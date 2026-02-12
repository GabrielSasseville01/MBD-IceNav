"""
Model-based diffusion controller for ship ice navigation (AUTO-IceNav style).

Optimizes a control sequence (body-frame force/torque) via reverse diffusion:
sample many controls, rollout with 3-DOF vessel dynamics, score with costmap +
path tracking + waypoint anchors, softmax-weight update. Uses the same
CostMap and world frame as the lattice planner.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from medea_ice_ship.vessel_dynamics import get_mss_style_3dof, rollout_3dof
from medea_ice_ship.properties import R_xy


class ModelBasedDiffusionController:
    """
    Model-based diffusion over control sequences (AUTO-IceNav style).

    Reverse SDE diffusion: at each step sample control sequences, rollout with
    3-DOF dynamics, score with costmap + path tracking + waypoint anchors,
    update mean control by reward-weighted average. Output is trajectory (x, y, psi).
    """

    def __init__(
        self,
        horizon: int,
        M3: np.ndarray,
        D3: np.ndarray,
        diffusion_noise_clipping: float = 1e6,
        num_samples: int = 128,
        num_diffusion_steps: int = 30,
        temperature: float = 0.1,
        beta0: float = 1e-4,
        betaT: float = 1e-2,
        seed: int = 0,
        surge_min: float = 1e5,
        surge_max: float = 2e6,
        sway_max: float = 1e6,
        yaw_max: float = 5e6,
        ship_length: float = 0.0,
        ship_beam: float = 0.0,
    ):
        """
        Args:
            horizon: Number of control steps (trajectory length = horizon+1).
            M3, D3: 3×3 mass and damping matrices for 3-DOF dynamics.
            diffusion_noise_clipping: Clip control samples to ± this (for scaling).
            num_samples: Samples per diffusion step.
            num_diffusion_steps: Reverse diffusion steps.
            temperature: Softmax temperature for reward weighting.
            beta0, betaT: Diffusion schedule (linear beta).
            seed: Random seed.
            surge_min/max, sway_max, yaw_max: Control limits [N], [N], [N·m].
            ship_length, ship_beam: Ship dimensions [m]. If > 0, cost is evaluated at
                hull corners (same as lattice) so the path avoids ice for the full footprint.
        """
        self.horizon = horizon
        self.M3 = M3
        self.D3 = D3
        self.ship_length = float(ship_length)
        self.ship_beam = float(ship_beam)
        self.diffusion_noise_clipping = diffusion_noise_clipping
        self.num_samples = num_samples
        self.num_diffusion_steps = num_diffusion_steps
        self.temperature = temperature
        self.beta0 = beta0
        self.betaT = betaT
        self.surge_min = surge_min
        self.surge_max = surge_max
        self.sway_max = sway_max
        self.yaw_max = yaw_max
        self.nu_dim = 3

        np.random.seed(seed)
        self._compute_diffusion_schedule()

        self.u_seq_mean = np.zeros((horizon, self.nu_dim))
        self.waypoint_anchors: Dict[int, np.ndarray] = {}
        self.enforce_forward_motion = True
        self.target_surge_speed: Optional[float] = None
        self.predicted_trajectory: Optional[np.ndarray] = None

    def _compute_diffusion_schedule(self) -> None:
        betas = np.linspace(self.beta0, self.betaT, self.num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)
        sigmas = np.sqrt(1.0 - alphas_bar)
        alphas_bar_rolled = np.roll(alphas_bar, 1)
        alphas_bar_rolled[0] = 1.0
        Sigmas_cond = (1.0 - alphas) * (1.0 - np.sqrt(alphas_bar_rolled)) / (1.0 - alphas_bar)
        sigmas_cond = np.sqrt(np.maximum(Sigmas_cond, 0))
        sigmas_cond[0] = 0.0
        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.sigmas = sigmas
        self.sigmas_cond = sigmas_cond

    def _clip_controls(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if u.ndim == 2:
            u = u.copy()
            u[:, 0] = np.clip(u[:, 0], self.surge_min, self.surge_max)
            u[:, 1] = np.clip(u[:, 1], -self.sway_max, self.sway_max)
            u[:, 2] = np.clip(u[:, 2], -self.yaw_max, self.yaw_max)
        else:
            u = u.copy()
            u[0] = np.clip(u[0], self.surge_min, self.surge_max)
            u[1] = np.clip(u[1], -self.sway_max, self.sway_max)
            u[2] = np.clip(u[2], -self.yaw_max, self.yaw_max)
        return u

    def _cost_at_pose(self, costmap: Any, x: float, y: float, heading: float) -> float:
        """
        Cost at ship pose: center and (if ship_length/ship_beam > 0) hull corners.
        Matches lattice planner footprint logic so diffusion avoids ice for full hull.
        """
        c_center = costmap.get_cost_world(x, y)
        if self.ship_length <= 0 or self.ship_beam <= 0:
            return float(c_center) if np.isfinite(c_center) else 1e6
        c, s = np.cos(heading), np.sin(heading)
        half_l = self.ship_length / 2
        half_w = self.ship_beam / 2
        corners_body = [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
        cost = float(c_center) if np.isfinite(c_center) else 1e6
        for dx, dy in corners_body:
            wx = x + c * dx - s * dy
            wy = y + s * dx + c * dy
            cc = costmap.get_cost_world(wx, wy)
            if np.isfinite(cc):
                cost = max(cost, float(cc))
            else:
                cost = max(cost, 1e6)
        return cost

    def rollout_trajectory(
        self,
        eta_start: np.ndarray,
        nu_start: np.ndarray,
        u_seq: np.ndarray,
        dt: float,
        costmap: Optional[Any] = None,
        goal_y: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        """Roll out one trajectory; return (trajectory, reward, states_list)."""
        u_seq = self._clip_controls(u_seq)
        trajectory = rollout_3dof(eta_start, nu_start, u_seq, dt, self.M3, self.D3)
        states_list = [trajectory[k].copy() for k in range(len(trajectory))]
        total_cost = 0.0
        for t in range(len(trajectory)):
            eta = trajectory[t]
            if costmap is not None:
                c = self._cost_at_pose(costmap, float(eta[0]), float(eta[1]), float(eta[2]))
                if np.isfinite(c):
                    # Strong weight so trajectories through ice get heavily penalized vs lattice path
                    total_cost += c * 100.0
                else:
                    total_cost += 10000.0
            total_cost += np.sum((u_seq[min(t, len(u_seq) - 1)] / self.surge_max) ** 2) * 0.001
        goal_reward = 0.0
        if goal_y is not None and len(trajectory) > 0:
            y_progress = trajectory[-1, 1] - eta_start[1]
            if y_progress < 0.1:
                goal_reward = -1000.0
        total_reward = -total_cost + goal_reward
        return trajectory, total_reward, states_list

    def objective_function(
        self,
        u_seq_batch: np.ndarray,
        eta_start: np.ndarray,
        nu_start: np.ndarray,
        local_path: Optional[np.ndarray],
        dt: float,
        costmap: Optional[Any] = None,
        goal_y: Optional[float] = None,
    ) -> np.ndarray:
        """Evaluate rewards for a batch of control sequences (higher is better)."""
        rewards = np.zeros(self.num_samples)
        for i in range(self.num_samples):
            trajectory, base_reward, _ = self.rollout_trajectory(
                eta_start, nu_start, u_seq_batch[i], dt, costmap, goal_y=goal_y
            )
            path_reward = 0.0
            if local_path is not None:
                for t in range(min(len(trajectory) - 1, len(local_path))):
                    pos_err = np.sum((trajectory[t + 1, :2] - local_path[t, :2]) ** 2)
                    ang_err = (trajectory[t + 1, 2] - local_path[t, 2] + np.pi) % (2 * np.pi) - np.pi
                    path_reward -= (pos_err + 50.0 * ang_err ** 2) * 1.0
            waypoint_reward = 0.0
            for anchor_t, anchor_wp in self.waypoint_anchors.items():
                if anchor_t < len(trajectory):
                    dist = np.linalg.norm(trajectory[anchor_t, :2] - anchor_wp[:2])
                    if dist < 100.0:
                        waypoint_reward += 5000.0 / (1.0 + dist / 10.0)
                    else:
                        waypoint_reward -= dist * 200.0
            rewards[i] = base_reward + path_reward + waypoint_reward
        return rewards

    def reverse_diffusion_step(
        self,
        i: int,
        Ybar_i: np.ndarray,
        eta_start: np.ndarray,
        nu_start: np.ndarray,
        local_path: Optional[np.ndarray],
        dt: float,
        costmap: Optional[Any] = None,
        goal_y: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Single reverse diffusion step (reward-weighted update)."""
        Yi = Ybar_i * np.sqrt(self.alphas_bar[i])
        eps_u = np.random.normal(0, 1, (self.num_samples, self.horizon, self.nu_dim))
        Y0s = eps_u * self.sigmas[i] + Ybar_i
        if self.enforce_forward_motion and Ybar_i.size > 0:
            mean_surge = max(self.surge_min, float(np.mean(Ybar_i[:, 0])))
            Y0s[:, :, 0] = np.clip(
                mean_surge + eps_u[:, :, 0] * self.sigmas[i] * 0.2,
                self.surge_min,
                self.surge_max,
            )
        Y0s = self._clip_controls(Y0s)
        rewards = self.objective_function(
            Y0s, eta_start, nu_start, local_path, dt, costmap, goal_y=goal_y
        )
        rew_mean = float(np.mean(rewards))
        rew_std = max(float(np.std(rewards)), 1e-4)
        logp0 = (rewards - rew_mean) / rew_std / self.temperature
        weights = np.exp(logp0 - np.max(logp0))
        weights = weights / weights.sum()
        Ybar = np.einsum("n,nij->ij", weights, Y0s)
        Ybar = self._clip_controls(Ybar)
        Ybar[:, 0] = np.maximum(self.surge_min, Ybar[:, 0])
        score = (1.0 / (1.0 - self.alphas_bar[i])) * (
            -Yi + np.sqrt(self.alphas_bar[i]) * Ybar
        )
        Yim1 = (1.0 / np.sqrt(self.alphas[i])) * (Yi + (1.0 - self.alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / np.sqrt(self.alphas_bar[i - 1])
        Ybar_im1 = self._clip_controls(Ybar_im1)
        Ybar_im1[:, 0] = np.maximum(self.surge_min, Ybar_im1[:, 0])
        return Ybar_im1, rew_mean

    def run_diffusion(
        self,
        pose: np.ndarray,
        nu: np.ndarray,
        local_path: np.ndarray,
        dt: float,
        costmap: Optional[Any] = None,
        goal_y: Optional[float] = None,
        warm_start_controls: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run reverse diffusion to get optimized control sequence, then return
        predicted trajectory (horizon+1, 3) [x, y, psi].
        """
        if nu is None or nu.size == 0:
            nu = np.array([2.0, 0.0, 0.0], dtype=float)
        pose = np.asarray(pose, dtype=float)
        nu = np.asarray(nu, dtype=float)
        if local_path is None or len(local_path) == 0:
            local_path = np.tile(pose, (self.horizon, 1))
        else:
            local_path = np.asarray(local_path, dtype=float)
            if len(local_path) < self.horizon:
                last = local_path[-1:]
                local_path = np.vstack([local_path, np.tile(last, (self.horizon - len(local_path), 1))])
            else:
                local_path = local_path[: self.horizon]
        if warm_start_controls is not None and warm_start_controls.shape[0] == self.horizon:
            Ybar_i = self._clip_controls(warm_start_controls.copy())
        else:
            Ybar_i = np.zeros((self.horizon, self.nu_dim))
            Ybar_i[:, 0] = self.surge_min
        for i in range(self.num_diffusion_steps - 1, 0, -1):
            Ybar_i, _ = self.reverse_diffusion_step(
                i, Ybar_i, pose, nu, local_path, dt, costmap, goal_y=goal_y
            )
        self.u_seq_mean = Ybar_i
        self.predicted_trajectory = rollout_3dof(pose, nu, self.u_seq_mean, dt, self.M3, self.D3)
        return self.predicted_trajectory.copy()
