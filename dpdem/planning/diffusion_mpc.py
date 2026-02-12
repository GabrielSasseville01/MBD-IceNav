"""
Diffusion MPC–style planner for ice navigation (alternative to lattice A*).

Receding-horizon MPC that generates candidate trajectories and selects the best
according to costmap cost and goal distance. This implementation uses
**sampling-based rollout** (random sequences of motion primitives) as the
trajectory proposal mechanism. In the general Diffusion MPC literature, a learned
diffusion model generates proposals (Sample–Score–Rank); a diffusion model could
be plugged in later to replace _sample_rollout() for higher-quality proposals.

Note: AUTO-IceNav (T-RO 2025 / ICRA 2023) does *not* use a diffusion model; it
uses lattice A* with a kinetic-energy cost function plus NLP refinement. The
*LatticePlanner* in this codebase corresponds to that approach. This class is an
alternative, experimental sampling-based MPC.

Compatible with the same CostMap and world frame as LatticePlanner; plan() returns
Nx3 [x, y, heading] path for path_to_ship_trajectory().
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Union

from .primitives import Primitives, Primitive
from .cost_map import CostMap


class DiffusionMPCPlanner:
    """
    Sampling-based MPC planner for ice navigation.

    Generates num_samples candidate paths by rolling out horizon_steps using
    (randomly) sampled motion primitives. Each path is scored by costmap integral
    plus goal distance. The best path is returned. Compatible with MEDEA
    costmap and path_to_ship_trajectory().

    Design note: In the general Diffusion MPC literature, trajectory proposals
    come from a learned diffusion model; here we use random primitive rollouts
    as a placeholder. A learned diffusion model can replace _sample_rollout() later
    for higher-quality proposals. AUTO-IceNav itself uses lattice A* + KE cost + NLP,
    not diffusion (see docs/DIFFUSION_MPC.md).
    """

    def __init__(
        self,
        primitives: Primitives,
        cost_map: CostMap,
        ship_length: float,
        ship_width: float,
        goal_tolerance: float = 5.0,
        horizon_steps: int = 12,
        num_samples: int = 64,
        goal_weight: float = 2.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            primitives: Motion primitive set (same as lattice planner).
            cost_map: Cost map for path cost (same frame as simulation).
            ship_length: Ship length [m] (for compatibility).
            ship_width: Ship width [m] (for compatibility).
            goal_tolerance: Distance to goal to consider “reached” [m].
            horizon_steps: Number of primitive steps per rollout.
            num_samples: Number of candidate rollouts to score.
            goal_weight: Weight for goal distance in score (higher = prefer reaching goal).
            seed: Random seed for reproducibility.
        """
        self.primitives = primitives
        self.cost_map = cost_map
        self.ship_length = ship_length
        self.ship_width = ship_width
        self.goal_tolerance = goal_tolerance
        self.horizon_steps = horizon_steps
        self.num_samples = num_samples
        self.goal_weight = goal_weight
        self._rng = np.random.default_rng(seed)

        # Stats (for compatibility with lattice planner usage)
        self.nodes_expanded = 0
        self.nodes_generated = num_samples

    def plan(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float],
        max_iterations: Optional[int] = None,
        return_search_history: bool = False,
    ) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], List]]:
        """
        Plan a path from start to goal using sampling-based MPC.

        Args:
            start: (x, y, heading) in world coordinates [m, m, rad].
            goal: (x, y) in world coordinates [m, m].
            max_iterations: Ignored (kept for interface compatibility).
            return_search_history: If True, returns (path, []) for compatibility.

        Returns:
            Path as Nx3 [x, y, heading], or None if no valid path found.
            If return_search_history is True, returns (path, []).
        """
        gx, gy = goal
        sx, sy, shead = start

        # Bounds check
        ox, oy = self.cost_map.x_origin, self.cost_map.y_origin
        if not (ox <= sx < ox + self.cost_map.width and oy <= sy < oy + self.cost_map.height):
            return (None, []) if return_search_history else None
        if not (ox <= gx < ox + self.cost_map.width and oy <= gy < oy + self.cost_map.height):
            return (None, []) if return_search_history else None

        best_path: Optional[np.ndarray] = None
        best_score = np.inf

        for _ in range(self.num_samples):
            path = self._sample_rollout(start, goal)
            if path is None or len(path) < 2:
                continue
            score = self._score_path(path, goal)
            if score < best_score:
                best_score = score
                best_path = path

        if best_path is None:
            return (None, []) if return_search_history else None
        if return_search_history:
            return (best_path, [])
        return best_path

    def _sample_rollout(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        """Generate one candidate path by random primitive rollout."""
        x, y, heading = start
        gx, gy = goal
        heading_idx = self.primitives.world_heading_to_index(heading)
        segments: List[np.ndarray] = []
        # First point
        current = np.array([[x, y, heading]], dtype=float)

        for _ in range(self.horizon_steps):
            prims = self.primitives.get_primitives(heading_idx)
            if not prims:
                break
            prim = prims[self._rng.integers(0, len(prims))]
            world_path = self.primitives.transform_primitive(prim, x, y, heading)
            # Append from index 1 to avoid duplicate point
            segments.append(world_path[1:])
            # Update state from end of primitive
            x, y = float(world_path[-1, 0]), float(world_path[-1, 1])
            heading = float(world_path[-1, 2])
            heading_idx = prim.end_heading
            # Early exit if we reached goal
            if np.hypot(x - gx, y - gy) <= self.goal_tolerance:
                break

        if not segments:
            return None
        path = np.vstack([current] + segments)
        return path

    def _score_path(self, path: np.ndarray, goal: Tuple[float, float]) -> float:
        """Score path: costmap integral + goal_weight * distance to goal."""
        path_cost = self.cost_map.get_path_cost(path)
        end = path[-1, :2]
        goal_dist = np.hypot(end[0] - goal[0], end[1] - goal[1])
        return path_cost + self.goal_weight * goal_dist
