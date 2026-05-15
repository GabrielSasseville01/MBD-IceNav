"""
Per-class seed generation and Hernandez reference-frame homotopy classifier.

The key contribution: instead of seeding all homotopy classes from the same
initial trajectory, each class is seeded from a geometric path that routes
through the class's specific passage sequence.  This fixes the exploration
failure observed in sequential-gate scenarios where the S-curve homotopy
class has near-zero probability under the Gaussian diffusion prior.

Reference: Hernandez et al. (2015) "Comparison of methods for homotopy class
identification and planning" — reference-frame approach.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

import jax.numpy as jnp


# ── Per-class seed generation ─────────────────────────────────────────────────

def generate_per_class_waypoints(
    obstacles: list,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    start_heading: float = float(np.pi / 2),
) -> List[np.ndarray]:
    """
    For each homotopy class (2^n_obs total), build a (N, 3) waypoint array
    [x, y, heading] that routes through the class's passage sequence.

    Bit k of the class index = 1  →  route through obstacle k's free corridor
                                       (cx_free, the widest gap).
    Bit k of the class index = 0  →  route through the obstacle body centre
                                       (cx_obs; will hit ice, but MBD corrects).

    Obstacles are visited bottom-to-top (ascending cy) so the route follows
    the natural encounter order for a ship moving from start to goal.

    Parameters
    ----------
    obstacles : list of (cx_obs, cx_free, cy, ry)
        From detect_obstacles_from_costmap().
    start_xy : (x, y) in world coordinates.
    goal_xy  : (x, y) in world coordinates.
    start_heading : initial heading [rad]. Default π/2 (north).

    Returns
    -------
    List of 2^n_obs arrays, each (N, 3) float64 with columns [x, y, heading].
    Index c in the list corresponds to class label c from make_homotopy_classifier
    or make_rf_homotopy_classifier (same bit assignment as those functions).
    """
    n_obs = len(obstacles)
    if n_obs == 0:
        dx = goal_xy[0] - start_xy[0]
        dy = goal_xy[1] - start_xy[1]
        hdg = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 0.5 else start_heading
        wpts = np.array([
            [start_xy[0], start_xy[1], start_heading],
            [goal_xy[0],  goal_xy[1],  hdg],
        ], dtype=np.float64)
        return [wpts]

    # Visit obstacles in y-ascending order (bottom-to-top encounter)
    y_order = sorted(range(n_obs), key=lambda i: float(obstacles[i][2]))

    n_classes = 1 << n_obs
    result: List[np.ndarray] = []

    for cls_idx in range(n_classes):
        wpts: list = [(float(start_xy[0]), float(start_xy[1]), float(start_heading))]

        for orig_idx in y_order:
            obs        = obstacles[orig_idx]
            cx_obs, cx_free, cy, ry = obs[:4]
            # Optional 5th element: list of all free corridor x-centres (widest first)
            all_passages: list = list(obs[4]) if len(obs) > 4 and obs[4] else [cx_free]

            bit = (cls_idx >> orig_idx) & 1
            if bit == 1:
                # Route through the widest free corridor (cx_free = all_passages[0])
                passage_x = float(cx_free)
            else:
                # Route through the SECOND corridor if one exists, else obstacle body.
                # This seeds classes that nominally "miss" corridor 0 with a valid
                # alternative route rather than an infeasible through-ice path.
                passage_x = float(all_passages[1]) if len(all_passages) > 1 else float(cx_obs)

            prev = wpts[-1]
            dx   = passage_x - prev[0]
            dy   = float(cy) - prev[1]
            hdg  = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 0.5 else prev[2]
            wpts.append((passage_x, float(cy), hdg))

        prev = wpts[-1]
        dx   = goal_xy[0] - prev[0]
        dy   = goal_xy[1] - prev[1]
        hdg  = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 0.5 else prev[2]
        wpts.append((float(goal_xy[0]), float(goal_xy[1]), hdg))

        result.append(np.array(wpts, dtype=np.float64))

    return result


# ── Hernandez reference-frame classifier ─────────────────────────────────────

def make_rf_homotopy_classifier(
    obstacles: list,
    center: Tuple[float, float],
):
    """
    Hernandez reference-frame homotopy classifier (JAX, vmappable).

    Draws a ray from `center` through each obstacle centroid.  A trajectory
    is assigned bit k = 1 if its mean x-position in obstacle k's y-band lies
    on the SAME SIDE of the k-th ray as cx_free_k (the free corridor centre).

    Side is determined by the sign of the 2-D cross product:
        cross_k = ray_dir_k × (mean_x_k − center_x,  cy_k − center_y)
    Bit k = 1  when  sign(cross_k) == sign_free_k,  where sign_free_k is the
    sign computed at (cx_free_k, cy_k).

    This is more principled than plain mean-x distance comparison because
    the reference frame encodes the topological structure of the obstacle
    layout relative to the start.

    Parameters
    ----------
    obstacles : list of (cx_obs, cx_free, cy, ry)
        From detect_obstacles_from_costmap().
    center : (x, y) — reference point, typically the ship start position.

    Returns
    -------
    classify_fn : callable  traj_xy (T, 2) → int32
    n_classes   : int       2 ** len(obstacles)
    """
    n_obs = len(obstacles)
    if n_obs == 0:
        return (lambda traj_xy: jnp.int32(0)), 1

    c = np.asarray(center, dtype=np.float64).ravel()[:2]

    ray_dirs_np   = np.zeros((n_obs, 2), dtype=np.float64)
    free_signs_np = np.zeros(n_obs,      dtype=np.float64)

    for k, obs in enumerate(obstacles):
        cx_obs, cx_free, cy, ry = obs[:4]
        # Ray: from c toward obstacle centroid
        d = np.array([float(cx_obs) - c[0], float(cy) - c[1]])
        d_norm = float(np.linalg.norm(d))
        d = d / d_norm if d_norm > 1e-6 else np.array([0.0, 1.0])
        ray_dirs_np[k] = d

        # Sign of cross product at the free-corridor position
        dx_free = float(cx_free) - c[0]
        dy_obs  = float(cy)      - c[1]
        cross_free = d[0] * dy_obs - d[1] * dx_free
        free_signs_np[k] = float(np.sign(cross_free)) if abs(cross_free) > 1e-6 else 1.0

    center_j     = jnp.array(c,             dtype=jnp.float32)
    ray_dirs_j   = jnp.array(ray_dirs_np,   dtype=jnp.float32)    # (n_obs, 2)
    free_signs_j = jnp.array(free_signs_np, dtype=jnp.float32)    # (n_obs,)
    obs_cy_j     = jnp.array([float(o[2]) for o in obstacles], dtype=jnp.float32)
    obs_ry_j     = jnp.array([float(o[3]) for o in obstacles], dtype=jnp.float32)
    powers       = jnp.array([1 << k for k in range(n_obs)],    dtype=jnp.int32)

    def classify(traj_xy):
        """traj_xy : (T, 2) → int32 class label."""
        xs = traj_xy[:, 0]
        ys = traj_xy[:, 1]

        # Mean x in each obstacle's y-band
        in_band = jnp.abs(ys[None, :] - obs_cy_j[:, None]) <= obs_ry_j[:, None]   # (n_obs, T)
        n_in    = in_band.sum(axis=1).astype(jnp.float32)                           # (n_obs,)
        sum_x   = (xs[None, :] * in_band).sum(axis=1)
        mean_x  = jnp.where(n_in > 0, sum_x / jnp.maximum(n_in, 1.0), xs[0])      # (n_obs,)

        # Cross product: ray_dir_k × (mean_x_k − center_x,  cy_k − center_y)
        dx    = mean_x   - center_j[0]                                              # (n_obs,)
        dy    = obs_cy_j - center_j[1]                                              # (n_obs,)
        cross = ray_dirs_j[:, 0] * dy - ray_dirs_j[:, 1] * dx                      # (n_obs,)

        bits = (jnp.sign(cross) == free_signs_j).astype(jnp.int32)
        return jnp.sum(bits * powers)

    return classify, (1 << n_obs)
