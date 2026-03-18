"""
Vectorized geometry operations for the JAX ice-ship simulation.

Replaces the Python nested for-loops in simulation.py (lines 347–455) with
fully batched JAX operations.  All if/else branches use jnp.where so the
functions are JIT-compilable and GPU-compatible.

Key functions
-------------
point_to_ship_collision_batched
    Ship hull collision query for m points simultaneously (replaces scalar
    point_to_ship_collision in simulation.py).

compute_all_query_points
    Rotates all ice element polygons (padded, uniform shape) to world frame
    and generates vertices + edge sample points in one vectorized operation.

compute_all_ship_collisions
    Combines the two above: returns per-element max-penetration contact
    (pen, nx, ny, contact_x, contact_y) as (n,) arrays.

point_in_polygon_batched
    Vectorized ray-casting PIP test for m points against one polygon (V,).

ship_hull_vs_ice_batched
    Tests all ship hull polygon points against all ice polygons (padded)
    and returns per-element max-penetration for the "hull inside ice" case.

compute_all_collisions_combined
    Combines compute_all_ship_collisions (ice vertices/edge points vs ship hull)
    and ship_hull_vs_ice_batched (ship hull points vs ice polygons) into one
    JIT-compiled call that returns the maximum-penetration contact per element.
"""

import jax
import jax.numpy as jnp
from functools import partial


# ── Ship hull collision: ice points vs ship ───────────────────────────────────

@jax.jit
def point_to_ship_collision_batched(
    px: jnp.ndarray,       # (m,) x-coordinates of query points [m]
    py: jnp.ndarray,       # (m,) y-coordinates
    ship_x: float,
    ship_y: float,
    ship_heading: float,   # [rad]
    half_len: float,
    half_beam: float,
    bow_taper_start: float = 0.6,
) -> tuple:  # (pen (m,), nx (m,), ny (m,))
    """
    Vectorized ship hull collision test.

    Tests m points simultaneously against the ship hull, returning
    penetration depth and outward normal for each point.

    Replaces the scalar point_to_ship_collision() called inside the
    triple nested loop in simulation.py lines 358–386.
    All if/else branches converted to jnp.where (JIT-compatible).

    Parameters
    ----------
    px, py           : (m,) world-frame coordinates of query points
    ship_x, ship_y   : ship centroid position [m]
    ship_heading     : ship yaw [rad]  (bow faces in +ship_heading direction)
    half_len         : ship half-length [m]
    half_beam        : ship half-beam [m]
    bow_taper_start  : x_rel threshold where bow taper begins (0 = midship, 1 = bow)

    Returns
    -------
    pen : (m,) penetration depth [m]; 0.0 if no collision
    nx  : (m,) world-frame contact normal x-component
    ny  : (m,) world-frame contact normal y-component
    """
    # ── Rotate points to ship body frame ─────────────────────────────────────
    c_neg = jnp.cos(-ship_heading)
    s_neg = jnp.sin(-ship_heading)
    dx = px - ship_x
    dy = py - ship_y
    rel_x = c_neg * dx - s_neg * dy
    rel_y = s_neg * dx + c_neg * dy
    x_rel = rel_x / half_len  # normalised: -1 = bow, +1 = stern (matches scalar convention)

    # ── Bow taper geometry ────────────────────────────────────────────────────
    # Bow is at the NEGATIVE x end (x_rel < -bow_taper_start), matching scalar convention.
    in_bow = x_rel < -bow_taper_start
    t_bow = jnp.clip((-x_rel - bow_taper_start) / (1.0 - bow_taper_start), 0.0, 1.0)
    local_hb_bow = half_beam * (1.0 - 0.9 * t_bow ** 1.5)
    local_hb_bow = jnp.maximum(local_hb_bow, half_beam * 0.05)
    local_hb = jnp.where(in_bow, local_hb_bow, half_beam)

    # ── Bounds check (outside hull → zero penetration) ────────────────────────
    inside = (
        (rel_x >= -half_len) & (rel_x <= half_len)
        & (jnp.abs(rel_y) <= local_hb)
    )

    # ── Penetration depth and body-frame normal ───────────────────────────────
    # dist_to_bow: distance to bow face at rel_x = -half_len
    dist_to_bow  = rel_x + half_len           # = rel_x - (-half_len)
    dist_to_side = local_hb - jnp.abs(rel_y)

    # Bow region: angled normal pointing away from hull surface
    bow_angle = 0.3 + 0.5 * t_bow            # angle of bow normal (rad)
    nx_bow = -jnp.cos(bow_angle)             # outward from bow face = -x direction in body
    ny_bow = jnp.sign(rel_y) * jnp.sin(bow_angle)
    pen_bow = jnp.minimum(dist_to_bow, dist_to_side) + 0.5  # offset for robustness

    # Non-bow region: side contact or bow face contact (no stern face — matches scalar)
    side_contact = dist_to_side <= dist_to_bow
    nx_body_nonbow = jnp.where(side_contact, 0.0, -1.0)   # bow face normal: -x
    ny_body_nonbow = jnp.where(side_contact, jnp.sign(rel_y), 0.0)
    pen_nonbow = jnp.where(side_contact, dist_to_side, dist_to_bow)

    # Select bow vs body
    nx_body = jnp.where(in_bow, nx_bow, nx_body_nonbow)
    ny_body = jnp.where(in_bow, ny_bow, ny_body_nonbow)
    pen_raw = jnp.where(in_bow, pen_bow, pen_nonbow)

    # ── Rotate normal back to world frame ─────────────────────────────────────
    c_h = jnp.cos(ship_heading)
    s_h = jnp.sin(ship_heading)
    nx_world = c_h * nx_body - s_h * ny_body
    ny_world = s_h * nx_body + c_h * ny_body

    # Zero out where outside hull
    pen = jnp.where(inside, jnp.maximum(pen_raw, 0.0), 0.0)
    nx_out = jnp.where(inside, nx_world, 0.0)
    ny_out = jnp.where(inside, ny_world, 0.0)

    return pen, nx_out, ny_out


# ── Query-point generation: all ice vertices + edge samples ───────────────────

def compute_all_query_points(
    local_polys_padded: jnp.ndarray,  # (n, max_V, 2)
    poly_mask: jnp.ndarray,           # (n, max_V) bool
    positions: jnp.ndarray,           # (n, 2)
    rotations: jnp.ndarray,           # (n,)
    n_edge_samples: int,
) -> tuple:  # (pts (n, P, 2), mask (n, P)) where P = max_V * (1 + n_edge_samples)
    """
    Rotate all padded ice polygons to world frame and generate query points.

    For each ice element, produces:
      - All real+padded polygon vertices in world frame
      - n_edge_samples interior points per edge

    Replaces the per-floe rotation (simulation.py lines 352–356) and the
    edge sampling loops (lines 367–378) with batched operations.

    Parameters
    ----------
    local_polys_padded : (n, max_V, 2)  body-frame vertices (padded)
    poly_mask          : (n, max_V) bool True = real vertex
    positions          : (n, 2)  world-frame centroids
    rotations          : (n,)    element yaw angles [rad]
    n_edge_samples     : number of interior sample points per edge

    Returns
    -------
    all_pts  : (n, P, 2)  world-frame query points
    all_mask : (n, P) bool True = point from a real vertex/edge
    """
    n, max_V, _ = local_polys_padded.shape

    # ── Rotation matrices per element ─────────────────────────────────────────
    c = jnp.cos(rotations)  # (n,)
    s = jnp.sin(rotations)  # (n,)

    # Rotate local vertices to world frame: (n, max_V, 2)
    vx_loc = local_polys_padded[:, :, 0]  # (n, max_V)
    vy_loc = local_polys_padded[:, :, 1]
    vx_world = c[:, None] * vx_loc - s[:, None] * vy_loc + positions[:, 0:1]
    vy_world = s[:, None] * vx_loc + c[:, None] * vy_loc + positions[:, 1:2]
    world_verts = jnp.stack([vx_world, vy_world], axis=-1)  # (n, max_V, 2)

    if n_edge_samples == 0:
        # No edge samples: only vertices
        return world_verts, poly_mask

    # ── Edge sample points ────────────────────────────────────────────────────
    # For edge (v_k, v_{k+1 mod max_V}), generate n_edge_samples interior pts.
    # t values: 1/(K+1), 2/(K+1), ..., K/(K+1)
    K = n_edge_samples
    t = jnp.linspace(1.0 / (K + 1), K / (K + 1), K)  # (K,)

    # Next-vertex via roll along vertex axis
    v_curr = world_verts                          # (n, max_V, 2)
    v_next = jnp.roll(world_verts, -1, axis=1)   # (n, max_V, 2)

    # (n, max_V, K, 2)
    edge_pts = (
        v_curr[:, :, None, :] * (1.0 - t)[None, None, :, None]
        + v_next[:, :, None, :] * t[None, None, :, None]
    )

    # ── Concatenate: vertices then edge samples ────────────────────────────────
    # Shape: (n, max_V, 1+K, 2) → flatten middle dims → (n, max_V*(1+K), 2)
    all_pts = jnp.concatenate(
        [world_verts[:, :, None, :], edge_pts], axis=2
    ).reshape(n, max_V * (1 + K), 2)

    # Mask: a sample point is valid iff its originating vertex is valid
    all_mask = jnp.repeat(poly_mask, 1 + K, axis=1)  # (n, max_V*(1+K))

    return all_pts, all_mask


# ── Per-element max-penetration (ice points → ship hull) ──────────────────────

def compute_all_ship_collisions(
    local_polys_padded: jnp.ndarray,  # (n, max_V, 2)
    poly_mask: jnp.ndarray,           # (n, max_V) bool
    positions: jnp.ndarray,           # (n, 2)
    rotations: jnp.ndarray,           # (n,)
    ship_x: float,
    ship_y: float,
    ship_heading: float,
    half_len: float,
    half_beam: float,
    n_edge_samples: int = 3,
) -> tuple:  # (pen (n,), nx (n,), ny (n,), cx (n,), cy (n,))
    """
    For each ice element, find the maximum-penetration contact with the ship hull.

    Replaces the triple nested loop at simulation.py lines 347–386:
        for i in range(n):
            for vx, vy in world_poly:       ← vertex loop
                point_to_ship_collision(...)
            for j in range(n_verts):
                for k in range(n_edge_samples):  ← edge sample loop
                    point_to_ship_collision(...)

    Returns
    -------
    pen_max : (n,) maximum penetration per element [m]
    nx_best : (n,) contact normal x at max-penetration point
    ny_best : (n,) contact normal y
    cx_best : (n,) world-frame x of max-penetration contact point
    cy_best : (n,) world-frame y
    """
    n = positions.shape[0]

    # Generate all query points for all elements simultaneously
    all_pts, all_mask = compute_all_query_points(
        local_polys_padded, poly_mask, positions, rotations, n_edge_samples
    )
    # all_pts  : (n, P, 2)
    # all_mask : (n, P)

    P = all_pts.shape[1]

    # Flatten to (n*P, 2) for batched collision test
    pts_flat = all_pts.reshape(-1, 2)
    pen_flat, nx_flat, ny_flat = point_to_ship_collision_batched(
        pts_flat[:, 0], pts_flat[:, 1],
        ship_x, ship_y, ship_heading, half_len, half_beam,
    )

    # Zero out invalid (padded) points so they can never be selected
    mask_flat = all_mask.reshape(-1)
    pen_flat = jnp.where(mask_flat, pen_flat, 0.0)

    # Reshape to (n, P) and take argmax per element
    pen_2d = pen_flat.reshape(n, P)
    nx_2d  = nx_flat.reshape(n, P)
    ny_2d  = ny_flat.reshape(n, P)
    cx_2d  = all_pts[:, :, 0]  # (n, P)
    cy_2d  = all_pts[:, :, 1]

    best_idx = jnp.argmax(pen_2d, axis=1)  # (n,)
    idx = jnp.arange(n)
    pen_max = pen_2d[idx, best_idx]
    nx_best = nx_2d[idx, best_idx]
    ny_best = ny_2d[idx, best_idx]
    cx_best = cx_2d[idx, best_idx]
    cy_best = cy_2d[idx, best_idx]

    return pen_max, nx_best, ny_best, cx_best, cy_best


# ── Vectorized point-in-polygon (ray casting) ─────────────────────────────────

@jax.jit
def point_in_polygon_batched(
    px: jnp.ndarray,    # (m,) query x-coordinates
    py: jnp.ndarray,    # (m,) query y-coordinates
    poly_x: jnp.ndarray,  # (V,) polygon vertex x (closed, no repeat of first)
    poly_y: jnp.ndarray,  # (V,) polygon vertex y
) -> jnp.ndarray:         # (m,) bool
    """
    Vectorized ray-casting point-in-polygon test.

    Tests m query points against one polygon with V vertices simultaneously.
    All branches replaced with jnp.where (JIT-compatible).

    Replaces the scalar point_in_polygon() called in simulation.py line 381.
    """
    xi = poly_x             # (V,)
    xj = jnp.roll(poly_x, -1)
    yi = poly_y             # (V,)
    yj = jnp.roll(poly_y, -1)

    # Broadcast: (m, 1) vs (1, V)
    py_ = py[:, None]
    px_ = px[:, None]
    yi_ = yi[None, :]
    yj_ = yj[None, :]
    xi_ = xi[None, :]
    xj_ = xj[None, :]

    # Edge crossing condition (winding rule)
    cond_y = (yi_ > py_) != (yj_ > py_)
    x_int = (xj_ - xi_) * (py_ - yi_) / (yj_ - yi_ + 1e-10) + xi_
    crosses = cond_y & (px_ < x_int)           # (m, V)

    # Inside if odd number of crossings
    inside = jnp.sum(crosses.astype(jnp.int32), axis=1) % 2 == 1  # (m,)
    return inside


@jax.jit
def distance_point_to_segment_batched(
    px: jnp.ndarray,  # (m,)
    py: jnp.ndarray,  # (m,)
    ax: jnp.ndarray,  # (V,) segment start x
    ay: jnp.ndarray,  # (V,) segment start y
    bx: jnp.ndarray,  # (V,) segment end x
    by: jnp.ndarray,  # (V,) segment end y
) -> tuple:  # (dist (m, V), qx (m, V), qy (m, V))
    """
    Distance from each of m points to each of V segments.
    Returns distance and closest point on segment, vectorized.
    """
    # (m, 1) vs (1, V)
    px_ = px[:, None]; py_ = py[:, None]
    ax_ = ax[None, :]; ay_ = ay[None, :]
    bx_ = bx[None, :]; by_ = by[None, :]

    vx = bx_ - ax_; vy = by_ - ay_
    wx = px_ - ax_; wy = py_ - ay_

    c1 = wx * vx + wy * vy           # (m, V)
    c2 = vx * vx + vy * vy + 1e-20   # (m, V)
    t = jnp.clip(c1 / c2, 0.0, 1.0)  # (m, V)

    qx = ax_ + t * vx                # (m, V)
    qy = ay_ + t * vy
    dist = jnp.sqrt((px_ - qx) ** 2 + (py_ - qy) ** 2 + 1e-20)
    return dist, qx, qy


@jax.jit
def point_inside_polygon_penetration_batched(
    px: jnp.ndarray,        # (m,) query x
    py: jnp.ndarray,        # (m,) query y
    poly_x: jnp.ndarray,    # (V,) polygon vertices x
    poly_y: jnp.ndarray,    # (V,) polygon vertices y
    centroid_x: float,
    centroid_y: float,
) -> tuple:  # (pen (m,), nx (m,), ny (m,))
    """
    For points inside a polygon, compute penetration to nearest edge and outward normal.

    Replaces the scalar point_inside_polygon_penetration() in geometry.py (called
    at simulation.py line 382 for ship hull points inside ice polygon).

    Parameters
    ----------
    px, py       : (m,) query points
    poly_x, poly_y : (V,) polygon vertices (closed, first != last)
    centroid_x/y : polygon centroid for outward-normal orientation

    Returns
    -------
    pen : (m,) penetration [m]; 0 if point outside polygon
    nx  : (m,) outward normal x
    ny  : (m,) outward normal y
    """
    V = poly_x.shape[0]

    # ── PIP check ─────────────────────────────────────────────────────────────
    inside = point_in_polygon_batched(px, py, poly_x, poly_y)  # (m,)

    # ── Distance to each edge ─────────────────────────────────────────────────
    xi = poly_x; xj = jnp.roll(poly_x, -1)
    yi = poly_y; yj = jnp.roll(poly_y, -1)

    dist_mv, qx_mv, qy_mv = distance_point_to_segment_batched(px, py, xi, yi, xj, yj)
    # dist_mv, qx_mv, qy_mv : (m, V)

    # ── Outward normals per edge ───────────────────────────────────────────────
    ex = xj - xi   # (V,)
    ey = yj - yi
    # Perpendicular: (ey, -ex) — CCW polygon: left normal is inward, right is outward
    nx_edge = ey   # (V,)
    ny_edge = -ex
    nlen = jnp.sqrt(nx_edge ** 2 + ny_edge ** 2 + 1e-20)
    nx_edge = nx_edge / nlen
    ny_edge = ny_edge / nlen

    # Ensure outward direction (should point away from centroid)
    # Use representative points on each edge (midpoint)
    mid_x = (xi + xj) / 2.0
    mid_y = (yi + yj) / 2.0
    dot_centroid = (mid_x - centroid_x) * nx_edge + (mid_y - centroid_y) * ny_edge
    # If dot < 0, normal points inward → flip
    flip = dot_centroid < 0.0
    nx_edge = jnp.where(flip, -nx_edge, nx_edge)
    ny_edge = jnp.where(flip, -ny_edge, ny_edge)

    # ── Per-query-point: closest edge → penetration and normal ────────────────
    best_edge = jnp.argmin(dist_mv, axis=1)  # (m,)
    m = px.shape[0]
    midx = jnp.arange(m)

    pen_raw = dist_mv[midx, best_edge]         # (m,)
    nx_best = nx_edge[best_edge]               # (m,)
    ny_best = ny_edge[best_edge]

    pen = jnp.where(inside, pen_raw, 0.0)
    nx_out = jnp.where(inside, nx_best, 0.0)
    ny_out = jnp.where(inside, ny_best, 0.0)

    return pen, nx_out, ny_out


# ── Ship hull points vs all ice polygons ──────────────────────────────────────

def ship_hull_vs_ice_batched(
    ship_hull_world: jnp.ndarray,     # (S, 2)  ship hull pts in world frame
    world_polys: jnp.ndarray,         # (n, max_V, 2)  ice polygons (world frame, padded)
    poly_mask: jnp.ndarray,           # (n, max_V) bool
    ice_centroids: jnp.ndarray,       # (n, 2)
) -> tuple:  # (pen (n,), nx (n,), ny (n,), cx (n,), cy (n,))
    """
    For each ice element, find max-penetration of any ship hull point inside it.

    Replaces simulation.py lines 380–385:
        for sx, sy in ship_hull_pts:
            if point_in_polygon((sx, sy), world_poly):
                test_pen, ... = point_inside_polygon_penetration(...)

    This function is NOT decorated with @jax.jit because it uses a Python-level
    loop over ice elements (n can vary), but the inner operations are batched
    over the S hull points.  If n is large you can vmap over elements instead.
    """
    S = ship_hull_world.shape[0]
    n = world_polys.shape[0]
    max_V = world_polys.shape[1]

    # We need to test S hull points against each of n ice polygons.
    # Strategy: for each ice element (n loop — small to moderate), test all S hull points.
    # This avoids a (S, n, V) cross-product tensor that could be memory-intensive.

    pen_best = jnp.zeros(n)
    nx_best  = jnp.zeros(n)
    ny_best  = jnp.zeros(n)
    cx_best  = jnp.zeros(n)
    cy_best  = jnp.zeros(n)

    hx = ship_hull_world[:, 0]  # (S,)
    hy = ship_hull_world[:, 1]  # (S,)

    # For each element, test all hull points
    def test_one_element(i, carry):
        pen_best, nx_best, ny_best, cx_best, cy_best = carry
        poly_verts = world_polys[i]          # (max_V, 2)
        pmask = poly_mask[i]                 # (max_V,)
        centroid = ice_centroids[i]          # (2,)

        # Real vertices only
        vx = poly_verts[:, 0]
        vy = poly_verts[:, 1]
        # Apply mask to centroid computation (masked mean)
        n_real = jnp.maximum(jnp.sum(pmask.astype(jnp.float64)), 1.0)
        cx_c = jnp.sum(jnp.where(pmask, vx, 0.0)) / n_real
        cy_c = jnp.sum(jnp.where(pmask, vy, 0.0)) / n_real

        pen_s, nx_s, ny_s = point_inside_polygon_penetration_batched(
            hx, hy, vx, vy, cx_c, cy_c
        )  # (S,) each

        best_s = jnp.argmax(pen_s)
        pen_i = pen_s[best_s]
        nx_i  = nx_s[best_s]
        ny_i  = ny_s[best_s]
        cx_i  = hx[best_s]
        cy_i  = hy[best_s]

        # Only update if better than current (from ice-vertex tests)
        update = pen_i > pen_best[i]
        pen_best = pen_best.at[i].set(jnp.where(update, pen_i, pen_best[i]))
        nx_best  = nx_best.at[i].set(jnp.where(update, nx_i,  nx_best[i]))
        ny_best  = ny_best.at[i].set(jnp.where(update, ny_i,  ny_best[i]))
        cx_best  = cx_best.at[i].set(jnp.where(update, cx_i,  cx_best[i]))
        cy_best  = cy_best.at[i].set(jnp.where(update, cy_i,  cy_best[i]))
        return pen_best, nx_best, ny_best, cx_best, cy_best

    pen_best, nx_best, ny_best, cx_best, cy_best = jax.lax.fori_loop(
        0, n, test_one_element,
        (pen_best, nx_best, ny_best, cx_best, cy_best),
    )

    return pen_best, nx_best, ny_best, cx_best, cy_best


# ── Combined collision detection call (ice vertices/edges + hull PIP) ─────────

def compute_all_collisions_combined(
    local_polys_padded: jnp.ndarray,   # (n, max_V, 2)  body-frame
    poly_mask: jnp.ndarray,            # (n, max_V) bool
    positions: jnp.ndarray,            # (n, 2)
    rotations: jnp.ndarray,            # (n,)
    ship_x: float,
    ship_y: float,
    ship_heading: float,
    half_len: float,
    half_beam: float,
    ship_hull_local: jnp.ndarray,     # (S, 2)  ship hull in body frame
    n_edge_samples: int = 3,
) -> tuple:  # (pen (n,), nx (n,), ny (n,), cx (n,), cy (n,))
    """
    Full collision detection for all ice elements in one call.

    1. Tests ice polygon vertices + edge samples against the ship hull.
    2. Tests ship hull polygon points against each ice polygon.
    3. Takes element-wise maximum over both.

    This replaces the full collision block in simulate_ice_ship_interaction
    (simulation.py lines 347–455) with two batched operations.

    Parameters
    ----------
    local_polys_padded : (n, max_V, 2)  local-frame vertices (padded)
    poly_mask          : (n, max_V) bool
    positions          : (n, 2)  current ice centroids
    rotations          : (n,)    current ice rotations
    ship_x, ship_y     : ship position [m]
    ship_heading       : ship yaw [rad]
    half_len, half_beam: ship dimensions [m]
    ship_hull_local    : (S, 2)  ship hull vertices in body frame
    n_edge_samples     : number of edge sample points per ice polygon edge

    Returns
    -------
    pen : (n,) maximum penetration per ice element [m]
    nx  : (n,) contact normal x (outward from ship)
    ny  : (n,) contact normal y
    cx  : (n,) world-frame contact point x
    cy  : (n,) world-frame contact point y
    """
    # ── Pass 1: ice vertices/edges → ship hull ────────────────────────────────
    pen1, nx1, ny1, cx1, cy1 = compute_all_ship_collisions(
        local_polys_padded, poly_mask, positions, rotations,
        ship_x, ship_y, ship_heading, half_len, half_beam,
        n_edge_samples=n_edge_samples,
    )

    # ── Rotate world polygons for pass 2 ─────────────────────────────────────
    # (Reuse the same rotation logic — could cache from pass 1 in future)
    c = jnp.cos(rotations)
    s = jnp.sin(rotations)
    vx_loc = local_polys_padded[:, :, 0]
    vy_loc = local_polys_padded[:, :, 1]
    vx_w = c[:, None] * vx_loc - s[:, None] * vy_loc + positions[:, 0:1]
    vy_w = s[:, None] * vx_loc + c[:, None] * vy_loc + positions[:, 1:2]
    world_polys = jnp.stack([vx_w, vy_w], axis=-1)  # (n, max_V, 2)

    # ── Ship hull world-frame points ──────────────────────────────────────────
    c_h = jnp.cos(ship_heading)
    s_h = jnp.sin(ship_heading)
    R_ship = jnp.array([[c_h, -s_h], [s_h, c_h]])
    ship_hull_world = ship_hull_local @ R_ship.T + jnp.array([ship_x, ship_y])  # (S, 2)

    # ── Pass 2: ship hull points → ice polygons ───────────────────────────────
    pen2, nx2, ny2, cx2, cy2 = ship_hull_vs_ice_batched(
        ship_hull_world, world_polys, poly_mask, positions
    )

    # ── Element-wise maximum ──────────────────────────────────────────────────
    use_pass2 = pen2 > pen1
    pen = jnp.where(use_pass2, pen2, pen1)
    nx  = jnp.where(use_pass2, nx2,  nx1)
    ny  = jnp.where(use_pass2, ny2,  ny1)
    cx  = jnp.where(use_pass2, cx2,  cx1)
    cy  = jnp.where(use_pass2, cy2,  cy1)

    return pen, nx, ny, cx, cy
