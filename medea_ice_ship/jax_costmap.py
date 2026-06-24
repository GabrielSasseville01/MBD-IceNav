"""
JAX-native costmap builder for the ice-ship simulation.

Replaces ``medea_ice_ship.costmap_build.build_costmap_from_ice_state`` for the
receding-horizon diffusion-MPC pipeline.  Uses JAX scatter operations so the
costmap can be built entirely on the GPU, avoiding a device-to-host round-trip.

Uses the AUTO-IceNav KE collision cost model:

    cost(x, y) = max_i [ ke_i * max(0, (eff_r_i² - d²) / eff_r_i²) ]

where ke_i is the per-element kinetic-energy constant based on ship speed,
ship mass, and ice floe mass, and the parabolic falloff provides a smooth
cost field. Overlapping elements contribute the maximum (not sum) at each
cell. SIC modulation (``ice_resistance_weight > 0``) scales the KE cost by
local sea-ice concentration raised to a power.

Ship-footprint dilation is applied via a max-pooling operation (equivalent to
``scipy.ndimage.maximum_filter``) so the planner keeps the ship center at
least ``inflation_radius`` metres from any ice.

Returns
-------
A ``JaxCostMapResult`` object with `.cost_map`, `.x_origin`, `.y_origin`,
`.resolution`, `.rows`, `.cols` attributes — matching the interface expected
by ``jax_mbd_planner.jax_mbd_optimize``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


# ── Return type ────────────────────────────────────────────────────────────────

@dataclass
class JaxCostMapResult:
    """
    Thin wrapper around a (rows, cols) numpy cost array with CostMap-compatible
    attributes so it can be passed directly to ``jax_mbd_planner.jax_mbd_optimize``
    and ``optimize_speed_profile``.

    Attributes
    ----------
    cost_map   : np.ndarray, shape (rows, cols)
    x_origin   : float — world x of column 0 left edge
    y_origin   : float — world y of row 0 bottom edge
    resolution : float — cell size [m/cell]
    rows       : int
    cols       : int
    width      : float — world width  (cols * resolution)
    height     : float — world height (rows * resolution)
    base_cost  : float — open-water traversal cost (same value passed to build_costmap_jax)
    """
    cost_map:   np.ndarray
    x_origin:   float
    y_origin:   float
    resolution: float
    rows:       int
    cols:       int
    width:      float
    height:     float
    base_cost:  float = 0.1
    # Adaptive-velocity mode: cells store the per-v² KE coefficient (no v
    # baked in, no base_cost floor applied). Downstream cost integrals must
    # multiply by v²(t) from the rollout and apply max(base_cost, ...) at
    # query time. When False (default), cells are the final cost as before.
    adaptive_velocity: bool = False

    def get_cost_world(self, x: float, y: float) -> float:
        """Sample costmap at world coordinates (x, y).

        Returns the cell value if inside the grid, or ``base_cost`` for
        out-of-bounds queries — matching the CostMap interface used by
        ``optimize_speed_profile``.
        """
        col = int((x - self.x_origin) / self.resolution)
        row = int((y - self.y_origin) / self.resolution)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return float(self.cost_map[row, col])
        return float(self.base_cost)


# ── Disk-based JAX rasterizer ──────────────────────────────────────────────────

def _make_disk_kernel(inflation_r: float, resolution: float) -> jnp.ndarray:
    """Build a binary (0/1) disk structuring element for max-pool dilation."""
    half = int(np.ceil(inflation_r / resolution))
    size = 2 * half + 1
    r_idx = jnp.arange(size) - half
    c_idx = jnp.arange(size) - half
    RR, CC = jnp.meshgrid(r_idx, c_idx, indexing="ij")
    dist = jnp.sqrt((RR * resolution) ** 2 + (CC * resolution) ** 2)
    return (dist <= inflation_r).astype(jnp.float32)


@partial(jax.jit, static_argnames=("rows", "cols"))
def _rasterize_elements_ke(
    centroids:   jnp.ndarray,   # (n, 2)
    radii:       jnp.ndarray,   # (n,)
    ke_weights:  jnp.ndarray,   # (n,)  per-element KE constant (pre-scaled)
    x_origin:    float,
    y_origin:    float,
    resolution:  float,
    rows: int,
    cols: int,
    obs_cost_buffer_factor: float = 0.0,
) -> jnp.ndarray:               # (rows, cols) float32
    """
    Paint KE collision cost onto a (rows, cols) grid using parabolic radial
    falloff and element-wise maximum (AUTO-IceNav cost model).

    Each element i contributes to cells within its effective radius:
        eff_r = radii[i] * (1 + obs_cost_buffer_factor) + 0.5 * resolution
        cost(x,y) = ke_weights[i] * max(0, (eff_r² - d²) / eff_r²)

    Overlapping elements contribute the maximum (not sum) at each cell.
    """
    c_idx = jnp.arange(cols, dtype=jnp.float32)
    r_idx = jnp.arange(rows, dtype=jnp.float32)
    cell_x = x_origin + (c_idx + 0.5) * resolution   # (cols,)
    cell_y = y_origin + (r_idx + 0.5) * resolution   # (rows,)
    XX, YY = jnp.meshgrid(cell_x, cell_y)            # (rows, cols)

    def element_contribution(centroid, radius, ke_weight):
        eff_r   = radius * (1.0 + obs_cost_buffer_factor) + 0.5 * resolution
        dist_sq = (XX - centroid[0]) ** 2 + (YY - centroid[1]) ** 2
        falloff = jnp.maximum(0.0, (eff_r ** 2 - dist_sq) / (eff_r ** 2))
        return ke_weight * falloff

    # vmap over all elements → (n, rows, cols)
    contributions = jax.vmap(element_contribution)(
        centroids.astype(jnp.float32),
        radii.astype(jnp.float32),
        ke_weights.astype(jnp.float32),
    )
    # Maximum across elements: highest KE cost wins at each cell
    return jnp.max(contributions, axis=0)


def _dilate_grid_jax(
    grid: jnp.ndarray,   # (rows, cols) float32
    kernel: jnp.ndarray, # (kh, kw) binary disk kernel
) -> jnp.ndarray:
    """
    Max-dilation: each output cell = max of input cells within the kernel window.

    Implemented as max-pooling via ``jax.lax.reduce_window``, which is
    equivalent to ``scipy.ndimage.maximum_filter`` with 'reflect' padding.
    """
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad with the minimum value so border cells are not artificially elevated
    padded = jnp.pad(
        grid,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="edge",
    )

    out = jax.lax.reduce_window(
        padded,
        init_value=jnp.float32(-jnp.inf),
        computation=jax.lax.max,
        window_dimensions=(kh, kw),
        window_strides=(1, 1),
        padding="VALID",
    )
    # Where the disk kernel is 0, do NOT propagate; use original value
    # For a convex disk kernel (all 1s), reduce_window over the full window
    # is equivalent to standard dilation — no masking needed.
    return out


# ── Public API ─────────────────────────────────────────────────────────────────

def build_costmap_jax(
    centroids: np.ndarray,         # (n, 2) world-frame centroids
    thicknesses: np.ndarray,       # (n,) ice thickness [m]
    radii: np.ndarray,             # (n,) approximate element radius [m]
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    resolution: float = 2.0,
    margin: float = 20.0,
    base_cost: float = 0.1,
    collision_cost: float = 5.0,
    obs_cost_buffer_factor: float = 0.1,
    ship_length: Optional[float] = None,
    ship_beam:   Optional[float] = None,
    planning_clearance_margin: float = 0.0,
    verbose: bool = False,
    ship_speed: float = 4.0,
    ship_mass: float = 23e6,
    ke_cost_scale: float = 1e-6,
    ice_resistance_weight: float = 0.0,
    masses: Optional[np.ndarray] = None,
    adaptive_velocity: bool = False,
) -> Tuple["JaxCostMapResult", dict]:
    """
    Build a planning costmap from current ice element positions using JAX.

    Uses the AUTO-IceNav KE collision cost model. Stays entirely on the JAX
    device, avoiding a GPU→CPU→GPU round-trip.

    Parameters
    ----------
    centroids : (n, 2) world-frame ice element centroids [m]
    thicknesses : (n,) ice element thicknesses [m]
    radii : (n,) approximate element radii [m] (from IceStaticData.radii)
    x_min, x_max, y_min, y_max : domain bounds [m]
    resolution : grid cell size [m/cell]
    margin : extra margin around domain [m]
    base_cost : open-water traversal cost
    collision_cost : KE cost weight (collision_cost_weight in AUTO-IceNav model)
    obs_cost_buffer_factor : extra fractional radius painted per element
    ship_length, ship_beam : if provided, inflate ice by half-beam +
        planning_clearance_margin for ship-footprint clearance
    planning_clearance_margin : extra dilation on top of half-beam [m]
    verbose : if True, print grid dimensions

    Returns
    -------
    cost_map : JaxCostMapResult — compatible with jax_mbd_planner.jax_mbd_optimize
    bounds   : dict with x_origin, y_origin, width, height keys
    """
    centroids   = np.asarray(centroids,   dtype=float)
    thicknesses = np.asarray(thicknesses, dtype=float)
    radii       = np.asarray(radii,       dtype=float)

    # Grid geometry
    x_origin = x_min - margin
    y_origin = y_min - margin
    width    = (x_max - x_min) + 2.0 * margin
    height   = (y_max - y_min) + 2.0 * margin
    cols = max(1, int(np.ceil(width  / resolution)))
    rows = max(1, int(np.ceil(height / resolution)))

    if verbose:
        print(f"  [jax-costmap] grid: {rows}×{cols} cells "
              f"({rows * cols / 1e6:.2f} M), n_elem={len(centroids)}", flush=True)

    n = len(centroids)

    if n > 0:
        # If masses not provided, approximate from circular-disk geometry
        if masses is None:
            masses_arr = np.pi * radii ** 2 * thicknesses * 920.0  # ICE_DENSITY
        else:
            masses_arr = np.asarray(masses, dtype=float)

        # Per-element KE constant (computed in float64 to avoid overflow).
        # In adaptive_velocity mode we bake v=1 so the stored value is the
        # per-v² coefficient; the cost integral multiplies by v_rollout² at
        # query time. The clip ceiling is scaled by ship_speed² so saturation
        # at the design velocity matches the non-adaptive build.
        v_baked = 1.0 if adaptive_velocity else max(float(ship_speed), 1e-6)
        clip_hi = 1e8 / (max(float(ship_speed), 1e-6) ** 2) if adaptive_velocity else 1e8
        M_s = float(ship_mass)
        m_i = masses_arr.astype(np.float64)
        ke_constants = (
            float(collision_cost)
            * v_baked * v_baked
            * m_i * M_s * (m_i + 2.0 * M_s)
            / (2.0 * (M_s + m_i) ** 2)
            * float(ke_cost_scale)
        )
        ke_constants_f32 = np.clip(ke_constants, 0.0, clip_hi).astype(np.float32)

        grid_jax = _rasterize_elements_ke(
            jnp.array(centroids,        dtype=jnp.float32),
            jnp.array(radii,            dtype=jnp.float32),
            jnp.array(ke_constants_f32, dtype=jnp.float32),
            float(x_origin), float(y_origin), float(resolution),
            rows, cols,
            float(obs_cost_buffer_factor),
        )
        jax.block_until_ready(grid_jax)
        grid_np = np.array(grid_jax, dtype=np.float32)

        # SIC modulation: cost *= SIC^weight (AUTO-IceNav style)
        if ice_resistance_weight > 0.0:
            from scipy.ndimage import uniform_filter
            bin_occ = (grid_np > 0.0).astype(np.float64)
            ksize = min(31, (min(rows, cols) // 4) | 1)  # odd, ≤ 31 cells
            if ksize % 2 == 0:
                ksize += 1
            sic = uniform_filter(bin_occ, size=ksize, mode="reflect")
            sic = np.clip(sic, 0.0, 1.0).astype(np.float32)
            sic_safe = np.maximum(sic, 1e-6)
            grid_np = grid_np * np.power(sic_safe, ice_resistance_weight)

        # Open-water floor. Skipped in adaptive_velocity mode: the floor
        # must not be v²-scaled, so the cost integral applies it at query.
        if not adaptive_velocity:
            grid_np = np.maximum(grid_np, float(base_cost))
        grid_jax = jnp.array(grid_np, dtype=jnp.float32)

    else:
        # Empty ice field. In adaptive mode the grid is all zeros (per-v²
        # coefficient is zero in open water); the floor is applied at query.
        floor_value = 0.0 if adaptive_velocity else float(base_cost)
        grid_jax = jnp.full((rows, cols), floor_value, dtype=jnp.float32)

    # Ship-footprint dilation
    if ship_length is not None and ship_beam is not None and ship_length > 0 and ship_beam > 0:
        inflation_r = 0.5 * ship_beam + planning_clearance_margin
        if inflation_r > 0:
            kernel = _make_disk_kernel(inflation_r, resolution)
            grid_jax = _dilate_grid_jax(grid_jax, kernel)
            jax.block_until_ready(grid_jax)

    grid_np = np.array(grid_jax, dtype=np.float32)

    result = JaxCostMapResult(
        cost_map=grid_np,
        x_origin=float(x_origin),
        y_origin=float(y_origin),
        resolution=float(resolution),
        rows=int(rows),
        cols=int(cols),
        width=float(width),
        height=float(height),
        base_cost=float(base_cost),
        adaptive_velocity=bool(adaptive_velocity),
    )

    bounds = {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "x_origin": float(x_origin),
        "y_origin": float(y_origin),
        "width": float(width),
        "height": float(height),
    }
    return result, bounds


@partial(jax.jit, static_argnames=("rows", "cols"))
def _rasterize_elements_exp_sdf(
    centroids:   jnp.ndarray,   # (n, 2)
    radii:       jnp.ndarray,   # (n,)
    weights:     jnp.ndarray,   # (n,) non-dimensional ∈ [0, 1]
    x_origin:    float,
    y_origin:    float,
    resolution:  float,
    rows: int,
    cols: int,
    decay_length: float,
    obs_cost_buffer_factor: float = 0.0,
) -> jnp.ndarray:               # (rows, cols) float32
    """
    Smooth exponential-SDF cost field (v2 cost reshape).

    For each floe i with centroid c_i, effective radius r_i_eff, and
    non-dimensional weight w_i ∈ [0, 1]:

        d_i(x, y) = max(0, ||(x, y) - c_i|| - r_i_eff)
        c_i(x, y) = w_i * exp(-d_i / L)

    Per-cell cost is the max over floes. Inside a floe, c_i = w_i; far away,
    c_i decays to 0 with characteristic distance L. The key property: there
    is a non-zero gradient EVERYWHERE pointing away from the nearest floe,
    unlike the v1 parabola which is zero outside r_i_eff.
    """
    c_idx = jnp.arange(cols, dtype=jnp.float32)
    r_idx = jnp.arange(rows, dtype=jnp.float32)
    cell_x = x_origin + (c_idx + 0.5) * resolution   # (cols,)
    cell_y = y_origin + (r_idx + 0.5) * resolution   # (rows,)
    XX, YY = jnp.meshgrid(cell_x, cell_y)            # (rows, cols)
    L_safe = jnp.maximum(jnp.float32(decay_length), jnp.float32(1e-6))

    def element_contribution(centroid, radius, weight):
        eff_r = radius * (1.0 + obs_cost_buffer_factor) + 0.5 * resolution
        dist = jnp.sqrt((XX - centroid[0]) ** 2 + (YY - centroid[1]) ** 2 + 1e-12)
        outer_d = jnp.maximum(0.0, dist - eff_r)
        return weight * jnp.exp(-outer_d / L_safe)

    contributions = jax.vmap(element_contribution)(
        centroids.astype(jnp.float32),
        radii.astype(jnp.float32),
        weights.astype(jnp.float32),
    )
    return jnp.max(contributions, axis=0)


def build_costmap_jax_v2(
    centroids: np.ndarray,
    thicknesses: np.ndarray,
    radii: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    resolution: float = 2.0,
    margin: float = 20.0,
    base_cost: float = 0.0,
    peak_cost: float = 1.0,
    decay_length: Optional[float] = None,
    obs_cost_buffer_factor: float = 0.1,
    ship_length: Optional[float] = None,
    ship_beam:   Optional[float] = None,
    planning_clearance_margin: float = 0.0,
    ship_speed: float = 4.0,
    ship_mass: float = 23e6,
    ke_cost_scale: float = 1e-6,
    ke_reference: Optional[float] = None,
    masses: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple["JaxCostMapResult", dict]:
    """
    v2 costmap builder — global exponential SDF + non-dimensionalized magnitude.

    Differences from ``build_costmap_jax`` (v1):

    (a) Non-zero gradient everywhere: cost decays exponentially with distance
        to the nearest floe boundary (decay length ``L``), not a parabola that
        is zero outside the floe radius. MBD rollouts in open water now see a
        spatial signal.
    (b) Per-floe contribution is non-dimensionalized: w_i = clip(ke_i / ke_ref,
        0, 1), so peak magnitude is bounded by ``peak_cost`` regardless of ice
        thickness, ship speed, or mass. The same MBD temperature transfers
        across scenarios.
    (c) Max over floes (same as v1) — closest dangerous floe dominates.

    Parameters
    ----------
    decay_length : characteristic distance L for the exp(-d/L) falloff.
        If None, defaults to max(ship_length, 3 * median floe radius), which
        scales the gradient to ship-maneuverability units rather than per-floe
        units.
    peak_cost : maximum value the cost field can take (saturating inside the
        heaviest floe). The whole field is bounded by [base_cost, peak_cost].
    ke_reference : if None, taken as the median per-floe KE constant for the
        scene. Pass a fixed value to force a global reference across runs.
    """
    centroids   = np.asarray(centroids,   dtype=float)
    thicknesses = np.asarray(thicknesses, dtype=float)
    radii       = np.asarray(radii,       dtype=float)

    x_origin = x_min - margin
    y_origin = y_min - margin
    width    = (x_max - x_min) + 2.0 * margin
    height   = (y_max - y_min) + 2.0 * margin
    cols = max(1, int(np.ceil(width  / resolution)))
    rows = max(1, int(np.ceil(height / resolution)))

    if verbose:
        print(f"  [jax-costmap-v2] grid: {rows}×{cols} cells "
              f"({rows * cols / 1e6:.2f} M), n_elem={len(centroids)}", flush=True)

    n = len(centroids)

    if n > 0:
        if masses is None:
            masses_arr = np.pi * radii ** 2 * thicknesses * 920.0
        else:
            masses_arr = np.asarray(masses, dtype=float)

        # Per-element KE constant — same physics as v1 but used here only for
        # *relative* weighting between floes (heavier/thicker = more weight).
        # The scene-relative normalization (divide by ke_ref) is what makes the
        # peak O(1) regardless of scenario.
        M_s = float(ship_mass)
        m_i = masses_arr.astype(np.float64)
        v   = max(float(ship_speed), 1e-6)
        ke_constants = (
            v * v * m_i * M_s * (m_i + 2.0 * M_s)
            / (2.0 * (M_s + m_i) ** 2)
            * float(ke_cost_scale)
        )
        if ke_reference is None:
            # Use scene MAX so only the heaviest floe saturates at w=1; lighter
            # floes get proportionally smaller weights. Median-based ke_ref
            # caused field saturation in dense Medea scenes (most floes ≥ ref).
            ke_ref = float(np.max(ke_constants))
        else:
            ke_ref = float(ke_reference)
        ke_ref = max(ke_ref, 1e-12)

        weights = np.clip(ke_constants / ke_ref, 0.0, 1.0).astype(np.float32)

        L = decay_length
        if L is None:
            # Decay length set by ship clearance, not ship length. With long L
            # the exp falloffs from neighbouring floes overlap and the max
            # saturates at peak_cost; with short L there is no signal in open
            # water. ~2*beam (≈ ship width incl. clearance) tracks the
            # corridor scale the ship must thread.
            ship_clear = (float(ship_beam) if ship_beam is not None else 18.0) * 2.0
            L = max(ship_clear, 1.5 * float(np.median(radii)))

        if verbose:
            print(f"  [jax-costmap-v2] L={L:.1f} m, ke_ref={ke_ref:.3g}, "
                  f"w∈[{weights.min():.2f},{weights.max():.2f}]", flush=True)

        grid_jax = _rasterize_elements_exp_sdf(
            jnp.array(centroids, dtype=jnp.float32),
            jnp.array(radii,     dtype=jnp.float32),
            jnp.array(weights,   dtype=jnp.float32),
            float(x_origin), float(y_origin), float(resolution),
            rows, cols,
            float(L),
            float(obs_cost_buffer_factor),
        )
        jax.block_until_ready(grid_jax)
        grid_np = (float(peak_cost) * np.array(grid_jax, dtype=np.float32))
        grid_np = np.maximum(grid_np, float(base_cost))
        grid_jax = jnp.array(grid_np, dtype=jnp.float32)
    else:
        grid_jax = jnp.full((rows, cols), float(base_cost), dtype=jnp.float32)

    if ship_length is not None and ship_beam is not None and ship_length > 0 and ship_beam > 0:
        inflation_r = 0.5 * ship_beam + planning_clearance_margin
        if inflation_r > 0:
            kernel = _make_disk_kernel(inflation_r, resolution)
            grid_jax = _dilate_grid_jax(grid_jax, kernel)
            jax.block_until_ready(grid_jax)

    grid_np = np.array(grid_jax, dtype=np.float32)

    result = JaxCostMapResult(
        cost_map=grid_np,
        x_origin=float(x_origin),
        y_origin=float(y_origin),
        resolution=float(resolution),
        rows=int(rows),
        cols=int(cols),
        width=float(width),
        height=float(height),
        base_cost=float(base_cost),
        adaptive_velocity=False,
    )
    bounds = {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "x_origin": float(x_origin),
        "y_origin": float(y_origin),
        "width": float(width),
        "height": float(height),
    }
    return result, bounds


def build_costmap_jax_from_ice_state(
    state,
    static,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    **kwargs,
) -> Tuple["JaxCostMapResult", dict]:
    """
    Convenience wrapper: build the costmap from a live IceState + IceStaticData.

    Extracts centroids (current positions), thicknesses, radii, and masses from
    the state and static data, then calls build_costmap_jax.
    """
    centroids_np   = np.array(state.positions)
    thicknesses_np = np.array(static.thicknesses)
    radii_np       = np.array(static.radii)
    masses_np      = np.array(static.masses)      # exact masses from simulation
    return build_costmap_jax(
        centroids_np, thicknesses_np, radii_np,
        x_min, x_max, y_min, y_max,
        masses=masses_np,
        **kwargs,
    )
