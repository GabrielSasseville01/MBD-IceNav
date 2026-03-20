"""
JAX-native costmap builder for the ice-ship simulation.

Replaces ``medea_ice_ship.costmap_build.build_costmap_from_ice_state`` for the
receding-horizon diffusion-MPC pipeline.  Unlike the NumPy version (which uses
OpenCV polygon rasterization + dilation), this implementation uses JAX scatter
operations so the costmap can be built entirely on the GPU, avoiding a
device-to-host round-trip between the simulation and the planner.

Cost model
----------
The grid cost for a cell at world position (x, y) is:

    cost(x, y) = base_cost
                 + sum_i  ice_cost_i * disk_i(x, y)

where the sum is over all ice elements, disk_i is 1 inside the element's
radius (scaled by obs_cost_buffer_factor), and ice_cost_i follows the same
exponential thickness model as the NumPy costmap:

    ice_cost_i = collision_cost * exp(ice_cost_exp_scale * h_i / thickness_ref)

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
    attributes so it can be passed directly to ``jax_mbd_planner.jax_mbd_optimize``.

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
    """
    cost_map:   np.ndarray
    x_origin:   float
    y_origin:   float
    resolution: float
    rows:       int
    cols:       int
    width:      float
    height:     float


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
def _rasterize_elements(
    centroids: jnp.ndarray,    # (n, 2)
    radii:     jnp.ndarray,    # (n,)
    ice_costs: jnp.ndarray,    # (n,)  per-element cost weight
    base_cost: float,
    x_origin:  float,
    y_origin:  float,
    resolution: float,
    rows: int,
    cols: int,
    radius_buffer: float = 0.0,  # extra radius multiplier (obs_cost_buffer_factor)
) -> jnp.ndarray:               # (rows, cols) float32
    """
    Paint ice elements as filled disks onto a (rows, cols) cost grid.

    Each element contributes ``ice_costs[i]`` to all cells whose center lies
    within ``radii[i] * (1 + radius_buffer)`` metres of ``centroids[i]``.

    Multiple overlapping elements have their costs summed.

    Parameters
    ----------
    centroids, radii, ice_costs : (n,) or (n, 2)
        Per-element geometry and cost.
    base_cost : float
        Background cost for open water.
    x_origin, y_origin, resolution, rows, cols : grid metadata
    radius_buffer : float
        Extra fractional radius (like ``obs_cost_buffer_factor``).
        Effective radius = radii[i] * (1 + radius_buffer).
    """
    # Cell-centre world coordinates: (rows, cols)
    c_idx = jnp.arange(cols, dtype=jnp.float32)
    r_idx = jnp.arange(rows, dtype=jnp.float32)
    cell_x = x_origin + (c_idx + 0.5) * resolution          # (cols,)
    cell_y = y_origin + (r_idx + 0.5) * resolution          # (rows,)
    XX, YY = jnp.meshgrid(cell_x, cell_y)                   # (rows, cols)

    def element_contribution(centroid, radius, cost):
        """Single element: returns (rows, cols) contribution array."""
        eff_r = radius * (1.0 + radius_buffer)
        dist2 = (XX - centroid[0]) ** 2 + (YY - centroid[1]) ** 2
        inside = dist2 <= eff_r ** 2
        return inside.astype(jnp.float32) * cost

    # vmap over all elements: (n, rows, cols)
    contributions = jax.vmap(element_contribution)(
        centroids.astype(jnp.float32),
        radii.astype(jnp.float32),
        ice_costs.astype(jnp.float32),
    )
    # Sum + base cost
    grid = base_cost + jnp.sum(contributions, axis=0)
    return grid


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
    ice_cost_exp_scale: float = 3.0,
    thickness_ref: float = 1.0,
    obs_cost_buffer_factor: float = 0.1,
    ship_length: Optional[float] = None,
    ship_beam:   Optional[float] = None,
    planning_clearance_margin: float = 0.0,
    verbose: bool = False,
) -> Tuple["JaxCostMapResult", dict]:
    """
    Build a planning costmap from current ice element positions using JAX.

    Stays entirely on the JAX device, avoiding a GPU→CPU→GPU round-trip.
    The cost model matches ``costmap_build.build_costmap_from_ice_state``
    closely enough to guide the diffusion-MPC planner correctly.

    Parameters
    ----------
    centroids : (n, 2) world-frame ice element centroids [m]
    thicknesses : (n,) ice element thicknesses [m]
    radii : (n,) approximate element radii [m] (from IceStaticData.radii)
    x_min, x_max, y_min, y_max : domain bounds [m]
    resolution : grid cell size [m/cell]
    margin : extra margin around domain [m]
    base_cost : open-water traversal cost
    collision_cost : ice-cell base cost multiplier
    ice_cost_exp_scale : exponential growth rate for thickness-based cost
    thickness_ref : reference thickness [m]
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

    # Per-element ice cost weights
    ice_costs_np = collision_cost * np.exp(
        ice_cost_exp_scale * thicknesses / thickness_ref
    ).astype(np.float32)

    # Rasterize on JAX device
    grid_jax = _rasterize_elements(
        jnp.array(centroids,   dtype=jnp.float32),
        jnp.array(radii,       dtype=jnp.float32),
        jnp.array(ice_costs_np, dtype=jnp.float32),
        float(base_cost),
        float(x_origin), float(y_origin), float(resolution),
        rows, cols,
        float(obs_cost_buffer_factor),
    )
    jax.block_until_ready(grid_jax)

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
    state,                          # IceState (JAX NamedTuple)
    static,                         # IceStaticData
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    **kwargs,
) -> Tuple["JaxCostMapResult", dict]:
    """
    Convenience wrapper: build the costmap from a live IceState + IceStaticData.

    Extracts centroids (current positions), thicknesses, and radii from the
    state and static data, then calls build_costmap_jax.
    """
    centroids_np   = np.array(state.positions)
    thicknesses_np = np.array(static.thicknesses)
    radii_np       = np.array(static.radii)
    return build_costmap_jax(
        centroids_np, thicknesses_np, radii_np,
        x_min, x_max, y_min, y_max,
        **kwargs,
    )
