"""
JAX-compatible state representations for the 2-D ice-ship simulation.

All mutable simulation state is collected into NamedTuples so that
jax.lax.scan / jax.jit can carry and transform them without retracing.

Immutable per-simulation data (padded polygon geometry, masses, bond
topology, …) lives in IceStaticData and is captured as a closure inside
jit-compiled functions.

Usage
-----
from medea_ice_ship.jax_ice_state import (
    IceState, IceStaticData, ShipDynamicsData,
    build_static_data, build_initial_ice_state, build_ship_dynamics_data,
)
"""

from typing import NamedTuple, List, Tuple, Optional
import numpy as np
import jax.numpy as jnp

from medea_ice_ship.geometry import polygon_area, polygon_centroid
from medea_ice_ship.properties import IceProperties


# ──────────────────────────────────────────────────────────────────────────────
# Mutable state (carried through jax.lax.scan / returned from @jax.jit)
# ──────────────────────────────────────────────────────────────────────────────

class IceState(NamedTuple):
    """All mutable ice-field state at a single simulation instant."""
    positions:          jnp.ndarray  # (n, 2)  ice centroid world positions [m]
    velocities:         jnp.ndarray  # (n, 2)  ice centroid velocities [m/s]
    rotations:          jnp.ndarray  # (n,)    ice element yaw angles [rad]
    angular_vels:       jnp.ndarray  # (n,)    ice element angular velocities [rad/s]
    submergence:        jnp.ndarray  # (n,)    vertical submergence depth [m]
    bond_intact:        jnp.ndarray  # (n_bonds,) bool — True while bond not failed
    bond_damage:        jnp.ndarray  # (n_bonds,) damage variable D ∈ [0, 1]
    bond_delta_m_0:     jnp.ndarray  # (n_bonds,) equivalent deformation at damage onset
    bond_delta_m_max:   jnp.ndarray  # (n_bonds,) historical max equivalent deformation
    bond_delta_s:       jnp.ndarray  # (n_bonds,) accumulated tangential deformation [m]
    elem_active:        jnp.ndarray  # (n,) bool — True once the ship has touched this floe's parent
    dpdem_floe_eligible: jnp.ndarray # (n_floes,) bool — True once collision force exceeds fracture threshold


class ShipState(NamedTuple):
    """Mutable ship state (pose + velocity + controller integrals)."""
    eta:      jnp.ndarray  # (3,) [x, y, psi] world-frame pose [m, m, rad]
    nu:       jnp.ndarray  # (3,) [u, v, r] body-frame velocity [m/s, m/s, rad/s]
    n_actual: jnp.ndarray  # (6,) propeller RPMs (AutoIceNav path; zeros otherwise)
    e_int:    jnp.ndarray  # (3,) integral error accumulator for DP controller


# ──────────────────────────────────────────────────────────────────────────────
# Immutable per-simulation data (captured in closures, never updated)
# ──────────────────────────────────────────────────────────────────────────────

class IceStaticData(NamedTuple):
    """
    Precomputed, immutable data for the ice field.

    Polygon vertices are padded to a uniform shape (n, max_V, 2) so JAX can
    treat them as a regular array.  `poly_mask` indicates which entries are
    real vertices (True) vs. padding (False).
    """
    # Geometry (padded)
    local_polys_padded: jnp.ndarray  # (n, max_V, 2) local (body-frame) vertices
    poly_mask:          jnp.ndarray  # (n, max_V) bool  True = real vertex
    poly_n_verts:       jnp.ndarray  # (n,) int  actual vertex count per element

    # Physical properties
    masses:      jnp.ndarray  # (n,)
    inertia:     jnp.ndarray  # (n,)
    radii:       jnp.ndarray  # (n,)  effective radius for ice-ice contact
    areas:       jnp.ndarray  # (n,)
    thicknesses: jnp.ndarray  # (n,)

    # Bond topology & material
    bond_i:        jnp.ndarray  # (n_bonds,) int  index of first bonded element
    bond_j:        jnp.ndarray  # (n_bonds,) int  index of second bonded element
    rest_lengths:  jnp.ndarray  # (n_bonds,) equilibrium bond length [m]
    bond_strengths: jnp.ndarray # (n_bonds,) breaking force [N]
    bond_areas:    jnp.ndarray  # (n_bonds,) cross-sectional area [m²]
    sigma_bar_bond: jnp.ndarray # (n_bonds,) normal strength [Pa]

    # Ice-ice contact pairs (upper-triangle, precomputed once)
    pair_i: jnp.ndarray  # (n_pairs,) int
    pair_j: jnp.ndarray  # (n_pairs,) int

    # Floe ancestry (for activation gating)
    initial_floe_of: jnp.ndarray  # (n,) int  parent-floe index for each element

    # DPDEM bond stiffness constants
    k_n: float  # normal stiffness scale [Pa]
    k_s: float  # shear stiffness scale [Pa]


class ShipDynamicsData(NamedTuple):
    """Precomputed ship dynamics constants (mass matrix inverse, gain matrices, …)."""
    M3inv:  jnp.ndarray  # (3, 3)  inverse of mass + added-mass matrix
    D3:     jnp.ndarray  # (3, 3)  linear damping matrix
    mass:   float        # ship mass [kg]
    Izz:    float        # yaw moment of inertia [kg·m²]

    # AutoIceNav DP controller (pre-computed gains)
    Kp:     jnp.ndarray  # (3, 3)  proportional gain matrix
    Kd:     jnp.ndarray  # (3, 3)  derivative gain matrix
    Ki:     jnp.ndarray  # (3, 3)  integral gain matrix (optional; zeros if not used)
    B_pinv: jnp.ndarray  # (6, 3)  pseudoinverse of thruster config matrix
    n_max:  jnp.ndarray  # (6,)    maximum propeller RPM

    # Fixed hull geometry in local (body) frame — used for hull-vs-ice PIP
    ship_hull_local: jnp.ndarray  # (S, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Builder functions  (NumPy  →  JAX arrays, called once at simulation start)
# ──────────────────────────────────────────────────────────────────────────────

def _pad_polygons(
    local_polys: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad a list of (V_i, 2) polygon arrays to a uniform (n, max_V, 2) array.

    Returns
    -------
    padded : (n, max_V, 2)  — padding repeats the last real vertex
    mask   : (n, max_V) bool — True for real vertices
    n_verts: (n,) int        — actual vertex count per element
    """
    n = len(local_polys)
    n_verts = np.array([len(p) for p in local_polys], dtype=np.int32)
    max_V = int(n_verts.max())

    padded = np.zeros((n, max_V, 2), dtype=np.float64)
    mask = np.zeros((n, max_V), dtype=bool)

    for i, poly in enumerate(local_polys):
        nv = len(poly)
        padded[i, :nv] = poly
        # Repeat last real vertex for padding positions
        padded[i, nv:] = poly[-1]
        mask[i, :nv] = True

    return padded, mask, n_verts


def build_static_data(
    polygons: List[np.ndarray],    # world-frame vertices per floe
    centroids: np.ndarray,         # (n, 2)
    bonds: List[Tuple[int, int]],  # [(i, j), ...]
    thicknesses: np.ndarray,       # (n,)
    parent_ids: Optional[np.ndarray],  # (n,) or None
    cfg,                           # SimulationConfig
) -> IceStaticData:
    """
    Build the IceStaticData NamedTuple from raw simulation inputs.

    This is called once at simulation start (outside any JIT boundary).
    """
    n = len(polygons)

    # ── Geometry ──────────────────────────────────────────────────────────────
    areas = np.array([polygon_area(p) for p in polygons], dtype=np.float64)
    radii = np.sqrt(areas / np.pi)
    masses = areas * thicknesses * IceProperties.ICE_DENSITY
    inertia = np.maximum(0.5 * masses * radii ** 2, 1e-10)

    # Local (body-frame) polygon vertices (subtract centroid)
    local_polys = []
    for poly, c in zip(polygons, centroids):
        local_polys.append((poly - polygon_centroid(poly)).astype(np.float64))

    padded, mask, n_verts = _pad_polygons(local_polys)

    # ── Bonds ──────────────────────────────────────────────────────────────────
    n_bonds = len(bonds)
    if n_bonds > 0:
        bond_i_arr = np.array([b[0] for b in bonds], dtype=np.int32)
        bond_j_arr = np.array([b[1] for b in bonds], dtype=np.int32)
        rest_lengths = np.array(
            [np.linalg.norm(centroids[i] - centroids[j]) for i, j in bonds],
            dtype=np.float64,
        )
        bond_strengths_arr = np.zeros(n_bonds, dtype=np.float64)
        bond_areas_arr = np.zeros(n_bonds, dtype=np.float64)
        for b, (i, j) in enumerate(bonds):
            h = min(thicknesses[i], thicknesses[j])
            w = min((radii[i] + radii[j]) / 2, cfg.max_bond_width)
            bond_strengths_arr[b] = IceProperties.breaking_force(h, w)
            bond_areas_arr[b] = w * h
        sigma_bar_bond = np.maximum(
            bond_strengths_arr / np.maximum(bond_areas_arr, 1e-10), 1e6
        )
    else:
        bond_i_arr = np.zeros(0, dtype=np.int32)
        bond_j_arr = np.zeros(0, dtype=np.int32)
        rest_lengths = np.zeros(0, dtype=np.float64)
        bond_strengths_arr = np.zeros(0, dtype=np.float64)
        bond_areas_arr = np.zeros(0, dtype=np.float64)
        sigma_bar_bond = np.zeros(0, dtype=np.float64)

    # ── DPDEM stiffness ────────────────────────────────────────────────────────
    nu_r = getattr(cfg, "poisson_ratio", 0.3)
    E = IceProperties.YOUNGS_MODULUS
    prefactor = E * (1.0 - nu_r) / ((1.0 + nu_r) * (1.0 - 2.0 * nu_r))
    k_n = prefactor
    k_s = prefactor * (1.0 - 2.0 * nu_r) / (2.0 * (1.0 - nu_r))

    # ── Ice-ice contact pairs (upper triangle) ─────────────────────────────────
    pair_i_arr = np.repeat(np.arange(n), np.arange(n - 1, -1, -1))
    pair_j_arr = np.concatenate([np.arange(i + 1, n) for i in range(n)])

    # ── Parent-floe ancestry ──────────────────────────────────────────────────
    if parent_ids is not None:
        initial_floe_of = np.asarray(parent_ids, dtype=np.int32)
    else:
        # Each element is its own "floe" (no tessellation info)
        initial_floe_of = np.arange(n, dtype=np.int32)

    return IceStaticData(
        local_polys_padded=jnp.array(padded),
        poly_mask=jnp.array(mask),
        poly_n_verts=jnp.array(n_verts),
        masses=jnp.array(masses),
        inertia=jnp.array(inertia),
        radii=jnp.array(radii),
        areas=jnp.array(areas),
        thicknesses=jnp.array(thicknesses),
        bond_i=jnp.array(bond_i_arr),
        bond_j=jnp.array(bond_j_arr),
        rest_lengths=jnp.array(rest_lengths),
        bond_strengths=jnp.array(bond_strengths_arr),
        bond_areas=jnp.array(bond_areas_arr),
        sigma_bar_bond=jnp.array(sigma_bar_bond),
        pair_i=jnp.array(pair_i_arr),
        pair_j=jnp.array(pair_j_arr),
        initial_floe_of=jnp.array(initial_floe_of),
        k_n=float(k_n),
        k_s=float(k_s),
    )


def build_initial_ice_state(
    n: int,
    centroids: np.ndarray,   # (n, 2)
    n_bonds: int,
    initial_floe_of: np.ndarray,  # (n,) int
) -> IceState:
    """Build the initial (zero-velocity, undamaged) IceState."""
    n_floes = int(initial_floe_of.max()) + 1 if n > 0 else 0
    return IceState(
        positions=jnp.array(centroids, dtype=jnp.float64),
        velocities=jnp.zeros((n, 2), dtype=jnp.float64),
        rotations=jnp.zeros(n, dtype=jnp.float64),
        angular_vels=jnp.zeros(n, dtype=jnp.float64),
        submergence=jnp.zeros(n, dtype=jnp.float64),
        bond_intact=jnp.ones(n_bonds, dtype=bool),
        bond_damage=jnp.zeros(n_bonds, dtype=jnp.float64),
        bond_delta_m_0=jnp.zeros(n_bonds, dtype=jnp.float64),
        bond_delta_m_max=jnp.zeros(n_bonds, dtype=jnp.float64),
        bond_delta_s=jnp.zeros(n_bonds, dtype=jnp.float64),
        elem_active=jnp.zeros(n, dtype=bool),
        dpdem_floe_eligible=jnp.zeros(n_floes, dtype=bool),
    )


def build_ship_dynamics_data(
    ship_props,    # ShipProperties object
    cfg,           # SimulationConfig
    ship_length: float,
    ship_beam: float,
) -> ShipDynamicsData:
    """
    Build ShipDynamicsData from ship properties and config.

    Uses the Fossen bis-scaled supply vessel model (matching AUTO-IceNav / paper):
      M3 = mass * Tbis_inv @ Mbis @ Tbis_inv
      D3 = mass * sqrt(g/L) * Tbis_inv @ Dbis @ Tbis_inv
    Controller gains from pole placement:
      Kp = wn^2 @ diag(M3),  Kd = 2*zeta*wn @ diag(M3) - diag(D3)
    Integral term disabled (Ki = 0) to match AUTO-IceNav default.
    """
    from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF

    m = ship_props.mass
    Izz = (1.0 / 12.0) * m * (ship_length ** 2 + ship_beam ** 2)

    # Fossen bis-scaled matrices — matches AUTO-IceNav Supply exactly
    _ship = FossenSupply3DOF(mass=m, length=ship_length)
    M3    = _ship.M3
    D3    = _ship.D3
    M3inv = _ship.M3inv

    # Pole-placement gains (matching AUTO-IceNav DPpolePlacement)
    M3_diag = np.diag(np.diag(M3))
    D3_diag = np.diag(np.diag(D3))
    Kp = _ship.wn @ _ship.wn @ M3_diag
    Kd = 2.0 * _ship.zeta @ _ship.wn @ M3_diag - D3_diag
    Ki = np.zeros((3, 3))  # integral disabled (commented out in AUTO-IceNav)

    # Thruster config from FossenSupply3DOF
    B      = _ship.B                                    # (3, 6)
    B_pinv = _ship.B.T @ np.linalg.inv(_ship.B @ _ship.B.T)  # (6, 3)
    n_max  = _ship.n_max                                # (6,)

    # Build ship hull polygon in local frame (for PIP tests)
    n_pts = 30
    half_len = ship_length / 2
    half_beam = ship_beam / 2
    lx, ly = [], []
    for i in range(n_pts + 1):
        t = i / n_pts
        x_loc = half_len - t * ship_length
        x_rel = x_loc / half_len
        taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5 if x_rel > 0.6 else 1.0
        taper = max(taper, 0.02)
        lx.append(x_loc); ly.append(half_beam * taper)
    for i in range(n_pts, -1, -1):
        t = i / n_pts
        x_loc = half_len - t * ship_length
        x_rel = x_loc / half_len
        taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5 if x_rel > 0.6 else 1.0
        taper = max(taper, 0.02)
        lx.append(x_loc); ly.append(-half_beam * taper)
    ship_hull_local = np.column_stack([lx, ly])

    return ShipDynamicsData(
        M3inv=jnp.array(M3inv),
        D3=jnp.array(D3),
        mass=float(_ship.mass),
        Izz=float(Izz),
        Kp=jnp.array(Kp),
        Kd=jnp.array(Kd),
        Ki=jnp.array(Ki),
        B_pinv=jnp.array(B_pinv),
        n_max=jnp.array(n_max),
        ship_hull_local=jnp.array(ship_hull_local),
    )


def build_initial_ship_state(
    ship_pos: np.ndarray,   # (2,)
    ship_heading: float,
) -> ShipState:
    """Zero-velocity initial ship state."""
    eta = jnp.array([ship_pos[0], ship_pos[1], ship_heading], dtype=jnp.float64)
    return ShipState(
        eta=eta,
        nu=jnp.zeros(3, dtype=jnp.float64),
        n_actual=jnp.zeros(6, dtype=jnp.float64),
        e_int=jnp.zeros(3, dtype=jnp.float64),
    )
