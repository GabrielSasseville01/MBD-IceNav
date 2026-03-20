"""
JAX-vectorized 2D ice-ship interaction simulation.

Drop-in replacement for medea_ice_ship.simulation.simulate_ice_ship_interaction
that compiles the inner physics loop with JAX JIT and runs the per-element
dynamics on GPU (or fast CPU XLA) rather than Python loops.

Architecture
------------
Python frame loop (outer)
  ├─ find_connected_components()        NumPy BFS — cannot JIT
  ├─ ocean_current.sample_batch()       NumPy / PyTorch — stays Python
  ├─ replan_callback()                  Python callback
  └─ _frame_step_jax(...)               @jax.jit compiled per-frame call
        ├─ compute_all_collisions_combined()   [jax_geometry.py]
        ├─ apply_thrust_cap_jax()
        ├─ update_dpdem_eligibility_jax()
        ├─ jax.lax.scan over n_ice_steps
        │     └─ _substep():
        │           ├─ _compute_bond_forces_jax()
        │           ├─ _compute_ice_ice_forces_jax()
        │           ├─ _compute_drag_jax()
        │           ├─ _integrate_per_element_jax()
        │           └─ _update_bonds_jax()
        └─ ship_step_jax()              [jax_vessel_dynamics.py]

Key simplifications vs. NumPy simulation
-----------------------------------------
1. Per-element integration instead of rigid-body component grouping.
   (component grouping requires variable-size Python lists — not JAX compatible)
2. Polygon collision uses padded uniform arrays (n, max_V, 2) with masking.
3. np.add.at scatter → jnp .at[].add() functional scatter.
4. All if/else branches → jnp.where.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Optional, Callable

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from medea_ice_ship.properties import IceProperties, R_xy, ssa
from medea_ice_ship.config import SimulationConfig, get_default_simulation_config
from medea_ice_ship.ocean_current import OceanCurrentModel, NoCurrentModel
from medea_ice_ship.geometry import create_ship_hull_at_position_heading

from medea_ice_ship.jax_ice_state import (
    IceState, IceStaticData, ShipDynamicsData, ShipState,
    build_static_data, build_initial_ice_state,
    build_ship_dynamics_data, build_initial_ship_state,
)
from medea_ice_ship.jax_bond_fracture import update_bond_damage_state
from medea_ice_ship.jax_geometry import compute_all_collisions_combined
from medea_ice_ship.jax_vessel_dynamics import ship_step_jax


# ── Inner physics sub-functions (all @jax.jit) ────────────────────────────────

@jax.jit
def _compute_bond_forces_jax(
    state: IceState,
    static: IceStaticData,
    k_n: float,
    bond_stiffness_factor: float,
) -> tuple:  # (forces (n,2), moments (n,))
    """Bond spring forces including damage softening (Eq. 7, bond term)."""
    n = state.positions.shape[0]
    n_bonds = static.bond_i.shape[0]

    if n_bonds == 0:
        return jnp.zeros((n, 2)), jnp.zeros(n)

    d_bond = state.positions[static.bond_j] - state.positions[static.bond_i]  # (n_bonds, 2)
    dist_bond = jnp.linalg.norm(d_bond, axis=1)
    dist_safe = jnp.maximum(dist_bond, 1e-10)
    bond_dir = d_bond / dist_safe[:, None]

    avg_thickness = (static.thicknesses[static.bond_i] + static.thicknesses[static.bond_j]) * 0.5
    stiffness = IceProperties.YOUNGS_MODULUS * avg_thickness * bond_stiffness_factor
    stretch = dist_bond - static.rest_lengths
    F_mag = stiffness * stretch * (1.0 - state.bond_damage)

    # Zero out broken bonds
    F_vec = bond_dir * F_mag[:, None]
    F_vec = jnp.where(state.bond_intact[:, None], F_vec, 0.0)

    # Only apply forces when both endpoints have the same active status
    both_active = state.elem_active[static.bond_i] & state.elem_active[static.bond_j]
    both_frozen = (~state.elem_active[static.bond_i]) & (~state.elem_active[static.bond_j])
    bond_force_ok = both_active | both_frozen
    F_vec = F_vec * bond_force_ok[:, None].astype(jnp.float64)

    # Scatter forces
    forces = jnp.zeros((n, 2))
    forces = forces.at[static.bond_i].add(F_vec)
    forces = forces.at[static.bond_j].add(-F_vec)

    # Bond torque (r × F, 2D cross product gives scalar)
    bond_cross = 0.5 * (d_bond[:, 0] * F_vec[:, 1] - d_bond[:, 1] * F_vec[:, 0])
    moments = jnp.zeros(n)
    moments = moments.at[static.bond_i].add(bond_cross)
    moments = moments.at[static.bond_j].add(-bond_cross)

    return forces, moments


@jax.jit
def _compute_ice_ice_forces_jax(
    state: IceState,
    static: IceStaticData,
    hertz_factor: float,
    hertz_exponent: float,
    ice_ice_friction: float,
    friction_cap: float,
) -> tuple:  # (forces (n,2), moments (n,))
    """Ice-ice contact forces via Hertz repulsion + Coulomb friction (Eq. 7, contact term)."""
    n = state.positions.shape[0]
    n_pairs = static.pair_i.shape[0]

    if n_pairs == 0:
        return jnp.zeros((n, 2)), jnp.zeros(n)

    d_vec = state.positions[static.pair_j] - state.positions[static.pair_i]  # (n_pairs, 2)
    dist_ij = jnp.linalg.norm(d_vec, axis=1)
    min_dist = static.radii[static.pair_i] + static.radii[static.pair_j]
    overlap = jnp.maximum(min_dist - dist_ij, 0.0)
    overlap = jnp.where(dist_ij > 1e-6, overlap, 0.0)

    direction = d_vec / jnp.maximum(dist_ij[:, None], 1e-10)

    # Normal force (Hertz)
    f_n_mag = hertz_factor * overlap ** hertz_exponent
    f_n_vec = direction * f_n_mag[:, None]
    has_overlap = overlap > 0.0

    # Friction (Coulomb, opposes relative tangential velocity)
    v_rel = state.velocities[static.pair_j] - state.velocities[static.pair_i]
    v_t = v_rel - jnp.sum(v_rel * direction, axis=1, keepdims=True) * direction
    v_t_mag = jnp.linalg.norm(v_t, axis=1)
    t_hat = -v_t / jnp.maximum(v_t_mag[:, None], 1e-10)
    f_t_mag = jnp.minimum(ice_ice_friction * f_n_mag, friction_cap)
    f_t_vec = t_hat * f_t_mag[:, None]
    valid_friction = has_overlap & (v_t_mag > 1e-6)
    f_t_vec = jnp.where(valid_friction[:, None], f_t_vec, 0.0)
    f_n_vec = jnp.where(has_overlap[:, None], f_n_vec, 0.0)

    # Scatter contact forces (action/reaction)
    forces = jnp.zeros((n, 2))
    forces = forces.at[static.pair_i].add(-f_n_vec + f_t_vec)
    forces = forces.at[static.pair_j].add(f_n_vec - f_t_vec)

    # Torques (r × F, 2D)
    F_on_i = -f_n_vec - f_t_vec
    cross_val = -(d_vec[:, 0] * F_on_i[:, 1] - d_vec[:, 1] * F_on_i[:, 0])
    cross_val = jnp.where(has_overlap, cross_val, 0.0)
    moments = jnp.zeros(n)
    moments = moments.at[static.pair_i].add(0.5 * cross_val)
    moments = moments.at[static.pair_j].add(-0.5 * cross_val)

    return forces, moments


@jax.jit
def _compute_drag_jax(
    state: IceState,
    static: IceStaticData,
    v_water: jnp.ndarray,   # (n, 2) ocean current velocities
    c_d_drag: float,
    drag_ref_thickness: float,
    linear_drag_factor: float,
) -> jnp.ndarray:  # (n, 2) drag forces
    """Quadratic (or linear at low speed) drag on ice elements (Eq. 9)."""
    RHO_WATER = IceProperties.WATER_DENSITY
    A_drag = static.areas * (static.thicknesses / drag_ref_thickness)  # (n,)

    v_rel = state.velocities - v_water  # (n, 2)
    v_rel_mag = jnp.linalg.norm(v_rel, axis=1, keepdims=True)  # (n, 1)
    v_rel_mag_safe = jnp.maximum(v_rel_mag, 1e-12)

    drag_force = jnp.where(
        v_rel_mag > 1e-6,
        (v_rel / v_rel_mag_safe) * (
            0.5 * RHO_WATER * c_d_drag * A_drag[:, None]
            * v_rel_mag_safe ** 2
        ),
        0.5 * RHO_WATER * c_d_drag * A_drag[:, None] * linear_drag_factor * v_rel,
    )
    return drag_force


@jax.jit
def _compute_rotational_drag_jax(
    state: IceState,
    static: IceStaticData,
    drag_coefficient_moment: float,
) -> jnp.ndarray:  # (n,) rotational drag moments
    """Rotational drag moment (Eq. 8, hydrodynamic drag term)."""
    RHO_WATER = IceProperties.WATER_DENSITY
    A_sub_area = static.areas * jnp.minimum(
        state.submergence / jnp.maximum(static.thicknesses, 1e-10), 1.0
    )
    Md = (
        -0.5 * drag_coefficient_moment * RHO_WATER
        * (static.radii ** 3)
        * state.angular_vels * jnp.abs(state.angular_vels)
        * A_sub_area
    )
    return Md


@jax.jit
def _integrate_per_element_jax(
    state: IceState,
    forces: jnp.ndarray,    # (n, 2)
    moments: jnp.ndarray,   # (n,)
    static: IceStaticData,
    dt_ice: float,
    angular_damping: float,
    angular_cap_lo: float,
    angular_cap_hi: float,
    velocity_clip: float,
    position_clip: float,
) -> IceState:
    """
    Per-element Verlet integration (replaces component rigid-body loop).

    Simplification vs. simulation.py lines 619–640:
    Each element integrates its own forces independently.  The DPDEM bond
    spring forces (already included in `forces`) couple bonded elements.
    """
    n = state.positions.shape[0]

    # Linear acceleration and velocity update
    accel = forces / static.masses[:, None]
    new_vel = jnp.clip(
        state.velocities + accel * dt_ice,
        -velocity_clip, velocity_clip,
    )
    new_pos = state.positions + new_vel * dt_ice

    # Angular update
    alpha = moments / static.inertia
    new_avel = state.angular_vels * angular_damping + alpha * dt_ice
    new_avel = jnp.clip(new_avel, angular_cap_lo, angular_cap_hi)
    new_rot = state.rotations + new_avel * dt_ice

    # Sanitize NaN/Inf (replaces the NaN check in simulation.py lines 644-661)
    new_vel = jnp.where(jnp.isfinite(new_vel), new_vel, 0.0)
    new_pos = jnp.where(jnp.isfinite(new_pos), new_pos, state.positions)
    new_pos = jnp.clip(new_pos, -position_clip, position_clip)

    # Apply active mask: frozen elements don't move
    active = state.elem_active[:, None]
    new_vel = jnp.where(active, new_vel, state.velocities)
    new_pos = jnp.where(active, new_pos, state.positions)
    new_avel = jnp.where(state.elem_active, new_avel, state.angular_vels)
    new_rot  = jnp.where(state.elem_active, new_rot,  state.rotations)

    return state._replace(
        positions=new_pos,
        velocities=new_vel,
        rotations=new_rot,
        angular_vels=new_avel,
    )


@partial(jax.jit, static_argnames=("n_steps",))
def _stress_diffusion_jax(
    element_stress: jnp.ndarray,   # (n,)
    bond_intact: jnp.ndarray,      # (n_bonds,) bool
    bond_i: jnp.ndarray,           # (n_bonds,)
    bond_j: jnp.ndarray,           # (n_bonds,)
    stress_transfer: float,
    n_steps: int,
) -> jnp.ndarray:                   # (n,)
    """
    Stress diffusion over bonded network (simulation.py lines 676-682).
    Replaces the Python for _ in range(cfg.stress_diffusion_steps) loop.
    """
    def diffuse_step(stress, _):
        stress_i = stress[bond_i]
        stress_j = stress[bond_j]
        transfer_i_to_j = bond_intact & (stress_i > stress_j)
        transfer_j_to_i = bond_intact & (stress_j >= stress_i)
        new_stress = stress
        new_stress = new_stress.at[bond_j].add(
            jnp.where(transfer_i_to_j, stress_i * stress_transfer, 0.0)
        )
        new_stress = new_stress.at[bond_i].add(
            jnp.where(transfer_j_to_i, stress_j * stress_transfer, 0.0)
        )
        return new_stress, None

    stress_out, _ = jax.lax.scan(diffuse_step, element_stress, None, length=n_steps)
    return stress_out


def _make_frame_step(
    static: IceStaticData,
    cfg: SimulationConfig,
    ship_dyn: ShipDynamicsData,
    ship_length: float,
    ship_beam: float,
    n_ice_steps: int,
    dt: float,
    dt_ice: float,
    use_auto_icenav: bool,
    B_mat: jnp.ndarray,   # (3, 6) thruster config
    T_n: float,
):
    """
    Factory: returns a JIT-compiled single-frame physics step.

    The returned function takes (IceState, ShipState, setpoint, v_water,
    ship_collision_force_ext, dpdem_eligible_ext) and returns updated
    (IceState, ShipState, diagnostics).

    Most scalar config values are captured as Python scalars in the closure
    so JAX can treat them as compile-time constants.
    """
    # Scalar constants extracted from config (captured in closure)
    half_len  = ship_length / 2.0
    half_beam = ship_beam / 2.0
    n_edge_samples = getattr(cfg, "n_edge_samples", 3)
    bond_stiffness_factor = cfg.bond_stiffness_factor
    hertz_factor   = cfg.hertz_factor
    hertz_exponent = cfg.hertz_exponent
    ice_ice_friction = cfg.ice_ice_friction
    friction_cap   = cfg.friction_cap
    c_d_drag       = cfg.c_d_drag
    drag_ref_thickness = cfg.drag_ref_thickness
    linear_drag_factor = cfg.linear_drag_factor
    drag_coefficient_moment = getattr(cfg, "drag_coefficient_moment", 0.4)
    angular_damping = cfg.angular_damping
    angular_cap_lo, angular_cap_hi = cfg.angular_cap
    velocity_clip  = float(np.max(np.abs(cfg.velocity_clip)))
    vc_ship        = tuple(float(v) for v in cfg.ship_velocity_clip)  # (surge, sway, yaw)
    position_clip  = float(getattr(cfg, "position_clip_m", 1e4))
    stress_transfer = cfg.stress_transfer
    stress_diffusion_steps = cfg.stress_diffusion_steps
    pen_crit_ref   = cfg.pen_crit_ref
    pen_crit_thk_ref = cfg.pen_crit_thickness_ref
    w_contact_cap  = cfg.w_contact_cap
    w_contact_factor = cfg.w_contact_radius_factor
    side_push_factor = cfg.side_push_factor
    submergence_rate = cfg.submergence_rate
    submergence_pen_cap = cfg.submergence_pen_cap
    rise_time_const = cfg.rise_time_const
    effective_strength_collision = cfg.effective_strength_collision
    max_fracture_thickness = getattr(cfg, "max_fracture_thickness", 15.0)
    max_ship_ice_force = cfg.max_ship_ice_force  # None or float (N)
    bond_cohesion   = cfg.bond_cohesion
    bond_internal_friction = cfg.bond_internal_friction
    bond_G_c_I      = cfg.bond_G_c_I
    bond_G_c_II     = cfg.bond_G_c_II
    bond_eta_bk     = cfg.bond_eta_bk

    @jax.jit
    def _substep(carry, _):
        """
        One ice physics sub-step (scanned over n_ice_steps per frame).

        carry = (IceState, ship_collision_forces_on_ice (n,2), v_water (n,2))
        """
        state, ship_forces, v_water = carry

        # Accumulate forces
        f_bond, m_bond = _compute_bond_forces_jax(
            state, static, static.k_n, bond_stiffness_factor,
        )
        f_ice, m_ice = _compute_ice_ice_forces_jax(
            state, static, hertz_factor, hertz_exponent,
            ice_ice_friction, friction_cap,
        )
        f_drag = _compute_drag_jax(
            state, static, v_water,
            c_d_drag, drag_ref_thickness, linear_drag_factor,
        )
        m_rdrag = _compute_rotational_drag_jax(state, static, drag_coefficient_moment)

        forces  = f_bond + f_ice - f_drag + ship_forces
        moments = m_bond + m_ice + m_rdrag

        # Integrate
        state = _integrate_per_element_jax(
            state, forces, moments, static, dt_ice,
            angular_damping, angular_cap_lo, angular_cap_hi,
            velocity_clip, position_clip,
        )
        return (state, ship_forces, v_water), None

    @jax.jit
    def frame_step(
        state: IceState,
        ship_state: ShipState,
        setpoint: jnp.ndarray,      # (3,) [x_d, y_d, psi_d]
        setpoint_next: jnp.ndarray, # (3,) next waypoint for velocity feedforward
        v_water: jnp.ndarray,       # (n, 2) ocean current at ice positions
        dpdem_eligible: jnp.ndarray,# (n_floes,) bool
    ):
        """
        One complete simulation frame:
          1. Collision detection (ice ↔ ship)
          2. Contact force + thrust cap
          3. n_ice_steps inner physics sub-steps (scanned)
          4. Bond damage + fracture update
          5. Ship dynamics update
        """
        ship_x   = ship_state.eta[0]
        ship_y   = ship_state.eta[1]
        ship_psi = ship_state.eta[2]
        n = state.positions.shape[0]

        # ── 1. Collision detection ─────────────────────────────────────────────
        pen, nx, ny, cx, cy = compute_all_collisions_combined(
            static.local_polys_padded, static.poly_mask,
            state.positions, state.rotations,
            ship_x, ship_y, ship_psi,
            half_len, half_beam,
            ship_dyn.ship_hull_local,
            n_edge_samples=n_edge_samples,
        )
        # pen, nx, ny, cx, cy : (n,)

        # ── 2. Contact forces on ice and reaction on ship ──────────────────────
        has_collision = pen > 0.0

        # Force magnitude: F = F_break * min(pen / pen_crit, 1)
        w_contact = jnp.minimum(w_contact_factor * static.radii, w_contact_cap)
        # breaking_force: matches IceProperties.breaking_force(t, w) = sigma_f * w * t^2
        #   sigma_f = FLEXURAL_STRENGTH_BASE * sqrt(t)
        sigma_f = IceProperties.FLEXURAL_STRENGTH_BASE * jnp.sqrt(static.thicknesses)
        F_break = sigma_f * w_contact * static.thicknesses ** 2
        pen_crit = pen_crit_ref * (static.thicknesses / pen_crit_thk_ref)
        force_mag = F_break * jnp.minimum(pen / jnp.maximum(pen_crit, 1e-10), 1.0)
        force_mag = jnp.where(has_collision, force_mag, 0.0)

        # Ice forces from ship contact (body push + side push)
        ship_on_ice_x = force_mag * nx
        ship_on_ice_y = force_mag * ny

        # Side push (lateral)
        rel_y_to_ship = state.positions[:, 1] - ship_y
        side_dir = jnp.sign(rel_y_to_ship)
        side_push = side_push_factor * force_mag * side_dir
        ship_on_ice_y = ship_on_ice_y + side_push

        ship_on_ice = jnp.stack([ship_on_ice_x, ship_on_ice_y], axis=1)  # (n, 2)

        # ── Thrust cap: scale down all contact forces when total exceeds max_ship_ice_force ──
        # Matches simulation.py lines 458-466.  Without this cap, JAX uses raw F_break per
        # element; NumPy scales each ship_collision_force[i] proportionally when the sum
        # exceeds cfg.max_ship_ice_force, which can prevent break_collision from triggering
        # for lightly-contacted elements.
        if max_ship_ice_force is not None:
            F_coll_total = jnp.linalg.norm(jnp.sum(ship_on_ice, axis=0))
            thrust_scale = jnp.where(
                F_coll_total > max_ship_ice_force,
                max_ship_ice_force / jnp.maximum(F_coll_total, 1e-6),
                1.0,
            )
            ship_on_ice = ship_on_ice * thrust_scale
            force_mag = force_mag * thrust_scale  # also scale force used for fracture

        # Reaction on ship (Newton's 3rd)
        F_ice_on_ship_xy = -jnp.sum(ship_on_ice, axis=0)  # (2,)
        # Moment about ship: sum of r × F for each contact
        r_x = state.positions[:, 0] - ship_x
        r_y = state.positions[:, 1] - ship_y
        M_ice_on_ship = jnp.sum(
            jnp.where(has_collision, r_x * (-ship_on_ice_y) - r_y * (-ship_on_ice_x), 0.0)
        )
        tau_ice_world = jnp.array([F_ice_on_ship_xy[0], F_ice_on_ship_xy[1], M_ice_on_ship])

        # ── 3. Submergence update ──────────────────────────────────────────────
        pen_clamped = jnp.minimum(pen, submergence_pen_cap)
        new_submergence = jnp.where(
            has_collision,
            jnp.minimum(state.submergence + submergence_rate * pen_clamped * dt, static.thicknesses),
            jnp.maximum(0.0, state.submergence - (1.0 / rise_time_const) * state.submergence * dt),
        )

        # Buoyancy force from submergence (pushes ice away from ship)
        z_sub = jnp.minimum(new_submergence, static.thicknesses)
        RHO_W = IceProperties.WATER_DENSITY
        G = 9.81
        F_buoy = RHO_W * G * static.areas * z_sub
        ship_on_ice_x = ship_on_ice_x + jnp.where(has_collision, F_buoy * nx, 0.0)
        ship_on_ice_y = ship_on_ice_y + jnp.where(has_collision, F_buoy * ny, 0.0)
        ship_on_ice = jnp.stack([ship_on_ice_x, ship_on_ice_y], axis=1)

        state = state._replace(submergence=new_submergence)

        # ── 4. Activate floes touched by ship ─────────────────────────────────
        # elem_active: scatter collision mask to parent floe, then back to elements
        new_elem_active = state.elem_active
        # For each colliding element, activate all elements sharing its parent floe
        collided_floes = jnp.zeros(dpdem_eligible.shape[0], dtype=jnp.int32)
        collided_floes = collided_floes.at[static.initial_floe_of].max(
            has_collision.astype(jnp.int32)
        ).astype(bool)
        new_elem_active = new_elem_active | collided_floes[static.initial_floe_of]
        state = state._replace(elem_active=new_elem_active)

        # ── 5. DPDEM eligibility update ────────────────────────────────────────
        # Bonds where collision force exceeds fracture threshold become eligible
        if static.bond_i.shape[0] > 0:
            min_bond_strength = jnp.min(static.bond_strengths)
            threshold = min_bond_strength * effective_strength_collision
            new_dpdem_eligible = dpdem_eligible | (
                collided_floes & (
                    jnp.zeros(dpdem_eligible.shape[0], dtype=jnp.float64).at[
                        static.initial_floe_of
                    ].max(force_mag).astype(jnp.float64) > threshold
                )
            )
        else:
            new_dpdem_eligible = dpdem_eligible

        # ── 6. Inner ice substep scan ──────────────────────────────────────────
        (state, _, _), _ = jax.lax.scan(
            _substep,
            (state, ship_on_ice, v_water),
            None,
            length=n_ice_steps,
        )

        # ── 7. Bond stress accumulation and fracture ───────────────────────────
        if static.bond_i.shape[0] > 0:
            # Element stress from collision + bond forces
            element_stress = force_mag.copy() if False else force_mag  # (n,) — ship collision force
            d_bond = state.positions[static.bond_j] - state.positions[static.bond_i]
            dist_bond = jnp.linalg.norm(d_bond, axis=1)
            dist_safe = jnp.maximum(dist_bond, 1e-10)
            bond_dir = d_bond / dist_safe[:, None]
            avg_thickness_bond = (
                static.thicknesses[static.bond_i] + static.thicknesses[static.bond_j]
            ) * 0.5
            stiffness_bond = IceProperties.YOUNGS_MODULUS * avg_thickness_bond * bond_stiffness_factor
            stretch_bond = jnp.abs(dist_bond - static.rest_lengths)
            bond_force_mag = stiffness_bond * stretch_bond

            element_stress = element_stress.at[static.bond_i].add(
                jnp.where(state.bond_intact, bond_force_mag, 0.0)
            )
            element_stress = element_stress.at[static.bond_j].add(
                jnp.where(state.bond_intact, bond_force_mag, 0.0)
            )

            element_stress = _stress_diffusion_jax(
                element_stress, state.bond_intact,
                static.bond_i, static.bond_j,
                stress_transfer, stress_diffusion_steps,
            )

            # DPDEM damage update
            both_ep_active = new_elem_active[static.bond_i] & new_elem_active[static.bond_j]
            floe_ok = (
                new_dpdem_eligible[static.initial_floe_of[static.bond_i]]
                & new_dpdem_eligible[static.initial_floe_of[static.bond_j]]
            )
            bond_breakable = (
                (static.thicknesses[static.bond_i] <= max_fracture_thickness)
                & (static.thicknesses[static.bond_j] <= max_fracture_thickness)
            )
            bond_near_ship = (force_mag[static.bond_i] > 0) | (force_mag[static.bond_j] > 0)

            # Crack-front: elements adjacent to broken bonds
            broken_mask = ~state.bond_intact
            elem_at_crack = jnp.zeros(n, dtype=jnp.int32)
            elem_at_crack = elem_at_crack.at[static.bond_i].max(
                broken_mask.astype(jnp.int32)
            )
            elem_at_crack = elem_at_crack.at[static.bond_j].max(
                broken_mask.astype(jnp.int32)
            ).astype(bool)
            bond_at_crack = elem_at_crack[static.bond_i] | elem_at_crack[static.bond_j]

            # DPDEM eligibility: limit to bonds in direct ship contact OR at the crack front.
            # NOTE: bond_at_crack propagation is kept for geometric eligibility, but the delta_n
            # used in the DPDEM stress criterion is zeroed for non-contact bonds (see below).
            # This prevents per-element elastic oscillations (which have critical strain ~0.18 mm
            # vs k_n=6.73 GPa / sigma_bar~300 kPa) from triggering instant far-field fracture.
            bond_dpdem_ok = both_ep_active & floe_ok & (bond_near_ship | bond_at_crack) & bond_breakable

            # Tangential deformation increment.
            # Gate delta_s accumulation to bonds in DIRECT ship contact only (not crack-front).
            # Crack-front bonds inherit large tangential velocities from bond-force transmission
            # through the per-element elastic network.  In the NumPy rigid-body simulation,
            # elements within the same component share a single COM velocity so internal
            # delta_s is zero; only inter-component (crack-tip) bonds see relative sliding.
            # Per-element integration cannot enforce this constraint, so we suppress ds_incr
            # for non-contact bonds — analogous to delta_n_dpdem zeroing above.
            v_rel = state.velocities[static.bond_j] - state.velocities[static.bond_i]
            v_n_proj = jnp.sum(v_rel * bond_dir, axis=1)
            v_t_vec = v_rel - v_n_proj[:, None] * bond_dir
            ds_incr = jnp.linalg.norm(v_t_vec, axis=1) * dt_ice
            ds_gate = bond_near_ship & both_ep_active & floe_ok & bond_breakable
            new_bond_delta_s = state.bond_delta_s + jnp.where(ds_gate, ds_incr, 0.0)

            # Stress for fracture criterion.
            # For bonds NOT in direct ship contact, suppress delta_n (set to 0) so that
            # sub-millimetre elastic oscillations from per-element integration do not
            # cause instant damage accumulation (k_n=6.73 GPa → critical strain 0.18 mm).
            # In the NumPy simulation, rigid-body integration keeps internal delta_n ≈ 0;
            # this approximates that behaviour without requiring BFS component integration.
            L0 = jnp.maximum(static.rest_lengths, 1e-10)
            delta_n = dist_bond - static.rest_lengths
            delta_n_dpdem = jnp.where(bond_near_ship, delta_n, 0.0)
            sigma_n_el = static.k_n * (delta_n_dpdem / L0)
            sigma_t_el = static.k_s * (new_bond_delta_s / L0)
            sigma_n = sigma_n_el * (1.0 - state.bond_damage)
            sigma_t = sigma_t_el * (1.0 - state.bond_damage)

            new_damage, new_delta_m_max, new_delta_m_0, new_intact = update_bond_damage_state(
                delta_n, new_bond_delta_s, sigma_n, sigma_t,
                static.sigma_bar_bond,
                bond_cohesion, bond_internal_friction,
                bond_G_c_I, bond_G_c_II, bond_eta_bk,
                state.bond_damage, state.bond_delta_m_max, state.bond_delta_m_0,
                state.bond_intact,
            )

            # Apply damage only to DPDEM-eligible bonds
            new_damage  = jnp.where(bond_dpdem_ok, new_damage,  state.bond_damage)
            new_delta_m_max = jnp.where(bond_dpdem_ok, new_delta_m_max, state.bond_delta_m_max)
            new_delta_m_0   = jnp.where(bond_dpdem_ok, new_delta_m_0,   state.bond_delta_m_0)
            new_intact_dpdem = jnp.where(bond_dpdem_ok & bond_breakable, new_intact, state.bond_intact)

            # Direct collision breaking (regardless of DPDEM eligibility)
            collision_on_bond = (force_mag[static.bond_i] > 0) | (force_mag[static.bond_j] > 0)
            collision_force_on_bond = jnp.maximum(force_mag[static.bond_i], force_mag[static.bond_j])
            eff_strength = static.bond_strengths * effective_strength_collision
            break_collision = (
                state.bond_intact & collision_on_bond
                & (collision_force_on_bond > eff_strength)
                & bond_breakable
            )
            final_intact = new_intact_dpdem & ~break_collision

            state = state._replace(
                bond_damage=new_damage,
                bond_delta_m_max=new_delta_m_max,
                bond_delta_m_0=new_delta_m_0,
                bond_delta_s=new_bond_delta_s,
                bond_intact=final_intact,
            )

        # Update dpdem eligibility on state
        state = state._replace(dpdem_floe_eligible=new_dpdem_eligible)

        # ── 8. Ship dynamics update ────────────────────────────────────────────
        eta_new, nu_new, n_actual_new, e_int_new = ship_step_jax(
            ship_state.eta, ship_state.nu, ship_state.n_actual, ship_state.e_int,
            setpoint, setpoint_next, tau_ice_world,
            ship_dyn, B_mat, T_n, dt,
            use_auto_icenav=use_auto_icenav,
            vc_surge=vc_ship[0], vc_sway=vc_ship[1], vc_yaw=vc_ship[2],
        )
        ship_state_new = ShipState(
            eta=eta_new,
            nu=nu_new,
            n_actual=n_actual_new,
            e_int=e_int_new,
        )

        # Diagnostics (for Python-level callbacks)
        diag = {
            "pen_max": pen,
            "force_mag": force_mag,
            "F_ice_on_ship": jnp.array([F_ice_on_ship_xy[0], F_ice_on_ship_xy[1]]),
            "J_cumul_incr": jnp.linalg.norm(tau_ice_world[:2]) * dt,
        }

        return state, ship_state_new, diag

    return frame_step


# ── Public interface ───────────────────────────────────────────────────────────

def jax_simulate_ice_ship_interaction(
    polygons,
    centroids,
    bonds,
    ship_pos,
    ship_length,
    ship_beam,
    n_frames,
    dt,
    drift_vel,
    thicknesses,
    ship_props,
    ship_trajectory=None,
    return_ship_diagnostics=False,
    config=None,
    replan_every_n_frames=None,
    replan_callback=None,
    frame_callback=None,
    kinematic_ship=False,
    parent_ids=None,
    ocean_current=None,
    ship_drift_only=False,
    goal_point=None,
    goal_y_line=None,
    goal_tolerance=15.0,
    use_auto_icenav_ship: bool = False,
):
    """
    JAX-vectorized replacement for simulate_ice_ship_interaction().

    Identical signature and return format.  Dispatches the inner physics to
    a @jax.jit compiled function; the outer Python frame loop handles
    callbacks, BFS, and history recording.

    Notes
    -----
    - First call triggers XLA compilation (5–30 s depending on model size).
    - Requires JAX installed; set jax_enable_x64=True for float64 precision.
    - If no ship_trajectory is provided, the ship is stationary.
    """
    cfg = config or get_default_simulation_config()
    if ocean_current is None:
        ocean_current = NoCurrentModel()

    n = len(polygons)

    # ── Build static data (one-time, NumPy → JAX) ─────────────────────────────
    print("  [JAX] Building static data...", flush=True)
    static = build_static_data(polygons, centroids, bonds, thicknesses, parent_ids, cfg)
    ship_dyn = build_ship_dynamics_data(ship_props, cfg, ship_length, ship_beam)

    # ── Initial state ──────────────────────────────────────────────────────────
    ship_start_pos  = np.array(ship_pos)
    ship_start_head = 0.0
    if ship_trajectory is not None and len(ship_trajectory) > 0:
        ship_start_pos  = np.array(ship_trajectory[0][0])
        ship_start_head = float(ship_trajectory[0][1])

    state = build_initial_ice_state(
        n, centroids, len(bonds),
        np.array(static.initial_floe_of),
    )
    ship_state = build_initial_ship_state(ship_start_pos, ship_start_head)

    # ── Time stepping parameters ───────────────────────────────────────────────
    nu_r = getattr(cfg, "poisson_ratio", 0.3)
    E_ice = IceProperties.YOUNGS_MODULUS
    G_shear = E_ice / (2.0 * (1.0 + nu_r))
    rho_ice = IceProperties.ICE_DENSITY
    radii_np = np.array(static.radii)
    r_min = float(radii_np.min())
    b_safe = getattr(cfg, "rayleigh_safety_factor", 0.1)
    dt_rayleigh = (
        b_safe * np.pi * r_min / (0.163 * nu_r + 0.877)
        * np.sqrt(rho_ice / G_shear)
    )
    print(f"  [JAX] Rayleigh dt (advisory): {dt_rayleigh:.4f} s (dt={dt:.4f} s)", flush=True)
    use_rayleigh_cap = getattr(cfg, "use_rayleigh_dt_cap", False)
    dt_ice = min(dt, dt_rayleigh) if use_rayleigh_cap else dt
    n_ice_steps = max(1, int(round(dt / dt_ice))) if dt_ice > 0 else 1

    # ── AutoIceNav thruster parameters ────────────────────────────────────────
    B_mat = jnp.array(ship_dyn.B_pinv.T @ jnp.eye(6) + 0.0)  # placeholder; overwrite below
    T_n_val = 1.0
    try:
        from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
        _s = FossenSupply3DOF(mass=ship_props.mass, length=ship_length)
        B_mat = jnp.array(_s.B)
        T_n_val = _s.T_n
    except Exception:
        B_mat = jnp.zeros((3, 6))

    use_auto = use_auto_icenav_ship and ship_trajectory is not None and not kinematic_ship

    # ── Build JIT frame step ───────────────────────────────────────────────────
    print("  [JAX] Compiling frame_step...", flush=True)
    frame_step = _make_frame_step(
        static, cfg, ship_dyn,
        ship_length, ship_beam,
        n_ice_steps, dt, dt_ice,
        use_auto, B_mat, T_n_val,
    )

    # Warm-up JIT (compile with dummy data to avoid latency on first real frame)
    t_compile_start = time.perf_counter()
    v_water_zero = jnp.zeros((n, 2), dtype=jnp.float64)
    setpoint_dummy = ship_state.eta
    _ = frame_step(state, ship_state, setpoint_dummy, setpoint_dummy, v_water_zero, state.dpdem_floe_eligible)
    jax.block_until_ready(_)
    t_compile = time.perf_counter() - t_compile_start
    print(f"  [JAX] Compilation done in {t_compile:.1f} s", flush=True)

    # ── History buffers ────────────────────────────────────────────────────────
    positions_history   = [np.array(state.positions)]
    rotations_history   = [np.array(state.rotations)]
    bonds_history       = [np.array(state.bond_intact)]
    ship_history        = []
    submergence_history = [np.array(state.submergence)]
    ship_diagnostics    = [] if return_ship_diagnostics else None
    J_cumul = 0.0
    commanded_trajectory = []

    # ── DPDEM min threshold for eligibility ───────────────────────────────────
    bond_strengths_np = np.array(static.bond_strengths)
    min_collision_threshold = (
        float(bond_strengths_np.min() * cfg.effective_strength_collision)
        if len(bond_strengths_np) > 0
        else float("inf")
    )

    use_ship_dynamics = ship_trajectory is not None and not kinematic_ship
    half_len  = ship_length / 2.0
    half_beam = ship_beam / 2.0

    print(f"  [JAX] Starting {n_frames}-frame simulation ({n} ice elements, {len(bonds)} bonds)", flush=True)

    # ── Main Python frame loop ─────────────────────────────────────────────────
    for frame in range(n_frames):

        # ── Receding-horizon replan (Python callback, cannot JIT) ────────────
        if (
            use_ship_dynamics
            and replan_callback is not None
            and replan_every_n_frames is not None
            and replan_every_n_frames > 0
            and frame % replan_every_n_frames == 0
        ):
            pos_np = np.array(state.positions)
            rot_np = np.array(state.rotations)
            world_polygons = []
            local_polys_np = np.array(static.local_polys_padded)
            mask_np        = np.array(static.poly_mask)
            n_verts_np     = np.array(static.poly_n_verts)
            for i in range(n):
                nv = int(n_verts_np[i])
                lp = local_polys_np[i, :nv]
                c_r, s_r = np.cos(rot_np[i]), np.sin(rot_np[i])
                R = np.array([[c_r, -s_r], [s_r, c_r]])
                world_polygons.append(lp @ R.T + pos_np[i])
            eta_np = np.array(ship_state.eta)
            n_remaining = n_frames - frame
            new_traj = replan_callback(
                frame, world_polygons, pos_np.copy(), np.array(static.thicknesses),
                eta_np[:2].copy(), float(eta_np[2]), n_remaining,
            )
            if new_traj is not None and len(new_traj) >= 1:
                end_slice = frame + len(new_traj)
                ship_trajectory[frame:end_slice] = new_traj
                if end_slice < len(ship_trajectory):
                    ship_trajectory[end_slice:] = [new_traj[-1]] * (len(ship_trajectory) - end_slice)

        # ── Get current ship setpoint ─────────────────────────────────────────
        if ship_trajectory is not None:
            idx      = min(frame, len(ship_trajectory) - 1)
            idx_next = min(frame + 1, len(ship_trajectory) - 1)
            sp_pos, sp_heading = ship_trajectory[idx]
            sp_pos_next, sp_heading_next = ship_trajectory[idx_next]
            setpoint      = jnp.array([float(sp_pos[0]),      float(sp_pos[1]),      float(sp_heading)])
            setpoint_next = jnp.array([float(sp_pos_next[0]), float(sp_pos_next[1]), float(sp_heading_next)])
        else:
            setpoint = setpoint_next = ship_state.eta  # stationary

        # ── Sample ocean current ──────────────────────────────────────────────
        pos_np = np.array(state.positions)
        v_water_np = ocean_current.sample_batch(pos_np)  # (n, 2) NumPy
        v_water = jnp.array(v_water_np)

        # ── ONE JIT frame step ────────────────────────────────────────────────
        dpdem_elig = state.dpdem_floe_eligible
        state, ship_state, diag = frame_step(state, ship_state, setpoint, setpoint_next, v_water, dpdem_elig)
        jax.block_until_ready(state.positions)

        # ── Diagnostics and history (Python) ─────────────────────────────────
        J_cumul += float(diag["J_cumul_incr"])

        eta_np = np.array(ship_state.eta)
        ship_history.append((eta_np[:2].copy(), float(eta_np[2])))
        if ship_trajectory is not None:
            commanded_trajectory.append((np.array(sp_pos), float(sp_heading)))

        positions_history.append(np.array(state.positions))
        rotations_history.append(np.array(state.rotations))
        bonds_history.append(np.array(state.bond_intact))
        submergence_history.append(np.array(state.submergence))

        if return_ship_diagnostics and ship_diagnostics is not None:
            F_ice = np.array(diag["F_ice_on_ship"])
            ship_diagnostics.append({
                "t": frame * dt,
                "F_ice_ship": F_ice,
                "ship_pos": eta_np[:2].copy(),
                "impulse_cumul": J_cumul,
            })

        # ── Frame callback ────────────────────────────────────────────────────
        if frame_callback is not None:
            frame_callback(
                frame,
                np.array(state.positions),
                np.array(state.rotations),
                np.array(state.bond_intact),
                eta_np[:2],
                float(eta_np[2]),
            )

        # ── Goal check ────────────────────────────────────────────────────────
        ship_xy = eta_np[:2]
        if goal_y_line is not None:
            if ship_xy[1] >= goal_y_line - goal_tolerance:
                print(f"  [JAX] Goal y-line reached at frame {frame}", flush=True)
                break
        elif goal_point is not None:
            goal_arr = np.array(goal_point)
            if np.linalg.norm(ship_xy - goal_arr) <= goal_tolerance:
                print(f"  [JAX] Goal point reached at frame {frame}", flush=True)
                break

        ocean_current.step()

    # ── Build return value matching simulate_ice_ship_interaction signature ────
    damage_summary = {"impulse_total": J_cumul}

    if return_ship_diagnostics:
        if ship_diagnostics:
            ship_diagnostics[-1]["damage_summary"] = damage_summary
        return (
            positions_history,
            rotations_history,
            bonds_history,
            ship_history,
            submergence_history,
            ship_diagnostics,
        )
    else:
        return (
            positions_history,
            rotations_history,
            bonds_history,
            ship_history,
            submergence_history,
            damage_summary,
        )


# ── Full JAX scan: entire simulation compiled into one XLA program ─────────

def simulate_full_jax_scan(
    polygons,
    centroids,
    bonds,
    ship_pos,
    ship_length,
    ship_beam,
    n_frames,
    dt,
    drift_vel,
    thicknesses,
    ship_props,
    ship_trajectory=None,
    config=None,
    parent_ids=None,
    v_water_all=None,
    use_auto_icenav_ship: bool = False,
):
    """
    Full JAX scan simulation: entire frame loop compiled into one XLA program.

    Unlike jax_simulate_ice_ship_interaction(), this function uses
    jax.lax.scan so the Python frame-dispatch overhead between frames is
    eliminated.  The first call triggers a full XLA compilation (may take
    longer than the per-frame version); subsequent calls with the same
    shapes execute in a single GPU kernel dispatch.

    Limitations vs. jax_simulate_ice_ship_interaction()
    -------------------------------------------------------
    * No replan_callback — ship_trajectory must be fully pre-specified.
    * No frame_callback — real-time visualisation is not supported.
    * No goal-based early stopping.
    * Ocean current must be pre-computed; pass v_water_all=(n_frames, n, 2)
      or None (defaults to zero current).

    Parameters
    ----------
    polygons, centroids, bonds, ship_pos, ship_length, ship_beam,
    n_frames, dt, drift_vel, thicknesses, ship_props :
        Same as jax_simulate_ice_ship_interaction().
    ship_trajectory : list of (pos, heading) or None
        Pre-computed trajectory for all n_frames.
    config : SimulationConfig or None
    parent_ids : (n,) int array or None
    v_water_all : (n_frames, n, 2) array or None
        Per-frame ocean-current velocities at each ice element.
        If None, zero current is assumed.
    use_auto_icenav_ship : bool

    Returns
    -------
    Same 6-tuple as jax_simulate_ice_ship_interaction() (non-diagnostics).
    """
    cfg = config or get_default_simulation_config()
    n = len(polygons)

    # ── Build static data ──────────────────────────────────────────────────
    print("  [JAX-scan] Building static data...", flush=True)
    static   = build_static_data(polygons, centroids, bonds, thicknesses, parent_ids, cfg)
    ship_dyn = build_ship_dynamics_data(ship_props, cfg, ship_length, ship_beam)

    # ── Initial state ──────────────────────────────────────────────────────
    ship_start_pos  = np.array(ship_pos)
    ship_start_head = 0.0
    if ship_trajectory is not None and len(ship_trajectory) > 0:
        ship_start_pos  = np.array(ship_trajectory[0][0])
        ship_start_head = float(ship_trajectory[0][1])

    state      = build_initial_ice_state(n, centroids, len(bonds), np.array(static.initial_floe_of))
    ship_state = build_initial_ship_state(ship_start_pos, ship_start_head)

    # ── Time stepping ──────────────────────────────────────────────────────
    nu_r    = getattr(cfg, "poisson_ratio", 0.3)
    E_ice   = IceProperties.YOUNGS_MODULUS
    G_shear = E_ice / (2.0 * (1.0 + nu_r))
    rho_ice = IceProperties.ICE_DENSITY
    r_min   = float(np.array(static.radii).min())
    b_safe  = getattr(cfg, "rayleigh_safety_factor", 0.1)
    dt_rayleigh = (
        b_safe * np.pi * r_min / (0.163 * nu_r + 0.877)
        * np.sqrt(rho_ice / G_shear)
    )
    use_rayleigh_cap = getattr(cfg, "use_rayleigh_dt_cap", False)
    dt_ice     = min(dt, dt_rayleigh) if use_rayleigh_cap else dt
    n_ice_steps = max(1, int(round(dt / dt_ice))) if dt_ice > 0 else 1

    # ── AutoIceNav thruster parameters ────────────────────────────────────
    B_mat   = jnp.zeros((3, 6), dtype=jnp.float64)
    T_n_val = 1.0
    try:
        from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
        _s    = FossenSupply3DOF(mass=ship_props.mass, length=ship_length)
        B_mat = jnp.array(_s.B)
        T_n_val = _s.T_n
    except Exception:
        pass
    use_auto = use_auto_icenav_ship and ship_trajectory is not None

    # ── Build JIT frame step (same factory as per-frame version) ──────────
    print("  [JAX-scan] Building frame step...", flush=True)
    frame_step = _make_frame_step(
        static, cfg, ship_dyn,
        ship_length, ship_beam,
        n_ice_steps, dt, dt_ice,
        use_auto, B_mat, T_n_val,
    )

    # Warm-up: compile frame_step first so we can time the scan separately
    _dummy_vw = jnp.zeros((n, 2), dtype=jnp.float64)
    _dummy_sp = ship_state.eta
    _ = frame_step(state, ship_state, _dummy_sp, _dummy_sp, _dummy_vw,
                   state.dpdem_floe_eligible)
    jax.block_until_ready(_)

    # ── Pre-compute per-frame setpoints ────────────────────────────────────
    if ship_trajectory is not None:
        sp_rows = []
        for frame in range(n_frames):
            idx      = min(frame, len(ship_trajectory) - 1)
            idx_next = min(frame + 1, len(ship_trajectory) - 1)
            sp_pos,  sp_head  = ship_trajectory[idx]
            sp_posn, sp_headn = ship_trajectory[idx_next]
            sp_rows.append([
                float(sp_pos[0]),  float(sp_pos[1]),  float(sp_head),
                float(sp_posn[0]), float(sp_posn[1]), float(sp_headn),
            ])
        setpoints_all = jnp.array(sp_rows, dtype=jnp.float64)   # (n_frames, 6)
    else:
        eta0 = ship_state.eta
        row  = jnp.concatenate([eta0, eta0])[None, :]            # (1, 6)
        setpoints_all = jnp.repeat(row, n_frames, axis=0)        # (n_frames, 6)

    # ── Pre-compute per-frame water velocities ─────────────────────────────
    if v_water_all is not None:
        v_water_jax = jnp.array(v_water_all, dtype=jnp.float64)  # (n_frames, n, 2)
    else:
        v_water_jax = jnp.zeros((n_frames, n, 2), dtype=jnp.float64)

    # ── Capture initial arrays for history prepend ─────────────────────────
    init_positions   = np.array(state.positions)
    init_rotations   = np.zeros(n, dtype=np.float64)
    init_bonds       = np.ones(len(bonds), dtype=bool)
    init_submergence = np.zeros(n, dtype=np.float64)

    # ── Define the scan body ───────────────────────────────────────────────
    def _scan_step(carry, xs):
        """One frame of physics: (IceState, ShipState) → updated carry + outputs."""
        state, ship_state = carry
        setpoint_row, v_water = xs         # (6,), (n, 2)

        setpoint      = setpoint_row[:3]
        setpoint_next = setpoint_row[3:]
        dpdem_elig    = state.dpdem_floe_eligible

        state, ship_state, diag = frame_step(
            state, ship_state, setpoint, setpoint_next, v_water, dpdem_elig,
        )

        hist = (
            state.positions,         # (n, 2)
            state.rotations,         # (n,)
            state.bond_intact,       # (n_bonds,)
            ship_state.eta,          # (3,)
            state.submergence,       # (n,)
            diag["J_cumul_incr"],    # scalar
            diag["F_ice_on_ship"],   # (2,)
        )
        return (state, ship_state), hist

    # ── Compile + run ──────────────────────────────────────────────────────
    print(f"  [JAX-scan] Compiling & running {n_frames}-frame scan ({n} elements)...",
          flush=True)
    t0 = time.perf_counter()

    (final_state, final_ship), histories = jax.lax.scan(
        _scan_step,
        (state, ship_state),
        (setpoints_all, v_water_jax),
    )
    jax.block_until_ready(final_state.positions)

    t_total = time.perf_counter() - t0
    print(f"  [JAX-scan] Done in {t_total:.1f} s", flush=True)

    # ── Unpack histories ───────────────────────────────────────────────────
    pos_all, rot_all, bond_all, eta_all, sub_all, j_all, fship_all = histories
    # shapes: (n_frames, n, 2), (n_frames, n), (n_frames, n_bonds),
    #         (n_frames, 3),    (n_frames, n), (n_frames,), (n_frames, 2)

    pos_all   = np.array(pos_all)
    rot_all   = np.array(rot_all)
    bond_all  = np.array(bond_all)
    eta_all   = np.array(eta_all)
    sub_all   = np.array(sub_all)
    j_all     = np.array(j_all)
    fship_all = np.array(fship_all)

    # Convert to list format matching jax_simulate_ice_ship_interaction
    positions_history   = [init_positions]   + [pos_all[f]  for f in range(n_frames)]
    rotations_history   = [init_rotations]   + [rot_all[f]  for f in range(n_frames)]
    bonds_history       = [init_bonds]       + [bond_all[f] for f in range(n_frames)]
    submergence_history = [init_submergence] + [sub_all[f]  for f in range(n_frames)]

    ship_history = [
        (eta_all[f, :2].copy(), float(eta_all[f, 2]))
        for f in range(n_frames)
    ]

    J_cumul = float(np.sum(j_all))
    damage_summary = {"impulse_total": J_cumul}

    return (
        positions_history,
        rotations_history,
        bonds_history,
        ship_history,
        submergence_history,
        damage_summary,
    )


# ── Low-level primitives for chunked/segmented simulation ─────────────────────

from typing import NamedTuple as _NamedTuple


class JaxSimContext(_NamedTuple):
    """
    Compiled simulation context created once, reused across chunk calls.

    Holds the JIT-compiled frame_step function (already warmed up) plus
    a pre-compiled scan_segment function that wraps jax.lax.scan so the
    XLA program is compiled once and reused on every chunk call.
    """
    frame_step: object          # JIT-compiled frame step callable
    scan_segment: object        # Compiled segment runner: (state, ship, sp_K6, vw_Kn2) -> (state, ship, hist)
    static: IceStaticData       # Immutable static data (geometry, masses, bonds)
    ship_dyn: object            # ShipDynamicsData
    n: int                      # Number of ice elements
    n_bonds: int                # Number of bonds
    dt: float
    chunk_size: int             # Canonical chunk length (frames per segment)


def build_jax_sim_context(
    polygons,
    centroids,
    bonds,
    ship_pos,
    ship_length,
    ship_beam,
    n_frames,
    dt,
    drift_vel,
    thicknesses,
    ship_props,
    config=None,
    parent_ids=None,
    use_auto_icenav_ship: bool = False,
    chunk_size: int = 50,
    verbose: bool = True,
) -> "JaxSimContext":
    """
    One-time setup: build static data, compile frame_step, warm up JIT.

    Returns a JaxSimContext that can be passed to _run_jax_segment() and
    simulate_jax_chunked_replan() without re-compiling on each call.

    Parameters
    ----------
    chunk_size : int
        The fixed scan length used by _run_jax_segment.  All segment calls
        must use arrays of exactly this length so XLA re-uses cached kernels.
        Pass ``replan_every_n_frames`` here.
    """
    cfg = config or get_default_simulation_config()
    n = len(polygons)

    if verbose:
        print("  [JAX-ctx] Building static data...", flush=True)
    static = build_static_data(polygons, centroids, bonds, thicknesses, parent_ids, cfg)
    ship_dyn = build_ship_dynamics_data(ship_props, cfg, ship_length, ship_beam)

    # Time-stepping parameters
    nu_r = getattr(cfg, "poisson_ratio", 0.3)
    E_ice = IceProperties.YOUNGS_MODULUS
    G_shear = E_ice / (2.0 * (1.0 + nu_r))
    rho_ice = IceProperties.ICE_DENSITY
    r_min = float(np.array(static.radii).min())
    b_safe = getattr(cfg, "rayleigh_safety_factor", 0.1)
    dt_rayleigh = (
        b_safe * np.pi * r_min / (0.163 * nu_r + 0.877)
        * np.sqrt(rho_ice / G_shear)
    )
    use_rayleigh_cap = getattr(cfg, "use_rayleigh_dt_cap", False)
    dt_ice = min(dt, dt_rayleigh) if use_rayleigh_cap else dt
    n_ice_steps = max(1, int(round(dt / dt_ice))) if dt_ice > 0 else 1

    # AutoIceNav thruster config
    B_mat = jnp.zeros((3, 6), dtype=jnp.float64)
    T_n_val = 1.0
    try:
        from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
        _s = FossenSupply3DOF(mass=ship_props.mass, length=ship_length)
        B_mat = jnp.array(_s.B)
        T_n_val = _s.T_n
    except Exception:
        pass
    use_auto = use_auto_icenav_ship

    if verbose:
        print("  [JAX-ctx] Compiling frame_step...", flush=True)
    t0 = time.perf_counter()
    frame_step = _make_frame_step(
        static, cfg, ship_dyn,
        ship_length, ship_beam,
        n_ice_steps, dt, dt_ice,
        use_auto, B_mat, T_n_val,
    )

    # Build scan_segment once so the closure is stable across all chunk calls.
    # Defining _scan_step here (not inside _run_jax_segment) ensures JAX sees
    # the same Python function object every time → no recompilation per chunk.
    def _scan_step(carry, xs):
        _state, _ship = carry
        setpoint_row, v_water = xs          # (6,), (n, 2)
        setpoint      = setpoint_row[:3]
        setpoint_next = setpoint_row[3:]
        dpdem_elig    = _state.dpdem_floe_eligible
        _state, _ship, diag = frame_step(
            _state, _ship, setpoint, setpoint_next, v_water, dpdem_elig,
        )
        hist = (
            _state.positions,
            _state.rotations,
            _state.bond_intact,
            _ship.eta,
            _state.submergence,
            diag["J_cumul_incr"],
            diag["F_ice_on_ship"],
        )
        return (_state, _ship), hist

    def scan_segment(state, ship_state, setpoints_K6, v_water_Kn2):
        (final_state, final_ship), histories = jax.lax.scan(
            _scan_step,
            (state, ship_state),
            (setpoints_K6, v_water_Kn2),
        )
        jax.block_until_ready(final_state.positions)
        return final_state, final_ship, histories

    # Warm-up: compile frame_step and the full scan_segment with chunk-sized data
    _dummy_state = build_initial_ice_state(n, centroids, len(bonds), np.array(static.initial_floe_of))
    _dummy_ship  = build_initial_ship_state(np.array(ship_pos), 0.0)
    _dummy_vw_K  = jnp.zeros((chunk_size, n, 2), dtype=jnp.float64)
    _dummy_eta   = _dummy_ship.eta
    _dummy_sp_K  = jnp.tile(jnp.concatenate([_dummy_eta, _dummy_eta])[None, :], (chunk_size, 1))
    _ = scan_segment(_dummy_state, _dummy_ship, _dummy_sp_K, _dummy_vw_K)
    jax.block_until_ready(_[0].positions)
    if verbose:
        print(f"  [JAX-ctx] Compilation done in {time.perf_counter() - t0:.1f} s", flush=True)

    return JaxSimContext(
        frame_step=frame_step,
        scan_segment=scan_segment,
        static=static,
        ship_dyn=ship_dyn,
        n=n,
        n_bonds=len(bonds),
        dt=dt,
        chunk_size=chunk_size,
    )


def _run_jax_segment(
    ctx: JaxSimContext,
    state: "IceState",
    ship_state: "ShipState",
    setpoints_K6: jnp.ndarray,   # (K, 6)  [x,y,psi, x_next,y_next,psi_next]
    v_water_Kn2: jnp.ndarray,    # (K, n, 2)
) -> tuple:
    """
    Run exactly K frames via jax.lax.scan using a pre-compiled frame_step.

    K must equal ctx.chunk_size so XLA can reuse cached kernels across calls.

    Returns
    -------
    final_state  : IceState
    final_ship   : ShipState
    histories    : tuple of stacked arrays (K frames):
        (positions (K,n,2), rotations (K,n), bond_intact (K,n_bonds),
         eta (K,3), submergence (K,n), J_cumul_incr (K,), F_ice_on_ship (K,2))
    """
    return ctx.scan_segment(state, ship_state, setpoints_K6, v_water_Kn2)


def _extract_world_polygons_np(state, static):
    """
    Convert current IceState + IceStaticData to a list of world-frame polygon arrays.

    Vectorized NumPy replacement for the Python for-loop in the replan callback
    (jax_simulate_ice_ship_interaction L833-844).  Runs ~n× faster than the loop.

    Returns
    -------
    world_polygons : list of (nv_i, 2) ndarray, one per element
    """
    pos_np   = np.array(state.positions)      # (n, 2)
    rot_np   = np.array(state.rotations)      # (n,)
    lp_np    = np.array(static.local_polys_padded)  # (n, max_V, 2)
    n_verts  = np.array(static.poly_n_verts)  # (n,)

    c = np.cos(rot_np)   # (n,)
    s = np.sin(rot_np)   # (n,)

    # Build (n, 2, 2) rotation matrices without a Python loop
    R = np.stack([
        np.stack([ c, -s], axis=1),   # (n, 2) first row
        np.stack([ s,  c], axis=1),   # (n, 2) second row
    ], axis=1)  # (n, 2, 2)

    # Rotate all padded vertices at once: (n, max_V, 2) @ (n, 2, 2)^T + (n, 1, 2)
    world_padded = np.einsum("nvi,nji->nvj", lp_np, R) + pos_np[:, None, :]  # (n, max_V, 2)

    # Slice each element to its actual vertex count
    world_polygons = [world_padded[i, : int(n_verts[i])] for i in range(len(n_verts))]
    return world_polygons


def simulate_jax_chunked_replan(
    polygons,
    centroids,
    bonds,
    ship_pos,
    ship_length,
    ship_beam,
    n_frames,
    dt,
    drift_vel,
    thicknesses,
    ship_props,
    ship_trajectory=None,
    return_ship_diagnostics: bool = False,
    config=None,
    replan_every_n_frames=None,
    replan_callback=None,
    frame_callback=None,
    parent_ids=None,
    ocean_current=None,
    goal_point=None,
    goal_y_line=None,
    goal_tolerance: float = 15.0,
    use_auto_icenav_ship: bool = False,
    verbose: bool = True,
):
    """
    GPU-accelerated simulation with receding-horizon replanning.

    Unlike ``jax_simulate_ice_ship_interaction`` (which has an O(N_frames)
    Python loop), this function runs the physics in fixed-length JAX scan
    segments of ``replan_every_n_frames`` frames, so the Python loop runs
    only O(N_frames / replan_every_n_frames) times — one iteration per replan
    event.  All heavy computation (physics + diffusion-MPC planning) executes
    on the JAX device.

    The interface is identical to ``jax_simulate_ice_ship_interaction`` so
    it can be used as a drop-in replacement in ``run_medea_ship_fracture.py``.

    Limitations vs. ``jax_simulate_ice_ship_interaction``
    ------------------------------------------------------
    * Goal-based early stopping is checked at chunk boundaries only (not
      every frame), so the ship may overshoot by up to ``replan_every_n_frames``
      frames.
    * ``frame_callback`` is called once per chunk (with the final frame of
      that chunk), not once per frame.
    * If ``replan_every_n_frames`` is None or 0, falls back to a single-chunk
      ``simulate_full_jax_scan`` call (no replanning).
    """
    cfg = config or get_default_simulation_config()
    if ocean_current is None:
        ocean_current = NoCurrentModel()

    n = len(polygons)
    chunk = replan_every_n_frames if (replan_every_n_frames and replan_every_n_frames > 0) else n_frames

    # ── One-time setup ─────────────────────────────────────────────────────────
    ctx = build_jax_sim_context(
        polygons, centroids, bonds, ship_pos,
        ship_length, ship_beam, n_frames, dt, drift_vel,
        thicknesses, ship_props,
        config=cfg, parent_ids=parent_ids,
        use_auto_icenav_ship=use_auto_icenav_ship,
        chunk_size=chunk,
        verbose=verbose,
    )

    # ── Initial state ──────────────────────────────────────────────────────────
    ship_start_pos  = np.array(ship_pos)
    ship_start_head = 0.0
    if ship_trajectory is not None and len(ship_trajectory) > 0:
        ship_start_pos  = np.array(ship_trajectory[0][0])
        ship_start_head = float(ship_trajectory[0][1])

    state      = build_initial_ice_state(n, centroids, len(bonds), np.array(ctx.static.initial_floe_of))
    ship_state = build_initial_ship_state(ship_start_pos, ship_start_head)

    # ── History buffers ────────────────────────────────────────────────────────
    positions_history   = [np.array(state.positions)]
    rotations_history   = [np.array(state.rotations)]
    bonds_history       = [np.array(state.bond_intact)]
    submergence_history = [np.array(state.submergence)]
    ship_history        = []
    ship_diagnostics    = [] if return_ship_diagnostics else None
    J_cumul             = 0.0

    if verbose:
        print(f"  [JAX-chunked] Starting {n_frames}-frame simulation "
              f"({n} elements, chunk={chunk}, "
              f"n_chunks={int(np.ceil(n_frames / chunk))})", flush=True)

    frame_start = 0
    use_ship_dynamics = ship_trajectory is not None

    while frame_start < n_frames:
        frame_end = min(frame_start + chunk, n_frames)
        K = frame_end - frame_start   # actual frames in this chunk

        # ── Replan callback (if any) ───────────────────────────────────────────
        # Call at every chunk boundary including frame 0, matching the
        # per-frame loop which evaluates the callback when frame % replan_every == 0.
        if (
            use_ship_dynamics
            and replan_callback is not None
        ):
            world_polygons = _extract_world_polygons_np(state, ctx.static)
            pos_np         = np.array(state.positions)
            eta_np         = np.array(ship_state.eta)
            n_remaining    = n_frames - frame_start

            new_traj = replan_callback(
                frame_start, world_polygons, pos_np.copy(),
                np.array(ctx.static.thicknesses),
                eta_np[:2].copy(), float(eta_np[2]),
                n_remaining,
            )
            if new_traj is not None and len(new_traj) >= 1:
                end_slice = frame_start + len(new_traj)
                ship_trajectory[frame_start:end_slice] = new_traj
                if end_slice < len(ship_trajectory):
                    ship_trajectory[end_slice:] = [new_traj[-1]] * (len(ship_trajectory) - end_slice)

        # ── Build per-frame setpoints for this chunk ───────────────────────────
        if ship_trajectory is not None:
            traj_len = len(ship_trajectory)
            sp_rows = []
            for f in range(frame_start, frame_end):
                idx      = min(f, traj_len - 1)
                idx_next = min(f + 1, traj_len - 1)
                sp_pos,  sp_head  = ship_trajectory[idx]
                sp_posn, sp_headn = ship_trajectory[idx_next]
                sp_rows.append([
                    float(sp_pos[0]),  float(sp_pos[1]),  float(sp_head),
                    float(sp_posn[0]), float(sp_posn[1]), float(sp_headn),
                ])
            setpoints_K6 = jnp.array(sp_rows, dtype=jnp.float64)
        else:
            eta0 = ship_state.eta
            row  = jnp.concatenate([eta0, eta0])[None, :]
            setpoints_K6 = jnp.repeat(row, K, axis=0)

        # ── Sample ocean current for this chunk ────────────────────────────────
        # Use GPU-native sampling when available (avoids K × GPU↔CPU transfers).
        # sample_chunk_jax() samples once at chunk-start positions and tiles K
        # times — exact for static fields (No/Uniform), an O(dt_chunk) approx
        # for slowly-evolving fields (_GridCurrentModel).
        # LearnedCurrentModel (time-varying) falls back to the NumPy loop.
        if hasattr(ocean_current, "sample_chunk_jax"):
            v_water_Kn2 = ocean_current.sample_chunk_jax(
                K, ctx.n, pos_jax=state.positions, dtype=jnp.float64,
            )
            for _ in range(K):
                ocean_current.step()
        else:
            pos_np = np.array(state.positions)
            v_rows = []
            for _ in range(K):
                v_rows.append(ocean_current.sample_batch(pos_np))
                ocean_current.step()
            v_water_Kn2 = jnp.array(np.stack(v_rows, axis=0), dtype=jnp.float64)

        # ── If K < chunk (last short chunk), pad to chunk size for scan ────────
        pad = chunk - K
        if pad > 0:
            last_sp = setpoints_K6[-1:, :]
            setpoints_K6 = jnp.concatenate([setpoints_K6, jnp.repeat(last_sp, pad, axis=0)], axis=0)
            last_vw = v_water_Kn2[-1:, :, :]
            v_water_Kn2  = jnp.concatenate([v_water_Kn2,  jnp.repeat(last_vw,  pad, axis=0)], axis=0)

        # ── Run the JAX scan segment ───────────────────────────────────────────
        final_state, final_ship, hist = _run_jax_segment(
            ctx, state, ship_state, setpoints_K6, v_water_Kn2,
        )

        # Unpack histories (shape: (chunk, ...) — trim to K)
        pos_K, rot_K, bond_K, eta_K, sub_K, j_K, fship_K = hist
        pos_K    = np.array(pos_K[:K])
        rot_K    = np.array(rot_K[:K])
        bond_K   = np.array(bond_K[:K])
        eta_K    = np.array(eta_K[:K])
        sub_K    = np.array(sub_K[:K])
        j_K      = np.array(j_K[:K])
        fship_K  = np.array(fship_K[:K])

        for f in range(K):
            positions_history.append(pos_K[f])
            rotations_history.append(rot_K[f])
            bonds_history.append(bond_K[f])
            submergence_history.append(sub_K[f])
            ship_history.append((eta_K[f, :2].copy(), float(eta_K[f, 2])))
            J_cumul += float(j_K[f])

            if return_ship_diagnostics and ship_diagnostics is not None:
                t_frame = (frame_start + f) * dt
                ship_diagnostics.append({
                    "t": t_frame,
                    "F_ice_ship": fship_K[f],
                    "ship_pos": eta_K[f, :2].copy(),
                    "impulse_cumul": J_cumul,
                })

        # ── Frame callback (called once with final frame of chunk) ─────────────
        if frame_callback is not None:
            frame_callback(
                frame_end - 1,
                pos_K[-1],
                rot_K[-1],
                bond_K[-1],
                eta_K[-1, :2],
                float(eta_K[-1, 2]),
            )

        # ── Goal check at chunk boundary ───────────────────────────────────────
        ship_xy = eta_K[-1, :2]
        if goal_y_line is not None:
            if ship_xy[1] >= goal_y_line - goal_tolerance:
                if verbose:
                    print(f"  [JAX-chunked] Goal y-line reached at frame {frame_end - 1}", flush=True)
                # Trim to the actual frame where goal was first crossed
                for _f in range(K):
                    if eta_K[_f, 1] >= goal_y_line - goal_tolerance:
                        _actual = frame_start + _f + 1  # keep up to this frame
                        positions_history   = positions_history[:_actual + 1]
                        rotations_history   = rotations_history[:_actual + 1]
                        bonds_history       = bonds_history[:_actual + 1]
                        submergence_history = submergence_history[:_actual + 1]
                        ship_history        = ship_history[:_actual]
                        if ship_diagnostics:
                            ship_diagnostics = ship_diagnostics[:_actual]
                        break
                state = final_state
                ship_state = final_ship
                break
        elif goal_point is not None:
            goal_arr = np.array(goal_point)
            if np.linalg.norm(ship_xy - goal_arr) <= goal_tolerance:
                if verbose:
                    print(f"  [JAX-chunked] Goal point reached at frame {frame_end - 1}", flush=True)
                state = final_state
                ship_state = final_ship
                break

        state      = final_state
        ship_state = final_ship
        frame_start = frame_end

    # ── Build return value ─────────────────────────────────────────────────────
    damage_summary = {"impulse_total": J_cumul}

    if return_ship_diagnostics:
        if ship_diagnostics:
            ship_diagnostics[-1]["damage_summary"] = damage_summary
        return (
            positions_history,
            rotations_history,
            bonds_history,
            ship_history,
            submergence_history,
            ship_diagnostics,
        )
    else:
        return (
            positions_history,
            rotations_history,
            bonds_history,
            ship_history,
            submergence_history,
            damage_summary,
        )
