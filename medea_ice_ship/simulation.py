"""Ice–ship interaction simulation: collision, fracture, ship dynamics."""

import os
import numpy as np
from medea_ice_ship.properties import IceProperties, R_xy, ssa
from medea_ice_ship.geometry import (
    polygon_area,
    polygon_centroid,
    create_ship_hull_at_position_heading,
    point_in_polygon,
    point_inside_polygon_penetration,
)
from medea_ice_ship.bonds import find_connected_components
from medea_ice_ship.bond_fracture import update_bond_damage_state
from medea_ice_ship.config import SimulationConfig, get_default_simulation_config
from medea_ice_ship.vessel_dynamics import get_mss_style_3dof, coriolis_3dof
from medea_ice_ship.ocean_current import OceanCurrentModel, NoCurrentModel


def point_to_ship_collision(px, py, ship_pos, ship_length, ship_beam, bow_taper_start=0.6, ship_heading=0.0):
    """Check collision with ship hull. ship_heading: angle in radians (0 = bow toward -x)."""
    cx, cy = ship_pos
    half_len = ship_length / 2
    half_beam = ship_beam / 2
    dx = px - cx
    dy = py - cy
    c, s = np.cos(-ship_heading), np.sin(-ship_heading)
    rel_x = c * dx - s * dy
    rel_y = s * dx + c * dy
    x_rel = rel_x / half_len

    if x_rel < -bow_taper_start:
        t = (-x_rel - bow_taper_start) / (1 - bow_taper_start)
        t = min(t, 1.0)
        local_half_beam = half_beam * (1 - 0.9 * t ** 1.5)
        local_half_beam = max(local_half_beam, half_beam * 0.05)
    else:
        local_half_beam = half_beam

    if rel_x < -half_len or rel_x > half_len:
        return 0, 0, 0
    if abs(rel_y) > local_half_beam:
        return 0, 0, 0

    dist_to_bow = rel_x - (-half_len)
    dist_to_side = local_half_beam - abs(rel_y)

    if x_rel < -bow_taper_start:
        t = (-x_rel - bow_taper_start) / (1 - bow_taper_start)
        t = min(t, 1.0)
        bow_angle = 0.3 + 0.5 * t
        if rel_y >= 0:
            nx_local, ny_local = -np.cos(bow_angle), np.sin(bow_angle)
        else:
            nx_local, ny_local = -np.cos(bow_angle), -np.sin(bow_angle)
        pen = min(dist_to_bow, dist_to_side) + 0.5
    else:
        if dist_to_bow < dist_to_side:
            nx_local, ny_local = -1, 0
            pen = dist_to_bow
        else:
            nx_local = 0
            ny_local = 1 if rel_y > 0 else -1
            pen = dist_to_side

    nx = np.cos(ship_heading) * nx_local - np.sin(ship_heading) * ny_local
    ny = np.sin(ship_heading) * nx_local + np.cos(ship_heading) * ny_local
    return max(pen, 0), nx, ny


def simulate_ice_ship_interaction(
    polygons, centroids, bonds, ship_pos, ship_length, ship_beam,
    n_frames, dt, drift_vel, thicknesses, ship_props,
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
    Simulate ice-ship interaction with fracture mechanics.

    config: SimulationConfig or None for defaults.
    replan_every_n_frames: If set with replan_callback, call callback every N frames to get a new
        trajectory segment (receding-horizon, AUTO-IceNav style). Callback receives current ice state
        and ship state and returns a list of (position_2d, heading) for the remainder, or None to keep current.
    replan_callback: Callable (frame, world_polygons, centroids, thicknesses, ship_pos, ship_heading, n_remaining)
        -> Optional[list of (position_2d, heading)] of length n_remaining + 1.
    frame_callback: Optional callable(frame, positions, rotations, bond_intact, ship_pos, ship_heading)
        called at the end of each frame (e.g. for real-time 2D view). Called with current state copies.
    ocean_current: OceanCurrentModel or None (no current).  When set, drag
        uses relative velocity v_ice - v_water(x,y) per Eq. 9, and the ship
        damping uses relative velocity in the body frame.
    ship_drift_only: If True and using ship dynamics, do not apply PD control
        (tau_control = 0) so the ship drifts with currents/ice forces only.
    goal_point: Optional (x, y) [m]. If set, simulation stops when ship is within goal_tolerance of this point.
    goal_y_line: Optional y [m] (e.g. top edge). If set, simulation stops when ship_pos[1] >= goal_y_line - goal_tolerance.
        Takes precedence over goal_point when both are set (line goal for full-map).
    goal_tolerance: Distance [m] for goal reached check. Default 15.0.

    Returns (positions_history, rotations_history, bonds_history, ship_history, submergence_history[, ship_diagnostics or damage_summary]).
    When ship_trajectory is not None and return_ship_diagnostics is False, the 6th element is damage_summary
    (dict with 'impulse_total': cumulative impulse from ice on ship). When return_ship_diagnostics is True,
    the 6th element is ship_diagnostics (list) and the last entry includes 'damage_summary'.
    """
    cfg = config or get_default_simulation_config()
    if ocean_current is None:
        ocean_current = NoCurrentModel()
    has_current = ocean_current.is_active
    if has_current:
        print(f"  Ocean current: {type(ocean_current).__name__}", flush=True)
    n = len(polygons)
    areas = np.array([polygon_area(p) for p in polygons])
    masses = areas * thicknesses * IceProperties.ICE_DENSITY
    radii = np.array([np.sqrt(a / np.pi) for a in areas])
    # Moment of inertia (2D, about z): I = 0.5 * m * r^2 (disk approximation)
    inertia = np.maximum(0.5 * masses * radii ** 2, 1e-10)

    local_polys = []
    for poly in polygons:
        centroid = polygon_centroid(poly)
        local_polys.append(poly - centroid)

    positions = centroids.copy()
    velocities = np.zeros_like(positions)
    rotations = np.zeros(n)
    angular_vels = np.zeros(n)
    submergence = np.zeros(n)

    bond_intact = np.ones(len(bonds), dtype=bool)
    rest_lengths = np.array([np.linalg.norm(centroids[i] - centroids[j]) for i, j in bonds])

    bond_strengths = np.zeros(len(bonds))
    bond_areas = np.zeros(len(bonds))
    bond_i = np.array([b[0] for b in bonds])
    bond_j = np.array([b[1] for b in bonds])
    for b, (i, j) in enumerate(bonds):
        h = min(thicknesses[i], thicknesses[j])
        w = min((radii[i] + radii[j]) / 2, cfg.max_bond_width)
        bond_strengths[b] = IceProperties.breaking_force(h, w)
        bond_areas[b] = w * h

    use_dpdem_bonds = getattr(cfg, "use_dpdem_style_bonds", True)
    if use_dpdem_bonds and len(bonds) > 0:
        bond_damage = np.zeros(len(bonds))
        bond_delta_m_0 = np.zeros(len(bonds))
        bond_delta_m_max = np.zeros(len(bonds))
        bond_delta_s = np.zeros(len(bonds))
        nu = getattr(cfg, "poisson_ratio", 0.3)
        E = IceProperties.YOUNGS_MODULUS
        prefactor = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        k_n = prefactor
        k_s = prefactor * (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
        sigma_bar_bond = np.maximum(bond_strengths / np.maximum(bond_areas, 1e-10), 1e6)
        print("  DPDEM-style bond fracture enabled (hybrid criterion, damage, Mohr-Coulomb)", flush=True)
    else:
        bond_damage = bond_delta_m_0 = bond_delta_m_max = bond_delta_s = None
        k_n = k_s = sigma_bar_bond = None

    if len(bonds) > 0:
        print(f"  Bond strengths: {bond_strengths.min()/1e6:.2f} - {bond_strengths.max()/1e6:.2f} MN", flush=True)

    # Rayleigh-based time step (Eq. 11): Δt = b·πr/(0.163ν+0.877)·√(ρ/G); advisory or cap
    nu_r = getattr(cfg, "poisson_ratio", 0.3)
    E_ice = IceProperties.YOUNGS_MODULUS
    G_shear = E_ice / (2.0 * (1.0 + nu_r))
    rho_ice = IceProperties.ICE_DENSITY
    r_min = float(radii.min())
    b_safe = getattr(cfg, "rayleigh_safety_factor", 0.1)
    dt_rayleigh = b_safe * np.pi * r_min / (0.163 * nu_r + 0.877) * np.sqrt(rho_ice / G_shear)
    print(f"  Rayleigh dt (advisory): {dt_rayleigh:.4f} s (requested dt={dt:.4f} s)", flush=True)
    use_rayleigh_cap = getattr(cfg, "use_rayleigh_dt_cap", False)
    dt_ice = min(dt, dt_rayleigh) if use_rayleigh_cap else dt
    if use_rayleigh_cap and dt_rayleigh < dt:
        print(f"  Using ice dt capped at Rayleigh: {dt_ice:.4f} s", flush=True)
    n_ice_steps = max(1, int(round(dt / dt_ice))) if dt_ice > 0 else 1
    if n_ice_steps > 1:
        print(f"  Ice sub-steps per frame: {n_ice_steps} (ice time {n_ice_steps * dt_ice:.4f} s/frame)", flush=True)

    use_bond_forces = getattr(cfg, "use_bond_forces_in_dynamics", True)

    # Precompute pair indices for ice-ice contact (upper triangle i < j).
    pair_i = np.repeat(np.arange(n), np.arange(n - 1, -1, -1))
    pair_j = np.concatenate([np.arange(i + 1, n) for i in range(n)])

    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    bonds_history = [bond_intact.copy()]
    assert bond_intact.all(), "Initial bond_intact must be all True"
    ship_history = []
    submergence_history = [submergence.copy()]

    ship_impact_force = (ship_props.impact_force(drift_vel, np.mean(thicknesses))
                         if (drift_vel > 0 and len(thicknesses) > 0) else 0.0)
    print(f"  Ship impact force: {ship_impact_force/1e6:.2f} MN", flush=True)
    collision_count = 0
    ever_collided = set()

    # Per-floe activation: elements stay frozen until the ship touches their floe.
    # Use satellite parent IDs (from tessellation) if provided; otherwise fall back to
    # bond-connected components (which may merge nearby floes).
    if parent_ids is not None:
        initial_floe_of = np.asarray(parent_ids, dtype=int)
    else:
        _init_comps = find_connected_components(n, bonds, np.ones(len(bonds), dtype=bool))
        initial_floe_of = np.zeros(n, dtype=int)
        for _fid, _comp in enumerate(_init_comps):
            for _ei in _comp:
                initial_floe_of[_ei] = _fid
    elem_active = np.zeros(n, dtype=bool)

    # Per-floe DPDEM eligibility: only enable progressive damage when the ship's
    # (thrust-capped) collision force can plausibly initiate fracture — i.e. when
    # the force exceeds the collision-based breaking threshold.  For thick ice the
    # ship force is too small and the floe stays "unbreakable" (pre-DPDEM behaviour).
    n_floes = int(initial_floe_of.max()) + 1 if len(initial_floe_of) > 0 else 0
    dpdem_floe_eligible = np.zeros(n_floes, dtype=bool)
    if len(bonds) > 0:
        min_collision_threshold = np.min(bond_strengths * cfg.effective_strength_collision)
    else:
        min_collision_threshold = np.inf

    use_ship_dynamics = ship_trajectory is not None and not kinematic_ship
    ship_diagnostics = [] if (use_ship_dynamics and return_ship_diagnostics) else None
    J_cumul = 0.0  # cumulative impulse from ice on ship (damage proxy)
    N_HULL_SEG = 10  # longitudinal segments for per-hull impulse (AUTO-IceNav style)
    hull_impulse_segments = []  # list of (x_body, Jx, Jy) per segment
    hull_impulse_contacts = None  # per-contact (AUTO-IceNav): list of (cx_body, cy_body, Jx_body, Jy_body)
    if ship_trajectory is not None:
        hull_impulse_contacts = []
        half_len = ship_length / 2
        seg_len = ship_length / N_HULL_SEG
        hull_impulse_segments = [
            [-half_len + (k + 0.5) * seg_len, 0.0, 0.0] for k in range(N_HULL_SEG)
        ]  # [x_center_body, Jx, Jy]
    commanded_trajectory = []  # (pos_2d, heading) per frame — what we commanded the ship to follow (for costmap movie)
    if use_ship_dynamics:
        ship_pos_dyn = np.array(ship_trajectory[0][0], dtype=float)
        ship_heading_dyn = float(ship_trajectory[0][1])
        ship_u, ship_v, ship_r = 0.0, 0.0, 0.0
        m = ship_props.mass
        Izz = (1.0 / 12.0) * m * (ship_length**2 + ship_beam**2)
        if getattr(cfg, "use_mss_style_dynamics", False):
            M3, D3 = get_mss_style_3dof(
                m, Izz,
                ship_damping_xy=cfg.ship_damping_xy,
                ship_damping_psi=cfg.ship_damping_psi,
                added_mass_surge_frac=cfg.mss_added_mass_surge_frac,
                added_mass_sway_frac=cfg.mss_added_mass_sway_frac,
                added_mass_yaw_frac=cfg.mss_added_mass_yaw_frac,
            )
        else:
            M3 = np.diag([m, m, Izz])
            D3 = np.diag([m * cfg.ship_damping_xy, m * cfg.ship_damping_xy, Izz * cfg.ship_damping_psi])
        M3inv = np.linalg.inv(M3)
        Kp_xy = ship_props.mass * cfg.Kp_xy_scale
        Kp_psi = cfg.Kp_psi
        Kd_ship = cfg.Kd_ship

    auto_icenav_adapter = None
    if use_ship_dynamics and use_auto_icenav_ship:
        from medea_ice_ship.auto_icenav_ship import AutoIceNavShip
        eta0 = [float(ship_pos_dyn[0]), float(ship_pos_dyn[1]), float(ship_heading_dyn)]
        nu0 = [0.0, 0.0, 0.0]
        dt_sim = dt
        target_speed = getattr(cfg, "planning_target_speed", 4.0)
        max_accel = getattr(cfg, "planning_max_accel", 0.5)
        auto_icenav_adapter = AutoIceNavShip(
            eta0=eta0,
            nu0=nu0,
            dt=dt_sim,
            target_speed=target_speed,
            max_accel=max_accel,
        )
        path_xyh = np.array(
            [[p[0][0], p[0][1], p[1]] for p in ship_trajectory],
            dtype=float,
        )
        auto_icenav_adapter.init_trajectory_tracking(path_xyh)
        print("  AUTO-IceNav ship dynamics enabled (Fossen 3-DOF + DP + trajectory setpoints).", flush=True)

    for frame in range(n_frames):
        # Receding-horizon replan (AUTO-IceNav style): update trajectory from current ice state
        if (
            use_ship_dynamics
            and replan_callback is not None
            and replan_every_n_frames is not None
            and replan_every_n_frames > 0
            and frame % replan_every_n_frames == 0
        ):
            n_remaining = n_frames - frame
            world_polygons = []
            for i in range(n):
                c, s = np.cos(rotations[i]), np.sin(rotations[i])
                R = np.array([[c, -s], [s, c]])
                world_polygons.append(local_polys[i] @ R.T + positions[i])
            new_traj = replan_callback(
                frame, world_polygons, positions.copy(), thicknesses,
                ship_pos_dyn.copy(), float(ship_heading_dyn), n_remaining,
            )
            if new_traj is not None and len(new_traj) >= 1:
                end_slice = frame + len(new_traj)
                ship_trajectory[frame:end_slice] = new_traj
                if end_slice < len(ship_trajectory):
                    ship_trajectory[end_slice:] = [new_traj[-1]] * (len(ship_trajectory) - end_slice)

        if use_ship_dynamics:
            ship_pos_frame = ship_pos_dyn.copy()
            ship_heading = ship_heading_dyn
        elif ship_trajectory is not None:
            ship_pos_frame, ship_heading = ship_trajectory[min(frame, len(ship_trajectory) - 1)]
        else:
            ship_pos_frame = ship_pos
            ship_heading = 0.0

        # Thick-ice mode: decide at frame start so we never fragment components or carry over broken bonds.
        max_fracture_thickness = getattr(cfg, "max_fracture_thickness", 15.0)
        all_ice_thick = len(thicknesses) > 0 and np.all(thicknesses > max_fracture_thickness)
        if all_ice_thick and len(bonds) > 0:
            bond_intact[:] = True
            components = find_connected_components(n, bonds, np.ones(len(bonds), dtype=bool))
        else:
            components = find_connected_components(n, bonds, bond_intact)

        forces = np.zeros_like(positions)
        moments = np.zeros(n)
        ship_collision_force = np.zeros(n)
        F_ice_ship = np.zeros(2)
        M_ice_ship = 0.0
        F_ice_ship_collision = np.zeros(2)
        M_ice_ship_collision = 0.0
        F_ship_on_ice = np.zeros((n, 2))

        ship_hull_pts = create_ship_hull_at_position_heading(
            ship_length, ship_beam, ship_pos_frame, ship_heading
        )

        for i in range(n):
            px, py = positions[i]
            pen, nx, ny = 0, 0, 0
            contact_x, contact_y = px, py  # contact point (world) for hull impulse
            angle = rotations[i]
            c, s = np.cos(angle), np.sin(angle)
            rotated = np.zeros_like(local_polys[i])
            rotated[:, 0] = local_polys[i][:, 0] * c - local_polys[i][:, 1] * s
            rotated[:, 1] = local_polys[i][:, 0] * s + local_polys[i][:, 1] * c
            world_poly = rotated + positions[i]

            for vx, vy in world_poly:
                test_pen, test_nx, test_ny = point_to_ship_collision(
                    vx, vy, ship_pos_frame, ship_length, ship_beam, ship_heading=ship_heading
                )
                if test_pen > pen:
                    pen, nx, ny = test_pen, test_nx, test_ny
                    contact_x, contact_y = vx, vy

            n_verts = len(world_poly)
            for j in range(n_verts):
                v1 = world_poly[j]
                v2 = world_poly[(j + 1) % n_verts]
                for k in range(1, cfg.n_edge_samples + 1):
                    t = k / (cfg.n_edge_samples + 1)
                    pt = (1 - t) * v1 + t * v2
                    test_pen, test_nx, test_ny = point_to_ship_collision(
                        pt[0], pt[1], ship_pos_frame, ship_length, ship_beam, ship_heading=ship_heading
                    )
                    if test_pen > pen:
                        pen, nx, ny = test_pen, test_nx, test_ny
                        contact_x, contact_y = pt[0], pt[1]

            for sx, sy in ship_hull_pts:
                if point_in_polygon((sx, sy), world_poly):
                    test_pen, test_nx, test_ny = point_inside_polygon_penetration(sx, sy, world_poly)
                    if test_pen > pen:
                        pen, nx, ny = test_pen, test_nx, test_ny
                        contact_x, contact_y = sx, sy

            if pen > 0:
                collision_count += 1
                ever_collided.add(i)
                elem_active[initial_floe_of == initial_floe_of[i]] = True
                w_contact = min(cfg.w_contact_radius_factor * radii[i], cfg.w_contact_cap)
                F_break = IceProperties.breaking_force(thicknesses[i], w_contact)
                # Thickness-dependent pen_crit: thick ice requires more penetration before force reaches F_break (realistic bending).
                pen_crit_i = cfg.pen_crit_ref * (thicknesses[i] / cfg.pen_crit_thickness_ref)
                force_mag = F_break * min(pen / pen_crit_i, 1.0)
                forces[i, 0] += force_mag * nx
                forces[i, 1] += force_mag * ny
                ship_collision_force[i] = force_mag
                rel_y = py - ship_pos_frame[1]
                side_direction = 1 if rel_y > 0 else -1
                side_push = cfg.side_push_factor * force_mag * side_direction
                forces[i, 1] += side_push
                F_ship_i = np.array([force_mag * nx, force_mag * ny + side_push])
                F_ship_on_ice[i] = F_ship_i
                F_ice_ship_collision -= F_ship_i
                # Per-hull-segment and per-contact impulse (AUTO-IceNav style): store in body frame
                rel_world = np.array([contact_x - ship_pos_frame[0], contact_y - ship_pos_frame[1]])
                contact_body = R_xy(ship_heading).T @ rel_world
                J_world = F_ship_i * dt
                J_body = R_xy(ship_heading).T @ J_world
                if hull_impulse_contacts is not None:
                    # Approach speed along contact normal (for plot-only scaling: bow = approaching = large, stern/wake = receding = small)
                    ship_vel_world = (
                        R_xy(ship_heading) @ np.array([ship_u, ship_v])
                        if use_ship_dynamics
                        else np.zeros(2)
                    )
                    rel_vel = ship_vel_world - velocities[i]
                    # n = outward normal from hull; (ship - ice)·n > 0 when ship is moving into ice
                    approach_speed = max(0.0, float(np.dot(rel_vel, np.array([nx, ny]))))
                    hull_impulse_contacts.append((
                        float(contact_body[0]), float(contact_body[1]),
                        float(J_body[0]), float(J_body[1]),
                        approach_speed,
                    ))
                if hull_impulse_segments:
                    dx = contact_body[0]
                    half_len = ship_length / 2
                    seg_len = ship_length / len(hull_impulse_segments)
                    k = int((dx + half_len) / seg_len)
                    k = max(0, min(k, len(hull_impulse_segments) - 1))
                    hull_impulse_segments[k][1] += J_body[0]
                    hull_impulse_segments[k][2] += J_body[1]
                r_vec = positions[i] - ship_pos_frame
                M_ice_ship_collision += r_vec[0] * (-F_ship_i[1]) - r_vec[1] * (-F_ship_i[0])
                submergence[i] = min(
                    submergence[i] + cfg.submergence_rate * min(pen, cfg.submergence_pen_cap) * dt,
                    thicknesses[i]
                )
                z_sub = min(submergence[i], thicknesses[i])
                RHO_W = IceProperties.WATER_DENSITY
                G = 9.81
                F_buoy = RHO_W * G * areas[i] * z_sub
                forces[i, 0] += F_buoy * nx
                forces[i, 1] += F_buoy * ny
                F_ice_ship[0] -= F_buoy * nx
                F_ice_ship[1] -= F_buoy * ny
                r_vec_b = positions[i] - ship_pos_frame
                M_ice_ship += F_buoy * (r_vec_b[1] * nx - r_vec_b[0] * ny)
                # Mf: moment from buoyancy (rf × Ff); rf = center of buoyancy to CM, use contact point
                r_cb = np.array([contact_x - positions[i, 0], contact_y - positions[i, 1]])
                moments[i] += r_cb[0] * (F_buoy * ny) - r_cb[1] * (F_buoy * nx)
                # Mc: moment from contact (r × F) for Eq. 8
                r_contact = np.array([contact_x - positions[i, 0], contact_y - positions[i, 1]])
                moments[i] += r_contact[0] * F_ship_i[1] - r_contact[1] * F_ship_i[0]

        # Optional thrust cap: limit ice-on-ship collision force so ship cannot break very thick ice by unlimited push.
        if cfg.max_ship_ice_force is not None:
            F_coll_mag = np.linalg.norm(F_ice_ship_collision)
            if F_coll_mag > cfg.max_ship_ice_force and F_coll_mag > 1e-6:
                scale = cfg.max_ship_ice_force / F_coll_mag
                for i in range(n):
                    if ship_collision_force[i] > 0:
                        forces[i] -= (1.0 - scale) * F_ship_on_ice[i]
                        ship_collision_force[i] *= scale
                F_ice_ship_collision *= scale
                M_ice_ship_collision *= scale

        # Dissipation cap (inelastic collision): limit collision impulse so the ship loses KE (stops) but does not bounce.
        # Apply to collision force/moment only, using collision direction (not total F_ice_ship), so we don't reverse approach velocity.
        if (
            use_ship_dynamics
            and getattr(cfg, "collision_impulse_dissipation_cap", True)
            and np.linalg.norm(F_ice_ship_collision) > 1e-6
        ):
            v_ship_world = R_xy(ship_heading_dyn) @ np.array([ship_u, ship_v])
            F_coll_mag = float(np.linalg.norm(F_ice_ship_collision))
            F_coll_dir = F_ice_ship_collision / F_coll_mag
            # v_approach = ship speed toward ice (along -F_collision direction); positive when approaching
            v_approach = max(0.0, float(-np.dot(v_ship_world, F_coll_dir)))
            J_coll_mag = F_coll_mag * dt
            J_max = ship_props.mass * v_approach * getattr(cfg, "collision_impulse_cap_factor", 1.0)
            if J_coll_mag > J_max and J_max > 0:
                scale = J_max / J_coll_mag
                F_ice_ship_collision = F_ice_ship_collision * scale
                M_ice_ship_collision = M_ice_ship_collision * scale

        F_ice_ship += F_ice_ship_collision
        M_ice_ship += M_ice_ship_collision

        # For zero-restitution clamp: remember collision direction in world frame (ship must not move away from ice)
        collision_dir_world = None
        if np.linalg.norm(F_ice_ship_collision) > 1e-6:
            collision_dir_world = F_ice_ship_collision / np.linalg.norm(F_ice_ship_collision)

        # After thrust cap: check if the (capped) collision force is large enough to
        # plausibly initiate fracture for each active floe.  Once eligible, stays eligible.
        for i in range(n):
            if ship_collision_force[i] > min_collision_threshold and thicknesses[i] <= max_fracture_thickness:
                dpdem_floe_eligible[initial_floe_of[i]] = True

        no_collision = ship_collision_force <= 0
        submergence[no_collision] = np.maximum(0, submergence[no_collision] - (1.0 / cfg.rise_time_const) * submergence[no_collision] * dt)

        bow_contact_force = 0.0
        c_r, s_r = np.cos(ship_heading), np.sin(ship_heading)
        for i in range(n):
            if ship_collision_force[i] <= 0:
                continue
            dx = positions[i, 0] - ship_pos_frame[0]
            dy = positions[i, 1] - ship_pos_frame[1]
            body_x = c_r * dx + s_r * dy
            if body_x > 0:
                bow_contact_force += ship_collision_force[i]
        if bow_contact_force > 0:
            F_rubble_mag = min(cfg.rubble_coef * bow_contact_force, cfg.rubble_cap_n)
            forward = np.array([c_r, s_r])
            F_ice_ship -= F_rubble_mag * forward

        # Ship damage proxy: cumulative impulse from ice on ship (AUTO-IceNav style; used for risk/cost)
        J_cumul += np.linalg.norm(F_ice_ship) * dt

        for ice_step in range(n_ice_steps):
            if ice_step > 0:
                forces = np.zeros_like(positions)
                moments = np.zeros(n)

            # ── Force accumulation (bonds, ice-ice contact, drag) ──────────
            RHO_WATER = IceProperties.WATER_DENSITY
            A_drag = areas * (thicknesses / cfg.drag_ref_thickness)

            if use_bond_forces and len(bonds) > 0:
                d_bond_pre = positions[bond_j] - positions[bond_i]
                dist_bond_pre = np.linalg.norm(d_bond_pre, axis=1)
                dist_bond_pre_safe = np.maximum(dist_bond_pre, 1e-10)
                bond_dir_pre = d_bond_pre / dist_bond_pre_safe[:, np.newaxis]
                avg_thickness_pre = (thicknesses[bond_i] + thicknesses[bond_j]) * 0.5
                stiffness_pre = IceProperties.YOUNGS_MODULUS * avg_thickness_pre * cfg.bond_stiffness_factor
                if use_dpdem_bonds:
                    F_mag_pre = stiffness_pre * (dist_bond_pre - rest_lengths) * (1.0 - bond_damage)
                else:
                    F_mag_pre = stiffness_pre * (dist_bond_pre - rest_lengths)
                F_bond_vec = bond_dir_pre * F_mag_pre[:, np.newaxis]
                F_bond_vec = np.where(bond_intact[:, np.newaxis], F_bond_vec, 0.0)
                both_active = elem_active[bond_i] & elem_active[bond_j]
                both_frozen = ~elem_active[bond_i] & ~elem_active[bond_j]
                bond_force_ok = both_active | both_frozen
                F_bond_vec *= bond_force_ok[:, np.newaxis].astype(float)
                np.add.at(forces, bond_i, F_bond_vec)
                np.add.at(forces, bond_j, -F_bond_vec)
                if np.any(elem_active):
                    bond_cross = 0.5 * (d_bond_pre[:, 0] * F_bond_vec[:, 1] - d_bond_pre[:, 1] * F_bond_vec[:, 0])
                    np.add.at(moments, bond_i, bond_cross)
                    np.add.at(moments, bond_j, -bond_cross)
    
            d_vec = positions[pair_j] - positions[pair_i]
            dist_ij = np.linalg.norm(d_vec, axis=1)
            min_dist = radii[pair_i] + radii[pair_j]
            overlap = np.where((dist_ij < min_dist) & (dist_ij > 1e-6), min_dist - dist_ij, 0.0)
            olap_mask = overlap > 0
            f_n_vec = np.zeros_like(d_vec)
            f_t_vec = np.zeros_like(d_vec)
            if np.any(olap_mask):
                direction = np.zeros_like(d_vec)
                direction[olap_mask] = d_vec[olap_mask] / dist_ij[olap_mask, np.newaxis]
                f_n_mag = cfg.hertz_factor * overlap ** cfg.hertz_exponent
                f_n_vec = direction * f_n_mag[:, np.newaxis]
                np.add.at(forces, pair_i, -f_n_vec)
                np.add.at(forces, pair_j, f_n_vec)
                v_rel = velocities[pair_j] - velocities[pair_i]
                v_t = v_rel - np.sum(v_rel * direction, axis=1, keepdims=True) * direction
                v_t_mag = np.linalg.norm(v_t, axis=1)
                friction_mask = olap_mask & (v_t_mag > 1e-6)
                if np.any(friction_mask):
                    t_hat = np.zeros_like(v_t)
                    t_hat[friction_mask] = -v_t[friction_mask] / v_t_mag[friction_mask, np.newaxis]
                    f_t_mag = np.minimum(cfg.ice_ice_friction * f_n_mag, cfg.friction_cap)
                    f_t_vec = t_hat * f_t_mag[:, np.newaxis]
                    np.add.at(forces, pair_i, f_t_vec)
                    np.add.at(forces, pair_j, -f_t_vec)
                if np.any(elem_active):
                    F_on_i = -f_n_vec - f_t_vec
                    cross_val = np.zeros(len(pair_i))
                    cross_val[olap_mask] = -(d_vec[olap_mask, 0] * F_on_i[olap_mask, 1] - d_vec[olap_mask, 1] * F_on_i[olap_mask, 0])
                    np.add.at(moments, pair_i, 0.5 * cross_val)
                    np.add.at(moments, pair_j, -0.5 * cross_val)
    
            if ship_trajectory is None:
                for comp in components:
                    total_mass_c = sum(masses[ci] for ci in comp)
                    drift_force = drift_vel * total_mass_c * cfg.drift_factor
                    for ci in comp:
                        forces[ci, 0] += drift_force * masses[ci] / total_mass_c
    
            # Eq. 9: Fd = 0.5·CFd·ρw·Σ Asub·(v_ice - v_water)·|v_ice - v_water|
            v_water = ocean_current.sample_batch(positions)  # (n, 2)
            v_rel = velocities - v_water
            v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)
            v_rel_mag_safe = np.maximum(v_rel_mag, 1e-12)
            v_rel_mag_1d = v_rel_mag_safe.ravel()
            drag_force = np.where(
                v_rel_mag > 1e-6,
                (v_rel / v_rel_mag_safe) * (0.5 * RHO_WATER * cfg.c_d_drag * np.expand_dims(A_drag, axis=1) * (v_rel_mag_1d ** 2)[:, np.newaxis]),
                0.5 * RHO_WATER * cfg.c_d_drag * np.expand_dims(A_drag, axis=1) * cfg.linear_drag_factor * v_rel,
            )
            forces -= drag_force
    
            A_sub_area = areas * np.minimum(submergence / np.maximum(thicknesses, 1e-10), 1.0)
            CMd = getattr(cfg, "drag_coefficient_moment", 0.4)
            Md = -0.5 * CMd * RHO_WATER * (radii ** 3) * angular_vels * np.abs(angular_vels) * A_sub_area
            moments += Md
    
            # ── Rigid-body component integration (legacy) ─────────────────
            # Each bonded component shares a single velocity (COM), maintaining
            # relative positions.  This prevents internal "bouncing" while the
            # DPDEM damage criterion still operates on end-of-frame bond stress.
            new_velocities = velocities.copy()
            new_positions = positions.copy()
            for comp in components:
                if not any(elem_active[ci] for ci in comp) and not has_current:
                    continue
                if len(comp) == 1:
                    i = comp[0]
                    accel = forces[i] / masses[i]
                    new_velocities[i] += accel * dt_ice
                    new_velocities[i] = np.clip(new_velocities[i], cfg.velocity_clip[0], cfg.velocity_clip[1])
                    new_positions[i] += new_velocities[i] * dt_ice
                else:
                    total_mass = sum(masses[ci] for ci in comp)
                    com = sum(masses[ci] * positions[ci] for ci in comp) / total_mass
                    total_force = sum(forces[ci] for ci in comp)
                    comp_vel = sum(masses[ci] * velocities[ci] for ci in comp) / total_mass
                    accel = total_force / total_mass
                    comp_vel += accel * dt_ice
                    comp_vel = np.clip(comp_vel, cfg.velocity_clip[0], cfg.velocity_clip[1])
                    new_com = com + comp_vel * dt_ice
                    for ci in comp:
                        new_velocities[ci] = comp_vel
                        new_positions[ci] = new_com + (positions[ci] - com)
            velocities = new_velocities
            positions = new_positions
    
            # Sanitize after sub-stepping
            if np.any(elem_active):
                bad_vel = ~np.isfinite(velocities)
                velocities = np.where(bad_vel, 0.0, velocities)
                velocities = np.clip(velocities, cfg.velocity_clip[0], cfg.velocity_clip[1])
                bad_pos = ~np.isfinite(positions)
                if np.any(bad_pos):
                    positions = np.where(bad_pos, positions_history[-1], positions)
                    still_bad = ~np.isfinite(positions)
                    if np.any(still_bad):
                        domain_cx = float(np.nanmean(positions[:, 0])) if np.any(np.isfinite(positions[:, 0])) else 0.0
                        domain_cy = float(np.nanmean(positions[:, 1])) if np.any(np.isfinite(positions[:, 1])) else 0.0
                        if not np.isfinite(domain_cx):
                            domain_cx = 0.0
                        if not np.isfinite(domain_cy):
                            domain_cy = 0.0
                        positions = np.where(still_bad, np.array([domain_cx, domain_cy]), positions)
                position_domain = getattr(cfg, "position_clip_m", 1e4)
                positions = np.clip(positions, -position_domain, position_domain)
    
            element_stress = ship_collision_force.copy()
            # Bond stress accumulation and breaking (vectorized).
            intact = bond_intact
            d_bond = positions[bond_j] - positions[bond_i]
            dist_bond = np.linalg.norm(d_bond, axis=1)
            dist_bond_safe = np.maximum(dist_bond, 1e-10)
            bond_dir = d_bond / dist_bond_safe[:, np.newaxis]
            stretch_bond = np.abs(dist_bond - rest_lengths)
            avg_thickness_bond = (thicknesses[bond_i] + thicknesses[bond_j]) * 0.5
            stiffness_bond = IceProperties.YOUNGS_MODULUS * avg_thickness_bond * cfg.bond_stiffness_factor
            bond_force_mag = stiffness_bond * stretch_bond
            np.add.at(element_stress, bond_i, np.where(intact, bond_force_mag, 0))
            np.add.at(element_stress, bond_j, np.where(intact, bond_force_mag, 0))
            for _ in range(cfg.stress_diffusion_steps):
                new_stress = element_stress.copy()
                transfer_i_to_j = intact & (element_stress[bond_i] > element_stress[bond_j])
                transfer_j_to_i = intact & (element_stress[bond_j] >= element_stress[bond_i])
                np.add.at(new_stress, bond_j, np.where(transfer_i_to_j, element_stress[bond_i] * cfg.stress_transfer, 0))
                np.add.at(new_stress, bond_i, np.where(transfer_j_to_i, element_stress[bond_j] * cfg.stress_transfer, 0))
                element_stress = new_stress
    
            if use_dpdem_bonds and len(bonds) > 0:
                both_ep_active = elem_active[bond_i] & elem_active[bond_j]
                floe_ok = (
                    dpdem_floe_eligible[initial_floe_of[bond_i]]
                    & dpdem_floe_eligible[initial_floe_of[bond_j]]
                )
                # Ice thicker than max_fracture_thickness cannot fracture (unbreakable / land-like).
                bond_breakable_thickness = (thicknesses[bond_i] <= max_fracture_thickness) & (thicknesses[bond_j] <= max_fracture_thickness)

                # Spatial locality: DPDEM damage only on bonds at or near the crack front.
                #  (a) at least one endpoint is directly colliding with the ship, OR
                #  (b) at least one endpoint is shared with an already-broken bond (crack tip).
                bond_near_ship = (ship_collision_force[bond_i] > 0) | (ship_collision_force[bond_j] > 0)
                elem_at_crack = np.zeros(n, dtype=bool)
                broken_mask = ~bond_intact
                if np.any(broken_mask):
                    elem_at_crack[bond_i[broken_mask]] = True
                    elem_at_crack[bond_j[broken_mask]] = True
                bond_at_crack = elem_at_crack[bond_i] | elem_at_crack[bond_j]
                bond_dpdem_ok = both_ep_active & floe_ok & (bond_near_ship | bond_at_crack) & bond_breakable_thickness
    
                v_rel = velocities[bond_j] - velocities[bond_i]
                v_n = np.sum(v_rel * bond_dir, axis=1)
                v_t_vec = v_rel - v_n[:, np.newaxis] * bond_dir
                ds_incr = np.linalg.norm(v_t_vec, axis=1) * dt_ice
                bond_delta_s[:] = bond_delta_s + np.where(bond_dpdem_ok, ds_incr, 0.0)
                if frame > 0 and np.any(bond_dpdem_ok):
                    delta_n = dist_bond - rest_lengths
                    delta_s = bond_delta_s
                    L0 = np.maximum(rest_lengths, 1e-10)
                    sigma_n_elastic = k_n * (delta_n / L0)
                    sigma_t_elastic = k_s * (delta_s / L0)
                    sigma_n = sigma_n_elastic * (1.0 - bond_damage)
                    sigma_t = sigma_t_elastic * (1.0 - bond_damage)
                    bond_damage[:], bond_delta_m_max[:], bond_delta_m_0[:], new_intact = update_bond_damage_state(
                        delta_n, delta_s, sigma_n, sigma_t, sigma_bar_bond,
                        cfg.bond_cohesion, cfg.bond_internal_friction,
                        cfg.bond_G_c_I, cfg.bond_G_c_II, cfg.bond_eta_bk,
                        bond_damage, bond_delta_m_max, bond_delta_m_0, intact,
                    )
                    # Only apply damage-model result to bonds that are breakable by thickness (thick ice stays intact)
                    if not all_ice_thick:
                        bond_intact[:] = np.where(bond_breakable_thickness, new_intact, bond_intact)
                # Ship collision can still break bonds (only when there is actual contact)
                if not all_ice_thick:
                    collision_on_bond = (ship_collision_force[bond_i] > 0) | (ship_collision_force[bond_j] > 0)
                    collision_force_on_bond = np.maximum(ship_collision_force[bond_i], ship_collision_force[bond_j])
                    effective_strength = bond_strengths * cfg.effective_strength_collision
                    break_collision = intact & collision_on_bond & (collision_force_on_bond > effective_strength) & bond_breakable_thickness
                    bond_intact[break_collision] = False
            else:
                # Legacy breaking: force vs effective strength
                bond_breakable_thickness = (thicknesses[bond_i] <= max_fracture_thickness) & (thicknesses[bond_j] <= max_fracture_thickness)
                bond_force = stiffness_bond * stretch_bond + np.maximum(element_stress[bond_i], element_stress[bond_j]) * cfg.propagated_stress_factor
                propagated_stress_bond = np.maximum(element_stress[bond_i], element_stress[bond_j])
                collision_on_bond = (ship_collision_force[bond_i] > 0) | (ship_collision_force[bond_j] > 0)
                collision_force_on_bond = np.maximum(ship_collision_force[bond_i], ship_collision_force[bond_j])
                effective_strength = np.where(
                    collision_on_bond,
                    bond_strengths * cfg.effective_strength_collision,
                    np.where(
                        propagated_stress_bond > bond_strengths * cfg.effective_strength_stress_threshold,
                        bond_strengths * cfg.effective_strength_stress,
                        bond_strengths,
                    ),
                )
                break_collision = intact & collision_on_bond & (collision_force_on_bond > effective_strength) & bond_breakable_thickness
                break_force = intact & (bond_force > effective_strength) & bond_breakable_thickness
                if not all_ice_thick:
                    bond_intact[break_collision | break_force] = False
    
            # Eq. 8: I·dω/dt = Mc + Mb + Mh; only update active elements
            if np.any(elem_active):
                alpha = moments / inertia
                angular_vels[elem_active] += alpha[elem_active] * dt_ice
                angular_vels[elem_active] += np.random.randn(int(elem_active.sum())) * cfg.angular_noise
                angular_vels[elem_active] *= cfg.angular_damping
                angular_vels = np.clip(angular_vels, cfg.angular_cap[0], cfg.angular_cap[1])
            rotations += angular_vels * dt_ice
    
        if use_ship_dynamics:
            psi = ship_heading_dyn
            nu = np.array([ship_u, ship_v, ship_r])
            F_ice_body = R_xy(psi).T @ F_ice_ship
            tau_ice = np.array([F_ice_body[0], F_ice_body[1], M_ice_ship])
            if auto_icenav_adapter is not None:
                auto_icenav_adapter.set_env_load_body(tau_ice)
                eta, nu_out = auto_icenav_adapter.step()
                ship_pos_dyn[0] = float(eta[0])
                ship_pos_dyn[1] = float(eta[1])
                ship_heading_dyn = float(eta[2])
                ship_u, ship_v, ship_r = float(nu_out[0]), float(nu_out[1]), float(nu_out[2])
                if collision_dir_world is not None and getattr(cfg, "collision_zero_restitution", True):
                    v_world = R_xy(ship_heading_dyn) @ np.array([ship_u, ship_v])
                    away = float(np.dot(v_world, collision_dir_world))
                    if away > 0:
                        v_world = v_world - away * collision_dir_world
                        nu_xy = R_xy(ship_heading_dyn).T @ v_world
                        ship_u, ship_v = float(nu_xy[0]), float(nu_xy[1])
                ship_pos_frame = ship_pos_dyn.copy()
                ship_heading = ship_heading_dyn
                commanded_trajectory.append((ship_pos_dyn.copy(), ship_heading_dyn))
                ship_history.append((ship_pos_dyn.copy(), float(ship_heading_dyn)))
                if ship_diagnostics is not None:
                    ship_vel_world = R_xy(ship_heading_dyn) @ np.array([ship_u, ship_v])
                    ship_diagnostics.append({
                        't': frame * dt,
                        'F_ice_ship': F_ice_ship.copy(),
                        'ship_vel': ship_vel_world,
                        'ship_pos': ship_pos_dyn.copy(),
                        'desired_pos': ship_pos_dyn.copy(),
                        'desired_vel': np.zeros(2),
                        'impulse_cumul': J_cumul,
                    })
            else:
                tr = ship_trajectory
                idx = min(frame, len(tr) - 1)
                idx_next = min(frame + 1, len(tr) - 1)
                desired_pos = np.array(tr[idx][0])
                desired_heading = tr[idx][1]
                e_xy_world = desired_pos - ship_pos_dyn
                e_xy_body = R_xy(psi).T @ e_xy_world
                e_psi = ssa(desired_heading - psi)
                desired_vel_world = (np.array(tr[idx_next][0]) - desired_pos) / dt if idx_next > idx else np.zeros(2)
                nu_des_xy = R_xy(psi).T @ desired_vel_world
                nu_des = np.array([nu_des_xy[0], nu_des_xy[1], 0.0])
                tau_control = np.array([
                    Kp_xy * e_xy_body[0] + Kd_ship * (nu_des[0] - nu[0]),
                    Kp_xy * e_xy_body[1] + Kd_ship * (nu_des[1] - nu[1]),
                    Kp_psi * e_psi - Kd_ship * nu[2],
                ])
                if ship_drift_only:
                    tau_control = np.zeros(3)
                if getattr(cfg, "use_mss_style_dynamics", False):
                    C = coriolis_3dof(M3, nu)
                else:
                    C = np.zeros((3, 3))
                    C[0, 2] = -m * nu[1]
                    C[1, 2] = m * nu[0]
                    C[2, 0], C[2, 1] = -C[0, 2], -C[1, 2]
                # Current effect on ship: damping uses relative velocity (Fossen body-frame)
                if has_current:
                    v_c_world = ocean_current.sample(ship_pos_dyn[0], ship_pos_dyn[1])
                    v_c_body = R_xy(psi).T @ v_c_world
                    nu_r_ship = nu - np.array([v_c_body[0], v_c_body[1], 0.0])
                else:
                    nu_r_ship = nu
                nu_dot = M3inv @ (tau_control + tau_ice - D3 @ nu_r_ship - C @ nu)
                nu = nu + nu_dot * dt
                vc = cfg.ship_velocity_clip
                nu[0] = np.clip(nu[0], -vc[0], vc[0])
                nu[1] = np.clip(nu[1], -vc[1], vc[1])
                nu[2] = np.clip(nu[2], -vc[2], vc[2])
                # Zero restitution: if we had collision, remove any velocity component away from ice (no bounce)
                if (
                    collision_dir_world is not None
                    and getattr(cfg, "collision_zero_restitution", True)
                ):
                    v_world = R_xy(psi) @ np.array([nu[0], nu[1]])
                    away = float(np.dot(v_world, collision_dir_world))
                    if away > 0:
                        v_world = v_world - away * collision_dir_world
                        nu_xy = R_xy(psi).T @ v_world
                        nu[0], nu[1] = nu_xy[0], nu_xy[1]
                ship_u, ship_v, ship_r = nu[0], nu[1], nu[2]
                u_g, v_g = R_xy(psi) @ np.array([ship_u, ship_v])
                ship_pos_dyn[0] += dt * u_g
                ship_pos_dyn[1] += dt * v_g
                ship_heading_dyn = (ship_heading_dyn + dt * ship_r) % (2 * np.pi)
                commanded_trajectory.append((desired_pos.copy(), desired_heading))
                ship_history.append((ship_pos_dyn.copy(), float(ship_heading_dyn)))
                if ship_diagnostics is not None:
                    desired_vel = desired_vel_world
                    ship_vel_world = R_xy(ship_heading_dyn) @ np.array([ship_u, ship_v])
                    ship_diagnostics.append({
                        't': frame * dt,
                        'F_ice_ship': F_ice_ship.copy(),
                        'ship_vel': ship_vel_world,
                        'ship_pos': ship_pos_dyn.copy(),
                        'desired_pos': desired_pos,
                        'desired_vel': desired_vel,
                        'impulse_cumul': J_cumul,
                    })
        else:
            if ship_trajectory is not None and frame < len(ship_trajectory):
                commanded_trajectory.append((np.array(ship_trajectory[frame][0]), float(ship_trajectory[frame][1])))
            ship_history.append((ship_pos_frame, ship_heading))

        ocean_current.step()

        # Failsafe: thick-ice bonds must never be stored as broken (unbreakable / land-like)
        if len(bonds) > 0:
            unbreakable = (thicknesses[bond_i] > max_fracture_thickness) | (thicknesses[bond_j] > max_fracture_thickness)
            bond_intact[unbreakable] = True

        positions_history.append(positions.copy())
        rotations_history.append(rotations.copy())
        bonds_history.append(bond_intact.copy())
        submergence_history.append(submergence.copy())

        # Sanity: when all ice is above max_fracture_thickness, no bonds must break
        if len(bonds) > 0 and np.all(thicknesses > max_fracture_thickness):
            assert bond_intact.all(), "Thick-ice invariant: no bonds should break when all thicknesses > max_fracture_thickness"

        if frame_callback is not None:
            frame_callback(
                frame,
                positions.copy(),
                rotations.copy(),
                bond_intact.copy(),
                ship_pos_frame.copy(),
                ship_heading,
            )

        # Debug: print ship position frequently when goal is set (every 10 frames)
        if (goal_y_line is not None or goal_point is not None) and frame % 10 == 0:
            sx, sy = float(ship_pos_frame[0]), float(ship_pos_frame[1])
            if goal_y_line is not None:
                thresh = goal_y_line - goal_tolerance
                past = "yes" if sy >= thresh else "no"
                print(f"  [goal debug] frame {frame + 1}  ship (x,y)=({sx:.1f}, {sy:.1f})  goal_y_line={goal_y_line:.0f}  threshold={thresh:.0f}  past? {past}", flush=True)
            else:
                print(f"  [goal debug] frame {frame + 1}  ship (x,y)=({sx:.1f}, {sy:.1f})  goal_point=({goal_point[0]:.1f}, {goal_point[1]:.1f})", flush=True)

        # Early stop when goal is reached (optional)
        if goal_y_line is not None:
            if ship_pos_frame[1] >= goal_y_line - goal_tolerance:
                print(f"  Goal reached at frame {frame + 1} (y >= {goal_y_line - goal_tolerance:.0f} m).", flush=True)
                break
        elif goal_point is not None:
            dist = np.hypot(ship_pos_frame[0] - goal_point[0], ship_pos_frame[1] - goal_point[1])
            if dist <= goal_tolerance:
                print(f"  Goal reached at frame {frame + 1} (distance {dist:.0f} m <= {goal_tolerance:.0f} m).", flush=True)
                break

    # Optional diagnostic: verify first frames have intact bonds (set DPDEM_BOND_DIAG=1 to enable)
    if os.environ.get("DPDEM_BOND_DIAG") == "1" and len(bonds) > 0 and len(bonds_history) >= 2:
        n_intact_0 = int(np.sum(bonds_history[0]))
        n_intact_1 = int(np.sum(bonds_history[1]))
        first_broken_frame = next((k for k in range(len(bonds_history)) if not bonds_history[k].all()), None)
        print(f"  [Bond diagnostic] history[0] intact: {n_intact_0}/{len(bonds)}, history[1] intact: {n_intact_1}/{len(bonds)}, first frame with any broken: {first_broken_frame}", flush=True)
    dpdem_bond_stats = None
    if collision_count > 0:
        print(f"  Total collisions: {collision_count}", flush=True)
        bonded_elements = set()
        for i, j in bonds:
            bonded_elements.add(i)
            bonded_elements.add(j)
        collided_with_bonds = ever_collided & bonded_elements
        collided_without_bonds = ever_collided - bonded_elements
        print(f"  Collided elements: {len(ever_collided)} total, "
              f"{len(collided_with_bonds)} bonded, {len(collided_without_bonds)} unbonded", flush=True)
    else:
        print("  WARNING: No collisions detected!", flush=True)

    if use_dpdem_bonds:
        if len(bonds) > 0:
            n_damaged = (bond_damage > 0.01).sum()
            n_failed_by_damage = (bond_damage >= 1.0).sum()
            dpdem_bond_stats = {
                'n_damaged': int(n_damaged),
                'n_bonds': len(bonds),
                'mean_damage': float(bond_damage.mean()),
                'max_damage': float(bond_damage.max()),
                'n_failed_by_damage': int(n_failed_by_damage),
            }
            print(
                f"  DPDEM bond stats: bonds with damage>0.01: {n_damaged}/{len(bonds)}, "
                f"mean damage: {bond_damage.mean():.4f}, max: {bond_damage.max():.4f}, "
                f"failed by damage (D>=1): {n_failed_by_damage}",
                flush=True,
            )
        else:
            print("  DPDEM bond stats: (no bonds)", flush=True)

    damage_summary = {'impulse_total': J_cumul} if use_ship_dynamics else None
    if damage_summary is not None:
        if hull_impulse_segments:
            damage_summary['hull_impulse_segments'] = [
                (x, Jx, Jy) for x, Jx, Jy in hull_impulse_segments
            ]
        if hull_impulse_contacts is not None and hull_impulse_contacts:
            damage_summary['hull_impulse_contacts'] = hull_impulse_contacts
        if commanded_trajectory:
            damage_summary['commanded_trajectory'] = commanded_trajectory
    elif kinematic_ship and commanded_trajectory:
        damage_summary = {'commanded_trajectory': commanded_trajectory}
    if damage_summary is not None and dpdem_bond_stats is not None:
        damage_summary['dpdem_bond_stats'] = dpdem_bond_stats

    if return_ship_diagnostics:
        if ship_diagnostics is not None and damage_summary is not None:
            ship_diagnostics.append({'damage_summary': damage_summary})
        return positions_history, rotations_history, bonds_history, ship_history, submergence_history, ship_diagnostics
    if damage_summary is not None:
        return positions_history, rotations_history, bonds_history, ship_history, submergence_history, damage_summary
    return positions_history, rotations_history, bonds_history, ship_history, submergence_history
