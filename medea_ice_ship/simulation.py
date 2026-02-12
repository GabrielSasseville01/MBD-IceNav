"""Ice–ship interaction simulation: collision, fracture, ship dynamics."""

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
from medea_ice_ship.config import SimulationConfig, get_default_simulation_config
from medea_ice_ship.vessel_dynamics import get_mss_style_3dof, coriolis_3dof


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

    Returns (positions_history, rotations_history, bonds_history, ship_history, submergence_history[, ship_diagnostics or damage_summary]).
    When ship_trajectory is not None and return_ship_diagnostics is False, the 6th element is damage_summary
    (dict with 'impulse_total': cumulative impulse from ice on ship). When return_ship_diagnostics is True,
    the 6th element is ship_diagnostics (list) and the last entry includes 'damage_summary'.
    """
    cfg = config or get_default_simulation_config()
    n = len(polygons)
    areas = np.array([polygon_area(p) for p in polygons])
    masses = areas * thicknesses * IceProperties.ICE_DENSITY
    radii = np.array([np.sqrt(a / np.pi) for a in areas])

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
    bond_i = np.array([b[0] for b in bonds])
    bond_j = np.array([b[1] for b in bonds])
    for b, (i, j) in enumerate(bonds):
        h = min(thicknesses[i], thicknesses[j])
        w = min((radii[i] + radii[j]) / 2, cfg.max_bond_width)
        bond_strengths[b] = IceProperties.breaking_force(h, w)

    if len(bonds) > 0:
        print(f"  Bond strengths: {bond_strengths.min()/1e6:.2f} - {bond_strengths.max()/1e6:.2f} MN", flush=True)

    # Precompute pair indices for ice-ice contact (upper triangle i < j).
    pair_i = np.repeat(np.arange(n), np.arange(n - 1, -1, -1))
    pair_j = np.concatenate([np.arange(i + 1, n) for i in range(n)])

    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    bonds_history = [bond_intact.copy()]
    ship_history = []
    submergence_history = [submergence.copy()]

    ship_impact_force = (ship_props.impact_force(drift_vel, np.mean(thicknesses))
                         if (drift_vel > 0 and len(thicknesses) > 0) else 0.0)
    print(f"  Ship impact force: {ship_impact_force/1e6:.2f} MN", flush=True)
    collision_count = 0
    ever_collided = set()

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

        components = find_connected_components(n, bonds, bond_intact)
        forces = np.zeros_like(positions)
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
                angular_vels[i] += cfg.angular_impulse_scale * side_direction * min(pen, cfg.angular_impulse_pen_cap)

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
        F_ice_ship += F_ice_ship_collision
        M_ice_ship += M_ice_ship_collision

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

        # Ice-ice contact (vectorized over pairs i < j).
        d_vec = positions[pair_j] - positions[pair_i]
        dist = np.linalg.norm(d_vec, axis=1)
        min_dist = radii[pair_i] + radii[pair_j]
        overlap = np.where((dist < min_dist) & (dist > 1e-6), min_dist - dist, 0.0)
        active = overlap > 0
        if np.any(active):
            direction = np.zeros_like(d_vec)
            direction[active] = d_vec[active] / dist[active, np.newaxis]
            f_n_mag = cfg.hertz_factor * overlap ** cfg.hertz_exponent
            # Normal force: direction points i→j; force on i = -direction*f_n_mag (repel), on j = +direction*f_n_mag.
            f_n_vec = direction * f_n_mag[:, np.newaxis]
            np.add.at(forces, pair_i, -f_n_vec)
            np.add.at(forces, pair_j, f_n_vec)
            v_rel = velocities[pair_j] - velocities[pair_i]
            v_t = v_rel - np.sum(v_rel * direction, axis=1, keepdims=True) * direction
            v_t_mag = np.linalg.norm(v_t, axis=1)
            friction_active = active & (v_t_mag > 1e-6)
            if np.any(friction_active):
                t_hat = np.zeros_like(v_t)
                t_hat[friction_active] = -v_t[friction_active] / v_t_mag[friction_active, np.newaxis]
                f_t_mag = np.minimum(cfg.ice_ice_friction * f_n_mag, cfg.friction_cap)
                f_t_vec = t_hat * f_t_mag[:, np.newaxis]
                np.add.at(forces, pair_i, f_t_vec)
                np.add.at(forces, pair_j, -f_t_vec)

        if ship_trajectory is None:
            for comp in components:
                total_mass = sum(masses[i] for i in comp)
                drift_force = drift_vel * total_mass * cfg.drift_factor
                for i in comp:
                    forces[i, 0] += drift_force * masses[i] / total_mass

        RHO_WATER = IceProperties.WATER_DENSITY
        v_mag = np.linalg.norm(velocities, axis=1, keepdims=True)
        v_mag_safe = np.maximum(v_mag, 1e-12)
        v_mag_1d = v_mag_safe.ravel()
        A_drag = areas * (thicknesses / cfg.drag_ref_thickness)
        # Quadratic when v_mag > 1e-6, else linear (same formulas as before, vectorized).
        drag_force = np.where(
            v_mag > 1e-6,
            (velocities / v_mag_safe) * (0.5 * RHO_WATER * cfg.c_d_drag * np.expand_dims(A_drag, axis=1) * (v_mag_1d ** 2)[:, np.newaxis]),
            0.5 * RHO_WATER * cfg.c_d_drag * np.expand_dims(A_drag, axis=1) * cfg.linear_drag_factor * velocities,
        )
        forces -= drag_force

        new_velocities = velocities.copy()
        new_positions = positions.copy()
        for comp in components:
            if len(comp) == 1:
                i = comp[0]
                accel = forces[i] / masses[i]
                new_velocities[i] += accel * dt
                new_velocities[i] = np.clip(new_velocities[i], cfg.velocity_clip[0], cfg.velocity_clip[1])
                new_positions[i] += new_velocities[i] * dt
            else:
                total_mass = sum(masses[i] for i in comp)
                com = sum(masses[i] * positions[i] for i in comp) / total_mass
                total_force = sum(forces[i] for i in comp)
                comp_vel = sum(masses[i] * velocities[i] for i in comp) / total_mass
                accel = total_force / total_mass
                comp_vel += accel * dt
                comp_vel = np.clip(comp_vel, cfg.velocity_clip[0], cfg.velocity_clip[1])
                for i in comp:
                    new_velocities[i] = comp_vel
                    rel_pos = positions[i] - com
                    new_com = com + comp_vel * dt
                    new_positions[i] = new_com + rel_pos

        velocities = new_velocities
        positions = new_positions

        element_stress = ship_collision_force.copy()
        # Bond stress accumulation (vectorized).
        intact = bond_intact
        d_bond = positions[bond_j] - positions[bond_i]
        dist_bond = np.linalg.norm(d_bond, axis=1)
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
        # Bond breaking (vectorized condition evaluation).
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
        break_collision = intact & collision_on_bond & (collision_force_on_bond > effective_strength)
        break_force = intact & (bond_force > effective_strength)
        bond_intact[break_collision | break_force] = False

        angular_vels += np.random.randn(n) * cfg.angular_noise
        angular_vels *= cfg.angular_damping
        angular_vels = np.clip(angular_vels, cfg.angular_cap[0], cfg.angular_cap[1])
        rotations += angular_vels * dt

        if use_ship_dynamics:
            psi = ship_heading_dyn
            nu = np.array([ship_u, ship_v, ship_r])
            F_ice_body = R_xy(psi).T @ F_ice_ship
            tau_ice = np.array([F_ice_body[0], F_ice_body[1], M_ice_ship])
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
            if getattr(cfg, "use_mss_style_dynamics", False):
                C = coriolis_3dof(M3, nu)
            else:
                C = np.zeros((3, 3))
                C[0, 2] = -m * nu[1]
                C[1, 2] = m * nu[0]
                C[2, 0], C[2, 1] = -C[0, 2], -C[1, 2]
            nu_dot = M3inv @ (tau_control + tau_ice - D3 @ nu - C @ nu)
            nu = nu + nu_dot * dt
            vc = cfg.ship_velocity_clip
            nu[0] = np.clip(nu[0], -vc[0], vc[0])
            nu[1] = np.clip(nu[1], -vc[1], vc[1])
            nu[2] = np.clip(nu[2], -vc[2], vc[2])
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

        positions_history.append(positions.copy())
        rotations_history.append(rotations.copy())
        bonds_history.append(bond_intact.copy())
        submergence_history.append(submergence.copy())

        if frame_callback is not None:
            frame_callback(
                frame,
                positions.copy(),
                rotations.copy(),
                bond_intact.copy(),
                ship_pos_frame.copy(),
                ship_heading,
            )

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

    if return_ship_diagnostics:
        if ship_diagnostics is not None and damage_summary is not None:
            ship_diagnostics.append({'damage_summary': damage_summary})
        return positions_history, rotations_history, bonds_history, ship_history, submergence_history, ship_diagnostics
    if damage_summary is not None:
        return positions_history, rotations_history, bonds_history, ship_history, submergence_history, damage_summary
    return positions_history, rotations_history, bonds_history, ship_history, submergence_history
