"""2D and 3D animation for MEDEA ice–ship fracture simulation."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from medea_ice_ship.geometry import polygon_centroid, create_ship_hull_at_position_heading, point_in_polygon
from medea_ice_ship.config import VisualizationConfig, get_default_visualization_config


def animate_ice_ship_2d(
    polygons, positions_history, rotations_history, bonds_history, bonds,
    ship_history, ship_length, ship_beam, initial_centroids, thicknesses,
    output_file='medea_ship_fracture.gif', fps=20
):
    """Animate ice-ship interaction in 2D. Handles n_elements=0 (clear water: ship only)."""
    n_frames = len(positions_history)
    n_elements = len(polygons)
    local_polys = []
    for poly in polygons:
        centroid = polygon_centroid(poly)
        local_polys.append(poly - centroid)
    if len(ship_history) > 0:
        ship_positions = np.array([h[0] for h in ship_history])
        x_min = ship_positions[:, 0].min() - ship_length - 20
        x_max = ship_positions[:, 0].max() + ship_length + 20
        y_min = ship_positions[:, 1].min() - ship_beam - 20
        y_max = ship_positions[:, 1].max() + ship_beam + 20
    else:
        x_min, x_max, y_min, y_max = -50, 50, -50, 50
    if n_elements > 0:
        all_pos = np.concatenate(positions_history)
        x_min = min(x_min, all_pos[:, 0].min() - 20)
        x_max = max(x_max, all_pos[:, 0].max() + 20)
        y_min = min(y_min, all_pos[:, 1].min() - 20)
        y_max = max(y_max, all_pos[:, 1].max() + 20)
    fig, ax = plt.subplots(figsize=(16, 12))
    water_color = '#1A4D70'
    ship_color = '#404040'
    if n_elements > 0 and len(thicknesses) > 0:
        thickness_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
        ice_colors = [f'#{int(240-100*t):02x}{int(250-80*t):02x}{int(255-40*t):02x}' for t in thickness_norm]
    else:
        ice_colors = []

    def rotate_polygon(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        return poly @ np.array([[c, -s], [s, c]]).T

    print(f"Creating 2D animation with {n_frames} frames...", flush=True)

    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_facecolor(water_color)
        positions = positions_history[frame]
        rotations = rotations_history[frame]
        intact = bonds_history[frame]
        broken_neighbors = {i: set() for i in range(n_elements)}
        for b, (i, j) in enumerate(bonds):
            if not intact[b]:
                broken_neighbors[i].add(j)
                broken_neighbors[j].add(i)
        for i in range(n_elements):
            rotated = rotate_polygon(local_polys[i], rotations[i])
            world_poly = rotated + positions[i]
            has_broken = len(broken_neighbors[i]) > 0
            edge_color = '#1A4D70' if has_broken else ice_colors[i]
            edge_width = 1.5 if has_broken else 0.5
            patch = MplPolygon(world_poly, closed=True, facecolor=ice_colors[i], edgecolor=edge_color,
                              linewidth=edge_width, alpha=0.95)
            ax.add_patch(patch)
        if frame < len(ship_history):
            ship_pos_frame, ship_heading = ship_history[frame]
            ship_outline = create_ship_hull_at_position_heading(
                ship_length, ship_beam, ship_pos_frame, ship_heading
            )
            ship_patch = MplPolygon(ship_outline, closed=True, facecolor=ship_color,
                                   edgecolor='#202020', linewidth=2, alpha=0.95)
            ax.add_patch(ship_patch)
        n_broken = np.sum(~intact)
        ax.set_title(f'DPDEM Ice-Ship Simulation (MEDEA Satellite Floes)\n'
                     f'Frame {frame}/{n_frames-1} | Bonds: {n_broken}/{len(bonds)} broken', fontsize=14)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    print(f"Saving to {output_file}...", flush=True)
    anim.save(output_file, writer=PillowWriter(fps=fps))
    print("Done!", flush=True)
    plt.close(fig)


def simulate_3d_ice_physics(positions_history, bonds_history, bonds, ship_history,
                           ship_length, ship_beam, ship_draft, n_elements,
                           submergence_history=None):
    """Z from submergence when available, else heuristic. Ice under ship clamped to hull bottom."""
    n_frames = len(positions_history)
    z_positions = [np.zeros(n_elements) for _ in range(n_frames)]
    tilts_x = [np.zeros(n_elements) for _ in range(n_frames)]
    tilts_y = [np.zeros(n_elements) for _ in range(n_frames)]
    use_submergence = (submergence_history is not None and len(submergence_history) == n_frames and
                      all(len(s) == n_elements for s in submergence_history))
    if use_submergence:
        z_ship_bottom = -ship_draft
        for frame in range(n_frames):
            z_positions[frame] = -np.array(submergence_history[frame], dtype=float)
            ship_pos, ship_psi = ship_history[min(frame, len(ship_history) - 1)]
            hull_pts = create_ship_hull_at_position_heading(ship_length, ship_beam, ship_pos, ship_psi)
            positions = positions_history[frame]
            for i in range(n_elements):
                if point_in_polygon((positions[i, 0], positions[i, 1]), hull_pts):
                    z_positions[frame][i] = min(z_positions[frame][i], z_ship_bottom)
        return z_positions, tilts_x, tilts_y
    for frame in range(1, n_frames):
        positions = positions_history[frame]
        intact = bonds_history[frame]
        ship_pos, ship_psi = ship_history[min(frame, len(ship_history) - 1)]
        ship_x, ship_y = ship_pos[0], ship_pos[1]
        prev_z = z_positions[frame - 1].copy()
        prev_tilt_x = tilts_x[frame - 1].copy()
        broken_count = np.zeros(n_elements)
        for b, (i, j) in enumerate(bonds):
            if not intact[b]:
                broken_count[i] += 1
                broken_count[j] += 1
        new_z = prev_z.copy()
        new_tilt_x = prev_tilt_x.copy()
        new_tilt_y = tilts_y[frame - 1].copy()
        for i in range(n_elements):
            px, py = positions[i]
            dx, dy = px - ship_x, py - ship_y
            c, s = np.cos(-ship_psi), np.sin(-ship_psi)
            local_x = c * dx - s * dy
            local_y = s * dx + c * dy
            dist_from_bow = ship_length / 2 - local_x
            if 0 < dist_from_bow < ship_length * 0.4 and abs(local_y) < ship_beam:
                push_factor = 1 - dist_from_bow / (ship_length * 0.4)
                if broken_count[i] > 0:
                    target_z = -ship_draft - push_factor * 2.0
                    new_z[i] += (target_z - new_z[i]) * 0.15
                    new_tilt_x[i] += (0.4 * push_factor - new_tilt_x[i]) * 0.1
                else:
                    new_tilt_x[i] += (0.2 * push_factor - new_tilt_x[i]) * 0.05
            elif dist_from_bow < 0:
                if broken_count[i] > 0 and abs(local_y) < ship_beam * 0.8:
                    new_z[i] += (-ship_draft * 0.9 - new_z[i]) * 0.05
                elif new_z[i] < -0.5:
                    new_z[i] += (0 - new_z[i]) * 0.03
            new_tilt_x[i] *= 0.95
            new_tilt_y[i] *= 0.95
            dist_to_ship = np.sqrt((px - ship_x)**2 + (py - ship_y)**2)
            if new_z[i] < -0.3 and dist_to_ship > ship_length:
                new_z[i] += 0.02
        hull_pts = create_ship_hull_at_position_heading(ship_length, ship_beam, ship_pos, ship_psi)
        for i in range(n_elements):
            if point_in_polygon((positions[i, 0], positions[i, 1]), hull_pts):
                new_z[i] = min(new_z[i], -ship_draft)
        z_positions[frame] = new_z
        tilts_x[frame] = new_tilt_x
        tilts_y[frame] = new_tilt_y
    return z_positions, tilts_x, tilts_y


def animate_ice_ship_3d(
    polygons, positions_history, rotations_history, bonds_history, bonds,
    ship_history, ship_length, ship_beam, ship_draft,
    initial_centroids, thicknesses,
    output_file='medea_ship_fracture_3d.gif', fps=15, frame_skip=2,
    submergence_history=None,
    viz_config=None,
):
    """3D animation with z from submergence; color by submerged (blue) vs surfaced (light)."""
    vc = viz_config or get_default_visualization_config()
    n_frames = len(positions_history)
    n_elements = len(polygons)
    local_polys = [poly - polygon_centroid(poly) for poly in polygons]
    all_pos = np.concatenate(positions_history)
    x_min, x_max = all_pos[:, 0].min() - 50, all_pos[:, 0].max() + 50
    y_min, y_max = all_pos[:, 1].min() - 50, all_pos[:, 1].max() + 50
    domain_width = x_max - x_min
    domain_height = y_max - y_min
    print(f"  Computing 3D ice dynamics (bending, submerging)...", flush=True)
    z_positions, tilts_x, tilts_y = simulate_3d_ice_physics(
        positions_history, bonds_history, bonds, ship_history,
        ship_length, ship_beam, ship_draft, n_elements,
        submergence_history=submergence_history,
    )
    color_by_submergence = submergence_history is not None
    SURFACED_FACE, SURFACED_SIDE = vc.surfaced_face, vc.surfaced_side
    SUBMERGED_FACE, SUBMERGED_SIDE = vc.submerged_face, vc.submerged_side
    Z_SURFACE_THRESHOLD = vc.z_surface_threshold
    thickness_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
    ice_colors = [f'#{int(230-80*t):02x}{int(245-60*t):02x}{int(255-30*t):02x}' for t in thickness_norm]
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    frames_to_render = list(range(0, n_frames, frame_skip))
    if frames_to_render[-1] != n_frames - 1:
        frames_to_render.append(n_frames - 1)
    print(f"  Creating 3D animation with {len(frames_to_render)} frames...", flush=True)

    def rotate_poly(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        return poly @ np.array([[c, -s], [s, c]]).T

    def update(frame_idx):
        frame = frames_to_render[frame_idx]
        ax.clear()
        ship_pos, ship_psi = ship_history[min(frame, len(ship_history) - 1)]
        ship_x, ship_y = ship_pos[0], ship_pos[1]
        positions = positions_history[frame]
        rotations = rotations_history[frame]
        intact = bonds_history[frame]
        z_pos = z_positions[frame]
        tilt_x, tilt_y = tilts_x[frame], tilts_y[frame]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(-ship_draft - 3, 5)
        ax.set_box_aspect([domain_width, domain_height, (ship_draft + 8) * 2])
        ax.view_init(elev=vc.view_elev, azim=vc.view_azim_start + frame * vc.view_azim_per_frame)
        for i in range(n_elements):
            rotated = rotate_poly(local_polys[i], rotations[i])
            world_poly = rotated + positions[i]
            z_center, tx, ty, h = z_pos[i], tilt_x[i], tilt_y[i], thicknesses[i]
            if color_by_submergence:
                submerged = z_center < -Z_SURFACE_THRESHOLD
                face_color = SUBMERGED_FACE if submerged else SURFACED_FACE
                side_color = SUBMERGED_SIDE if submerged else SURFACED_SIDE
            else:
                face_color, side_color = ice_colors[i], '#C8E0F0'
            center_xy = positions[i]
            top_verts = []
            bottom_verts = []
            for v in world_poly:
                dx = v[0] - center_xy[0]
                dy = v[1] - center_xy[1]
                z_offset = dx * np.sin(tx) + dy * np.sin(ty)
                top_verts.append([v[0], v[1], z_center + h * 0.1 + z_offset])
                bottom_verts.append([v[0], v[1], z_center - h * 0.9 + z_offset])
            top_verts = np.array(top_verts)
            bottom_verts = np.array(bottom_verts)
            has_breaks = any(not intact[b] for b, (pi, pj) in enumerate(bonds) if pi == i or pj == i)
            edge_color = '#2080A0' if has_breaks else '#A0C8D8'
            ax.add_collection3d(Poly3DCollection(
                [top_verts], facecolor=face_color, edgecolor=edge_color,
                linewidth=0.5 if has_breaks else 0.2, alpha=0.9
            ))
            n_verts = len(world_poly)
            for j in range(n_verts):
                k = (j + 1) % n_verts
                side = np.array([bottom_verts[j], bottom_verts[k], top_verts[k], top_verts[j]])
                ax.add_collection3d(Poly3DCollection(
                    [side], facecolor=side_color, edgecolor=edge_color, linewidth=0.1, alpha=0.8
                ))
        c, s = np.cos(ship_psi), np.sin(ship_psi)
        half_l, half_b = ship_length / 2, ship_beam / 2
        deck_height = 1.5
        n_sections = 10
        hull_verts = []
        for i in range(n_sections + 1):
            t = i / n_sections
            x_local = half_l - t * ship_length
            x_rel = x_local / half_l
            taper = max(1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5 if x_rel > 0.6 else 1.0, 0.05)
            local_beam = ship_beam * taper
            local_draft = ship_draft * taper
            local_deck = deck_height * taper
            hull_verts.extend([
                [x_local, -local_beam/2, local_deck], [x_local, local_beam/2, local_deck],
                [x_local, -local_beam/2*0.9, 0], [x_local, local_beam/2*0.9, 0],
                [x_local, 0, -local_draft],
            ])
        hull_verts = np.array(hull_verts)
        R_3d = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        hull_verts_world = hull_verts @ R_3d.T + np.array([ship_x, ship_y, 0])
        verts_per_section = 5
        deck_color, hull_above, hull_below = '#8B4513', '#C0C0C0', '#8B0000'
        for i in range(n_sections):
            base = i * verts_per_section
            next_base = (i + 1) * verts_per_section
            deck_face = [hull_verts_world[base], hull_verts_world[next_base],
                         hull_verts_world[next_base + 1], hull_verts_world[base + 1]]
            ax.add_collection3d(Poly3DCollection(
                [deck_face], facecolor=deck_color, edgecolor='#5D3A1A', linewidth=0.5, alpha=0.95
            ))
            port_top = [hull_verts_world[base], hull_verts_world[base+2], hull_verts_world[next_base+2], hull_verts_world[next_base]]
            ax.add_collection3d(Poly3DCollection([port_top], facecolor=hull_above, edgecolor='#808080', linewidth=0.3, alpha=0.95))
            starboard_top = [hull_verts_world[base+1], hull_verts_world[next_base+1], hull_verts_world[next_base+3], hull_verts_world[base+3]]
            ax.add_collection3d(Poly3DCollection([starboard_top], facecolor=hull_above, edgecolor='#808080', linewidth=0.3, alpha=0.95))
            port_bot = [hull_verts_world[base+2], hull_verts_world[base+4], hull_verts_world[next_base+4], hull_verts_world[next_base+2]]
            ax.add_collection3d(Poly3DCollection([port_bot], facecolor=hull_below, edgecolor='#5C0000', linewidth=0.3, alpha=0.95))
            starboard_bot = [hull_verts_world[base+3], hull_verts_world[next_base+3], hull_verts_world[next_base+4], hull_verts_world[base+4]]
            ax.add_collection3d(Poly3DCollection([starboard_bot], facecolor=hull_below, edgecolor='#5C0000', linewidth=0.3, alpha=0.95))
        n_broken = np.sum(~intact)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        title = f'DPDEM 3D Ice-Ship Simulation (MEDEA)\nFrame {frame}/{n_frames-1} | Bonds: {n_broken}/{len(bonds)} broken'
        if color_by_submergence:
            title += '\nBlue = submerged | Light = at surface'
        ax.set_title(title)
        return []

    anim = FuncAnimation(fig, update, frames=len(frames_to_render), blit=False)
    print(f"  Saving to {output_file}...", flush=True)
    anim.save(output_file, writer=PillowWriter(fps=fps))
    print("  Done!", flush=True)
    plt.close(fig)
