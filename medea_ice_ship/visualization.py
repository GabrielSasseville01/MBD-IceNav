"""2D and 3D animation for MEDEA ice–ship fracture simulation."""

import multiprocessing
import os
import pickle
import shutil
import tempfile
import warnings
import numpy as np
import matplotlib
if os.environ.get('DPDEM_LIVE_VIEW') != '1':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib import colors as mpl_colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from medea_ice_ship.geometry import (
    polygon_centroid,
    create_ship_hull_at_position_heading,
    point_in_polygon,
    closest_point_and_outward_normal,
)
from medea_ice_ship.config import VisualizationConfig, get_default_visualization_config


def make_live_2d_callback(
    bonds,
    local_polys,
    ship_length,
    ship_beam,
    thicknesses,
    extent,
    n_frames,
    figsize=(12, 8),
):
    """
    Create (fig, ax, callback) for real-time 2D view during simulation.

    Use with simulate_ice_ship_interaction(..., frame_callback=callback).
    extent: (x_min, x_max, y_min, y_max). After the sim, call plt.show() to keep the window open.
    """
    x_min, x_max, y_min, y_max = extent
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    water_color = "#1A4D70"
    ship_color = "#404040"
    n_elements = len(local_polys)
    if n_elements > 0 and thicknesses is not None and len(thicknesses) == n_elements:
        t = thicknesses
        thickness_norm = (t - t.min()) / (t.max() - t.min() + 0.01)
        ice_colors = [
            f"#{int(240 - 100 * v):02x}{int(250 - 80 * v):02x}{int(255 - 40 * v):02x}"
            for v in thickness_norm
        ]
    else:
        ice_colors = ["#C0E0F0"] * max(n_elements, 1)

    def callback(frame, positions, rotations, bond_intact, ship_pos, ship_heading):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_facecolor(water_color)
        broken_neighbors = {i: set() for i in range(n_elements)}
        for b, (i, j) in enumerate(bonds):
            if not bond_intact[b]:
                broken_neighbors[i].add(j)
                broken_neighbors[j].add(i)
        for i in range(n_elements):
            c, s = np.cos(rotations[i]), np.sin(rotations[i])
            world_poly = local_polys[i] @ np.array([[c, -s], [s, c]]).T + positions[i]
            has_broken = len(broken_neighbors[i]) > 0
            edge_color = "#1A4D70" if has_broken else ice_colors[i]
            edge_width = 1.5 if has_broken else 0.5
            patch = MplPolygon(
                world_poly,
                closed=True,
                facecolor=ice_colors[i],
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=0.95,
            )
            ax.add_patch(patch)
        ship_outline = create_ship_hull_at_position_heading(
            ship_length, ship_beam, ship_pos, ship_heading
        )
        ship_patch = MplPolygon(
            ship_outline,
            closed=True,
            facecolor=ship_color,
            edgecolor="#202020",
            linewidth=2,
            alpha=0.95,
        )
        ax.add_patch(ship_patch)
        n_broken = np.sum(~bond_intact)
        ax.set_title(
            f"DPDEM Ice-Ship (live)\nFrame {frame}/{n_frames - 1} | Bonds: {n_broken}/{len(bonds)} broken",
            fontsize=12,
        )
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

    return fig, ax, callback


def plot_hull_impact_vectors(
    ship_length: float,
    ship_beam: float,
    hull_impulse_segments: list = None,
    hull_impulse_contacts: list = None,
    output_file: str = "hull_impact_vectors.png",
    figsize: tuple = (10, 6),
    max_arrow_size: float = None,
):
    """
    Plot impulse vectors at impact locations on the ship hull (AUTO-IceNav style).

    When hull_impulse_contacts is provided: one line segment per contact (like
    AUTO-IceNav impact_locations_impulse.pdf). Line length is normalized so no
    segment exceeds max_arrow_size; color encodes impulse magnitude (plasma).
    Otherwise uses segment-aggregated data with quiver arrows.

    Args:
        ship_length: Ship length [m]
        ship_beam: Ship beam [m]
        hull_impulse_segments: Optional list of (x_body, Jx, Jy) per segment (fallback)
        hull_impulse_contacts: Optional list of (cx_body, cy_body, Jx, Jy) per contact
        output_file: Output path (PNG)
        figsize: Figure size (inches)
        max_arrow_size: Max line length in plot units (default 0.12 * ship_length)
    """
    contacts = hull_impulse_contacts if hull_impulse_contacts else []
    segments = hull_impulse_segments if hull_impulse_segments else []
    if not contacts and not segments:
        return
    half_beam = ship_beam / 2
    half_len = ship_length / 2
    hull_outline = create_ship_hull_at_position_heading(
        ship_length, ship_beam, (0.0, 0.0), 0.0
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    # Hull: AUTO-IceNav uses fill=False, alpha=0.1; we keep a light fill for visibility
    ax.add_patch(
        MplPolygon(
            hull_outline,
            fill=True,
            facecolor="#404040",
            edgecolor="#202020",
            linewidth=2,
            alpha=0.9,
        )
    )
    use_contacts = len(contacts) > 0
    if use_contacts:
        # AUTO-IceNav style: one line per contact, length capped, color = magnitude
        if max_arrow_size is None:
            max_arrow_size = 0.24 * ship_length
        max_impulse = max(
            np.sqrt(c[2] ** 2 + c[3] ** 2) for c in contacts
        ) or 1.0
        cmap = plt.get_cmap("plasma")
        # Optional approach-speed scaling (plot-only): 5-tuple (cx, cy, Jx, Jy, approach_speed)
        has_approach = contacts and len(contacts[0]) >= 5
        max_approach = (
            max(c[4] for c in contacts if len(c) >= 5) if has_approach else 1.0
        ) or 1.0
        segs = []
        color_list = []
        for c in contacts:
            cx, cy, Jx, Jy = c[0], c[1], c[2], c[3]
            approach_speed = c[4] if len(c) >= 5 else None
            norm_impulse = np.sqrt(Jx * Jx + Jy * Jy)
            if norm_impulse < 1e-12:
                continue
            # Project onto hull and get outward normal (AUTO-IceNav: segments perpendicular to hull, outside ship)
            hx, hy, nx, ny = closest_point_and_outward_normal(cx, cy, hull_outline)
            # Segment: head on hull, tail outside along outward normal; length = magnitude
            length = min(norm_impulse / max_impulse, 1.0) * max_arrow_size
            # Scale length by approach speed (plot-only): bow (approaching) larger, stern/wake (receding) smaller
            if approach_speed is not None and max_approach > 1e-6:
                length *= max(0.2, approach_speed / max_approach)
            tail_x = hx + nx * length
            tail_y = hy + ny * length
            segs.append([(tail_x, tail_y), (hx, hy)])
            color_list.append(cmap(min(norm_impulse / max_impulse, 1.0)))
        if segs:
            lc = LineCollection(
                segs,
                colors=color_list,
                linestyles="-",
                linewidths=1,
                alpha=0.7,
                rasterized=True,
            )
            ax.add_collection(lc)
            ax.set_xlim(np.min(hull_outline[:, 0]) - max_arrow_size, np.max(hull_outline[:, 0]) + max_arrow_size)
            ax.set_ylim(np.min(hull_outline[:, 1]) - max_arrow_size, np.max(hull_outline[:, 1]) + max_arrow_size)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_impulse))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.95)
            cbar.set_label("Impulse magnitude (N·s)", fontsize=10)
    else:
        # Segment-aggregated fallback: quiver
        xs = np.array([s[0] for s in segments])
        jxs = np.array([s[1] for s in segments])
        jys = np.array([s[2] for s in segments])
        eps = 1e-9
        y_contact = np.where(jys > eps, -half_beam, np.where(jys < -eps, half_beam, 0.0))
        head_x, head_y = xs, y_contact
        j_mag = np.sqrt(jxs**2 + jys**2)
        if j_mag.size > 0 and j_mag.max() > 1e-3:
            scale = 0.3 * ship_length / j_mag.max()
            tail_x = head_x - jxs * scale
            tail_y = head_y - jys * scale
            q = ax.quiver(
                tail_x, tail_y, jxs * scale, jys * scale,
                j_mag, cmap="hot", scale=1.0, scale_units="xy",
                width=0.008, alpha=0.9,
            )
            cbar = fig.colorbar(q, ax=ax, shrink=0.8)
            cbar.set_label("Accumulated impulse [N·s]", fontsize=10)
    ax.set_xlabel("x (body frame) [m]", fontsize=11)
    ax.set_ylabel("y (body frame) [m]", fontsize=11)
    ax.set_title("Impulse vectors at impact locations (AUTO-IceNav style)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Process-global data for parallel workers (set by pool initializer)
_DATA_2D = None
_DATA_3D = None
_DATA_COSTMAP = None
_RENDER_DPI = 100


def _get_ffmpeg_path():
    """Return path to ffmpeg: bundled (imageio-ffmpeg) first, then system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _draw_2d_frame_to_array(data, frame_idx):
    """Draw a single 2D frame; return RGB array (H, W, 3) uint8. Used by parallel workers."""
    frame = data["frames_to_render"][frame_idx]
    n_elements = data["n_elements"]
    positions = data["positions_history"][frame]
    rotations = data["rotations_history"][frame]
    intact = data["bonds_history"][frame]
    bonds = data["bonds"]
    local_polys = data["local_polys"]
    ice_colors = data["ice_colors"]
    broken_neighbors = {i: set() for i in range(n_elements)}
    for b, (i, j) in enumerate(bonds):
        if not intact[b]:
            broken_neighbors[i].add(j)
            broken_neighbors[j].add(i)

    fig, ax = plt.subplots(figsize=data["figsize"], dpi=data["dpi"])
    ax.set_xlim(data["x_min"], data["x_max"])
    ax.set_ylim(data["y_min"], data["y_max"])
    ax.set_aspect("equal")
    ax.set_facecolor(data["water_color"])
    for i in range(n_elements):
        c, s = np.cos(rotations[i]), np.sin(rotations[i])
        rotated = local_polys[i] @ np.array([[c, -s], [s, c]]).T
        world_poly = rotated + positions[i]
        has_broken = len(broken_neighbors[i]) > 0
        edge_color = "#1A4D70" if has_broken else ice_colors[i]
        edge_width = 1.5 if has_broken else 0.5
        patch = MplPolygon(world_poly, closed=True, facecolor=ice_colors[i], edgecolor=edge_color, linewidth=edge_width, alpha=0.95)
        ax.add_patch(patch)
    if frame < len(data["ship_history"]):
        ship_pos_frame, ship_heading = data["ship_history"][frame]
        ship_outline = create_ship_hull_at_position_heading(data["ship_length"], data["ship_beam"], ship_pos_frame, ship_heading)
        ship_patch = MplPolygon(ship_outline, closed=True, facecolor=data["ship_color"], edgecolor="#202020", linewidth=2, alpha=0.95)
        ax.add_patch(ship_patch)
    n_broken = np.sum(~intact)
    ax.set_title(f"DPDEM Ice-Ship Simulation (MEDEA Satellite Floes)\nFrame {frame}/{data['n_frames']-1} | Bonds: {n_broken}/{len(bonds)} broken", fontsize=14)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    fig.canvas.draw()
    # buffer_rgba() is (height, width, 4); take RGB for imageio (tostring_rgb removed in newer matplotlib).
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _init_worker_2d(data_path):
    global _DATA_2D
    with open(data_path, "rb") as f:
        _DATA_2D = pickle.load(f)


def _worker_2d(frame_idx):
    return (frame_idx, _draw_2d_frame_to_array(_DATA_2D, frame_idx))


def _draw_3d_frame_to_array(data, frame_idx):
    """Draw a single 3D frame; return RGB array (H, W, 3) uint8. Used by parallel workers."""
    frame = data["frames_to_render"][frame_idx]
    n_elements = data["n_elements"]
    ship_pos, ship_psi = data["ship_history"][min(frame, len(data["ship_history"]) - 1)]
    ship_x, ship_y = ship_pos[0], ship_pos[1]
    positions = data["positions_history"][frame]
    rotations = data["rotations_history"][frame]
    intact = data["bonds_history"][frame]
    z_pos = data["z_positions"][frame]
    tilt_x = data["tilts_x"][frame]
    tilt_y = data["tilts_y"][frame]
    bonds = data["bonds"]
    thicknesses = data["thicknesses"]
    local_polys = data["local_polys"]
    ship_draft = data["ship_draft"]
    ship_length = data["ship_length"]
    ship_beam = data["ship_beam"]

    fig = plt.figure(figsize=data["figsize"], dpi=data["dpi"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(data["x_min"], data["x_max"])
    ax.set_ylim(data["y_min"], data["y_max"])
    ax.set_zlim(-ship_draft - 3, 5)
    ax.set_box_aspect([data["domain_width"], data["domain_height"], (ship_draft + 8) * 2])
    ax.view_init(elev=data["view_elev"], azim=data["view_azim_start"] + frame * data["view_azim_per_frame"])

    def rotate_poly(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        return poly @ np.array([[c, -s], [s, c]]).T

    for i in range(n_elements):
        rotated = rotate_poly(local_polys[i], rotations[i])
        world_poly = rotated + positions[i]
        z_center, tx, ty, h = z_pos[i], tilt_x[i], tilt_y[i], thicknesses[i]
        if data["color_by_submergence"]:
            submerged = z_center < -data["Z_SURFACE_THRESHOLD"]
            face_color = data["SUBMERGED_FACE"] if submerged else data["SURFACED_FACE"]
            side_color = data["SUBMERGED_SIDE"] if submerged else data["SURFACED_SIDE"]
        else:
            face_color, side_color = data["ice_colors"][i], "#C8E0F0"
        center_xy = positions[i]
        top_verts = []
        bottom_verts = []
        for v in world_poly:
            dx, dy = v[0] - center_xy[0], v[1] - center_xy[1]
            z_offset = dx * np.sin(tx) + dy * np.sin(ty)
            top_verts.append([v[0], v[1], z_center + h * 0.1 + z_offset])
            bottom_verts.append([v[0], v[1], z_center - h * 0.9 + z_offset])
        top_verts = np.array(top_verts)
        bottom_verts = np.array(bottom_verts)
        has_breaks = any(not intact[b] for b, (pi, pj) in enumerate(bonds) if pi == i or pj == i)
        edge_color = "#2080A0" if has_breaks else "#A0C8D8"
        ax.add_collection3d(Poly3DCollection([top_verts], facecolor=face_color, edgecolor=edge_color, linewidth=0.5 if has_breaks else 0.2, alpha=0.9))
        n_verts = len(world_poly)
        for j in range(n_verts):
            k = (j + 1) % n_verts
            side = np.array([bottom_verts[j], bottom_verts[k], top_verts[k], top_verts[j]])
            ax.add_collection3d(Poly3DCollection([side], facecolor=side_color, edgecolor=edge_color, linewidth=0.1, alpha=0.8))

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
    deck_color, hull_above, hull_below = "#8B4513", "#C0C0C0", "#8B0000"
    for i in range(n_sections):
        base = i * verts_per_section
        next_base = (i + 1) * verts_per_section
        deck_face = np.array([hull_verts_world[base], hull_verts_world[next_base], hull_verts_world[next_base + 1], hull_verts_world[base + 1]])
        ax.add_collection3d(Poly3DCollection([deck_face], facecolor=deck_color, edgecolor="#5D3A1A", linewidth=0.5, alpha=0.95))
        port_top = np.array([hull_verts_world[base], hull_verts_world[base+2], hull_verts_world[next_base+2], hull_verts_world[next_base]])
        ax.add_collection3d(Poly3DCollection([port_top], facecolor=hull_above, edgecolor="#808080", linewidth=0.3, alpha=0.95))
        starboard_top = np.array([hull_verts_world[base+1], hull_verts_world[next_base+1], hull_verts_world[next_base+3], hull_verts_world[base+3]])
        ax.add_collection3d(Poly3DCollection([starboard_top], facecolor=hull_above, edgecolor="#808080", linewidth=0.3, alpha=0.95))
        port_bot = np.array([hull_verts_world[base+2], hull_verts_world[base+4], hull_verts_world[next_base+4], hull_verts_world[next_base+2]])
        ax.add_collection3d(Poly3DCollection([port_bot], facecolor=hull_below, edgecolor="#5C0000", linewidth=0.3, alpha=0.95))
        starboard_bot = np.array([hull_verts_world[base+3], hull_verts_world[next_base+3], hull_verts_world[next_base+4], hull_verts_world[base+4]])
        ax.add_collection3d(Poly3DCollection([starboard_bot], facecolor=hull_below, edgecolor="#5C0000", linewidth=0.3, alpha=0.95))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    n_broken = np.sum(~intact)
    title = f"DPDEM 3D Ice-Ship Simulation (MEDEA)\nFrame {frame}/{data['n_frames']-1} | Bonds: {n_broken}/{len(bonds)} broken"
    if data["color_by_submergence"]:
        title += "\nBlue = submerged | Light = at surface"
    ax.set_title(title)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _init_worker_3d(data_path):
    global _DATA_3D
    with open(data_path, "rb") as f:
        _DATA_3D = pickle.load(f)


def _worker_3d(frame_idx):
    return (frame_idx, _draw_3d_frame_to_array(_DATA_3D, frame_idx))


def _animation_writer(output_file, fps):
    """Use FFMpegWriter for .mp4 (faster, smaller), PillowWriter for .gif. Uses bundled ffmpeg (imageio-ffmpeg) or system ffmpeg; falls back to GIF if neither available."""
    out = str(output_file)
    if out.lower().endswith(".mp4"):
        ffmpeg_path = _get_ffmpeg_path()
        if ffmpeg_path is None:
            output_file = out[:-4] + ".gif"
            print("  ffmpeg not found; saving as GIF instead of MP4 (install imageio-ffmpeg or system ffmpeg for MP4).", flush=True)
            return PillowWriter(fps=fps), output_file
        import matplotlib
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        # Faster encoding (slightly larger file); use 'ultrafast' for max speed.
        return FFMpegWriter(fps=fps, extra_args=["-preset", "veryfast"]), output_file


def _draw_costmap_frame_to_array(data, frame_idx):
    """Draw one frame of the costmap movie (AUTO-IceNav style): costmap + ice + planned path (red) + actual path (blue) + ship. Used by parallel workers."""
    k = data["frames_to_render"][frame_idx]
    ship_history = data["ship_history"]
    path = data.get("path")
    ship_length = data["ship_length"]
    ship_beam = data["ship_beam"]
    x_min_ax, x_max_ax = data["x_min_ax"], data["x_max_ax"]
    y_min_ax, y_max_ax = data["y_min_ax"], data["y_max_ax"]

    # Costmap: use snapshot for frame k if provided (AUTO-IceNav: costmap updated every replan, not every frame)
    snapshots = data.get("costmap_snapshots")
    if snapshots:
        cm = None
        for (fidx, cminst) in snapshots:
            if fidx <= k:
                cm = cminst
        if cm is None:
            cm = snapshots[0][1]
        cost_grid = np.asarray(cm.cost_map)
        x_origin = getattr(cm, "x_origin", 0)
        y_origin = getattr(cm, "y_origin", 0)
        resolution = getattr(cm, "resolution", 2.0)
        cols, rows = cost_grid.shape[1], cost_grid.shape[0]
    else:
        cost_grid = data["cost_grid"]
        x_origin = data["x_origin"]
        y_origin = data["y_origin"]
        resolution = data["resolution"]
        cols, rows = data["cols"], data["rows"]

    # Ice: either moving (per-frame positions/rotations) or static polygons
    positions_history = data.get("positions_history")
    rotations_history = data.get("rotations_history")
    local_polygons = data.get("local_polygons")
    static_polygons = data.get("polygons") or []

    X = x_origin + np.arange(cols + 1) * resolution
    Y = y_origin + np.arange(rows + 1) * resolution
    cost = np.asarray(cost_grid)
    norm = mpl_colors.LogNorm(vmin=max(0.5, float(cost.min())), vmax=float(cost.max()))

    fig, ax = plt.subplots(figsize=data["figsize"], dpi=data["dpi"])
    ax.pcolormesh(X, Y, cost, norm=norm, cmap="viridis", shading="flat")

    if positions_history is not None and rotations_history is not None and local_polygons is not None and k < len(positions_history):
        # Draw ice in current frame (moved/rotated by simulation)
        pos = positions_history[k]
        rot = rotations_history[k]
        for i, local_poly in enumerate(local_polygons):
            c, s = np.cos(rot[i]), np.sin(rot[i])
            R = np.array([[c, -s], [s, c]])
            world_poly = np.asarray(local_poly, dtype=float) @ R.T + np.asarray(pos[i], dtype=float)
            closed = np.vstack([world_poly, world_poly[0:1]])
            ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=0.4, alpha=0.9)
    else:
        for poly in static_polygons:
            closed = np.vstack([poly, poly[0:1]])
            ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=0.4, alpha=0.9)
    # AUTO-IceNav colors: planned path = red, actual path = blue
    if path is not None and len(path) > 0:
        ax.plot(path[:, 0], path[:, 1], "r-", linewidth=1.5, alpha=0.9, label="Planned path")
    trail_end = min(k + 1, len(ship_history))
    if trail_end >= 2:
        trail_x = [ship_history[i][0][0] for i in range(trail_end)]
        trail_y = [ship_history[i][0][1] for i in range(trail_end)]
        ax.plot(trail_x, trail_y, "b-", linewidth=2, label="Actual path", zorder=5)
    if k < len(ship_history):
        ship_pos, ship_heading = ship_history[k][0], ship_history[k][1]
        ax.plot(ship_pos[0], ship_pos[1], "ko", markersize=6, zorder=6)
        hull = create_ship_hull_at_position_heading(ship_length, ship_beam, ship_pos, ship_heading)
        patch = MplPolygon(hull, closed=True, facecolor="#404040", edgecolor="black", linewidth=1.5, alpha=0.9, zorder=6)
        ax.add_patch(patch)
    if len(ship_history) > 0:
        start_pos = ship_history[0][0]
        ax.plot(start_pos[0], start_pos[1], "go", markersize=10, label="Start", zorder=7)
    if path is not None and len(path) > 0:
        ax.plot(path[-1, 0], path[-1, 1], "gs", markersize=10, label="Goal", zorder=7)
    ax.set_xlim(x_min_ax, x_max_ax)
    ax.set_ylim(y_min_ax, y_max_ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Costmap + ship trajectory (AUTO-IceNav style)\nFrame {k}/{data['n_frames']-1}")
    ax.grid(True, alpha=0.3)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _init_worker_costmap(data_path):
    global _DATA_COSTMAP
    with open(data_path, "rb") as f:
        _DATA_COSTMAP = pickle.load(f)


def _worker_costmap(frame_idx):
    return (frame_idx, _draw_costmap_frame_to_array(_DATA_COSTMAP, frame_idx))


def animate_costmap_movie(
    cost_map,
    bounds,
    ship_history,
    path=None,
    polygons=None,
    positions_history=None,
    rotations_history=None,
    local_polygons=None,
    costmap_snapshots=None,
    ship_length=158.0,
    ship_beam=28.0,
    output_file="costmap_movie.mp4",
    fps=20,
    frame_skip=1,
    n_workers=1,
    figsize=(10.08, 6.08),
):
    """
    Generate a costmap movie (AUTO-IceNav style): costmap as background, moving ice,
    planned path (red), actual path (blue), ship and start/goal. The costmap can be
    updated every X frames (e.g. at replan times) via costmap_snapshots.

    cost_map: default costmap (used for all frames if costmap_snapshots is None)
    bounds: dict with x_min, x_max, y_min, y_max, x_origin, y_origin, etc.
    ship_history: list of (position_2d, heading) per frame
    path: Nx3 planned path [x,y,heading] or None
    polygons: list of polygon arrays for static ice outline, or None (ignored if moving ice provided)
    positions_history, rotations_history, local_polygons: per-frame ice state for moving ice
    costmap_snapshots: optional list of (frame_idx, cost_map) so the costmap is updated only at
        those frames (e.g. at replan times), not every frame. Each cost_map must have .cost_map,
        .x_origin, .y_origin, .resolution, .cols, .rows. Like AUTO-IceNav planner_plots.
    """
    n_frames = len(ship_history)
    if n_frames == 0:
        print("animate_costmap_movie: ship_history is empty, skipping.", flush=True)
        return
    frames_to_render = list(range(0, n_frames, frame_skip))
    if frames_to_render[-1] != n_frames - 1:
        frames_to_render.append(n_frames - 1)
    n_render = len(frames_to_render)
    resolution = getattr(cost_map, "resolution", 2.0)
    x_origin = getattr(cost_map, "x_origin", bounds.get("x_origin", 0))
    y_origin = getattr(cost_map, "y_origin", bounds.get("y_origin", 0))
    cols = getattr(cost_map, "cols", cost_map.cost_map.shape[1])
    rows = getattr(cost_map, "rows", cost_map.cost_map.shape[0])
    ship_positions = np.array([h[0] for h in ship_history])
    x_min_ax = float(ship_positions[:, 0].min() - ship_length - 20)
    x_max_ax = float(ship_positions[:, 0].max() + ship_length + 20)
    y_min_ax = float(ship_positions[:, 1].min() - ship_beam - 20)
    y_max_ax = float(ship_positions[:, 1].max() + ship_beam + 20)
    if path is not None and len(path) > 0:
        x_min_ax = min(x_min_ax, float(path[:, 0].min()) - 20)
        x_max_ax = max(x_max_ax, float(path[:, 0].max()) + 20)
        y_min_ax = min(y_min_ax, float(path[:, 1].min()) - 20)
        y_max_ax = max(y_max_ax, float(path[:, 1].max()) + 20)
    # Same view extent as 2D simulation when ice history is provided
    if positions_history is not None and len(positions_history) > 0:
        all_pos = np.concatenate(positions_history)
        x_min_ax = min(x_min_ax, float(all_pos[:, 0].min()) - 20)
        x_max_ax = max(x_max_ax, float(all_pos[:, 0].max()) + 20)
        y_min_ax = min(y_min_ax, float(all_pos[:, 1].min()) - 20)
        y_max_ax = max(y_max_ax, float(all_pos[:, 1].max()) + 20)

    data = {
        "cost_grid": cost_map.cost_map,
        "x_origin": x_origin,
        "y_origin": y_origin,
        "resolution": resolution,
        "cols": cols,
        "rows": rows,
        "ship_history": ship_history,
        "path": np.asarray(path) if path is not None else None,
        "polygons": list(polygons) if polygons is not None else [],
        "positions_history": positions_history,
        "rotations_history": rotations_history,
        "local_polygons": list(local_polygons) if local_polygons is not None else None,
        "costmap_snapshots": list(costmap_snapshots) if costmap_snapshots else None,
        "ship_length": ship_length,
        "ship_beam": ship_beam,
        "frames_to_render": frames_to_render,
        "n_frames": n_frames,
        "x_min_ax": x_min_ax, "x_max_ax": x_max_ax,
        "y_min_ax": y_min_ax, "y_max_ax": y_max_ax,
        "figsize": figsize,
        "dpi": _RENDER_DPI,
    }
    writer, out_path = _animation_writer(output_file, fps)
    use_mp4 = str(output_file).lower().endswith(".mp4") and out_path.lower().endswith(".mp4")

    if n_workers > 1 and use_mp4 and n_render > 0:
        print(f"Creating costmap movie with {n_render} frames (parallel, n_workers={n_workers})...", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            n_workers_actual = min(n_workers, n_render, multiprocessing.cpu_count() or 4)
            ctx = multiprocessing.get_context("spawn")
            import imageio
            print(f"  Writing to {out_path}...", flush=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="subprocess")
                pool = ctx.Pool(n_workers_actual, initializer=_init_worker_costmap, initargs=(tmp_path,))
                with pool:
                    pending = {}
                    next_idx = 0
                    with imageio.get_writer(out_path, format="FFMPEG", mode="I", fps=fps, codec="libx264", output_params=["-preset", "veryfast"]) as vid:
                        for (idx, arr) in pool.imap_unordered(_worker_costmap, range(n_render), chunksize=max(1, n_render // (n_workers_actual * 4))):
                            pending[idx] = arr
                            while next_idx in pending:
                                vid.append_data(pending.pop(next_idx))
                                next_idx += 1
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        print(f"  Costmap movie saved: {out_path}", flush=True)
        return

    fig, ax = plt.subplots(figsize=figsize)
    print(f"Creating costmap movie with {n_render} frames (frame_skip={frame_skip})...", flush=True)

    positions_h = positions_history
    rotations_h = rotations_history
    local_polys = list(local_polygons) if local_polygons is not None else None
    snapshots = list(costmap_snapshots) if costmap_snapshots else None

    def update(frame_idx):
        k = frames_to_render[frame_idx]
        ax.clear()
        if snapshots:
            cm = None
            for (fidx, cminst) in snapshots:
                if fidx <= k:
                    cm = cminst
            if cm is None:
                cm = snapshots[0][1]
            cost = np.asarray(cm.cost_map)
            x_orig = getattr(cm, "x_origin", 0)
            y_orig = getattr(cm, "y_origin", 0)
            res = getattr(cm, "resolution", 2.0)
            ccols, crows = cost.shape[1], cost.shape[0]
        else:
            cost = np.asarray(cost_map.cost_map)
            x_orig, y_orig, res = x_origin, y_origin, resolution
            ccols, crows = cols, rows
        X = x_orig + np.arange(ccols + 1) * res
        Y = y_orig + np.arange(crows + 1) * res
        norm = mpl_colors.LogNorm(vmin=max(0.5, float(cost.min())), vmax=float(cost.max()))
        ax.pcolormesh(X, Y, cost, norm=norm, cmap="viridis", shading="flat")
        if positions_h is not None and rotations_h is not None and local_polys is not None and k < len(positions_h):
            pos = positions_h[k]
            rot = rotations_h[k]
            for i, local_poly in enumerate(local_polys):
                c, s = np.cos(rot[i]), np.sin(rot[i])
                R = np.array([[c, -s], [s, c]])
                world_poly = np.asarray(local_poly, dtype=float) @ R.T + np.asarray(pos[i], dtype=float)
                closed = np.vstack([world_poly, world_poly[0:1]])
                ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=0.4, alpha=0.9)
        else:
            for poly in (polygons or []):
                closed = np.vstack([poly, poly[0:1]])
                ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=0.4, alpha=0.9)
        if path is not None and len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], "r-", linewidth=1.5, alpha=0.9, label="Planned path")
        trail_end = min(k + 1, len(ship_history))
        if trail_end >= 2:
            trail_x = [ship_history[i][0][0] for i in range(trail_end)]
            trail_y = [ship_history[i][0][1] for i in range(trail_end)]
            ax.plot(trail_x, trail_y, "b-", linewidth=2, label="Actual path", zorder=5)
        if k < len(ship_history):
            ship_pos, ship_heading = ship_history[k][0], ship_history[k][1]
            ax.plot(ship_pos[0], ship_pos[1], "ko", markersize=6, zorder=6)
            hull = create_ship_hull_at_position_heading(ship_length, ship_beam, ship_pos, ship_heading)
            ax.add_patch(MplPolygon(hull, closed=True, facecolor="#404040", edgecolor="black", linewidth=1.5, alpha=0.9, zorder=6))
        if len(ship_history) > 0:
            ax.plot(ship_history[0][0][0], ship_history[0][0][1], "go", markersize=10, label="Start", zorder=7)
        if path is not None and len(path) > 0:
            ax.plot(path[-1, 0], path[-1, 1], "gs", markersize=10, label="Goal", zorder=7)
        ax.set_xlim(x_min_ax, x_max_ax)
        ax.set_ylim(y_min_ax, y_max_ax)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Costmap + ship trajectory (AUTO-IceNav style)\nFrame {k}/{n_frames-1}")
        ax.grid(True, alpha=0.3)
        return []

    anim = FuncAnimation(fig, update, frames=n_render, blit=False)
    print(f"  Saving to {out_path}...", flush=True)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"  Costmap movie saved: {out_path}", flush=True)


def animate_ice_ship_2d(
    polygons, positions_history, rotations_history, bonds_history, bonds,
    ship_history, ship_length, ship_beam, initial_centroids, thicknesses,
    output_file='medea_ship_fracture.gif', fps=20, frame_skip=1, n_workers=1,
    figsize=(16, 12),
    view_bounds=None,
):
    """Animate ice-ship interaction in 2D. n_workers>1 uses parallel rendering (faster). frame_skip>1 renders every Nth frame.
    view_bounds: optional (x_min, x_max, y_min, y_max) to fix the visible domain (e.g. show full grid); if None, auto from ship + ice."""
    n_frames = len(positions_history)
    frames_to_render = list(range(0, n_frames, frame_skip))
    if frames_to_render[-1] != n_frames - 1:
        frames_to_render.append(n_frames - 1)
    n_render = len(frames_to_render)
    n_elements = len(polygons)
    local_polys = []
    for poly in polygons:
        centroid = polygon_centroid(poly)
        local_polys.append(poly - centroid)
    if view_bounds is not None:
        x_min, x_max, y_min, y_max = view_bounds
    elif len(ship_history) > 0:
        ship_positions = np.array([h[0] for h in ship_history])
        x_min = float(ship_positions[:, 0].min() - ship_length - 20)
        x_max = float(ship_positions[:, 0].max() + ship_length + 20)
        y_min = float(ship_positions[:, 1].min() - ship_beam - 20)
        y_max = float(ship_positions[:, 1].max() + ship_beam + 20)
        if n_elements > 0:
            all_pos = np.concatenate(positions_history)
            x_min = min(x_min, float(all_pos[:, 0].min() - 20))
            x_max = max(x_max, float(all_pos[:, 0].max() + 20))
            y_min = min(y_min, float(all_pos[:, 1].min() - 20))
            y_max = max(y_max, float(all_pos[:, 1].max() + 20))
    else:
        x_min, x_max, y_min, y_max = -50.0, 50.0, -50.0, 50.0
        if n_elements > 0:
            all_pos = np.concatenate(positions_history)
            x_min = min(x_min, float(all_pos[:, 0].min() - 20))
            x_max = max(x_max, float(all_pos[:, 0].max() + 20))
            y_min = min(y_min, float(all_pos[:, 1].min() - 20))
            y_max = max(y_max, float(all_pos[:, 1].max() + 20))
    water_color = '#1A4D70'
    ship_color = '#404040'
    if n_elements > 0 and len(thicknesses) > 0:
        thickness_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
        ice_colors = [f'#{int(240-100*t):02x}{int(250-80*t):02x}{int(255-40*t):02x}' for t in thickness_norm]
    else:
        ice_colors = []

    writer, out_path = _animation_writer(output_file, fps)
    use_mp4 = str(output_file).lower().endswith(".mp4") and out_path.lower().endswith(".mp4")

    if n_workers > 1 and use_mp4 and n_render > 0:
        print(f"Creating 2D animation with {n_render} frames (parallel, n_workers={n_workers})...", flush=True)
        data = {
            "positions_history": positions_history,
            "rotations_history": rotations_history,
            "bonds_history": bonds_history,
            "bonds": bonds,
            "frames_to_render": frames_to_render,
            "n_frames": n_frames,
            "n_elements": n_elements,
            "local_polys": local_polys,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "ice_colors": ice_colors,
            "ship_history": ship_history,
            "ship_length": ship_length,
            "ship_beam": ship_beam,
            "water_color": water_color,
            "ship_color": ship_color,
            "figsize": figsize,
            "dpi": _RENDER_DPI,
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            n_workers_actual = min(n_workers, n_render, multiprocessing.cpu_count() or 4)
            ctx = multiprocessing.get_context("spawn")
            import imageio
            print(f"  Writing to {out_path}...", flush=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="subprocess")
                pool = ctx.Pool(n_workers_actual, initializer=_init_worker_2d, initargs=(tmp_path,))
                with pool:
                    pending = {}
                    next_idx = 0
                    with imageio.get_writer(out_path, format="FFMPEG", mode="I", fps=fps, codec="libx264", output_params=["-preset", "veryfast"]) as vid:
                        for (idx, arr) in pool.imap_unordered(_worker_2d, range(n_render), chunksize=max(1, n_render // (n_workers_actual * 4))):
                            pending[idx] = arr
                            while next_idx in pending:
                                vid.append_data(pending.pop(next_idx))
                                next_idx += 1
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        print(f"Saving to {out_path}...", flush=True)
        print("Done!", flush=True)
        return

    fig, ax = plt.subplots(figsize=figsize)
    def rotate_polygon(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        return poly @ np.array([[c, -s], [s, c]]).T

    print(f"Creating 2D animation with {n_render} frames (frame_skip={frame_skip})...", flush=True)

    def update(frame_idx):
        frame = frames_to_render[frame_idx]
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

    anim = FuncAnimation(fig, update, frames=n_render, blit=False)
    print(f"Saving to {out_path}...", flush=True)
    anim.save(out_path, writer=writer)
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
    figsize=(12, 8.96),  # 1200×896 px at 100 DPI; divisible by 16 for FFmpeg macro_block_size
    n_workers=1,
):
    """3D animation with z from submergence. n_workers>1 uses parallel rendering (faster)."""
    vc = viz_config or get_default_visualization_config()
    n_frames = len(positions_history)
    n_elements = len(polygons)
    local_polys = [poly - polygon_centroid(poly) for poly in polygons]
    all_pos = np.concatenate(positions_history)
    x_min, x_max = float(all_pos[:, 0].min() - 50), float(all_pos[:, 0].max() + 50)
    y_min, y_max = float(all_pos[:, 1].min() - 50), float(all_pos[:, 1].max() + 50)
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
    frames_to_render = list(range(0, n_frames, frame_skip))
    if frames_to_render[-1] != n_frames - 1:
        frames_to_render.append(n_frames - 1)
    n_render = len(frames_to_render)
    writer, out_path = _animation_writer(output_file, fps)
    use_mp4 = str(output_file).lower().endswith(".mp4") and out_path.lower().endswith(".mp4")

    if n_workers > 1 and use_mp4 and n_render > 0:
        print(f"  Creating 3D animation with {n_render} frames (parallel, n_workers={n_workers})...", flush=True)
        data = {
            "positions_history": positions_history,
            "rotations_history": rotations_history,
            "bonds_history": bonds_history,
            "bonds": bonds,
            "ship_history": ship_history,
            "z_positions": z_positions,
            "tilts_x": tilts_x,
            "tilts_y": tilts_y,
            "thicknesses": thicknesses,
            "local_polys": local_polys,
            "n_elements": n_elements,
            "n_frames": n_frames,
            "frames_to_render": frames_to_render,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "domain_width": domain_width,
            "domain_height": domain_height,
            "ship_length": ship_length,
            "ship_beam": ship_beam,
            "ship_draft": ship_draft,
            "color_by_submergence": color_by_submergence,
            "SURFACED_FACE": SURFACED_FACE, "SURFACED_SIDE": SURFACED_SIDE,
            "SUBMERGED_FACE": SUBMERGED_FACE, "SUBMERGED_SIDE": SUBMERGED_SIDE,
            "Z_SURFACE_THRESHOLD": Z_SURFACE_THRESHOLD,
            "ice_colors": ice_colors,
            "view_elev": vc.view_elev,
            "view_azim_start": vc.view_azim_start,
            "view_azim_per_frame": vc.view_azim_per_frame,
            "figsize": figsize,
            "dpi": _RENDER_DPI,
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            n_workers_actual = min(n_workers, n_render, multiprocessing.cpu_count() or 4)
            ctx = multiprocessing.get_context("spawn")
            import imageio
            print(f"  Writing to {out_path}...", flush=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="subprocess")
                pool = ctx.Pool(n_workers_actual, initializer=_init_worker_3d, initargs=(tmp_path,))
                with pool:
                    pending = {}
                    next_idx = 0
                    with imageio.get_writer(out_path, format="FFMPEG", mode="I", fps=fps, codec="libx264", output_params=["-preset", "veryfast"]) as vid:
                        for (idx, arr) in pool.imap_unordered(_worker_3d, range(n_render), chunksize=max(1, n_render // (n_workers_actual * 4))):
                            pending[idx] = arr
                            while next_idx in pending:
                                vid.append_data(pending.pop(next_idx))
                                next_idx += 1
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        print("  Done!", flush=True)
        return

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    print(f"  Creating 3D animation with {n_render} frames...", flush=True)

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
    writer, out_path = _animation_writer(output_file, fps)
    print(f"  Saving to {out_path}...", flush=True)
    anim.save(out_path, writer=writer)
    print("  Done!", flush=True)
    plt.close(fig)
