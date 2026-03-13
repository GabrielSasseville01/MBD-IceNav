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
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib import colors as mpl_colors
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from medea_ice_ship.geometry import (
    polygon_centroid,
    create_ship_hull_at_position_heading,
    point_in_polygon,
    closest_point_and_outward_normal,
)
from medea_ice_ship.bonds import find_connected_components
from medea_ice_ship.config import VisualizationConfig, get_default_visualization_config

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False


def _rigid_correct_positions_if_thick_ice(positions_history, bonds, thicknesses, max_fracture_thickness, n_elements, initial_floe_of=None, rotations_history=None):
    """When all thicknesses > max_fracture_thickness, return (positions_history, rotations_history) with each
    component drawn rigidly: same relative positions and same rotation as frame 0 (no gaps, no spin).
    If initial_floe_of is provided (parent floe ID per cell), use one component per floe.
    Returns (positions_out, rotations_out); if no correction applied, returns (positions_history, rotations_history)."""
    if not positions_history or thicknesses is None or len(thicknesses) != n_elements:
        return positions_history, (rotations_history if rotations_history is not None else None)
    if max_fracture_thickness is None or np.any(thicknesses <= max_fracture_thickness):
        return positions_history, rotations_history
    n = n_elements
    if initial_floe_of is not None and len(initial_floe_of) == n_elements:
        floe_ids = np.asarray(initial_floe_of, dtype=int)
        components = [np.where(floe_ids == k)[0].tolist() for k in np.unique(floe_ids)]
        print(f"  [Thick-ice] Rigid display: {len(components)} floes (from parent IDs)", flush=True)
    elif bonds and len(bonds) > 0:
        intact_all = np.ones(len(bonds), dtype=bool)
        components = find_connected_components(n, bonds, intact_all)
    else:
        return positions_history, rotations_history
    pos0 = np.asarray(positions_history[0], dtype=float)
    out_pos = []
    for pos_f in positions_history:
        pos_f = np.asarray(pos_f, dtype=float)
        new_pos = pos_f.copy()
        for comp in components:
            if len(comp) == 0:
                continue
            comp = np.array(comp)
            com0 = np.mean(pos0[comp], axis=0)
            com_f = np.mean(pos_f[comp], axis=0)
            offset = com_f - com0
            for i in comp:
                new_pos[i] = pos0[i] + offset
        out_pos.append(new_pos)
    # Keep initial rotations for all frames so each floe doesn't spin or show per-cell rotation drift.
    # Use frame-0 rotation per cell (no override) so polygons stay aligned with the rigid-corrected positions.
    if rotations_history is not None and len(rotations_history) > 0:
        rot0 = np.asarray(rotations_history[0], dtype=float)
        out_rot = [rot0.copy() for _ in range(len(positions_history))]
        return out_pos, out_rot
    return out_pos, rotations_history


def _build_merged_floe_polygons_per_frame(local_polys, positions_history, rotations_history, initial_floe_of, ice_colors):
    """When all ice is thick and initial_floe_of is set, build one merged polygon per floe per frame (rigid body: frame-0 shape + COM offset).
    Returns (merged_display_per_frame, floe_ids) where merged_display_per_frame[f] = [(poly Nx2, color), ...] for each floe, or (None, []) if not applicable."""
    if not _HAS_SHAPELY or not local_polys or not positions_history or initial_floe_of is None or len(initial_floe_of) != len(local_polys):
        return None, []
    n_elements = len(local_polys)
    floe_ids = np.unique(np.asarray(initial_floe_of, dtype=int)).tolist()
    pos0 = np.asarray(positions_history[0], dtype=float)
    rot0 = np.asarray(rotations_history[0], dtype=float)

    def rotate_poly(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.asarray(poly, dtype=float) @ np.array([[c, -s], [s, c]]).T

    # Build merged polygon per floe in frame-0 world coordinates
    merged_by_floe = {}
    floe_color = {}
    for fid in floe_ids:
        indices = [i for i in range(n_elements) if initial_floe_of[i] == fid]
        if not indices:
            continue
        geoms = []
        for i in indices:
            world_poly = rotate_poly(local_polys[i], rot0[i]) + pos0[i]
            if len(world_poly) >= 3:
                geoms.append(ShapelyPolygon(world_poly))
        if not geoms:
            continue
        union = unary_union(geoms)
        if union.is_empty:
            continue
        if hasattr(union, 'geoms'):
            # MultiPolygon: collect all parts (e.g. floe split by narrow gap)
            polys_for_floe = []
            for g in union.geoms:
                if not g.is_empty and g.area >= 1e-10:
                    polys_for_floe.append(np.array(g.exterior.coords))
            if not polys_for_floe:
                continue
            merged_by_floe[fid] = polys_for_floe
        else:
            if union.area < 1e-10:
                continue
            ext = np.array(union.exterior.coords)
            merged_by_floe[fid] = [ext]
        floe_color[fid] = ice_colors[indices[0]] if indices[0] < len(ice_colors) else "#C0E0F0"

    if not merged_by_floe:
        return None, []

    # Per-frame: translate merged polygon(s) by (com_f - com0)
    com0_by_floe = {fid: np.mean(pos0[[i for i in range(n_elements) if initial_floe_of[i] == fid]], axis=0) for fid in merged_by_floe}
    merged_display_per_frame = []
    for f in range(len(positions_history)):
        pos_f = np.asarray(positions_history[f], dtype=float)
        frame_list = []
        for fid in floe_ids:
            if fid not in merged_by_floe:
                continue
            indices = [i for i in range(n_elements) if initial_floe_of[i] == fid]
            com_f = np.mean(pos_f[indices], axis=0)
            offset = com_f - com0_by_floe[fid]
            for poly in merged_by_floe[fid]:
                frame_list.append((poly + offset, floe_color[fid]))
        merged_display_per_frame.append(frame_list)
    return merged_display_per_frame, floe_ids


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


def animate_lattice_search(
    cost_map,
    search_history: list,
    path,
    start: tuple,
    goal: tuple,
    output_file: str = "lattice_search.mp4",
    frame_stride: int = 1,
    fps: int = 15,
    figsize: tuple = (10, 8),
):
    """
    Generate a movie showing the A* lattice graph search evolution.

    Frames show the costmap, start/goal, and the growing set of expanded nodes
    and edges (motion primitives) as the search progresses. The final chosen path
    is drawn at the end.

    Args:
        cost_map: CostMap (dpdem.planning.cost_map) with .cost_map, .x_origin, .y_origin, .width, .height
        search_history: From planner.plan(..., return_search_history=True); list of
            ( (x, y, heading_idx), [ (path_world Nx3, (end_x, end_y)), ... ] )
        path: Nx3 [x, y, heading] or None (the planned path)
        start: (x, y, heading)
        goal: (x, y)
        output_file: Output path (.mp4 or .gif)
        frame_stride: Show every Nth expansion (1 = every step; 5 = fewer, faster video)
        fps: Frames per second
        figsize: Figure size (inches)
    """
    ox = cost_map.x_origin
    oy = cost_map.y_origin
    w = cost_map.width
    h = cost_map.height
    # cost_map: row = (y - y_origin)/resolution, so row 0 = bottom (y_origin); imshow origin='lower' draws row 0 at bottom
    grid = np.asarray(cost_map.cost_map)

    # Build cumulative nodes and edges per display frame
    all_nodes = []  # list of (x, y) for each expansion
    all_edges = []  # list of (path_xy Nx2) for each edge
    for (xyh, edges) in search_history:
        x, y, _ = xyh
        all_nodes.append((x, y))
        for (path_world, _) in edges:
            all_edges.append(path_world[:, :2].copy())

    # Display frames: show state after expansion 0, stride, 2*stride, ... then final
    n_expansions = len(search_history)
    if n_expansions == 0:
        frame_indices = [0]
    else:
        frame_indices = list(range(0, n_expansions, frame_stride))
        if frame_indices[-1] != n_expansions - 1:
            frame_indices.append(n_expansions - 1)

    n_frames = len(frame_indices)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlim(ox, ox + w)
    ax.set_ylim(oy, oy + h)

    # Static: costmap background (log scale so open water, thin ice, thick ice are all visible)
    grid_max = float(np.nanmax(grid))
    grid_min = float(np.nanmin(grid[grid > 0])) if np.any(grid > 0) else 1.0
    vmin = max(0.5, grid_min * 0.5)
    vmax = max(2.0, grid_max)
    # Avoid 0 for LogNorm (would be -inf)
    grid_display = np.where(grid > 0, grid, vmin)
    im = ax.imshow(
        grid_display,
        extent=[ox, ox + w, oy, oy + h],
        origin="lower",
        cmap="viridis",
        alpha=0.85,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Start and goal (persistent)
    ax.plot(start[0], start[1], "go", markersize=12, label="Start", zorder=5)
    ax.plot(goal[0], goal[1], "r*", markersize=16, label="Goal", zorder=5)

    # Dynamic: edges (LineCollection of segments) and nodes (scatter), path (line)
    edge_collection = LineCollection([], colors="b", linewidths=0.5, alpha=0.7, label="Explored edges")
    ax.add_collection(edge_collection)
    node_scatter = ax.scatter([], [], c="cyan", s=6, alpha=0.9, label="Expanded nodes", zorder=3)
    path_line = ax.plot([], [], "r-", linewidth=2.5, label="Chosen path", zorder=4)[0]

    def init():
        edge_collection.set_segments([])
        node_scatter.set_offsets(np.empty((0, 2)))
        path_line.set_data([], [])
        return edge_collection, node_scatter, path_line

    def update(frame_idx):
        k = frame_indices[frame_idx]
        # Nodes up to and including expansion k
        nodes_xy = all_nodes[: k + 1]
        if nodes_xy:
            node_scatter.set_offsets(np.array(nodes_xy))
        else:
            node_scatter.set_offsets(np.empty((0, 2)))

        # Edges: list of (Nx2) arrays for each primitive path
        segments = []
        for (_, edges) in search_history[: k + 1]:
            for (path_world, _) in edges:
                segments.append(path_world[:, :2])
        edge_collection.set_segments(segments if segments else [[]])

        # On last frame, draw chosen path
        if path is not None and frame_idx == n_frames - 1 and len(path) > 0:
            path_line.set_data(path[:, 0], path[:, 1])
        else:
            path_line.set_data([], [])

        ax.set_title(f"A* lattice search — expansion {k + 1}/{n_expansions}" if n_expansions else "A* lattice search (no expansions)")
        return edge_collection, node_scatter, path_line

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )
    writer, out_path = _animation_writer(output_file, fps)
    anim.save(out_path, writer=writer, dpi=120)
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


def _remaining_planned_path_from_snapshots(ship_pos, path_snapshots, frame_idx):
    """
    Return the planned path from the ship onward for a given frame (for 2D/3D movies).
    path_snapshots: list of (frame_idx, path Nx3). Uses latest path with snapshot frame <= frame_idx.
    Returns (Nx2) array [ship_pos, ...waypoints to goal] or None if no path.
    """
    if not path_snapshots or frame_idx is None:
        return None
    sorted_snapshots = sorted(path_snapshots, key=lambda t: t[0])
    path = None
    for (fidx, p) in sorted_snapshots:
        if fidx <= frame_idx:
            path = p
    if path is None and sorted_snapshots:
        path = sorted_snapshots[0][1]
    if path is None or len(path) < 2:
        return None
    path_xy = np.asarray(path)[:, :2]
    ship_pos = np.asarray(ship_pos).ravel()[:2]
    # Trim to segment from ship onward: closest waypoint index, then from there to end
    dists = np.linalg.norm(path_xy - ship_pos, axis=1)
    trim_idx = int(np.argmin(dists))
    remaining = np.vstack([ship_pos.reshape(1, 2), path_xy[trim_idx:]])
    if len(remaining) < 2:
        return None
    return remaining


def _draw_2d_frame_to_array(data, frame_idx):
    """Draw a single 2D frame; return RGB array (H, W, 3) uint8. Used by parallel workers."""
    frame = data["frames_to_render"][frame_idx]
    n_elements = data["n_elements"]
    intact = data["bonds_history"][frame]
    bonds = data["bonds"]
    thicknesses_arr = data.get("thicknesses")
    max_ft = data.get("max_fracture_thickness", 15.0)
    local_polys = data["local_polys"]
    ice_colors = data["ice_colors"]
    broken_neighbors = {i: set() for i in range(n_elements)}
    for b, (i, j) in enumerate(bonds):
        if not intact[b]:
            # Only show as broken if both endpoints are at or below max_fracture_thickness (thick ice never appears fractured)
            if thicknesses_arr is not None and len(thicknesses_arr) > max(i, j) and max_ft is not None:
                if thicknesses_arr[i] <= max_ft and thicknesses_arr[j] <= max_ft:
                    broken_neighbors[i].add(j)
                    broken_neighbors[j].add(i)
            else:
                broken_neighbors[i].add(j)
                broken_neighbors[j].add(i)

    fig, ax = plt.subplots(figsize=data["figsize"], dpi=data["dpi"])
    ax.set_xlim(data["x_min"], data["x_max"])
    ax.set_ylim(data["y_min"], data["y_max"])
    ax.set_aspect("equal")
    ax.set_facecolor(data["water_color"])
    cviz = data.get("current_viz")
    if cviz is not None:
        cX, cY, cU, cV = cviz
        cMag = np.sqrt(cU**2 + cV**2)
        max_mag = float(np.max(cMag)) if cMag.size > 0 else 1e-6
        x_r, y_r = data["x_max"] - data["x_min"], data["y_max"] - data["y_min"]
        arrow_scale = (0.25 * min(x_r, y_r)) / (max_mag + 1e-12)
        cU_plot, cV_plot = cU * arrow_scale, cV * arrow_scale
        ax.quiver(cX, cY, cU_plot, cV_plot, cMag, cmap='cool', alpha=0.45,
                  angles='xy', scale_units='xy', scale=1,
                  zorder=0, width=0.003)
    if data.get("merged_floe_display_per_frame") is not None:
        # Thick-ice: one merged polygon per floe (no internal edges)
        for poly, color in data["merged_floe_display_per_frame"][frame]:
            patch = MplPolygon(poly, closed=True, facecolor=color, edgecolor=color, linewidth=0.5, alpha=0.95)
            ax.add_patch(patch)
    else:
        positions = data["positions_history"][frame]
        rotations = data["rotations_history"][frame]
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
        # Planned path from ship onward (updates at replan; only show remaining segment)
        path_snapshots = data.get("path_snapshots")
        if path_snapshots is not None:
            remaining = _remaining_planned_path_from_snapshots(ship_pos_frame, path_snapshots, frame)
            if remaining is not None and len(remaining) >= 2:
                ax.plot(remaining[:, 0], remaining[:, 1], "r-", linewidth=1.5, alpha=0.9, label="Planned path (remaining)", zorder=4)
    goal_y_line = data.get("goal_y_line")
    if goal_y_line is not None:
        ax.axhline(goal_y_line, color="lime", ls="--", lw=2, label=f"Goal line y={goal_y_line:.0f}", zorder=4)
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
    ship_x, ship_y = float(ship_pos[0]), float(ship_pos[1])
    if not np.isfinite(ship_x):
        ship_x = 0.0
    if not np.isfinite(ship_y):
        ship_y = 0.0
    positions = np.asarray(data["positions_history"][frame], dtype=float)
    rotations = np.asarray(data["rotations_history"][frame], dtype=float)
    intact = data["bonds_history"][frame]
    z_pos = np.asarray(data["z_positions"][frame], dtype=float)
    tilt_x = np.asarray(data["tilts_x"][frame], dtype=float)
    tilt_y = np.asarray(data["tilts_y"][frame], dtype=float)
    # Sanitize so 3D projection never gets degenerate (avoids LinAlgError: Singular matrix)
    x_center = 0.5 * (data["x_min"] + data["x_max"])
    y_center = 0.5 * (data["y_min"] + data["y_max"])
    bad_pos = ~np.isfinite(positions)
    positions = np.where(bad_pos, np.array([x_center, y_center]), positions)
    bad_rot = ~np.isfinite(rotations)
    rotations = np.where(bad_rot, 0.0, rotations)
    z_pos = np.where(np.isfinite(z_pos), z_pos, 0.0)
    tilt_x = np.where(np.isfinite(tilt_x), tilt_x, 0.0)
    tilt_y = np.where(np.isfinite(tilt_y), tilt_y, 0.0)
    bonds = data["bonds"]
    thicknesses = data["thicknesses"]
    local_polys = data["local_polys"]
    ship_draft = data["ship_draft"]
    ship_length = data["ship_length"]
    ship_beam = data["ship_beam"]
    dw = max(float(data["domain_width"]), 1.0)
    dh = max(float(data["domain_height"]), 1.0)
    z_span = max((ship_draft + 8) * 2, 1.0)

    fig = plt.figure(figsize=data["figsize"], dpi=data["dpi"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(data["x_min"], data["x_max"])
    ax.set_ylim(data["y_min"], data["y_max"])
    ax.set_zlim(-ship_draft - 3, 5)
    ax.set_box_aspect([dw, dh, z_span])
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
        max_ft = data.get("max_fracture_thickness", 15.0)
        def _bond_break_visible(b, pi, pj):
            if not intact[b]:
                if thicknesses is not None and len(thicknesses) > max(pi, pj) and max_ft is not None:
                    return thicknesses[pi] <= max_ft and thicknesses[pj] <= max_ft
                return True
            return False
        has_breaks = any(_bond_break_visible(b, pi, pj) for b, (pi, pj) in enumerate(bonds) if pi == i or pj == i)
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
    path_snapshots = data.get("path_snapshots")
    if path_snapshots is not None:
        remaining = _remaining_planned_path_from_snapshots(np.array([ship_x, ship_y]), path_snapshots, frame)
        if remaining is not None and len(remaining) >= 2:
            z_path = np.zeros(len(remaining))
            ax.plot(remaining[:, 0], remaining[:, 1], z_path, "r-", linewidth=1.5, alpha=0.9)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    n_broken = np.sum(~intact)
    title = f"DPDEM 3D Ice-Ship Simulation (MEDEA)\nFrame {frame}/{data['n_frames']-1} | Bonds: {n_broken}/{len(bonds)} broken"
    if data["color_by_submergence"]:
        title += "\nBlue = submerged | Light = at surface"
    ax.set_title(title)
    try:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    except np.linalg.LinAlgError:
        # Degenerate 3D view (e.g. NaN/inf in data); return a gray frame so video completes
        w, h = int(data["figsize"][0] * data["dpi"]), int(data["figsize"][1] * data["dpi"])
        buf = np.full((h, w, 3), 128, dtype=np.uint8)
    finally:
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
    """Draw one frame of the costmap movie (plan-only style): costmap + ice + planned path (red) + actual path (blue) + ship."""
    k = data["frames_to_render"][frame_idx]
    ship_history = data["ship_history"]
    path = data.get("path")
    ship_length = data["ship_length"]
    ship_beam = data["ship_beam"]
    x_min_ax, x_max_ax = data["x_min_ax"], data["x_max_ax"]
    y_min_ax, y_max_ax = data["y_min_ax"], data["y_max_ax"]
    thicknesses = data.get("thicknesses")
    goal_y_line = data.get("goal_y_line")

    # Costmap: use snapshot for frame k if provided
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
        extent = [x_origin, x_origin + cm.width, y_origin, y_origin + cm.height]
    else:
        cost_grid = data["cost_grid"]
        x_origin = data["x_origin"]
        y_origin = data["y_origin"]
        res = data.get("resolution", 2.0)
        ccols, rrows = cost_grid.shape[1], cost_grid.shape[0]
        extent = [x_origin, x_origin + ccols * res, y_origin, y_origin + rrows * res]

    cost = np.asarray(cost_grid)
    vmax = np.percentile(cost[cost > 0], 95) if np.any(cost > 0) else 1.0

    fig, ax = plt.subplots(figsize=data["figsize"], dpi=data["dpi"])
    ax.imshow(cost, origin="lower", extent=extent, cmap="YlOrRd", alpha=0.5,
              vmin=0, vmax=float(vmax))

    # Ice: either moving (per-frame) or static; use PatchCollection with thickness when available
    positions_history = data.get("positions_history")
    rotations_history = data.get("rotations_history")
    local_polygons = data.get("local_polygons")
    static_polygons = data.get("polygons") or []

    if thicknesses is not None and len(thicknesses) > 0:
        patches = []
        colors_th = []
        if positions_history is not None and rotations_history is not None and local_polygons is not None and k < len(positions_history):
            pos = positions_history[k]
            rot = rotations_history[k]
            for i, local_poly in enumerate(local_polygons):
                if i >= len(thicknesses):
                    break
                c, s = np.cos(rot[i]), np.sin(rot[i])
                R = np.array([[c, -s], [s, c]])
                world_poly = np.asarray(local_poly, dtype=float) @ R.T + np.asarray(pos[i], dtype=float)
                patches.append(MplPolygon(world_poly, closed=True))
                colors_th.append(thicknesses[i])
        else:
            for i, poly in enumerate(static_polygons):
                if i >= len(thicknesses):
                    break
                patches.append(MplPolygon(poly, closed=True))
                colors_th.append(thicknesses[i])
        if patches:
            pc = PatchCollection(patches, cmap="Blues", edgecolor="#555555", linewidth=0.3, alpha=0.6)
            pc.set_array(np.array(colors_th))
            pc.set_clim(np.min(thicknesses), np.max(thicknesses))
            ax.add_collection(pc)
            fig.colorbar(pc, ax=ax, shrink=0.5, label="Thickness [m]", location="left")
    else:
        if positions_history is not None and rotations_history is not None and local_polygons is not None and k < len(positions_history):
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

    # Planned path
    draw_path = data.get("path")
    path_snapshots = data.get("path_snapshots")
    if path_snapshots:
        path_snapshots_sorted = sorted(path_snapshots, key=lambda t: t[0])
        draw_path = None
        for (fidx, p) in path_snapshots_sorted:
            if fidx <= k:
                draw_path = p
        if draw_path is None and path_snapshots_sorted:
            draw_path = path_snapshots_sorted[0][1]
    if draw_path is not None and len(draw_path) > 0:
        draw_path = np.asarray(draw_path)
        ax.plot(draw_path[:, 0], draw_path[:, 1], "r-", lw=2.5, label="Planned path", zorder=5)
        ax.plot(draw_path[0, 0], draw_path[0, 1], "go", ms=12, zorder=6, label="Start")
        ax.plot(draw_path[-1, 0], draw_path[-1, 1], "r*", ms=16, zorder=6, label="End")

    # Goal line (plan-only style)
    if goal_y_line is not None:
        ax.axhline(goal_y_line, color="lime", ls="--", lw=2, label=f"Goal line y={goal_y_line:.0f}", zorder=4)

    trail_end = min(k + 1, len(ship_history))
    if trail_end >= 2:
        trail_x = [ship_history[i][0][0] for i in range(trail_end)]
        trail_y = [ship_history[i][0][1] for i in range(trail_end)]
        ax.plot(trail_x, trail_y, "b-", lw=2, label="Actual path", zorder=5)
    if k < len(ship_history):
        ship_pos, ship_heading = ship_history[k][0], ship_history[k][1]
        ax.plot(ship_pos[0], ship_pos[1], "ko", markersize=6, zorder=6)
        hull = create_ship_hull_at_position_heading(ship_length, ship_beam, ship_pos, ship_heading)
        ax.add_patch(MplPolygon(hull, closed=True, fc="orange", ec="black", lw=1.5, alpha=0.8, zorder=6))
    if len(ship_history) > 0 and (draw_path is None or len(draw_path) == 0):
        ax.plot(ship_history[0][0][0], ship_history[0][0][1], "go", ms=12, label="Start", zorder=7)
    if goal_y_line is None:
        goal_xy = data.get("goal")
        if goal_xy is None and draw_path is not None and len(draw_path) > 0:
            goal_xy = np.asarray(draw_path[-1])[:2]
        elif goal_xy is None and path is not None and len(path) > 0:
            goal_xy = np.asarray(path[-1])[:2]
        if goal_xy is not None:
            g = np.asarray(goal_xy).ravel()[:2]
            ax.plot(g[0], g[1], "mx", ms=14, mew=3, zorder=6, label="Goal")
    ax.set_xlim(x_min_ax, x_max_ax)
    ax.set_ylim(y_min_ax, y_max_ax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Costmap + ship trajectory\nFrame {k}/{data['n_frames']-1}")
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
    path_snapshots=None,
    goal=None,
    goal_y_line=None,
    polygons=None,
    thicknesses=None,
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
    figsize=(12, 12),
):
    """
    Generate a costmap movie (plan-only style): costmap (YlOrRd) as background, ice polygons
    with thickness colorbar (Blues), planned path (red), actual path (blue), ship (orange),
    start (green circle), end (red star), goal line (lime dashed) when goal_y_line is set.

    cost_map: default costmap (used for all frames if costmap_snapshots is None)
    bounds: dict with x_min, x_max, y_min, y_max, x_origin, y_origin, etc.
    ship_history: list of (position_2d, heading) per frame
    path: Nx3 planned path [x,y,heading] or None (used when path_snapshots is None)
    path_snapshots: optional list of (frame_idx, path) so the red planned path updates at replan times.
    goal: fixed (x, y) for point goals; when goal_y_line is set, path end is shown as red star.
    goal_y_line: when set (e.g. y=700), draw horizontal goal line and show path end as red star.
    polygons: list of polygon arrays for static ice (or base for moving ice with local_polygons).
    thicknesses: per-polygon thickness [m] for ice colorbar (Blues); enables thickness shading.
    positions_history, rotations_history, local_polygons: per-frame ice state for moving ice.
    costmap_snapshots: optional list of (frame_idx, cost_map) for costmap updates at replan times.
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
    margin = 25
    x_min_ax = float(ship_positions[:, 0].min() - ship_length - margin)
    x_max_ax = float(ship_positions[:, 0].max() + ship_length + margin)
    y_min_ax = float(ship_positions[:, 1].min() - ship_beam - margin)
    y_max_ax = float(ship_positions[:, 1].max() + ship_beam + margin)
    if path is not None and len(path) > 0:
        x_min_ax = min(x_min_ax, float(path[:, 0].min()) - margin)
        x_max_ax = max(x_max_ax, float(path[:, 0].max()) + margin)
        y_min_ax = min(y_min_ax, float(path[:, 1].min()) - margin)
        y_max_ax = max(y_max_ax, float(path[:, 1].max()) + margin)
    if goal is not None:
        g = np.asarray(goal).ravel()[:2]
        x_min_ax = min(x_min_ax, float(g[0]) - margin)
        x_max_ax = max(x_max_ax, float(g[0]) + margin)
        y_min_ax = min(y_min_ax, float(g[1]) - margin)
        y_max_ax = max(y_max_ax, float(g[1]) + margin)
    # Same view extent as 2D simulation when ice history is provided
    if positions_history is not None and len(positions_history) > 0:
        all_pos = np.concatenate(positions_history)
        x_min_ax = min(x_min_ax, float(all_pos[:, 0].min()) - margin)
        x_max_ax = max(x_max_ax, float(all_pos[:, 0].max()) + margin)
        y_min_ax = min(y_min_ax, float(all_pos[:, 1].min()) - margin)
        y_max_ax = max(y_max_ax, float(all_pos[:, 1].max()) + margin)

    data = {
        "cost_grid": cost_map.cost_map,
        "x_origin": x_origin,
        "y_origin": y_origin,
        "resolution": resolution,
        "cols": cols,
        "rows": rows,
        "ship_history": ship_history,
        "path": np.asarray(path) if path is not None else None,
        "path_snapshots": list(path_snapshots) if path_snapshots else None,
        "goal": np.asarray(goal, dtype=float).ravel()[:2] if goal is not None else None,
        "goal_y_line": goal_y_line,
        "polygons": list(polygons) if polygons is not None else [],
        "thicknesses": np.asarray(thicknesses) if thicknesses is not None else None,
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

    thicknesses_h = np.asarray(thicknesses) if thicknesses is not None else None
    goal_y_line_h = goal_y_line
    if thicknesses_h is not None and len(thicknesses_h) > 0:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap=plt.cm.Blues, norm=Normalize(vmin=float(thicknesses_h.min()), vmax=float(thicknesses_h.max())))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, label="Thickness [m]", location="left")

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
            extent = [x_orig, x_orig + cm.width, y_orig, y_orig + cm.height]
        else:
            cost = np.asarray(cost_map.cost_map)
            x_orig, y_orig, res = x_origin, y_origin, resolution
            ccols, crows = cols, rows
            extent = [x_orig, x_orig + ccols * res, y_orig, y_orig + crows * res]
        vmax = np.percentile(cost[cost > 0], 95) if np.any(cost > 0) else 1.0
        ax.imshow(cost, origin="lower", extent=extent, cmap="YlOrRd", alpha=0.5, vmin=0, vmax=float(vmax))
        if thicknesses_h is not None and len(thicknesses_h) > 0:
            patches = []
            colors_th = []
            if positions_h is not None and rotations_h is not None and local_polys is not None and k < len(positions_h):
                pos = positions_h[k]
                rot = rotations_h[k]
                for i, local_poly in enumerate(local_polys):
                    if i >= len(thicknesses_h):
                        break
                    c, s = np.cos(rot[i]), np.sin(rot[i])
                    R = np.array([[c, -s], [s, c]])
                    world_poly = np.asarray(local_poly, dtype=float) @ R.T + np.asarray(pos[i], dtype=float)
                    patches.append(MplPolygon(world_poly, closed=True))
                    colors_th.append(thicknesses_h[i])
            else:
                for i, poly in enumerate(polygons or []):
                    if i >= len(thicknesses_h):
                        break
                    patches.append(MplPolygon(poly, closed=True))
                    colors_th.append(thicknesses_h[i])
            if patches:
                pc = PatchCollection(patches, cmap="Blues", edgecolor="#555555", linewidth=0.3, alpha=0.6)
                pc.set_array(np.array(colors_th))
                pc.set_clim(np.min(thicknesses_h), np.max(thicknesses_h))
                ax.add_collection(pc)
        else:
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
        draw_path = path
        if path_snapshots:
            path_snapshots_sorted = sorted(path_snapshots, key=lambda t: t[0])
            draw_path = None
            for (fidx, p) in path_snapshots_sorted:
                if fidx <= k:
                    draw_path = p
            if draw_path is None and path_snapshots_sorted:
                draw_path = path_snapshots_sorted[0][1]
        if draw_path is not None and len(draw_path) > 0:
            draw_path = np.asarray(draw_path)
            ax.plot(draw_path[:, 0], draw_path[:, 1], "r-", lw=2.5, label="Planned path", zorder=5)
            ax.plot(draw_path[0, 0], draw_path[0, 1], "go", ms=12, zorder=6, label="Start")
            ax.plot(draw_path[-1, 0], draw_path[-1, 1], "r*", ms=16, zorder=6, label="End")
        if goal_y_line_h is not None:
            ax.axhline(goal_y_line_h, color="lime", ls="--", lw=2, label=f"Goal line y={goal_y_line_h:.0f}", zorder=4)
        trail_end = min(k + 1, len(ship_history))
        if trail_end >= 2:
            trail_x = [ship_history[i][0][0] for i in range(trail_end)]
            trail_y = [ship_history[i][0][1] for i in range(trail_end)]
            ax.plot(trail_x, trail_y, "b-", lw=2, label="Actual path", zorder=5)
        if k < len(ship_history):
            ship_pos, ship_heading = ship_history[k][0], ship_history[k][1]
            ax.plot(ship_pos[0], ship_pos[1], "ko", markersize=6, zorder=6)
            hull = create_ship_hull_at_position_heading(ship_length, ship_beam, ship_pos, ship_heading)
            ax.add_patch(MplPolygon(hull, closed=True, fc="orange", ec="black", lw=1.5, alpha=0.8, zorder=6))
        if len(ship_history) > 0 and (draw_path is None or len(draw_path) == 0):
            ax.plot(ship_history[0][0][0], ship_history[0][0][1], "go", ms=12, label="Start", zorder=7)
        if goal_y_line_h is None:
            goal_xy = goal
            if goal_xy is None and path is not None and len(path) > 0:
                goal_xy = np.asarray(path[-1])[:2]
            if goal_xy is not None:
                g = np.asarray(goal_xy).ravel()[:2]
                ax.plot(g[0], g[1], "mx", ms=14, mew=3, zorder=6, label="Goal")
        ax.set_xlim(x_min_ax, x_max_ax)
        ax.set_ylim(y_min_ax, y_max_ax)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"Costmap + ship trajectory\nFrame {k}/{n_frames-1}")
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
    ocean_current=None,
    path_snapshots=None,
    goal_y_line=None,
    max_fracture_thickness=15.0,
    initial_floe_of=None,
):
    """Animate ice-ship interaction in 2D. n_workers>1 uses parallel rendering (faster). frame_skip>1 renders every Nth frame.
    view_bounds: optional (x_min, x_max, y_min, y_max) to fix the visible domain (e.g. show full grid); if None, auto from ship + ice.
    ocean_current: OceanCurrentModel or None; when set, draws a quiver plot of the current field behind the ice.
    path_snapshots: optional list of (frame_idx, path Nx3); when set, draws the remaining planned path from the ship (updates at replan).
    goal_y_line: optional y [m]; when set, draws a horizontal goal line (lime dashed) like in the costmap movie."""
    n_frames = len(positions_history)
    frames_to_render = list(range(0, n_frames, frame_skip))
    if frames_to_render[-1] != n_frames - 1:
        frames_to_render.append(n_frames - 1)
    n_render = len(frames_to_render)
    n_elements = len(polygons)
    # When all ice is thick, draw each component rigidly so the movie never shows gaps (fixes any sim drift).
    positions_history, rotations_history = _rigid_correct_positions_if_thick_ice(
        positions_history, bonds, thicknesses, max_fracture_thickness, n_elements, initial_floe_of=initial_floe_of, rotations_history=rotations_history
    )
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

    # When thick-ice rigid display is active, draw one merged polygon per floe (no internal cell edges).
    merged_floe_display_per_frame = None
    if (initial_floe_of is not None and thicknesses is not None and max_fracture_thickness is not None
            and len(thicknesses) == n_elements and np.all(thicknesses > max_fracture_thickness)):
        merged_floe_display_per_frame, _ = _build_merged_floe_polygons_per_frame(
            local_polys, positions_history, rotations_history, initial_floe_of, ice_colors
        )
        if merged_floe_display_per_frame is not None:
            print(f"  [Thick-ice] Drawing {len(merged_floe_display_per_frame[0])} floes as merged polygons (no internal edges)", flush=True)

    writer, out_path = _animation_writer(output_file, fps)
    use_mp4 = str(output_file).lower().endswith(".mp4") and out_path.lower().endswith(".mp4")

    _current_viz_data = None
    if ocean_current is not None and getattr(ocean_current, 'is_active', False):
        _current_viz_data = ocean_current.get_field_for_viz(x_min, x_max, y_min, y_max, nx=18, ny=18)

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
            "current_viz": _current_viz_data,
            "path_snapshots": list(path_snapshots) if path_snapshots else None,
            "goal_y_line": goal_y_line,
            "thicknesses": np.asarray(thicknesses) if thicknesses is not None and len(thicknesses) == n_elements else None,
            "max_fracture_thickness": max_fracture_thickness,
        }
        if merged_floe_display_per_frame is not None:
            data["merged_floe_display_per_frame"] = merged_floe_display_per_frame
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
            # Save last frame as PNG for visual verification (e.g. thick-ice no-fracture check)
            last_frame_path = os.path.splitext(out_path)[0] + "_last_frame.png"
            try:
                last_buf = _draw_2d_frame_to_array(data, n_render - 1)
                imageio.imwrite(last_frame_path, last_buf)
                print(f"  Saved last frame: {last_frame_path}", flush=True)
            except Exception as e:
                print(f"  Could not save last frame PNG: {e}", flush=True)
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

    current_viz = None
    if ocean_current is not None and getattr(ocean_current, 'is_active', False):
        current_viz = ocean_current.get_field_for_viz(x_min, x_max, y_min, y_max, nx=18, ny=18)

    def update(frame_idx):
        frame = frames_to_render[frame_idx]
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_facecolor(water_color)
        if current_viz is not None:
            cX, cY, cU, cV = current_viz
            cMag = np.sqrt(cU**2 + cV**2)
            max_mag = float(np.max(cMag)) if cMag.size > 0 else 1e-6
            arrow_scale = (0.25 * min(x_max - x_min, y_max - y_min)) / (max_mag + 1e-12)
            cU_plot, cV_plot = cU * arrow_scale, cV * arrow_scale
            ax.quiver(cX, cY, cU_plot, cV_plot, cMag,
                      cmap='cool', alpha=0.45,
                      angles='xy', scale_units='xy', scale=1,
                      zorder=0, width=0.003)
        intact = bonds_history[frame]
        n_broken = np.sum(~intact)
        if merged_floe_display_per_frame is not None:
            for poly, color in merged_floe_display_per_frame[frame]:
                patch = MplPolygon(poly, closed=True, facecolor=color, edgecolor=color, linewidth=0.5, alpha=0.95)
                ax.add_patch(patch)
        else:
            positions = positions_history[frame]
            rotations = rotations_history[frame]
            broken_neighbors = {i: set() for i in range(n_elements)}
            for b, (i, j) in enumerate(bonds):
                if not intact[b]:
                    if thicknesses is not None and len(thicknesses) > max(i, j) and max_fracture_thickness is not None:
                        if thicknesses[i] <= max_fracture_thickness and thicknesses[j] <= max_fracture_thickness:
                            broken_neighbors[i].add(j)
                            broken_neighbors[j].add(i)
                    else:
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
            if path_snapshots is not None:
                remaining = _remaining_planned_path_from_snapshots(ship_pos_frame, path_snapshots, frame)
                if remaining is not None and len(remaining) >= 2:
                    ax.plot(remaining[:, 0], remaining[:, 1], "r-", linewidth=1.5, alpha=0.9, zorder=4)
        if goal_y_line is not None:
            ax.axhline(goal_y_line, color="lime", ls="--", lw=2, label=f"Goal line y={goal_y_line:.0f}", zorder=4)
        n_broken = np.sum(~intact)
        ax.set_title(f'DPDEM Ice-Ship Simulation (MEDEA Satellite Floes)\n'
                     f'Frame {frame}/{n_frames-1} | Bonds: {n_broken}/{len(bonds)} broken', fontsize=14)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        return []

    anim = FuncAnimation(fig, update, frames=n_render, blit=False)
    print(f"Saving to {out_path}...", flush=True)
    anim.save(out_path, writer=writer)
    last_frame_path = os.path.splitext(out_path)[0] + "_last_frame.png"
    try:
        update(n_render - 1)
        fig.savefig(last_frame_path, dpi=fig.get_dpi() or 100)
        print(f"  Saved last frame: {last_frame_path}", flush=True)
    except Exception as e:
        print(f"  Could not save last frame PNG: {e}", flush=True)
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
    path_snapshots=None,
    max_fracture_thickness=15.0,
    initial_floe_of=None,
):
    """3D animation with z from submergence. n_workers>1 uses parallel rendering (faster).
    path_snapshots: optional list of (frame_idx, path Nx3); when set, draws remaining planned path at z=0."""
    vc = viz_config or get_default_visualization_config()
    n_frames = len(positions_history)
    n_elements = len(polygons)
    positions_history, rotations_history = _rigid_correct_positions_if_thick_ice(
        positions_history, bonds, thicknesses, max_fracture_thickness, n_elements, initial_floe_of=initial_floe_of, rotations_history=rotations_history
    )
    local_polys = [poly - polygon_centroid(poly) for poly in polygons]
    all_pos = np.concatenate(positions_history)
    x_min_raw = np.nanmin(all_pos[:, 0])
    x_max_raw = np.nanmax(all_pos[:, 0])
    y_min_raw = np.nanmin(all_pos[:, 1])
    y_max_raw = np.nanmax(all_pos[:, 1])
    if not np.isfinite(x_min_raw):
        x_min_raw, x_max_raw = 0.0, 100.0
    if not np.isfinite(y_min_raw):
        y_min_raw, y_max_raw = 0.0, 100.0
    x_min, x_max = float(x_min_raw - 50), float(x_max_raw + 50)
    y_min, y_max = float(y_min_raw - 50), float(y_max_raw + 50)
    if x_max <= x_min:
        x_min, x_max = x_min - 50, x_min + 50
    if y_max <= y_min:
        y_min, y_max = y_min - 50, y_min + 50
    domain_width = max(x_max - x_min, 1.0)
    domain_height = max(y_max - y_min, 1.0)
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
            "path_snapshots": list(path_snapshots) if path_snapshots else None,
            "max_fracture_thickness": max_fracture_thickness,
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
            has_breaks = any(
                not intact[b] and (thicknesses[pi] <= max_fracture_thickness and thicknesses[pj] <= max_fracture_thickness)
                for b, (pi, pj) in enumerate(bonds) if pi == i or pj == i
            )
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
        if path_snapshots is not None:
            remaining = _remaining_planned_path_from_snapshots(np.array([ship_x, ship_y]), path_snapshots, frame)
            if remaining is not None and len(remaining) >= 2:
                z_path = np.zeros(len(remaining))
                ax.plot(remaining[:, 0], remaining[:, 1], z_path, 'r-', linewidth=1.5, alpha=0.9)
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
