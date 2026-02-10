"""Tessellate satellite floes into Voronoi cells for fracture simulation."""

import numpy as np
from medea_ice_ship.geometry import polygon_area, polygon_centroid, point_in_polygon, clip_cell_to_floe


def tessellate_floe_internally(floe_polygon, cells_per_100m2=0.5, min_cells=8, seed=None):
    """
    Tessellate a floe into Voronoi cells. Returns list of (cell_polygon, centroid) tuples.
    """
    from scipy.spatial import Voronoi

    if seed is not None:
        np.random.seed(seed)

    min_xy = floe_polygon.min(axis=0)
    max_xy = floe_polygon.max(axis=0)
    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]
    floe_area = polygon_area(floe_polygon)

    n_cells = max(min_cells, int(floe_area / 100.0 * cells_per_100m2))
    n_cells = min(n_cells, 150)

    points = []
    aspect = max(0.1, width / max(height, 1))
    ny = max(2, int(np.sqrt(n_cells / aspect)))
    nx = max(2, int(n_cells / ny))
    dx = width / nx
    dy = height / ny

    for i in range(nx):
        for j in range(ny):
            x = min_xy[0] + (i + 0.5) * dx + np.random.uniform(-0.3 * dx, 0.3 * dx)
            y = min_xy[1] + (j + 0.5) * dy + np.random.uniform(-0.3 * dy, 0.3 * dy)
            if point_in_polygon((x, y), floe_polygon):
                points.append([x, y])

    target_spacing = np.sqrt(floe_area / max(n_cells, 1)) * 0.5
    max_attempts = n_cells * 100
    attempts = 0

    while len(points) < n_cells and attempts < max_attempts:
        x = np.random.uniform(min_xy[0], max_xy[0])
        y = np.random.uniform(min_xy[1], max_xy[1])
        if point_in_polygon((x, y), floe_polygon):
            too_close = False
            for px, py in points:
                if (x - px)**2 + (y - py)**2 < target_spacing**2:
                    too_close = True
                    break
            if not too_close:
                points.append([x, y])
        attempts += 1

    if len(points) < min_cells // 2:
        attempts = 0
        while len(points) < min_cells and attempts < max_attempts:
            x = np.random.uniform(min_xy[0], max_xy[0])
            y = np.random.uniform(min_xy[1], max_xy[1])
            if point_in_polygon((x, y), floe_polygon):
                points.append([x, y])
            attempts += 1

    if len(points) < 3:
        centroid = polygon_centroid(floe_polygon)
        return [(floe_polygon.copy(), centroid)]

    points = np.array(points)
    margin = max(width, height) * 2
    mirrors = []
    for dx_m, dy_m in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        shifted = points.copy()
        shifted[:, 0] += dx_m * margin
        shifted[:, 1] += dy_m * margin
        mirrors.append(shifted)
    all_points = np.vstack([points] + mirrors)

    try:
        vor = Voronoi(all_points)
    except Exception:
        centroid = polygon_centroid(floe_polygon)
        return [(floe_polygon.copy(), centroid)]

    cells = []
    for i in range(len(points)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        cell_verts = np.array([vor.vertices[v] for v in region])
        clipped_verts = clip_cell_to_floe(cell_verts, floe_polygon)
        if clipped_verts is None or len(clipped_verts) < 3:
            continue
        cell_centroid = polygon_centroid(clipped_verts)
        cells.append((clipped_verts, cell_centroid))

    if len(cells) == 0:
        centroid = polygon_centroid(floe_polygon)
        return [(floe_polygon.copy(), centroid)]
    return cells


def tessellate_all_floes(polygons, centroids, cells_per_100m2=0.5, seed=42):
    """
    Tessellate all satellite floes into cells.
    Returns (cell_polygons, cell_centroids, parent_floe_ids).
    """
    all_cells = []
    all_centroids = []
    floe_ids = []
    np.random.seed(seed)

    for floe_idx, poly in enumerate(polygons):
        floe_area = polygon_area(poly)
        cells = tessellate_floe_internally(poly, cells_per_100m2=cells_per_100m2, seed=None)
        print(f"    Floe {floe_idx}: area={floe_area:.0f}m², cells={len(cells)}", flush=True)
        for cell_poly, cell_centroid in cells:
            all_cells.append(cell_poly)
            all_centroids.append(cell_centroid)
            floe_ids.append(floe_idx)

    return all_cells, np.array(all_centroids), np.array(floe_ids)
