"""Polygon and ship-hull geometry utilities for MEDEA ice–ship simulation."""

import numpy as np


def polygon_area(verts):
    """Compute area using shoelace formula."""
    n = len(verts)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i, 0] * verts[j, 1]
        area -= verts[j, 0] * verts[i, 1]
    return abs(area) / 2


def polygon_centroid(verts):
    return np.mean(verts, axis=0)


def create_ship_hull_at_position_heading(length, beam, position, heading):
    """Create ship hull outline at position with given heading (radians). Bow faces heading direction."""
    cx, cy = position
    half_len = length / 2
    half_beam = beam / 2
    n_pts = 30

    local_x = []
    local_y = []
    for i in range(n_pts + 1):
        t = i / n_pts
        x_local = half_len - t * length
        x_rel = x_local / half_len
        if x_rel > 0.6:
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.02)
        local_x.append(x_local)
        local_y.append(half_beam * taper)
    for i in range(n_pts, -1, -1):
        t = i / n_pts
        x_local = half_len - t * length
        x_rel = x_local / half_len
        if x_rel > 0.6:
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.02)
        local_x.append(x_local)
        local_y.append(-half_beam * taper)

    local_verts = np.column_stack([local_x, local_y])
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    world_verts = local_verts @ R.T + np.array([cx, cy])
    return world_verts


def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside


def _distance_point_to_segment(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to segment (A,B). Returns (dist, closest_x, closest_y)."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c1 = wx * vx + wy * vy
    c2 = vx * vx + vy * vy
    if c2 < 1e-20:
        return np.sqrt(wx * wx + wy * wy), ax, ay
    if c1 <= 0:
        d = np.sqrt(wx * wx + wy * wy)
        return d, ax, ay
    if c1 >= c2:
        dx, dy = px - bx, py - by
        return np.sqrt(dx * dx + dy * dy), bx, by
    b = c1 / c2
    cx, cy = ax + b * vx, ay + b * vy
    dx, dy = px - cx, py - cy
    return np.sqrt(dx * dx + dy * dy), cx, cy


def closest_point_on_polygon_boundary(px, py, polygon):
    """
    Return the closest point on the closed polygon boundary to (px, py).
    polygon: Nx2 array of vertices (closed: first and last need not repeat).
    """
    poly = np.asarray(polygon)
    n = len(poly)
    best_d = np.inf
    best_qx, best_qy = px, py
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        d, qx, qy = _distance_point_to_segment(px, py, ax, ay, bx, by)
        if d < best_d:
            best_d, best_qx, best_qy = d, qx, qy
    return best_qx, best_qy


def closest_point_and_outward_normal(px, py, polygon):
    """
    Return (qx, qy, nx, ny): closest point on the closed polygon boundary to (px, py),
    and the outward unit normal at that point (perpendicular to the edge, pointing
    away from the polygon interior). polygon: Nx2 array (closed).
    """
    poly = np.asarray(polygon)
    n = len(poly)
    centroid = polygon_centroid(poly)
    best_d = np.inf
    best_qx, best_qy = px, py
    best_nx, best_ny = 0.0, 0.0
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        d, qx, qy = _distance_point_to_segment(px, py, ax, ay, bx, by)
        if d < best_d:
            # Edge vector and perpendicular (outward = right of edge in CCW order)
            ex, ey = bx - ax, by - ay
            # Perp: (ey, -ex) is 90° counterclockwise from (ex, ey)
            nx, ny = ey, -ex
            nlen = np.sqrt(nx * nx + ny * ny) + 1e-20
            nx, ny = nx / nlen, ny / nlen
            # Outward: normal should point away from centroid
            if (qx - centroid[0]) * nx + (qy - centroid[1]) * ny < 0:
                nx, ny = -nx, -ny
            best_d, best_qx, best_qy = d, qx, qy
            best_nx, best_ny = nx, ny
    return best_qx, best_qy, best_nx, best_ny


def point_inside_polygon_penetration(px, py, polygon):
    """
    When point (px, py) is inside polygon, return penetration (distance to nearest edge)
    and outward normal from that edge. If point is not inside, return (0, 0, 0).
    """
    poly = np.asarray(polygon)
    n = len(poly)
    if n < 3:
        return 0.0, 0.0, 0.0
    if not point_in_polygon((px, py), poly):
        return 0.0, 0.0, 0.0
    best_pen = np.inf
    best_nx, best_ny = 0.0, 0.0
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dist, _, _ = _distance_point_to_segment(px, py, ax, ay, bx, by)
        ex, ey = bx - ax, by - ay
        nx = ey
        ny = -ex
        nlen = np.sqrt(nx * nx + ny * ny) + 1e-20
        nx, ny = nx / nlen, ny / nlen
        cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
        dot = (px - cx) * nx + (py - cy) * ny
        if dot > 0:
            nx, ny = -nx, -ny
        if dist < best_pen:
            best_pen = dist
            best_nx, best_ny = nx, ny
    return best_pen, best_nx, best_ny


def clip_cell_to_floe(cell_verts, floe_polygon):
    """Clip a Voronoi cell to the floe boundary using shapely. Returns Nx2 array or None."""
    from shapely.geometry import Polygon as ShapelyPolygon

    try:
        cell = ShapelyPolygon(cell_verts)
        floe = ShapelyPolygon(floe_polygon)
        if not cell.is_valid:
            cell = cell.buffer(0)
        if not floe.is_valid:
            floe = floe.buffer(0)
        clipped = cell.intersection(floe)
        if clipped.is_empty:
            return None
        if clipped.geom_type == 'MultiPolygon':
            clipped = max(clipped.geoms, key=lambda p: p.area)
        if clipped.geom_type == 'Polygon' and not clipped.is_empty:
            coords = np.array(clipped.exterior.coords[:-1])
            if len(coords) >= 3:
                return coords
        return None
    except Exception:
        return None
