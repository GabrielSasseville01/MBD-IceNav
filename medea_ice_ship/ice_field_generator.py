"""
Random ice field generator inspired by AUTO-IceNav's generate_rand_exp.py.

Generates ice fields at specified concentrations using random convex polygon
placement with overlap rejection.  Polygon shapes follow the CGLab convex
generation algorithm (same as AUTO-IceNav).  No external dependencies beyond
numpy and shapely (already in DPDEM).

Typical usage:
    fields = generate_ice_fields(
        concentrations=[0.2, 0.3, 0.4, 0.5],
        map_shape=(1400, 1400),
        seed=42,
    )
    # fields[0.3] -> list of dicts with 'polygons', 'centroids', 'thicknesses', ...
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

# ── ice material constants (AUTO-IceNav defaults) ──────────────────────────
ICE_DENSITY = 900.0   # kg/m³
ICE_THICKNESS_DEFAULT = 1.0  # m


def _random_convex_polygon(
    diameter: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    num_vertices_range: Tuple[int, int] = (10, 20),
    final_vertices_range: Optional[Tuple[int, int]] = (5, 12),
    rng: np.random.RandomState | None = None,
) -> Optional[np.ndarray]:
    """Generate a random convex polygon inscribed in a circle of *diameter*.

    Algorithm: https://cglab.ca/~sander/misc/ConvexGeneration/convex.html
    (same as AUTO-IceNav ship_ice_planner.geometry.polygon.generate_polygon).
    """
    if rng is None:
        rng = np.random.RandomState()
    n = rng.randint(*num_vertices_range)
    xs = sorted(rng.uniform(0, diameter, size=n))
    ys = sorted(rng.uniform(0, diameter, size=n))

    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = ys[0], ys[-1]

    last_top = x_min
    last_bot = x_min
    x_vec = []
    for i in range(1, n - 1):
        if rng.random() < 0.5:
            x_vec.append(xs[i] - last_top)
            last_top = xs[i]
        else:
            x_vec.append(last_bot - xs[i])
            last_bot = xs[i]
    x_vec.append(x_max - last_top)
    x_vec.append(last_bot - x_max)

    last_left = y_min
    last_right = y_min
    y_vec = []
    for i in range(1, n - 1):
        if rng.random() < 0.5:
            y_vec.append(ys[i] - last_left)
            last_left = ys[i]
        else:
            y_vec.append(last_right - ys[i])
            last_right = ys[i]
    y_vec.append(y_max - last_left)
    y_vec.append(last_right - y_max)
    rng.shuffle(y_vec)

    pairs = sorted(zip(x_vec, y_vec), key=lambda p: np.arctan2(p[0], p[1]))
    x, y = 0.0, 0.0
    min_x = 0.0
    min_y = 0.0
    points = []
    for dx, dy in pairs:
        points.append((x, y))
        x += dx
        y += dy
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    pts = np.array(points) + np.array([xs[0] - min_x, ys[0] - min_y])
    # centre at origin
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    pts[:, 0] += origin[0] - cx
    pts[:, 1] += origin[1] - cy

    if final_vertices_range is not None:
        nv = rng.randint(*final_vertices_range)
        if len(pts) > nv:
            step = max(1, len(pts) // nv)
            pts = pts[::step]

    if len(pts) < 3:
        return None
    return pts


def _poly_area(verts: np.ndarray) -> float:
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _polygons_overlap(p1: np.ndarray, p2: np.ndarray) -> bool:
    """Fast approximate overlap via Shapely."""
    try:
        sp1 = ShapelyPolygon(p1)
        sp2 = ShapelyPolygon(p2)
        if not sp1.is_valid or not sp2.is_valid:
            return True
        return sp1.intersects(sp2)
    except Exception:
        return True


def generate_ice_field(
    concentration: float,
    map_shape: Tuple[float, float] = (1400.0, 1400.0),
    ice_start_y: float = 0.0,
    min_radius: float = 5.0,
    max_radius: float = 80.0,
    min_area: float = 50.0,
    thickness_range: Tuple[float, float] = (0.3, 3.5),
    max_attempts: int = 50000,
    tol: float = 0.02,
    seed: int = 42,
) -> Dict:
    """
    Generate a single random ice field at a target concentration.

    The field spans x ∈ [-map_shape[1]/2, map_shape[1]/2],
                    y ∈ [ice_start_y, ice_start_y + map_shape[0]].
    Concentration = sum(polygon areas) / total ice domain area.

    Returns dict with keys:
        polygons:      list of Nx2 vertex arrays
        centroids:     Nx2 array
        thicknesses:   length-N array
        concentration: actual achieved concentration
        map_shape:     (length, width)
        domain:        (x_min, x_max, y_min, y_max)
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    length, width = map_shape
    x_lo, x_hi = -width / 2, width / 2
    y_lo, y_hi = ice_start_y, ice_start_y + length
    total_area = length * width

    polygons: List[np.ndarray] = []
    placed_shapely: List[ShapelyPolygon] = []
    total_poly_area = 0.0
    desired_area = concentration * total_area

    for _ in range(max_attempts):
        if total_poly_area >= desired_area - tol * total_area:
            break

        r = rng.uniform(min_radius, max_radius)
        cx = rng.uniform(x_lo + r, x_hi - r)
        cy = rng.uniform(y_lo + r, y_hi - r)

        verts = _random_convex_polygon(
            diameter=r * 2, origin=(cx, cy), rng=rng,
        )
        if verts is None:
            continue

        # clip to domain
        verts[:, 0] = np.clip(verts[:, 0], x_lo, x_hi)
        verts[:, 1] = np.clip(verts[:, 1], y_lo, y_hi)

        area = _poly_area(verts)
        if area < min_area:
            continue

        # would overshoot concentration
        if total_poly_area + area > desired_area + tol * total_area:
            continue

        try:
            sp = ShapelyPolygon(verts)
            if not sp.is_valid:
                continue
        except Exception:
            continue

        overlap = False
        for existing in placed_shapely:
            if sp.intersects(existing):
                overlap = True
                break
        if overlap:
            continue

        polygons.append(verts)
        placed_shapely.append(sp)
        total_poly_area += area

    actual_conc = total_poly_area / total_area
    centroids = np.array([p.mean(axis=0) for p in polygons])

    # Assign thickness with structure: correlate with floe size and add spatial
    # variation so the field has natural thick/thin regions (corridors).
    n_floes = len(polygons)
    if n_floes > 0:
        areas = np.array([_poly_area(p) for p in polygons])
        # Size component: larger floes → thicker (normalised 0–1)
        log_areas = np.log1p(areas)
        size_frac = (log_areas - log_areas.min()) / max(log_areas.max() - log_areas.min(), 1e-6)

        # Spatial component: Perlin-like low-frequency noise via random blobs.
        # Pick a few random "thick centres"; distance to nearest centre controls
        # a smooth spatial thickness bias.
        n_blobs = max(3, n_floes // 15)
        blob_cx = rng.uniform(x_lo, x_hi, size=n_blobs)
        blob_cy = rng.uniform(y_lo, y_hi, size=n_blobs)
        blob_r = rng.uniform(0.15, 0.35, size=n_blobs) * min(width, length)
        spatial_frac = np.zeros(n_floes)
        for i in range(n_floes):
            fx, fy = centroids[i]
            dists = np.sqrt((blob_cx - fx) ** 2 + (blob_cy - fy) ** 2)
            influence = np.sum(np.exp(-0.5 * (dists / blob_r) ** 2))
            spatial_frac[i] = influence
        spatial_frac = (spatial_frac - spatial_frac.min()) / max(spatial_frac.max() - spatial_frac.min(), 1e-6)

        # Combine: 40% size, 40% spatial, 20% random noise
        combined = 0.4 * size_frac + 0.4 * spatial_frac + 0.2 * rng.uniform(0, 1, size=n_floes)
        combined = np.clip(combined, 0.0, 1.0)

        t_lo, t_hi = thickness_range
        thicknesses = t_lo + combined * (t_hi - t_lo)
    else:
        thicknesses = np.array([], dtype=float)

    return {
        "polygons": polygons,
        "centroids": centroids,
        "thicknesses": thicknesses,
        "concentration": actual_conc,
        "map_shape": map_shape,
        "domain": (x_lo, x_hi, y_lo, y_hi),
    }


def generate_ice_fields(
    concentrations: Sequence[float] = (0.2, 0.3, 0.4, 0.5),
    n_per_concentration: int = 1,
    map_shape: Tuple[float, float] = (1400.0, 1400.0),
    seed: int = 42,
    **kwargs,
) -> Dict[float, List[Dict]]:
    """Generate multiple ice fields at several concentrations.

    Returns {concentration: [field_dict, ...]} where each field_dict has the
    same schema as generate_ice_field().
    """
    result: Dict[float, List[Dict]] = {}
    for conc in concentrations:
        fields = []
        for i in range(n_per_concentration):
            field_seed = seed + int(conc * 1000) + i
            field = generate_ice_field(
                concentration=conc,
                map_shape=map_shape,
                seed=field_seed,
                **kwargs,
            )
            fields.append(field)
            print(
                f"  Concentration {conc:.0%} trial {i}: "
                f"{len(field['polygons'])} floes, "
                f"actual {field['concentration']:.1%}",
                flush=True,
            )
        result[conc] = fields
    return result
