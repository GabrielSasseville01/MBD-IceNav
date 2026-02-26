"""MEDEA data loading, scaling, and simple synthetic ice field creation."""

import pickle
import numpy as np


def load_medea_floes(pickle_path: str) -> tuple:
    """Load MEDEA ice floe data from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    polygons = [np.array(floe['vertices']) for floe in data]
    centroids = np.array([np.array(floe['centre']) for floe in data])
    areas = np.array([floe['area'] for floe in data])

    metadata = {
        'n_floes': len(polygons),
        'areas': areas,
        'source': 'MEDEA',
    }
    return polygons, centroids, metadata


def scale_floes_to_world(polygons, centroids, meters_per_pixel, center_domain=True,
                         subsample_region=None):
    """Convert pixel coordinates to world coordinates."""
    scaled_polygons = [p * meters_per_pixel for p in polygons]
    scaled_centroids = centroids * meters_per_pixel

    if center_domain:
        all_verts = np.vstack(scaled_polygons)
        center = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2
        scaled_polygons = [p - center for p in scaled_polygons]
        scaled_centroids = scaled_centroids - center

    if subsample_region is not None:
        x_min, x_max, y_min, y_max = subsample_region
        keep = []
        for i, c in enumerate(scaled_centroids):
            if x_min <= c[0] <= x_max and y_min <= c[1] <= y_max:
                keep.append(i)
        if len(keep) > 0:
            scaled_polygons = [scaled_polygons[i] for i in keep]
            scaled_centroids = scaled_centroids[keep]

    return scaled_polygons, scaled_centroids


def create_simple_ice_field(seed=42):
    """
    Create a minimal synthetic ice field for fast testing: 3 floes of different
    sizes. Returns (sat_polygons, sat_centroids) in world coordinates [m].
    """
    np.random.seed(seed)
    floes_spec = [
        (0.0, 0.0, 30.0, 20.0),
        (-55.0, 10.0, 15.0, 11.0),
        (50.0, -25.0, 24.0, 16.0),
    ]
    sat_polygons = []
    sat_centroids = []
    for cx, cy, w, h in floes_spec:
        poly = np.array([
            [cx - w, cy - h],
            [cx + w, cy - h],
            [cx + w, cy + h],
            [cx - w, cy + h],
        ])
        sat_polygons.append(poly)
        sat_centroids.append(np.array([cx, cy]))
    sat_centroids = np.array(sat_centroids)
    return sat_polygons, sat_centroids


def create_sparse_toy_ice_field(
    seed=42,
    domain_x=(-120.0, 120.0),
    domain_y=(-35.0, 35.0),
    n_floes=12,
    n_icebergs=3,
    iceberg_thickness=10.0,
    moderate_thickness_range=(0.5, 2.0),
    floe_size_range=(8.0, 35.0),
    min_gap=25.0,
):
    """
    Create a sparse toy ice field with varying thickness: a few thick "icebergs"
    (~10 m) and several moderate floes (0.5–2 m). For testing costmap/planning
    and fracture behaviour.

    Returns:
        sat_polygons: list of 2D polygon arrays [m]
        sat_centroids: (n_floes, 2) array [m]
        sat_thicknesses: (n_floes,) array [m] — use these instead of assign_thickness_by_area
    """
    np.random.seed(seed)
    x_min, x_max = domain_x
    y_min, y_max = domain_y
    sat_polygons = []
    sat_centroids = []
    sat_thicknesses = []

    # Decide which floes are icebergs (thick) vs moderate
    indices = np.arange(n_floes)
    np.random.shuffle(indices)
    iceberg_indices = set(indices[:n_icebergs])

    positions = []
    for _ in range(n_floes * 20):  # allow many attempts to place
        w = np.random.uniform(*floe_size_range)
        h = np.random.uniform(w * 0.4, w * 0.8)
        cx = np.random.uniform(x_min + w, x_max - w)
        cy = np.random.uniform(y_min + h, y_max - h)
        ok = True
        for (ox, oy, ow, oh) in positions:
            if abs(cx - ox) < (w + ow) / 2 + min_gap and abs(cy - oy) < (h + oh) / 2 + min_gap:
                ok = False
                break
        if ok:
            positions.append((cx, cy, w, h))
            if len(positions) >= n_floes:
                break
    if len(positions) < n_floes:
        # fallback: allow smaller gaps
        while len(positions) < n_floes:
            w = np.random.uniform(*floe_size_range)
            h = np.random.uniform(w * 0.4, w * 0.8)
            cx = np.random.uniform(x_min + w, x_max - w)
            cy = np.random.uniform(y_min + h, y_max - h)
            positions.append((cx, cy, w, h))

    for i, (cx, cy, w, h) in enumerate(positions[:n_floes]):
        poly = np.array([
            [cx - w, cy - h],
            [cx + w, cy - h],
            [cx + w, cy + h],
            [cx - w, cy + h],
        ])
        sat_polygons.append(poly)
        sat_centroids.append(np.array([cx, cy]))
        if i in iceberg_indices:
            t = iceberg_thickness
        else:
            t = np.random.uniform(*moderate_thickness_range)
        sat_thicknesses.append(t)

    sat_centroids = np.array(sat_centroids)
    sat_thicknesses = np.array(sat_thicknesses, dtype=float)
    return sat_polygons, sat_centroids, sat_thicknesses
