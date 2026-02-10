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
