"""Bond creation, thickness assignment, and connected-component finding for MEDEA ice."""

import numpy as np


def create_proximity_bonds(centroids, radii, threshold_factor=1.5):
    """Create bonds between nearby floes/cells."""
    bonds = []
    n = len(centroids)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            threshold = (radii[i] + radii[j]) * threshold_factor
            if dist < threshold:
                bonds.append((i, j))
    return bonds


def assign_thickness_by_area(areas, min_thickness=0.5, max_thickness=3.0, power=0.3):
    """Assign ice thickness correlated with floe area."""
    area_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1)
    thickness_norm = area_norm ** power
    thicknesses = min_thickness + thickness_norm * (max_thickness - min_thickness)
    thicknesses += np.random.randn(len(areas)) * 0.1
    return np.clip(thicknesses, min_thickness, max_thickness)


def find_connected_components(n, bonds, intact):
    """Find connected components of bonded elements."""
    adj = {i: [] for i in range(n)}
    for b, (i, j) in enumerate(bonds):
        if intact[b]:
            adj[i].append(j)
            adj[j].append(i)
    visited = set()
    components = []
    for start in range(n):
        if start in visited:
            continue
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)
    return components
