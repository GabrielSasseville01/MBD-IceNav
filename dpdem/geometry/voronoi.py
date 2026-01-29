"""
Voronoi tessellation for generating ice element geometries.

Uses SciPy at initialization time (not JAX-native, but only runs once).
"""

import numpy as np
from scipy.spatial import Voronoi
from typing import List, Tuple, Optional
import jax.numpy as jnp


def generate_voronoi_polygons(
    n_elements: int,
    domain_size: Tuple[float, float],
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate random Voronoi polygons in a 2D domain.
    
    Args:
        n_elements: Approximate number of elements (actual may be fewer due to boundary)
        domain_size: (width, height) of the domain [m]
        seed: Random seed for reproducibility
    
    Returns:
        polygons: List of (n_vertices, 2) arrays, one per polygon
        centroids: (n_polygons, 2) array of polygon centroids
    """
    if seed is not None:
        np.random.seed(seed)
    
    width, height = domain_size
    
    # Generate random seed points with margin to avoid edge effects
    margin = max(width, height) * 0.1
    points = np.random.rand(n_elements, 2)
    points[:, 0] = points[:, 0] * width
    points[:, 1] = points[:, 1] * height
    
    # Add mirror points around the boundary to handle edge cells
    # This ensures all interior cells are bounded
    mirror_points = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        shifted = points.copy()
        shifted[:, 0] += dx * width
        shifted[:, 1] += dy * height
        mirror_points.append(shifted)
    
    all_points = np.vstack([points] + mirror_points)
    
    # Compute Voronoi tessellation
    vor = Voronoi(all_points)
    
    # Extract polygons for original points only
    polygons = []
    centroids = []
    
    for i in range(n_elements):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        # Skip unbounded regions (shouldn't happen with mirrors, but check)
        if -1 in region or len(region) < 3:
            continue
        
        # Get vertices in order
        vertices = vor.vertices[region]
        
        # Clip to domain
        vertices_clipped = clip_polygon_to_rectangle(
            vertices, 0, width, 0, height
        )
        
        if vertices_clipped is not None and len(vertices_clipped) >= 3:
            polygons.append(vertices_clipped)
            centroid = np.mean(vertices_clipped, axis=0)
            centroids.append(centroid)
    
    return polygons, np.array(centroids)


def clip_voronoi_to_domain(
    polygons: List[np.ndarray],
    domain_bounds: Tuple[float, float, float, float],
) -> List[np.ndarray]:
    """
    Clip polygons to rectangular domain bounds.
    
    Args:
        polygons: List of (n_vertices, 2) polygon vertex arrays
        domain_bounds: (x_min, x_max, y_min, y_max)
    
    Returns:
        List of clipped polygons
    """
    x_min, x_max, y_min, y_max = domain_bounds
    
    clipped = []
    for poly in polygons:
        result = clip_polygon_to_rectangle(poly, x_min, x_max, y_min, y_max)
        if result is not None and len(result) >= 3:
            clipped.append(result)
    
    return clipped


def clip_polygon_to_rectangle(
    vertices: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Optional[np.ndarray]:
    """
    Clip a convex polygon to a rectangle using Sutherland-Hodgman algorithm.
    
    Args:
        vertices: (n, 2) array of polygon vertices in order
        x_min, x_max, y_min, y_max: Rectangle bounds
    
    Returns:
        Clipped polygon vertices or None if completely outside
    """
    def clip_edge(vertices, x1, y1, x2, y2):
        """Clip polygon against a single edge of the clip rectangle."""
        if len(vertices) == 0:
            return vertices
        
        output = []
        
        def inside(p):
            return (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0
        
        def intersection(p1, p2):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dx_clip = x2 - x1
            dy_clip = y2 - y1
            
            denom = dx * dy_clip - dy * dx_clip
            if abs(denom) < 1e-10:
                return p1  # Parallel, return start point
            
            t = ((x1 - p1[0]) * dy_clip - (y1 - p1[1]) * dx_clip) / denom
            return np.array([p1[0] + t * dx, p1[1] + t * dy])
        
        prev = vertices[-1]
        prev_inside = inside(prev)
        
        for curr in vertices:
            curr_inside = inside(curr)
            
            if curr_inside:
                if not prev_inside:
                    output.append(intersection(prev, curr))
                output.append(curr)
            elif prev_inside:
                output.append(intersection(prev, curr))
            
            prev = curr
            prev_inside = curr_inside
        
        return output
    
    # Clip against each edge of the rectangle
    output = list(vertices)
    
    # Left edge (x = x_min)
    output = clip_edge(output, x_min, y_min, x_min, y_max)
    # Right edge (x = x_max)
    output = clip_edge(output, x_max, y_max, x_max, y_min)
    # Bottom edge (y = y_min)
    output = clip_edge(output, x_max, y_min, x_min, y_min)
    # Top edge (y = y_max)
    output = clip_edge(output, x_min, y_max, x_max, y_max)
    
    if len(output) < 3:
        return None
    
    return np.array(output)


def compute_polygon_area(vertices: np.ndarray) -> float:
    """
    Compute area of a 2D polygon using the shoelace formula.
    
    Args:
        vertices: (n, 2) array of polygon vertices in order
    
    Returns:
        Polygon area (positive for counter-clockwise ordering)
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def compute_polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """
    Compute centroid of a 2D polygon.
    
    Args:
        vertices: (n, 2) array of polygon vertices in order
    
    Returns:
        (2,) array of centroid coordinates
    """
    n = len(vertices)
    area = compute_polygon_area(vertices)
    
    if area < 1e-10:
        return np.mean(vertices, axis=0)
    
    cx, cy = 0.0, 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
        cx += (vertices[i, 0] + vertices[j, 0]) * cross
        cy += (vertices[i, 1] + vertices[j, 1]) * cross
    
    cx /= 6.0 * area
    cy /= 6.0 * area
    
    return np.array([abs(cx), abs(cy)])


def order_polygon_vertices_ccw(vertices: np.ndarray) -> np.ndarray:
    """
    Order polygon vertices counter-clockwise around their centroid.
    
    Args:
        vertices: (n, 2) array of polygon vertices
    
    Returns:
        (n, 2) array of vertices in counter-clockwise order
    """
    centroid = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - centroid[1], 
                        vertices[:, 0] - centroid[0])
    order = np.argsort(angles)
    return vertices[order]
