"""
Geometry module for DPDEM.

Handles Voronoi tessellation, polyhedral extrusion, dilated polyhedra
(Minkowski sums), and mesh utilities.
"""

from .voronoi import generate_voronoi_polygons, clip_voronoi_to_domain
from .polyhedra import (
    extrude_polygon_to_polyhedron,
    compute_polyhedron_volume,
    compute_polyhedron_centroid,
    compute_polyhedron_inertia_tensor,
)
from .support import support_dilated_polyhedron, support_polyhedron
from .envelope import (
    signed_distance_to_polyhedron,
    contact_potential,
    compute_contact_overlap,
)
from .mesh import (
    triangulate_polyhedron_surface,
    compute_triangle_area,
    compute_triangle_normal,
)
