"""
Geometry module for DPDEM.

Handles Voronoi tessellation, polyhedral extrusion, dilated polyhedra
(Minkowski sums), mesh utilities, and image-based floe extraction.
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

# Image-based floe extraction (optional OpenCV dependency)
try:
    from .image_floes import (
        segment_ice_floes,
        extract_floe_polygons,
        load_floes_from_pickle,
        apply_scale_calibration,
        process_image_to_polygons,
        HAS_OPENCV,
    )
except ImportError:
    HAS_OPENCV = False
