"""
Ice floe extraction from satellite imagery.

This module provides functions to extract ice floe polygon shapes from
satellite images using watershed segmentation and contour detection.
Adapted from AUTO-IceNav (GabrielSasseville01/AUTO-IceNav) for use with
DPDEM's realistic 3D ice floe model.
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

# OpenCV import with fallback for environments without it
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None


__all__ = [
    'segment_ice_floes',
    'extract_floe_polygons',
    'load_floes_from_pickle',
    'apply_scale_calibration',
    'process_image_to_polygons',
    'HAS_OPENCV',
]


def _check_opencv():
    """Raise ImportError if OpenCV is not available."""
    if not HAS_OPENCV:
        raise ImportError(
            "OpenCV is required for image processing. "
            "Install it with: pip install opencv-python"
        )


def segment_ice_floes(
    image: np.ndarray,
    dt_threshold: float = 0.2,
    morph_kernel_size: int = 3,
    morph_iterations: int = 2,
    dilate_iterations: int = 3,
) -> np.ndarray:
    """
    Segment ice floes from an image using the watershed algorithm.
    
    This function uses OpenCV's watershed algorithm to separate individual
    ice floes from a satellite or overhead image. The algorithm:
    1. Converts to grayscale and applies OTSU thresholding
    2. Uses morphological operations to remove noise
    3. Applies distance transform to find sure foreground
    4. Uses watershed to segment individual floes
    
    Args:
        image: Input BGR image as numpy array, shape (H, W, 3)
        dt_threshold: Distance transform threshold factor (0-1).
            Higher values require more separation between floes.
        morph_kernel_size: Size of morphological kernel for opening/dilation.
        morph_iterations: Number of iterations for morphological opening.
        dilate_iterations: Number of iterations for background dilation.
    
    Returns:
        markers: Integer array where each unique positive value represents
            a distinct ice floe. Value 1 is background (water), -1 marks
            watershed boundaries between floes.
    
    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If image has wrong shape or dtype.
    
    Example:
        >>> image = cv2.imread("satellite.png")
        >>> markers = segment_ice_floes(image, dt_threshold=0.3)
        >>> n_floes = len(np.unique(markers)) - 2  # Exclude background and boundaries
    """
    _check_opencv()
    
    if image is None or len(image.shape) != 3:
        raise ValueError("Input must be a 3-channel BGR image")
    
    img = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # OTSU thresholding - automatically determines optimal threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    
    # Sure background - dilate the opening
    sure_bg = cv2.dilate(opening, kernel, iterations=dilate_iterations)
    
    # Distance transform to find sure foreground
    dist_transform = cv2.distanceTransform(
        src=opening, distanceType=cv2.DIST_L2, maskSize=5
    )
    
    # Threshold the distance transform
    _, sure_fg = cv2.threshold(
        dist_transform, dt_threshold * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region - difference between sure background and foreground
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label connected components in sure foreground
    n_labels, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels so background is not 0 but 1
    markers = markers + 1
    
    # Mark unknown region with 0
    markers[unknown == 255] = 0
    
    # Apply watershed algorithm
    markers = cv2.watershed(img, markers)
    
    return markers


def extract_floe_polygons(
    image: np.ndarray,
    markers: Optional[np.ndarray] = None,
    min_area: float = 50.0,
    epsilon_factor: float = 0.005,
    max_vertices: Optional[int] = None,
) -> List[Dict]:
    """
    Extract polygon representations of ice floes from a segmented image.
    
    Given an image (either pre-segmented with unique colors per floe, or
    raw with markers from watershed), extracts the contours of each floe
    and simplifies them to polygon representations.
    
    Args:
        image: Input BGR image. If markers is None, expects a pre-segmented
            image where each floe has a unique color.
        markers: Optional watershed markers array. If provided, creates masks
            from the markers instead of using color-based segmentation.
        min_area: Minimum contour area in pixels to keep. Smaller contours
            are filtered out.
        epsilon_factor: Controls polygon simplification. Smaller values keep
            more detail. The actual epsilon is epsilon_factor * arc_length.
        max_vertices: Maximum number of vertices per polygon. If specified,
            further simplifies polygons exceeding this limit.
    
    Returns:
        List of dictionaries, each containing:
            - 'vertices': (N, 2) array of polygon vertices in pixel coordinates
            - 'area': Polygon area in pixels squared
            - 'centre': (x, y) tuple of centroid coordinates
            - 'radius': Maximum distance from centre to any vertex
    
    Raises:
        ImportError: If OpenCV is not installed.
    
    Example:
        >>> markers = segment_ice_floes(image)
        >>> floes = extract_floe_polygons(image, markers, min_area=100)
        >>> print(f"Found {len(floes)} ice floes")
    """
    _check_opencv()
    
    all_polygons = []
    
    if markers is not None:
        # Extract polygons from watershed markers
        unique_markers = np.unique(markers)
        # Filter out background (1) and boundaries (-1)
        floe_markers = unique_markers[(unique_markers > 1)]
        
        for marker_id in floe_markers:
            # Create binary mask for this floe
            mask = np.uint8(markers == marker_id) * 255
            
            polygons = _extract_polygons_from_mask(
                mask, min_area, epsilon_factor, max_vertices
            )
            all_polygons.extend(polygons)
    else:
        # Extract from pre-segmented image (unique colors)
        image_uint8 = image.astype(np.uint8)
        pixels = image_uint8.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        # Filter out black (water/background)
        unique_colors = unique_colors[~np.all(unique_colors == 0, axis=1)]
        
        for color in unique_colors:
            # Create binary mask using pure numpy (cross-platform consistent)
            color = np.array(color, dtype=np.uint8)
            mask = np.all(image_uint8 == color, axis=2).astype(np.uint8) * 255
            
            polygons = _extract_polygons_from_mask(
                mask, min_area, epsilon_factor, max_vertices
            )
            all_polygons.extend(polygons)
    
    return all_polygons


def _extract_polygons_from_mask(
    mask: np.ndarray,
    min_area: float,
    epsilon_factor: float,
    max_vertices: Optional[int],
) -> List[Dict]:
    """Extract polygon(s) from a binary mask."""
    polygons = []
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        # Ensure contiguous array
        contour = np.ascontiguousarray(contour, dtype=np.int32)
        
        # Filter by area
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Simplify contour
        arc_length = cv2.arcLength(contour, True)
        epsilon = max(epsilon_factor * arc_length, 0.1)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Further simplify if max_vertices specified
        if max_vertices is not None and len(approx) > max_vertices:
            # Increase epsilon iteratively until we're under the limit
            for mult in [2, 4, 8, 16]:
                approx = cv2.approxPolyDP(contour, epsilon * mult, True)
                if len(approx) <= max_vertices:
                    break
        
        # Need at least 3 vertices for a polygon
        if len(approx) < 3:
            continue
        
        # Extract vertices as float64
        vertices = approx[:, 0, :].astype(np.float64)
        
        # Compute centroid
        centre = tuple(np.mean(vertices, axis=0))
        
        # Compute radius (max distance from centre)
        distances = np.linalg.norm(vertices - centre, axis=1)
        radius = float(np.max(distances))
        
        # Compute area using shoelace formula
        polygon_area = _polygon_area(vertices)
        
        polygons.append({
            'vertices': vertices,
            'area': polygon_area,
            'centre': centre,
            'radius': radius,
        })
    
    return polygons


def _polygon_area(vertices: np.ndarray) -> float:
    """Calculate polygon area using the shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def load_floes_from_pickle(path: Union[str, Path]) -> List[Dict]:
    """
    Load ice floe polygons from an AUTO-IceNav pickle file.
    
    AUTO-IceNav stores extracted ice floe polygons in pickle format with
    the same dictionary structure used by extract_floe_polygons().
    
    Args:
        path: Path to the pickle file containing ice floe data.
    
    Returns:
        List of floe dictionaries with 'vertices', 'area', 'centre', 'radius'.
    
    Raises:
        FileNotFoundError: If the pickle file doesn't exist.
        ValueError: If the pickle file has unexpected format.
    
    Example:
        >>> floes = load_floes_from_pickle("ice_floes_MEDEA.pkl")
        >>> print(f"Loaded {len(floes)} floes")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Validate format
    if not isinstance(data, list):
        raise ValueError(f"Expected list of floe dictionaries, got {type(data)}")
    
    if len(data) > 0:
        required_keys = {'vertices'}
        if not required_keys.issubset(data[0].keys()):
            raise ValueError(
                f"Floe dictionary missing required keys. "
                f"Expected at least {required_keys}, got {data[0].keys()}"
            )
    
    # Ensure vertices are numpy arrays and add missing fields
    normalized = []
    for floe in data:
        vertices = np.asarray(floe['vertices'], dtype=np.float64)
        
        # Compute missing fields if needed
        if 'centre' not in floe:
            centre = tuple(np.mean(vertices, axis=0))
        else:
            centre = floe['centre']
        
        if 'radius' not in floe:
            distances = np.linalg.norm(vertices - np.array(centre), axis=1)
            radius = float(np.max(distances))
        else:
            radius = floe['radius']
        
        if 'area' not in floe:
            area = _polygon_area(vertices)
        else:
            area = floe['area']
        
        normalized.append({
            'vertices': vertices,
            'area': area,
            'centre': centre,
            'radius': radius,
        })
    
    return normalized


def apply_scale_calibration(
    floes: List[Dict],
    meters_per_pixel: float,
    domain_offset: Tuple[float, float] = (0.0, 0.0),
    center_domain: bool = False,
) -> List[np.ndarray]:
    """
    Convert floe polygons from pixel coordinates to world coordinates (meters).
    
    Applies scale transformation and optional offset/centering to convert
    image-space polygons to simulation-ready world coordinates.
    
    Args:
        floes: List of floe dictionaries from extract_floe_polygons() or
            load_floes_from_pickle().
        meters_per_pixel: Scale factor - how many meters each pixel represents.
        domain_offset: (x, y) offset to add to all coordinates after scaling.
            Useful for positioning the ice field in the simulation domain.
        center_domain: If True, centers the floe field at the origin by
            computing the bounding box and offsetting accordingly.
    
    Returns:
        List of (N, 2) numpy arrays containing polygon vertices in meters.
    
    Example:
        >>> floes = extract_floe_polygons(image, markers)
        >>> # Convert with 2.5 meters per pixel
        >>> polygons_m = apply_scale_calibration(floes, meters_per_pixel=2.5)
    """
    if len(floes) == 0:
        return []
    
    # Extract and scale vertices
    scaled_polygons = []
    for floe in floes:
        vertices = floe['vertices'].copy()
        vertices *= meters_per_pixel
        scaled_polygons.append(vertices)
    
    # Compute bounding box if centering
    if center_domain:
        all_vertices = np.vstack(scaled_polygons)
        min_coords = np.min(all_vertices, axis=0)
        max_coords = np.max(all_vertices, axis=0)
        center = (min_coords + max_coords) / 2.0
        
        for i in range(len(scaled_polygons)):
            scaled_polygons[i] -= center
    
    # Apply offset
    if domain_offset != (0.0, 0.0):
        offset = np.array(domain_offset)
        for i in range(len(scaled_polygons)):
            scaled_polygons[i] += offset
    
    return scaled_polygons


def process_image_to_polygons(
    image_path: Union[str, Path],
    meters_per_pixel: float,
    crop_bounds: Optional[Tuple[int, int, int, int]] = None,
    dt_threshold: float = 0.2,
    min_area_pixels: float = 50.0,
    epsilon_factor: float = 0.005,
    center_domain: bool = True,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Complete pipeline: load image, segment, extract, and scale polygons.
    
    This is a convenience function that runs the full image processing
    pipeline from an image file to world-coordinate polygon arrays ready
    for DPDEM simulation.
    
    Args:
        image_path: Path to the satellite/overhead image file.
        meters_per_pixel: Scale factor for coordinate conversion.
        crop_bounds: Optional (x, y, width, height) to crop the image.
        dt_threshold: Distance transform threshold for watershed.
        min_area_pixels: Minimum floe area in pixels to keep.
        epsilon_factor: Polygon simplification factor.
        center_domain: If True, centers the resulting domain at origin.
    
    Returns:
        Tuple of:
            - List of (N, 2) polygon vertex arrays in meters
            - Dict of metadata including domain size, number of floes, etc.
    
    Raises:
        ImportError: If OpenCV is not installed.
        FileNotFoundError: If image file doesn't exist.
    
    Example:
        >>> polygons, metadata = process_image_to_polygons(
        ...     "satellite.png",
        ...     meters_per_pixel=2.5,
        ...     crop_bounds=(0, 0, 500, 500),
        ... )
        >>> print(f"Domain size: {metadata['domain_size']} meters")
    """
    _check_opencv()
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Crop if specified
    if crop_bounds is not None:
        x, y, w, h = crop_bounds
        image = image[y:y+h, x:x+w, :]
    
    image_shape = image.shape[:2]
    
    # Segment
    markers = segment_ice_floes(image, dt_threshold=dt_threshold)
    
    # Extract polygons
    floes = extract_floe_polygons(
        image, markers,
        min_area=min_area_pixels,
        epsilon_factor=epsilon_factor,
    )
    
    # Scale to world coordinates
    polygons = apply_scale_calibration(
        floes,
        meters_per_pixel=meters_per_pixel,
        center_domain=center_domain,
    )
    
    # Compute metadata
    if len(polygons) > 0:
        all_vertices = np.vstack(polygons)
        min_coords = np.min(all_vertices, axis=0)
        max_coords = np.max(all_vertices, axis=0)
        domain_size = tuple(max_coords - min_coords)
    else:
        domain_size = (
            image_shape[1] * meters_per_pixel,
            image_shape[0] * meters_per_pixel
        )
    
    # Compute total and average areas
    areas = [_polygon_area(p) for p in polygons]
    total_area = sum(areas) if areas else 0.0
    avg_area = np.mean(areas) if areas else 0.0
    
    metadata = {
        'n_floes': len(polygons),
        'domain_size': domain_size,
        'meters_per_pixel': meters_per_pixel,
        'image_shape': image_shape,
        'total_ice_area': total_area,
        'average_floe_area': avg_area,
        'crop_bounds': crop_bounds,
    }
    
    return polygons, metadata


def get_ice_binary_mask(markers: np.ndarray) -> np.ndarray:
    """
    Convert watershed markers to binary ice/water mask.
    
    Args:
        markers: Watershed markers from segment_ice_floes().
    
    Returns:
        Boolean array where True indicates ice and False indicates water.
    """
    # Background is 1, boundaries are -1, floes are > 1
    binary = markers.copy()
    binary[binary != 1] = 0
    return ~binary.astype(bool)


def get_ice_edges(markers: np.ndarray) -> np.ndarray:
    """
    Extract ice floe edges from watershed markers.
    
    Args:
        markers: Watershed markers from segment_ice_floes().
    
    Returns:
        Boolean array where True indicates an ice edge (floe boundary).
    """
    edges = markers.copy()
    edges[edges != -1] = 0
    edges[edges == -1] = 1
    return edges.astype(bool)


def visualize_segmentation(
    image: np.ndarray,
    markers: np.ndarray,
    floes: Optional[List[Dict]] = None,
    show_boundaries: bool = True,
    boundary_color: Tuple[int, int, int] = (0, 0, 255),
    polygon_color: Tuple[int, int, int] = (0, 255, 0),
    polygon_thickness: int = 2,
) -> np.ndarray:
    """
    Create a visualization of the segmentation results.
    
    Args:
        image: Original BGR image.
        markers: Watershed markers from segment_ice_floes().
        floes: Optional list of extracted floe polygons to overlay.
        show_boundaries: If True, highlight watershed boundaries.
        boundary_color: BGR color for boundaries.
        polygon_color: BGR color for polygon outlines.
        polygon_thickness: Line thickness for polygon outlines.
    
    Returns:
        BGR image with segmentation visualization overlaid.
    """
    _check_opencv()
    
    output = image.copy()
    
    # Draw watershed boundaries
    if show_boundaries:
        output[markers == -1] = boundary_color
    
    # Draw extracted polygons
    if floes is not None:
        for floe in floes:
            vertices = floe['vertices'].astype(np.int32)
            cv2.polylines(output, [vertices], True, polygon_color, polygon_thickness)
    
    return output
