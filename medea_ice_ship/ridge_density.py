"""
Ridge density map generation from satellite imagery (ported from AUTO-IceNav).

Implements methods to generate ridge density maps from optical satellite imagery,
adapting the Lensu & Similä (2022) approach for SAR to optical imagery.

Based on:
- Lensu, M. & Similä, M. (2022). "Ice Ridge Density Signatures in High-Resolution SAR Images"
  https://tc.copernicus.org/articles/16/4363/2022/
"""

import numpy as np
from typing import Optional

from scipy import ndimage
from skimage import feature

try:
    import cv2
except ImportError:
    cv2 = None


def compute_ridge_density_from_optical_image(
    image: np.ndarray,
    ice_mask: Optional[np.ndarray] = None,
    window_size: int = 25,
    method: str = "texture_variance",
    auto_scale_window: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute ridge density map from optical satellite imagery.

    Adapts the Lensu & Similä (2022) density-based approach for optical imagery.
    In optical images, ridges appear as:
    - Texture variations (rough vs smooth ice)
    - Edge density (many edges = ridging zones)
    - Shadow patterns (height variations)

    Args:
        image: Input image (grayscale or RGB). If RGB, will be converted to grayscale.
            Can be uint8 (0-255) or float (0-1). Higher bit depth images will be normalized.
        ice_mask: Binary mask where 1 = ice, 0 = water. If None, will be estimated.
        window_size: Size of local window for density computation (pixels).
        method: Method to use:
            - 'texture_variance': Local variance in texture (good for optical)
            - 'edge_density': Density of edges (ridges create many edges)
            - 'gradient_magnitude': Magnitude of gradients (height variations)
            - 'combined': Combination of all methods
        auto_scale_window: If True, scale window_size based on image resolution
            (larger images get proportionally larger windows).
        verbose: If True, print progress and statistics.

    Returns:
        Ridge density map (0-1, where 1 = maximum ridging).
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for ridge density computation")

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Normalize to 0-1 range
    gray = np.asarray(gray, dtype=np.float64)
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float64) / 255.0
    elif gray.dtype == np.uint16:
        gray = gray.astype(np.float64) / 65535.0
    elif gray.max() > 1.0:
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)

    # Auto-scale window size based on image resolution
    original_window_size = window_size
    if auto_scale_window:
        base_resolution = 1000.0
        image_resolution = np.sqrt(gray.shape[0] * gray.shape[1])
        scale_factor = image_resolution / base_resolution
        window_size = int(window_size * scale_factor)
        window_size = max(15, min(100, window_size))
        if verbose:
            print(
                f"  Auto-scaled window size: {original_window_size} → {window_size} "
                f"(scale factor: {scale_factor:.2f})"
            )

    if verbose:
        print(f"  Processing image: {gray.shape[1]}×{gray.shape[0]} pixels")
        print(f"  Window size: {window_size}×{window_size} pixels")
        print(f"  Method: {method}")

    # Estimate ice mask if not provided
    if ice_mask is None:
        if verbose:
            print("  Computing ice/water mask...")
        _, ice_mask = cv2.threshold(
            (gray * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        ice_mask = ice_mask.astype(bool)
        if verbose:
            ice_pixels = np.sum(ice_mask)
            total_pixels = ice_mask.size
            print(
                f"  Ice mask: {ice_pixels}/{total_pixels} pixels "
                f"({100 * ice_pixels / total_pixels:.1f}% ice)"
            )

    # Initialize density map
    density_map = np.zeros_like(gray)

    if method == "texture_variance" or method == "combined":
        if verbose:
            print("  Computing texture variance...")
        kernel = np.ones((window_size, window_size)) / (window_size**2)
        local_mean = ndimage.convolve(gray, kernel, mode="reflect")
        local_variance = ndimage.convolve(
            (gray - local_mean) ** 2, kernel, mode="reflect"
        )
        variance_normalized = np.clip(
            local_variance / (local_variance.max() + 1e-6), 0, 1
        )
        if verbose:
            print(
                f"  Texture variance: min={variance_normalized.min():.4f}, "
                f"max={variance_normalized.max():.4f}"
            )
        if method == "texture_variance":
            density_map = variance_normalized
        else:
            density_map = density_map + variance_normalized * 0.4

    if method == "edge_density" or method == "combined":
        if verbose:
            print("  Computing edge density...")
        edges = feature.canny(gray, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
        kernel = np.ones((window_size, window_size))
        edge_density = ndimage.convolve(edges.astype(np.float64), kernel, mode="reflect")
        edge_density = edge_density / (window_size**2)
        if verbose:
            print(
                f"  Edge density: min={edge_density.min():.4f}, "
                f"max={edge_density.max():.4f}"
            )
        if method == "edge_density":
            density_map = edge_density
        else:
            density_map = density_map + edge_density * 0.3

    if method == "gradient_magnitude" or method == "combined":
        if verbose:
            print("  Computing gradient magnitude...")
        grad_y, grad_x = np.gradient(gray)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        kernel = np.ones((window_size, window_size)) / (window_size**2)
        local_grad_density = ndimage.convolve(
            gradient_magnitude, kernel, mode="reflect"
        )
        local_grad_density = np.clip(
            local_grad_density / (local_grad_density.max() + 1e-6), 0, 1
        )
        if verbose:
            print(
                f"  Gradient density: min={local_grad_density.min():.4f}, "
                f"max={local_grad_density.max():.4f}"
            )
        if method == "gradient_magnitude":
            density_map = local_grad_density
        else:
            density_map = density_map + local_grad_density * 0.3

    # Apply ice mask (set water areas to 0)
    if verbose:
        print("  Applying ice mask and normalizing...")
    density_map[~ice_mask] = 0.0

    # Normalize final map to 0-1
    if density_map.max() > 0:
        density_map = density_map / density_map.max()

    if verbose:
        print(
            f"  Final density: min={density_map.min():.4f}, max={density_map.max():.4f}, "
            f"mean={density_map.mean():.4f}"
        )

    return density_map


def compute_ridge_density_from_concentration_grid(
    concentration: np.ndarray,
    floe_edges: Optional[np.ndarray] = None,
    window_size: int = 25,
    compression_threshold: float = 0.7,
) -> np.ndarray:
    """
    Compute ridge density map from ice concentration grid.

    Infers ridging from high concentration + compression patterns,
    floe edge density, and concentration gradients.

    Args:
        concentration: Ice concentration grid (0-1, where 1 = 100% ice).
        floe_edges: Binary mask of floe edges (optional).
        window_size: Size of local window for density computation (pixels).
        compression_threshold: Concentration threshold for compression detection.

    Returns:
        Ridge density map (0-1, where 1 = maximum ridging).
    """
    density_map = np.zeros_like(concentration)
    high_conc_mask = concentration >= compression_threshold

    grad_y, grad_x = np.gradient(concentration)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    kernel = np.ones((window_size, window_size)) / (window_size**2)
    local_grad = ndimage.convolve(gradient_magnitude, kernel, mode="reflect")
    compression_density = (concentration * local_grad) * high_conc_mask.astype(
        np.float64
    )

    if floe_edges is not None:
        edge_density = ndimage.convolve(
            floe_edges.astype(np.float64), kernel, mode="reflect"
        )
        edge_density = edge_density / (window_size**2)
        density_map = 0.6 * compression_density + 0.4 * edge_density
    else:
        density_map = compression_density

    if density_map.max() > 0:
        density_map = density_map / density_map.max()

    return density_map
