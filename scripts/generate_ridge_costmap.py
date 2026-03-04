"""
Generate ridge density costmap from satellite imagery for path planning.

Ported from AUTO-IceNav (ship_ice_planner/image_process/generate_ridge_costmap.py).
Creates a ridge density costmap for the planning costmap to avoid ridges (Lensu et al. 2022).

Usage:
  # Test without real data:
  uv run python scripts/generate_ridge_costmap.py --synthetic

  # With MEDEA images (uv add gdown, then: uv run gdown 1jV6xLCtFoe4qCWu_pOy-OnXI583lJ3Q3 --output data/):
  uv run python scripts/generate_ridge_costmap.py \\
    --input data/MEDEA_fram_20100629.png \\
    --satellite data/MEDEA_fram_20100629_satellite.tif \\
    --output data/ridge_costmap_MEDEA_fram_20100629_scale05_fixed.pkl \\
    --crop_size 600 --costmap_scale 0.5 --method combined --window_size 25 --visualize

  See docs/RIDGE_COSTMAP.md for full instructions.
"""

import argparse
import pickle
import sys
from pathlib import Path

# Ensure project root is on path when run as script (scripts/generate_ridge_costmap.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from scipy.ndimage import zoom

from medea_ice_ship.ridge_density import compute_ridge_density_from_optical_image


def downsample_costmap(
    costmap: np.ndarray, target_scale: float, original_scale: float
) -> np.ndarray:
    """
    Resample a costmap to match target scale, preserving physical dimensions.

    Args:
        costmap: Original costmap (H, W)
        target_scale: Target scale (meters per pixel) for simulation
        original_scale: Original scale (meters per pixel) of input image

    Returns:
        Resampled costmap (may be upsampled or downsampled depending on scales)
    """
    scale_factor = original_scale / target_scale
    if abs(scale_factor - 1.0) < 1e-6:
        return costmap

    target_h = int(costmap.shape[0] * scale_factor)
    target_w = int(costmap.shape[1] * scale_factor)
    resampled = zoom(
        costmap, (target_h / costmap.shape[0], target_w / costmap.shape[1]), order=1
    )
    return resampled


def generate_ridge_costmap(
    input_image_path: str,
    satellite_image_path: str | None = None,
    crop_size: int = 600,
    costmap_scale: float = 1.0,
    method: str = "combined",
    window_size: int = 25,
    output_path: str | None = None,
    visualize: bool = False,
    polygon_file: str | None = None,
) -> dict:
    """
    Generate a ridge density costmap from satellite imagery.

    Args:
        input_image_path: Path to segmented PNG image (used for cropping/extent)
        satellite_image_path: Optional path to original satellite TIF (better for ridge detection)
        crop_size: Size to crop image to (square crop from top-left, pixels)
        costmap_scale: Scale for output costmap (meters per pixel). Lower = coarser
        method: Method for ridge density ('texture_variance', 'edge_density', 'gradient_magnitude', 'combined')
        window_size: Window size for ridge density computation
        output_path: Output path for pickle file
        visualize: Whether to save visualization
        polygon_file: Optional path to ice floe polygon pickle (for overlay, AUTO-IceNav style)

    Returns:
        output_data dict with ridge_costmap, scale, width, length, shape, metadata
    """
    input_pixel_scale = 1.0  # meters per pixel for segmented image

    print(f"Loading segmented image: {input_image_path}")
    segmented_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if segmented_image is None:
        raise FileNotFoundError(f"Could not load image: {input_image_path}")
    segmented_h, segmented_w = segmented_image.shape[:2]
    print(f"  Segmented image shape: {segmented_image.shape}")

    if satellite_image_path and Path(satellite_image_path).exists():
        print(f"Loading satellite image: {satellite_image_path}")
        satellite_image = cv2.imread(satellite_image_path, cv2.IMREAD_UNCHANGED)
        if satellite_image is None:
            raise ValueError(f"Could not load TIF image: {satellite_image_path}")
        print(
            f"  Loaded TIF: shape={satellite_image.shape}, dtype={satellite_image.dtype}, "
            f"min={satellite_image.min()}, max={satellite_image.max()}"
        )

        scale_h = satellite_image.shape[0] / segmented_h
        scale_w = satellite_image.shape[1] / segmented_w
        print(f"  Scale factors: {scale_h:.2f}x (height), {scale_w:.2f}x (width)")

        satellite_crop_h = int(crop_size * scale_h)
        satellite_crop_w = int(crop_size * scale_w)
        print(
            f"  Cropping satellite image to {satellite_crop_h}x{satellite_crop_w} pixels"
        )
        satellite_image = satellite_image[:satellite_crop_h, :satellite_crop_w]

        print("Computing ridge density map from satellite image...")
        ridge_density = compute_ridge_density_from_optical_image(
            satellite_image,
            window_size=window_size,
            method=method,
            auto_scale_window=True,
            verbose=True,
        )
        satellite_pixel_scale = input_pixel_scale / scale_h
    else:
        if satellite_image_path:
            print(
                f"Warning: Satellite image not found at {satellite_image_path}, using segmented image"
            )
        image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {input_image_path}")
        image = image[:crop_size, :crop_size, :]

        print("Computing ridge density map from segmented image...")
        ridge_density = compute_ridge_density_from_optical_image(
            image,
            window_size=window_size,
            method=method,
            auto_scale_window=False,
            verbose=True,
        )
        satellite_pixel_scale = input_pixel_scale

    print(f"\nRidge density map shape: {ridge_density.shape}")
    print(f"Ridge density range: [{ridge_density.min():.4f}, {ridge_density.max():.4f}]")
    print(f"Ridge density mean: {ridge_density.mean():.4f}")

    print(
        f"\nDownsampling to costmap scale: {costmap_scale} m/pixel "
        f"(from {satellite_pixel_scale} m/pixel)"
    )
    ridge_costmap = downsample_costmap(
        ridge_density, costmap_scale, satellite_pixel_scale
    )
    print(f"Downsampled costmap shape: {ridge_costmap.shape}")
    print(
        f"Downsampled density range: [{ridge_costmap.min():.4f}, {ridge_costmap.max():.4f}]"
    )

    width_m = ridge_costmap.shape[1] * costmap_scale
    length_m = ridge_costmap.shape[0] * costmap_scale

    output_data = {
        "ridge_costmap": ridge_costmap,
        "scale": costmap_scale,
        "width": width_m,
        "length": length_m,
        "shape": ridge_costmap.shape,
        "metadata": {
            "input_image": input_image_path,
            "satellite_image": satellite_image_path,
            "crop_size": crop_size,
            "method": method,
            "window_size": window_size,
            "original_shape": ridge_density.shape,
            "original_scale": satellite_pixel_scale,
            "downsample_factor": satellite_pixel_scale / costmap_scale,
        },
    }

    if output_path is None:
        input_stem = Path(input_image_path).stem
        output_path = str(Path("data") / f"ridge_costmap_{input_stem}.pkl")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"\n✓ Saved ridge costmap to: {output_path}")
    print(f"  Dimensions: {width_m:.1f} m × {length_m:.1f} m")
    print(f"  Grid size: {ridge_costmap.shape[1]} × {ridge_costmap.shape[0]} pixels")
    print(f"  Scale: {costmap_scale} m/pixel")

    if visualize:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection

        # Match AUTO-IceNav Plot convention: origin='lower', extent in meters
        # Image row 0 = top of crop; flip so row 0 = y=0 (bottom) for costmap alignment
        ridge_density_plot = np.flipud(ridge_density)
        ridge_costmap_plot = np.flipud(ridge_costmap)

        # World extent in meters (matches costmap convention)
        extent_orig = [
            0,
            ridge_density.shape[1] * satellite_pixel_scale,
            0,
            ridge_density.shape[0] * satellite_pixel_scale,
        ]
        extent_costmap = [0, width_m, 0, length_m]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        im1 = axes[0].imshow(
            ridge_density_plot,
            cmap="hot",
            origin="lower",
            extent=extent_orig,
            vmin=0,
            vmax=1,
            aspect="equal",
        )
        axes[0].set_title(
            f"Original Ridge Density Map\n{ridge_density.shape[1]}×{ridge_density.shape[0]} pixels"
        )
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        plt.colorbar(im1, ax=axes[0], label="Ridge Density")

        im2 = axes[1].imshow(
            ridge_costmap_plot,
            cmap="hot",
            origin="lower",
            extent=extent_costmap,
            vmin=0,
            vmax=1,
            aspect="equal",
        )
        axes[1].set_title(
            f"Downsampled Ridge Costmap\n{ridge_costmap.shape[1]}×{ridge_costmap.shape[0]} pixels\n"
            f"{costmap_scale} m/pixel"
        )
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("y (m)")
        plt.colorbar(im2, ax=axes[1], label="Ridge Density")

        # Optional: overlay ice polygon outlines (AUTO-IceNav style, cyan)
        if polygon_file and Path(polygon_file).exists():
            try:
                with open(polygon_file, "rb") as f:
                    polygons = pickle.load(f)
                for ax in axes:
                    patches_list = []
                    for poly in polygons:
                        verts = (
                            np.asarray(poly["vertices"])
                            if isinstance(poly, dict) and "vertices" in poly
                            else np.asarray(poly)
                        )
                        if len(verts) >= 3:
                            patches_list.append(MplPolygon(verts, closed=True, fill=False))
                    if patches_list:
                        pc = PatchCollection(
                            patches_list,
                            edgecolor="cyan",
                            facecolor="none",
                            linewidth=0.5,
                            alpha=0.7,
                        )
                        ax.add_collection(pc)
                print(f"  Overlaid {len(polygons)} ice polygons from {polygon_file}")
            except Exception as e:
                print(f"  Warning: could not load polygons from {polygon_file}: {e}")

        plt.tight_layout()

        vis_path = str(output_path).replace(".pkl", "_visualization.png")
        plt.savefig(vis_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Visualization saved to: {vis_path}")

    return output_data


def create_synthetic_test_image(path: Path, size: int = 400) -> None:
    """Create a synthetic test image with textured (ridged) regions."""
    np.random.seed(42)
    img = np.ones((size, size, 3), dtype=np.float64) * 180
    img += np.random.randint(-20, 20, (size, size, 3)).astype(np.float64)
    img[100:200, 100:300] += np.random.randint(-50, 50, (100, 200, 3)).astype(np.float64)
    img = np.clip(img, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    print(f"Created synthetic test image: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ridge density costmap from satellite imagery for path planning"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input segmented image path (used for extent/cropping)",
    )
    parser.add_argument(
        "--satellite",
        type=str,
        default=None,
        help="Optional: Original satellite TIF image path (better for ridge detection)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle file path",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=600,
        help="Size to crop image to (square crop from top-left, pixels)",
    )
    parser.add_argument(
        "--costmap_scale",
        type=float,
        default=1.0,
        help="Scale for output costmap (meters per pixel). Lower = coarser. Default: 1.0",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="combined",
        choices=["texture_variance", "edge_density", "gradient_magnitude", "combined"],
        help="Method for ridge density computation",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=25,
        help="Window size for ridge density computation (pixels)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization of ridge density maps",
    )
    parser.add_argument(
        "--polygon",
        type=str,
        default=None,
        help="Optional: path to ice floe polygon pickle for overlay (AUTO-IceNav style)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create synthetic test image and run pipeline (for testing without real data)",
    )
    args = parser.parse_args()

    if args.synthetic:
        out_dir = Path(__file__).resolve().parent.parent / "outputs" / "ridge_density_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        synth_path = out_dir / "synthetic_test.png"
        create_synthetic_test_image(synth_path, size=400)
        input_path = str(synth_path)
        output_path = str(out_dir / "ridge_costmap_synthetic_test.pkl")
        crop_size = min(400, args.crop_size)
    else:
        input_path = args.input or "data/MEDEA_fram_20100629.png"
        output_path = args.output
        crop_size = args.crop_size

    generate_ridge_costmap(
        input_image_path=input_path,
        satellite_image_path=args.satellite,
        crop_size=crop_size,
        costmap_scale=args.costmap_scale,
        method=args.method,
        window_size=args.window_size,
        output_path=output_path,
        visualize=args.visualize or args.synthetic,
        polygon_file=args.polygon,
    )


if __name__ == "__main__":
    main()
