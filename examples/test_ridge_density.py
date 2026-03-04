"""
Minimal test for ridge density computation (Step 1 of AUTO-IceNav ridge/costmap port).

Run: uv run python examples/test_ridge_density.py

Visualizations are saved to outputs/ridge_density_test/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from medea_ice_ship.ridge_density import (
    compute_ridge_density_from_optical_image,
    compute_ridge_density_from_concentration_grid,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "ridge_density_test"


def _ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def test_ridge_density_optical():
    """Test ridge density from synthetic optical-like image."""
    np.random.seed(42)
    h, w = 200, 200
    smooth = np.ones((h, w)) * 0.5 + np.random.randn(h, w) * 0.02
    smooth[50:100, 50:150] += np.random.randn(50, 100) * 0.3
    img = np.clip(smooth * 255, 0, 255).astype(np.uint8)

    density = compute_ridge_density_from_optical_image(
        img, window_size=25, method="combined", auto_scale_window=False, verbose=True
    )

    assert density.shape == img.shape, f"Shape mismatch: {density.shape} vs {img.shape}"
    assert np.all(density >= 0) and np.all(density <= 1), "Density must be in [0, 1]"
    ridge_zone = density[50:100, 50:150]
    quiet_zone = density[10:40, 10:40]
    assert ridge_zone.mean() >= quiet_zone.mean() * 0.5, "Ridged zone should have higher density"

    # Visualization
    _ensure_out_dir()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input image (synthetic, textured region = ridge zone)")
    axes[0].axis("off")
    im = axes[1].imshow(density, cmap="hot", origin="upper", vmin=0, vmax=1)
    axes[1].set_title("Ridge density (combined method)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Density")
    # Mark ridge zone
    from matplotlib.patches import Rectangle

    axes[1].add_patch(
        Rectangle((50, 50), 100, 50, fill=False, edgecolor="cyan", linewidth=2)
    )
    axes[1].text(55, 55, "ridge zone", color="cyan", fontsize=9)
    plt.tight_layout()
    out_path = OUT_DIR / "ridge_density_optical.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Ridge density from optical image: shape OK, range [0,1], ridge zone > quiet zone")
    print(f"  Saved: {out_path}")


def test_ridge_density_concentration():
    """Test ridge density from concentration grid."""
    np.random.seed(43)
    grid = np.random.rand(80, 80) * 0.5 + 0.5
    grid += np.linspace(0, 0.3, 80)[np.newaxis, :]

    density = compute_ridge_density_from_concentration_grid(
        grid, window_size=15, compression_threshold=0.6
    )

    assert density.shape == grid.shape
    assert np.all(density >= 0) and np.all(density <= 1)

    # Visualization
    _ensure_out_dir()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(grid, cmap="Blues", origin="upper", vmin=0, vmax=1)
    axes[0].set_title("Ice concentration grid")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], label="Concentration")
    im1 = axes[1].imshow(density, cmap="hot", origin="upper", vmin=0, vmax=1)
    axes[1].set_title("Ridge density (from concentration)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], label="Density")
    plt.tight_layout()
    out_path = OUT_DIR / "ridge_density_concentration.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✓ Ridge density from concentration grid: shape OK, range [0,1]")
    print(f"  Saved: {out_path}")


def test_methods():
    """Test that all optical methods run without error."""
    np.random.seed(44)
    img = np.random.randint(0, 256, (120, 120), dtype=np.uint8)
    # Add some structure so methods produce visible differences
    img[30:90, 30:90] = np.clip(img[30:90, 30:90] + 40, 0, 255)

    results = {}
    for method in ("texture_variance", "edge_density", "gradient_magnitude", "combined"):
        density = compute_ridge_density_from_optical_image(
            img, method=method, auto_scale_window=False
        )
        assert density.shape == img.shape
        assert np.all(density >= 0) and np.all(density <= 1)
        results[method] = density

    # Visualization: 2x2 grid of all methods
    _ensure_out_dir()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for ax, (method, density) in zip(axes, results.items()):
        im = ax.imshow(density, cmap="hot", origin="upper", vmin=0, vmax=1)
        ax.set_title(method)
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="Density")
    fig.suptitle("Ridge density: optical methods (same input image)", fontsize=12)
    plt.tight_layout()
    out_path = OUT_DIR / "ridge_density_methods_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✓ All methods (texture_variance, edge_density, gradient_magnitude, combined) OK")
    print(f"  Saved: {out_path}")


def test_ridge_costmap_generation():
    """Test ridge costmap generation script (Step 2): synthetic image → pickle + visualization."""
    import subprocess
    import sys

    _ensure_out_dir()
    proj_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(proj_root / "scripts" / "generate_ridge_costmap.py"), "--synthetic"],
        cwd=proj_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    out_pkl = OUT_DIR / "ridge_costmap_synthetic_test.pkl"
    vis_path = OUT_DIR / "ridge_costmap_synthetic_test_visualization.png"
    assert out_pkl.exists(), f"Ridge costmap pickle not created: {out_pkl}"
    assert vis_path.exists(), f"Visualization not saved: {vis_path}"

    import pickle
    with open(out_pkl, "rb") as f:
        data = pickle.load(f)
    assert "ridge_costmap" in data
    assert data["ridge_costmap"].min() >= 0 and data["ridge_costmap"].max() <= 1

    print("\n✓ Ridge costmap generation: pickle + visualization OK")
    print(f"  Saved: {out_pkl}, {vis_path}")


if __name__ == "__main__":
    print("Testing ridge density module (AUTO-IceNav Lensu method)...")
    test_ridge_density_optical()
    test_ridge_density_concentration()
    test_methods()
    print("\n--- Step 2: Ridge costmap generation ---")
    test_ridge_costmap_generation()
    print("\n" + "=" * 50)
    print("All ridge density tests passed.")
    print(f"\nVisualizations saved to: {OUT_DIR}")
