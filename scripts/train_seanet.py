#!/usr/bin/env python3
"""
Train the SeaCurrentRNN ocean current model.

Two data sources:

  --synthetic  (recommended, no downloads)
      Generates divergence-free Fourier-mode current fields and creates a
      training sequence of slowly-evolving frames.  Everything is self-
      contained — no external data needed.

  --data path/to/ECCO4.pkl   (real oceanographic data)
      Uses preprocessed ECCO4 monthly surface velocity fields.  Run
      ``scripts/preprocess_ecco4.py`` first to create the pickle.

Outputs:
    data/seanet.pth          — trained model weights
    data/seanet_x0.npy       — initial current field (H, W, 2)
                               used by --current learned at runtime

Usage:
    # Quick synthetic training (works out of the box):
    python scripts/train_seanet.py --synthetic

    # With real ECCO4 data:
    python scripts/train_seanet.py --data data/ECCO4.pkl

    # Then run simulation with learned currents:
    uv run python run_medea_ship_fracture.py --simple \\
        --current learned \\
        --seanet-weights data/seanet.pth \\
        --current-field data/seanet_x0.npy
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path so medea_ice_ship is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medea_ice_ship.seanet import SeaCurrentRNN, upsample_dataset
from medea_ice_ship.ocean_current import generate_synthetic_current_field


def _generate_synthetic_sequence(
    n_frames: int = 200,
    H: int = 64,
    W: int = 64,
    max_speed: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Create a (T, H, W, 2) sequence of slowly-evolving current fields.

    Each frame is a divergence-free Fourier field with smoothly drifting
    phases, simulating a realistic slow evolution of ocean currents.
    """
    rng = np.random.RandomState(seed)
    n_modes = 5
    X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, W),
                        np.linspace(0, 2 * np.pi, H))

    kxs = rng.randint(1, 4, size=n_modes)
    kys = rng.randint(1, 4, size=n_modes)
    amps = rng.uniform(0.2, 1.0, size=n_modes)
    phases = rng.uniform(0, 2 * np.pi, size=n_modes)
    # Phase drift rate per frame (slow evolution)
    drift_rates = rng.uniform(-0.03, 0.03, size=n_modes)

    seq = np.zeros((n_frames, H, W, 2))
    for t in range(n_frames):
        field = np.zeros((H, W, 2))
        for m in range(n_modes):
            phase_t = phases[m] + drift_rates[m] * t
            arg = kxs[m] * X + kys[m] * Y + phase_t
            field[:, :, 0] += amps[m] * kys[m] * np.cos(arg)
            field[:, :, 1] -= amps[m] * kxs[m] * np.cos(arg)
        peak = np.sqrt(field[:, :, 0] ** 2 + field[:, :, 1] ** 2).max()
        if peak > 1e-10:
            field *= max_speed / peak
        seq[t] = field
    return seq


def main():
    parser = argparse.ArgumentParser(
        description="Train SeaCurrentRNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true",
                     help="Generate synthetic training data (no downloads needed)")
    src.add_argument("--data", type=str, default=None,
                     help="Path to preprocessed ECCO4.pkl")
    parser.add_argument("--epochs", type=int, default=64,
                        help="Training epochs (default: 64)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Spatial resolution H=W for synthetic data (default: 64)")
    parser.add_argument("--n-frames", type=int, default=200,
                        help="Number of synthetic training frames (default: 200)")
    parser.add_argument("--out-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "data"),
                        help="Output directory for weights + initial field")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "seanet.pth"
    x0_path = out_dir / "seanet_x0.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Prepare data ──────────────────────────────────────────────────
    if args.synthetic:
        print(f"Generating synthetic training data "
              f"({args.n_frames} frames, {args.resolution}x{args.resolution})...")
        seq = _generate_synthetic_sequence(
            n_frames=args.n_frames, H=args.resolution, W=args.resolution)
        inputs = torch.from_numpy(seq[:-1].astype(np.float32))   # (T-1, H, W, 2)
        targets = torch.from_numpy(seq[1:].astype(np.float32))
        H, W = args.resolution, args.resolution
        # Save the first frame as the initial field for simulation
        np.save(x0_path, seq[0])
        print(f"  Saved initial field: {x0_path}")
    else:
        import pickle
        print(f"Loading ECCO4 data from {args.data}...")
        with open(args.data, "rb") as f:
            raw = pickle.load(f)
        raw = np.array(raw) if not isinstance(raw, np.ndarray) else raw
        dataset = torch.from_numpy(raw[:, :2, :, :]).float()  # (T, 2, H, W)
        series = upsample_dataset(dataset, factor=30)          # (T', 2, H, W)
        series = series.permute(0, 2, 3, 1)                   # (T', H, W, 2)
        inputs = series[:-1]
        targets = series[1:]
        H, W = int(inputs.shape[1]), int(inputs.shape[2])
        # Save first frame
        np.save(x0_path, inputs[0].numpy())
        print(f"  Saved initial field: {x0_path}")

    print(f"  Training samples: {inputs.shape[0]},  field size: {H}x{W}")

    # ── Build model ───────────────────────────────────────────────────
    model = SeaCurrentRNN(hidden_size=128, input_height=H, input_width=W)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Training loop ─────────────────────────────────────────────────
    n_batches = math.ceil(inputs.shape[0] / args.batch)
    print(f"\nTraining for {args.epochs} epochs, {n_batches} batches/epoch...")
    for epoch in range(1, args.epochs + 1):
        losses = []
        for b in range(n_batches):
            bx = inputs[b * args.batch:(b + 1) * args.batch].to(device)
            by = targets[b * args.batch:(b + 1) * args.batch].to(device)
            preds = model(bx)
            loss = torch.mean((preds - by) ** 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        print(f"  Epoch {epoch:03d}/{args.epochs} | MSE {avg_loss:.6f}")
        torch.save(model.state_dict(), weights_path)

    # ── Final evaluation ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds = model(inputs[:args.batch].to(device))
        mae = torch.mean(torch.abs(preds.cpu() - targets[:args.batch])).item()
    print(f"\nFinal MAE (first batch): {mae:.6f}")
    print(f"Saved weights: {weights_path}")
    print(f"Saved initial field: {x0_path}")
    print(f"\nTo use in simulation:")
    print(f"  uv run python run_medea_ship_fracture.py --simple \\")
    print(f"      --current learned \\")
    print(f"      --seanet-weights {weights_path} \\")
    print(f"      --current-field {x0_path}")


if __name__ == "__main__":
    main()
