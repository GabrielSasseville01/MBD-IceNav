"""
Ocean current models for ice–ship simulation.

Three modes:
  - NoCurrentModel      : zero current everywhere (default)
  - UniformCurrentModel : constant (u, v) everywhere
  - LearnedCurrentModel : spatially-varying field evolved by SeaCurrentRNN

Usage in simulation (Eq. 9):
  Drag is computed on the *relative* velocity  v_ice - v_water(x, y),
  so ice embedded in a current drifts at equilibrium with it.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


# ── Base ──────────────────────────────────────────────────────────────────

class OceanCurrentModel(ABC):
    """Abstract base for ocean current models."""

    @abstractmethod
    def sample(self, x: float, y: float) -> np.ndarray:
        """Return (2,) water velocity [u, v] at world position (x, y)."""

    def sample_batch(self, positions: np.ndarray) -> np.ndarray:
        """Return (n, 2) water velocities for (n, 2) world positions."""
        return np.array([self.sample(p[0], p[1]) for p in positions])

    def sample_chunk_jax(self, K: int, n: int, pos_jax=None, dtype=None):
        """Return (K, n, 2) JAX array of water velocities for a chunk of K frames.

        Default: samples once at *pos_jax* positions (or zeros if None) and
        tiles the result K times — correct for static current fields.
        Time-varying subclasses (LearnedCurrentModel) should override if
        per-frame accuracy is required; the base implementation is a
        conservative-but-correct approximation for any model.

        Parameters
        ----------
        K       : number of frames in the chunk
        n       : number of ice elements
        pos_jax : JAX (n, 2) array of ice positions at chunk start, or None
        dtype   : JAX dtype (default jnp.float64)
        """
        import jax.numpy as jnp
        dt = dtype if dtype is not None else jnp.float64
        if pos_jax is not None:
            pos_np = np.array(pos_jax)                      # one GPU→CPU transfer
            v_np = self.sample_batch(pos_np)                # (n, 2) NumPy
            v_jax = jnp.array(v_np, dtype=dt)               # CPU→GPU
            return jnp.broadcast_to(v_jax[None, :, :], (K, n, 2))
        return jnp.zeros((K, n, 2), dtype=dt)

    def step(self):
        """Advance internal state by one simulation frame (no-op for static models)."""

    def get_field_for_viz(self, x_min: float, x_max: float,
                          y_min: float, y_max: float,
                          nx: int = 20, ny: int = 20):
        """Return (X, Y, U, V) meshgrid arrays for a quiver plot.

        Returns None when there is nothing to draw (e.g. NoCurrentModel).
        """
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(xs, ys)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(ny):
            for j in range(nx):
                vel = self.sample(X[i, j], Y[i, j])
                U[i, j] = vel[0]
                V[i, j] = vel[1]
        return X, Y, U, V

    @property
    def is_active(self) -> bool:
        """True when the model produces non-zero currents (used to gate viz)."""
        return True


# ── No current ────────────────────────────────────────────────────────────

class NoCurrentModel(OceanCurrentModel):
    """Zero current everywhere (default)."""

    def sample(self, x: float, y: float) -> np.ndarray:
        return np.zeros(2)

    def sample_batch(self, positions: np.ndarray) -> np.ndarray:
        return np.zeros_like(positions)

    def sample_chunk_jax(self, K: int, n: int, pos_jax=None, dtype=None):
        """Return (K, n, 2) zeros on the JAX device — no host-device transfer."""
        import jax.numpy as jnp
        dt = dtype if dtype is not None else jnp.float64
        return jnp.zeros((K, n, 2), dtype=dt)

    def get_field_for_viz(self, *args, **kwargs):
        return None

    @property
    def is_active(self) -> bool:
        return False


# ── Uniform ───────────────────────────────────────────────────────────────

class UniformCurrentModel(OceanCurrentModel):
    """Constant (u, v) current everywhere.

    Args:
        speed: current speed [m/s].
        direction_rad: direction in radians (0 = +x, pi/2 = +y).
    """

    def __init__(self, speed: float, direction_rad: float):
        self.velocity = np.array([
            speed * np.cos(direction_rad),
            speed * np.sin(direction_rad),
        ])

    def sample(self, x: float, y: float) -> np.ndarray:
        return self.velocity.copy()

    def sample_batch(self, positions: np.ndarray) -> np.ndarray:
        return np.tile(self.velocity, (len(positions), 1))

    def sample_chunk_jax(self, K: int, n: int, pos_jax=None, dtype=None):
        """Return (K, n, 2) broadcast of the constant velocity — no position needed."""
        import jax.numpy as jnp
        dt = dtype if dtype is not None else jnp.float64
        vel_jax = jnp.array(self.velocity, dtype=dt)   # (2,)
        return jnp.broadcast_to(vel_jax[None, None, :], (K, n, 2))


# ── Grid (internal, used by LearnedCurrentModel) ─────────────────────────

class _GridCurrentModel(OceanCurrentModel):
    """Bilinear interpolation from a regular (H, W, 2) velocity grid."""

    def __init__(self, field: np.ndarray,
                 x_min: float, x_max: float,
                 y_min: float, y_max: float):
        self.field = np.asarray(field, dtype=np.float64)
        assert self.field.ndim == 3 and self.field.shape[2] == 2
        self.H, self.W = self.field.shape[:2]
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self._inv_dx = max(self.W - 1, 1) / max(x_max - x_min, 1e-6)
        self._inv_dy = max(self.H - 1, 1) / max(y_max - y_min, 1e-6)

    def sample(self, x: float, y: float) -> np.ndarray:
        gx = np.clip((x - self.x_min) * self._inv_dx, 0, self.W - 1)
        gy = np.clip((y - self.y_min) * self._inv_dy, 0, self.H - 1)
        ix = min(int(np.floor(gx)), self.W - 2)
        iy = min(int(np.floor(gy)), self.H - 2)
        fx, fy = gx - ix, gy - iy
        return (self.field[iy, ix]         * (1 - fx) * (1 - fy)
              + self.field[iy, ix + 1]     * fx       * (1 - fy)
              + self.field[iy + 1, ix]     * (1 - fx) * fy
              + self.field[iy + 1, ix + 1] * fx       * fy)

    def sample_batch(self, positions: np.ndarray) -> np.ndarray:
        gx = np.clip((positions[:, 0] - self.x_min) * self._inv_dx, 0, self.W - 1)
        gy = np.clip((positions[:, 1] - self.y_min) * self._inv_dy, 0, self.H - 1)
        ix = np.minimum(np.floor(gx).astype(int), self.W - 2)
        iy = np.minimum(np.floor(gy).astype(int), self.H - 2)
        fx = (gx - ix)[:, np.newaxis]
        fy = (gy - iy)[:, np.newaxis]
        return (self.field[iy, ix]         * (1 - fx) * (1 - fy)
              + self.field[iy, ix + 1]     * fx       * (1 - fy)
              + self.field[iy + 1, ix]     * (1 - fx) * fy
              + self.field[iy + 1, ix + 1] * fx       * fy)

    def update_field(self, new_field: np.ndarray):
        self.field = np.asarray(new_field, dtype=np.float64)

    def sample_chunk_jax(self, K: int, n: int, pos_jax=None, dtype=None):
        """Sample at chunk-start positions; tile K times (static-field approximation).

        Uses one GPU→CPU transfer instead of K, then broadcasts the result.
        For LearnedCurrentModel the field evolves; see that class for overrides.
        """
        import jax.numpy as jnp
        dt = dtype if dtype is not None else jnp.float64
        if pos_jax is not None:
            pos_np = np.array(pos_jax)                  # one GPU→CPU transfer
            v_np = self.sample_batch(pos_np)             # (n, 2) bilinear NumPy
            v_jax = jnp.array(v_np, dtype=dt)           # CPU→GPU (n, 2)
            return jnp.broadcast_to(v_jax[None, :, :], (K, n, 2))
        return jnp.zeros((K, n, 2), dtype=dt)


# ── Learned (SeaCurrentRNN) ──────────────────────────────────────────────

class LearnedCurrentModel(_GridCurrentModel):
    """Spatially-varying ocean current field evolved by SeaCurrentRNN.

    The field is a ``(H, W, 2)`` grid of ``(u, v)`` velocities sampled via
    bilinear interpolation.  Every *update_every_n_steps* simulation frames
    the SeaCurrentRNN (physics-informed shallow-water RNN) rolls the field
    forward one step, producing a time-evolving current.

    Setup (two options):

    A) **ECCO4 real data** — download ECCO4 surface velocity from NASA,
       preprocess into ``data/ECCO4.pkl`` + ``data/ecco4_x0.pth``, then
       train the RNN to produce ``data/seanet.pth``.  See
       ``scripts/preprocess_ecco4.py`` and ``scripts/train_seanet.py``.

    B) **Synthetic data (no download needed)** — run
       ``scripts/train_seanet.py --synthetic`` which generates divergence-free
       Fourier-mode training sequences and trains the RNN on them.  The
       resulting weights + initial field are saved to ``data/``.

    Args:
        initial_field: (H, W, 2) starting current field.
        x_min, x_max, y_min, y_max: world bounds.
        seanet_weights_path: path to trained ``seanet.pth``.
        initial_field_path: path to the initial field file (ecco4_x0.pth or
            synthetic_x0.npy).  Ignored when *initial_field* is given
            directly.
        hidden_size: RNN hidden dimension (must match the saved weights).
        update_every_n_steps: evolve the field every N simulation frames.
        noise_strength: multiplicative scale applied to the raw field.
    """

    def __init__(self, initial_field: np.ndarray,
                 x_min: float, x_max: float,
                 y_min: float, y_max: float,
                 seanet_weights_path: Optional[str] = None,
                 hidden_size: int = 128,
                 update_every_n_steps: int = 1000,
                 noise_strength: float = 0.5):
        self._noise_strength = noise_strength
        super().__init__(initial_field * noise_strength,
                         x_min, x_max, y_min, y_max)
        self._raw_field = initial_field.copy()
        self._update_every = update_every_n_steps
        self._step_count = 0

        try:
            import torch
            from medea_ice_ship.seanet import SeaCurrentRNN
        except ImportError:
            print("  [ocean_current] PyTorch not available; "
                  "learned model will use static field.  "
                  "Install torch:  pip install torch", flush=True)
            self._has_torch = False
            return

        self._has_torch = True
        H, W = initial_field.shape[:2]
        self._device = torch.device("cpu")
        self._seanet = SeaCurrentRNN(
            hidden_size=hidden_size,
            input_height=H,
            input_width=W,
        ).to(self._device)

        if seanet_weights_path is not None:
            state = torch.load(seanet_weights_path,
                               map_location=self._device, weights_only=False)
            self._seanet.load_state_dict(state)
            print(f"  [ocean_current] Loaded SeaCurrentRNN weights "
                  f"from {seanet_weights_path}", flush=True)
        else:
            print("  [ocean_current] SeaCurrentRNN initialised with "
                  "random weights (no trained model)", flush=True)

        self._seanet.eval()
        self._torch_field = torch.from_numpy(
            initial_field.astype(np.float32)
        ).unsqueeze(0)  # (1, H, W, 2)

    def step(self):
        self._step_count += 1
        if not self._has_torch:
            return
        if self._step_count % self._update_every != 0:
            return
        import torch
        with torch.no_grad():
            self._torch_field = self._seanet(self._torch_field)
        new_raw = self._torch_field[0].numpy().copy()
        self._raw_field = new_raw
        self.update_field(new_raw * self._noise_strength)


# ── Synthetic field generation ────────────────────────────────────────────

def generate_synthetic_current_field(H: int, W: int,
                                     n_modes: int = 5,
                                     max_speed: float = 0.5,
                                     seed: int = 42) -> np.ndarray:
    """Generate a smooth, divergence-free field via random Fourier stream functions.

    Returns:
        (H, W, 2) array of (u, v) velocities with peak magnitude *max_speed*.
    """
    rng = np.random.RandomState(seed)
    field = np.zeros((H, W, 2))
    X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, W),
                        np.linspace(0, 2 * np.pi, H))
    for _ in range(n_modes):
        kx = rng.randint(1, 4)
        ky = rng.randint(1, 4)
        amp = rng.uniform(0.2, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        arg = kx * X + ky * Y + phase
        field[:, :, 0] += amp * ky * np.cos(arg)
        field[:, :, 1] -= amp * kx * np.cos(arg)
    peak = np.sqrt(field[:, :, 0] ** 2 + field[:, :, 1] ** 2).max()
    if peak > 1e-10:
        field *= max_speed / peak
    return field


# ── Factory ───────────────────────────────────────────────────────────────

def create_ocean_current_model(
    mode: str,
    domain: Tuple[float, float, float, float],
    speed: float = 0.3,
    direction_rad: float = 0.0,
    field_path: Optional[str] = None,
    seanet_weights_path: Optional[str] = None,
    grid_resolution: int = 64,
    update_every_n_steps: int = 1000,
    noise_strength: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> OceanCurrentModel:
    """Create an ocean current model.

    Args:
        mode: ``"none"`` | ``"uniform"`` | ``"learned"``.
        domain: (x_min, x_max, y_min, y_max) world bounding box.
        speed: current speed [m/s] (uniform) or peak speed of the
               synthetic initial field (learned without *field_path*).
        direction_rad: direction [rad] for uniform mode.
        field_path: path to an initial current field (``.npy``, ``.pkl``,
                    ``.pth``) for learned mode.  If *None* a synthetic
                    divergence-free field is generated.
        seanet_weights_path: path to trained ``seanet.pth`` (learned mode).
        grid_resolution: spatial resolution for the synthetic field.
        update_every_n_steps: how often to evolve the learned field.
        noise_strength: scale factor for the learned field.
        seed: random seed for synthetic field generation.
    """
    x_min, x_max, y_min, y_max = domain

    if mode == "none":
        return NoCurrentModel()

    if mode == "uniform":
        return UniformCurrentModel(speed, direction_rad)

    if mode == "learned":
        if field_path is not None:
            field = _load_field(field_path)
        else:
            field = generate_synthetic_current_field(
                grid_resolution, grid_resolution, seed=seed, max_speed=speed)
        return LearnedCurrentModel(
            field, x_min, x_max, y_min, y_max,
            seanet_weights_path=seanet_weights_path,
            update_every_n_steps=update_every_n_steps,
            noise_strength=noise_strength,
        )

    raise ValueError(
        f"Unknown ocean current mode: {mode!r}.  "
        f"Choose from: none, uniform, learned."
    )


def _load_field(path: str) -> np.ndarray:
    """Load (H, W, 2) current field from .npy / .pkl / .pth."""
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        field = np.load(path)
    elif ext in (".pkl", ".pickle"):
        import pickle
        with open(path, "rb") as f:
            field = pickle.load(f)
    elif ext == ".pth":
        import torch
        raw = torch.load(path, map_location="cpu", weights_only=False)
        field = raw.numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
    else:
        raise ValueError(f"Unsupported field file format: {ext}")
    field = np.asarray(field, dtype=np.float64)
    if field.ndim == 4:
        field = field[0]
    if field.ndim == 3 and field.shape[0] == 2:
        field = field.transpose(1, 2, 0)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError(f"Expected (H, W, 2), got {field.shape}")
    return field
