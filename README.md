# DPDEM: Dilated-Polyhedron-based Discrete Element Method

Implementation of ice-ship interaction simulation based on:

> Wang et al. (2024) - "Interaction between ice cover and floating platform simulated by dilated-polyhedron-based DEM" - Ocean Engineering 301

## MEDEA Ship–Ice Fracture Simulation

The main demo uses **satellite-derived ice floes** (MEDEA Fram Strait, Hwang et al. 2022) with DPDEM-style bond fracture and ship–ice interaction.

### 2D and 3D outputs

| 2D | 3D |
|----|-----|
| [medea_ship_fracture.gif](medea_ship_fracture.gif) | [medea_ship_fracture_3d.gif](medea_ship_fracture_3d.gif) |

Run `uv run python run_medea_ship_fracture.py` to regenerate these animations (use `--simple` for a small synthetic ice field).

- **Ice**: MEDEA polygons, tessellated into cells with bonds; fracture when bonds break under ship load.
- **Ship**: Prescribed path (e.g. diagonal across the domain), tapered hull, heading-aware collision.
- **3D**: Same simulation with added z/tilt for bending and submergence under the hull.
- **Parallelization**: For MP4 output, 2D and 3D animations use multi-worker rendering (`n_workers` in config; default 4). Use `n_workers=1` for sequential rendering; GIF output does not use parallel workers.

### Usage

Use `uv run python ...` to run with the project’s environment (or `pip install -r requirements.txt` and `python ...`).

```bash
# Install dependencies (or use uv)
pip install -r requirements.txt

# Full run (simulation + 2D + 3D animations); outputs MP4 by default
uv run python run_medea_ship_fracture.py

# Simple synthetic ice field (3 floes, fast for testing)
uv run python run_medea_ship_fracture.py --simple

# Simulation only (no animations)
uv run python run_medea_ship_fracture.py --quick
uv run python run_medea_ship_fracture.py --simple --quick

# Skip 3D animation
uv run python run_medea_ship_fracture.py --no-3d

# Full ice map, ship diagonal bottom-left → top-right
uv run python run_medea_ship_fracture.py --full-map

# Reproducibility: fix random seed
uv run python run_medea_ship_fracture.py --simple --seed 42

# Save state for replay (use --simple for simple scenario; --quick skips animations after saving)
uv run python run_medea_ship_fracture.py --simple --quick --save-state scenario.pkl

# Replay: generate 2D and 3D animations from saved state (no simulation)
uv run python run_medea_ship_fracture.py --load-state scenario.pkl
uv run python run_medea_ship_fracture.py --load-state scenario.pkl --no-3d   # 2D only

# Output GIF instead of MP4 (slower, larger files; no parallel rendering)
uv run python run_medea_ship_fracture.py --simple --gif
```

**Flags summary:** `--quick` (simulation only), `--no-3d` (skip 3D), `--simple` (synthetic 3-floe field), `--full-map` (full MEDEA map, diagonal ship), `--seed N`, `--save-state FILE`, `--load-state FILE`, `--gif` (GIF instead of MP4). Optional: `--thick-ice M`, `--thrust-cap N`, `--no-thrust-cap`.

### Reproducibility (Section 5.1)

The script sets a single random seed at startup (default from config, overridable with `--seed`). This makes tessellation, thickness noise (`assign_thickness_by_area`), and small angular velocity noise deterministic. Same seed ⇒ same ice field layout and same simulation outcome.

### Scenario replay (Section 5.2)

Use `--save-state FILE` to write the full simulation state (positions, rotations, bonds, ship history, submergence, etc.) to a pickle file. Use `--load-state FILE` to skip the simulation and regenerate only the 2D and 3D animations from that state. For the simple synthetic scenario, run with `--simple` when saving (e.g. `--simple --quick --save-state scenario.pkl`).

Requires MEDEA data: `data/ice_floes_MEDEA_fram_20100629.pkl` (see `data/` or AUTO-IceNav MEDEA dataset).

## Control and autonomous navigation

The `dpdem` library includes **planning** and **control** modules used for autonomous navigation in ice:

- **Planning**: `dpdem.planning` — lattice planner, cost map, primitives.
- **Control**: `dpdem.control` — DP controller, GNC, trajectory setpoints.
- **Navigation**: `dpdem.navigation` — `AutonomousNavigator`, `NavigationConfig`.

Examples:

```bash
# Costmap from simple ice (visual)
uv run python examples/visualize_costmap.py

# Run planner and plot path on costmap
uv run python examples/visualize_planned_path.py
```
With `--plan`, the run script uses **receding-horizon planning** (AUTO-IceNav style): the costmap is rebuilt from the current ice state every N frames (config `planning_replan_every_n_frames`, default 60), and the trajectory is updated from the new plan. It also generates a **costmap movie**: costmap as background, ship trajectory drawing over time (`costmap_simple_ship_fracture.mp4` or `costmap_medea_ship_fracture.mp4`).

## Features

- **Voronoi ice elements**: Irregular polygon shapes (paper Fig. 2); MEDEA floes tessellated internally for fracture.
- **Bond mechanics**: Adjacent elements bonded, fracture under stress (mixed-mode, paper Eqs. 14–19).
- **Contact**: Hertz–Mindlin–style contact (paper Eqs. 9–10); in MEDEA script, simplified repulsion and ship collision.
- **3D visualization**: Ice z-offset and tilt from ship-relative position; tapered 3D hull.

## References

- Wang, X., Liu, L., Wang, S., & Ji, S. (2024). Interaction between ice cover and floating platform simulated by dilated-polyhedron-based DEM. *Ocean Engineering*, 301, 117413.
- MEDEA ice data: Hwang et al. (2022), multi-satellite floe size distribution (2000–2020); Fram Strait 2010-06-29.
