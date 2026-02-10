# DPDEM: Dilated-Polyhedron-based Discrete Element Method

Implementation of ice-ship interaction simulation based on:

> Wang et al. (2024) - "Interaction between ice cover and floating platform simulated by dilated-polyhedron-based DEM" - Ocean Engineering 301

## MEDEA Ship–Ice Fracture Simulation

The main demo uses **satellite-derived ice floes** (MEDEA Fram Strait, Hwang et al. 2022) with DPDEM-style bond fracture and ship–ice interaction.

### 2D and 3D outputs

| 2D | 3D |
|----|-----|
| [medea_ship_fracture.gif](medea_ship_fracture.gif) | [medea_ship_fracture_3d.gif](medea_ship_fracture_3d.gif) |

Run `python run_medea_ship_fracture.py` to regenerate these animations.

- **Ice**: MEDEA polygons, tessellated into cells with bonds; fracture when bonds break under ship load.
- **Ship**: Prescribed path (e.g. diagonal across the domain), tapered hull, heading-aware collision.
- **3D**: Same simulation with added z/tilt for bending and submergence under the hull.

### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Full run (simulation + 2D + 3D animations)
python run_medea_ship_fracture.py

# Simulation only (no GIFs)
python run_medea_ship_fracture.py --quick

# Skip 3D animation
python run_medea_ship_fracture.py --no-3d

# Full ice map, ship diagonal bottom-left → top-right
python run_medea_ship_fracture.py --full-map

# Simple synthetic ice field (fast, for testing improvements)
python run_medea_ship_fracture.py --simple

# Reproducibility: fix random seed (tessellation, thickness noise, angular noise)
python run_medea_ship_fracture.py --simple --seed 42

# Save state after simulation for replay without re-running
python run_medea_ship_fracture.py --simple --save-state scenario.pkl

# Replay: load state and generate 2D/3D animations only (no simulation)
python run_medea_ship_fracture.py --load-state scenario.pkl
```

### Reproducibility (Section 5.1)

The script sets a single random seed at startup (default `42`, overridable with `--seed`). This makes tessellation, thickness noise (`assign_thickness_by_area`), and small angular velocity noise deterministic. Same seed ⇒ same ice field layout and same simulation outcome.

### Scenario replay (Section 5.2)

Use `--save-state FILE` to write the full simulation state (positions, rotations, bonds, ship history, submergence, etc.) to a pickle file. Use `--load-state FILE` to skip the simulation and regenerate only the 2D and 3D animations from that state.

Requires MEDEA data: `data/ice_floes_MEDEA_fram_20100629.pkl` (see `data/` or AUTO-IceNav MEDEA dataset).

## Control and autonomous navigation

The `dpdem` library includes **planning** and **control** modules used for autonomous navigation in ice:

- **Planning**: `dpdem.planning` — lattice planner, cost map, primitives.
- **Control**: `dpdem.control` — DP controller, GNC, trajectory setpoints.
- **Navigation**: `dpdem.navigation` — `AutonomousNavigator`, `NavigationConfig`.

Example:

```bash
python examples/autonomous_navigation.py
```

## Features

- **Voronoi ice elements**: Irregular polygon shapes (paper Fig. 2); MEDEA floes tessellated internally for fracture.
- **Bond mechanics**: Adjacent elements bonded, fracture under stress (mixed-mode, paper Eqs. 14–19).
- **Contact**: Hertz–Mindlin–style contact (paper Eqs. 9–10); in MEDEA script, simplified repulsion and ship collision.
- **3D visualization**: Ice z-offset and tilt from ship-relative position; tapered 3D hull.

## References

- Wang, X., Liu, L., Wang, S., & Ji, S. (2024). Interaction between ice cover and floating platform simulated by dilated-polyhedron-based DEM. *Ocean Engineering*, 301, 117413.
- MEDEA ice data: Hwang et al. (2022), multi-satellite floe size distribution (2000–2020); Fram Strait 2010-06-29.
