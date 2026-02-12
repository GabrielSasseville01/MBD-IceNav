# Diffusion MPC planner (AUTO-IceNav style)

This module provides a **Diffusion MPC–style** path planner for ice navigation. It is an **alternative** to the lattice A* planner and shares the same costmap, world frame, and ship trajectory interface.

## Relation to AUTO-IceNav

**AUTO-IceNav (T-RO 2025 / ICRA 2023) does *not* use a diffusion model.** The published AUTO-IceNav approach is:

1. **Lattice-based A* search** on motion primitives (Dubins-like).
2. A **kinetic-energy cost function** that minimizes expected kinetic energy loss from ship–ice collisions (rather than simple collision avoidance).
3. **NLP refinement**: the lattice path is used as an initial guess for an optimization step that produces a locally optimal path.

In this codebase, the **lattice planner** (`LatticePlanner`, `--planner lattice`) is the one that matches AUTO-IceNav’s structure: A* on primitives with a costmap. Our costmap uses mass/collision (and optional damage) cost rather than the exact KE-loss formulation in the paper, but the pipeline is the same (lattice → optional NLP via `--optimize-path`).

With `--planner diffusion-mpc` we now implement the **same pipeline** as AUTO-IceNav's diffusion planner: lattice A* then **model-based diffusion** over control sequences (sample → rollout with 3-DOF dynamics → score with costmap + path + waypoint anchors → reward-weighted update). See `medea_ice_ship/diffusion_controller.py` and `optimize_with_diffusion()` in `medea_ice_ship/planning.py`.

## Implementation in this codebase

- **Model-based diffusion (used with `--planner diffusion-mpc`):** After lattice A*, we run `optimize_with_diffusion()` which builds a `ModelBasedDiffusionController` with our 3-DOF dynamics (M, D, Coriolis) and costmap. Control sequences (body-frame force/torque) are optimized via reverse diffusion (sample → rollout → score with costmap + path tracking + waypoint anchors → softmax-weight update). Output is an optimized (x, y, psi) trajectory, which is converted to `ship_trajectory` for the simulator.
- **Sampling-based fallback:** The class `DiffusionMPCPlanner` in `dpdem/planning/diffusion_mpc.py` is a simpler alternative (random primitive rollouts, no learned model). It is not used by default when `--planner diffusion-mpc` is set; the run script uses lattice + model-based diffusion instead.

## Usage

- **Run script:** Use `--plan --planner diffusion-mpc` to use this alternative planner:
  ```bash
  uv run python run_medea_ship_fracture.py --simple --plan --planner diffusion-mpc
  ```
- **Receding horizon:** With `--plan --planner diffusion-mpc`, replanning uses lattice A* + model-based diffusion as well.
- **Config:** In `medea_ice_ship.config.RunConfig`:
  - `planning_planner`: `"lattice"` (default) or `"diffusion_mpc"`
  - `planning_diffusion_controller_horizon`: time steps for diffusion controller (default 80)
  - `planning_diffusion_num_samples`: samples per diffusion step (default 64)
  - `planning_diffusion_num_diffusion_steps`: reverse diffusion steps (default 30)
  - `planning_diffusion_temperature`, `planning_diffusion_beta0`, `planning_diffusion_betaT`: diffusion schedule
  - `planning_diffusion_seed`: optional random seed

## Interface

- **Optimization:** `medea_ice_ship.planning.optimize_with_diffusion(continuous_path, start_pose, goal, cost_map, run_cfg, start_nu=...)`  
  Refines a path (e.g. from lattice A*) using model-based diffusion; returns optimized (N, 3) path.
- **Controller:** `medea_ice_ship.diffusion_controller.ModelBasedDiffusionController`  
  Reverse diffusion over control sequences; uses `medea_ice_ship.vessel_dynamics.rollout_3dof` and CostMap for scoring.

## Notes

- The lattice search visualization movie is only generated for the lattice planner; with `--planner diffusion-mpc` it is skipped.
- Path refinement (NLP optimizer) can be used with Diffusion MPC via `--optimize-path` as with the lattice planner.
