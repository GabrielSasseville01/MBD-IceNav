# Full-map simulation performance

A full-map run can take many hours. This guide explains where time is spent and how to speed things up.

## Where time is spent

1. **Physics simulation** (main loop in `medea_ice_ship/simulation.py`)
   - **CPU-only** (NumPy/Python). No GPU use.
   - Cost per frame: O(n) with n = number of ice cells (collision detection, bond forces, ice–ice contact, ship dynamics).
   - Total: `n_frames × per_frame_cost`. With target-speed mode, `n_frames` is derived from path length (e.g. ~1300 m at 4 m/s with dt=0.1 s → ~3250 frames).

2. **Replanning** (receding horizon)
   - Every `replan_every_n_frames` (default 120): rebuild costmap, run A*, and optionally run JAX planner (diffusion/gradient/MPPI).
   - **Costmap + A***: CPU. Use `--fast-lattice` on full-map for coarser, faster A*.
   - **JAX planners** (diffusion-mpc, gradient, mppi, etc.): run on **GPU** if JAX is installed with CUDA; otherwise CPU. GPU only accelerates this part, not the physics.

3. **Animations** (after simulation)
   - 2D, 3D, and costmap movies: CPU, parallelized with `n_workers`. Skip with `--quick` or `--no-3d`.

## Speedups (no code changes)

| Goal | Flags / config |
|------|-----------------|
| **Fewer replans** | `--replan-every 240` or `--replan-every 360` (fewer, cheaper replans). |
| **No replanning** | `--no-replan` (single initial plan for whole run; largest gain). |
| **Faster A*** | `--fast-lattice` (coarser grid, fewer headings on full-map). |
| **Lattice only** | `--planner lattice` (no JAX diffusion/gradient at replan). |
| **Skip animations** | `--quick` (simulation only) or `--no-3d` (skip 3D movie). |
| **Fewer frames** | `--frames 1500` (cap frames; shorter simulated time). |
| **Less frequent animations** | Increase `frame_skip_2d` / `frame_skip_3d` in config (fewer rendered frames). |

## GPU

- **Physics simulation**: Not GPU-accelerated. The loop is NumPy/Python; porting to JAX would be a large change.
- **Planning**: JAX planners (diffusion-mpc, gradient, mppi, sqp, gradcem) **use GPU automatically** when JAX is installed with CUDA (e.g. `pip install jax[cuda12]`). Replanning then runs on GPU; the rest of the run (simulation, costmap, A*) stays on CPU.
- To see which device JAX uses, check the planner log line (e.g. `JAX Gradient [CUDA (NVIDIA ...)]`).

## Suggested full-map commands

**Faster run (single plan, no 3D, quick):**
```bash
uv run python run_medea_ship_fracture.py --full-map --plan --planner lattice \
  --no-replan --fast-lattice --no-3d --quick --output-dir outputs/fast
```

**Faster run with replanning and 2D movie:**
```bash
uv run python run_medea_ship_fracture.py --full-map --plan --planner lattice \
  --replan-every 240 --fast-lattice --no-3d --output-dir outputs/fast
```

**After a run:** Check the printed **TIMING DIAGNOSTICS** to see whether simulation or planning dominates; then adjust `--replan-every`, `--planner`, or `--frames` accordingly.

---

## How hard would it be to run the physics on GPU?

**Summary: non-trivial (weeks to a few months), not a small change.**

The simulation in `medea_ice_ship/simulation.py` is a single Python loop over frames (~840 lines). Each frame does:

| Part | CPU today | GPU port difficulty |
|------|-----------|----------------------|
| **Ship–ice collision** | Per-element Python loop; per element: rotate polygon, test vertices/edges vs ship, `point_in_polygon` (ship pts in ice). | **Hard.** Variable polygon sizes (variable vertex count per cell). Would need fixed max vertices + masking, or a different representation (e.g. SDF). Point-in-polygon and penetration in JAX with fixed max verts is doable but careful. |
| **find_connected_components** | BFS over bond graph; returns list of variable-length components. | **Hard.** JAX wants fixed shapes. Would need a fixed max number of components and max size per component, or a fixed-iteration scan, and no Python set/dict. |
| **Bond forces** | `np.add.at(forces, bond_i, F_bond_vec)` (scatter-add). | **Easy.** JAX has `jnp.index_add`; already vectorized over bonds. |
| **Ice–ice contact** | All-pairs or neighbor pairs; `np.add.at` for forces. | **Medium.** O(n²) or O(n×neighbors); can be vmap’d over pairs. Scatter is fine. Need to ensure pair list is built in a JAX-friendly way. |
| **Component-wise integration** | `for comp in components`: COM velocity, then update positions per component. | **Hard.** Variable-length components, Python loops. Would need to either pad components and mask, or replace with a single per-particle integration (would change physics slightly unless we keep rigid-body COM per component in a fixed-shape form). |
| **Stress diffusion** | A few steps of scatter-update. | **Easy.** Same as bond forces; JAX scatter. |
| **DPDEM bond damage** | `update_bond_damage_state` in NumPy (vectorized). | **Easy.** Swap to `jnp`, no structural change. |
| **Ship dynamics** | Small O(1) NumPy (3-DOF). | **Easy.** Trivial in JAX. |
| **Replan callback** | Called from Python every N frames. | **Stays CPU/host.** Copy state device→host, call Python callback, then continue. No need to put callback on GPU. |

So the main blockers are:

1. **Variable structure**: Polygons (variable verts per cell), connected components (variable number and size). Requires fixed-shape + masking or a redesigned representation.
2. **Control flow**: Lots of `if pen > 0`, `for comp in components`, etc. Must be rewritten with `jnp.where`, `lax.cond`, or `lax.scan`.
3. **Geometry**: `point_in_polygon` and `point_inside_polygon_penetration` are Python loops over vertices. Need JAX versions (e.g. ray-cast with fixed max verts).

**Rough effort:** A focused, experienced person could get a **first GPU version** (same behaviour, no replan) in **~2–6 weeks** (JAX state, vmap ship–ice with padded polygons, JAX point-in-polygon, scatter for bonds/contact, fixed-shape or simplified components). Making it **feature-complete and validated** (replan callback, ocean current, all bond/damage options, matching current results) could add another **few weeks to a couple of months**. So overall: **moderate to high difficulty**, mainly due to variable-sized data and graph/component handling.

**Alternatives:** Use **Numba** to JIT-compile the hot loops (ship–ice, bond/contact) on CPU; less work than a full JAX GPU port and can give a noticeable speedup without changing the overall structure.
