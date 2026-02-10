# Sea-Ice Model Improvements Checklist

Use this list to improve the simulation before and alongside autonomous navigation.  
**Test each item on the simple ice field** (`--simple`) so runs stay fast.

---

## 0. Simple test setup (do first)

- [ ] **0.1** Add a `--simple` mode: small synthetic ice field, few cells, few frames.
- [ ] **0.2** Verify simple run completes quickly and produces 2D + 3D outputs.
- [ ] **0.3** Use simple mode as the default test for every subsequent checkpoint.

**How to run the simple test:**
```bash
# Full run (simulation + 2D + 3D GIFs) — writes simple_ship_fracture.gif, simple_ship_fracture_3d.gif
python run_medea_ship_fracture.py --simple

# Simulation only (no GIFs, fastest)
python run_medea_ship_fracture.py --simple --quick
```

---

## 1. Ice physics (tackle next)

- [ ] **1.1** Use the DPDEM library bond model (elastic–viscous stress, mixed-mode fracture, B–K) instead of the ad-hoc force-based fracture. *(Reverted: previous integration made fractured floes stick together; revisit with different coupling or per-element dynamics.)*
- [x] **1.2** Drive stress propagation from actual bond forces (or make diffusion consistent with bond stiffness).
- [x] **1.3** Add tangential friction (Coulomb) to ice–ice contact so sliding and pile-up are modeled.
- [x] **1.4** Improve drag (e.g. depend on submerged area / thickness or concentration).

---

## 2. Ship–ice interaction

- [x] **2.1** Add simple ship dynamics (e.g. surge–sway–yaw) so ice loads affect the ship.  
  **Demo:** `python examples/ship_dynamics_demo.py` → `ship_dynamics_demo.png`.  
  **What to look for:** actual path lags/deviates from desired when in ice; speed dips when ice force spikes; position error grows in contact and can recover after clearing.
- [x] **2.2** Tie collision force to ice mechanics (e.g. Lindqvist-style or breaking force) instead of purely phenomenological scaling.  
  **Implemented:** Normal force = F_break * min(pen/pen_crit, 1) with F_break = breaking_force(thickness, w_contact), pen_crit = 0.5 m; side push = 0.2 × normal force.
- [x] **2.3** Harden collision detection (more sample points or polygon–polygon) to avoid missed/jumpy contacts.  
  **Implemented:** (1) Ice→ship: 4 sample points along each ice edge (in addition to vertices). (2) Ship→ice: ship hull points from `create_ship_hull_at_position_heading` tested with point-in-polygon vs each ice element; penetration/normal from `point_inside_polygon_penetration` (distance to nearest edge, outward normal).
- [x] **2.4** Optional: simple ice accumulation / rubble resistance at the bow.  
  **Implemented:** Bow rubble resistance: ice elements in the forward half (body_x > 0) contribute to a lumped “bow contact force”; an extra force opposes surge: F_rubble = −min(0.25 × bow_contact_force, 5 MN) × forward_direction, added to F_ice_ship so the ship feels additional resistance when pushing through ice at the bow.

---

## 3. 3D and thickness

- [x] **3.1** Optional: add simple 3D effects (e.g. buoyancy/effective force from thickness, or vertical motion of ice under the ship).  
  **Implemented:** Per-element submergence state (m) when ice is pressed under the ship: increases with penetration while in contact (capped at thickness), decays when not in contact (0.5 s time constant). Buoyancy force F_buoy = ρ_water × g × area × min(submergence, thickness) applied on ice away from ship (normal direction); reaction force and moment on ship included so the ship feels extra resistance from “pushing ice down.”
- [ ] **3.2** Support observed or prescribed thickness (e.g. from satellite) instead of area-based heuristic only.

---

## 4. Numerics and stability

- [ ] **4.1** Consider adaptive time step or smaller fixed dt when contacts are stiff.
- [ ] **4.2** Replace or soften hard velocity clamping; fix root causes of instability if needed.
- [ ] **4.3** Optionally couple rotation to cluster motion or a simple moment balance.

---

## 5. Reproducibility and structure

- [x] **5.1** Fix random seed(s) for tessellation and any stochastic parts; document in README.  
  **Implemented:** Single seed at startup (`--seed`, default 42); used for create_simple_ice_field, tessellate_all_floes, and drives assign_thickness_by_area + angular_vels noise. Documented in README.
- [x] **5.2** Add save/load of full state (positions, velocities, bonds, ship) for scenario replay.  
  **Implemented:** `--save-state FILE` writes pickle (polygons, centroids, bonds, thicknesses, positions/rotations/bonds/ship/submergence history, ship params). `--load-state FILE` skips simulation and regenerates 2D/3D animations only.
- [ ] **5.3** Refactor into modules (geometry, bonds, contact, integration, visualization) where useful.
- [ ] **5.4** Centralize parameters (config or dataclass) for easier tuning and reporting.

---

## 6. Preparing for autonomy

- [ ] **6.1** Add an observation layer (e.g. what a sensor would see) on top of full state.
- [ ] **6.2** Expose ice load (force/moment on ship) as a clear interface for ship dynamics and validation.

---

## Order of work

1. **Complete §0** (simple test setup) and confirm it runs.
2. **Work through §1** (Ice physics) using the simple test after each change.
3. Then proceed with §2–§6 as needed for navigation and validation.
