# Equations of Motion and Time Integration — Implementation Checklist

This document maps the physics summary (Eqs 7–11, Verlet, etc.) to the code in `simulation.py` and related modules.

---

## Part 1: Equations of Motion

### Equation 7 — Translational motion (m · du/dt = Fc + Fb + Fh + G)

- **Forces** (per frame, before integration):
  - **Fc**: Contact forces — ship collision (`forces[i] += ...` in ship loop), ice–ice contact (`np.add.at(forces, pair_i, -f_n_vec)` etc.).
  - **Fb**: Bond forces — `F_bond_vec` from bond stiffness/stress, `np.add.at(forces, bond_i, F_bond_vec)` and `bond_j`.
  - **Fh**: Hydrodynamic — buoyancy in ship-contact loop; drag: `forces -= drag_force` (quadratic/linear).
  - **G**: Not applied in 2D horizontal simulation (no vertical gravity on ice in this layer).
- **Integration**: `accel = forces / masses[:, np.newaxis]`, then either:
  - **Verlet** (when `use_verlet_position_update`): `new_positions = positions + velocities*dt_ice + 0.5*accel*dt_ice**2`, `new_velocities = velocities + accel*dt_ice`.
  - **Euler**: `new_velocities = velocities + accel*dt_ice`, `new_positions = positions + new_velocities*dt_ice`.
- **Time step**: Ice uses `dt_ice` (either `dt` or Rayleigh-capped when `use_rayleigh_dt_cap`).

**Location**: `simulation.py` — force accumulation (~264–488), then per-particle block (~499–533) using `dt_ice`.

---

### Equation 8 — Rotational motion (I · dω/dt = Mc + Mb + Mh)

- **I**: Moment of inertia — `inertia = 0.5 * masses * radii**2` (disk, 2D), computed once.
- **Moments** (per frame):
  - **Mc**: Ship contact — `r_contact × F_ship`; ice–ice contact — lever `0.5*d_vec`, cross with force on `i`.
  - **Mb**: Bond — lever `0.5*d_bond_pre`, cross with `F_bond_vec`; `np.add.at(moments, bond_i, bond_cross)`.
  - **Mh**: Buoyancy moment (Mf) in ship loop; rotational drag (Md): `Md = -0.5*CMd*ρw*radii**3*ω*|ω|*A_sub`, `moments += Md`.
- **Integration**: `alpha = moments / inertia`, `angular_vels += alpha*dt_ice`, damping/noise/clip, then `rotations += angular_vels*dt_ice`.

**Location**: `simulation.py` — moments accumulated in ship and bond and ice–ice blocks; integration ~631–636.

---

## Part 2: Hydrodynamic forces (Equation 9)

- **Ff (buoyancy)**: Applied in ship-collision loop when `pen > 0`; submergence from penetration; force magnitude from buoyancy model.
- **Fd (drag)**: `drag_force` from `0.5*ρw*c_d_drag*A_drag*v²` (or linear at low speed); `forces -= drag_force`.

**Location**: `simulation.py` — buoyancy in ship loop; drag ~478–488.

---

## Part 3: Hydrodynamic moments (Equation 10)

- **Mf**: In ship-contact loop, `r_cb × F_buoy` (r_cb from contact to cell center).
- **Md**: `Md = -0.5*CMd*ρw*radii**3*ω*|ω|*A_sub`; `moments += Md` after translational drag.

**Location**: `simulation.py` — Mf in ship loop; Md ~490–493.

---

## Part 4: Time step (Equation 11)

- **Rayleigh**: `dt_rayleigh = b*π*r_min/(0.163*ν+0.877)*√(ρ/G)`; `G = E/(2(1+ν))`.
- **Usage**: When `use_rayleigh_dt_cap` is True, `dt_ice = min(dt, dt_rayleigh)`; otherwise `dt_ice = dt`.
- **Where dt_ice is used**: All ice integration (positions, velocities, rotations, bond_delta_s). Ship and frame time use `dt`.

**Location**: `simulation.py` — ~148–161 (compute and print); all ice updates use `dt_ice`.

---

## Part 5: Time integration

- **Translation**: Verlet (position then velocity) or Euler; both use `dt_ice`.
- **Rotation**: 2D scalar angle; `rotations += angular_vels*dt_ice` (no quaternions in this 2D path).
- **Bond slip**: `bond_delta_s += |v_t|*dt_ice`.

---

## Bond fracture and “instant fracture” fix

- **DPDEM damage update** runs only when **frame > 0**. At frame 0 we do not call `update_bond_damage_state`, so `bond_intact` stays all True and no bond can fail before any motion or contact.
- **In `bond_fracture.py`**:
  - Damage is applied only when stress is meaningful: `sigma_m_0_raw >= _SIGMA_MIN_FOR_DAMAGE` (1e4 Pa).
  - Damage initiation requires deformation above noise: `delta_m > _DELTA_M_MIN_INITIATE` (1e-5 m).
  - Damage increase per step is capped: `_DAMAGE_RATE_CAP` (0.15) so fracture progresses over several frames.

**Location**: `simulation.py` — DPDEM block guarded by `if frame > 0:` (~584–606); `bond_fracture.py` — `update_bond_damage_state`.

---

## History and animation

- **positions_history**: Initial state at index 0; then one entry per completed frame (so index `k` = state after step `k-1` for k≥1).
- **bonds_history**: Same indexing; with the frame-0 skip, indices 0 and 1 both have all bonds intact.
- **Animation**: Uses `frames_to_render` (e.g. 0, 2, 4, …); frame index 0 shows `positions_history[0]` and `bonds_history[0]` (initial intact ice).
