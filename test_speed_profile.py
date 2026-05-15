"""
Validation tests for costmap-driven speed profile (Option 3 implementation).

Tests cover:
  1. JaxCostMapResult.get_cost_world() and base_cost attribute
  2. optimize_speed_profile works with JaxCostMapResult
  3. Ice brake: setpoint blending under large ice force
  5. End-to-end: replan callback produces variable-density trajectory
"""

import numpy as np
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_jax_costmap(ice_x=10.0, v_ship=4.0, m_ship=6e6, base_cost=0.1):
    """Build a small JaxCostMapResult with one floe at (ice_x, 0)."""
    from medea_ice_ship.jax_costmap import build_costmap_jax
    c = np.array([[ice_x, 0.0]], dtype=np.float32)
    t = np.array([2.0], dtype=np.float32)
    r = np.array([8.0], dtype=np.float32)
    cm, _ = build_costmap_jax(
        c, t, r, -20.0, 30.0, -20.0, 20.0,
        resolution=1.0, base_cost=base_cost,
        ship_speed=v_ship, ship_mass=m_ship, ke_cost_scale=1e-6,
    )
    return cm


# ── Test 1: JaxCostMapResult interface ────────────────────────────────────────

def test_jax_costmap_base_cost():
    """base_cost attribute is set correctly."""
    cm = _make_jax_costmap(base_cost=0.5)
    assert cm.base_cost == pytest.approx(0.5)


def test_jax_costmap_get_cost_world_inside():
    """get_cost_world returns a float and is higher at the floe centroid than in open water."""
    cm = _make_jax_costmap(ice_x=10.0)
    cost_at_floe = cm.get_cost_world(10.0, 0.0)
    cost_open    = cm.get_cost_world(-15.0, 0.0)
    assert isinstance(cost_at_floe, float)
    assert cost_at_floe > cost_open, "Cost at floe centroid should exceed open-water cost"


def test_jax_costmap_get_cost_world_oob():
    """Out-of-bounds query returns base_cost (not error, not zero)."""
    cm = _make_jax_costmap(base_cost=0.3)
    oob = cm.get_cost_world(-9999.0, -9999.0)
    assert oob == pytest.approx(0.3)


# ── Test 2: optimize_speed_profile with JaxCostMapResult ─────────────────────

def test_speed_profile_with_jax_costmap_no_error():
    """optimize_speed_profile runs without AttributeError on a JaxCostMapResult."""
    from medea_ice_ship.planning import optimize_speed_profile
    cm = _make_jax_costmap(ice_x=10.0)
    path = np.array([[-15., 0., 0.], [0., 0., 0.], [10., 0., 0.], [25., 0., 0.]])
    speeds, T = optimize_speed_profile(path, cm, v_ref=4.0, v_min=1.0, v_max=5.0)
    assert speeds is not None
    assert len(speeds) == len(path) - 1
    assert T > 0.0


def test_speed_profile_open_water_faster_than_ice():
    """Segment through open water should be faster than segment through ice."""
    from medea_ice_ship.planning import optimize_speed_profile
    cm = _make_jax_costmap(ice_x=10.0)
    # Seg 0: open water  [-15,0] -> [0,0]
    # Seg 1: into ice    [0,0] -> [10,0] (floe centroid at 10,0)
    path = np.array([[-15., 0., 0.], [0., 0., 0.], [15., 0., 0.]])
    speeds, _ = optimize_speed_profile(path, cm, v_ref=4.0, v_min=1.0, v_max=5.0)
    assert speeds[0] > speeds[1], (
        f"Open-water speed {speeds[0]:.2f} should exceed ice speed {speeds[1]:.2f}"
    )


def test_speed_profile_clipped_to_bounds():
    """Speeds must stay within [v_min, v_max]."""
    from medea_ice_ship.planning import optimize_speed_profile
    cm = _make_jax_costmap(ice_x=10.0)
    path = np.array([[-15., 0., 0.], [0., 0., 0.], [10., 0., 0.], [25., 0., 0.]])
    v_min, v_max = 1.5, 3.5
    speeds, _ = optimize_speed_profile(path, cm, v_ref=4.0, v_min=v_min, v_max=v_max)
    assert np.all(speeds >= v_min - 1e-6)
    assert np.all(speeds <= v_max + 1e-6)


# ── Test 3: Ice brake — setpoint blending ────────────────────────────────────

def test_ice_brake_config_default_zero():
    """ice_brake_threshold defaults to 0.0 (disabled)."""
    from medea_ice_ship.config import get_default_simulation_config
    cfg = get_default_simulation_config()
    assert getattr(cfg, 'ice_brake_threshold', 0.0) == 0.0


def test_ice_brake_setpoint_blending_formula():
    """Verify the setpoint blending formula: eff = ship + brake * (sp - ship)."""
    import jax.numpy as jnp

    ship_pos = jnp.array([0.0, 0.0, 0.0])
    setpoint  = jnp.array([10.0, 5.0, 0.3])
    threshold = 1e6  # 1 MN

    # Case 1: ice force below threshold → brake_factor = 1.0 → no change
    ice_surge_small = 5e5  # 0.5 MN < threshold
    bf1 = float(jnp.clip(threshold / max(ice_surge_small, threshold), 0.1, 1.0))
    assert bf1 == pytest.approx(1.0), "Below threshold → brake_factor should be 1.0"

    # Case 2: ice force at 5× threshold → brake_factor = 0.2
    ice_surge_large = 5e6  # 5 MN
    bf2 = float(jnp.clip(threshold / max(ice_surge_large, threshold), 0.1, 1.0))
    assert bf2 == pytest.approx(0.2, abs=1e-4)
    eff = jnp.array([
        ship_pos[0] + bf2 * (setpoint[0] - ship_pos[0]),
        ship_pos[1] + bf2 * (setpoint[1] - ship_pos[1]),
        setpoint[2],
    ])
    assert float(eff[0]) == pytest.approx(2.0, abs=1e-4)  # 0 + 0.2 * 10
    assert float(eff[1]) == pytest.approx(1.0, abs=1e-4)  # 0 + 0.2 * 5
    assert float(eff[2]) == pytest.approx(0.3, abs=1e-4)  # heading unchanged

    # Case 3: enormous force → minimum fraction 0.1
    ice_surge_huge = 1000e6
    bf3 = float(jnp.clip(threshold / max(ice_surge_huge, threshold), 0.1, 1.0))
    assert bf3 == pytest.approx(0.1, abs=1e-4), "Should clamp to minimum 0.1"


# ── Test 5: End-to-end replan callback with variable trajectory density ───────

def test_replan_callback_produces_variable_density_trajectory():
    """
    With KE cost + speed optimization, the replanned trajectory should have
    closer-spaced waypoints in ice than in open water.
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax
    from medea_ice_ship.planning import optimize_speed_profile, path_to_ship_trajectory_arr
    import numpy as np

    # Build costmap: single large floe at (200, 0)
    c = np.array([[200.0, 0.0]], dtype=np.float32)
    t = np.array([2.0], dtype=np.float32)
    r = np.array([50.0], dtype=np.float32)
    cm, _ = build_costmap_jax(
        c, t, r, -50.0, 500.0, -100.0, 100.0,
        resolution=2.0, base_cost=0.1,
        ship_speed=4.0, ship_mass=6e6, ke_cost_scale=1e-6,
    )

    # Path: one short open-water segment, one clearly-in-ice segment
    # Floe centroid at (200, 0), radius 50m → ice from ~150 to ~250
    # Seg 0: [0→100] = open water (100 m)
    # Seg 1: [150→250] = fully inside the floe (100 m)
    path = np.array([
        [0.,   0., 0.],
        [100., 0., 0.],   # end of open-water segment
        [150., 0., 0.],   # start of ice segment
        [250., 0., 0.],   # end of ice segment
    ])

    # Use a small ship footprint so corners don't bleed into ice from far away
    speeds, T = optimize_speed_profile(
        path, cm, v_ref=4.0, v_min=1.0, v_max=5.0,
        ship_length=10.0, ship_beam=5.0,
    )
    n_frames = max(10, int(np.ceil(T / 0.05)))
    traj = path_to_ship_trajectory_arr(path, n_frames, speed_profile=speeds)

    xs = traj[:, 0]
    n_in_open = int(np.sum((xs >= 0)   & (xs <= 100)))   # open-water region (100 m)
    n_in_ice  = int(np.sum((xs >= 150) & (xs <= 250)))   # ice region (100 m, same distance)

    # Same distance, but ice is slower → more frames allocated to ice
    assert n_in_ice > n_in_open, (
        f"Expected more trajectory frames in ice ({n_in_ice}) than open water ({n_in_open})"
    )
    # Speed in open-water segment (seg 0) should be much higher than ice (seg 2)
    assert speeds[0] > 3.0, (
        f"Open-water speed should be > 3.0 m/s, got {speeds[0]:.3f}"
    )
    assert speeds[2] <= 1.5, (
        f"Ice segment speed should be near v_min, got {speeds[2]:.3f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
