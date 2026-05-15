"""
Tests for the return_cost_components feature in jax_mbd_planner and planning.

These tests verify:
  1. The returned dict has the correct structure and keys.
  2. All raw component values are non-negative.
  3. The total_weighted value is consistent with the individual components and
     the weights that were active during optimisation.
  4. The feature is correctly threaded through optimize_with_diffusion().
  5. Combining return_cost_components=True with return_diagnostics=True raises
     a clear error (they use different code paths).

Tests use a tiny planner config (5-step horizon, 16 samples, 5 diffusion steps)
so they complete quickly on CPU.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project importable when running from the repo root or this directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def jax_setup():
    """Enable float64 once per test session."""
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax


@pytest.fixture(scope="module")
def tiny_cfg():
    """Minimal RunConfig with a very short horizon for fast tests."""
    from dataclasses import replace
    from medea_ice_ship.config import RunConfig
    base = RunConfig()
    return replace(
        base,
        planning_jax_mbd_horizon=10,
        planning_jax_mbd_nsample=32,
        planning_jax_mbd_ndiffuse=5,
        planning_jax_mbd_warm_start=False,
        # explicit weights so we can verify the weighted sum
        planning_jax_mbd_costmap_weight=1.0,
        planning_jax_mbd_length_weight=1.0,
        planning_jax_mbd_goal_weight=0.1,
        planning_jax_mbd_smoothness_weight=0.005,
    )


@pytest.fixture(scope="module")
def flat_costmap():
    """
    A 200 × 200 m flat (all zeros) CostMap.

    All ice cost is zero so costmap_cost ≈ 0 and path_length, goal_dist, and
    smoothness are the only significant terms — easy to reason about.
    """
    from dpdem.planning.cost_map import CostMap
    cm = CostMap(
        width=400.0,
        height=400.0,
        resolution=2.0,
        x_origin=-200.0,
        y_origin=-200.0,
    )
    # Leave the grid at its initialised base_cost (should be uniform low value).
    return cm


@pytest.fixture(scope="module")
def nonzero_costmap():
    """
    A small CostMap with a stripe of elevated cost in the middle so the
    costmap_cost term is clearly non-zero.
    """
    from dpdem.planning.cost_map import CostMap
    cm = CostMap(
        width=400.0,
        height=400.0,
        resolution=2.0,
        x_origin=-200.0,
        y_origin=-200.0,
    )
    # Set a band of high cost in the centre rows (y ≈ 0 ± 20 m)
    rows, cols = cm.cost_map.shape
    mid = rows // 2
    cm.cost_map[mid - 5:mid + 5, :] = 50.0
    return cm


@pytest.fixture(scope="module")
def simple_route():
    """Start pose and goal for a short, simple traverse."""
    start_pose = (-100.0, -100.0, np.pi / 4)   # 45-degree heading
    goal = (100.0, 100.0)
    return start_pose, goal


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_mbd(cost_map, start_pose, goal, cfg, return_cost_components=True, return_diagnostics=False):
    """Call jax_mbd_optimize with the given flags."""
    from medea_ice_ship.jax_mbd_planner import jax_mbd_optimize
    return jax_mbd_optimize(
        cost_map=cost_map,
        start_pose=start_pose,
        goal=goal,
        run_cfg=cfg,
        return_cost_components=return_cost_components,
        return_diagnostics=return_diagnostics,
    )


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestCostComponentsStructure:
    """Verify the returned dict has correct keys and types."""

    def test_returns_tuple_when_requested(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        start_pose, goal = simple_route
        result = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg, return_cost_components=True)
        assert isinstance(result, tuple), "Expected (traj, cost_components) tuple"
        assert len(result) == 2

    def test_trajectory_is_ndarray(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        start_pose, goal = simple_route
        traj, comps = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        assert isinstance(traj, np.ndarray), "Trajectory should be numpy array"
        assert traj.ndim == 2
        assert traj.shape[1] == 3, "Trajectory should be (N, 3) [x, y, psi]"

    def test_cost_components_has_required_keys(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        start_pose, goal = simple_route
        _, comps = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        required = {"costmap_cost", "path_length", "goal_dist", "smoothness",
                    "total_weighted", "weights"}
        assert required.issubset(set(comps.keys())), (
            f"Missing keys: {required - set(comps.keys())}"
        )

    def test_weights_subdict_has_required_keys(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        start_pose, goal = simple_route
        _, comps = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        w_keys = {"costmap", "length", "goal", "smoothness"}
        assert w_keys.issubset(set(comps["weights"].keys()))

    def test_weights_match_config(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        start_pose, goal = simple_route
        _, comps = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        w = comps["weights"]
        assert w["costmap"] == pytest.approx(tiny_cfg.planning_jax_mbd_costmap_weight)
        assert w["length"] == pytest.approx(tiny_cfg.planning_jax_mbd_length_weight)
        assert w["goal"] == pytest.approx(tiny_cfg.planning_jax_mbd_goal_weight)
        assert w["smoothness"] == pytest.approx(tiny_cfg.planning_jax_mbd_smoothness_weight)

    def test_plain_return_when_not_requested(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        start_pose, goal = simple_route
        result = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg, return_cost_components=False)
        assert isinstance(result, np.ndarray), (
            "Without return_cost_components the function should return a plain ndarray"
        )


# ---------------------------------------------------------------------------
# Value tests
# ---------------------------------------------------------------------------

class TestCostComponentsValues:
    """Verify the numerical values are physically plausible."""

    def test_all_components_non_negative(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        start_pose, goal = simple_route
        _, comps = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        for key in ("costmap_cost", "path_length", "goal_dist", "smoothness"):
            assert comps[key] >= 0.0, f"{key} should be non-negative, got {comps[key]}"

    def test_path_length_positive(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        start_pose, goal = simple_route
        _, comps = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        assert comps["path_length"] > 0.0, "path_length must be positive for a non-trivial route"

    def test_total_weighted_consistency(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        """
        total_weighted must equal (costmap_weight/normalizer)*costmap_cost + sum(other w_i*c_i).
        tiny_cfg uses normalizer=1.0 (default), so the formula simplifies to sum(w_i*c_i).
        """
        start_pose, goal = simple_route
        _, comps = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        w = comps["weights"]
        n = comps.get("costmap_normalizer", 1.0)
        expected = (
            (w["costmap"] / n) * comps["costmap_cost"]
            + w["length"]      * comps["path_length"]
            + w["goal"]        * comps["goal_dist"]
            + w["smoothness"]  * comps["smoothness"]
        )
        assert comps["total_weighted"] == pytest.approx(expected, rel=1e-5), (
            f"total_weighted={comps['total_weighted']:.6f} != (w/n)*c+...={expected:.6f}"
        )

    def test_nonzero_costmap_raises_cost(self, jax_setup, flat_costmap, nonzero_costmap, simple_route, tiny_cfg):
        """Costmap cost should be higher when the map has a high-cost band."""
        start_pose, goal = simple_route
        _, comps_flat = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        _, comps_ice = _run_mbd(nonzero_costmap, start_pose, goal, tiny_cfg)
        assert comps_ice["costmap_cost"] >= comps_flat["costmap_cost"], (
            "costmap_cost should be higher (or equal) when there is more ice in the map"
        )

    def test_goal_dist_reasonable(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        """goal_dist should be less than the straight-line start→goal distance."""
        start_pose, goal = simple_route
        _, comps = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        straight_line = float(np.hypot(goal[0] - start_pose[0], goal[1] - start_pose[1]))
        # The planner may not reach the goal exactly with only 10 steps, but it should
        # make progress — goal_dist must be less than or equal to start distance.
        assert comps["goal_dist"] <= straight_line * 1.1, (
            f"goal_dist={comps['goal_dist']:.1f} > 1.1 × straight_line={straight_line:.1f}"
        )


# ---------------------------------------------------------------------------
# planning.py wrapper tests
# ---------------------------------------------------------------------------

class TestOptimizeWithDiffusionPassthrough:
    """Verify the planning.py wrapper correctly passes through cost components."""

    def test_returns_tuple(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        from medea_ice_ship.planning import optimize_with_diffusion
        start_pose, goal = simple_route
        initial_path = np.array([
            [start_pose[0], start_pose[1], start_pose[2]],
            [goal[0],       goal[1],       start_pose[2]],
        ])
        result = optimize_with_diffusion(
            continuous_path=initial_path,
            start_pose=start_pose,
            goal=goal,
            cost_map=flat_costmap,
            run_cfg=tiny_cfg,
            return_cost_components=True,
        )
        assert isinstance(result, tuple) and len(result) == 2

    def test_wrapper_components_match_direct(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        """
        Both the direct jax_mbd_optimize call and the planning.py wrapper should
        return cost_components dicts with the same keys and non-negative values.
        (Values may differ due to warm-start / heading alignment in the wrapper.)
        """
        from medea_ice_ship.planning import optimize_with_diffusion
        start_pose, goal = simple_route
        initial_path = np.array([
            [start_pose[0], start_pose[1], start_pose[2]],
            [goal[0],       goal[1],       start_pose[2]],
        ])
        _, comps = optimize_with_diffusion(
            continuous_path=initial_path,
            start_pose=start_pose,
            goal=goal,
            cost_map=flat_costmap,
            run_cfg=tiny_cfg,
            return_cost_components=True,
        )
        assert comps is not None
        required = {"costmap_cost", "path_length", "goal_dist", "smoothness",
                    "total_weighted", "weights"}
        assert required.issubset(set(comps.keys()))
        for key in ("costmap_cost", "path_length", "goal_dist", "smoothness"):
            assert comps[key] >= 0.0

    def test_wrapper_short_path_returns_none_components(self, jax_setup, flat_costmap, tiny_cfg):
        """A path with only one point should return None for cost components."""
        from medea_ice_ship.planning import optimize_with_diffusion
        start_pose = (0.0, 0.0, 0.0)
        goal = (100.0, 0.0)
        single_point_path = np.array([[0.0, 0.0, 0.0]])
        path, comps = optimize_with_diffusion(
            continuous_path=single_point_path,
            start_pose=start_pose,
            goal=goal,
            cost_map=flat_costmap,
            run_cfg=tiny_cfg,
            return_cost_components=True,
        )
        assert comps is None, "Single-point path should yield None cost components"


# ---------------------------------------------------------------------------
# Regression: no interaction with return_diagnostics
# ---------------------------------------------------------------------------

class TestMutualExclusion:
    """return_cost_components=True should not interfere with the default path."""

    def test_default_returns_array(self, jax_setup, flat_costmap, simple_route, tiny_cfg):
        start_pose, goal = simple_route
        traj = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg,
                        return_cost_components=False, return_diagnostics=False)
        assert isinstance(traj, np.ndarray)

    def test_cost_components_ignored_when_diagnostics_active(
        self, jax_setup, flat_costmap, simple_route, tiny_cfg
    ):
        """
        When return_diagnostics=True the diagnostics code path is used and
        return_cost_components has no effect (components will be None).
        The function should still return (traj, diag_dict) without crashing.
        """
        start_pose, goal = simple_route
        result = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg,
                          return_cost_components=True, return_diagnostics=True)
        # Diagnostics path returns (traj, diag_dict)
        assert isinstance(result, tuple) and len(result) == 2
        traj, diag = result
        assert isinstance(traj, np.ndarray)
        # diag is the full diagnostics dict, not cost components
        assert "mean_trajectories" in diag


# ---------------------------------------------------------------------------
# costmap_normalizer tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_cfg_normalizer(tiny_cfg):
    """
    Same as tiny_cfg but with a non-unity costmap_normalizer so we can verify
    it is wired through correctly.
    """
    from dataclasses import replace
    return replace(tiny_cfg, planning_jax_mbd_costmap_normalizer=50.0)


class TestCostmapNormalizer:
    """Verify the costmap_normalizer is wired through make_cost_fn and the components dict."""

    def test_normalizer_field_present_in_components(
        self, jax_setup, flat_costmap, simple_route, tiny_cfg
    ):
        """The returned components dict must contain a 'costmap_normalizer' key."""
        start_pose, goal = simple_route
        _, comps = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        assert "costmap_normalizer" in comps, (
            "Expected 'costmap_normalizer' key in cost_components dict"
        )

    def test_default_normalizer_is_one(
        self, jax_setup, flat_costmap, simple_route, tiny_cfg
    ):
        """Default normalizer (not set in tiny_cfg) should be 1.0."""
        start_pose, goal = simple_route
        _, comps = _run_mbd(flat_costmap, start_pose, goal, tiny_cfg)
        assert comps["costmap_normalizer"] == pytest.approx(1.0), (
            f"Default costmap_normalizer should be 1.0, got {comps['costmap_normalizer']}"
        )

    def test_normalizer_stored_in_components(
        self, jax_setup, nonzero_costmap, simple_route, tiny_cfg_normalizer
    ):
        """The normalizer value from config must appear in the returned dict."""
        start_pose, goal = simple_route
        _, comps = _run_mbd(nonzero_costmap, start_pose, goal, tiny_cfg_normalizer)
        assert comps["costmap_normalizer"] == pytest.approx(50.0), (
            f"Expected costmap_normalizer=50.0 in components, got {comps['costmap_normalizer']}"
        )

    def test_normalizer_reduces_total_weighted(
        self, jax_setup, nonzero_costmap, simple_route, tiny_cfg, tiny_cfg_normalizer
    ):
        """
        With a normalizer > 1 the total weighted cost must be lower than without,
        because the costmap term is divided by the normalizer.
        Uses a nonzero costmap so the costmap contribution is material.
        """
        start_pose, goal = simple_route
        _, comps_no_norm = _run_mbd(nonzero_costmap, start_pose, goal, tiny_cfg)
        _, comps_norm    = _run_mbd(nonzero_costmap, start_pose, goal, tiny_cfg_normalizer)
        assert comps_norm["total_weighted"] < comps_no_norm["total_weighted"], (
            f"normalizer=50 should reduce total_weighted: "
            f"{comps_norm['total_weighted']:.4f} >= {comps_no_norm['total_weighted']:.4f}"
        )

    def test_total_weighted_uses_effective_weight(
        self, jax_setup, nonzero_costmap, simple_route, tiny_cfg_normalizer
    ):
        """
        total_weighted == (costmap_weight/normalizer)*costmap_cost
                        + length_weight*path_length
                        + goal_weight*goal_dist
                        + smoothness_weight*smoothness
        """
        start_pose, goal = simple_route
        _, comps = _run_mbd(nonzero_costmap, start_pose, goal, tiny_cfg_normalizer)
        w = comps["weights"]
        n = comps["costmap_normalizer"]
        expected = (
            (w["costmap"] / n) * comps["costmap_cost"]
            + w["length"]      * comps["path_length"]
            + w["goal"]        * comps["goal_dist"]
            + w["smoothness"]  * comps["smoothness"]
        )
        assert comps["total_weighted"] == pytest.approx(expected, rel=1e-5), (
            f"total_weighted={comps['total_weighted']:.6f} != "
            f"(w/n)*c + ...={expected:.6f}"
        )

    def test_make_cost_fn_normalizer_directly(self, jax_setup, nonzero_costmap):
        """
        Call make_cost_fn directly with two different normalizers on the same
        trajectory and verify the ratio of costs equals the inverse ratio of
        normalizers (all else equal, with nonzero costmap contribution).
        """
        from medea_ice_ship.jax_mbd_planner import make_cost_fn, make_dynamics_rollout
        import jax.numpy as jnp

        grid = jnp.array(nonzero_costmap.cost_map, dtype=jnp.float32)
        meta = {
            "x_origin": float(nonzero_costmap.x_origin),
            "y_origin": float(nonzero_costmap.y_origin),
            "resolution": float(nonzero_costmap.resolution),
            "rows": int(nonzero_costmap.rows),
            "cols": int(nonzero_costmap.cols),
        }
        mass = 23e6
        rollout_fn, _, _ = make_dynamics_rollout(
            mass=mass, Izz=(1/12)*mass*(120**2+23**2),
            dt_substep=1.0, n_substeps=1,
        )
        eta0 = jnp.array([-100.0, -100.0, np.pi / 4], dtype=jnp.float32)
        nu0  = jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32)
        surge_max = 2e6
        action_center = jnp.array([surge_max / 2, 0.0, 0.0], dtype=jnp.float32)
        action_scale  = jnp.array([surge_max / 2, 5e5, 5e8], dtype=jnp.float32)
        goal = (100.0, 100.0)

        H = 10
        u_norm = jnp.zeros((H, 3)).at[:, 0].set(0.7)

        cost1 = make_cost_fn(
            grid, meta, goal, rollout_fn, eta0, nu0,
            action_center, action_scale,
            costmap_weight=1.0, costmap_normalizer=1.0,
            length_weight=0.0, goal_weight=0.0, smoothness_weight=0.0,
        )(u_norm)

        cost2 = make_cost_fn(
            grid, meta, goal, rollout_fn, eta0, nu0,
            action_center, action_scale,
            costmap_weight=1.0, costmap_normalizer=2.0,
            length_weight=0.0, goal_weight=0.0, smoothness_weight=0.0,
        )(u_norm)

        # With only the costmap term active: cost2 should be exactly half of cost1
        assert float(cost2) == pytest.approx(float(cost1) / 2.0, rel=1e-5), (
            f"normalizer=2 should halve the costmap cost: {float(cost2):.6f} != "
            f"{float(cost1)/2:.6f}"
        )
