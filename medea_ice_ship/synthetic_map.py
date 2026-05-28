"""
Synthetic rectangular ice field for the experiment suite.

Map geometry (fixed):
    300 m wide (x) × 750 m long (y)
    x ∈ [0, 300],  y ∈ [0, 750]
    Ship start : (150, 50),  heading = +y (π/2 rad)
    Goal line  : y = 700

Block A — full factorial (no currents and with currents):
    concentrations : [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    thickness modes: uniform_thin (1.0 m), uniform_thick (3.0 m),
                     bimodal (80 % at 1.0 m + 20 % at 3.0 m)
    currents       : False, True
    → 6 × 3 × 2 = 36 scenarios

Block B — concentration trend (no currents):
    concentrations : [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    thickness modes: uniform_thin, uniform_thick, bimodal
    → 7 × 3 = 21 scenarios

Block C — Block A grid with rigid floes (no tessellation, no fracture):
    Identical scenario factorial to Block A, but each satellite floe is
    treated as a single rigid body (no internal Voronoi cells, no bonds),
    so fracture cannot occur. Used to isolate the effect of fracture on
    planner performance. → 36 scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from medea_ice_ship.ice_field_generator import generate_ice_field
from medea_ice_ship.ocean_current import NoCurrentModel, OceanCurrentModel, UniformCurrentModel

# ── Fixed map geometry ─────────────────────────────────────────────────────
MAP_LENGTH: float = 750.0   # m  (y-axis, direction of travel)
MAP_WIDTH:  float = 300.0   # m  (x-axis)

X_LO: float = 0.0
X_HI: float = MAP_WIDTH          #  300
Y_LO: float = 0.0
Y_HI: float = MAP_LENGTH         #  750

SHIP_START = np.array([MAP_WIDTH / 2, 50.0])   # (150, 50) — centre of map
GOAL_Y_LINE: float = 700.0
GOAL_POINT  = (MAP_WIDTH / 2, GOAL_Y_LINE)      # (150, 700)
START_HEADING: float = np.pi / 2   # pointing along +y

FLOE_RADIUS_MIN: float = 3.0
FLOE_RADIUS_MAX: float = 25.0

# Uniform synthetic current for "with currents" scenarios
CURRENT_U: float = 0.3   # m/s in +x
CURRENT_V: float = 0.0   # m/s in +y

# ── Scenario grids ─────────────────────────────────────────────────────────
BLOCK_A_CONCENTRATIONS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BLOCK_B_CONCENTRATIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

THICKNESS_MODES = ["uniform_thin", "uniform_thick", "bimodal"]

BIMODAL_THICK_FRACTION = 0.20   # 20 % of floes get 3.0 m
BIMODAL_THICK_VALUE    = 3.0    # m
BIMODAL_THIN_VALUE     = 1.0    # m
UNIFORM_THIN_VALUE     = 1.0    # m
UNIFORM_THICK_VALUE    = 3.0    # m


@dataclass(frozen=True)
class ScenarioSpec:
    block:          str    # 'A', 'B', or 'C'
    scenario_id:    str    # human-readable key, e.g. 'A_conc05_thin_curr'
    concentration:  float
    thickness_mode: str    # 'uniform_thin' | 'uniform_thick' | 'bimodal'
    with_currents:  bool
    no_fracture:    bool = False   # If True, skip tessellation + bonds: each
                                   # satellite floe stays as one rigid body.


def _scenario_id(block: str, conc: float, tm: str, curr: bool) -> str:
    c   = f"conc{int(round(conc * 100)):02d}"
    t   = {"uniform_thin": "thin", "uniform_thick": "thick", "bimodal": "bimodal"}[tm]
    sfx = "_curr" if curr else ""
    return f"{block}_{c}_{t}{sfx}"


def list_scenarios(block: str) -> List[ScenarioSpec]:
    """Return all ScenarioSpec objects for the requested block."""
    if block == "A":
        return [
            ScenarioSpec(
                block="A",
                scenario_id=_scenario_id("A", conc, tm, curr),
                concentration=conc,
                thickness_mode=tm,
                with_currents=curr,
            )
            for conc in BLOCK_A_CONCENTRATIONS
            for tm   in THICKNESS_MODES
            for curr in [False, True]
        ]
    if block == "B":
        return [
            ScenarioSpec(
                block="B",
                scenario_id=_scenario_id("B", conc, tm, False),
                concentration=conc,
                thickness_mode=tm,
                with_currents=False,
            )
            for conc in BLOCK_B_CONCENTRATIONS
            for tm   in THICKNESS_MODES
        ]
    if block == "C":
        # Block A grid, rigid floes (no_fracture=True)
        return [
            ScenarioSpec(
                block="C",
                scenario_id=_scenario_id("C", conc, tm, curr),
                concentration=conc,
                thickness_mode=tm,
                with_currents=curr,
                no_fracture=True,
            )
            for conc in BLOCK_A_CONCENTRATIONS
            for tm   in THICKNESS_MODES
            for curr in [False, True]
        ]
    raise ValueError(f"Unknown block: {block!r}  (expected 'A', 'B', or 'C')")


def get_scenario(block: str, scenario_id: str) -> ScenarioSpec:
    """Look up a single ScenarioSpec by its scenario_id string."""
    for spec in list_scenarios(block):
        if spec.scenario_id == scenario_id:
            return spec
    raise ValueError(f"Scenario {scenario_id!r} not found in block {block!r}")


ICE_Y_START: float = 100.0   # m — clear water below ship start
ICE_Y_END:   float = GOAL_Y_LINE  # m — no ice past the goal line


def make_ice_field(spec: ScenarioSpec, seed: int) -> dict:
    """
    Generate an ice field for the given scenario and random seed.

    Ice is placed in x ∈ [0, MAP_WIDTH] and y ∈ [ICE_Y_START, ICE_Y_END].
    generate_ice_field() centers x at 0, so we shift by +MAP_WIDTH/2 afterwards.

    Returns the generate_ice_field() dict (keys: polygons, centroids,
    thicknesses, concentration, domain) with thicknesses overridden
    according to the scenario's thickness_mode.

    A separate RNG (seed + 10_000) is used for thickness assignment so
    that the ice layout (positions, shapes) is determined solely by seed.
    """
    ice_height = ICE_Y_END - ICE_Y_START  # 600 m

    field = generate_ice_field(
        concentration=spec.concentration,
        map_shape=(ice_height, MAP_WIDTH),
        ice_start_y=ICE_Y_START,
        min_radius=FLOE_RADIUS_MIN,
        max_radius=FLOE_RADIUS_MAX,
        seed=seed,
    )

    # Shift x from [-MAP_WIDTH/2, +MAP_WIDTH/2] to [0, MAP_WIDTH]
    x_shift = np.array([MAP_WIDTH / 2.0, 0.0])
    field["polygons"]  = [p + x_shift for p in field["polygons"]]
    field["centroids"] = field["centroids"] + x_shift
    field["domain"]    = (X_LO, X_HI, ICE_Y_START, ICE_Y_END)

    n = len(field["polygons"])
    rng = np.random.RandomState(seed + 10_000)

    if spec.thickness_mode == "uniform_thin":
        field["thicknesses"] = np.full(n, UNIFORM_THIN_VALUE, dtype=np.float64)
    elif spec.thickness_mode == "uniform_thick":
        field["thicknesses"] = np.full(n, UNIFORM_THICK_VALUE, dtype=np.float64)
    elif spec.thickness_mode == "bimodal":
        mask = rng.random(n) < BIMODAL_THICK_FRACTION
        field["thicknesses"] = np.where(mask, BIMODAL_THICK_VALUE, BIMODAL_THIN_VALUE).astype(np.float64)

    return field


def make_current_model(spec: ScenarioSpec) -> OceanCurrentModel:
    """Return the ocean-current model for this scenario."""
    if spec.with_currents:
        return UniformCurrentModel(CURRENT_U, CURRENT_V)
    return NoCurrentModel()
