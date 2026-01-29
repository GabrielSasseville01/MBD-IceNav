"""
Bond model module for DPDEM.

Handles inter-element bonds, elastic/viscous stresses, and bond forces.
"""

from .model import (
    compute_bond_strain,
    compute_elastic_stress,
    compute_viscous_stress,
    compute_bond_force,
    initialize_bonds,
)
from .fracture import (
    compute_strength_criterion,
    compute_mohr_coulomb_strength,
    compute_damage_variable,
    compute_benzeggagh_kenane_energy,
    check_bond_failure,
    update_bond_state,
)
