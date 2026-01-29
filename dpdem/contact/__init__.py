"""
Contact detection and force computation module for DPDEM.

Handles neighbor lists, contact detection between dilated polyhedra,
and Hertz-Mindlin contact forces.
"""

from .neighbors import (
    create_neighbor_list,
    update_neighbor_list,
    get_neighbor_pairs,
)
from .detection import (
    compute_contact,
    compute_contact_point,
    compute_contact_normal,
    detect_all_contacts,
)
from .forces import (
    hertz_normal_force,
    mindlin_tangential_force,
    compute_contact_force,
    compute_contact_moment,
)
