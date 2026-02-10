"""Ice and ship physical properties; 2D rotation and angle helpers."""

import numpy as np


class IceProperties:
    """Physical properties of sea ice."""
    ICE_DENSITY = 920  # kg/m³
    WATER_DENSITY = 1025  # kg/m³
    FLEXURAL_STRENGTH_BASE = 0.3e6  # Pa (reduced for visible fracturing)
    YOUNGS_MODULUS = 5e9  # Pa

    @staticmethod
    def flexural_strength(thickness):
        h_ref = 1.0
        return IceProperties.FLEXURAL_STRENGTH_BASE * np.sqrt(thickness / h_ref)

    @staticmethod
    def breaking_force(thickness, width):
        """Force required to break ice in bending."""
        sigma_f = IceProperties.flexural_strength(thickness)
        return sigma_f * width * thickness ** 2


class ShipProperties:
    """Physical properties of the ship."""
    def __init__(self, length, beam, draft, displacement=None):
        self.length = length
        self.beam = beam
        self.draft = draft

        if displacement is None:
            block_coef = 0.6
            self.displacement = length * beam * draft * block_coef * IceProperties.WATER_DENSITY
        else:
            self.displacement = displacement

        self.mass = self.displacement

    def impact_force(self, velocity, ice_thickness):
        """Estimate ice impact force."""
        C_force = 5.0
        return C_force * (self.mass ** 0.7) * (velocity ** 0.8) * (ice_thickness ** 0.5)


def R_xy(psi):
    """Rotation matrix body -> world for 2D (Fossen / AUTO-IceNav). (x_world, y_world)^T = R_xy(psi) @ (u, v)^T."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def ssa(angle):
    """Smallest-signed angle in [-pi, pi) (AUTO-IceNav gnc.ssa)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
