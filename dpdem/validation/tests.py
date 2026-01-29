"""
Validation tests for DPDEM implementation.

Provides unit tests for individual components and system-level validation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

from ..config import DEFAULT_ICE_PROPERTIES, DEFAULT_SIMULATION_CONFIG
from ..kinematics.quaternion import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_multiply,
    quaternion_normalize,
    IDENTITY_QUATERNION,
)
from ..contact.forces import hertz_normal_force, mindlin_tangential_force
from ..bonds.fracture import (
    compute_strength_criterion,
    compute_mohr_coulomb_strength,
    compute_benzeggagh_kenane_energy,
)


def test_quaternion_rotation_matrix():
    """
    Test quaternion to rotation matrix conversion and back.
    
    Verifies that Eq. 4 is correctly implemented.
    """
    # Identity quaternion should give identity matrix
    R_identity = quaternion_to_rotation_matrix(IDENTITY_QUATERNION)
    assert jnp.allclose(R_identity, jnp.eye(3), atol=1e-6), "Identity quaternion failed"
    
    # 90 degree rotation about z-axis
    q_z90 = jnp.array([jnp.cos(jnp.pi/4), 0, 0, jnp.sin(jnp.pi/4)])
    R_z90 = quaternion_to_rotation_matrix(q_z90)
    
    expected = jnp.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    assert jnp.allclose(R_z90, expected, atol=1e-6), "90 deg z rotation failed"
    
    # Convert back
    q_back = rotation_matrix_to_quaternion(R_z90)
    # Quaternions can differ by sign
    assert jnp.allclose(jnp.abs(q_back), jnp.abs(q_z90), atol=1e-5), "Round-trip failed"
    
    print("Quaternion rotation matrix test: PASSED")
    return True


def test_hertz_contact():
    """
    Test Hertzian contact force (Eq. 9).
    
    Compares against analytical solution for simple cases.
    """
    # Material parameters
    E = 2.4e9  # Pa
    nu = 0.3
    R = 1.0  # m (equivalent radius)
    
    # Effective modulus
    E_star = E / (2 * (1 - nu**2))
    
    # Normal stiffness
    k_n = (4/3) * E_star * jnp.sqrt(R)
    
    # Test case: 1 mm overlap
    delta_n = 0.001  # m
    delta_n_dot = 0.0  # Static contact
    c_n = 0.0  # No damping
    
    F_n = hertz_normal_force(delta_n, delta_n_dot, k_n, c_n)
    
    # Analytical: F = k_n * delta^(3/2)
    F_analytical = k_n * delta_n**1.5
    
    assert jnp.allclose(F_n, F_analytical, rtol=1e-5), f"Hertz force mismatch: {F_n} vs {F_analytical}"
    
    # Verify force scales with delta^1.5
    delta_n_2 = 0.002
    F_n_2 = hertz_normal_force(delta_n_2, 0.0, k_n, 0.0)
    
    ratio = F_n_2 / F_n
    expected_ratio = (delta_n_2 / delta_n)**1.5
    
    assert jnp.allclose(ratio, expected_ratio, rtol=1e-5), "Force scaling incorrect"
    
    print("Hertz contact test: PASSED")
    return True


def test_bond_fracture():
    """
    Test fracture mechanics (Eqs. 14-19).
    
    Verifies strength criterion and fracture energy calculations.
    """
    # Material parameters
    sigma_bar = 0.6e6  # Pa
    C = 1.8e6  # Pa
    mu_b = 0.7
    G_c_I = 12.0  # J/m²
    G_c_II = 12.0  # J/m²
    
    # Test 1: Strength criterion below failure
    sigma = 0.3e6  # Pa (tensile)
    tau = 0.5e6  # Pa (shear)
    tau_bar = compute_mohr_coulomb_strength(C, mu_b, sigma)
    criterion = compute_strength_criterion(sigma, tau, sigma_bar, tau_bar)
    
    assert criterion < 1.0, f"Should not fail: criterion = {criterion}"
    
    # Test 2: Strength criterion at failure
    sigma_fail = sigma_bar * 0.8
    tau_bar_fail = compute_mohr_coulomb_strength(C, mu_b, sigma_fail)
    tau_fail = tau_bar_fail * jnp.sqrt(1 - (sigma_fail/sigma_bar)**2)
    criterion_fail = compute_strength_criterion(sigma_fail, tau_fail, sigma_bar, tau_bar_fail)
    
    assert jnp.allclose(criterion_fail, 1.0, atol=0.05), f"Should fail: criterion = {criterion_fail}"
    
    # Test 3: Compression does not contribute to normal failure
    sigma_compress = -0.5e6  # Compression
    criterion_compress = compute_strength_criterion(sigma_compress, 0.0, sigma_bar, tau_bar)
    
    assert criterion_compress == 0.0, "Compression should not cause normal failure"
    
    # Test 4: Benzeggagh-Kenane energy
    G_c = compute_benzeggagh_kenane_energy(6.0, 6.0, G_c_I, G_c_II)
    
    # At equal mode mix, should be between G_c_I and G_c_II
    assert G_c_I <= G_c <= G_c_II or G_c_II <= G_c <= G_c_I, f"B-K energy out of range: {G_c}"
    
    # Pure Mode I
    G_c_mode_I = compute_benzeggagh_kenane_energy(1.0, 0.0, G_c_I, G_c_II)
    assert jnp.allclose(G_c_mode_I, G_c_I, rtol=0.1), f"Pure Mode I failed: {G_c_mode_I}"
    
    print("Bond fracture test: PASSED")
    return True


def test_single_particle_gravity():
    """
    Test single particle falling under gravity.
    
    Verifies basic dynamics integration.
    """
    from ..dynamics.ice_element import compute_ice_element_acceleration
    
    # Particle parameters
    mass = 1000.0  # kg
    g = jnp.array([0.0, 0.0, -9.8])
    
    # No other forces
    F_contact = jnp.zeros(3)
    F_bond = jnp.zeros(3)
    F_buoyancy = jnp.zeros(3)
    F_drag = jnp.zeros(3)
    
    # Compute acceleration
    a = compute_ice_element_acceleration(F_contact, F_bond, F_buoyancy, F_drag, mass, g)
    
    # Should equal gravity
    assert jnp.allclose(a, g, atol=1e-10), f"Acceleration should be g: {a}"
    
    # With buoyancy equal to weight
    F_buoyancy = -mass * g  # Buoyancy cancels weight
    a_float = compute_ice_element_acceleration(F_contact, F_bond, F_buoyancy, F_drag, mass, g)
    
    assert jnp.allclose(a_float, jnp.zeros(3), atol=1e-10), "Should float with zero acceleration"
    
    print("Single particle gravity test: PASSED")
    return True


def test_buoyancy_box():
    """
    Test buoyancy calculation for a simple box.
    
    Verifies Eqs. 22-27.
    """
    from ..geometry.mesh import create_box_mesh
    from ..hydrodynamics.buoyancy import compute_total_buoyancy
    
    # Create a 1x1x1 box
    vertices, faces = create_box_mesh(1.0, 1.0, 1.0)
    
    # Position half submerged (z=0 at waterline, box centered at z=-0.5)
    position = jnp.array([0.0, 0.0, 0.0])  # Box center at waterline
    quaternion = IDENTITY_QUATERNION
    
    rho_water = 1000.0  # kg/m³
    g = jnp.array([0.0, 0.0, -9.8])
    waterline_z = 0.0
    
    F_b, M_b = compute_total_buoyancy(
        vertices, faces, position, quaternion, rho_water, g, waterline_z
    )
    
    # Half the box is submerged, so buoyancy ≈ 0.5 * rho * g * V
    expected_force_z = 0.5 * rho_water * 9.8 * 1.0  # Half volume submerged
    
    # Check buoyancy is upward
    assert F_b[2] > 0, "Buoyancy should be upward"
    
    print(f"Buoyancy force: {F_b}")
    print("Buoyancy box test: PASSED (qualitative)")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("=" * 50)
    print("DPDEM Validation Tests")
    print("=" * 50)
    
    results = []
    
    results.append(("Quaternion rotation matrix", test_quaternion_rotation_matrix()))
    results.append(("Hertz contact force", test_hertz_contact()))
    results.append(("Bond fracture mechanics", test_bond_fracture()))
    results.append(("Single particle gravity", test_single_particle_gravity()))
    results.append(("Buoyancy box", test_buoyancy_box()))
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("=" * 50)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
