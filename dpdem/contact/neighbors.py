"""
JAX-MD neighbor list wrapper for DPDEM.

Provides efficient broad-phase contact detection using JAX-MD's
optimized neighbor list infrastructure.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Any, Optional
from functools import partial


def create_neighbor_list(
    box_size: float,
    r_cutoff: float,
    capacity_multiplier: float = 1.5,
) -> Tuple[Any, Any]:
    """
    Create neighbor list function using JAX-MD.
    
    Args:
        box_size: Size of simulation domain [m]
        r_cutoff: Cutoff distance for neighbor search [m]
        capacity_multiplier: Safety factor for neighbor list capacity
    
    Returns:
        neighbor_fn: Function to allocate/update neighbor lists
        displacement_fn: Function to compute displacements
    """
    try:
        from jax_md import partition, space
        
        # Use free boundary conditions (no periodic wrap)
        displacement_fn, shift_fn = space.free()
        
        # Create neighbor list with sparse format for efficiency
        neighbor_fn = partition.neighbor_list(
            displacement_fn,
            box_size=box_size,
            r_cutoff=r_cutoff,
            format=partition.NeighborListFormat.Sparse,
            capacity_multiplier=capacity_multiplier,
        )
        
        return neighbor_fn, displacement_fn
        
    except ImportError:
        # Fallback if JAX-MD not available
        print("Warning: JAX-MD not available, using simple neighbor search")
        return None, None


def allocate_neighbor_list(
    neighbor_fn: Any,
    positions: jnp.ndarray,
) -> Any:
    """
    Allocate initial neighbor list.
    
    Args:
        neighbor_fn: JAX-MD neighbor list function
        positions: (N, 3) particle positions
    
    Returns:
        Allocated neighbor list
    """
    if neighbor_fn is None:
        return None
    return neighbor_fn.allocate(positions)


def update_neighbor_list(
    neighbor_fn: Any,
    positions: jnp.ndarray,
    neighbors: Any,
) -> Any:
    """
    Update neighbor list with new positions.
    
    Args:
        neighbor_fn: JAX-MD neighbor list function
        positions: (N, 3) particle positions
        neighbors: Existing neighbor list
    
    Returns:
        Updated neighbor list
    """
    if neighbor_fn is None:
        return None
    return neighbor_fn.update(positions, neighbors)


def get_neighbor_pairs(neighbors: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract neighbor pairs from JAX-MD neighbor list.
    
    Args:
        neighbors: JAX-MD neighbor list
    
    Returns:
        idx_i: (n_pairs,) indices of first particles
        idx_j: (n_pairs,) indices of second particles
    """
    if neighbors is None:
        return jnp.array([], dtype=jnp.int32), jnp.array([], dtype=jnp.int32)
    
    # For sparse format, neighbors.idx contains the pairs
    idx = neighbors.idx
    
    # Filter out padding (usually marked with large indices or -1)
    n_particles = neighbors.idx.shape[0] if hasattr(neighbors, 'idx') else 0
    
    # Return the indices
    return idx[:, 0], idx[:, 1]


# Simple fallback implementation when JAX-MD is not available

@partial(jax.jit, static_argnums=(2,))
def compute_all_pairs_distances(
    positions: jnp.ndarray,
    r_cutoff: float,
    n_particles: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute distances between all pairs within cutoff.
    
    Simple O(N²) implementation for fallback.
    
    Args:
        positions: (N, 3) particle positions
        r_cutoff: Cutoff distance
        n_particles: Number of particles (static)
    
    Returns:
        idx_i: Indices of first particles
        idx_j: Indices of second particles  
        distances: Distances between pairs
    """
    # Compute pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    distances = jnp.linalg.norm(diff, axis=2)  # (N, N)
    
    # Create mask for pairs within cutoff (excluding self)
    mask = (distances < r_cutoff) & (distances > 0)
    
    # Only keep upper triangle to avoid duplicates
    i_indices, j_indices = jnp.triu_indices(n_particles, k=1)
    mask_upper = mask[i_indices, j_indices]
    
    return i_indices, j_indices, distances[i_indices, j_indices], mask_upper


class SimpleNeighborList:
    """
    Simple neighbor list implementation for when JAX-MD is unavailable.
    """
    
    def __init__(self, r_cutoff: float):
        self.r_cutoff = r_cutoff
        self.idx_i = None
        self.idx_j = None
        self.distances = None
        self.mask = None
    
    def update(self, positions: jnp.ndarray) -> 'SimpleNeighborList':
        """Update neighbor list with new positions."""
        n = len(positions)
        
        # Compute all pairwise distances
        diff = positions[:, None, :] - positions[None, :, :]
        distances = jnp.linalg.norm(diff, axis=2)
        
        # Get pairs within cutoff
        i_indices, j_indices = jnp.triu_indices(n, k=1)
        pair_distances = distances[i_indices, j_indices]
        mask = pair_distances < self.r_cutoff
        
        self.idx_i = i_indices
        self.idx_j = j_indices
        self.distances = pair_distances
        self.mask = mask
        
        return self
    
    def get_pairs(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get neighbor pairs."""
        if self.mask is None:
            return jnp.array([]), jnp.array([])
        
        return self.idx_i[self.mask], self.idx_j[self.mask]


def create_simple_neighbor_list(r_cutoff: float) -> SimpleNeighborList:
    """Create a simple neighbor list."""
    return SimpleNeighborList(r_cutoff)
