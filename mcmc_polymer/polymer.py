"""Polymer configuration classes and utilities."""

import numpy as np
from typing import Tuple, Optional


class PolymerChain:
    """Represents a polymer chain configuration."""
    
    def __init__(self, n_monomers: int, bond_length: float = 1.0, 
                 excluded_volume_radius: float = 0.5):
        """
        Initialize polymer chain.
        
        Parameters
        ----------
        n_monomers : int
            Number of monomers in the chain
        bond_length : float
            Length of each bond between monomers
        excluded_volume_radius : float
            Excluded volume radius for hard sphere interactions
        """
        self.n_monomers = n_monomers
        self.bond_length = bond_length
        self.excluded_volume_radius = excluded_volume_radius
        
        # Initialize positions (3D coordinates)
        self.positions = np.zeros((n_monomers, 3))
        self.weight = 1.0
    
    def get_distance(self, i: int, j: int) -> float:
        """Calculate distance between monomers i and j."""
        return np.linalg.norm(self.positions[i] - self.positions[j])
    
    def check_overlap(self, monomer_idx: int) -> bool:
        """
        Check if monomer at monomer_idx overlaps with any other monomer.
        
        Returns
        -------
        bool
            True if overlap detected, False otherwise
        """
        for i in range(monomer_idx):
            distance = self.get_distance(i, monomer_idx)
            if distance < 2 * self.excluded_volume_radius:
                return True
        return False
    
    def copy(self) -> 'PolymerChain':
        """Create a copy of the polymer chain."""
        chain = PolymerChain(self.n_monomers, self.bond_length, 
                            self.excluded_volume_radius)
        chain.positions = self.positions.copy()
        chain.weight = self.weight
        return chain
