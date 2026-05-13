"""Rosenbluth algorithm implementation for polymer MCMC simulation."""

import numpy as np
from typing import List, Tuple
from .polymer import PolymerChain


class RosenbluthMCMC:
    """MCMC sampler using the Rosenbluth algorithm for polymer chains."""
    
    def __init__(self, n_monomers: int = 100, bond_length: float = 1.0,
                 excluded_volume_radius: float = 0.5, n_trial_positions: int = 10,
                 random_seed: Optional[int] = None):
        """
        Initialize Rosenbluth MCMC sampler.
        
        Parameters
        ----------
        n_monomers : int
            Number of monomers in the polymer chain
        bond_length : float
            Length of bonds between consecutive monomers
        excluded_volume_radius : float
            Excluded volume radius for hard sphere interactions
        n_trial_positions : int
            Number of trial positions to sample for each segment
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.n_monomers = n_monomers
        self.bond_length = bond_length
        self.excluded_volume_radius = excluded_volume_radius
        self.n_trial_positions = n_trial_positions
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_trial_positions(self, current_position: np.ndarray,
                                 n_positions: int) -> np.ndarray:
        """
        Generate trial positions for next segment.
        
        Positions are sampled uniformly on a sphere of radius bond_length
        centered at current_position.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current monomer position
        n_positions : int
            Number of trial positions to generate
        
        Returns
        -------
        np.ndarray
            Array of shape (n_positions, 3) containing trial positions
        """
        # Generate random unit vectors
        u = np.random.randn(n_positions, 3)
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        
        # Scale by bond length and add to current position
        trial_positions = current_position + self.bond_length * u
        return trial_positions
    
    def grow_chain(self, chain: PolymerChain, start_segment: int = 1) -> Tuple[PolymerChain, float]:
        """
        Grow polymer chain using Rosenbluth method.
        
        Parameters
        ----------
        chain : PolymerChain
            Initial polymer chain (can be partially grown)
        start_segment : int
            Starting segment index (default 1, after first monomer)
        
        Returns
        -------
        chain : PolymerChain
            Grown polymer chain
        rosenbluth_weight : float
            Weight of the generated configuration
        """
        chain = chain.copy()
        rosenbluth_weight = 1.0
        
        for segment_idx in range(start_segment, self.n_monomers):
            # Generate trial positions
            trial_positions = self.generate_trial_positions(
                chain.positions[segment_idx - 1], 
                self.n_trial_positions
            )
            
            # Count valid positions (no overlaps)
            valid_positions = []
            for pos in trial_positions:
                chain.positions[segment_idx] = pos
                if not chain.check_overlap(segment_idx):
                    valid_positions.append(pos)
            
            if not valid_positions:
                # No valid positions found - configuration rejected
                return chain, 0.0
            
            # Randomly select one valid position
            valid_positions = np.array(valid_positions)
            selected_idx = np.random.randint(len(valid_positions))
            chain.positions[segment_idx] = valid_positions[selected_idx]
            
            # Update Rosenbluth weight
            rosenbluth_weight *= len(valid_positions) / self.n_trial_positions
        
        chain.weight = rosenbluth_weight
        return chain, rosenbluth_weight
    
    def run(self, n_steps: int, initial_chain: Optional[PolymerChain] = None) -> List[PolymerChain]:
        """
        Run MCMC simulation.
        
        Parameters
        ----------
        n_steps : int
            Number of MCMC steps
        initial_chain : PolymerChain, optional
            Initial chain configuration. If None, a random chain is generated.
        
        Returns
        -------
        trajectories : List[PolymerChain]
            List of accepted configurations
        """
        if initial_chain is None:
            # Generate initial chain
            current_chain = PolymerChain(self.n_monomers, self.bond_length,
                                        self.excluded_volume_radius)
            current_chain, _ = self.grow_chain(current_chain)
        else:
            current_chain = initial_chain.copy()
        
        trajectories = [current_chain.copy()]
        n_accepted = 0
        
        for step in range(n_steps):
            # Generate trial chain
            trial_chain, trial_weight = self.grow_chain(current_chain)
            
            # Metropolis acceptance criterion based on Rosenbluth weights
            if trial_weight > 0:
                acceptance_prob = min(1.0, trial_weight / current_chain.weight)
                
                if np.random.rand() < acceptance_prob:
                    current_chain = trial_chain
                    n_accepted += 1
            
            trajectories.append(current_chain.copy())
        
        acceptance_rate = n_accepted / n_steps
        print(f"Acceptance rate: {acceptance_rate:.3f}")
        
        return trajectories
