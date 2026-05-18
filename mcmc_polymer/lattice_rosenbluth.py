"""Rosenbluth algorithm on a simple cubic lattice (Rosenbluth & Rosenbluth, 1955)."""

import numpy as np
from typing import List, Optional, Tuple


# Six unit directions on the simple cubic lattice
_DIRECTIONS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=float)
_Z   = 6  # coordination number (used for the first bond: no backwards direction)
_Z1  = 5  # z - 1 (used for all subsequent bonds: backwards direction always excluded)


class LatticeRosenbluthMCMC:
    """
    Rosenbluth chain-growth sampler on a simple cubic lattice.

    At each step the six nearest-neighbor sites are inspected; occupied sites
    are excluded (self-avoiding constraint).  The Rosenbluth weight contribution
    for step i is  k_i / 6, where k_i is the number of free neighbors.
    The overall chain weight is the product of these contributions.
    """

    def __init__(self, n_monomers: int, bond_length: float = 1.0,
                 random_seed: Optional[int] = None):
        self.n_monomers = n_monomers
        self.bond_length = bond_length
        if random_seed is not None:
            np.random.seed(random_seed)

    def grow_chain(self, start: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, float]:
        """
        Grow one self-avoiding walk from *start* (default origin).

        Returns
        -------
        positions : ndarray, shape (n_monomers, 3)
        weight    : float  — Rosenbluth weight; 0.0 if a dead-end was reached
        """
        if start is None:
            start = np.zeros(3)

        positions = np.zeros((self.n_monomers, 3))
        positions[0] = start
        occupied: set = {tuple(start.astype(int))}
        weight = 1.0

        for i in range(1, self.n_monomers):
            neighbors = positions[i - 1] + self.bond_length * _DIRECTIONS
            free = [n for n in neighbors if tuple(n.astype(int)) not in occupied]

            if not free:
                return positions, 0.0

            weight *= len(free) / (_Z if i == 1 else _Z1)
            chosen = free[np.random.randint(len(free))]
            positions[i] = chosen
            occupied.add(tuple(chosen.astype(int)))

        return positions, weight

    def sample(self, n_chains: int, start: Optional[np.ndarray] = None
               ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Draw *n_chains* independent Rosenbluth-weighted samples.

        Returns
        -------
        chains  : list of ndarray, each shape (n_monomers, 3)
        weights : ndarray of Rosenbluth weights for accepted chains
        """
        if start is None:
            start = np.zeros(3)

        chains, weights = [], []
        for _ in range(n_chains):
            pos, w = self.grow_chain(start)
            chains.append(pos)
            weights.append(w)
        return chains, np.array(weights)
