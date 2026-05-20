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
    and wall-excluded sites are rejected.  The Rosenbluth weight contribution
    for step i is  k_i / 6, where k_i is the number of free neighbors.
    The overall chain weight is the product of these contributions.

    Walls are impenetrable planes that the chain cannot cross or touch.
    Each wall is a tuple ``(axis, coord, side)`` where:

        axis  : int   — lattice axis perpendicular to the wall (0=x, 1=y, 2=z)
        coord : int   — wall position along that axis
        side  : str   — which side (including the wall itself) is excluded:
                        '<'  →  sites with position[axis] <= coord are blocked
                             (wall at coord, chain must stay strictly above)
                        '>'  →  sites with position[axis] >= coord are blocked
                             (wall at coord, chain must stay strictly below)

    Example — slit geometry between z=0 and z=10::

        walls = [(2, 0, '<'), (2, 10, '>')]
        sampler = LatticeRosenbluthMCMC(n_monomers=17, walls=walls)
        # chain lives in z = 1 … 9
    """

    def __init__(
        self,
        n_monomers: int,
        bond_length: float = 1.0,
        walls: Optional[List[Tuple]] = None,
        random_seed: Optional[int] = None,
    ):
        self.n_monomers  = n_monomers
        self.bond_length = bond_length
        self.walls       = walls if walls is not None else []
        if random_seed is not None:
            np.random.seed(random_seed)

    # ── wall check ────────────────────────────────────────────────────────────

    def _is_wall_excluded(self, site: np.ndarray) -> bool:
        """Return True if *site* is inside or on any wall."""
        coord = site.astype(int)
        for axis, wall_coord, side in self.walls:
            if side == '<' and coord[axis] <= wall_coord:
                return True
            if side == '>' and coord[axis] >= wall_coord:
                return True
        return False

    # ── chain growth ──────────────────────────────────────────────────────────

    def grow_chain(self, start: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, float]:
        """
        Grow one self-avoiding walk from *start* (default origin).

        Neighbors that are already occupied or blocked by a wall are excluded.

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
            free = [
                n for n in neighbors
                if tuple(n.astype(int)) not in occupied
                and not self._is_wall_excluded(n)
            ]

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
