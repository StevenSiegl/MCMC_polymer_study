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


class MultiChainRosenbluthMCMC:
    """
    Joint Rosenbluth chain-growth sampler for multiple self-avoiding chains.

    All chains grow simultaneously in round-robin order: at monomer index i,
    chain 0 places its i-th monomer first, then chain 1, then chain 2, etc.
    The shared excluded-site set is updated after every placed monomer, so
    inter-chain excluded volume is enforced throughout growth — chains cannot
    overlap each other or any wall.

    The joint Rosenbluth weight is the product of all per-step weights:

        W = prod_{i, k}  free_neighbors_{i,k} / z

    where free_neighbors_{i,k} is the number of unoccupied, non-wall sites
    available to chain k at growth step i.

    Parameters
    ----------
    n_chains : int
        Number of chains to grow simultaneously.
    n_monomers : int or list of int
        Number of monomers per chain.  Pass a single int to give all chains
        the same length, or a list of length *n_chains* to set each chain's
        length individually.
    bond_length : float
        Lattice step size (default 1.0).
    walls : list of (axis, coord, side) tuples, optional
        Hard walls shared by all chains.  Same convention as
        LatticeRosenbluthMCMC: side='<' excludes coord and below,
        side='>' excludes coord and above.
    random_seed : int, optional
    """

    def __init__(
        self,
        n_chains: int,
        n_monomers,          # int or list[int]
        bond_length: float = 1.0,
        walls: Optional[List[Tuple]] = None,
        random_seed: Optional[int] = None,
    ):
        if n_chains < 1:
            raise ValueError("n_chains must be >= 1")

        # normalise n_monomers to a per-chain list
        if isinstance(n_monomers, int):
            chain_lengths = [n_monomers] * n_chains
        else:
            chain_lengths = list(n_monomers)
            if len(chain_lengths) != n_chains:
                raise ValueError(
                    f"n_monomers list length ({len(chain_lengths)}) "
                    f"must match n_chains ({n_chains})"
                )

        self.n_chains      = n_chains
        self.chain_lengths = chain_lengths   # list of per-chain monomer counts
        self.bond_length   = bond_length
        self.walls         = walls if walls is not None else []
        if random_seed is not None:
            np.random.seed(random_seed)

    def _is_wall_excluded(self, site: np.ndarray) -> bool:
        coord = site.astype(int)
        for axis, wall_coord, side in self.walls:
            if side == '<' and coord[axis] <= wall_coord:
                return True
            if side == '>' and coord[axis] >= wall_coord:
                return True
        return False

    def grow_chains(
        self,
        starts: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], float]:
        """
        Grow *n_chains* self-avoiding walks jointly from *starts*.

        Chains grow in round-robin order at each monomer step, sharing one
        excluded-site set.  A dead-end in any chain returns weight 0.

        Parameters
        ----------
        starts : list of ndarray, each shape (3,)
            Starting lattice position for each chain.  Must be distinct.

        Returns
        -------
        chains : list of n_chains arrays; chain k has shape (chain_lengths[k], 3)
        weight : float — joint Rosenbluth weight; 0.0 on any dead-end
        """
        if len(starts) != self.n_chains:
            raise ValueError(f"Expected {self.n_chains} starts, got {len(starts)}")

        chains   = [np.zeros((self.chain_lengths[k], 3)) for k in range(self.n_chains)]
        occupied: set = set()
        for k, start in enumerate(starts):
            chains[k][0] = start
            occupied.add(tuple(start.astype(int)))

        weight   = 1.0
        max_len  = max(self.chain_lengths)

        for i in range(1, max_len):
            for k in range(self.n_chains):
                if i >= self.chain_lengths[k]:
                    continue   # this chain is already fully grown

                neighbors = chains[k][i - 1] + self.bond_length * _DIRECTIONS
                free = [
                    n for n in neighbors
                    if tuple(n.astype(int)) not in occupied
                    and not self._is_wall_excluded(n)
                ]

                if not free:
                    return chains, 0.0

                weight *= len(free) / (_Z if i == 1 else _Z1)
                chosen = free[np.random.randint(len(free))]
                chains[k][i] = chosen
                occupied.add(tuple(chosen.astype(int)))

        return chains, weight

    def sample(
        self,
        n_samples: int,
        starts: List[np.ndarray],
    ) -> Tuple[List[List[np.ndarray]], np.ndarray]:
        """
        Draw *n_samples* joint Rosenbluth-weighted multi-chain configurations.

        Parameters
        ----------
        n_samples : int
        starts : list of ndarray, each shape (3,)

        Returns
        -------
        configs : list of n_samples entries; each entry is a list of
                  n_chains arrays of shape (n_monomers, 3)
        weights : ndarray of shape (n_samples,)
        """
        configs, weights = [], []
        for _ in range(n_samples):
            chains, w = self.grow_chains(starts)
            configs.append(chains)
            weights.append(w)
        return configs, np.array(weights)
