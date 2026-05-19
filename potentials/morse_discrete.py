"""Discretized Morse potential with cutoff for the simple cubic lattice."""

import numpy as np


class DiscreteMorsePotential:
    """
    Morse potential discretized to integer lattice distances along one axis.

    Because the chain lives on a simple cubic lattice, distances along any
    axis are integers.  A lookup table is precomputed once at construction
    for integer distances r = 0, 1, …, cutoff.  Monomers beyond the cutoff
    contribute zero energy (truncated potential).

    Monomers with a negative signed distance (i.e. inside/behind the wall)
    receive a large repulsive penalty instead of the Morse value.

    Parameters
    ----------
    D_e : float
        Well depth (energy units).  Must be > 0.
    a : float
        Controls the width of the potential well.
    r_e : float
        Equilibrium distance from the wall (need not be an integer; the
        table is evaluated at integer r values around it).
    axis : int
        Lattice axis along which the potential acts (0=x, 1=y, 2=z).
    wall_position : int
        Integer lattice coordinate of the wall along *axis*.
    cutoff : int
        Maximum integer distance at which the potential is non-zero.
        Monomers farther than *cutoff* contribute 0.
    wall_penalty : float
        Energy assigned to monomers inside the wall (r < 0).
        Should be large enough to make such configurations effectively
        forbidden (default 1e6).
    """

    def __init__(
        self,
        D_e: float,
        a: float,
        r_e: float,
        axis: int = 2,
        wall_position: int = 0,
        cutoff: int = 10,
        wall_penalty: float = 1e6,
    ):
        if D_e <= 0:
            raise ValueError("D_e must be positive")
        if a <= 0:
            raise ValueError("a must be positive")
        if cutoff < 0:
            raise ValueError("cutoff must be >= 0")

        self.axis          = axis
        self.wall_position = int(wall_position)
        self.cutoff        = int(cutoff)
        self.wall_penalty  = float(wall_penalty)

        # Precompute V at every integer distance 0 … cutoff
        r_int      = np.arange(0, cutoff + 1, dtype=float)
        self._table = D_e * (1.0 - np.exp(-a * (r_int - r_e))) ** 2 - D_e

    def __call__(self, chain: np.ndarray) -> float:
        """
        Return the total discretized Morse energy for *chain*.

        Parameters
        ----------
        chain : ndarray, shape (n_monomers, 3)
            Integer lattice coordinates.

        Returns
        -------
        float
        """
        dist = chain[:, self.axis].astype(int) - self.wall_position

        inside   = dist < 0
        in_range = (dist >= 0) & (dist <= self.cutoff)

        energy  = float(np.count_nonzero(inside)) * self.wall_penalty
        energy += float(np.sum(self._table[dist[in_range]]))
        return energy

    @property
    def table(self) -> np.ndarray:
        """Lookup table: table[r] = V(r) for r = 0 … cutoff."""
        return self._table.copy()
