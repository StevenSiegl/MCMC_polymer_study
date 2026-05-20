"""Discretized Morse potential with cutoff for the simple cubic lattice."""

import numpy as np


class DiscreteMorsePotential:
    """
    Morse potential discretized to integer lattice distances along one axis.

    The lookup table covers r = -inner_cutoff, …, 0, 1, …, cutoff using the
    exact Morse formula at each integer.  Three distance regions are handled:

        r < -inner_cutoff  →  wall_penalty  (hard repulsion cap)
        -inner_cutoff ≤ r ≤ cutoff  →  table lookup (Morse value)
        r > cutoff         →  0  (truncated tail)

    The negative half of the table represents the steeply repulsive inner wall
    of the Morse potential (V grows rapidly as r → -∞), evaluated for
    ``inner_cutoff`` grid points before switching to the fixed penalty.

    Parameters
    ----------
    D_e : float
        Well depth (energy units).  Must be > 0.
    a : float
        Controls the width of the potential well.
    r_e : float
        Equilibrium distance from the wall (need not be an integer).
    axis : int
        Lattice axis along which the potential acts (0=x, 1=y, 2=z).
    wall_position : int
        Integer lattice coordinate of the wall along *axis*.
    cutoff : int
        Maximum positive integer distance included in the table.
        Monomers farther than *cutoff* contribute 0.
    inner_cutoff : int
        Number of negative grid points included in the table before
        switching to *wall_penalty*.  E.g. inner_cutoff=3 evaluates the
        Morse formula at r = -3, -2, -1 and uses wall_penalty for r < -3.
    wall_penalty : float
        Energy assigned to monomers deeper than *inner_cutoff* inside the
        wall.  Should be large enough to make those configurations
        effectively forbidden (default 1e6).
    """

    def __init__(
        self,
        D_e: float,
        a: float,
        r_e: float,
        axis: int = 2,
        wall_position: int = 0,
        cutoff: int = 10,
        inner_cutoff: int = 3,
        wall_penalty: float = 1e6,
    ):
        if D_e <= 0:
            raise ValueError("D_e must be positive")
        if a <= 0:
            raise ValueError("a must be positive")
        if cutoff < 0:
            raise ValueError("cutoff must be >= 0")
        if inner_cutoff < 0:
            raise ValueError("inner_cutoff must be >= 0")

        self.axis          = axis
        self.wall_position = int(wall_position)
        self.cutoff        = int(cutoff)
        self.inner_cutoff  = int(inner_cutoff)
        self.wall_penalty  = float(wall_penalty)

        # Table covers r = -inner_cutoff … +cutoff  (length = inner_cutoff + cutoff + 1)
        # Index into table as: table[r + inner_cutoff]
        r_int        = np.arange(-inner_cutoff, cutoff + 1, dtype=float)
        self._table  = D_e * (1.0 - np.exp(-a * (r_int - r_e))) ** 2 - D_e
        self._offset = int(inner_cutoff)   # shift so index 0 → r = -inner_cutoff

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

        too_deep = dist < -self.inner_cutoff
        in_range = (dist >= -self.inner_cutoff) & (dist <= self.cutoff)
        # dist > cutoff  →  contributes 0, no action needed
        energy  = float(np.count_nonzero(too_deep)) * self.wall_penalty
        energy += float(np.sum(self._table[dist[in_range] + self._offset]))
        return energy

    @property
    def table(self) -> np.ndarray:
        """
        Full lookup table from r = -inner_cutoff to r = +cutoff.

        table[i] corresponds to r = i - inner_cutoff.
        """
        return self._table.copy()

    def table_as_dict(self) -> dict:
        """Return {r: V(r)} for all tabulated distances."""
        r_vals = np.arange(-self.inner_cutoff, self.cutoff + 1)
        return {int(r): float(v) for r, v in zip(r_vals, self._table)}
