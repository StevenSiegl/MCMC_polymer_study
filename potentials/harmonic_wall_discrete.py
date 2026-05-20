"""Discretized harmonic wall potential with cutoff for the simple cubic lattice."""

import numpy as np


class DiscreteHarmonicWallPotential:
    """
    Harmonic wall potential discretized to integer lattice distances.

    On the simple cubic lattice all distances along an axis are integers.
    A lookup table is precomputed once at construction for signed distances
    r = -cutoff, …, -1  (inside wall, penalized) and r >= 0 gives zero.

    Monomers farther inside the wall than *cutoff* lattice steps receive a
    fixed large penalty instead of the quadratic value, preventing overflow
    and making deeply buried configurations effectively forbidden.

        V(r) = k/2 · r²        for -cutoff <= r < 0
               wall_penalty     for r < -cutoff
               0                for r >= 0

    Parameters
    ----------
    k : float
        Spring constant (energy / length²).  Must be > 0.
    axis : int
        Lattice axis perpendicular to the wall (0=x, 1=y, 2=z).
    wall_position : int
        Integer lattice coordinate of the wall along *axis*.
    cutoff : int
        Maximum penetration depth (in lattice steps) handled by the
        quadratic table.  Deeper monomers get *wall_penalty*.
    wall_penalty : float
        Energy assigned to monomers more than *cutoff* steps inside the
        wall (default 1e6).
    """

    def __init__(
        self,
        k: float,
        axis: int = 2,
        wall_position: int = 0,
        cutoff: int = 10,
        wall_penalty: float = 1e6,
    ):
        if k <= 0:
            raise ValueError("k must be positive")
        if cutoff < 0:
            raise ValueError("cutoff must be >= 0")

        self.axis          = axis
        self.wall_position = int(wall_position)
        self.cutoff        = int(cutoff)
        self.wall_penalty  = float(wall_penalty)

        # Lookup table indexed by penetration depth p = 1 … cutoff
        # p = 0 means exactly at the wall → entry [0] = 0 (on free side)
        # For convenience store V at |r| for r in [-cutoff, 0]
        p = np.arange(0, cutoff + 1, dtype=float)
        self._table = 0.5 * float(k) * p ** 2   # table[p] = k/2 · p²

    def __call__(self, chain: np.ndarray) -> float:
        """
        Return the total discretized harmonic wall energy for *chain*.

        Parameters
        ----------
        chain : ndarray, shape (n_monomers, 3)
            Integer lattice coordinates.

        Returns
        -------
        float
        """
        r = chain[:, self.axis].astype(int) - self.wall_position

        on_free_side   = r >= 0
        shallow_inside = (r < 0) & (r >= -self.cutoff)   # -cutoff <= r < 0
        deep_inside    = r < -self.cutoff

        energy  = float(np.count_nonzero(deep_inside)) * self.wall_penalty
        energy += float(np.sum(self._table[np.abs(r[shallow_inside])]))
        return energy

    @property
    def table(self) -> np.ndarray:
        """Lookup table: table[p] = V(p) for penetration depth p = 0 … cutoff."""
        return self._table.copy()
