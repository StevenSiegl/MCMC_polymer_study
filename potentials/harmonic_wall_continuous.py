"""Continuous harmonic wall potential along one Cartesian axis."""

import numpy as np


class HarmonicWallPotential:
    """
    Harmonic potential confining monomers to one side of a wall.

    Each monomer that penetrates the wall (signed distance r < 0) is
    penalized quadratically.  Monomers on the free side (r >= 0) feel
    no force:

        V(r) = k/2 · r²   if r < 0
               0           if r >= 0

    where r = chain[:, axis] - wall_position.

    The total energy is the sum over all monomers.

    Parameters
    ----------
    k : float
        Spring constant (energy / length²).  Must be > 0.
    axis : int
        Cartesian axis perpendicular to the wall (0=x, 1=y, 2=z).
    wall_position : float
        Position of the wall along *axis*.
    """

    def __init__(
        self,
        k: float,
        axis: int = 2,
        wall_position: float = 0.0,
    ):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k             = float(k)
        self.axis          = axis
        self.wall_position = float(wall_position)

    def __call__(self, chain: np.ndarray) -> float:
        """
        Return the total harmonic wall energy for *chain*.

        Parameters
        ----------
        chain : ndarray, shape (n_monomers, 3)

        Returns
        -------
        float
        """
        r = chain[:, self.axis] - self.wall_position
        penetration = np.minimum(r, 0.0)          # zero for r >= 0
        return float(0.5 * self.k * np.sum(penetration ** 2))

    def energy_profile(self, r_min: float, r_max: float, n_points: int = 500):
        """
        Evaluate V(r) over a range of distances for plotting/inspection.

        Returns
        -------
        r : ndarray, shape (n_points,)
        v : ndarray, shape (n_points,)
        """
        r = np.linspace(r_min, r_max, n_points)
        v = np.where(r < 0.0, 0.5 * self.k * r ** 2, 0.0)
        return r, v
