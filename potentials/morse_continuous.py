"""Continuous Morse potential along one Cartesian axis (e.g. wall interaction)."""

import numpy as np


class MorsePotential:
    """
    Continuous Morse potential acting on each monomer along one axis.

    Models a polymer near an attractive wall.  Each monomer at signed distance
    r = chain[:, axis] - wall_position from the wall contributes:

        V(r) = D_e * (1 - exp(-a * (r - r_e)))^2 - D_e

    The total energy is the sum over all monomers.  The minimum V = -D_e is
    reached at r = r_e; the potential is repulsive for r < r_e and decays
    to zero for r → ∞.

    Parameters
    ----------
    D_e : float
        Well depth (energy units).  Must be > 0.
    a : float
        Controls the width of the potential well; larger a = narrower well.
    r_e : float
        Equilibrium distance from the wall (position of the minimum).
    axis : int
        Cartesian axis along which the potential acts (0=x, 1=y, 2=z).
    wall_position : float
        Reference position of the wall along *axis*.
    """

    def __init__(
        self,
        D_e: float,
        a: float,
        r_e: float,
        axis: int = 2,
        wall_position: float = 0.0,
    ):
        if D_e <= 0:
            raise ValueError("D_e must be positive")
        if a <= 0:
            raise ValueError("a must be positive")
        self.D_e           = D_e
        self.a             = a
        self.r_e           = r_e
        self.axis          = axis
        self.wall_position = wall_position

    def __call__(self, chain: np.ndarray) -> float:
        """
        Return the total Morse energy for *chain*.

        Parameters
        ----------
        chain : ndarray, shape (n_monomers, 3)

        Returns
        -------
        float
        """
        r = chain[:, self.axis] - self.wall_position
        v = self.D_e * (1.0 - np.exp(-self.a * (r - self.r_e))) ** 2 - self.D_e
        return float(np.sum(v))

    def energy_profile(self, r_min: float, r_max: float, n_points: int = 500):
        """
        Evaluate V(r) over a range of distances for plotting/inspection.

        Returns
        -------
        r : ndarray, shape (n_points,)
        v : ndarray, shape (n_points,)
        """
        r = np.linspace(r_min, r_max, n_points)
        v = self.D_e * (1.0 - np.exp(-self.a * (r - self.r_e))) ** 2 - self.D_e
        return r, v
