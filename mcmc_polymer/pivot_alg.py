"""Pivot algorithm sampler for lattice polymer chains."""

import itertools
import numpy as np
from typing import Callable, List, Optional, Tuple


def _build_cubic_symmetries() -> np.ndarray:
    """
    Generate all 48 elements of the cubic symmetry group Oh.

    Each element is a 3×3 signed-permutation matrix: exactly one non-zero
    entry per row and column, each ±1.  There are 3! permutations × 2³ sign
    combinations = 48 matrices, all of which map the simple cubic lattice to
    itself and have determinant ±1.
    """
    mats = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([-1, 1], repeat=3):
            m = np.zeros((3, 3), dtype=float)
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            mats.append(m)
    result = np.stack(mats)           # shape (48, 3, 3)
    assert len(result) == 48
    return result


_CUBIC_SYMMETRIES = _build_cubic_symmetries()


class PivotAlgorithm:
    """
    Pivot algorithm sampler for self-avoiding walks on the simple cubic lattice.

    At each step a random monomer is chosen as the pivot point.  The chain
    segment beyond the pivot is transformed by a random cubic-group symmetry
    operation.  The move is accepted via a Metropolis criterion:

        P_accept = min(1, w_new / w_old)

    where the total weight is:

        w(chain) = w_SAW(chain) * exp(-beta_pot * U_total(chain))
        U_total  = sum(U(chain) for U in potentials)

    Each callable in *potentials* receives the full (n_monomers, 3) chain
    array and returns a scalar energy — this covers any interaction type:
    monomer–monomer, monomer–wall, external field, confinement, etc.

    Parameters
    ----------
    n_steps : int
        Number of pivot moves to attempt per call to `run`.
    beta : float
        Inverse temperature for the self-avoidance soft-core weight
        w_SAW = exp(-beta * n_overlaps).  Use np.inf for the hard-core limit.
    potentials : list of callable, optional
        Each entry is ``U(chain: ndarray) -> float``.  The energies are summed
        to give the total interaction energy.  Pass an empty list (default) to
        run a pure SAW with no additional interactions.
    beta_pot : float
        Inverse temperature (1 / k_B T) applied to *all* potentials.
    """

    def __init__(
        self,
        n_steps: int = 100,
        beta: float = np.inf,
        potentials: Optional[List[Callable[[np.ndarray], float]]] = None,
        beta_pot: float = 1.0,
    ):
        self.n_steps          = n_steps
        self.beta             = beta
        self.potentials       = potentials if potentials is not None else []
        self.beta_pot         = beta_pot
        self.acceptance_rate  = None   # updated after every call to run()

    # ── public interface ──────────────────────────────────────────────────────

    def run(self, chain: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Apply `n_steps` pivot moves starting from *chain*.

        Parameters
        ----------
        chain : ndarray, shape (n_monomers, 3)
            Starting lattice configuration (integer coordinates).

        Returns
        -------
        final_chain : ndarray, shape (n_monomers, 3)
            Configuration after all accepted moves.
        trajectory : list of ndarray
            Chain state after every attempted step (accepted or not).
        """
        current    = chain.copy()
        w_current  = self._compute_weight(current)
        trajectory = []
        n_accepted = 0

        for _ in range(self.n_steps):
            pivot_idx, operation = self._select_pivot_and_operation(current)
            candidate = self._apply_pivot(current, pivot_idx, operation)
            w_candidate = self._compute_weight(candidate)
            # Metropolis acceptance: min(1, w_new / w_old)
            if w_current == 0 or (w_candidate / w_current) >= np.random.rand():
                current   = candidate
                w_current = w_candidate
                n_accepted += 1

            trajectory.append(current.copy())

        self.acceptance_rate = n_accepted / self.n_steps
        return current, trajectory

    # ── step sub-functions (placeholders) ────────────────────────────────────

    def _select_pivot_and_operation(
        self, chain: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """
        Choose a random pivot monomer index and a random cubic symmetry operation.

        Returns
        -------
        pivot_idx : int
            Index of the pivot monomer (1 … n_monomers-2 so both segments
            are non-empty).
        operation : ndarray, shape (3, 3)
            Rotation/reflection matrix from the cubic symmetry group.
        """
        # TODO: implement pivot selection strategy (e.g. bias toward centre)
        pivot_idx = self._pick_pivot(chain)
        operation = self._pick_symmetry()
        return pivot_idx, operation

    def _pick_pivot(self, chain: np.ndarray) -> int:
        """Return a uniformly random pivot index in [1, n_monomers-2]."""
        # TODO: placeholder — uniform random choice
        n = len(chain)
        return int(np.random.randint(1, n - 1))

    def _pick_symmetry(self) -> np.ndarray:
        """Return a uniformly random element of the cubic symmetry group."""
        # TODO: placeholder — sample from _CUBIC_SYMMETRIES
        idx = np.random.randint(len(_CUBIC_SYMMETRIES))
        return _CUBIC_SYMMETRIES[idx]

    def _apply_pivot(
        self, chain: np.ndarray, pivot_idx: int, operation: np.ndarray
    ) -> np.ndarray:
        """
        Apply *operation* to the chain segment after *pivot_idx*.

        The segment [pivot_idx+1 :] is rotated/reflected about the pivot
        monomer's coordinates.

        Parameters
        ----------
        chain : ndarray, shape (n_monomers, 3)
        pivot_idx : int
        operation : ndarray, shape (3, 3)

        Returns
        -------
        new_chain : ndarray, shape (n_monomers, 3)
        """
        # TODO: placeholder — translate to pivot origin, apply matrix, translate back
        new_chain = chain.copy()
        pivot_pos = chain[pivot_idx]
        segment   = chain[pivot_idx + 1:] - pivot_pos
        new_chain[pivot_idx + 1:] = (operation @ segment.T).T + pivot_pos
        return new_chain

    def _compute_weight(self, chain: np.ndarray) -> float:
        """
        Compute the total Boltzmann weight of *chain*.

        w = w_SAW * exp(-beta_pot * sum(U(chain) for U in self.potentials))

        Parameters
        ----------
        chain : ndarray, shape (n_monomers, 3)

        Returns
        -------
        float
        """
        n_overlaps = self._count_overlaps(chain)
        if np.isinf(self.beta):
            w_saw = 0.0 if n_overlaps > 0 else 1.0
        else:
            w_saw = float(np.exp(-self.beta * n_overlaps)) 
        if w_saw == 0.0 or not self.potentials:
            return w_saw

        u_total = sum(float(U(chain)) for U in self.potentials)
        return w_saw * float(np.exp(-self.beta_pot * u_total))

    def _count_overlaps(self, chain: np.ndarray) -> int:
        """
        Count the number of pairs of monomers sharing the same lattice site.

        Returns
        -------
        int : 0 means fully self-avoiding.
        """
        # TODO: placeholder — O(N) set approach; replace with faster method for large N
        sites = list(map(tuple, chain.astype(int)))
        return len(sites) - len(set(sites))

    def _is_self_avoiding(self, chain: np.ndarray) -> bool:
        """Return True if *chain* has no overlapping monomers."""
        return self._count_overlaps(chain) == 0

    def _compute_end_to_end_sq(self, chain: np.ndarray) -> float:
        """Return the squared end-to-end distance |r_N - r_0|²."""
        # TODO: placeholder
        return float(np.sum((chain[-1] - chain[0]) ** 2))

    def _compute_radius_of_gyration_sq(self, chain: np.ndarray) -> float:
        """Return the radius of gyration squared Rg² = mean |r_i - r_cm|²."""
        # TODO: placeholder
        r_cm = chain.mean(axis=0)
        return float(np.mean(np.sum((chain - r_cm) ** 2, axis=1)))
