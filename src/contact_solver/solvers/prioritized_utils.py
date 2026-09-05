"""Shared utilities for prioritized-planning solvers.

Provides:
    conflict_severity_order:
        Priority ordering by descending worst-case pairwise violation on the
        unconstrained straight-line (cubic) trajectories.

    PiecewiseCubicTrajectory:
        Internal representation of a single-robot continuous trajectory as a
        sequence of cubic segments. Used by PrioritizedSCI to hand prior-rank
        trajectories to the next single-robot subproblem.
"""

import numpy as np

from .cubic_utils import (
    cubic_coefficients,
    eval_cubic,
    eval_cubic_acc,
    eval_cubic_vel,
)


def conflict_severity_order(
    p0: np.ndarray,
    v0: np.ndarray,
    pf: np.ndarray,
    vf: np.ndarray,
    T: float,
    R: float,
    n_samples: int = 100,
) -> list[int]:
    """Sort robot indices by descending worst-case pairwise violation.

    For each robot i, compute its conflict score as the sum, over all other
    robots j, of (R - min_t ||x_i(t) - x_j(t)||)_+, where x_i, x_j are the
    unconstrained cubic trajectories between the boundary states. Robots with
    the most severe predicted violations are solved first.

    Args:
        p0, v0, pf, vf: (N, 2) boundary states.
        T: time horizon.
        R: minimum allowed pairwise distance.
        n_samples: number of time samples for the min-distance scan.

    Returns:
        List of robot indices in descending conflict-severity order.
    """
    N = p0.shape[0]
    coeffs = [cubic_coefficients(p0[i], v0[i], pf[i], vf[i], T) for i in range(N)]

    scores = np.zeros(N)
    ts = np.linspace(0.0, T, n_samples + 1)

    for i in range(N):
        for j in range(i + 1, N):
            d_min = np.inf
            for t in ts:
                d = np.linalg.norm(eval_cubic(coeffs[i], t) - eval_cubic(coeffs[j], t))
                if d < d_min:
                    d_min = d
            if d_min < R:
                severity = R - d_min
                scores[i] += severity
                scores[j] += severity

    return np.argsort(-scores).tolist()


class PiecewiseCubicTrajectory:
    """Piecewise cubic trajectory: list of (coeffs, t_start, t_end) segments."""

    def __init__(self, segments: list, T: float):
        self.segments = segments
        self.T = T

    def _find_segment(self, t):
        for coeffs, t_s, t_e in self.segments:
            if t_s <= t <= t_e + 1e-10:
                return coeffs, t_s
        coeffs, t_s, _ = self.segments[-1]
        return coeffs, t_s

    def eval_pos(self, t):
        coeffs, t_s = self._find_segment(t)
        return eval_cubic(coeffs, t - t_s)

    def eval_vel(self, t):
        coeffs, t_s = self._find_segment(t)
        return eval_cubic_vel(coeffs, t - t_s)

    def eval_acc(self, t):
        coeffs, t_s = self._find_segment(t)
        return eval_cubic_acc(coeffs, t - t_s)

    def sample(self, n_points: int = 1000):
        times = np.linspace(0.0, self.T, n_points)
        positions = np.array([self.eval_pos(t) for t in times])
        velocities = np.array([self.eval_vel(t) for t in times])
        accelerations = np.array([self.eval_acc(t) for t in times])
        return times, positions, velocities, accelerations

    def sample_at(self, times: np.ndarray):
        positions = np.array([self.eval_pos(t) for t in times])
        velocities = np.array([self.eval_vel(t) for t in times])
        accelerations = np.array([self.eval_acc(t) for t in times])
        return positions, velocities, accelerations
