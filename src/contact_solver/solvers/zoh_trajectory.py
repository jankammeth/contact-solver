"""Continuous double-integrator trajectories with zero-order-held acceleration."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import brentq


def _stationary_candidates(polynomial):
    """Isolate roots on [0,1] through derivative partitions, avoiding huge roots.

    Also retain derivative critical points: these cover repeated roots without
    relying on a sign change. Extra candidates are harmless for minimization.
    Degrees are at most three for the derivative of a quadratic squared norm.
    """
    polynomial = np.trim_zeros(polynomial, "b")
    if len(polynomial) <= 1:
        return []
    polynomial = polynomial / np.max(np.abs(polynomial))
    critical = _stationary_candidates(polynomial[1:] * np.arange(1, len(polynomial)))
    partition = sorted(set([0.0, 1.0, *critical]))
    values = np.polynomial.polynomial.polyval(partition, polynomial)
    candidates = list(critical)
    for i, (left, right) in enumerate(zip(partition[:-1], partition[1:], strict=True)):
        if values[i] == 0:
            candidates.append(left)
        if values[i + 1] == 0:
            candidates.append(right)
        if (
            np.signbit(values[i]) != np.signbit(values[i + 1])
            and values[i] != 0
            and values[i + 1] != 0
        ):
            candidates.append(
                brentq(
                    lambda u: np.polynomial.polynomial.polyval(u, polynomial),
                    left,
                    right,
                    xtol=1e-14,
                    rtol=4 * np.finfo(float).eps,
                )
            )
    return sorted(set(candidates))


def quadratic_distance_candidates(coefficients, duration, offset=0.0):
    """All stationary-distance candidates of a quadratic on a closed interval."""
    p, v, a = coefficients[:3] * np.array([1, duration, duration**2])[:, None]
    polynomial = np.array([p @ v, v @ v + 2 * p @ a, 3 * v @ a, 2 * a @ a])
    times = sorted(set([0.0, 1.0, *_stationary_candidates(polynomial)]))
    return [(offset + duration * u, float(np.linalg.norm(p + u * (v + u * a)))) for u in times]


@dataclass
class ZOHTrajectory:
    """a[:,k] acts on [k*h,(k+1)*h); final samples use the last interval.

    Integrate from the prescribed initial state, rather than interpolating the
    approximate QP position/velocity variables or silently repairing their defects.
    """

    h: float
    coefficients: np.ndarray  # robot, interval, polynomial power, coordinate

    @classmethod
    def from_controls(cls, initial_position, initial_velocity, acceleration, h):
        a = np.asarray(acceleration, dtype=float)
        if a.ndim != 3 or a.shape[1] < 1 or not np.all(np.isfinite(a)) or h <= 0:
            raise ValueError("Expected finite accelerations with shape (N, K, d) and h > 0")
        v0 = np.asarray(initial_velocity)[:, None, :]
        p0 = np.asarray(initial_position)[:, None, :]
        zero = np.zeros((a.shape[0], 1, a.shape[2]))
        starts_v = v0 + h * np.concatenate((zero, np.cumsum(a[:, :-1], axis=1)), axis=1)
        increments = h * starts_v + 0.5 * h**2 * a
        starts_p = p0 + np.concatenate((zero, np.cumsum(increments[:, :-1], axis=1)), axis=1)
        return cls(h, np.stack((starts_p, starts_v, 0.5 * a, np.zeros_like(a)), axis=2))

    @property
    def horizon(self):
        return self.h * self.coefficients.shape[1]

    def sample(self, times, derivative=0):
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or not np.all(np.isfinite(times))
            or np.any(times < 0)
            or np.any(times > self.horizon)
        ):
            raise ValueError("Sample times must be a finite 1-D array within [0, K*h]")
        if derivative not in (0, 1, 2):
            raise ValueError("ZOH trajectory supports derivative orders 0, 1, 2")
        knots = np.arange(self.coefficients.shape[1] + 1) * self.h
        segments = np.minimum(
            np.searchsorted(knots, times, side="right") - 1, self.coefficients.shape[1] - 1
        )
        local = times - self.h * segments
        c = self.coefficients[:, segments]
        if derivative == 2:
            return 2 * c[:, :, 2]
        if derivative == 1:
            return c[:, :, 1] + 2 * local[None, :, None] * c[:, :, 2]
        return c[:, :, 0] + local[None, :, None] * (c[:, :, 1] + local[None, :, None] * c[:, :, 2])

    def separation(self, obstacles=None):
        """Full-horizon polynomial checks, including the prescribed t=0 states."""
        n, k = self.coefficients.shape[:2]
        pair_best, obstacle_best = (np.inf, None, None), (np.inf, None, None)
        obstacles = [] if obstacles is None else obstacles
        for segment in range(k):
            for i, j in combinations(range(n), 2):
                c = self.coefficients[i, segment] - self.coefficients[j, segment]
                for t, distance in quadratic_distance_candidates(c, self.h, segment * self.h):
                    if distance < pair_best[0]:
                        pair_best = distance, (i, j), t
            for i in range(n):
                for j, obstacle in enumerate(obstacles):
                    c = self.coefficients[i, segment].copy()
                    c[0] -= obstacle
                    for t, distance in quadratic_distance_candidates(c, self.h, segment * self.h):
                        if distance < obstacle_best[0]:
                            obstacle_best = distance, (i, j), t
        return dict(
            min_distance=pair_best[0],
            closest_pair=pair_best[1],
            closest_time=pair_best[2],
            min_obstacle_distance=obstacle_best[0],
            closest_obstacle=obstacle_best[1],
            closest_obstacle_time=obstacle_best[2],
        )

    def extrema(self):
        """Coordinate extrema for continuous position/velocity and held acceleration."""
        p, v, half_a = (self.coefficients[:, :, power] for power in (0, 1, 2))
        a = 2 * half_a
        p_end, v_end = p + self.h * v + self.h**2 * half_a, v + self.h * a
        stationary = np.divide(-v, a, out=np.zeros_like(v), where=a != 0)
        stationary = np.clip(stationary, 0, self.h)
        p_stationary = p + stationary * v + stationary**2 * half_a
        return dict(
            position=(
                np.minimum(np.minimum(p, p_end), p_stationary).min(axis=(0, 1)),
                np.maximum(np.maximum(p, p_end), p_stationary).max(axis=(0, 1)),
            ),
            velocity=(np.minimum(v, v_end).min(axis=(0, 1)), np.maximum(v, v_end).max(axis=(0, 1))),
            acceleration=(a.min(axis=(0, 1)), a.max(axis=(0, 1))),
        )

    def acceleration_cost(self):
        return float(self.h * np.sum((2 * self.coefficients[:, :, 2]) ** 2))
