"""Cubic polynomial utilities for trajectory computation."""

import numpy as np
from scipy.optimize import brentq


def cubic_coefficients(p0, v0, pf, vf, T):
    """Interpolate both endpoint states, including on short positive intervals."""
    if not np.isfinite(T) or T <= 0:
        raise ValueError("Cubic duration must be finite and positive")
    a, b, pf, vf = (np.array(value, dtype=float, copy=True) for value in (p0, v0, pf, vf))
    if not all(np.isfinite(value).all() for value in (a, b, pf, vf)):
        raise ValueError("Cubic endpoint states must be finite")
    try:
        # Divide successively so T**2 or T**3 cannot underflow to zero first.
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            secant = (pf - a) / T
            c = (3 * secant - 2 * b - vf) / T
            d = ((-2 * secant + b + vf) / T) / T
    except FloatingPointError as exc:
        raise ValueError("Cubic coefficients exceed floating-point range") from exc
    return a, b, c, d


def eval_cubic(coeffs, t):
    a, b, c, d = coeffs
    return a + t * (b + t * (c + t * d))


def eval_cubic_vel(coeffs, t):
    a, b, c, d = coeffs
    return b + t * (2 * c + t * 3 * d)


def eval_cubic_acc(coeffs, t):
    a, b, c, d = coeffs
    return 2 * c + 6 * d * t


def eval_cubic_jerk(coeffs):
    return 6 * coeffs[3]


def unit_interval_candidates(polynomial):
    """Isolate roots on [0,1] by recursively partitioning at derivative roots.

    Retain derivative critical points as extra candidates, including repeated
    roots. Tiny leading coefficients are not discarded and huge roots outside
    the interval cannot spoil the companion-matrix eigenproblem.
    """
    # These polynomials have degree at most five. Scalar arithmetic avoids
    # repeatedly allocating NumPy arrays in every recursive partition and
    # every Brent callback, without changing the isolation procedure.
    coefficients = [float(value) for value in polynomial]
    while coefficients and coefficients[-1] == 0:
        coefficients.pop()
    if len(coefficients) <= 1:
        return []
    scale = max(map(abs, coefficients))
    coefficients = [value / scale for value in coefficients]
    if len(coefficients) == 2:
        root = -coefficients[0] / coefficients[1]
        return [float(root)] if 0 <= root <= 1 else []
    critical = unit_interval_candidates([i * value for i, value in enumerate(coefficients) if i])
    partition = sorted(set([0.0, 1.0, *critical]))

    def evaluate(u):
        result = coefficients[-1]
        for value in coefficients[-2::-1]:
            result = result * u + value
        return result

    values = [evaluate(u) for u in partition]
    candidates = list(critical)
    for i, (left, right) in enumerate(zip(partition[:-1], partition[1:], strict=True)):
        if values[i] == 0:
            candidates.append(left)
        if values[i + 1] == 0:
            candidates.append(right)
        if (
            values[i] != 0
            and values[i + 1] != 0
            and (values[i] < 0) != (values[i + 1] < 0)
        ):
            candidates.append(
                brentq(
                    evaluate,
                    left,
                    right,
                    xtol=1e-14,
                    rtol=4 * np.finfo(float).eps,
                )
            )
    return sorted(set(candidates))


def distance_candidates(r_coeffs, duration, offset=0.0):
    """Endpoints and all real stationary-distance candidates in any dimension.

    Form sum_i r_i r'_i in normalized segment time u in [0, 1]. Constant
    distance includes both endpoints. Derivative partitions isolate roots only
    on [0,1], including near-degenerate and repeated-root cases.
    No contact exclusion or sampled minimum test is used here.
    """
    coefficients = np.asarray(r_coeffs)
    if (
        coefficients.ndim != 2
        or coefficients.shape[0] != 4
        or not np.all(np.isfinite(coefficients))
        or not np.isfinite(duration)
        or duration < 0
    ):
        raise ValueError("Expected finite (4, d) coefficients and a nonnegative duration")
    if not np.any(coefficients[1:]):
        # Exactly constant relative position: both endpoints attain the minimum.
        distance = float(np.linalg.norm(coefficients[0]))
        return [(offset, distance), (offset + duration, distance)]
    scaled = coefficients.astype(float, copy=True)
    for power in range(1, 4):
        scaled[power:] *= duration
    polynomial = np.zeros(6)
    for coordinate in scaled.T:
        polynomial += np.convolve(coordinate, coordinate[1:] * [1, 2, 3])
    candidates = [0.0, 1.0, *unit_interval_candidates(polynomial)]
    return [
        (offset + u * duration, float(np.linalg.norm(eval_cubic(scaled, u))))
        for u in sorted(set(candidates))
    ]
