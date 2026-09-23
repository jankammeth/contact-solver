"""Cubic polynomial utilities for trajectory computation."""

import numpy as np


def cubic_coefficients(p0, v0, pf, vf, T):
    if T < 1e-10:
        return p0.copy(), v0.copy(), np.zeros_like(p0), np.zeros_like(p0)
    a = p0.copy()
    b = v0.copy()
    c = (3 * (pf - p0) - (2 * v0 + vf) * T) / T**2
    d = (-2 * (pf - p0) + (v0 + vf) * T) / T**3
    return a, b, c, d


def eval_cubic(coeffs, t):
    a, b, c, d = coeffs
    return a + b * t + c * t**2 + d * t**3


def eval_cubic_vel(coeffs, t):
    a, b, c, d = coeffs
    return b + 2 * c * t + 3 * d * t**2


def eval_cubic_acc(coeffs, t):
    a, b, c, d = coeffs
    return 2 * c + 6 * d * t


def eval_cubic_jerk(coeffs):
    return 6 * coeffs[3]


def find_distance_minima(r_coeffs, T_seg, t_offset=0.0):
    a, b, c, d = r_coeffs
    if T_seg < 1e-10:
        return []

    minima = []
    for absolute_t, dist in distance_candidates(r_coeffs, T_seg, t_offset):
        t = absolute_t - t_offset
        eps = 1e-8
        is_min = True
        if t > eps and np.linalg.norm(eval_cubic(r_coeffs, t - eps)) < dist - 1e-10:
            is_min = False
        if t < T_seg - eps and np.linalg.norm(eval_cubic(r_coeffs, t + eps)) < dist - 1e-10:
            is_min = False
        if is_min:
            minima.append((t_offset + t, dist))
    return minima


def distance_candidates(r_coeffs, duration, offset=0.0):
    """Endpoints and all real stationary-distance candidates in any dimension.

    Form sum_i r_i r'_i in normalized segment time u in [0, 1]. Constant
    distance includes both endpoints. Near-real polynomial roots use a 1e-7
    relative imaginary tolerance; roots within 1e-10 of [0, 1] are clamped.
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
    scaled = coefficients * np.array([1, duration, duration**2, duration**3])[:, None]
    polynomial = np.zeros(6)
    for coordinate in scaled.T:
        polynomial += np.convolve(coordinate, coordinate[1:] * [1, 2, 3])
    polynomial = np.trim_zeros(polynomial, "b")
    candidates = [0.0, 1.0]
    if len(polynomial) > 1:
        for root in np.polynomial.polynomial.polyroots(polynomial / np.max(np.abs(polynomial))):
            if abs(root.imag) <= 1e-7 * max(1.0, abs(root.real)):
                if -1e-10 <= root.real <= 1 + 1e-10:
                    candidates.append(float(np.clip(root.real, 0, 1)))
    return [
        (offset + u * duration, float(np.linalg.norm(eval_cubic(scaled, u))))
        for u in sorted(set(candidates))
    ]
