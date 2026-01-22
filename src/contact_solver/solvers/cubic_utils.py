"""Cubic polynomial utilities for trajectory computation."""

import numpy as np


def cubic_coefficients(p0, v0, pf, vf, T):
    if T < 1e-10:
        return p0.copy(), v0.copy(), np.zeros(2), np.zeros(2)
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

    rx = np.array([a[0], b[0], c[0], d[0]])
    ry = np.array([a[1], b[1], c[1], d[1]])
    vx = np.array([b[0], 2 * c[0], 3 * d[0]])
    vy = np.array([b[1], 2 * c[1], 3 * d[1]])

    dot_poly = np.convolve(rx, vx) + np.convolve(ry, vy)
    if np.max(np.abs(dot_poly)) < 1e-14:
        return []

    roots = np.roots(dot_poly[::-1])
    candidates = [0.0, T_seg]
    for root in roots:
        if np.isreal(root):
            t = np.real(root)
            if 1e-10 < t < T_seg - 1e-10:
                candidates.append(t)

    minima = []
    for t in candidates:
        r_t = eval_cubic(r_coeffs, t)
        dist = np.linalg.norm(r_t)
        eps = 1e-8
        is_min = True
        if t > eps and np.linalg.norm(eval_cubic(r_coeffs, t - eps)) < dist - 1e-10:
            is_min = False
        if t < T_seg - eps and np.linalg.norm(eval_cubic(r_coeffs, t + eps)) < dist - 1e-10:
            is_min = False
        if is_min:
            minima.append((t_offset + t, dist))
    return minima
