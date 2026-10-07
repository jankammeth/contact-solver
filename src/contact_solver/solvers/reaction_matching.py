"""Condensed contact equations using the clamped cubic Green function.

Intermediate knot states are eliminated exactly. This recovery formulation also
allows contact times to exchange order without a zero-length Hermite segment.
"""

import time

import numpy as np
from scipy.optimize import root

from .cubic_utils import cubic_coefficients


class ReactionMatching:
    def __init__(self, system):
        self.system = system
        self.m, self.d, self.T, self.R = system.m, system.d, system.T, system.R
        incidence = np.zeros((self.m, system.N))
        pairs = np.array(system.pairs)
        incidence[np.arange(self.m), pairs[:, 0]] = 1
        incidence[np.arange(self.m), pairs[:, 1]] = -1
        self.coupling = incidence @ incidence.T
        self.free = np.array(cubic_coefficients(system.r0, system.v0, system.rf, system.vf, self.T))
        self.pair_free = np.array(
            [self.free[:, q - 1] - (self.free[:, p - 1] if p else 0) for p, q in system.pairs]
        )

    def evaluate(self, w, jacobian=False):
        m, d, T, R = self.m, self.d, self.T, self.R
        z = w.reshape(m, d + 2)
        t, n, alpha = z[:, 0], z[:, 1 : 1 + d], z[:, -1]
        u, s = t[:, None], t[None, :]
        h = np.maximum(u - s, 0)
        A = s * (T - s) ** 2 / (2 * T * T)
        B = -(T + 2 * s) * (T - s) ** 2 / (6 * T**3)
        G = h**3 / 6 + A * u * u + B * u**3
        V = h * h / 2 + 2 * A * u + 3 * B * u * u
        C = self.coupling
        fc = self.pair_free
        r = ((fc[:, 3] * t[:, None] + fc[:, 2]) * t[:, None] + fc[:, 1]) * t[:, None] + fc[:, 0]
        v = (3 * fc[:, 3] * t[:, None] + 2 * fc[:, 2]) * t[:, None] + fc[:, 1]
        r += R * (C * G) @ (alpha[:, None] * n)
        v += R * (C * V) @ (alpha[:, None] * n)
        residual = np.column_stack(
            (r / R - n, np.sum(n * v, axis=1) * T / R, np.sum(n * n, axis=1) - 1)
        ).ravel()
        if not jacobian:
            return residual
        Ap = (T - s) * (T - 3 * s) / (2 * T * T)
        Bp = s * (T - s) / T**3
        Gs = -h * h / 2 + Ap * u * u + Bp * u**3
        Vs = -h + 2 * Ap * u + 3 * Bp * u * u
        acc = (
            6 * fc[:, 3] * t[:, None]
            + 2 * fc[:, 2]
            + R * (C * (h + 2 * A + 6 * B * u)) @ (alpha[:, None] * n)
        )
        J = np.zeros((m, d + 2, m, d + 2))
        for i in range(m):
            for j in range(m):
                J[i, :d, j, 0] = C[i, j] * Gs[i, j] * alpha[j] * n[j]
                J[i, :d, j, 1 : 1 + d] = C[i, j] * G[i, j] * alpha[j] * np.eye(d)
                J[i, :d, j, -1] = C[i, j] * G[i, j] * n[j]
                J[i, d, j, 0] = T * C[i, j] * Vs[i, j] * alpha[j] * (n[i] @ n[j])
                J[i, d, j, 1 : 1 + d] = T * C[i, j] * V[i, j] * alpha[j] * n[i]
                J[i, d, j, -1] = T * C[i, j] * V[i, j] * (n[i] @ n[j])
            J[i, :d, i, 0] += v[i] / R
            J[i, :d, i, 1 : 1 + d] -= np.eye(d)
            J[i, d, i, 0] += T * (n[i] @ acc[i]) / R
            J[i, d, i, 1 : 1 + d] += T * v[i] / R
            J[i, d + 1, i, 1 : 1 + d] = 2 * n[i]
        return J.reshape(m * (d + 2), m * (d + 2))

    def solve(self, initial, deadline, max_evaluations):
        z = initial.reshape(self.m, self.system.b)
        w = np.column_stack((z[:, : 1 + self.d], z[:, 1 + 2 * self.d])).ravel()

        def evaluate(x, jac=False):
            if time.perf_counter() >= deadline:
                raise TimeoutError("reaction recovery deadline")
            return self.evaluate(x, jac)

        result = root(
            evaluate,
            w,
            jac=lambda x: evaluate(x, True),
            method="hybr",
            options={"xtol": 1e-10, "maxfev": max_evaluations},
        )
        return result

    def coefficients(self, w):
        """Build local cubic coefficients directly, without endpoint subtraction."""
        z = w.reshape(self.m, self.d + 2)
        order = np.argsort(z[:, 0])
        times = np.concatenate(([0.0], z[order, 0], [self.T]))
        t = z[:, 0]
        A = t * (self.T - t) ** 2 / (2 * self.T**2)
        B = -(self.T + 2 * t) * (self.T - t) ** 2 / (6 * self.T**3)
        forces = (
            self.R
            * self.system.jerk_factors[:, :, None]
            * (z[:, -1, None] * z[:, 1 : 1 + self.d])[:, None, :]
        )
        result = []
        a, b, c, d = self.free
        for left in times[:-1]:
            h = np.maximum(left - t, 0)
            weights = np.array(
                (
                    h**3 / 6 + A * left**2 + B * left**3,
                    h * h / 2 + 2 * A * left + 3 * B * left**2,
                    h / 2 + A + 3 * B * left,
                    (left >= t) / 6 + B,
                )
            )
            free = np.array(
                (
                    ((d * left + c) * left + b) * left + a,
                    (3 * d * left + 2 * c) * left + b,
                    c + 3 * d * left,
                    d,
                )
            )
            result.append((free + np.einsum("ks,sjd->kjd", weights, forces)).transpose(1, 0, 2))
        return times, np.array(result), order
