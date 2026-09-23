"""Explicit-vector contact coordinates and local Hermite matching equations.

Robot indices are zero based: r[j] = x[0] - x[j + 1]. Blocks retain physical
contact times, unconstrained normal/velocity vectors, and the reaction mass.
Normalization and tangency are equations, never projections inside F or DF.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class VectorContact:
    pair: tuple[int, int]
    t_star: float
    normal: np.ndarray
    velocity: np.ndarray
    mass: float
    r_others: np.ndarray
    v_others: np.ndarray

    def to_vector(self):
        others = np.stack((self.r_others, self.v_others), axis=1).ravel()
        return np.concatenate(([self.t_star], self.normal, self.velocity, [self.mass], others))

    @classmethod
    def from_vector(cls, vector, pair, n_robots, dimension):
        x = np.asarray(vector)
        d = dimension
        if x.shape != (2 * d * (n_robots - 1) + 2,):
            raise ValueError("Wrong contact block size")
        others = x[2 * d + 2 :].reshape(n_robots - 2, 2, d)
        return cls(
            pair,
            x[0],
            x[1 : 1 + d].copy(),
            x[1 + d : 1 + 2 * d].copy(),
            x[1 + 2 * d],
            others[:, 0].copy(),
            others[:, 1].copy(),
        )


def hermite_junction_weights(h, at_end):
    """Acceleration and jerk weights on (r0, v0, r1, v1), and d/dh."""
    if at_end:
        acceleration = [6 / h**2, 2 / h, -6 / h**2, 4 / h]
        da = [-12 / h**3, -2 / h**2, 12 / h**3, -4 / h**2]
    else:
        acceleration = [-6 / h**2, -4 / h, 6 / h**2, -2 / h]
        da = [12 / h**3, 4 / h**2, -12 / h**3, 2 / h**2]
    jerk = [12 / h**3, 6 / h**2, -12 / h**3, 6 / h**2]
    dj = [-36 / h**4, -12 / h**3, 36 / h**4, -12 / h**3]
    return np.array([acceleration, jerk]), np.array([da, dj])


class MatchingSystem:
    """Square system with b = 2d(N-1)+2 equations per contact.

    Residual order: acceleration jumps, jerk balance, n.n-1, n.v.
    Pairs/order are fixed for each root solve. Invalid times are rejected, not
    sorted or clipped. The caller must keep all steps in the feasible domain.
    """

    def __init__(
        self,
        pairs,
        initial_position,
        initial_velocity,
        final_position,
        final_velocity,
        radius,
        horizon,
        min_gap,
    ):
        self.pairs = [tuple(pair) for pair in pairs]
        self.r0 = np.asarray(initial_position)
        self.v0 = np.asarray(initial_velocity)
        self.rf = np.asarray(final_position)
        self.vf = np.asarray(final_velocity)
        self.M, self.d = self.r0.shape
        self.N = self.M + 1
        self.m = len(pairs)
        self.b = 2 * self.M * self.d + 2
        self.R, self.T, self.min_gap = radius, horizon, min_gap
        self.position_maps = np.zeros((self.m, self.M, self.d, self.b))
        self.velocity_maps = np.zeros_like(self.position_maps)
        self.jerk_factors = np.zeros((self.m, self.M))
        for k, (p, q) in enumerate(self.pairs):
            if not 0 <= p < q < self.N:
                raise ValueError("Each contact requires 0 <= p < q < N")
            dr, dv = self.position_maps[k], self.velocity_maps[k]
            cursor = 2 * self.d + 2
            for j in range(self.M):
                if j != q - 1:
                    dr[j, :, cursor : cursor + self.d] = np.eye(self.d)
                    dv[j, :, cursor + self.d : cursor + 2 * self.d] = np.eye(self.d)
                    cursor += 2 * self.d
            if p:
                dr[q - 1] = dr[p - 1]
                dv[q - 1] = dv[p - 1]
            dr[q - 1, :, 1 : 1 + self.d] = self.R * np.eye(self.d)
            dv[q - 1, :, 1 + self.d : 1 + 2 * self.d] = np.eye(self.d)
            sigma = np.zeros(self.N)
            sigma[p], sigma[q] = 1, -1
            self.jerk_factors[k] = sigma[0] - sigma[1:]

    def unpack(self, x):
        x = np.asarray(x)
        if x.shape != (self.m * self.b,) or not np.all(np.isfinite(x)):
            raise ValueError("Expected a finite vector with m*b entries")
        return x.reshape(self.m, self.b)

    def knot_times(self, x):
        blocks = self.unpack(x)
        times = np.concatenate(([0], blocks[:, 0], [self.T]))
        if np.any(np.diff(times) < self.min_gap) or np.any(np.diff(times) <= 0):
            raise ValueError("Contact times must be ordered, interior, and respect the minimum gap")
        return times

    def states(self, x):
        z = self.unpack(x)
        times = self.knot_times(x)
        r = np.concatenate(
            (self.r0[None], np.einsum("kmdb,kb->kmd", self.position_maps, z), self.rf[None])
        )
        v = np.concatenate(
            (self.v0[None], np.einsum("kmdb,kb->kmd", self.velocity_maps, z), self.vf[None])
        )
        return times, r, v

    def evaluate(self, x, with_jacobian=False):
        z = self.unpack(x)
        times, r, v = self.states(x)
        md = self.M * self.d
        residual = np.zeros((self.m, self.b))
        blocks = np.zeros((self.m, 3, self.b, self.b)) if with_jacobian else None
        for k in range(self.m):
            for s, sign, at_end in ((k, -1, True), (k + 1, 1, False)):
                h = times[s + 1] - times[s]
                weights, dw = hermite_junction_weights(h, at_end)
                boundary = np.array([r[s], v[s], r[s + 1], v[s + 1]])
                residual[k, : 2 * md] += sign * np.einsum("ai,imd->amd", weights, boundary).ravel()
                if with_jacobian:
                    duration_derivative = np.einsum("ai,imd->amd", dw, boundary).ravel()
                    for knot, pos_col, time_sign in ((s, 0, -1), (s + 1, 2, 1)):
                        neighbor = knot - 1
                        if not 0 <= neighbor < self.m:
                            continue
                        block = blocks[k, neighbor - k + 1]
                        state_derivative = (
                            weights[:, pos_col, None, None, None] * self.position_maps[neighbor]
                            + weights[:, pos_col + 1, None, None, None]
                            * self.velocity_maps[neighbor]
                        )
                        block[: 2 * md] += sign * state_derivative.reshape(2 * md, self.b)
                        block[: 2 * md, 0] += sign * time_sign * duration_derivative
            n = z[k, 1 : 1 + self.d]
            velocity = z[k, 1 + self.d : 1 + 2 * self.d]
            alpha = z[k, 1 + 2 * self.d]
            factor = self.R * self.jerk_factors[k]
            residual[k, md : 2 * md] -= (alpha * factor[:, None] * n).ravel()
            residual[k, -2:] = [n @ n - 1, n @ velocity]
            if with_jacobian:
                block = blocks[k, 1]
                block[md : 2 * md, 1 : 1 + self.d] -= (
                    alpha * factor[:, None, None] * np.eye(self.d)
                ).reshape(md, self.d)
                block[md : 2 * md, 1 + 2 * self.d] -= (factor[:, None] * n).ravel()
                block[-2, 1 : 1 + self.d] = 2 * n
                block[-1, 1 : 1 + self.d] = velocity
                block[-1, 1 + self.d : 1 + 2 * self.d] = n
        return (residual.ravel(), blocks) if with_jacobian else residual.ravel()

    def jacobian(self, x):
        """Sparse analytical Jacobian; toarray() is useful for small derivative tests."""
        return blocks_to_sparse(self.evaluate(x, True)[1])

    def contacts(self, x):
        return [
            VectorContact.from_vector(z, pair, self.N, self.d)
            for z, pair in zip(self.unpack(x), self.pairs, strict=True)
        ]


def blocks_to_banded(blocks):
    """LAPACK general-band storage, without constructing a dense m*b square array."""
    m, _, b, _ = blocks.shape
    bandwidth = min(2 * b - 1, m * b - 1)
    band = np.zeros((2 * bandwidth + 1, m * b))
    rows, cols = np.indices((b, b))
    for k in range(m):
        for slot in range(3):
            neighbor = k + slot - 1
            if 0 <= neighbor < m:
                i, j = k * b + rows, neighbor * b + cols
                band[bandwidth + i - j, j] = blocks[k, slot]
    return bandwidth, band


def blocks_to_sparse(blocks):
    m, _, b, _ = blocks.shape
    rows, cols, values = [], [], []
    for k in range(m):
        for slot in range(3):
            neighbor = k + slot - 1
            if 0 <= neighbor < m:
                i, j = np.nonzero(blocks[k, slot])
                rows.extend(k * b + i)
                cols.extend(neighbor * b + j)
                values.extend(blocks[k, slot, i, j])
    return sparse.csr_matrix((values, (rows, cols)), shape=(m * b, m * b))
