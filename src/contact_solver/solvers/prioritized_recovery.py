"""Condensed matching recovery for prescribed moving obstacles."""

import time

import numpy as np
from scipy.optimize import root

from .cubic_utils import cubic_coefficients, eval_cubic, eval_cubic_vel
from .prioritized_utils import PiecewiseCubicTrajectory


def trajectory_key(p0, v0, pf, vf, contacts, obstacles, T, R):
    return (
        T,
        R,
        tuple(np.r_[p0, v0, pf, vf]),
        tuple((c.obstacle_idx, c.t_star, c.phi, c.nu) for c in contacts),
        tuple((index, id(obstacle)) for index, obstacle in obstacles.items()),
    )


def recover_reactions(self, p0, v0, pf, vf, contacts, obstacles, result, info, masses):
    from .prioritized_sci import _ObstacleContact, _ObstacleMatching

    m, T, R = len(contacts), self.T, self.R
    free = (
        np.array(cubic_coefficients(p0, v0, pf, vf, T))
        * np.array([1, T, T * T, T * T * T])[:, None]
    )
    indices = [c.obstacle_idx for c in contacts]

    def obstacle_state(obstacle, t):
        # Polynomial continuation is used only for rejected out-of-domain root
        # iterates. A returned solution must be strictly inside the horizon.
        segment = obstacle.segments[0]
        for c, a, b in obstacle.segments:
            if a <= t:
                segment = (c, a, b)
            else:
                break
        c, a, b = segment
        return eval_cubic(c, t - a), eval_cubic_vel(c, t - a) * T

    def states(w):
        z = w.reshape(m, 3)
        u, phi, gamma = z.T
        n = np.column_stack((np.cos(phi), np.sin(phi)))
        h = np.maximum(u[:, None] - u[None, :], 0)
        s = u[None, :]
        t = u[:, None]
        A = s * (1 - s) ** 2 / 2
        B = -(1 + 2 * s) * (1 - s) ** 2 / 6
        G = h**3 / 6 + A * t * t + B * t**3
        V = h * h / 2 + 2 * A * t + 3 * B * t * t
        r = ((free[3] * t + free[2]) * t + free[1]) * t + free[0] + R * G @ (gamma[:, None] * n)
        v = (3 * free[3] * t + 2 * free[2]) * t + free[1] + R * V @ (gamma[:, None] * n)
        obs = np.array(
            [obstacle_state(obstacles[index], u[k] * T) for k, index in enumerate(indices)]
        )
        return r - obs[:, 0], v - obs[:, 1], n

    def evaluate(w):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError
        r, v, n = states(w)
        return np.column_stack((r / R - n, np.sum(n * v, axis=1) / R)).ravel()

    initial = np.array([[c.t_star / T, c.phi, 0.0] for c in contacts])
    failed = np.array([[c.t_star / T, c.phi, mass * T**3] for c, mass in zip(result, masses, strict=True)])
    try:
        candidates = []
        for guess in (initial, failed):
            solved = root(
                evaluate,
                guess.ravel(),
                method="hybr",
                options={"xtol": 1e-10, "maxfev": max(500, 100 * m)},
            )
            error = float(np.max(np.abs(evaluate(solved.x))))
            self.recovery_attempts.append(
                dict(
                    contacts=m,
                    indices=indices,
                    initial=guess.tolist(),
                    success=bool(solved.success),
                    message=str(solved.message),
                    error=error,
                    solution=solved.x.reshape(m, 3).tolist(),
                )
            )
            self.matching_nfev += solved.nfev
            candidates.append((error, solved))
            if error <= self.options.sci_residual_tol:
                break
        solved = min(candidates, key=lambda item: item[0])[1]
        z = solved.x.reshape(m, 3)
        if (
            not np.isfinite(z).all()
            or np.max(np.abs(evaluate(solved.x))) > self.options.sci_residual_tol
        ):
            return result, info, masses
        order = np.argsort(z[:, 0])
        ts = z[order, 0] * T
        if np.min(np.diff(np.r_[0, ts, T])) <= self.options.sci_min_contact_gap:
            return result, info, masses
        r, v, n = states(solved.x)
        proposed = [
            _ObstacleContact(
                indices[k], z[k, 0] * T, z[k, 1], v[k] @ np.array([-n[k, 1], n[k, 0]]) / T
            )
            for k in order
        ]
        system = _ObstacleMatching(self, p0, v0, pf, vf, proposed, obstacles)
        residual = float(
            np.max(np.abs(system.evaluate(np.concatenate([c.to_vector() for c in proposed]))))
        )
        self.recovery_attempts[-1]["hermite_reconstruction_residual"] = residual
        # Preserve the actual condensed solution instead of reconstructing
        # short cubics by subtracting nearly equal endpoint states.
        u = z[:, 0]
        gamma = z[:, 2]
        normal = np.column_stack((np.cos(z[:, 1]), np.sin(z[:, 1])))
        A = u * (1 - u) ** 2 / 2
        B = -(1 + 2 * u) * (1 - u) ** 2 / 6
        knots = np.r_[0, z[order, 0], 1.0]
        segments = []
        for left, right in zip(knots[:-1], knots[1:], strict=True):
            h = np.maximum(left - u, 0)
            weights = np.array(
                [
                    h**3 / 6 + A * left**2 + B * left**3,
                    h * h / 2 + 2 * A * left + 3 * B * left**2,
                    h / 2 + A + 3 * B * left,
                    (left >= u) / 6 + B,
                ]
            )
            a, b, c, d = free
            f = np.array(
                [
                    ((d * left + c) * left + b) * left + a,
                    (3 * d * left + 2 * c) * left + b,
                    c + 3 * d * left,
                    d,
                ]
            )
            coefficients = (f + R * weights @ (gamma[:, None] * normal)) / np.array(
                [1, T, T * T, T * T * T]
            )[:, None]
            segments.append((tuple(coefficients), left * T, right * T))
        trajectory = PiecewiseCubicTrajectory(segments, T)
        boundary_error = max(
            np.max(np.abs(trajectory.eval_pos(0) - p0)),
            np.max(np.abs(trajectory.eval_vel(0) - v0)),
            np.max(np.abs(trajectory.eval_pos(T) - pf)),
            np.max(np.abs(trajectory.eval_vel(T) - vf)),
        )
        contact_error = max(
            np.max(
                np.abs(
                    trajectory.eval_pos(c.t_star)
                    - obstacles[c.obstacle_idx].eval_pos(c.t_star)
                    - R * np.array([np.cos(c.phi), np.sin(c.phi)])
                )
            )
            for c in proposed
        )
        tangent_error = max(
            abs(
                (trajectory.eval_vel(c.t_star) - obstacles[c.obstacle_idx].eval_vel(c.t_star))
                @ np.array([np.cos(c.phi), np.sin(c.phi)])
            )
            for c in proposed
        )
        if max(boundary_error, contact_error, tangent_error) > self.options.sci_residual_tol:
            return result, info, masses
        key = trajectory_key(p0, v0, pf, vf, proposed, obstacles, T, R)
        self._recovered_trajectory = (key, trajectory)
        residual = float(
            np.max(np.abs(system.evaluate(np.concatenate([c.to_vector() for c in proposed]))))
        )
        self.recovery_attempts[-1].update(
            direct_residual=residual,
            boundary_error=float(boundary_error),
            contact_error=float(contact_error),
            tangent_error=float(tangent_error),
        )
        if not np.isfinite(residual) or residual > self.options.sci_residual_tol:
            self._recovered_trajectory = None
            return result, info, masses
        new_info = dict(
            success=True,
            message="reaction_recovery",
            residual=residual,
            nfev=solved.nfev,
            iterations=None,
            sparse_fallbacks=0,
            reordered=bool(np.any(order != np.arange(m))),
        )
        self.roots.append(new_info)
        trajectory = self._build_trajectory(p0, v0, pf, vf, proposed, obstacles)
        checked_masses = [
            float(
                6
                * (trajectory.segments[k + 1][0][3] - trajectory.segments[k][0][3])
                @ np.array([np.cos(c.phi), np.sin(c.phi)])
                / R
            )
            for k, c in enumerate(proposed)
        ]
        return proposed, new_info, checked_masses
    except (TimeoutError, ValueError, FloatingPointError):
        self._recovered_trajectory = None
        return result, info, masses
