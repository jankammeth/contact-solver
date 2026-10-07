"""Prioritized contact-based trajectory solver.

Solves an N-robot trajectory problem by prioritized planning: robots are
solved one at a time in conflict-severity order, and each robot treats the
previously committed trajectories as moving obstacles. The single-robot
subproblem is solved by sequential contact insertion on the variational
problem with obstacle trajectories entering the contact equations as
prescribed data.

Per-contact unknowns reduce from (4N-5) (joint solve) to 3 (t_star, phi, nu)
when the contact partner is a prescribed time-varying obstacle:
    t_star:  contact time
    phi:     contact angle around the obstacle (rel_pos = R * [cos phi, sin phi])
    nu:      tangential relative velocity at contact

Per contact, the matching system imposes acceleration continuity (2) and
the radial jerk-jump condition (1): the kink in the robot's jerk across
the contact points along the contact normal. Tangency and the contact
condition are absorbed by the parameterization.

Compared to the joint contact solver, there is one such 3-dim block per
obstacle contact; no center-of-mass mode is needed because each subproblem
has only one decision-making robot.
"""

import time
from dataclasses import dataclass

import numpy as np

from ..config import Config
from .base import Solver
from .collision_search import COLLISION_BACKEND, minima
from .cubic_utils import (
    cubic_coefficients,
    eval_cubic_acc,
    eval_cubic_jerk,
)
from .matching_numerics import damped_newton, insertion_time
from .prioritized_recovery import recover_reactions, trajectory_key
from .prioritized_utils import PiecewiseCubicTrajectory, conflict_severity_order


def _shift_cubic(coeffs, s):
    """Re-expand a cubic a + b*t + c*t^2 + d*t^3 around t = s.

    Returns coefficients (a', b', c', d') such that
    p'(tau) = p(tau + s) for all tau. Operates per component.
    """
    a, b, c, d = coeffs
    return (
        a + b * s + c * s**2 + d * s**3,
        b + 2 * c * s + 3 * d * s**2,
        c + 3 * d * s,
        d,
    )


@dataclass
class _ObstacleContact:
    obstacle_idx: int
    t_star: float
    phi: float
    nu: float

    def to_vector(self):
        return np.array([self.t_star, self.phi, self.nu])


class PrioritizedSCI(Solver):
    """Sequential contact insertion with prioritized planning."""

    def __init__(self, config: Config, verbose: bool = False, priority_order=None):
        super().__init__(config)
        self.verbose = verbose
        self.matching_seconds = 0.0
        self.matching_solves = 0
        self.matching_nfev = 0
        if priority_order is not None and sorted(priority_order) != list(range(self.N)):
            raise ValueError("priority_order must be a permutation of robot indices")
        self.priority_order = list(priority_order) if priority_order is not None else None
        self.detection_tol = config.solver.sci_detection_tol
        self.options = config.solver
        if (
            self.options.sci_root_max_iterations < 0
            or self.options.sci_residual_tol <= 0
            or min(
                self.options.sci_mass_tol,
                self.options.sci_detection_tol,
                self.options.sci_min_contact_gap,
            )
            < 0
        ):
            raise ValueError("Invalid SCI tolerances or limits")
        if not np.isfinite(self.options.timeout) or self.options.timeout <= 0:
            raise ValueError("Solve timeout must be positive and finite")
        if (
            np.any(np.isfinite(self.pos_min))
            or np.any(np.isfinite(self.pos_max))
            or any(
                value is not None
                for value in (
                    self.vel_min,
                    self.vel_max,
                    self.acc_min,
                    self.acc_max,
                    self.jerk_min,
                    self.jerk_max,
                )
            )
        ):
            raise ValueError("PrioritizedSCI supports unconstrained dynamics and workspace only")

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------

    def generate_trajectories(self, verbose: bool | None = None):
        verbose = verbose if verbose is not None else self.verbose
        t_call_start = time.perf_counter()
        self.matching_seconds = 0.0
        self.matching_solves = 0
        self.matching_nfev = 0

        p0 = self.initial_positions.reshape(self.N, 2)
        pf = self.final_positions.reshape(self.N, 2)
        v0 = (
            self.initial_velocities.reshape(self.N, 2)
            if self.initial_velocities is not None
            else np.zeros((self.N, 2))
        )
        vf = (
            self.final_velocities.reshape(self.N, 2)
            if self.final_velocities is not None
            else np.zeros((self.N, 2))
        )

        order = (
            list(self.priority_order)
            if self.priority_order is not None
            else conflict_severity_order(p0, v0, pf, vf, self.T, self.R)
        )

        preprocessing_time = time.perf_counter() - t_call_start
        t_wall_start = time.perf_counter()
        self.deadline = t_wall_start + self.options.timeout
        self.rank_checks = []
        self.roots = []
        self.removed_contacts = []
        self.insertion_offsets = []
        self.recovery_attempts = []
        self._recovered_trajectory = None

        if verbose:
            print(f"PrioritizedSCI: N={self.N}, T={self.T:.3f}, R={self.R:.3f}m")
            print(f"  priority order: {order}")

        trajectories: dict[int, PiecewiseCubicTrajectory] = {}
        contacts_per_rank: list[int] = []
        max_per_pair_per_rank: list[int] = []
        time_per_rank: list[float] = []
        rank_converged: list[bool] = []

        for step, robot_i in enumerate(order):
            t_rank_start = time.perf_counter()
            obstacles = dict(trajectories)  # {prior_robot_id: trajectory}
            traj_i, n_contacts, max_per_pair, ok = self._solve_single_robot(
                p0[robot_i],
                v0[robot_i],
                pf[robot_i],
                vf[robot_i],
                obstacles,
                verbose,
            )
            trajectories[robot_i] = traj_i
            contacts_per_rank.append(n_contacts)
            max_per_pair_per_rank.append(max_per_pair)
            time_per_rank.append(time.perf_counter() - t_rank_start)
            rank_converged.append(ok)

            if verbose:
                print(
                    f"  rank {step + 1}/{self.N} (robot {robot_i}): "
                    f"{n_contacts} contacts (max {max_per_pair}/pair), {time_per_rank[-1]:.3f}s, "
                    f"{'ok' if ok else 'incomplete'}"
                )

        t_solve = time.perf_counter() - t_wall_start
        t_post_start = time.perf_counter()
        self._trajectories = trajectories
        times, positions, velocities, accelerations = self.sample()
        min_dist = min((r["min_distance"] for r in self.rank_checks), default=np.inf)
        converged = all(rank_converged)
        reason = (
            "converged"
            if converged
            else next(r["reason"] for r in self.rank_checks if not r["valid"])
        )

        t_post = time.perf_counter() - t_post_start

        return {
            "times": times,
            "trajectories": {
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
            },
            "metrics": {
                "timing": {
                    "total_time": t_solve,
                    "preprocessing_time": preprocessing_time,
                    "wall_time": time.perf_counter() - t_call_start,
                    "solve_time": t_solve,
                    "postprocess_time": t_post,
                    "per_rank_time": time_per_rank,
                },
                "converged": converged,
                "collision_backend": COLLISION_BACKEND,
                "rank_checks": self.rank_checks,
                "roots": self.roots,
                "recovery_attempts": self.recovery_attempts,
                "removed_contacts": self.removed_contacts,
                "insertion_offsets": self.insertion_offsets,
                "convergence_reason": reason,
                "matching_seconds": self.matching_seconds,
                "matching_solves": self.matching_solves,
                "nfev": self.matching_nfev,
                "num_contacts": sum(contacts_per_rank),
                "num_contacts_per_rank": contacts_per_rank,
                "max_contacts_single_pair": max(max_per_pair_per_rank)
                if max_per_pair_per_rank
                else 0,
                "max_contacts_per_pair_per_rank": max_per_pair_per_rank,
                "min_distance": min_dist,
                "min_obstacle_distance": np.inf,
                "priority_order": order,
            },
        }

    # ------------------------------------------------------------------
    #  Single-robot subproblem with moving obstacles
    # ------------------------------------------------------------------

    def _solve_single_robot(self, p0, v0, pf, vf, obstacles, verbose):
        self._recovered_trajectory = None
        contacts = []
        traj = self._build_trajectory(p0, v0, pf, vf, contacts, obstacles)
        updates = 0
        reason, residual, minimum_mass = "converged", 0.0, 0.0
        while True:
            violations = self._find_violations(traj, obstacles, contacts)
            minimum = self._last_minimum
            if not violations:
                break
            if time.perf_counter() >= getattr(self, "deadline", np.inf):
                reason = "timeout"
                break
            obs_idx, t_viol, _ = min(violations, key=lambda v: v[2])
            if t_viol <= 0 or t_viol >= self.T:
                reason = "infeasible_endpoint_separation"
                break
            try:
                t_guess = insertion_time(t_viol, [c.t_star for c in contacts], self.T)
            except ValueError:
                reason = "coincident_initialization_failed"
                break
            if t_guess != t_viol:
                self.insertion_offsets.append(
                    dict(obstacle=obs_idx, collision_time=t_viol, initial_time=t_guess)
                )
            if (
                min(abs(t_guess - t) for t in [0.0, self.T] + [c.t_star for c in contacts])
                < self.options.sci_min_contact_gap
            ):
                reason = "minimum_knot_gap"
                break
            contacts.append(self._init_contact(obs_idx, t_guess, traj, obstacles))
            contacts.sort(key=lambda c: c.t_star)
            updates += 1
            while True:
                contacts, info, masses = self._solve_contact_system(
                    p0, v0, pf, vf, contacts, obstacles
                )
                traj = self._build_trajectory(p0, v0, pf, vf, contacts, obstacles)
                residual = info["residual"]
                minimum_mass = float(min(masses, default=0.0))
                if not info["success"]:
                    reason = info["message"]
                    break
                if minimum_mass >= -self.options.sci_mass_tol:
                    break
                if time.perf_counter() >= getattr(self, "deadline", np.inf):
                    reason = "timeout"
                    break
                index = int(np.argmin(masses))
                removed = contacts.pop(index)
                self.removed_contacts.append(
                    dict(obstacle=removed.obstacle_idx, time=removed.t_star, mass=minimum_mass)
                )
                updates += 1
            if reason != "converged":
                # The failed solve changed the path; report its current clearance.
                self._find_violations(traj, obstacles, contacts)
                minimum = self._last_minimum
                break
        ok = reason == "converged"
        self.rank_checks.append(
            dict(
                valid=ok,
                reason=reason,
                residual=residual,
                minimum_mass=minimum_mass,
                min_distance=minimum,
                active_set_updates=updates,
            )
        )
        counts = [sum(c.obstacle_idx == j for c in contacts) for j in obstacles]
        return traj, len(contacts), max(counts, default=0), ok

    # ------------------------------------------------------------------
    #  Trajectory construction through contact knot states
    # ------------------------------------------------------------------

    def _build_trajectory(self, p0, v0, pf, vf, contacts, obstacles):
        cached = getattr(self, "_recovered_trajectory", None)
        if cached is not None and cached[0] == trajectory_key(
            p0, v0, pf, vf, contacts, obstacles, self.T, self.R
        ):
            return cached[1]
        T = self.T
        R = self.R

        if not contacts:
            coeffs = cubic_coefficients(p0, v0, pf, vf, T)
            return PiecewiseCubicTrajectory(segments=[(coeffs, 0.0, T)], T=T)

        sorted_c = contacts
        gaps = np.diff([0.0] + [c.t_star for c in contacts] + [T])
        if np.any(gaps <= 0) or np.any(gaps < self.options.sci_min_contact_gap):
            raise ValueError("Contact times must be ordered and strictly interior")
        times = [0.0] + [c.t_star for c in sorted_c] + [T]
        states = [(p0.copy(), v0.copy())]

        for c in sorted_c:
            obs = obstacles[c.obstacle_idx]
            obs_pos = obs.eval_pos(c.t_star)
            obs_vel = obs.eval_vel(c.t_star)
            normal = np.array([np.cos(c.phi), np.sin(c.phi)])
            tangent = np.array([-np.sin(c.phi), np.cos(c.phi)])
            pos = obs_pos + R * normal
            vel = obs_vel + c.nu * tangent
            states.append((pos, vel))

        states.append((pf.copy(), vf.copy()))

        segments = []
        for k in range(len(times) - 1):
            t_s, t_e = times[k], times[k + 1]
            p_s, vel_s = states[k]
            p_e, vel_e = states[k + 1]
            coeffs = cubic_coefficients(p_s, vel_s, p_e, vel_e, t_e - t_s)
            segments.append((coeffs, t_s, t_e))

        return PiecewiseCubicTrajectory(segments=segments, T=T)

    # ------------------------------------------------------------------
    #  Violation detection (exact, continuous-time)
    # ------------------------------------------------------------------

    def _find_violations(self, traj, obstacles, contacts):
        """Find the deepest violation over all common polynomial intervals."""
        self._last_minimum = np.inf
        if not obstacles:
            return []
        coefficients, durations, offsets, indices = [], [], [], []

        robot_bps = [seg[1] for seg in traj.segments] + [traj.T]

        for obs_idx, obs in obstacles.items():
            obs_bps = [seg[1] for seg in obs.segments] + [obs.T]
            bps = sorted(set(robot_bps + obs_bps))

            for u0, u1 in zip(bps[:-1], bps[1:], strict=False):
                if u1 <= u0:
                    continue
                rc, rt_s = traj._find_segment(u0 + 0.5 * (u1 - u0))
                oc, ot_s = obs._find_segment(u0 + 0.5 * (u1 - u0))
                rc_loc = _shift_cubic(rc, u0 - rt_s)
                oc_loc = _shift_cubic(oc, u0 - ot_s)
                rel = tuple(rc_loc[k] - oc_loc[k] for k in range(4))

                coefficients.append(rel)
                durations.append(u1 - u0)
                offsets.append(u0)
                indices.append(obs_idx)

        times, distances = minima(coefficients, durations)
        k = int(np.argmin(distances))
        self._last_minimum = float(distances[k])
        if distances[k] < self.R - self.detection_tol:
            return [(indices[k], float(offsets[k] + times[k] * durations[k]), float(distances[k]))]
        return []

    def _init_contact(self, obs_idx, t_star, traj, obstacles):
        obs = obstacles[obs_idx]
        rel_pos = traj.eval_pos(t_star) - obs.eval_pos(t_star)
        rel_vel = traj.eval_vel(t_star) - obs.eval_vel(t_star)

        d = np.linalg.norm(rel_pos)
        if d > 1e-10:
            phi = float(np.arctan2(rel_pos[1], rel_pos[0]))
        else:
            phi = float(np.arctan2(-rel_vel[0], rel_vel[1]))

        tangent = np.array([-np.sin(phi), np.cos(phi)])
        nu = float(np.dot(rel_vel, tangent))

        return _ObstacleContact(obstacle_idx=obs_idx, t_star=float(t_star), phi=phi, nu=nu)

    # ------------------------------------------------------------------
    #  Algebraic system: acceleration continuity + radial jerk jump
    # ------------------------------------------------------------------

    def _solve_contact_system(self, p0, v0, pf, vf, contacts, obstacles):
        if not contacts:
            return [], dict(success=True, message="empty_active_set", residual=0.0), []
        system = _ObstacleMatching(self, p0, v0, pf, vf, contacts, obstacles)
        x0 = np.concatenate([c.to_vector() for c in contacts])
        started = time.perf_counter()
        x, info = damped_newton(system, x0, getattr(self, "deadline", np.inf), self.options)
        self.matching_seconds += time.perf_counter() - started
        self.matching_solves += 1
        self.matching_nfev += info["nfev"]
        info["residual"] = float(np.max(np.abs(system.evaluate(x))))
        info["success"] = bool(
            info["success"] and info["residual"] <= self.options.sci_residual_tol
        )
        self.roots.append(info)
        result = system.contacts(x)
        traj = self._build_trajectory(p0, v0, pf, vf, result, obstacles)
        masses = [
            float(
                (eval_cubic_jerk(traj.segments[k + 1][0]) - eval_cubic_jerk(traj.segments[k][0]))
                @ np.array([np.cos(c.phi), np.sin(c.phi)])
                / self.R
            )
            for k, c in enumerate(result)
        ]
        if (
            self.options.sci_reaction_recovery
            and self.options.sci_root_max_iterations > 0
            and not info["success"]
            and info["message"] != "timeout"
        ):
            started = time.perf_counter()
            recovered = self._recover_reactions(
                p0, v0, pf, vf, contacts, obstacles, result, info, masses
            )
            self.matching_seconds += time.perf_counter() - started
            return recovered
        return result, info, masses

    def _recover_reactions(self, *args):
        return recover_reactions(self, *args)

    # ------------------------------------------------------------------
    #  Sampling and aggregate metrics
    # ------------------------------------------------------------------

    def sample(self, n=1000, times=None):
        times = np.linspace(0.0, self.T, n) if times is None else np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or not np.isfinite(times).all()
            or np.any(times < 0)
            or np.any(times > self.T)
        ):
            raise ValueError("Sample times must be finite and within [0, T]")
        values = [self._trajectories[i].sample_at(times) for i in range(self.N)]
        return times, *[[value[k] for value in values] for k in range(3)]


class _ObstacleMatching:
    """Ordered three-variable blocks; no projection inside the matching equations."""

    analytical_jacobian = False
    b = 3

    def __init__(self, solver, p0, v0, pf, vf, contacts, obstacles):
        self.solver, self.boundary, self.obstacles = solver, (p0, v0, pf, vf), obstacles
        self.indices = [c.obstacle_idx for c in contacts]
        self.m, self.T, self.min_gap = len(contacts), solver.T, solver.options.sci_min_contact_gap

    def knot_times(self, x):
        if not np.all(np.isfinite(x)):
            raise ValueError("Nonfinite contact state")
        times = np.r_[0.0, x.reshape(self.m, 3)[:, 0], self.T]
        if np.any(np.diff(times) <= 0) or np.any(np.diff(times) < self.min_gap):
            raise ValueError("Contact times must be ordered and interior")
        return times

    def contacts(self, x):
        self.knot_times(x)
        return [
            _ObstacleContact(j, *z) for j, z in zip(self.indices, x.reshape(self.m, 3), strict=True)
        ]

    def evaluate(self, x):
        contacts = self.contacts(x)
        traj = self.solver._build_trajectory(*self.boundary, contacts, self.obstacles)
        residual = np.zeros((self.m, 3))
        for k, c in enumerate(contacts):
            left, a, b = traj.segments[k]
            right = traj.segments[k + 1][0]
            residual[k, :2] = eval_cubic_acc(right, 0.0) - eval_cubic_acc(left, b - a)
            jump = eval_cubic_jerk(right) - eval_cubic_jerk(left)
            residual[k, 2] = jump @ np.array([-np.sin(c.phi), np.cos(c.phi)])
        return residual.ravel()
