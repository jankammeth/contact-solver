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
from scipy.optimize import fsolve

from ..config import Config
from .base import Solver
from .cubic_utils import (
    cubic_coefficients,
    eval_cubic,
    eval_cubic_acc,
    eval_cubic_jerk,
    eval_cubic_vel,
    find_distance_minima,
)
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

    def __init__(self, config: Config, verbose: bool = False):
        super().__init__(config)
        self.verbose = verbose
        self.max_contacts_per_robot = config.solver.sci_max_contacts
        self.max_contacts_per_obstacle = config.solver.sci_max_contacts_per_pair
        self.detection_tol = config.solver.sci_detection_tol
        self.fsolve_xtol = config.solver.fsolve_xtol

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------

    def generate_trajectories(self, verbose: bool | None = None):
        verbose = verbose if verbose is not None else self.verbose
        t_wall_start = time.perf_counter()

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

        order = conflict_severity_order(p0, v0, pf, vf, self.T, self.R)

        if verbose:
            print(f"PrioritizedSCI: N={self.N}, T={self.T:.3f}, R={self.R:.3f}m")
            print(f"  priority order (conflict severity): {order}")

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

        times, positions, velocities, accelerations = self._sample_all(trajectories)
        min_dist = self._compute_min_dist(positions)
        converged = (min_dist >= self.R - self.detection_tol) and all(rank_converged)
        reason = "converged" if converged else "max_contacts_or_violation"

        t_wall_total = time.perf_counter() - t_wall_start

        return {
            "trajectories": {
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
            },
            "metrics": {
                "timing": {
                    "total_time": t_wall_total,
                    "wall_time": t_wall_total,
                    "per_rank_time": time_per_rank,
                },
                "converged": converged,
                "convergence_reason": reason,
                "num_contacts": sum(contacts_per_rank),
                "num_contacts_per_rank": contacts_per_rank,
                "max_contacts_single_pair": max(max_per_pair_per_rank) if max_per_pair_per_rank else 0,
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
        contacts: list[_ObstacleContact] = []
        obs_count: dict[int, int] = {j: 0 for j in obstacles}

        # Initial trajectory (unconstrained cubic)
        traj = self._build_trajectory(p0, v0, pf, vf, contacts, obstacles)
        ok = True

        for iteration in range(self.max_contacts_per_robot + 1):
            violations = self._find_violations(traj, obstacles, contacts)

            if not violations:
                break

            elig = [v for v in violations if obs_count[v[0]] < self.max_contacts_per_obstacle]
            if not elig:
                ok = False
                break
            if iteration == self.max_contacts_per_robot:
                ok = False
                break

            worst = min(elig, key=lambda v: v[2])
            obs_idx, t_viol, _ = worst

            new_contact = self._init_contact(obs_idx, t_viol, traj, obstacles)
            contacts.append(new_contact)
            contacts.sort(key=lambda c: c.t_star)
            obs_count[obs_idx] += 1

            contacts = self._solve_contact_system(p0, v0, pf, vf, contacts, obstacles)
            traj = self._build_trajectory(p0, v0, pf, vf, contacts, obstacles)

        max_per_pair = max(obs_count.values()) if obs_count else 0
        return traj, len(contacts), max_per_pair, ok

    # ------------------------------------------------------------------
    #  Trajectory construction through contact knot states
    # ------------------------------------------------------------------

    def _build_trajectory(self, p0, v0, pf, vf, contacts, obstacles):
        T = self.T
        R = self.R

        if not contacts:
            coeffs = cubic_coefficients(p0, v0, pf, vf, T)
            return PiecewiseCubicTrajectory(segments=[(coeffs, 0.0, T)], T=T)

        sorted_c = sorted(contacts, key=lambda c: c.t_star)
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
        """Exact continuous-time violation search.

        On the intersection of a robot segment and an obstacle segment the
        relative trajectory is a single cubic, so the minima of the
        distance are located exactly by find_distance_minima (real roots
        of the degree-5 derivative of the squared distance, plus the
        subinterval endpoints). Minima within detection_tol in time of an
        existing contact with the same obstacle are skipped, mirroring
        the joint solver.
        """
        R = self.R
        tol = self.detection_tol
        violations = []

        contact_times = {}
        for c in contacts:
            contact_times.setdefault(c.obstacle_idx, []).append(c.t_star)

        robot_bps = [seg[1] for seg in traj.segments] + [traj.T]

        for obs_idx, obs in obstacles.items():
            obs_bps = [seg[1] for seg in obs.segments] + [obs.T]
            bps = sorted(set(robot_bps + obs_bps))
            tcs = contact_times.get(obs_idx, [])

            for u0, u1 in zip(bps[:-1], bps[1:], strict=False):
                if u1 - u0 < 1e-12:
                    continue
                rc, rt_s = traj._find_segment(u0 + 0.5 * (u1 - u0))
                oc, ot_s = obs._find_segment(u0 + 0.5 * (u1 - u0))
                rc_loc = _shift_cubic(rc, u0 - rt_s)
                oc_loc = _shift_cubic(oc, u0 - ot_s)
                rel = tuple(rc_loc[k] - oc_loc[k] for k in range(4))

                for t, dist in find_distance_minima(rel, u1 - u0, u0):
                    if dist < R - tol and not any(
                        abs(t - tc) < tol for tc in tcs
                    ):
                        violations.append((obs_idx, float(t), float(dist)))

        return violations

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
            return contacts

        T = self.T
        R = self.R
        K = len(contacts)
        obs_idxs = [c.obstacle_idx for c in contacts]

        def residuals(x):
            c_list = []
            for i in range(K):
                t_star = float(np.clip(x[3 * i], 1e-6, T - 1e-6))
                phi = float(x[3 * i + 1])
                nu = float(x[3 * i + 2])
                c_list.append(
                    _ObstacleContact(
                        obstacle_idx=obs_idxs[i], t_star=t_star, phi=phi, nu=nu
                    )
                )
            c_list.sort(key=lambda c: c.t_star)

            traj = self._build_trajectory(p0, v0, pf, vf, c_list, obstacles)

            res = []

            # Acceleration continuity at each interior knot (each contact)
            for i in range(K):
                coeffs_left = traj.segments[i][0]
                t_left = traj.segments[i][2] - traj.segments[i][1]
                acc_left = eval_cubic_acc(coeffs_left, t_left)
                coeffs_right = traj.segments[i + 1][0]
                acc_right = eval_cubic_acc(coeffs_right, 0.0)
                res.extend(acc_left - acc_right)

            # Radial jerk jump: the robot's jerk kink at each contact must
            # point along the contact normal (stationarity; the tangency
            # condition is already absorbed by the contact parameterization)
            for i, c in enumerate(c_list):
                jerk_left = eval_cubic_jerk(traj.segments[i][0])
                jerk_right = eval_cubic_jerk(traj.segments[i + 1][0])
                dj = jerk_right - jerk_left
                res.append(dj[0] * np.sin(c.phi) - dj[1] * np.cos(c.phi))

            return np.array(res)

        x0 = np.concatenate([c.to_vector() for c in contacts])
        fsolve_kwargs = {"full_output": True}
        if self.fsolve_xtol > 0:
            fsolve_kwargs["xtol"] = self.fsolve_xtol
        sol, _info, _ier, _msg = fsolve(residuals, x0, **fsolve_kwargs)

        result = []
        for i in range(K):
            t_star = float(np.clip(sol[3 * i], 1e-6, T - 1e-6))
            phi = float(sol[3 * i + 1])
            nu = float(sol[3 * i + 2])
            result.append(
                _ObstacleContact(
                    obstacle_idx=obs_idxs[i], t_star=t_star, phi=phi, nu=nu
                )
            )
        result.sort(key=lambda c: c.t_star)
        return result

    # ------------------------------------------------------------------
    #  Sampling and aggregate metrics
    # ------------------------------------------------------------------

    def _sample_all(self, trajectories, n_points: int = 1000):
        times = np.linspace(0.0, self.T, n_points)
        positions = [np.zeros((n_points, 2)) for _ in range(self.N)]
        velocities = [np.zeros((n_points, 2)) for _ in range(self.N)]
        accelerations = [np.zeros((n_points, 2)) for _ in range(self.N)]

        for i in range(self.N):
            pos, vel, acc = trajectories[i].sample_at(times)
            positions[i] = pos
            velocities[i] = vel
            accelerations[i] = acc

        return times, positions, velocities, accelerations

    def _compute_min_dist(self, positions):
        pos = np.asarray(positions)  # (N, K, 2)
        if pos.shape[0] < 2:
            return float("inf")
        ii, jj = np.triu_indices(pos.shape[0], k=1)
        diffs = pos[ii] - pos[jj]
        return float(np.min(np.linalg.norm(diffs, axis=2)))
