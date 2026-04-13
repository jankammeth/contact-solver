"""Contact-based trajectory optimization using sequential contact insertion."""

import time
from dataclasses import dataclass
from itertools import combinations

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
from .velocity_arcs import apply_velocity_bounds_to_trajectories


@dataclass
class _Contact:
    pair: tuple[int, int]
    t_star: float
    phi: float
    lam: float
    r_others: dict
    v_others: dict

    def to_vector(self):
        vec = [self.t_star, self.phi, self.lam]
        for i in sorted(self.r_others.keys()):
            vec.extend(self.r_others[i])
            vec.extend(self.v_others[i])
        return np.array(vec)

    @classmethod
    def from_vector(cls, x, pair, other_indices):
        r_others, v_others = {}, {}
        idx = 3
        for i in other_indices:
            r_others[i] = np.array([x[idx], x[idx + 1]])
            v_others[i] = np.array([x[idx + 2], x[idx + 3]])
            idx += 4
        return cls(pair=pair, t_star=x[0], phi=x[1], lam=x[2], r_others=r_others, v_others=v_others)


@dataclass
class _ObstacleContact:
    robot: int
    obstacle: int
    t_star: float
    phi: float

    def to_vector(self):
        return np.array([self.t_star, self.phi])

    @classmethod
    def from_vector(cls, x, robot, obstacle):
        return cls(robot=robot, obstacle=obstacle, t_star=x[0], phi=x[1])


class ContactSolver(Solver):
    """Contact-based trajectory solver using sequential contact insertion."""

    def __init__(
        self,
        config: Config,
        obstacle_positions: np.ndarray | None = None,
        verbose: bool = False,
        max_contacts: int | None = None,
        max_contacts_per_pair: int | None = None,
        apply_velocity_bounds: bool | None = None,
    ):
        super().__init__(config)
        self.verbose = verbose
        self.obstacle_positions = obstacle_positions
        self.max_contacts = max_contacts or config.solver.sci_max_contacts
        self.max_contacts_per_pair = max_contacts_per_pair or config.solver.sci_max_contacts_per_pair
        self.apply_velocity_bounds = (
            apply_velocity_bounds
            if apply_velocity_bounds is not None
            else config.solver.apply_velocity_bounds
        )
        self.detection_tol = config.solver.sci_detection_tol
        self.min_contact_gap = config.solver.sci_min_contact_gap
        self.fsolve_xtol = config.solver.fsolve_xtol

    def generate_trajectories(self, verbose: bool | None = None):
        verbose = verbose if verbose is not None else self.verbose
        t_start = time.time()

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

        r0 = {i: p0[0] - p0[i + 1] for i in range(self.N - 1)}
        v0_rel = {i: v0[0] - v0[i + 1] for i in range(self.N - 1)}
        rf = {i: pf[0] - pf[i + 1] for i in range(self.N - 1)}
        vf_rel = {i: vf[0] - vf[i + 1] for i in range(self.N - 1)}

        c_coeffs = cubic_coefficients(
            np.mean(p0, axis=0),
            np.mean(v0, axis=0),
            np.mean(pf, axis=0),
            np.mean(vf, axis=0),
            self.T,
        )

        pairs = list(combinations(range(self.N), 2))
        jerk_coeffs = self._compute_jerk_coeffs(pairs)

        contacts = []
        pair_cnt = {pair: 0 for pair in pairs}

        obs_contacts = []
        if self.obstacle_positions is not None and len(self.obstacle_positions) > 0:
            n_obs = len(self.obstacle_positions)
            obs_cnt = {(i, j): 0 for i in range(self.N) for j in range(n_obs)}
        else:
            obs_cnt = {}

        if verbose:
            print(f"ContactSolver: N={self.N}, T={self.T:.3f}, R={self.R:.3f}m")
            if self.obstacle_positions is not None:
                print(f"  {len(self.obstacle_positions)} obstacles")

        reason = "converged"
        for iteration in range(self.max_contacts + 1):
            segments = self._build_segments(contacts, r0, v0_rel, rf, vf_rel)

            violations = self._find_violations(segments, contacts, pairs)
            elig = [v for v in violations if pair_cnt[v[0]] < self.max_contacts_per_pair]

            obs_violations = self._find_obstacle_violations(segments, c_coeffs, obs_contacts)
            elig_obs = [
                v for v in obs_violations
                if obs_cnt.get((v[0], v[1]), 0) < self.max_contacts_per_pair
            ]

            if verbose:
                if violations or obs_violations:
                    worst_rr = min(violations, key=lambda v: v[2]) if violations else None
                    worst_ro = min(obs_violations, key=lambda v: v[3]) if obs_violations else None
                    msg = f"  Iter {iteration}: {len(violations)} robot-robot"
                    if obs_violations:
                        msg += f", {len(obs_violations)} robot-obstacle"
                    if worst_rr:
                        msg += f", worst RR: {worst_rr[0]} @ t={worst_rr[1]:.4f}s, d={worst_rr[2]:.4f}m"
                    if worst_ro:
                        msg += f", worst RO: r{worst_ro[0]}-o{worst_ro[1]} @ t={worst_ro[2]:.4f}s, d={worst_ro[3]:.4f}m"
                    print(msg)
                else:
                    print(f"  Iter {iteration}: Converged!")

            if not violations and not obs_violations:
                break

            if not elig and not elig_obs:
                reason = "max_contacts_per_pair"
                break

            if iteration == self.max_contacts:
                reason = "max_contacts"
                break

            worst_rr_dist = min(v[2] for v in elig) if elig else np.inf
            worst_ro_dist = min(v[3] for v in elig_obs) if elig_obs else np.inf

            if worst_rr_dist <= worst_ro_dist and elig:
                worst = min(elig, key=lambda v: v[2])
                pair, t_viol, _ = worst
                new_contact = self._init_contact(pair, t_viol, segments, pairs)
                contacts.append(new_contact)
                contacts.sort(key=lambda c: c.t_star)
                pair_cnt[pair] += 1

                contacts, _ = self._solve_system(contacts, pairs, jerk_coeffs, r0, v0_rel, rf, vf_rel)
            elif elig_obs:
                worst = min(elig_obs, key=lambda v: v[3])
                robot, obs_idx, t_viol, _ = worst
                new_obs_contact = self._init_obstacle_contact(robot, obs_idx, t_viol, segments, c_coeffs)
                obs_contacts.append(new_obs_contact)
                obs_contacts.sort(key=lambda c: c.t_star)
                obs_cnt[(robot, obs_idx)] = obs_cnt.get((robot, obs_idx), 0) + 1

        segments = self._build_segments(contacts, r0, v0_rel, rf, vf_rel)
        times, positions, velocities, accelerations = self._sample(segments, c_coeffs)
        contact_times = [c.t_star for c in contacts]
        arc_times = []

        v_max = self.config.problem.dynamics.vel_max
        if self.apply_velocity_bounds and v_max > 0:
            max_vel = max(np.max(np.abs(v)) for v in velocities)
            if max_vel > v_max + 1e-6:
                if verbose:
                    print(f"  Velocity exceeded: {max_vel:.4f} > {v_max:.4f}, inserting arcs...")
                positions, velocities, accelerations, arc_times = (
                    apply_velocity_bounds_to_trajectories(
                        times, positions, velocities, accelerations, contact_times, v_max, verbose
                    )
                )

        min_dist = self._compute_min_dist(positions)
        min_obs_dist = self._compute_min_obstacle_dist(positions) if self.obstacle_positions is not None else np.inf
        self.K = len(times)

        # Compute max contacts on any single pair
        max_contacts_single_pair = max(pair_cnt.values()) if pair_cnt else 0

        return {
            "trajectories": {
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
            },
            "metrics": {
                "timing": {"total_time": time.time() - t_start},
                "converged": min_dist >= self.R - self.detection_tol and min_obs_dist >= self.R - self.detection_tol,
                "convergence_reason": reason,
                "num_contacts": len(contacts),
                "num_obstacle_contacts": len(obs_contacts),
                "max_contacts_single_pair": max_contacts_single_pair,
                "min_distance": min_dist,
                "min_obstacle_distance": min_obs_dist,
                "num_velocity_arcs": len(arc_times) // 2,
            },
            "contact_times": contact_times,
            "arc_times": arc_times,
        }

    def _compute_jerk_coeffs(self, pairs):
        coeffs = {}
        for pair in pairs:
            a, b = pair
            c = {}
            for i in range(self.N - 1):
                val = 0.0
                if a == 0:
                    val += 1.0
                elif b == 0:
                    val -= 1.0
                if a == i + 1:
                    val -= 1.0
                elif b == i + 1:
                    val += 1.0
                c[i] = val
            coeffs[pair] = c
        return coeffs

    def _get_pair_expr(self, pair):
        a, b = pair
        if a == 0:
            return [b - 1], [1.0]
        return [a - 1, b - 1], [-1.0, 1.0]

    def _get_other_indices(self, pair):
        a, b = pair
        if a == 0:
            return [i for i in range(self.N - 1) if i != b - 1]
        return [i for i in range(self.N - 1) if i != b - 1]

    def _build_segments(self, contacts, r0, v0_rel, rf, vf_rel):
        if not contacts:
            return {
                i: [(cubic_coefficients(r0[i], v0_rel[i], rf[i], vf_rel[i], self.T), 0.0, self.T)]
                for i in range(self.N - 1)
            }

        sorted_c = sorted(contacts, key=lambda c: c.t_star)
        times = [0.0] + [c.t_star for c in sorted_c] + [self.T]

        states = {i: [(r0[i], v0_rel[i])] for i in range(self.N - 1)}
        for contact in sorted_c:
            rel_states = self._get_contact_states(contact)
            for i in range(self.N - 1):
                states[i].append(rel_states[i])
        for i in range(self.N - 1):
            states[i].append((rf[i], vf_rel[i]))

        segments = {i: [] for i in range(self.N - 1)}
        for seg_idx in range(len(times) - 1):
            t_s, t_e = times[seg_idx], times[seg_idx + 1]
            for i in range(self.N - 1):
                r_s, v_s = states[i][seg_idx]
                r_e, v_e = states[i][seg_idx + 1]
                segments[i].append((cubic_coefficients(r_s, v_s, r_e, v_e, t_e - t_s), t_s, t_e))
        return segments

    def _get_contact_states(self, contact):
        a, b = contact.pair
        normal = np.array([np.cos(contact.phi), np.sin(contact.phi)])
        tangent = np.array([-np.sin(contact.phi), np.cos(contact.phi)])
        delta_r = self.R * normal
        delta_v = contact.lam * tangent

        states = {}
        if a == 0:
            states[b - 1] = (delta_r.copy(), delta_v.copy())
            for i in contact.r_others:
                states[i] = (contact.r_others[i].copy(), contact.v_others[i].copy())
        else:
            r_a = contact.r_others[a - 1]
            v_a = contact.v_others[a - 1]
            states[a - 1] = (r_a.copy(), v_a.copy())
            states[b - 1] = (r_a + delta_r, v_a + delta_v)
            for i in contact.r_others:
                if i != a - 1:
                    states[i] = (contact.r_others[i].copy(), contact.v_others[i].copy())
        return states

    def _find_violations(self, segments, contacts, pairs):
        tol = self.detection_tol
        contact_times = [c.t_star for c in contacts]
        violations = []
        n_segs = len(segments[0])

        for seg_idx in range(n_segs):
            t_s = segments[0][seg_idx][1]
            t_e = segments[0][seg_idx][2]
            T_seg = t_e - t_s

            for pair in pairs:
                indices, coeffs = self._get_pair_expr(pair)
                combined = [np.zeros(2) for _ in range(4)]
                for idx, coef in zip(indices, coeffs, strict=False):
                    seg_coeffs = segments[idx][seg_idx][0]
                    for k in range(4):
                        combined[k] += coef * seg_coeffs[k]

                for t, dist in find_distance_minima(tuple(combined), T_seg, t_s):
                    if not any(abs(t - tc) < tol for tc in contact_times):
                        if dist < self.R - tol:
                            violations.append((pair, t, dist))
        return violations

    def _find_obstacle_violations(self, segments, c_coeffs, obs_contacts):
        tol = self.detection_tol
        if self.obstacle_positions is None or len(self.obstacle_positions) == 0:
            return []

        violations = []
        obs_contact_times = [c.t_star for c in obs_contacts]
        n_segs = len(segments[0])

        for robot in range(self.N):
            for obs_idx, obs_pos in enumerate(self.obstacle_positions):
                for seg_idx in range(n_segs):
                    t_s = segments[0][seg_idx][1]
                    t_e = segments[0][seg_idx][2]
                    T_seg = t_e - t_s

                    n_samples = 50
                    for k in range(n_samples + 1):
                        t_local = k * T_seg / n_samples
                        t = t_s + t_local

                        r_vals = {i: eval_cubic(segments[i][seg_idx][0], t_local) for i in range(self.N - 1)}
                        c_pos = eval_cubic(c_coeffs, t)
                        sum_r = sum(r_vals.values()) if r_vals else np.zeros(2)

                        if robot == 0:
                            p_robot = c_pos + sum_r / self.N
                        else:
                            p_robot = c_pos + sum_r / self.N - r_vals[robot - 1]

                        dist = np.linalg.norm(p_robot - obs_pos)

                        if dist < self.R - tol:
                            if not any(abs(t - tc) < tol for tc in obs_contact_times):
                                violations.append((robot, obs_idx, t, dist))

        return violations

    def _init_contact(self, pair, t_star, segments, pairs):
        seg_idx = next(i for i, s in enumerate(segments[0]) if s[1] <= t_star <= s[2] + 1e-10)
        t_local = t_star - segments[0][seg_idx][1]

        r_vals = {i: eval_cubic(segments[i][seg_idx][0], t_local) for i in range(self.N - 1)}
        v_vals = {i: eval_cubic_vel(segments[i][seg_idx][0], t_local) for i in range(self.N - 1)}

        indices, coeffs = self._get_pair_expr(pair)
        delta_r = sum(c * r_vals[i] for i, c in zip(indices, coeffs, strict=False))
        delta_v = sum(c * v_vals[i] for i, c in zip(indices, coeffs, strict=False))

        dist = np.linalg.norm(delta_r)
        phi = (
            np.arctan2(delta_r[1], delta_r[0])
            if dist > 1e-10
            else np.arctan2(-delta_v[0], delta_v[1])
        )
        tangent = np.array([-np.sin(phi), np.cos(phi)])
        lam = np.dot(delta_v, tangent)

        other_idx = self._get_other_indices(pair)
        return _Contact(
            pair=pair,
            t_star=t_star,
            phi=phi,
            lam=lam,
            r_others={i: r_vals[i] for i in other_idx},
            v_others={i: v_vals[i] for i in other_idx},
        )

    def _init_obstacle_contact(self, robot, obs_idx, t_star, segments, c_coeffs):
        seg_idx = next(i for i, s in enumerate(segments[0]) if s[1] <= t_star <= s[2] + 1e-10)
        t_local = t_star - segments[0][seg_idx][1]

        obs_pos = self.obstacle_positions[obs_idx]

        r_vals = {i: eval_cubic(segments[i][seg_idx][0], t_local) for i in range(self.N - 1)}
        c_pos = eval_cubic(c_coeffs, t_local + segments[0][seg_idx][1])
        sum_r = sum(r_vals.values()) if r_vals else np.zeros(2)

        if robot == 0:
            p_robot = c_pos + sum_r / self.N
        else:
            p_robot = c_pos + sum_r / self.N - r_vals[robot - 1]

        diff = p_robot - obs_pos
        dist = np.linalg.norm(diff)
        phi = np.arctan2(diff[1], diff[0]) if dist > 1e-10 else 0.0

        return _ObstacleContact(robot=robot, obstacle=obs_idx, t_star=t_star, phi=phi)

    def _solve_system(self, contacts, pairs, jerk_coeffs, r0, v0_rel, rf, vf_rel):
        if not contacts:
            return contacts, 0.0

        K = len(contacts)
        c_pairs = [c.pair for c in contacts]
        other_idx_list = [self._get_other_indices(p) for p in c_pairs]
        vars_per = 4 * self.N - 5

        T = self.T
        min_gap = self.min_contact_gap

        def equations(x):
            c_list = []
            for i in range(K):
                vec = x[vars_per * i : vars_per * (i + 1)].copy()
                vec[0] = np.clip(vec[0], min_gap, T - min_gap)
                c_list.append(_Contact.from_vector(vec, c_pairs[i], other_idx_list[i]))
            c_list.sort(key=lambda c: c.t_star)
            segs = self._build_segments(c_list, r0, v0_rel, rf, vf_rel)

            residuals = []
            for i, contact in enumerate(c_list):
                T_before = segs[0][i][2] - segs[0][i][1]

                for j in range(self.N - 1):
                    a_bef = eval_cubic_acc(segs[j][i][0], T_before)
                    a_aft = eval_cubic_acc(segs[j][i + 1][0], 0.0)
                    residuals.extend(a_bef - a_aft)

                jerks_bef = {j: eval_cubic_jerk(segs[j][i][0]) for j in range(self.N - 1)}
                jerks_aft = {j: eval_cubic_jerk(segs[j][i + 1][0]) for j in range(self.N - 1)}
                delta_j = {j: jerks_aft[j] - jerks_bef[j] for j in range(self.N - 1)}

                jc = jerk_coeffs[contact.pair]
                ref_idx = next((j for j, c in jc.items() if abs(c) > 1e-10), None)
                if ref_idx is None:
                    residuals.extend([0.0] * (2 * (self.N - 2) + 1))
                    continue

                for j in range(self.N - 1):
                    if j != ref_idx:
                        ratio = jc[j] / jc[ref_idx]
                        residuals.extend(delta_j[j] - ratio * delta_j[ref_idx])

                normal = np.array([np.cos(contact.phi), np.sin(contact.phi)])
                residuals.append(delta_j[ref_idx][0] * normal[1] - delta_j[ref_idx][1] * normal[0])

            return np.array(residuals)

        x0 = np.concatenate([c.to_vector() for c in contacts])
        fsolve_kwargs = {"full_output": True}
        if self.fsolve_xtol > 0:
            fsolve_kwargs["xtol"] = self.fsolve_xtol
        sol, info, ier, _ = fsolve(equations, x0, **fsolve_kwargs)

        residual = np.max(np.abs(info["fvec"])) if len(info["fvec"]) > 0 else 0.0

        result = []
        for i in range(K):
            vec = sol[vars_per * i : vars_per * (i + 1)].copy()
            vec[0] = np.clip(vec[0], min_gap, T - min_gap)
            result.append(_Contact.from_vector(vec, c_pairs[i], other_idx_list[i]))
        result.sort(key=lambda c: c.t_star)

        for i in range(1, len(result)):
            if result[i].t_star - result[i - 1].t_star < min_gap:
                result[i].t_star = result[i - 1].t_star + min_gap

        return result, residual

    def _sample(self, segments, c_coeffs, n=1000):
        times = np.linspace(0, self.T, n)
        positions = [np.zeros((n, 2)) for _ in range(self.N)]
        velocities = [np.zeros((n, 2)) for _ in range(self.N)]
        accelerations = [np.zeros((n, 2)) for _ in range(self.N)]

        for k, t in enumerate(times):
            for seg_idx, (_, t_s, t_e) in enumerate(segments[0]):
                if t_s <= t <= t_e + 1e-10:
                    t_loc = t - t_s
                    r_vals = {
                        i: eval_cubic(segments[i][seg_idx][0], t_loc) for i in range(self.N - 1)
                    }
                    v_rels = {
                        i: eval_cubic_vel(segments[i][seg_idx][0], t_loc) for i in range(self.N - 1)
                    }
                    a_rels = {
                        i: eval_cubic_acc(segments[i][seg_idx][0], t_loc) for i in range(self.N - 1)
                    }

                    c = eval_cubic(c_coeffs, t)
                    vc = eval_cubic_vel(c_coeffs, t)
                    ac = eval_cubic_acc(c_coeffs, t)

                    sum_r = sum(r_vals.values()) if r_vals else np.zeros(2)
                    sum_v = sum(v_rels.values()) if v_rels else np.zeros(2)
                    sum_a = sum(a_rels.values()) if a_rels else np.zeros(2)

                    positions[0][k] = c + sum_r / self.N
                    velocities[0][k] = vc + sum_v / self.N
                    accelerations[0][k] = ac + sum_a / self.N

                    for j in range(1, self.N):
                        positions[j][k] = positions[0][k] - r_vals[j - 1]
                        velocities[j][k] = velocities[0][k] - v_rels[j - 1]
                        accelerations[j][k] = accelerations[0][k] - a_rels[j - 1]
                    break

        return times, positions, velocities, accelerations

    def _compute_min_dist(self, positions):
        min_d = np.inf
        for k in range(len(positions[0])):
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    d = np.linalg.norm(positions[i][k] - positions[j][k])
                    min_d = min(min_d, d)
        return min_d

    def _compute_min_obstacle_dist(self, positions):
        if self.obstacle_positions is None or len(self.obstacle_positions) == 0:
            return np.inf

        min_d = np.inf
        for k in range(len(positions[0])):
            for i in range(self.N):
                for obs in self.obstacle_positions:
                    d = np.linalg.norm(positions[i][k] - obs)
                    min_d = min(min_d, d)
        return min_d
