"""Contact-based trajectory optimization using sequential contact insertion."""

import time
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import fsolve, root

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


# ---------------------------------------------------------------------------
# Vectorized cubic utilities (operate on stacked arrays)
# ---------------------------------------------------------------------------

def _cubic_coefficients_batch(p0, v0, pf, vf, T):
    """Compute cubic coefficients for multiple coordinates at once.

    Args:
        p0, v0, pf, vf: (M, 2) arrays of boundary conditions.
        T: scalar segment duration.

    Returns:
        (M, 4, 2) array of coefficients [a, b, c, d] for each coordinate.
    """
    M = p0.shape[0]
    coeffs = np.empty((M, 4, 2))
    coeffs[:, 0] = p0
    coeffs[:, 1] = v0
    if T < 1e-10:
        coeffs[:, 2] = 0.0
        coeffs[:, 3] = 0.0
    else:
        coeffs[:, 2] = (3 * (pf - p0) - (2 * v0 + vf) * T) / T**2
        coeffs[:, 3] = (-2 * (pf - p0) + (v0 + vf) * T) / T**3
    return coeffs


def _eval_cubic_acc_batch(coeffs, t):
    """Evaluate acceleration for stacked coefficients: (M, 4, 2) -> (M, 2)."""
    return 2 * coeffs[:, 2] + 6 * coeffs[:, 3] * t


def _eval_cubic_jerk_batch(coeffs):
    """Evaluate (constant) jerk for stacked coefficients: (M, 4, 2) -> (M, 2)."""
    return 6 * coeffs[:, 3]


# ---------------------------------------------------------------------------
# Linear map: boundary values -> cubic coefficients
# ---------------------------------------------------------------------------

def _bv_to_coeffs_matrix(T):
    """Return 4x4 matrix M such that [a,b,c,d]^T = M @ [p0,v0,pf,vf]^T (per component)."""
    if T < 1e-10:
        return np.eye(4)
    T2 = T * T
    T3 = T2 * T
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [-3 / T2, -2 / T, 3 / T2, -1 / T],
        [2 / T3, 1 / T2, -2 / T3, 1 / T2],
    ])


def _d_bv_to_coeffs_dT(T):
    """Derivative of the bv->coeffs matrix with respect to T."""
    if T < 1e-10:
        return np.zeros((4, 4))
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    return np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [6 / T3, 2 / T2, -6 / T3, 1 / T2],
        [-6 / T4, -2 / T3, 6 / T4, -2 / T3],
    ])


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

        # Performance flags
        self.use_analytical_jacobian = config.solver.use_analytical_jacobian
        self.root_method = config.solver.root_method
        self.vectorize_residuals = config.solver.vectorize_residuals
        self.vectorize_postprocess = config.solver.vectorize_postprocess

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
        t_solve_start = time.perf_counter()
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

        t_solve_time = time.perf_counter() - t_solve_start

        t_post_start = time.perf_counter()
        segments = self._build_segments(contacts, r0, v0_rel, rf, vf_rel)
        self._segments = segments
        self._c_coeffs = c_coeffs
        times, positions, velocities, accelerations = self._sample(segments, c_coeffs)
        contact_times = [c.t_star for c in contacts]
        arc_times = []

        v_max = self.config.problem.dynamics.vel_max
        if self.apply_velocity_bounds and v_max is not None and v_max > 0:
            max_vel = max(np.max(np.abs(v)) for v in velocities)
            if max_vel > v_max + 1e-6:
                if verbose:
                    print(f"  Velocity exceeded: {max_vel:.4f} > {v_max:.4f}, inserting arcs...")
                positions, velocities, accelerations, arc_times = (
                    apply_velocity_bounds_to_trajectories(
                        times, positions, velocities, accelerations, contact_times, v_max, verbose
                    )
                )

        if self.vectorize_postprocess:
            min_dist = self._compute_min_dist_vec(positions)
            min_obs_dist = (
                self._compute_min_obstacle_dist_vec(positions)
                if self.obstacle_positions is not None
                else np.inf
            )
        else:
            min_dist = self._compute_min_dist(positions)
            min_obs_dist = (
                self._compute_min_obstacle_dist(positions)
                if self.obstacle_positions is not None
                else np.inf
            )
        self.K = len(times)

        t_post_time = time.perf_counter() - t_post_start
        t_wall_time = time.perf_counter() - t_wall_start

        max_contacts_single_pair = max(pair_cnt.values()) if pair_cnt else 0

        return {
            "trajectories": {
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
            },
            "metrics": {
                "timing": {
                    "total_time": t_solve_time,
                    "solve_time": t_solve_time,
                    "postprocess_time": t_post_time,
                    "wall_time": t_wall_time,
                },
                "converged": min_dist >= self.R - self.detection_tol and min_obs_dist >= self.R - self.detection_tol,
                "convergence_reason": reason,
                "num_contacts": len(contacts),
                "num_obstacle_contacts": len(obs_contacts),
                "max_contacts_single_pair": max_contacts_single_pair,
                "min_distance": min_dist,
                "min_obstacle_distance": min_obs_dist,
            },
            "contact_times": contact_times,
            "arc_times": arc_times,
        }

    # ===================================================================
    #  Helpers (unchanged)
    # ===================================================================

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

    # ===================================================================
    #  _solve_system: routes to baseline or optimized paths
    # ===================================================================

    def _solve_system(self, contacts, pairs, jerk_coeffs, r0, v0_rel, rf, vf_rel):
        if not contacts:
            return contacts, 0.0

        K = len(contacts)
        c_pairs = [c.pair for c in contacts]
        other_idx_list = [self._get_other_indices(p) for p in c_pairs]
        vars_per = 4 * self.N - 5

        T = self.T
        min_gap = self.min_contact_gap
        M = self.N - 1

        # Precompute jerk_coeffs as array
        jc_arr = np.zeros((K, M))
        ref_indices = np.zeros(K, dtype=int)
        for k in range(K):
            jc = jerk_coeffs[c_pairs[k]]
            for j in range(M):
                jc_arr[k, j] = jc[j]
            ref_indices[k] = next((j for j in range(M) if abs(jc_arr[k, j]) > 1e-10), 0)

        # Choose equation function
        if self.vectorize_residuals:
            equations = self._make_equations_vec(
                K, M, vars_per, c_pairs, other_idx_list,
                jc_arr, ref_indices, r0, v0_rel, rf, vf_rel, T, min_gap)
        else:
            equations = self._make_equations_baseline(
                K, M, vars_per, c_pairs, other_idx_list,
                jerk_coeffs, r0, v0_rel, rf, vf_rel, T, min_gap)

        # Choose Jacobian
        jac_fn = None
        if self.use_analytical_jacobian:
            jac_fn = self._make_jacobian(
                K, M, vars_per, c_pairs, other_idx_list,
                jc_arr, ref_indices, r0, v0_rel, rf, vf_rel, T, min_gap)

        x0 = np.concatenate([c.to_vector() for c in contacts])

        # Choose solver
        if self.root_method == "fsolve":
            fsolve_kwargs = {"full_output": True}
            if self.fsolve_xtol > 0:
                fsolve_kwargs["xtol"] = self.fsolve_xtol
            if jac_fn is not None:
                fsolve_kwargs["fprime"] = jac_fn
            sol, info, ier, _ = fsolve(equations, x0, **fsolve_kwargs)
            residual = np.max(np.abs(info["fvec"])) if len(info["fvec"]) > 0 else 0.0
        else:
            root_kwargs = {"method": self.root_method}
            if jac_fn is not None:
                root_kwargs["jac"] = jac_fn
            if self.fsolve_xtol > 0:
                root_kwargs["tol"] = self.fsolve_xtol
            res = root(equations, x0, **root_kwargs)
            sol = res.x
            residual = np.max(np.abs(res.fun)) if len(res.fun) > 0 else 0.0

        result_contacts = []
        for i in range(K):
            vec = sol[vars_per * i : vars_per * (i + 1)].copy()
            vec[0] = np.clip(vec[0], min_gap, T - min_gap)
            result_contacts.append(_Contact.from_vector(vec, c_pairs[i], other_idx_list[i]))
        result_contacts.sort(key=lambda c: c.t_star)

        for i in range(1, len(result_contacts)):
            if result_contacts[i].t_star - result_contacts[i - 1].t_star < min_gap:
                result_contacts[i].t_star = result_contacts[i - 1].t_star + min_gap

        return result_contacts, residual

    # ===================================================================
    #  Baseline equations (original implementation)
    # ===================================================================

    def _make_equations_baseline(self, K, M, vars_per, c_pairs, other_idx_list,
                                  jerk_coeffs, r0, v0_rel, rf, vf_rel, T, min_gap):
        solver = self

        def equations(x):
            c_list = []
            for i in range(K):
                vec = x[vars_per * i : vars_per * (i + 1)].copy()
                vec[0] = np.clip(vec[0], min_gap, T - min_gap)
                c_list.append(_Contact.from_vector(vec, c_pairs[i], other_idx_list[i]))
            c_list.sort(key=lambda c: c.t_star)
            segs = solver._build_segments(c_list, r0, v0_rel, rf, vf_rel)

            residuals = []
            for i, contact in enumerate(c_list):
                T_before = segs[0][i][2] - segs[0][i][1]

                for j in range(M):
                    a_bef = eval_cubic_acc(segs[j][i][0], T_before)
                    a_aft = eval_cubic_acc(segs[j][i + 1][0], 0.0)
                    residuals.extend(a_bef - a_aft)

                jerks_bef = {j: eval_cubic_jerk(segs[j][i][0]) for j in range(M)}
                jerks_aft = {j: eval_cubic_jerk(segs[j][i + 1][0]) for j in range(M)}
                delta_j = {j: jerks_aft[j] - jerks_bef[j] for j in range(M)}

                jc = jerk_coeffs[contact.pair]
                ref_idx = next((j for j, c in jc.items() if abs(c) > 1e-10), None)
                if ref_idx is None:
                    residuals.extend([0.0] * (2 * (M - 1) + 1))
                    continue

                for j in range(M):
                    if j != ref_idx:
                        ratio = jc[j] / jc[ref_idx]
                        residuals.extend(delta_j[j] - ratio * delta_j[ref_idx])

                normal = np.array([np.cos(contact.phi), np.sin(contact.phi)])
                residuals.append(delta_j[ref_idx][0] * normal[1] - delta_j[ref_idx][1] * normal[0])

            return np.array(residuals)

        return equations

    # ===================================================================
    #  Vectorized equations
    # ===================================================================

    def _make_equations_vec(self, K, M, vars_per, c_pairs, other_idx_list,
                             jc_arr, ref_indices, r0, v0_rel, rf, vf_rel, T, min_gap):
        R = self.R
        r0_arr = np.array([r0[i] for i in range(M)])
        v0_arr = np.array([v0_rel[i] for i in range(M)])
        rf_arr = np.array([rf[i] for i in range(M)])
        vf_arr = np.array([vf_rel[i] for i in range(M)])
        pair_a = np.array([c_pairs[k][0] for k in range(K)])
        pair_b = np.array([c_pairs[k][1] for k in range(K)])
        other_idx_sorted = [sorted(other_idx_list[k]) for k in range(K)]

        def equations(x):
            t_stars = np.empty(K)
            phis = np.empty(K)
            lams = np.empty(K)
            r_ol = []
            v_ol = []
            for k in range(K):
                base = vars_per * k
                t_stars[k] = np.clip(x[base], min_gap, T - min_gap)
                phis[k] = x[base + 1]
                lams[k] = x[base + 2]
                ro, vo = {}, {}
                idx = base + 3
                for i in other_idx_sorted[k]:
                    ro[i] = x[idx:idx + 2].copy()
                    vo[i] = x[idx + 2:idx + 4].copy()
                    idx += 4
                r_ol.append(ro)
                v_ol.append(vo)

            order = np.argsort(t_stars)
            t_s = t_stars[order]
            ph = phis[order]
            la = lams[order]
            ro_s = [r_ol[o] for o in order]
            vo_s = [v_ol[o] for o in order]
            spa = pair_a[order]
            spb = pair_b[order]

            states_r = np.empty((K + 2, M, 2))
            states_v = np.empty((K + 2, M, 2))
            states_r[0] = r0_arr
            states_v[0] = v0_arr
            states_r[K + 1] = rf_arr
            states_v[K + 1] = vf_arr

            for k in range(K):
                cp, sp = np.cos(ph[k]), np.sin(ph[k])
                dr = R * np.array([cp, sp])
                dv = la[k] * np.array([-sp, cp])
                a, b = spa[k], spb[k]
                if a == 0:
                    states_r[k + 1, b - 1] = dr
                    states_v[k + 1, b - 1] = dv
                    for i in ro_s[k]:
                        states_r[k + 1, i] = ro_s[k][i]
                        states_v[k + 1, i] = vo_s[k][i]
                else:
                    ra = ro_s[k][a - 1]
                    va = vo_s[k][a - 1]
                    states_r[k + 1, a - 1] = ra
                    states_v[k + 1, a - 1] = va
                    states_r[k + 1, b - 1] = ra + dr
                    states_v[k + 1, b - 1] = va + dv
                    for i in ro_s[k]:
                        if i != a - 1:
                            states_r[k + 1, i] = ro_s[k][i]
                            states_v[k + 1, i] = vo_s[k][i]

            knot_times = np.concatenate([[0.0], t_s, [T]])
            seg_T = np.diff(knot_times)

            all_coeffs = np.empty((K + 1, M, 4, 2))
            for s in range(K + 1):
                all_coeffs[s] = _cubic_coefficients_batch(
                    states_r[s], states_v[s], states_r[s + 1], states_v[s + 1], seg_T[s])

            residuals = np.empty(K * vars_per)
            sjc = jc_arr[order]
            sref = ref_indices[order]

            for k in range(K):
                Tb = seg_T[k]
                acc_bef = _eval_cubic_acc_batch(all_coeffs[k], Tb)
                acc_aft = _eval_cubic_acc_batch(all_coeffs[k + 1], 0.0)
                base = k * vars_per
                residuals[base:base + 2 * M] = (acc_bef - acc_aft).ravel()

                jerk_bef = _eval_cubic_jerk_batch(all_coeffs[k])
                jerk_aft = _eval_cubic_jerk_batch(all_coeffs[k + 1])
                dj = jerk_aft - jerk_bef

                ref = sref[k]
                jck = sjc[k]
                offset = base + 2 * M

                if abs(jck[ref]) < 1e-10:
                    residuals[offset:offset + 2 * (M - 1) + 1] = 0.0
                else:
                    idx = offset
                    for j in range(M):
                        if j != ref:
                            ratio = jck[j] / jck[ref]
                            residuals[idx:idx + 2] = dj[j] - ratio * dj[ref]
                            idx += 2
                    n_vec = np.array([np.cos(ph[k]), np.sin(ph[k])])
                    residuals[idx] = dj[ref, 0] * n_vec[1] - dj[ref, 1] * n_vec[0]

            return residuals

        return equations

    # ===================================================================
    #  Analytical Jacobian (block-tridiagonal, fully analytical)
    # ===================================================================

    def _make_jacobian(self, K, M, vars_per, c_pairs, other_idx_list,
                        jc_arr, ref_indices, r0, v0_rel, rf, vf_rel, T, min_gap):
        R = self.R
        r0_arr = np.array([r0[i] for i in range(M)])
        v0_arr = np.array([v0_rel[i] for i in range(M)])
        rf_arr = np.array([rf[i] for i in range(M)])
        vf_arr = np.array([vf_rel[i] for i in range(M)])
        pair_a_arr = np.array([c_pairs[k][0] for k in range(K)])
        pair_b_arr = np.array([c_pairs[k][1] for k in range(K)])
        other_idx_sorted = [sorted(other_idx_list[k]) for k in range(K)]

        def jacobian(x):
            n_vars = K * vars_per
            J = np.zeros((n_vars, n_vars))

            t_stars_raw = np.empty(K)
            phis_raw = np.empty(K)
            lams_raw = np.empty(K)
            r_ol = []
            v_ol = []
            for k in range(K):
                base = vars_per * k
                t_stars_raw[k] = np.clip(x[base], min_gap, T - min_gap)
                phis_raw[k] = x[base + 1]
                lams_raw[k] = x[base + 2]
                ro, vo = {}, {}
                idx = base + 3
                for i in other_idx_sorted[k]:
                    ro[i] = x[idx:idx + 2].copy()
                    vo[i] = x[idx + 2:idx + 4].copy()
                    idx += 4
                r_ol.append(ro)
                v_ol.append(vo)

            order = np.argsort(t_stars_raw)
            t_stars = t_stars_raw[order]
            phis = phis_raw[order]
            lams = lams_raw[order]
            ros = [r_ol[o] for o in order]
            vos = [v_ol[o] for o in order]
            spa = pair_a_arr[order]
            spb = pair_b_arr[order]
            sjc = jc_arr[order]
            sref = ref_indices[order]

            states_r = np.empty((K + 2, M, 2))
            states_v = np.empty((K + 2, M, 2))
            states_r[0] = r0_arr
            states_v[0] = v0_arr
            states_r[K + 1] = rf_arr
            states_v[K + 1] = vf_arr

            for k in range(K):
                cpk, spk = np.cos(phis[k]), np.sin(phis[k])
                dr = R * np.array([cpk, spk])
                dv = lams[k] * np.array([-spk, cpk])
                a, b = spa[k], spb[k]
                if a == 0:
                    states_r[k + 1, b - 1] = dr
                    states_v[k + 1, b - 1] = dv
                    for i in ros[k]:
                        states_r[k + 1, i] = ros[k][i]
                        states_v[k + 1, i] = vos[k][i]
                else:
                    ra = ros[k][a - 1]
                    va = vos[k][a - 1]
                    states_r[k + 1, a - 1] = ra
                    states_v[k + 1, a - 1] = va
                    states_r[k + 1, b - 1] = ra + dr
                    states_v[k + 1, b - 1] = va + dv
                    for i in ros[k]:
                        if i != a - 1:
                            states_r[k + 1, i] = ros[k][i]
                            states_v[k + 1, i] = vos[k][i]

            knot_times = np.concatenate([[0.0], t_stars, [T]])
            seg_T = np.diff(knot_times)

            all_coeffs = np.empty((K + 1, M, 4, 2))
            Mlist = []
            dMlist = []
            for s in range(K + 1):
                Ts = seg_T[s]
                Mlist.append(_bv_to_coeffs_matrix(Ts))
                dMlist.append(_d_bv_to_coeffs_dT(Ts))
                all_coeffs[s] = _cubic_coefficients_batch(
                    states_r[s], states_v[s], states_r[s + 1], states_v[s + 1], Ts)

            for k in range(K):
                Tb = seg_T[k]
                ref = sref[k]
                jck = sjc[k]
                res_base = k * vars_per

                Mb = Mlist[k]
                Ma = Mlist[k + 1]
                dMb = dMlist[k]
                dMa = dMlist[k + 1]

                dacc_dcc_b = np.array([0.0, 0.0, 2.0, 6.0 * Tb])
                dacc_dcc_a = np.array([0.0, 0.0, 2.0, 0.0])

                for n in [k - 1, k, k + 1]:
                    if n < 0 or n >= K:
                        continue

                    orig_n = order[n]
                    var_base_n = orig_n * vars_per
                    phi_n = phis[n]
                    lam_n = lams[n]
                    cpn, spn = np.cos(phi_n), np.sin(phi_n)
                    a_n = spa[n]
                    b_n = spb[n]
                    knot_n = n + 1

                    for j in range(M):
                        row = res_base + 2 * j
                        d_dt_n = np.zeros(2)

                        if n == k:
                            bv_b = np.array([states_r[k, j], states_v[k, j],
                                             states_r[k + 1, j], states_v[k + 1, j]])
                            bv_a = np.array([states_r[k + 1, j], states_v[k + 1, j],
                                             states_r[k + 2, j], states_v[k + 2, j]])
                            jerk_b = 6.0 * all_coeffs[k, j, 3]
                            d_dt_n += jerk_b + np.einsum('i,ij->j', dacc_dcc_b, dMb @ bv_b)
                            d_dt_n -= -np.einsum('i,ij->j', dacc_dcc_a, dMa @ bv_a)
                        elif n == k - 1:
                            bv_b = np.array([states_r[k, j], states_v[k, j],
                                             states_r[k + 1, j], states_v[k + 1, j]])
                            jerk_b = 6.0 * all_coeffs[k, j, 3]
                            d_dt_n += -(jerk_b + np.einsum('i,ij->j', dacc_dcc_b, dMb @ bv_b))
                        elif n == k + 1:
                            bv_a = np.array([states_r[k + 1, j], states_v[k + 1, j],
                                             states_r[k + 2, j], states_v[k + 2, j]])
                            d_dt_n += -np.einsum('i,ij->j', dacc_dcc_a, dMa @ bv_a)

                        J[row:row + 2, var_base_n] += d_dt_n

                        d_res_d_sr = 0.0
                        d_res_d_sv = 0.0
                        if knot_n == k:
                            d_res_d_sr = dacc_dcc_b @ Mb[:, 0]
                            d_res_d_sv = dacc_dcc_b @ Mb[:, 1]
                        elif knot_n == k + 1:
                            d_res_d_sr = dacc_dcc_b @ Mb[:, 2] - dacc_dcc_a @ Ma[:, 0]
                            d_res_d_sv = dacc_dcc_b @ Mb[:, 3] - dacc_dcc_a @ Ma[:, 1]
                        elif knot_n == k + 2:
                            d_res_d_sr = -dacc_dcc_a @ Ma[:, 2]
                            d_res_d_sv = -dacc_dcc_a @ Ma[:, 3]

                        if d_res_d_sr != 0.0 or d_res_d_sv != 0.0:
                            self._fill_bv_jac(J, row, var_base_n, d_res_d_sr, d_res_d_sv,
                                              j, orig_n, a_n, b_n, lam_n,
                                              other_idx_sorted, R, spn, cpn, diagonal=True)

                    if abs(jck[ref]) < 1e-10:
                        continue

                    djerk_dbv_b = 6.0 * Mb[3, :]
                    djerk_dbv_a = 6.0 * Ma[3, :]
                    dMb_r3 = 6.0 * dMb[3, :]
                    dMa_r3 = 6.0 * dMa[3, :]

                    d_dj_d_sr_n = 0.0
                    d_dj_d_sv_n = 0.0
                    if knot_n == k:
                        d_dj_d_sr_n = -djerk_dbv_b[0]
                        d_dj_d_sv_n = -djerk_dbv_b[1]
                    elif knot_n == k + 1:
                        d_dj_d_sr_n = -djerk_dbv_b[2] + djerk_dbv_a[0]
                        d_dj_d_sv_n = -djerk_dbv_b[3] + djerk_dbv_a[1]
                    elif knot_n == k + 2:
                        d_dj_d_sr_n = djerk_dbv_a[2]
                        d_dj_d_sv_n = djerk_dbv_a[3]

                    offset = res_base + 2 * M
                    res_idx = offset

                    for j_res in range(M):
                        if j_res == ref:
                            continue
                        ratio = jck[j_res] / jck[ref]

                        for j_dep, sign in [(j_res, 1.0), (ref, -ratio)]:
                            bv_b_j = np.array([states_r[k, j_dep], states_v[k, j_dep],
                                               states_r[k + 1, j_dep], states_v[k + 1, j_dep]])
                            bv_a_j = np.array([states_r[k + 1, j_dep], states_v[k + 1, j_dep],
                                               states_r[k + 2, j_dep], states_v[k + 2, j_dep]])

                            d_dj_dt_n = np.zeros(2)
                            if n == k:
                                d_dj_dt_n = -(dMb_r3 @ bv_b_j) - (dMa_r3 @ bv_a_j)
                            elif n == k - 1:
                                d_dj_dt_n = (dMb_r3 @ bv_b_j)
                            elif n == k + 1:
                                d_dj_dt_n = (dMa_r3 @ bv_a_j)

                            J[res_idx:res_idx + 2, var_base_n] += sign * d_dj_dt_n

                            if d_dj_d_sr_n != 0.0 or d_dj_d_sv_n != 0.0:
                                self._fill_bv_jac(J, res_idx, var_base_n,
                                                  sign * d_dj_d_sr_n, sign * d_dj_d_sv_n,
                                                  j_dep, orig_n, a_n, b_n, lam_n,
                                                  other_idx_sorted, R, spn, cpn, diagonal=True)

                        res_idx += 2

                    dj_all = _eval_cubic_jerk_batch(all_coeffs[k + 1]) - _eval_cubic_jerk_batch(all_coeffs[k])
                    dj_ref = dj_all[ref]
                    cpk, spk = np.cos(phis[k]), np.sin(phis[k])

                    if n == k:
                        J[res_idx, var_base_n + 1] += dj_ref[0] * cpk + dj_ref[1] * spk

                    bv_b_ref = np.array([states_r[k, ref], states_v[k, ref],
                                         states_r[k + 1, ref], states_v[k + 1, ref]])
                    bv_a_ref = np.array([states_r[k + 1, ref], states_v[k + 1, ref],
                                         states_r[k + 2, ref], states_v[k + 2, ref]])
                    d_dj_ref_dt_n = np.zeros(2)
                    if n == k:
                        d_dj_ref_dt_n = -(dMb_r3 @ bv_b_ref) - (dMa_r3 @ bv_a_ref)
                    elif n == k - 1:
                        d_dj_ref_dt_n = (dMb_r3 @ bv_b_ref)
                    elif n == k + 1:
                        d_dj_ref_dt_n = (dMa_r3 @ bv_a_ref)

                    J[res_idx, var_base_n] += d_dj_ref_dt_n[0] * spk - d_dj_ref_dt_n[1] * cpk

                    if d_dj_d_sr_n != 0.0 or d_dj_d_sv_n != 0.0:
                        d_res_n_d_sr = np.array([d_dj_d_sr_n * spk, -d_dj_d_sr_n * cpk])
                        d_res_n_d_sv = np.array([d_dj_d_sv_n * spk, -d_dj_d_sv_n * cpk])
                        self._fill_bv_jac_normal(J, res_idx, var_base_n,
                                                  d_res_n_d_sr, d_res_n_d_sv,
                                                  ref, orig_n, a_n, b_n, lam_n,
                                                  other_idx_sorted, R, spn, cpn)

            return J

        return jacobian

    def _fill_bv_jac(self, J, row, var_base, d_res_d_sr, d_res_d_sv,
                      j, orig_k, a_r, b_r, lam, other_idx_sorted, R, sp, cp,
                      diagonal=True):
        oi = other_idx_sorted[orig_k]
        if a_r == 0:
            if j == b_r - 1:
                d_sr_dphi = R * np.array([-sp, cp])
                d_sv_dphi = lam * np.array([-cp, -sp])
                d_sv_dlam = np.array([-sp, cp])
                J[row:row + 2, var_base + 1] += d_res_d_sr * d_sr_dphi + d_res_d_sv * d_sv_dphi
                J[row:row + 2, var_base + 2] += d_res_d_sv * d_sv_dlam
            else:
                if j in oi:
                    pos = oi.index(j)
                    col_r = var_base + 3 + pos * 4
                    col_v = col_r + 2
                    J[row, col_r] += d_res_d_sr
                    J[row + 1, col_r + 1] += d_res_d_sr
                    J[row, col_v] += d_res_d_sv
                    J[row + 1, col_v + 1] += d_res_d_sv
        else:
            if j == a_r - 1:
                if (a_r - 1) in oi:
                    pos = oi.index(a_r - 1)
                    col_r = var_base + 3 + pos * 4
                    col_v = col_r + 2
                    J[row, col_r] += d_res_d_sr
                    J[row + 1, col_r + 1] += d_res_d_sr
                    J[row, col_v] += d_res_d_sv
                    J[row + 1, col_v + 1] += d_res_d_sv
            elif j == b_r - 1:
                d_sr_dphi = R * np.array([-sp, cp])
                d_sv_dphi = lam * np.array([-cp, -sp])
                d_sv_dlam = np.array([-sp, cp])
                J[row:row + 2, var_base + 1] += d_res_d_sr * d_sr_dphi + d_res_d_sv * d_sv_dphi
                J[row:row + 2, var_base + 2] += d_res_d_sv * d_sv_dlam
                if (a_r - 1) in oi:
                    pos = oi.index(a_r - 1)
                    col_r = var_base + 3 + pos * 4
                    col_v = col_r + 2
                    J[row, col_r] += d_res_d_sr
                    J[row + 1, col_r + 1] += d_res_d_sr
                    J[row, col_v] += d_res_d_sv
                    J[row + 1, col_v + 1] += d_res_d_sv
            else:
                if j in oi:
                    pos = oi.index(j)
                    col_r = var_base + 3 + pos * 4
                    col_v = col_r + 2
                    J[row, col_r] += d_res_d_sr
                    J[row + 1, col_r + 1] += d_res_d_sr
                    J[row, col_v] += d_res_d_sv
                    J[row + 1, col_v + 1] += d_res_d_sv

    def _fill_bv_jac_normal(self, J, row, var_base, d_res_d_sr, d_res_d_sv,
                             ref, orig_k, a_r, b_r, lam, other_idx_sorted, R, sp, cp):
        oi = other_idx_sorted[orig_k]
        if a_r == 0:
            if ref == b_r - 1:
                d_sr_dphi = R * np.array([-sp, cp])
                d_sv_dphi = lam * np.array([-cp, -sp])
                d_sv_dlam = np.array([-sp, cp])
                J[row, var_base + 1] += np.dot(d_res_d_sr, d_sr_dphi) + np.dot(d_res_d_sv, d_sv_dphi)
                J[row, var_base + 2] += np.dot(d_res_d_sv, d_sv_dlam)
            else:
                if ref in oi:
                    pos = oi.index(ref)
                    col_r = var_base + 3 + pos * 4
                    col_v = col_r + 2
                    J[row, col_r:col_r + 2] += d_res_d_sr
                    J[row, col_v:col_v + 2] += d_res_d_sv
        else:
            if ref == a_r - 1:
                if (a_r - 1) in oi:
                    pos = oi.index(a_r - 1)
                    col_r = var_base + 3 + pos * 4
                    col_v = col_r + 2
                    J[row, col_r:col_r + 2] += d_res_d_sr
                    J[row, col_v:col_v + 2] += d_res_d_sv
            elif ref == b_r - 1:
                d_sr_dphi = R * np.array([-sp, cp])
                d_sv_dphi = lam * np.array([-cp, -sp])
                d_sv_dlam = np.array([-sp, cp])
                J[row, var_base + 1] += np.dot(d_res_d_sr, d_sr_dphi) + np.dot(d_res_d_sv, d_sv_dphi)
                J[row, var_base + 2] += np.dot(d_res_d_sv, d_sv_dlam)
                if (a_r - 1) in oi:
                    pos = oi.index(a_r - 1)
                    col_r = var_base + 3 + pos * 4
                    col_v = col_r + 2
                    J[row, col_r:col_r + 2] += d_res_d_sr
                    J[row, col_v:col_v + 2] += d_res_d_sv
            else:
                if ref in oi:
                    pos = oi.index(ref)
                    col_r = var_base + 3 + pos * 4
                    col_v = col_r + 2
                    J[row, col_r:col_r + 2] += d_res_d_sr
                    J[row, col_v:col_v + 2] += d_res_d_sv

    # ===================================================================
    #  Sampling (unchanged)
    # ===================================================================

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

    def _sample_at(self, segments, c_coeffs, times):
        """Evaluate trajectories at explicit time points."""
        n = len(times)
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

    def sample(self, n=1000, times=None):
        """Re-evaluate the continuous trajectories at given time points.

        Args:
            n: Number of uniformly spaced points (used if times is None).
            times: Explicit array of time points to evaluate at.

        Must be called after generate_trajectories(). This is a post-processing
        step and is not part of the solve time.
        """
        if times is not None:
            return self._sample_at(self._segments, self._c_coeffs, times)
        return self._sample(self._segments, self._c_coeffs, n=n)

    # ===================================================================
    #  Min distance (original + vectorized)
    # ===================================================================

    def _compute_min_dist(self, positions):
        min_d = np.inf
        for k in range(len(positions[0])):
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    d = np.linalg.norm(positions[i][k] - positions[j][k])
                    min_d = min(min_d, d)
        return min_d

    def _compute_min_dist_vec(self, positions):
        """Vectorized minimum pairwise distance computation."""
        pos = np.array(positions)
        ii, jj = np.triu_indices(pos.shape[0], k=1)
        diffs = pos[ii] - pos[jj]
        dists = np.linalg.norm(diffs, axis=2)
        return float(np.min(dists))

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

    def _compute_min_obstacle_dist_vec(self, positions):
        """Vectorized minimum robot-obstacle distance computation."""
        if self.obstacle_positions is None or len(self.obstacle_positions) == 0:
            return np.inf
        pos = np.array(positions)
        obs = np.asarray(self.obstacle_positions)
        diffs = pos[:, :, np.newaxis, :] - obs[np.newaxis, np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=3)
        return float(np.min(dists))
