"""Prioritized SCP trajectory solver.

Solves an N-robot trajectory problem by prioritized planning: robots are
solved one at a time in conflict-severity order, and each robot treats the
previously committed grid-sampled trajectories as time-varying point
obstacles. The single-robot subproblem is a discrete-time SCP with
acceleration-only decision variables (same layout as the joint SCP solver
in this codebase) plus a set of (i-1) * K linearized obstacle constraints.

This is the SCP counterpart to PrioritizedSCI. The key cost difference is
that the obstacle constraint count grows as O(K * i) at rank i -- every prior
trajectory contributes one linearized constraint per grid point per SCP
iteration. PrioritizedSCI sidesteps this since each contact contributes
3 unknowns regardless of K.

Row-normalization of the linearized obstacle constraints is controlled by
`config.solver.prio_scp_normalize_obstacles` (see SolverConfig for the
rationale; the prioritized sweep config sets it to False).
"""

import time
from collections import Counter

import numpy as np
import scipy.sparse as sp

from ..collision import detect_collisions
from ..config import Config
from .base import Solver
from .constraints import Constraint, build_dynamics_constraints, normalize_constraint
from .osqp_utils import OSQPSettings, solve_qp
from .prioritized_utils import conflict_severity_order


class PrioritizedSCP(Solver):
    """Sequential convex programming with prioritized planning."""

    def __init__(self, config: Config, verbose: bool = False):
        super().__init__(config)
        self.verbose = verbose
        self.normalize_obstacle_constraints = bool(
            config.solver.prio_scp_normalize_obstacles
        )
        self.scp_convergence_rel = config.solver.scp_convergence_rel
        self.scp_convergence_abs = config.solver.scp_convergence_abs
        self.scp_max_iterations = config.solver.scp_max_iterations
        self.detection_tol = config.solver.scp_detection_tol
        self.osqp_settings = OSQPSettings.from_config(config)

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
            print(f"PrioritizedSCP: N={self.N}, K={self.K}, T={self.T:.3f}, R={self.R:.3f}m")
            print(f"  priority order (conflict severity): {order}")

        committed_positions: dict[int, np.ndarray] = {}
        committed_velocities: dict[int, np.ndarray] = {}
        committed_accelerations: dict[int, np.ndarray] = {}

        scp_iters_per_rank: list[int] = []
        time_per_rank: list[float] = []
        rank_reasons: list[str] = []

        for step, robot_i in enumerate(order):
            t_rank_start = time.perf_counter()

            obstacle_trajs = (
                np.stack([committed_positions[j] for j in committed_positions], axis=0)
                if committed_positions
                else None
            )

            pos_i, vel_i, acc_i, n_iters, reason = self._solve_rank(
                p0[robot_i], v0[robot_i], pf[robot_i], vf[robot_i],
                obstacle_trajs,
            )
            committed_positions[robot_i] = pos_i
            committed_velocities[robot_i] = vel_i
            committed_accelerations[robot_i] = acc_i
            scp_iters_per_rank.append(n_iters)
            time_per_rank.append(time.perf_counter() - t_rank_start)
            rank_reasons.append(reason)

            if verbose:
                print(
                    f"  rank {step + 1}/{self.N} (robot {robot_i}): "
                    f"{n_iters} SCP iters, {time_per_rank[-1]:.3f}s, {reason}"
                )

        positions = [committed_positions[i] for i in range(self.N)]
        velocities = [committed_velocities[i] for i in range(self.N)]
        accelerations = [committed_accelerations[i] for i in range(self.N)]

        # Joint feasibility check (grid-discrete)
        report = detect_collisions(positions, self.R, detection_tol=self.detection_tol)
        all_ranks_converged = all(r == "converged" for r in rank_reasons)
        joint_satisfied = report.all_satisfied
        converged = joint_satisfied and all_ranks_converged

        # Structured convergence reason
        if converged:
            reason = "converged"
        elif not all_ranks_converged:
            bad = [r for r in rank_reasons if r != "converged"]
            dominant = Counter(bad).most_common(1)[0][0]
            reason = f"rank_{dominant}"
        else:
            reason = "joint_violation_after_ranks_converged"

        t_wall_total = time.perf_counter() - t_wall_start

        metrics = {
            "timing": {
                "total_time": t_wall_total,
                "wall_time": t_wall_total,
                "per_rank_time": time_per_rank,
            },
            "converged": converged,
            "convergence_reason": reason,
            "all_ranks_converged": all_ranks_converged,
            "joint_satisfied": joint_satisfied,
            "n_joint_violations": report.n_robot_robot_violations,
            "worst_joint_violation": report.worst_violation,
            "scp_iterations_total": sum(scp_iters_per_rank),
            "scp_iterations_per_rank": scp_iters_per_rank,
            "rank_reasons": rank_reasons,
            "min_distance": _min_pairwise_distance(positions),
            "priority_order": order,
        }

        return {
            "trajectories": {
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
            },
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    #  Single-rank SCP: 1 robot vs (i-1) time-varying point obstacles
    # ------------------------------------------------------------------

    def _solve_rank(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        pf: np.ndarray,
        vf: np.ndarray,
        obstacle_trajs: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
        """Solve a single-robot SCP subproblem.

        Returns (positions, velocities, accelerations, n_scp_iters, reason).
        """
        p0_arr = p0.reshape(1, 2)
        v0_arr = v0.reshape(1, 2)
        pf_arr = pf.reshape(1, 2)
        vf_arr = vf.reshape(1, 2)

        dynamics = build_dynamics_constraints(
            K_i=[self.K],
            h=self.h,
            initial_states={"position": p0_arr, "velocity": v0_arr},
            final_states={"position": pf_arr, "velocity": vf_arr},
            box_limits={
                "jerk_min": self.jerk_min, "jerk_max": self.jerk_max,
                "acc_min": self.acc_min, "acc_max": self.acc_max,
                "vel_min": self.vel_min, "vel_max": self.vel_max,
                "pos_min": self.pos_min, "pos_max": self.pos_max,
            },
            dims=2,
        )
        dyn_list = self._dynamics_list(dynamics)

        total_vars = 2 * self.K
        P = sp.eye(total_vars, format="csc") * 2.0
        q = np.zeros(total_vars)

        # Initial unconstrained solve
        try:
            accel_flat, _ = solve_qp(P, q, dyn_list, settings=self.osqp_settings)
        except RuntimeError:
            zero = np.zeros((self.K, 2))
            return zero, zero, zero, 0, "osqp_failed"

        pos, vel = self._integrate(accel_flat, p0, v0)

        if obstacle_trajs is None or obstacle_trajs.shape[0] == 0:
            return pos, vel, accel_flat.reshape(self.K, 2), 0, "converged"

        # Iterate SCP
        prev_pos = pos
        prev_accel = accel_flat
        for it in range(self.scp_max_iterations):
            obs_constr = self._build_obstacle_constraints(prev_pos, obstacle_trajs, p0, v0)
            constraints = dyn_list + [obs_constr]

            try:
                accel_flat, _ = solve_qp(
                    P, q, constraints, warm_start=prev_accel, settings=self.osqp_settings
                )
            except RuntimeError:
                return prev_pos, vel, prev_accel.reshape(self.K, 2), it, "osqp_failed"

            pos, vel = self._integrate(accel_flat, p0, v0)

            abs_change = float(np.abs(pos - prev_pos).max())
            norm_prev = float(np.linalg.norm(prev_pos))
            rel_change = float(np.linalg.norm(pos - prev_pos)) / max(norm_prev, 1e-10)

            iterate_feasible = self._is_feasible(pos, obstacle_trajs)
            iterate_stabilized = (
                abs_change <= self.scp_convergence_abs
                and rel_change <= max(self.scp_convergence_rel, 1e-12)
            )

            prev_pos = pos
            prev_accel = accel_flat

            if iterate_stabilized and iterate_feasible:
                return pos, vel, accel_flat.reshape(self.K, 2), it + 1, "converged"

        return pos, vel, accel_flat.reshape(self.K, 2), self.scp_max_iterations, "max_iterations"

    # ------------------------------------------------------------------
    #  Linearized time-varying obstacle constraints (single robot)
    # ------------------------------------------------------------------

    def _build_obstacle_constraints(
        self,
        prev_pos: np.ndarray,
        obstacle_trajs: np.ndarray,
        p0: np.ndarray,
        v0: np.ndarray,
    ) -> Constraint:
        """One linearized constraint per (obstacle, timestep) pair.

        Row-normalized iff `self.normalize_obstacle_constraints` is True.
        """
        M = obstacle_trajs.shape[0]
        K = self.K
        h = self.h
        total_vars = 2 * K

        rows, cols, vals = [], [], []
        rhs_list = []
        row_idx = 0

        for m in range(M):
            obs_pos_traj = obstacle_trajs[m]
            for k in range(K):
                diff = prev_pos[k] - obs_pos_traj[k]
                d = float(np.linalg.norm(diff))
                if d < 1e-6:
                    angle = np.random.uniform(0, 2 * np.pi)
                    eta = np.array([np.cos(angle), np.sin(angle)])
                else:
                    eta = diff / d

                for mp in range(k + 1):
                    weight = h * h * (k + 1 - mp - 0.5)
                    rows.append(row_idx)
                    cols.append(2 * mp)
                    vals.append(eta[0] * weight)
                    rows.append(row_idx)
                    cols.append(2 * mp + 1)
                    vals.append(eta[1] * weight)

                init_pos_contrib = float(eta @ (p0 - obs_pos_traj[k]))
                init_vel_contrib = float(eta @ v0) * (k + 1) * h
                linearization_term = float(eta @ (prev_pos[k] - obs_pos_traj[k])) - d
                rhs_list.append(
                    self.R + linearization_term - init_pos_contrib - init_vel_contrib
                )
                row_idx += 1

        if row_idx == 0:
            return Constraint(sp.csc_matrix((0, total_vars)), np.array([]), np.array([]))

        matrix = sp.coo_matrix(
            (vals, (rows, cols)), shape=(row_idx, total_vars)
        ).tocsc()
        constraint = Constraint(matrix, np.array(rhs_list), np.full(row_idx, np.inf))

        if self.normalize_obstacle_constraints:
            return normalize_constraint(constraint)
        return constraint

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _dynamics_list(self, dynamics: dict[str, Constraint]) -> list[Constraint]:
        keys = []
        if "jerk" in dynamics:
            keys.append("jerk")
        if "acceleration" in dynamics:
            keys.append("acceleration")
        keys.extend(["velocity", "position"])
        return [dynamics[k] for k in keys]

    def _integrate(
        self, accel_flat: np.ndarray, p0: np.ndarray, v0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = self.h
        K = self.K
        a = accel_flat.reshape(K, 2)
        vel = v0 + h * np.cumsum(a, axis=0)
        k_idx = np.arange(K)
        cumsum_a = np.cumsum(a, axis=0)
        weighted_a = (k_idx[:, None] + 0.5) * a
        cumsum_weighted_a = np.cumsum(weighted_a, axis=0)
        pos = (
            p0
            + (k_idx + 1)[:, None] * h * v0
            + h * h * ((k_idx[:, None] + 1) * cumsum_a - cumsum_weighted_a)
        )
        return pos, vel

    def _is_feasible(self, pos: np.ndarray, obstacle_trajs: np.ndarray) -> bool:
        diffs = pos[None, :, :] - obstacle_trajs
        dists = np.linalg.norm(diffs, axis=2)
        return bool(np.all(dists >= self.R - self.detection_tol))


def _min_pairwise_distance(positions: list[np.ndarray]) -> float:
    N = len(positions)
    if N < 2:
        return float("inf")
    pos = np.asarray(positions)
    ii, jj = np.triu_indices(N, k=1)
    diffs = pos[ii] - pos[jj]
    return float(np.min(np.linalg.norm(diffs, axis=2)))
