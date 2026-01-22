"""SCP solver with acceleration-only decision variables."""

import time

import numpy as np
import scipy.sparse as sp

from ..collision import detect_collisions
from ..config import Config
from .base import Solver
from .constraints import Constraint, build_dynamics_constraints, normalize_constraint
from .osqp_utils import OSQPSettings, solve_qp


class SCP(Solver):

    def __init__(
        self,
        config: Config,
        obstacle_positions: np.ndarray | None = None,
        verbose: bool = False,
    ):
        super().__init__(config)
        self.verbose = verbose
        self.obstacle_positions = obstacle_positions

        self.scp_tolerance_rel = config.solver.scp_tolerance_rel
        self.scp_tolerance_abs = config.solver.scp_tolerance_abs
        self.scp_max_iterations = config.solver.scp_max_iterations
        self.osqp_settings = OSQPSettings.from_config(config)

        self.trajectories = None

    def generate_trajectories(self, verbose: bool | None = None):
        verbose = verbose if verbose is not None else self.verbose
        max_iterations = self.scp_max_iterations
        metrics = self._init_metrics()
        t_start = time.time()

        self.robot_timeframes = [[0, self.K] for _ in range(self.N)]
        self.K_i = [self.K] * self.N
        self.decision_var_offsets = np.cumsum([0] + [2 * self.K] * (self.N - 1))
        self.total_vars = 2 * self.K * self.N

        if verbose:
            print(f"SCP: {self.total_vars} variables ({self.N} robots, {self.K} timesteps)")
            if self.obstacle_positions is not None:
                print(f"  {len(self.obstacle_positions)} obstacles")

        t0 = time.time()
        self._precompute_dynamics()
        metrics["timing"]["precompute_constraints_time"] = time.time() - t0

        t0 = time.time()
        accel, osqp_info = self._solve_initial()
        positions, velocities = self._integrate(accel)
        metrics["timing"]["initial_trajectory_time"] = time.time() - t0
        metrics["osqp_metrics"]["initial_solve"] = osqp_info

        self._initial_guess_positions = [p.copy() for p in positions]
        self._initial_guess_velocities = [v.copy() for v in velocities]

        collisions = detect_collisions(
            positions, self.R,
            obstacle_positions=self.obstacle_positions
        )
        if collisions.all_satisfied:
            if verbose:
                print("Initial trajectory is collision-free")
            return self._finalize(
                accel, positions, velocities, metrics, t_start, "collision_free_initial"
            )

        t_scp = time.time()
        prev_accel = accel
        prev_pos = positions

        for iteration in range(max_iterations):
            if verbose:
                n_rr = collisions.n_robot_robot_violations
                n_ro = collisions.n_robot_obstacle_violations
                print(f"SCP Iteration {iteration + 1}: {n_rr} robot-robot, {n_ro} robot-obstacle")

            accel, osqp_info = self._solve_with_collisions(prev_accel, prev_pos)
            osqp_info["scp_iter"] = iteration + 1
            metrics["osqp_metrics"]["iterations"].append(osqp_info)

            positions, velocities = self._integrate(accel)

            converged, rel_change, abs_change = self._check_convergence(prev_pos, positions)
            if verbose:
                print(f"  Position change - rel: {rel_change:.6f}, abs: {abs_change:.6f}")

            if converged:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations.")
                metrics["timing"]["scp_iterations_time"] = time.time() - t_scp
                metrics["scp_iterations"] = iteration + 1
                return self._finalize(
                    accel, positions, velocities, metrics, t_start, "converged"
                )

            prev_accel = accel
            prev_pos = positions
            collisions = detect_collisions(
                positions, self.R,
                obstacle_positions=self.obstacle_positions
            )

        metrics["timing"]["scp_iterations_time"] = time.time() - t_scp
        metrics["scp_iterations"] = max_iterations
        return self._finalize(
            accel, positions, velocities, metrics, t_start, "max_iterations"
        )

    def _init_metrics(self) -> dict:
        return {
            "scp_iterations": 0,
            "converged": False,
            "convergence_reason": "",
            "timing": {},
            "osqp_metrics": {"initial_solve": {}, "iterations": []},
        }

    def _precompute_dynamics(self):
        N, h = self.N, self.h
        p0 = self.initial_positions.reshape(N, 2)
        v0 = self.initial_velocities.reshape(N, 2)
        pf = self.final_positions.reshape(N, 2)
        vf = self.final_velocities.reshape(N, 2)

        self.dynamics_constraints = build_dynamics_constraints(
            K_i=self.K_i,
            h=h,
            initial_states={"position": p0, "velocity": v0},
            final_states={"position": pf, "velocity": vf},
            box_limits={
                "jerk_min": self.jerk_min,
                "jerk_max": self.jerk_max,
                "acc_min": self.acc_min,
                "acc_max": self.acc_max,
                "vel_min": self.vel_min,
                "vel_max": self.vel_max,
                "pos_min": self.pos_min,
                "pos_max": self.pos_max,
            },
            dims=2,
        )

    def _dynamics_constraint_list(self) -> list[Constraint]:
        return [self.dynamics_constraints[k] for k in ["velocity", "position"]]

    def _build_cost_matrix(self) -> tuple[sp.csc_matrix, np.ndarray]:
        P = sp.eye(self.total_vars, format="csc") * 2.0
        q = np.zeros(self.total_vars)
        return P, q

    def _solve_initial(self) -> tuple[np.ndarray, dict]:
        P, q = self._build_cost_matrix()
        x, metrics = solve_qp(P, q, self._dynamics_constraint_list(), settings=self.osqp_settings)
        return x, metrics

    def _build_collision_constraints(self, prev_pos: list[np.ndarray]) -> Constraint:
        rows, cols, vals = [], [], []
        rhs_list = []
        row_idx = 0

        h = self.h
        p0 = self.initial_positions.reshape(self.N, 2)
        v0 = self.initial_velocities.reshape(self.N, 2)

        for i in range(self.N):
            for j in range(i + 1, self.N):
                for k in range(self.K):
                    pi_prev = prev_pos[i][k]
                    pj_prev = prev_pos[j][k]

                    diff = pi_prev - pj_prev
                    dist = np.linalg.norm(diff)
                    if dist < 1e-6:
                        angle = np.random.uniform(0, 2 * np.pi)
                        eta = np.array([np.cos(angle), np.sin(angle)])
                    else:
                        eta = diff / dist

                    offset_i = self.decision_var_offsets[i]
                    offset_j = self.decision_var_offsets[j]

                    for m in range(k + 1):
                        weight = h * h * (k + 1 - m - 0.5)
                        
                        rows.append(row_idx)
                        cols.append(offset_i + 2 * m)
                        vals.append(eta[0] * weight)
                        
                        rows.append(row_idx)
                        cols.append(offset_i + 2 * m + 1)
                        vals.append(eta[1] * weight)
                        
                        rows.append(row_idx)
                        cols.append(offset_j + 2 * m)
                        vals.append(-eta[0] * weight)
                        
                        rows.append(row_idx)
                        cols.append(offset_j + 2 * m + 1)
                        vals.append(-eta[1] * weight)

                    init_pos_contrib = eta @ (p0[i] - p0[j])
                    init_vel_contrib = eta @ (v0[i] - v0[j]) * (k + 1) * h
                    linearization_term = eta @ (pi_prev - pj_prev) - dist
                    
                    rhs_list.append(self.R + linearization_term - init_pos_contrib - init_vel_contrib)
                    row_idx += 1

        if row_idx == 0:
            return Constraint(sp.csc_matrix((0, self.total_vars)), np.array([]), np.array([]))

        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, self.total_vars)).tocsc()
        constraint = Constraint(matrix, np.array(rhs_list), np.full(row_idx, np.inf))
        return normalize_constraint(constraint)

    def _build_obstacle_constraints(self, prev_pos: list[np.ndarray]) -> Constraint:
        if self.obstacle_positions is None or len(self.obstacle_positions) == 0:
            return Constraint(sp.csc_matrix((0, self.total_vars)), np.array([]), np.array([]))

        rows, cols, vals = [], [], []
        rhs_list = []
        row_idx = 0

        h = self.h
        p0 = self.initial_positions.reshape(self.N, 2)
        v0 = self.initial_velocities.reshape(self.N, 2)

        for i in range(self.N):
            for k in range(self.K):
                pi_prev = prev_pos[i][k]
                offset_i = self.decision_var_offsets[i]

                for obs in self.obstacle_positions:
                    diff = pi_prev - obs
                    dist = np.linalg.norm(diff)

                    if dist < 1e-6:
                        angle = np.random.uniform(0, 2 * np.pi)
                        eta = np.array([np.cos(angle), np.sin(angle)])
                    else:
                        eta = diff / dist

                    for m in range(k + 1):
                        weight = h * h * (k + 1 - m - 0.5)
                        
                        rows.append(row_idx)
                        cols.append(offset_i + 2 * m)
                        vals.append(eta[0] * weight)
                        
                        rows.append(row_idx)
                        cols.append(offset_i + 2 * m + 1)
                        vals.append(eta[1] * weight)

                    init_pos_contrib = eta @ (p0[i] - obs)
                    init_vel_contrib = eta @ v0[i] * (k + 1) * h
                    linearization_term = eta @ (pi_prev - obs) - dist

                    rhs_list.append(self.R + linearization_term - init_pos_contrib - init_vel_contrib)
                    row_idx += 1

        if row_idx == 0:
            return Constraint(sp.csc_matrix((0, self.total_vars)), np.array([]), np.array([]))

        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, self.total_vars)).tocsc()
        constraint = Constraint(matrix, np.array(rhs_list), np.full(row_idx, np.inf))
        return normalize_constraint(constraint)

    def _solve_with_collisions(self, prev_accel: np.ndarray, prev_pos: list[np.ndarray]) -> tuple[np.ndarray, dict]:
        P, q = self._build_cost_matrix()
        coll_constr = self._build_collision_constraints(prev_pos)
        obs_constr = self._build_obstacle_constraints(prev_pos)
        constraints = self._dynamics_constraint_list() + [coll_constr, obs_constr]

        x, metrics = solve_qp(P, q, constraints, warm_start=prev_accel, settings=self.osqp_settings)
        return x, metrics

    def _integrate(self, accel_flat: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        h = self.h
        p0 = self.initial_positions.reshape(self.N, 2)
        v0 = self.initial_velocities.reshape(self.N, 2)

        positions, velocities = [], []

        for i in range(self.N):
            offset = self.decision_var_offsets[i]
            accel_i = accel_flat[offset : offset + 2 * self.K].reshape(self.K, 2)

            vel_i = v0[i] + h * np.cumsum(accel_i, axis=0)

            k_indices = np.arange(self.K)
            pos_i = p0[i] + (k_indices + 1)[:, np.newaxis] * h * v0[i]
            cumsum_a = np.cumsum(accel_i, axis=0)
            weighted_a = (k_indices[:, np.newaxis] + 0.5) * accel_i
            cumsum_weighted_a = np.cumsum(weighted_a, axis=0)
            pos_i += h**2 * ((k_indices[:, np.newaxis] + 1) * cumsum_a - cumsum_weighted_a)

            positions.append(pos_i)
            velocities.append(vel_i)

        return positions, velocities

    def _check_convergence(self, prev_pos, positions):
        max_abs_change = 0.0
        total_norm_sq = 0.0
        total_diff_sq = 0.0

        for prev, pos in zip(prev_pos, positions, strict=True):
            diff = pos - prev
            max_abs_change = max(max_abs_change, np.abs(diff).max())
            total_diff_sq += np.sum(diff**2)
            total_norm_sq += np.sum(prev**2)

        rel_change = np.sqrt(total_diff_sq) / max(np.sqrt(total_norm_sq), 1e-10)
        converged = (rel_change <= self.scp_tolerance_rel) and (
            max_abs_change <= self.scp_tolerance_abs
        )
        return converged, rel_change, max_abs_change

    def _finalize(self, accel_flat, positions, velocities, metrics, t_start, reason):
        accelerations = [
            accel_flat[self.decision_var_offsets[i] : self.decision_var_offsets[i] + 2 * self.K].reshape(self.K, 2)
            for i in range(self.N)
        ]

        self.trajectories = {
            "positions": positions,
            "velocities": velocities,
            "accelerations": accelerations,
        }

        metrics["timing"]["total_time"] = time.time() - t_start
        metrics["timing"].setdefault("scp_iterations_time", 0.0)
        metrics["timing"]["average_iteration_time"] = metrics["timing"][
            "scp_iterations_time"
        ] / max(metrics["scp_iterations"], 1)
        metrics["converged"] = reason in ("converged", "collision_free_initial")
        metrics["convergence_reason"] = reason

        collisions = detect_collisions(
            positions, self.R,
            obstacle_positions=self.obstacle_positions
        )
        metrics["collision_check"] = {
            "all_satisfied": collisions.all_satisfied,
            "n_robot_robot_violations": collisions.n_robot_robot_violations,
            "n_robot_obstacle_violations": collisions.n_robot_obstacle_violations,
            "worst_violation": collisions.worst_violation,
        }

        if self.verbose:
            print(f"Total time: {metrics['timing']['total_time']:.3f}s")
            print(f"SCP iterations: {metrics['scp_iterations']}, Converged: {metrics['converged']}")

        return {
            "trajectories": self.trajectories,
            "initial_guess": {
                "positions": self._initial_guess_positions,
                "velocities": self._initial_guess_velocities,
            },
            "metrics": metrics,
        }
