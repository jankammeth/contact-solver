"""Lifted SCP solver with explicit position and velocity decision variables."""

import time

import numpy as np
import scipy.sparse as sp

from ..collision import detect_collisions
from ..config import Config
from .base import Solver
from .constraints import Constraint, normalize_constraint
from .osqp_utils import OSQPSettings, solve_qp


class LiftedSCP(Solver):
    """Lifted SCP solver with explicit position/velocity/acceleration variables.
    
    Variable layout per robot with K timesteps:
    Each timestep k has: p_x[k], p_y[k], v_x[k], v_y[k], a_x[k], a_y[k]
    = 6 * K variables per robot
    """

    def __init__(self, config: Config, verbose: bool = False):
        super().__init__(config)
        self.verbose = verbose

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

        # Setup timeframes (full trajectory for all robots)
        self.robot_timeframes = [[0, self.K] for _ in range(self.N)]
        self.K_i = [self.K for _ in range(self.N)]
        self.start_times = [0 for _ in range(self.N)]

        # Variable layout
        self.vars_per_timestep = 6  # px, py, vx, vy, ax, ay
        vars_per_robot = [self.vars_per_timestep * K_i for K_i in self.K_i]
        self.robot_var_offsets = (
            np.concatenate([[0], np.cumsum(vars_per_robot[:-1])])
            if len(vars_per_robot) > 1
            else np.array([0])
        )
        self.total_vars = sum(vars_per_robot)

        if verbose:
            print(f"LiftedSCP: {self.total_vars} variables ({self.N} robots, {self.K} timesteps)")

        # Build constraints
        t0 = time.time()
        self._build_dynamics_constraints()
        self._build_bound_constraints()
        metrics["timing"]["precompute_constraints_time"] = time.time() - t0

        # Initial solve (no collision constraints)
        t0 = time.time()
        x, osqp_info = self._solve_initial()
        positions, velocities, accelerations = self._extract_trajectories(x)
        metrics["timing"]["initial_trajectory_time"] = time.time() - t0
        metrics["osqp_metrics"]["initial_solve"] = osqp_info

        self._initial_guess_positions = [p.copy() for p in positions]
        self._initial_guess_velocities = [v.copy() for v in velocities]

        # Check if already collision-free
        collisions = detect_collisions(positions, self.R, self.robot_timeframes)
        if collisions.all_satisfied:
            if verbose:
                print("Initial trajectory is collision-free")
            return self._finalize(
                positions, velocities, accelerations, metrics, t_start, "collision_free_initial"
            )

        # SCP iterations
        t_scp = time.time()
        prev_positions = positions
        prev_velocities = velocities
        prev_accelerations = accelerations

        for iteration in range(max_iterations):
            if verbose:
                n_violations = collisions.n_violations
                print(f"SCP Iteration {iteration + 1}: {n_violations} collisions")

            x, osqp_info = self._solve_with_collisions(
                prev_positions, prev_velocities, prev_accelerations
            )
            osqp_info["scp_iter"] = iteration + 1
            metrics["osqp_metrics"]["iterations"].append(osqp_info)

            positions, velocities, accelerations = self._extract_trajectories(x)

            converged, rel_change, abs_change = self._check_convergence(prev_positions, positions)
            if verbose:
                print(f"  Position change - rel: {rel_change:.6f}, abs: {abs_change:.6f}")

            if converged:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations.")
                metrics["timing"]["scp_iterations_time"] = time.time() - t_scp
                metrics["scp_iterations"] = iteration + 1
                return self._finalize(
                    positions, velocities, accelerations, metrics, t_start, "converged"
                )

            prev_positions = positions
            prev_velocities = velocities
            prev_accelerations = accelerations
            collisions = detect_collisions(positions, self.R, self.robot_timeframes)

        metrics["timing"]["scp_iterations_time"] = time.time() - t_scp
        metrics["scp_iterations"] = max_iterations
        return self._finalize(
            positions, velocities, accelerations, metrics, t_start, "max_iterations"
        )

    def _get_var_indices(self, robot: int, timestep: int) -> dict:
        """Get variable indices for a robot at a timestep."""
        base = self.robot_var_offsets[robot] + timestep * self.vars_per_timestep
        return {
            "px": base,
            "py": base + 1,
            "vx": base + 2,
            "vy": base + 3,
            "ax": base + 4,
            "ay": base + 5,
        }

    def _init_metrics(self) -> dict:
        return {
            "scp_iterations": 0,
            "converged": False,
            "convergence_reason": "",
            "timing": {},
            "osqp_metrics": {"initial_solve": {}, "iterations": []},
        }

    def _build_dynamics_constraints(self):
        """Build sparse dynamics constraints."""
        h = self.h
        h2 = 0.5 * h * h

        rows, cols, vals = [], [], []
        lower, upper = [], []
        row_idx = 0

        p0 = self.initial_positions.reshape(self.N, 2)
        v0 = self.initial_velocities.reshape(self.N, 2)
        pf = self.final_positions.reshape(self.N, 2)
        vf = self.final_velocities.reshape(self.N, 2)

        for i in range(self.N):
            K_i = self.K_i[i]

            # k=0: Initial conditions
            idx_0 = self._get_var_indices(i, 0)

            # Position: p[0] - 0.5*h²*a[0] = p0 + h*v0
            for d, (p_key, a_key) in enumerate([("px", "ax"), ("py", "ay")]):
                rows.append(row_idx)
                cols.append(idx_0[p_key])
                vals.append(1.0)

                rows.append(row_idx)
                cols.append(idx_0[a_key])
                vals.append(-h2)

                rhs = p0[i, d] + h * v0[i, d]
                lower.append(rhs)
                upper.append(rhs)
                row_idx += 1

            # Velocity: v[0] - h*a[0] = v0
            for d, (v_key, a_key) in enumerate([("vx", "ax"), ("vy", "ay")]):
                rows.append(row_idx)
                cols.append(idx_0[v_key])
                vals.append(1.0)

                rows.append(row_idx)
                cols.append(idx_0[a_key])
                vals.append(-h)

                lower.append(v0[i, d])
                upper.append(v0[i, d])
                row_idx += 1

            # k=1...K-1: Dynamics
            for k in range(1, K_i):
                idx_k = self._get_var_indices(i, k)
                idx_km1 = self._get_var_indices(i, k - 1)

                # Position: p[k] - p[k-1] - h*v[k-1] - 0.5*h²*a[k] = 0
                for p_key, v_key, a_key in [("px", "vx", "ax"), ("py", "vy", "ay")]:
                    rows.append(row_idx)
                    cols.append(idx_k[p_key])
                    vals.append(1.0)

                    rows.append(row_idx)
                    cols.append(idx_km1[p_key])
                    vals.append(-1.0)

                    rows.append(row_idx)
                    cols.append(idx_km1[v_key])
                    vals.append(-h)

                    rows.append(row_idx)
                    cols.append(idx_k[a_key])
                    vals.append(-h2)

                    lower.append(0.0)
                    upper.append(0.0)
                    row_idx += 1

                # Velocity: v[k] - v[k-1] - h*a[k] = 0
                for v_key, a_key in [("vx", "ax"), ("vy", "ay")]:
                    rows.append(row_idx)
                    cols.append(idx_k[v_key])
                    vals.append(1.0)

                    rows.append(row_idx)
                    cols.append(idx_km1[v_key])
                    vals.append(-1.0)

                    rows.append(row_idx)
                    cols.append(idx_k[a_key])
                    vals.append(-h)

                    lower.append(0.0)
                    upper.append(0.0)
                    row_idx += 1

            # Final conditions
            idx_f = self._get_var_indices(i, K_i - 1)

            for d, key in enumerate(["px", "py"]):
                rows.append(row_idx)
                cols.append(idx_f[key])
                vals.append(1.0)
                lower.append(pf[i, d])
                upper.append(pf[i, d])
                row_idx += 1

            for d, key in enumerate(["vx", "vy"]):
                rows.append(row_idx)
                cols.append(idx_f[key])
                vals.append(1.0)
                lower.append(vf[i, d])
                upper.append(vf[i, d])
                row_idx += 1

        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, self.total_vars)).tocsc()
        self.dynamics_constraint = Constraint(matrix, np.array(lower), np.array(upper))

    def _build_bound_constraints(self):
        """Build box constraints on velocities and accelerations."""
        rows, cols, vals = [], [], []
        lower, upper = [], []
        row_idx = 0

        for i in range(self.N):
            K_i = self.K_i[i]
            for k in range(K_i):
                idx = self._get_var_indices(i, k)

                for key in ["vx", "vy"]:
                    rows.append(row_idx)
                    cols.append(idx[key])
                    vals.append(1.0)
                    lower.append(self.vel_min)
                    upper.append(self.vel_max)
                    row_idx += 1

                for key in ["ax", "ay"]:
                    rows.append(row_idx)
                    cols.append(idx[key])
                    vals.append(1.0)
                    lower.append(self.acc_min)
                    upper.append(self.acc_max)
                    row_idx += 1

        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, self.total_vars)).tocsc()
        self.bound_constraint = Constraint(matrix, np.array(lower), np.array(upper))

    def _build_cost_matrix(self) -> tuple[sp.csc_matrix, np.ndarray]:
        """Build cost matrix for minimizing acceleration."""
        rows, cols, vals = [], [], []

        for i in range(self.N):
            K_i = self.K_i[i]
            for k in range(K_i):
                idx = self._get_var_indices(i, k)
                for key in ["ax", "ay"]:
                    rows.append(idx[key])
                    cols.append(idx[key])
                    vals.append(2.0)

        P = sp.coo_matrix((vals, (rows, cols)), shape=(self.total_vars, self.total_vars)).tocsc()
        q = np.zeros(self.total_vars)
        return P, q

    def _build_collision_constraints(self, prev_positions: list[np.ndarray]) -> Constraint:
        """Build linearized collision constraints."""
        rows, cols, vals = [], [], []
        rhs_list = []
        row_idx = 0

        for i in range(self.N):
            for j in range(i + 1, self.N):
                start_i, end_i = self.start_times[i], self.start_times[i] + self.K_i[i]
                start_j, end_j = self.start_times[j], self.start_times[j] + self.K_i[j]
                overlap_start = max(start_i, start_j)
                overlap_end = min(end_i, end_j)

                for k_global in range(overlap_start, overlap_end):
                    k_i = k_global - self.start_times[i]
                    k_j = k_global - self.start_times[j]

                    pi_prev = prev_positions[i][k_i]
                    pj_prev = prev_positions[j][k_j]

                    diff = pi_prev - pj_prev
                    dist = np.linalg.norm(diff)
                    if dist < 1e-6:
                        angle = np.random.uniform(0, 2 * np.pi)
                        eta = np.array([np.cos(angle), np.sin(angle)])
                    else:
                        eta = diff / dist

                    idx_i = self._get_var_indices(i, k_i)
                    idx_j = self._get_var_indices(j, k_j)

                    rows.append(row_idx)
                    cols.append(idx_i["px"])
                    vals.append(eta[0])

                    rows.append(row_idx)
                    cols.append(idx_i["py"])
                    vals.append(eta[1])

                    rows.append(row_idx)
                    cols.append(idx_j["px"])
                    vals.append(-eta[0])

                    rows.append(row_idx)
                    cols.append(idx_j["py"])
                    vals.append(-eta[1])

                    rhs = self.R + (eta @ (pi_prev - pj_prev) - dist)
                    rhs_list.append(rhs)
                    row_idx += 1

        if row_idx == 0:
            return Constraint(sp.csc_matrix((0, self.total_vars)), np.array([]), np.array([]))

        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, self.total_vars)).tocsc()
        constraint = Constraint(matrix, np.array(rhs_list), np.full(row_idx, np.inf))

        return normalize_constraint(constraint)

    def _solve_initial(self) -> tuple[np.ndarray, dict]:
        P, q = self._build_cost_matrix()
        constraints = [self.dynamics_constraint, self.bound_constraint]
        x, metrics = solve_qp(P, q, constraints, settings=self.osqp_settings)
        return x, metrics

    def _build_warm_start_vector(
        self,
        positions: list[np.ndarray],
        velocities: list[np.ndarray],
        accelerations: list[np.ndarray],
    ) -> np.ndarray:
        """Build OSQP warm start vector."""
        x = np.zeros(self.total_vars)

        for i in range(self.N):
            K_i = self.K_i[i]
            for k in range(K_i):
                idx = self._get_var_indices(i, k)
                x[idx["px"]] = positions[i][k, 0]
                x[idx["py"]] = positions[i][k, 1]
                x[idx["vx"]] = velocities[i][k, 0]
                x[idx["vy"]] = velocities[i][k, 1]
                x[idx["ax"]] = accelerations[i][k, 0]
                x[idx["ay"]] = accelerations[i][k, 1]

        return x

    def _solve_with_collisions(
        self,
        prev_positions: list[np.ndarray],
        prev_velocities: list[np.ndarray],
        prev_accelerations: list[np.ndarray],
    ) -> tuple[np.ndarray, dict]:
        P, q = self._build_cost_matrix()
        collision_constraint = self._build_collision_constraints(prev_positions)
        constraints = [self.dynamics_constraint, self.bound_constraint, collision_constraint]

        warm_start = self._build_warm_start_vector(
            prev_positions, prev_velocities, prev_accelerations
        )

        x, metrics = solve_qp(P, q, constraints, warm_start=warm_start, settings=self.osqp_settings)
        return x, metrics

    def _extract_trajectories(
        self, x: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        positions, velocities, accelerations = [], [], []

        for i in range(self.N):
            K_i = self.K_i[i]
            pos_i = np.zeros((K_i, 2))
            vel_i = np.zeros((K_i, 2))
            acc_i = np.zeros((K_i, 2))

            for k in range(K_i):
                idx = self._get_var_indices(i, k)
                pos_i[k] = [x[idx["px"]], x[idx["py"]]]
                vel_i[k] = [x[idx["vx"]], x[idx["vy"]]]
                acc_i[k] = [x[idx["ax"]], x[idx["ay"]]]

            positions.append(pos_i)
            velocities.append(vel_i)
            accelerations.append(acc_i)

        return positions, velocities, accelerations

    def _check_convergence(self, prev_positions, positions):
        max_abs_change = 0.0
        total_norm_sq = 0.0
        total_diff_sq = 0.0

        for prev_pos, pos in zip(prev_positions, positions, strict=True):
            diff = pos - prev_pos
            max_abs_change = max(max_abs_change, np.abs(diff).max())
            total_diff_sq += np.sum(diff**2)
            total_norm_sq += np.sum(prev_pos**2)

        rel_change = np.sqrt(total_diff_sq) / max(np.sqrt(total_norm_sq), 1e-10)
        converged = (rel_change <= self.scp_tolerance_rel) and (
            max_abs_change <= self.scp_tolerance_abs
        )
        return converged, rel_change, max_abs_change

    def _finalize(self, positions, velocities, accelerations, metrics, t_start, reason):
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

        collisions = detect_collisions(positions, self.R, self.robot_timeframes)
        metrics["collision_check"] = {
            "all_satisfied": collisions.all_satisfied,
            "n_violations": collisions.n_violations,
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
