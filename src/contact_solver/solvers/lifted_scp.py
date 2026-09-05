"""Lifted SCP solver with explicit position/velocity/acceleration variables."""

import time

import numpy as np
import scipy.sparse as sp

from ..collision import detect_collisions
from ..config import Config
from .base import Solver
from .constraints import Constraint, normalize_constraint
from .osqp_utils import OSQPSettings, solve_qp


class LiftedSCP(Solver):

    def __init__(
        self,
        config: Config,
        obstacle_positions: np.ndarray | None = None,
        verbose: bool = False,
    ):
        super().__init__(config)
        self.verbose = verbose
        self.obstacle_positions = obstacle_positions

        self.scp_convergence_rel = config.solver.scp_convergence_rel
        self.scp_convergence_abs = config.solver.scp_convergence_abs
        self.scp_max_iterations = config.solver.scp_max_iterations
        self.detection_tol = config.solver.scp_detection_tol
        self.osqp_settings = OSQPSettings.from_config(config)

        self.trajectories = None

    def generate_trajectories(self, verbose: bool | None = None):
        verbose = verbose if verbose is not None else self.verbose
        max_iterations = self.scp_max_iterations
        metrics = self._init_metrics()
        t_start = time.time()

        self.vars_per_timestep = 6
        self.vars_per_robot = self.vars_per_timestep * self.K
        self.robot_var_offsets = np.arange(self.N) * self.vars_per_robot
        self.total_vars = self.N * self.vars_per_robot

        if verbose:
            print(f"LiftedSCP: {self.total_vars} variables ({self.N} robots, {self.K} timesteps)")
            if self.obstacle_positions is not None:
                print(f"  {len(self.obstacle_positions)} obstacles")

        t0 = time.time()
        self._build_dynamics_constraints()
        self._build_bound_constraints()
        metrics["timing"]["precompute_constraints_time"] = time.time() - t0

        t0 = time.time()
        x, osqp_info = self._solve_initial()
        positions, velocities, accelerations = self._extract_trajectories(x)
        metrics["timing"]["initial_trajectory_time"] = time.time() - t0
        metrics["osqp_metrics"]["initial_solve"] = osqp_info

        self._initial_guess_positions = [p.copy() for p in positions]
        self._initial_guess_velocities = [v.copy() for v in velocities]

        collisions = detect_collisions(
            positions, self.R,
            obstacle_positions=self.obstacle_positions,
            detection_tol=self.detection_tol,
        )
        # force_scp_iterations: campaign mode (phase_diagram) supplies a
        # feasible initial guess on purpose; the constrained SCP loop must
        # still run, otherwise the guess is returned unoptimized.
        if collisions.all_satisfied and not getattr(self, "force_scp_iterations", False):
            if verbose:
                print("Initial trajectory is collision-free")
            return self._finalize(
                positions, velocities, accelerations, metrics, t_start, "collision_free_initial"
            )

        t_scp = time.time()
        prev_pos = positions
        prev_vel = velocities
        prev_acc = accelerations
        _debug_log = []  # diagnostic log for debugging failures

        for iteration in range(max_iterations):
            if verbose:
                n_rr = collisions.n_robot_robot_violations
                n_ro = collisions.n_robot_obstacle_violations
                print(f"SCP Iteration {iteration + 1}: {n_rr} robot-robot, {n_ro} robot-obstacle")

            x, osqp_info = self._solve_with_collisions(prev_pos, prev_vel, prev_acc)
            osqp_info["scp_iter"] = iteration + 1
            metrics["osqp_metrics"]["iterations"].append(osqp_info)

            positions, velocities, accelerations = self._extract_trajectories(x)

            converged, rel_change, abs_change = self._check_convergence(prev_pos, positions)
            if verbose:
                print(f"  Position change - rel: {rel_change:.6f}, abs: {abs_change:.6f}")

            prev_pos = positions
            prev_vel = velocities
            prev_acc = accelerations

            collisions = detect_collisions(
                positions, self.R,
                obstacle_positions=self.obstacle_positions,
                detection_tol=self.detection_tol,
            )

            # Compute min pairwise distance for diagnostics
            min_dist = np.inf
            min_pair = None
            min_k = None
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    for k in range(self.K):
                        d = np.linalg.norm(positions[i][k] - positions[j][k])
                        if d < min_dist:
                            min_dist = d
                            min_pair = (i, j)
                            min_k = k

            _debug_log.append({
                "iteration": iteration + 1,
                "converged": bool(converged),
                "rel_change": float(rel_change),
                "abs_change": float(abs_change),
                "collision_free": bool(collisions.all_satisfied),
                "n_violations": collisions.n_violations,
                "worst_violation": float(collisions.worst_violation),
                "min_dist": float(min_dist),
                "min_pair": min_pair,
                "min_k": min_k,
                "osqp_status": osqp_info.get("status", "?"),
                "osqp_pri_res": float(osqp_info.get("pri_res", 0)),
                "osqp_dua_res": float(osqp_info.get("dua_res", 0)),
                "osqp_iter": osqp_info.get("iter", 0),
            })

            if collisions.all_satisfied and converged:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations.")
                metrics["timing"]["scp_iterations_time"] = time.time() - t_scp
                metrics["scp_iterations"] = iteration + 1
                return self._finalize(
                    positions, velocities, accelerations, metrics, t_start, "converged"
                )

        metrics["timing"]["scp_iterations_time"] = time.time() - t_scp
        metrics["scp_iterations"] = max_iterations
        metrics["_debug_log"] = _debug_log  # attach debug log to failed runs
        return self._finalize(
            positions, velocities, accelerations, metrics, t_start, "max_iterations"
        )

    def _get_var_indices(self, robot: int, timestep: int) -> dict:
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
            idx_0 = self._get_var_indices(i, 0)

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

            for k in range(1, self.K):
                idx_k = self._get_var_indices(i, k)
                idx_km1 = self._get_var_indices(i, k - 1)

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

            idx_f = self._get_var_indices(i, self.K - 1)

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
        rows, cols, vals = [], [], []
        lower, upper = [], []
        row_idx = 0

        has_vel = self.vel_min is not None or self.vel_max is not None
        has_acc = self.acc_min is not None or self.acc_max is not None

        vl = self.vel_min if self.vel_min is not None else -np.inf
        vu = self.vel_max if self.vel_max is not None else np.inf
        al = self.acc_min if self.acc_min is not None else -np.inf
        au = self.acc_max if self.acc_max is not None else np.inf

        for i in range(self.N):
            for k in range(self.K):
                idx = self._get_var_indices(i, k)

                if has_vel:
                    for key in ["vx", "vy"]:
                        rows.append(row_idx)
                        cols.append(idx[key])
                        vals.append(1.0)
                        lower.append(vl)
                        upper.append(vu)
                        row_idx += 1

                if has_acc:
                    for key in ["ax", "ay"]:
                        rows.append(row_idx)
                        cols.append(idx[key])
                        vals.append(1.0)
                        lower.append(al)
                        upper.append(au)
                        row_idx += 1

        if row_idx == 0:
            self.bound_constraint = Constraint(
                sp.csc_matrix((0, self.total_vars)), np.array([]), np.array([])
            )
            return

        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(row_idx, self.total_vars)).tocsc()
        self.bound_constraint = Constraint(matrix, np.array(lower), np.array(upper))

    def _build_cost_matrix(self) -> tuple[sp.csc_matrix, np.ndarray]:
        rows, cols, vals = [], [], []

        for i in range(self.N):
            for k in range(self.K):
                idx = self._get_var_indices(i, k)
                for key in ["ax", "ay"]:
                    rows.append(idx[key])
                    cols.append(idx[key])
                    vals.append(2.0)

        P = sp.coo_matrix((vals, (rows, cols)), shape=(self.total_vars, self.total_vars)).tocsc()
        q = np.zeros(self.total_vars)
        return P, q

    def _build_collision_constraints(self, prev_pos: list[np.ndarray]) -> Constraint:
        rows, cols, vals = [], [], []
        rhs_list = []
        row_idx = 0

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

                    idx_i = self._get_var_indices(i, k)
                    idx_j = self._get_var_indices(j, k)

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

    def _build_obstacle_constraints(self, prev_pos: list[np.ndarray]) -> Constraint:
        if self.obstacle_positions is None or len(self.obstacle_positions) == 0:
            return Constraint(sp.csc_matrix((0, self.total_vars)), np.array([]), np.array([]))

        rows, cols, vals = [], [], []
        rhs_list = []
        row_idx = 0

        for i in range(self.N):
            for k in range(self.K):
                pi_prev = prev_pos[i][k]
                idx_i = self._get_var_indices(i, k)

                for j, obs in enumerate(self.obstacle_positions):
                    diff = pi_prev - obs
                    dist = np.linalg.norm(diff)

                    if dist < 1e-6:
                        angle = np.random.uniform(0, 2 * np.pi)
                        eta = np.array([np.cos(angle), np.sin(angle)])
                    else:
                        eta = diff / dist

                    rows.append(row_idx)
                    cols.append(idx_i["px"])
                    vals.append(eta[0])

                    rows.append(row_idx)
                    cols.append(idx_i["py"])
                    vals.append(eta[1])

                    rhs = self.R + (eta @ (pi_prev - obs) - dist)
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

    def _build_warm_start(self, positions, velocities, accelerations) -> np.ndarray:
        x = np.zeros(self.total_vars)

        for i in range(self.N):
            for k in range(self.K):
                idx = self._get_var_indices(i, k)
                x[idx["px"]] = positions[i][k, 0]
                x[idx["py"]] = positions[i][k, 1]
                x[idx["vx"]] = velocities[i][k, 0]
                x[idx["vy"]] = velocities[i][k, 1]
                x[idx["ax"]] = accelerations[i][k, 0]
                x[idx["ay"]] = accelerations[i][k, 1]

        return x

    def _solve_with_collisions(self, prev_pos, prev_vel, prev_acc) -> tuple[np.ndarray, dict]:
        P, q = self._build_cost_matrix()
        coll_constr = self._build_collision_constraints(prev_pos)
        obs_constr = self._build_obstacle_constraints(prev_pos)
        constraints = [
            self.dynamics_constraint,
            self.bound_constraint,
            coll_constr,
            obs_constr,
        ]

        warm_start = self._build_warm_start(prev_pos, prev_vel, prev_acc)

        x, metrics = solve_qp(P, q, constraints, warm_start=warm_start, settings=self.osqp_settings)

        # Record the collision-row dual slice of the most recent QP.
        # Row order in the stacked system: dynamics | bounds | collision
        # (pair-major, timestep-minor) | obstacle. For the regime
        # classifiers, duals on collision rows are the discrete estimate
        # of the contact multiplier measure (up to the constant row-
        # normalization factor).
        y = metrics.get("y")
        n_dyn = self.dynamics_constraint.matrix.shape[0]
        n_bnd = self.bound_constraint.matrix.shape[0]
        n_col = coll_constr.matrix.shape[0]
        if y is not None and n_col > 0:
            n_pairs = self.N * (self.N - 1) // 2
            coll_y = y[n_dyn + n_bnd : n_dyn + n_bnd + n_col]
            # OSQP sign convention: for lower-bounded rows (l <= Ax,
            # u = +inf) the active dual is NEGATIVE. Store -y so the
            # recorded profile is the nonnegative multiplier estimate.
            self._last_collision_duals = (-coll_y).reshape(n_pairs, self.K)
            self._last_dual_quality = {
                "pri_res": metrics.get("pri_res"),
                "dua_res": metrics.get("dua_res"),
                "polish_time": metrics.get("polish_time"),
                "status": metrics.get("status"),
            }
        return x, metrics

    def get_collision_duals(self):
        """Collision-row duals of the final SCP iteration.

        Returns:
            (duals, quality): duals is an (n_pairs, K) array (pairs in
            (i, j), i < j lexicographic order), or None if no
            collision-constrained QP was solved. quality is a dict with
            the OSQP residuals of that solve (gate on `dua_res` before
            trusting the profile).
        """
        return (
            getattr(self, "_last_collision_duals", None),
            getattr(self, "_last_dual_quality", None),
        )

    def _extract_trajectories(self, x: np.ndarray):
        positions, velocities, accelerations = [], [], []

        for i in range(self.N):
            pos_i = np.zeros((self.K, 2))
            vel_i = np.zeros((self.K, 2))
            acc_i = np.zeros((self.K, 2))

            for k in range(self.K):
                idx = self._get_var_indices(i, k)
                pos_i[k] = [x[idx["px"]], x[idx["py"]]]
                vel_i[k] = [x[idx["vx"]], x[idx["vy"]]]
                acc_i[k] = [x[idx["ax"]], x[idx["ay"]]]

            positions.append(pos_i)
            velocities.append(vel_i)
            accelerations.append(acc_i)

        return positions, velocities, accelerations

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
        converged = max_abs_change <= self.scp_convergence_abs
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

        collisions = detect_collisions(
            positions, self.R,
            obstacle_positions=self.obstacle_positions,
            detection_tol=self.detection_tol,
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
