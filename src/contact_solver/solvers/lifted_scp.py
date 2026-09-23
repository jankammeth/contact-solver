"""Sparse joint SCP in arbitrary dimension with held-acceleration trajectories."""

import time
from itertools import combinations

import numpy as np
import scipy.sparse as sp

from ..collision import detect_collisions
from ..config import Config
from .base import Solver
from .constraints import Constraint, normalize_constraint
from .osqp_utils import OSQPSettings, solve_qp
from .zoh_trajectory import ZOHTrajectory


class LiftedSCP(Solver):
    supports_nd_states = True

    def __init__(self, config: Config, obstacle_positions=None, verbose=False):
        super().__init__(config)
        self.verbose = verbose
        self.obstacle_positions = None
        if obstacle_positions is not None:
            obstacles = np.asarray(obstacle_positions, dtype=float)
            if obstacles.size == 0:
                obstacles = np.empty((0, self.d))
            if (
                obstacles.ndim != 2
                or obstacles.shape[1] != self.d
                or not np.all(np.isfinite(obstacles))
            ):
                raise ValueError("Obstacles must be finite with shape (M, d)")
            self.obstacle_positions = obstacles.copy()
        self.scp_convergence_rel = config.solver.scp_convergence_rel
        self.scp_convergence_abs = config.solver.scp_convergence_abs
        self.scp_max_iterations = config.solver.scp_max_iterations
        self.detection_tol = config.solver.scp_detection_tol
        self.validation_tol = config.solver.scp_validation_tol
        self.osqp_settings = OSQPSettings.from_config(config)
        # The schema's zero/zero defaults are not a valid OSQP tolerance pair.
        if self.osqp_settings.eps_abs == self.osqp_settings.eps_rel == 0:
            self.osqp_settings.eps_abs = OSQPSettings().eps_abs
            self.osqp_settings.eps_rel = OSQPSettings().eps_rel
        if (
            not isinstance(self.N, int)
            or self.N < 1
            or not np.isfinite(self.h)
            or self.h <= 0
            or not np.isfinite(self.R)
            or self.R <= 0
        ):
            raise ValueError("Require integer N >= 1 and positive finite timestep/radius")
        tolerances = [
            self.scp_convergence_rel,
            self.scp_convergence_abs,
            self.detection_tol,
            self.validation_tol,
            config.solver.timeout,
            self.osqp_settings.eps_abs,
            self.osqp_settings.eps_rel,
        ]
        if not np.all(np.isfinite(tolerances)) or min(tolerances) < 0 or self.validation_tol <= 0:
            raise ValueError("Invalid solver tolerances or timeout")
        if not isinstance(self.scp_max_iterations, int) or self.scp_max_iterations < 0:
            raise ValueError("scp_max_iterations must be a nonnegative integer")
        for lower, upper in (
            (self.vel_min, self.vel_max),
            (self.acc_min, self.acc_max),
            (self.jerk_min, self.jerk_max),
        ):
            if any(
                value is not None and (np.ndim(value) != 0 or np.isnan(value))
                for value in (lower, upper)
            ):
                raise ValueError("Dynamics bounds must be scalars (applied to all d components)")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("Lower dynamics bound exceeds upper bound")
        self.trajectory = None

    def _auto_time_horizon(self):
        if self.vel_max is not None and (not np.isfinite(self.vel_max) or self.vel_max <= 0):
            raise ValueError("Automatic horizon requires a positive finite vel_max")
        return max(self.h, super()._auto_time_horizon())

    def _prepare(self):
        if self.initial_positions is None or self.final_positions is None:
            raise ValueError("Set initial and final states before solving")
        if self.T is None or not np.isfinite(self.T) or self.T <= 0:
            raise ValueError("Time horizon must be positive and finite")
        self.K = int(round(self.T / self.h))
        if self.K < 1 or not np.isclose(self.K * self.h, self.T, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "LiftedSCP requires time_horizon to be an integer multiple of timestep"
            )
        self.grid_horizon = self.K * self.h
        self.times = np.arange(1, self.K + 1) * self.h
        self.vars_per_timestep = 3 * self.d
        self.vars_per_robot = self.vars_per_timestep * self.K
        self.robot_var_offsets = np.arange(self.N) * self.vars_per_robot
        self.total_vars = self.N * self.vars_per_robot

    def generate_trajectories(self, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        started = time.perf_counter()
        self._prepare()
        self.trajectory = self.trajectories = None
        self._last_collision_duals = self._last_dual_quality = None
        self._initial_guess_positions = self._initial_guess_velocities = None
        metrics = self._init_metrics()
        deadline = started + self.config.solver.timeout if self.config.solver.timeout else np.inf
        precompute = time.perf_counter()
        self._build_dynamics_constraints()
        self._build_bound_constraints()
        metrics["timing"]["precompute_constraints_time"] = time.perf_counter() - precompute
        if verbose:
            print(f"LiftedSCP: N={self.N}, d={self.d}, K={self.K}, variables={self.total_vars}")
        initial_started = time.perf_counter()
        try:
            x, info = self._solve_initial()
        except RuntimeError as error:
            metrics["qp_error"] = str(error)
            return self._finalize(None, None, None, metrics, started, "qp_failure")
        metrics["timing"]["initial_trajectory_time"] = time.perf_counter() - initial_started
        metrics["osqp_metrics"]["initial_solve"] = info
        if not self._valid_qp_vector(x):
            metrics["qp_error"] = "Nonfinite or incorrectly sized QP solution"
            return self._finalize(None, None, None, metrics, started, "qp_failure")
        positions, velocities, accelerations = self._extract_trajectories(x)
        self._initial_guess_positions = [p.copy() for p in positions]
        self._initial_guess_velocities = [v.copy() for v in velocities]
        collisions = self._grid_collisions(positions)
        reason = "max_iterations"
        iteration_started = time.perf_counter()
        if time.perf_counter() >= deadline:
            reason = "timeout"
        elif collisions.all_satisfied and not getattr(self, "force_scp_iterations", False):
            reason = "collision_free_initial"
        else:
            for iteration in range(self.scp_max_iterations):
                if time.perf_counter() >= deadline:
                    reason = "timeout"
                    break
                try:
                    x, info = self._solve_with_collisions(positions, velocities, accelerations)
                except RuntimeError as error:
                    metrics["qp_error"] = str(error)
                    reason = "qp_failure"
                    break
                metrics["scp_iterations"] = iteration + 1
                info["scp_iter"] = iteration + 1
                metrics["osqp_metrics"]["iterations"].append(info)
                if not self._valid_qp_vector(x):
                    metrics["qp_error"] = "Nonfinite or incorrectly sized QP solution"
                    reason = "qp_failure"
                    break
                next_pos, next_vel, next_acc = self._extract_trajectories(x)
                stable, rel_change, abs_change = self._check_convergence(positions, next_pos)
                positions, velocities, accelerations = next_pos, next_vel, next_acc
                collisions = self._grid_collisions(positions)
                metrics["_debug_log"].append(
                    dict(
                        iteration=iteration + 1,
                        converged=bool(stable),
                        rel_change=float(rel_change),
                        abs_change=float(abs_change),
                        collision_free=collisions.all_satisfied,
                        n_violations=collisions.n_violations,
                        osqp_status=info.get("status"),
                        osqp_pri_res=info.get("pri_res"),
                        osqp_dua_res=info.get("dua_res"),
                    )
                )
                if verbose:
                    print(
                        f"SCP iteration {iteration + 1}: change={abs_change:.3g}, grid violations={collisions.n_violations}"
                    )
                if time.perf_counter() >= deadline:
                    reason = "timeout"
                    break
                if collisions.all_satisfied and stable:
                    reason = "converged"
                    break
        metrics["timing"]["scp_iterations_time"] = time.perf_counter() - iteration_started
        return self._finalize(positions, velocities, accelerations, metrics, started, reason)

    def _valid_qp_vector(self, x):
        return x is not None and np.shape(x) == (self.total_vars,) and np.all(np.isfinite(x))

    def _grid_collisions(self, positions):
        return detect_collisions(
            positions,
            self.R,
            obstacle_positions=self.obstacle_positions,
            detection_tol=self.detection_tol,
        )

    def _get_var_indices(self, robot, timestep):
        base = self.robot_var_offsets[robot] + timestep * self.vars_per_timestep
        indices = {
            key: base + offset * self.d + np.arange(self.d)
            for offset, key in enumerate(("p", "v", "a"))
        }
        # Retain private planar aliases used by existing warm-start subclasses.
        if self.d == 2:
            indices.update(
                {
                    key + axis: indices[key][coordinate]
                    for key in ("p", "v", "a")
                    for coordinate, axis in enumerate(("x", "y"))
                }
            )
        return indices

    def _init_metrics(self):
        return dict(
            scp_iterations=0,
            converged=False,
            convergence_reason="",
            timing={},
            osqp_metrics=dict(initial_solve={}, iterations=[]),
            _debug_log=[],
        )

    def _build_dynamics_constraints(self):
        rows, cols, values, rhs = [], [], [], []

        def add(columns, coefficients, value):
            rows.extend([len(rhs)] * len(columns))
            cols.extend(columns)
            values.extend(coefficients)
            rhs.append(value)

        p0, v0, pf, vf = (
            a.reshape(self.N, self.d)
            for a in (
                self.initial_positions,
                self.initial_velocities,
                self.final_positions,
                self.final_velocities,
            )
        )
        for i in range(self.N):
            for k in range(self.K):
                current = self._get_var_indices(i, k)
                previous = self._get_var_indices(i, k - 1) if k else None
                for axis in range(self.d):
                    if k:
                        add(
                            [
                                current["p"][axis],
                                previous["p"][axis],
                                previous["v"][axis],
                                current["a"][axis],
                            ],
                            [1, -1, -self.h, -0.5 * self.h**2],
                            0,
                        )
                    else:
                        add(
                            [current["p"][axis], current["a"][axis]],
                            [1, -0.5 * self.h**2],
                            p0[i, axis] + self.h * v0[i, axis],
                        )
                for axis in range(self.d):
                    if k:
                        add(
                            [current["v"][axis], previous["v"][axis], current["a"][axis]],
                            [1, -1, -self.h],
                            0,
                        )
                    else:
                        add([current["v"][axis], current["a"][axis]], [1, -self.h], v0[i, axis])
            terminal = self._get_var_indices(i, self.K - 1)
            for key, target in (("p", pf), ("v", vf)):
                for axis in range(self.d):
                    add([terminal[key][axis]], [1], target[i, axis])
        matrix = sp.coo_matrix((values, (rows, cols)), shape=(len(rhs), self.total_vars)).tocsc()
        self.dynamics_constraint = Constraint(matrix, np.array(rhs), np.array(rhs))

    def _build_bound_constraints(self):
        rows, cols, values, lower, upper = [], [], [], [], []

        def add(columns, coefficients, lo, hi):
            rows.extend([len(lower)] * len(columns))
            cols.extend(columns)
            values.extend(coefficients)
            lower.append(lo)
            upper.append(hi)

        for i in range(self.N):
            for k in range(self.K):
                indices = self._get_var_indices(i, k)
                for key, lo, hi in (
                    ("v", self.vel_min, self.vel_max),
                    ("a", self.acc_min, self.acc_max),
                ):
                    if lo is not None or hi is not None:
                        for column in indices[key]:
                            add(
                                [column],
                                [1],
                                -np.inf if lo is None else lo,
                                np.inf if hi is None else hi,
                            )
                for axis, column in enumerate(indices["p"]):
                    if np.isfinite(self.pos_min[axis]) or np.isfinite(self.pos_max[axis]):
                        add([column], [1], self.pos_min[axis], self.pos_max[axis])
                if k and (self.jerk_min is not None or self.jerk_max is not None):
                    previous = self._get_var_indices(i, k - 1)
                    for axis in range(self.d):
                        add(
                            [previous["a"][axis], indices["a"][axis]],
                            [-1 / self.h, 1 / self.h],
                            -np.inf if self.jerk_min is None else self.jerk_min,
                            np.inf if self.jerk_max is None else self.jerk_max,
                        )
        matrix = sp.coo_matrix((values, (rows, cols)), shape=(len(lower), self.total_vars)).tocsc()
        self.bound_constraint = Constraint(matrix, np.array(lower), np.array(upper))

    def _build_cost_matrix(self):
        diagonal = np.zeros((self.N, self.K, 3, self.d))
        diagonal[:, :, 2] = 2
        return sp.diags(diagonal.ravel(), format="csc"), np.zeros(self.total_vars)

    def _linearization_direction(self, difference, travel):
        distance = np.linalg.norm(difference)
        if distance >= 1e-6:
            return difference / distance
        # At coincidence, choose a deterministic bypass normal orthogonal to travel.
        norm = np.linalg.norm(travel)
        eta = np.zeros(self.d)
        if norm > 1e-12:
            direction = travel / norm
            eta[np.argmin(np.abs(direction))] = 1
            eta -= (eta @ direction) * direction
            eta /= np.linalg.norm(eta)
        else:
            eta[0] = 1
        return eta

    def _collision_constraint(self, positions, obstacles=False):
        rows, cols, values, rhs = [], [], [], []
        travel = (self.final_positions - self.initial_positions).reshape(self.N, self.d)
        partners = (
            [(i, j) for i in range(self.N) for j in range(len(self.obstacle_positions))]
            if obstacles and self.obstacle_positions is not None
            else [] if obstacles else list(combinations(range(self.N), 2))
        )
        for i, j in partners:
            for k in range(self.K):
                other = self.obstacle_positions[j] if obstacles else positions[j][k]
                direction = travel[i] if obstacles else travel[i] - travel[j]
                eta = self._linearization_direction(positions[i][k] - other, direction)
                columns = list(self._get_var_indices(i, k)["p"])
                coefficients = list(eta)
                if not obstacles:
                    columns.extend(self._get_var_indices(j, k)["p"])
                    coefficients.extend(-eta)
                rows.extend([len(rhs)] * len(columns))
                cols.extend(columns)
                values.extend(coefficients)
                rhs.append(self.R + (eta @ other if obstacles else 0))
        matrix = sp.coo_matrix((values, (rows, cols)), shape=(len(rhs), self.total_vars)).tocsc()
        return normalize_constraint(Constraint(matrix, np.asarray(rhs), np.full(len(rhs), np.inf)))

    def _build_collision_constraints(self, prev_pos):
        return self._collision_constraint(prev_pos)

    def _build_obstacle_constraints(self, prev_pos):
        return self._collision_constraint(prev_pos, obstacles=True)

    def _solve_initial(self):
        p, q = self._build_cost_matrix()
        return solve_qp(
            p, q, [self.dynamics_constraint, self.bound_constraint], settings=self.osqp_settings
        )

    def _build_warm_start(self, positions, velocities, accelerations):
        arrays = [np.asarray(a) for a in (positions, velocities, accelerations)]
        if any(a.shape != (self.N, self.K, self.d) for a in arrays):
            raise ValueError("Warm-start states must have shape (N, K, d)")
        return np.stack(arrays, axis=2).ravel()

    def _solve_with_collisions(self, prev_pos, prev_vel, prev_acc):
        p, q = self._build_cost_matrix()
        collision = self._build_collision_constraints(prev_pos)
        obstacle = self._build_obstacle_constraints(prev_pos)
        constraints = [self.dynamics_constraint, self.bound_constraint, collision, obstacle]
        x, metrics = solve_qp(
            p,
            q,
            constraints,
            warm_start=self._build_warm_start(prev_pos, prev_vel, prev_acc),
            settings=self.osqp_settings,
        )
        y = metrics.get("y")
        if y is not None and collision.matrix.shape[0]:
            offset = (
                self.dynamics_constraint.matrix.shape[0] + self.bound_constraint.matrix.shape[0]
            )
            self._last_collision_duals = -y[offset : offset + collision.matrix.shape[0]].reshape(
                -1, self.K
            )
            self._last_dual_quality = {
                key: metrics.get(key) for key in ("pri_res", "dua_res", "polish_time", "status")
            }
        return x, metrics

    def get_collision_duals(self):
        return getattr(self, "_last_collision_duals", None), getattr(
            self, "_last_dual_quality", None
        )

    def _extract_trajectories(self, x):
        states = np.asarray(x).reshape(self.N, self.K, 3, self.d)
        return tuple([states[i, :, order].copy() for i in range(self.N)] for order in range(3))

    def _check_convergence(self, prev_pos, positions):
        difference = np.asarray(positions) - np.asarray(prev_pos)
        absolute = float(np.max(np.abs(difference)))
        relative = float(np.linalg.norm(difference) / max(np.linalg.norm(prev_pos), 1e-10))
        # Preserve the legacy absolute stopping rule; report relative change as before.
        return absolute <= self.scp_convergence_abs, relative, absolute

    def _validate_trajectory(self, positions, velocities, accelerations):
        p, v, a = map(np.asarray, (positions, velocities, accelerations))
        p0, v0, pf, vf = (
            array.reshape(self.N, self.d)
            for array in (
                self.initial_positions,
                self.initial_velocities,
                self.final_positions,
                self.final_velocities,
            )
        )
        left_p = np.concatenate((p0[:, None], p[:, :-1]), axis=1)
        left_v = np.concatenate((v0[:, None], v[:, :-1]), axis=1)
        error_p = np.max(np.abs(p - left_p - self.h * left_v - 0.5 * self.h**2 * a))
        error_v = np.max(np.abs(v - left_v - self.h * a))
        terminal_p = self.trajectory.sample([self.grid_horizon])[:, 0]
        terminal_v = self.trajectory.sample([self.grid_horizon], 1)[:, 0]
        endpoint_error = max(
            np.max(np.abs(p[:, -1] - pf)),
            np.max(np.abs(v[:, -1] - vf)),
            np.max(np.abs(terminal_p - pf)),
            np.max(np.abs(terminal_v - vf)),
        )
        drift = max(
            np.max(np.abs(self.trajectory.sample(self.times) - p)),
            np.max(np.abs(self.trajectory.sample(self.times, 1) - v)),
        )

        def violation(low, high, minimum, maximum):
            return float(max(0, np.max(minimum - low), np.max(high - maximum)))

        extrema = self.trajectory.extrema()
        bounds = {
            "position": (self.pos_min, self.pos_max),
            "velocity": (
                -np.inf if self.vel_min is None else self.vel_min,
                np.inf if self.vel_max is None else self.vel_max,
            ),
            "acceleration": (
                -np.inf if self.acc_min is None else self.acc_min,
                np.inf if self.acc_max is None else self.acc_max,
            ),
        }
        bound_errors = {key: violation(*extrema[key], *bounds[key]) for key in bounds}
        jerk = np.diff(a, axis=1) / self.h
        bound_errors["discrete_jerk"] = (
            violation(
                jerk.min(),
                jerk.max(),
                -np.inf if self.jerk_min is None else self.jerk_min,
                np.inf if self.jerk_max is None else self.jerk_max,
            )
            if jerk.size
            else 0.0
        )
        checks = dict(
            dynamics_position=float(error_p),
            dynamics_velocity=float(error_v),
            endpoint_error=float(endpoint_error),
            integrated_grid_error=float(drift),
            bounds=bound_errors,
        )
        checks["state_valid"] = max(error_p, error_v, endpoint_error, drift) <= self.validation_tol
        checks["bounds_valid"] = max(bound_errors.values()) <= self.validation_tol
        checks["valid"] = bool(checks["state_valid"] and checks["bounds_valid"])
        return checks

    def _finalize(self, positions, velocities, accelerations, metrics, started, reason):
        audit_started = time.perf_counter()
        numerical_converged = reason in ("converged", "collision_free_initial")
        metrics.update(
            dimension=self.d,
            n_timesteps=self.K,
            system_size=self.total_vars,
            vars_per_timestep=3 * self.d,
            grid_horizon=self.grid_horizon,
            numerical_converged=numerical_converged,
            continuous_feasible=False,
            tolerances=dict(separation=self.detection_tol, validation=self.validation_tol),
            stopping_reason=reason,
        )
        status = {
            "max_iterations": "iteration_limit",
            "qp_failure": "qp_failure",
            "timeout": "timeout",
        }.get(reason, "success")
        if positions is not None:
            self.trajectories = dict(
                positions=positions, velocities=velocities, accelerations=accelerations
            )
            self.trajectory = ZOHTrajectory.from_controls(
                self.initial_positions.reshape(self.N, self.d),
                self.initial_velocities.reshape(self.N, self.d),
                accelerations,
                self.h,
            )
            checks = self._validate_trajectory(positions, velocities, accelerations)
            separation = self.trajectory.separation(self.obstacle_positions)
            separation_ok = (
                min(separation["min_distance"], separation["min_obstacle_distance"])
                >= self.R - self.detection_tol
            )
            metrics.update(separation)
            metrics.update(
                checks=checks,
                continuous_feasible=bool(checks["valid"] and separation_ok),
                acceleration_cost=self.trajectory.acceleration_cost(),
            )
            grid_report = self._grid_collisions(positions)
            grid_min = min(
                (
                    float(np.linalg.norm(np.asarray(positions[i]) - positions[j], axis=1).min())
                    for i, j in combinations(range(self.N), 2)
                ),
                default=np.inf,
            )
            metrics["collision_check"] = dict(
                all_satisfied=grid_report.all_satisfied,
                n_robot_robot_violations=grid_report.n_robot_robot_violations,
                n_robot_obstacle_violations=grid_report.n_robot_obstacle_violations,
                worst_violation=max(
                    0,
                    self.R
                    - self.detection_tol
                    - min(
                        [
                            entry[3]
                            for entry in grid_report.robot_robot + grid_report.robot_obstacle
                        ],
                        default=np.inf,
                    ),
                ),
            )
            metrics["grid_min_distance"] = grid_min
            if status == "success":
                if not checks["state_valid"]:
                    status, reason = "invalid_solution", "state_validation_failed"
                elif not checks["bounds_valid"]:
                    status, reason = "continuous_infeasible", "continuous_bounds_violation"
                elif not separation_ok:
                    status, reason = "continuous_infeasible", "continuous_separation_violation"
        metrics.update(status=status, converged=status == "success", convergence_reason=reason)
        metrics["timing"]["validation_time"] = time.perf_counter() - audit_started
        metrics["timing"]["total_time"] = metrics["timing"]["solve_time"] = (
            time.perf_counter() - started
        )
        metrics["timing"].setdefault("scp_iterations_time", 0.0)
        metrics["timing"]["average_iteration_time"] = metrics["timing"][
            "scp_iterations_time"
        ] / max(metrics["scp_iterations"], 1)
        return dict(
            trajectories=self.trajectories,
            times=self.times.copy(),
            metrics=metrics,
            initial_guess=dict(
                positions=self._initial_guess_positions, velocities=self._initial_guess_velocities
            ),
        )

    def sample(self, n=1000, times=None):
        if self.trajectory is None:
            raise ValueError(
                "No continuous trajectory is available; solve successfully or inspect failure metrics"
            )
        times = (
            np.linspace(0, self.grid_horizon, n)
            if times is None
            else np.asarray(times, dtype=float)
        )
        return (times, *[list(self.trajectory.sample(times, order)) for order in (0, 1, 2)])
