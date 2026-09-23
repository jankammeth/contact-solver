"""Dimension, discretization, regression, and independent continuous-time checks."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from scipy import sparse

from contact_solver import (
    Config,
    ContactSolver,
    LiftedSCP,
    PrioritizedSCI,
    PrioritizedSCP,
    VectorContactSolver,
)
from contact_solver.config import DynamicsConfig, EnvironmentConfig, ProblemConfig, SolverConfig
from contact_solver.solvers.zoh_trajectory import ZOHTrajectory


def config(d=2, n=2, **options):
    defaults = dict(
        scp_detection_tol=1e-3,
        sci_detection_tol=1e-3,
        scp_convergence_abs=1e-5,
        osqp_eps_abs=1e-7,
        osqp_eps_rel=1e-7,
        osqp_max_iter=20000,
        scp_max_iterations=50,
        apply_velocity_bounds=False,
    )
    defaults.update(options)
    return Config(
        name="lifted_nd_test",
        problem=ProblemConfig(
            n_robots=n, dimension=d, time_horizon=6, timestep=0.1, min_distance=1
        ),
        solver=SolverConfig(**defaults),
    )


def crossing(d):
    start = np.zeros((2, d))
    start[:, :2] = [[-2, -0.2], [2, 0.2]]
    goal = start.copy()
    goal[:, 0] *= -1
    return start, goal


def solve(start, goal, cfg=None, velocities=None, obstacles=None):
    solver = LiftedSCP(cfg or config(start.shape[1], len(start)), obstacle_positions=obstacles)
    v0, vf = (None, None) if velocities is None else velocities
    solver.set_initial_states(start, v0)
    solver.set_final_states(goal, vf)
    return solver, solver.generate_trajectories()


def independent_checks(solver, result, expect_separation=True):
    """Integrate controls recursively and minimize a separate squared polynomial."""
    p, v, a = (
        np.asarray(result["trajectories"][key])
        for key in ("positions", "velocities", "accelerations")
    )
    previous_p = solver.initial_positions.reshape(solver.N, solver.d).copy()
    previous_v = solver.initial_velocities.reshape(solver.N, solver.d).copy()
    minimum, obstacle_minimum = np.inf, np.inf
    for k in range(solver.K):
        coefficients = np.stack((previous_p, previous_v, 0.5 * a[:, k]), axis=1)
        targets = [
            (coefficients[i] - coefficients[j], False) for i, j in combinations(range(solver.N), 2)
        ]
        if solver.obstacle_positions is not None:
            for robot in coefficients:
                for obstacle in solver.obstacle_positions:
                    relative = robot.copy()
                    relative[0] -= obstacle
                    targets.append((relative, True))
        for relative, is_obstacle in targets:
            polynomials = [Polynomial(axis) for axis in relative.T]
            squared = sum((polynomial**2 for polynomial in polynomials), Polynomial([0]))
            candidates = [0, solver.h]
            derivative = squared.deriv()
            derivative = derivative.trim(tol=1e-14 * max(np.max(np.abs(derivative.coef)), 1e-100))
            candidates.extend(
                float(root.real)
                for root in derivative.roots()
                if abs(root.imag) < 1e-7 and 0 <= root.real <= solver.h
            )
            distance = min(
                np.linalg.norm([polynomial(t) for polynomial in polynomials]) for t in candidates
            )
            if is_obstacle:
                obstacle_minimum = min(obstacle_minimum, distance)
            else:
                minimum = min(minimum, distance)
        previous_p += solver.h * previous_v + 0.5 * solver.h**2 * a[:, k]
        previous_v += solver.h * a[:, k]
        np.testing.assert_allclose(p[:, k], previous_p, atol=1e-5)
        np.testing.assert_allclose(v[:, k], previous_v, atol=1e-5)
    np.testing.assert_allclose(
        previous_p, solver.final_positions.reshape(solver.N, solver.d), atol=1e-5
    )
    np.testing.assert_allclose(
        previous_v, solver.final_velocities.reshape(solver.N, solver.d), atol=1e-5
    )
    np.testing.assert_allclose(
        [minimum, obstacle_minimum],
        [result["metrics"]["min_distance"], result["metrics"]["min_obstacle_distance"]],
        atol=1e-8,
    )
    if expect_separation:
        assert min(minimum, obstacle_minimum) >= solver.R - solver.detection_tol - 1e-9
    assert result["metrics"]["acceleration_cost"] == pytest.approx(
        solver.h * np.sum(a**2), rel=1e-12
    )
    return minimum


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_counts_packing_and_sparse_dynamics(d):
    rng = np.random.default_rng(7)
    n, k, h = 3, 4, 0.2
    a, p0, v0 = rng.normal(size=(n, k, d)), rng.normal(size=(n, d)), rng.normal(size=(n, d))
    p, v = np.empty_like(a), np.empty_like(a)
    last_p, last_v = p0.copy(), v0.copy()
    for t in range(k):
        p[:, t] = last_p + h * last_v + 0.5 * h**2 * a[:, t]
        v[:, t] = last_v + h * a[:, t]
        last_p, last_v = p[:, t], v[:, t]
    cfg = config(d, n)
    cfg.problem.time_horizon, cfg.problem.timestep = k * h, h
    solver = LiftedSCP(cfg)
    solver.set_initial_states(p0, v0)
    solver.set_final_states(p[:, -1], v[:, -1])
    solver._prepare()
    solver._build_dynamics_constraints()
    solver._build_bound_constraints()
    assert solver.total_vars == 3 * d * n * k
    assert solver.dynamics_constraint.matrix.shape == (2 * d * n * (k + 1), solver.total_vars)
    x = solver._build_warm_start(p, v, a)
    for expected, actual in zip((p, v, a), solver._extract_trajectories(x), strict=True):
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_allclose(
        solver.dynamics_constraint.matrix @ x, solver.dynamics_constraint.lower, atol=1e-14
    )
    P, q = solver._build_cost_matrix()
    assert sparse.isspmatrix_csc(P) and P.nnz == n * k * d
    assert 0.5 * x @ P @ x + q @ x == pytest.approx(np.sum(a**2))
    collision = solver._build_collision_constraints(p)
    assert collision.matrix.shape == (n * (n - 1) // 2 * k, len(x))
    assert collision.matrix.nnz <= n * (n - 1) // 2 * k * 2 * d
    if d == 2:
        index = solver._get_var_indices(1, 2)
        assert [index[key] for key in ("px", "py", "vx", "vy", "ax", "ay")] == list(range(36, 42))


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_embedded_and_rotated_planar_encounter(d):
    reference, ref = solve(*crossing(2))
    embedded, result = solve(*crossing(d))
    assert result["metrics"]["converged"]
    independent_checks(embedded, result)
    np.testing.assert_allclose(
        np.asarray(result["trajectories"]["positions"])[:, :, :2],
        ref["trajectories"]["positions"],
        atol=2e-5,
    )
    if d > 2:
        np.testing.assert_allclose(
            np.asarray(result["trajectories"]["positions"])[:, :, 2:], 0, atol=1e-10
        )
    rotation, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(d, d)))
    start, goal = crossing(d)
    rotated, rot_result = solve(start @ rotation, goal @ rotation)
    assert rot_result["metrics"]["converged"]
    independent_checks(rotated, rot_result)
    np.testing.assert_allclose(
        rot_result["trajectories"]["positions"],
        np.asarray(result["trajectories"]["positions"]) @ rotation,
        atol=5e-5,
    )
    assert rot_result["metrics"]["acceleration_cost"] == pytest.approx(
        ref["metrics"]["acceleration_cost"], rel=2e-6
    )


@pytest.mark.parametrize("d", [3, 5, 10])
def test_nonplanar_interaction_with_endpoint_velocities(d):
    start, goal = crossing(d)
    v0, vf = np.zeros_like(start), np.zeros_like(start)
    v0[:, 2], vf[:, 2] = [0.15, -0.1], [-0.1, 0.2]
    solver, result = solve(start, goal, velocities=(v0, vf))
    assert result["metrics"]["converged"] and result["metrics"]["scp_iterations"] > 0
    relative = np.diff(np.asarray(solver.sample(n=51)[1]), axis=0)[0]
    assert np.linalg.matrix_rank(relative, tol=1e-7) >= 3
    independent_checks(solver, result)


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_free_motion_all_coordinates(d):
    start = np.zeros((3, d))
    start[:, 0] = [0, 3, 6]
    goal = start + np.linspace(0.2, 1, d)
    v0, vf = np.tile(np.linspace(0.1, 0.3, d), (3, 1)), np.tile(np.linspace(-0.1, 0.1, d), (3, 1))
    solver, result = solve(start, goal, velocities=(v0, vf))
    assert result["metrics"]["converged"] and result["metrics"]["scp_iterations"] == 0
    assert result["times"][0] == solver.h and result["times"][-1] == solver.T
    independent_checks(solver, result)


REFERENCE = json.loads(
    (Path(__file__).parent / "data/lifted_scp_planar_reference.json").read_text()
)


@pytest.mark.parametrize("record", REFERENCE["records"], ids=lambda r: r["case"])
def test_prechange_planar_regression(record):
    solver, result = solve(
        np.array(record["initial_positions"]),
        np.array(record["final_positions"]),
        velocities=(np.array(record["initial_velocities"]), np.array(record["final_velocities"])),
    )
    assert record["converged"] and result["metrics"]["numerical_converged"]
    if record["case"] == "velocity":
        assert result["metrics"]["status"] == "continuous_infeasible"
    else:
        assert result["metrics"]["converged"]
    assert result["metrics"]["acceleration_cost"] == pytest.approx(record["cost"], rel=2e-5)
    if record["case"] != "head_on":
        for key, snapshots in record["snapshots"].items():
            np.testing.assert_allclose(
                np.asarray(result["trajectories"][key])[:, record["indices"]], snapshots, atol=2e-5
            )
    independent_checks(solver, result, expect_separation=record["case"] != "velocity")


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_translated_obstacles_and_constraint_offsets(d):
    center = np.linspace(3, 5, d)[None]
    start, goal = np.zeros((1, d)), np.zeros((1, d))
    start[0, :2], goal[0, :2] = [-2, -0.3], [2, -0.3]
    solver, result = solve(start + center, goal + center, config(d, 1), obstacles=center)
    assert result["metrics"]["converged"]
    independent_checks(solver, result)
    constraint = solver._build_obstacle_constraints(result["trajectories"]["positions"])
    x = solver._build_warm_start(**result["trajectories"])
    assert np.min(constraint.matrix @ x - constraint.lower) >= -1e-6
    assert constraint.matrix.shape[0] == solver.K


def test_position_velocity_acceleration_and_discrete_jerk_bounds():
    cfg = config(3, 1)
    cfg.problem.time_horizon = 2
    cfg.problem.environment = EnvironmentConfig(position_lower=[-0.1] * 3, position_upper=[2] * 3)
    cfg.problem.dynamics = DynamicsConfig(
        vel_min=-2, vel_max=2, acc_min=-3, acc_max=3, jerk_min=-6, jerk_max=6
    )
    start, goal = np.zeros((1, 3)), np.array([[0.2, 0.6, 1.0]])
    solver, result = solve(start, goal, cfg)
    assert result["metrics"]["converged"]
    independent_checks(solver, result)
    x = solver._build_warm_start(**result["trajectories"])
    bound = solver.bound_constraint
    assert bound.matrix.shape[0] == 3 * (3 * solver.K + solver.K - 1)
    values = bound.matrix @ x
    assert np.all(values >= bound.lower - 1e-6) and np.all(values <= bound.upper + 1e-6)
    positions, velocities, accelerations = map(np.asarray, solver.sample(n=101)[1:])
    assert positions.min() >= -0.1 - 1e-7 and positions.max() <= 2 + 1e-7
    assert np.abs(velocities).max() <= 2 + 1e-7 and np.abs(accelerations).max() <= 3 + 1e-7
    assert (
        np.max(np.abs(np.diff(result["trajectories"]["accelerations"], axis=1) / solver.h))
        <= 6 + 1e-7
    )


@pytest.mark.parametrize("d", [2, 5])
def test_grid_feasibility_does_not_hide_interior_collision(d):
    cfg = config(d)
    cfg.problem.time_horizon = cfg.problem.timestep = 1
    start = np.zeros((2, d))
    start[:, -1] = [-1, 1]
    goal = -start
    velocity = goal - start
    solver, result = solve(start, goal, cfg, (velocity, velocity))
    metrics = result["metrics"]
    assert metrics["numerical_converged"] and metrics["collision_check"]["all_satisfied"]
    assert metrics["grid_min_distance"] > 1.9
    assert not metrics["converged"] and metrics["status"] == "continuous_infeasible"
    assert metrics["min_distance"] < 1e-9 and metrics["closest_time"] == pytest.approx(0.5)
    independent_checks(solver, result, expect_separation=False)


def test_interior_position_bound_and_initial_collision_are_checked():
    cfg = config(2, 1)
    cfg.problem.time_horizon = cfg.problem.timestep = 1
    cfg.problem.environment = EnvironmentConfig(position_upper=[0.25, 1])
    solver, result = solve(
        np.zeros((1, 2)), np.zeros((1, 2)), cfg, (np.array([[2.0, 0]]), np.array([[-2.0, 0]]))
    )
    assert result["metrics"]["convergence_reason"] == "continuous_bounds_violation"
    assert result["metrics"]["checks"]["bounds"]["position"] == pytest.approx(0.25)
    cfg = config(3)
    cfg.problem.time_horizon = cfg.problem.timestep = 1
    start = np.zeros((2, 3))
    goal = np.array([[-2.0, 0, 0], [2.0, 0, 0]])
    _, result = solve(start, goal, cfg, (goal, goal))
    assert result["metrics"]["status"] == "continuous_infeasible"
    assert result["metrics"]["closest_time"] == 0


def test_constant_distance_and_all_pairs_and_segments():
    start = np.array([[0.0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [3, 0, 0, 0, 0.2]])
    trajectory = ZOHTrajectory.from_controls(start, np.zeros_like(start), np.zeros((3, 2, 5)), 0.5)
    separation = trajectory.separation()
    assert separation["closest_pair"] == (1, 2)
    assert separation["min_distance"] == pytest.approx(0.2)
    assert separation["closest_time"] == 0
    a = np.zeros((1, 2, 2))
    a[0, :, 0] = [1, -2]
    trajectory = ZOHTrajectory.from_controls(np.zeros((1, 2)), np.zeros((1, 2)), a, 0.3)
    assert trajectory.sample([0.3], 2)[0, 0, 0] == -2  # right-continuous control
    assert trajectory.sample([0.6], 2)[0, 0, 0] == -2


def test_degenerate_directions_are_deterministic():
    solver = LiftedSCP(config(5))
    for travel in (np.zeros(5), np.arange(5, dtype=float)):
        first = solver._linearization_direction(np.zeros(5), travel)
        second = solver._linearization_direction(np.zeros(5), travel)
        np.testing.assert_array_equal(first, second)
        assert np.linalg.norm(first) == pytest.approx(1)
        assert first @ travel == pytest.approx(0, abs=1e-12)
    start, goal = crossing(5)
    start[:, 1] = goal[:, 1] = 0
    _, result = solve(start, goal)
    assert result["metrics"]["converged"]


def test_failures_and_iteration_limits_are_distinct(monkeypatch):
    start, goal = crossing(3)
    _, limited = solve(start, goal, config(3, scp_max_iterations=0))
    assert limited["metrics"]["status"] == "iteration_limit"
    _, timed = solve(start, goal, config(3, timeout=1e-12))
    assert timed["metrics"]["status"] == "timeout"
    cfg = config(3)
    cfg.problem.dynamics = DynamicsConfig(acc_min=0, acc_max=0)
    _, failed = solve(start, goal, cfg)
    assert failed["metrics"]["status"] == "qp_failure" and failed["trajectories"] is None

    def fail(*args):
        raise RuntimeError("forced iteration QP failure")

    monkeypatch.setattr(LiftedSCP, "_solve_with_collisions", fail)
    _, failed_iteration = solve(start, goal)
    assert failed_iteration["metrics"]["status"] == "qp_failure"
    assert failed_iteration["trajectories"] is not None
    assert not failed_iteration["metrics"]["converged"]


def test_config_input_guards_and_time_grid(tmp_path):
    cfg = config(3)
    cfg.problem.environment = EnvironmentConfig(position_lower=[-5] * 3, position_upper=[5] * 3)
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = Config.from_yaml(path)
    np.testing.assert_array_equal(loaded.problem.environment.pos_min, [-5] * 3)
    solver = LiftedSCP(loaded)
    with pytest.raises(ValueError, match="shape"):
        solver.set_initial_states(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        solver.set_initial_states(np.full((2, 3), np.nan))
    with pytest.raises(ValueError, match="Obstacles"):
        LiftedSCP(cfg, obstacle_positions=np.zeros((2, 2)))
    for cls in (ContactSolver, PrioritizedSCP, PrioritizedSCI):
        legacy = cls(config(3))
        with pytest.raises(ValueError, match="only dimension=2"):
            legacy.set_initial_states(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="one entry"):
        EnvironmentConfig(position_lower=[0, 0]).bounds(3)
    with pytest.raises(ValueError, match="mix"):
        EnvironmentConfig(pos_x_min=0, position_lower=[0, 0]).bounds(2)
    legacy = EnvironmentConfig(pos_x_min=-1, pos_y_max=5)
    assert legacy.bounds(3)[0][2] == -np.inf and legacy.pos_max[1] == 5
    cfg = config(3)
    cfg.problem.time_horizon = 1.05
    with pytest.raises(ValueError, match="integer multiple"):
        solve(*crossing(3), cfg)
    cfg.problem.time_horizon, cfg.problem.timestep = 0.3, 0.1
    start = np.array([[0.0, 0, 0], [3.0, 0, 0]])
    solver, result = solve(start, start, cfg)
    assert solver.K == 3 and result["metrics"]["converged"]
    cfg.problem.time_horizon = None
    solver, result = solve(start, start, cfg)
    assert solver.K == 1 and result["metrics"]["converged"]


def test_joint_cost_and_separation_comparison():
    start, goal = crossing(3)
    cfg = config(3)
    lifted, result = solve(start, goal, cfg)
    vector = VectorContactSolver(cfg)
    vector.set_initial_states(start)
    vector.set_final_states(goal)
    sci = vector.generate_trajectories()
    from contact_solver.cli.vector_sci import acceleration_cost

    assert result["metrics"]["converged"] and sci["metrics"]["converged"]
    assert result["metrics"]["acceleration_cost"] == pytest.approx(
        acceleration_cost(vector, vector.trajectory.times), rel=1e-3
    )
    assert result["metrics"]["system_size"] == 3 * 3 * 2 * 60
    assert sci["metrics"]["system_size"] == 8
    assert sci["metrics"]["intrinsic_degrees_of_freedom"] == 6


def test_obstacle_collision_between_grid_nodes():
    cfg = config(5, 1)
    cfg.problem.time_horizon = cfg.problem.timestep = 1
    center = np.arange(5, dtype=float)[None]
    start, goal = center.copy(), center.copy()
    start[0, -1] -= 1
    goal[0, -1] += 1
    velocity = goal - start
    solver, result = solve(start, goal, cfg, (velocity, velocity), center)
    assert result["metrics"]["collision_check"]["all_satisfied"]
    assert result["metrics"]["status"] == "continuous_infeasible"
    assert result["metrics"]["min_obstacle_distance"] < 1e-10
    independent_checks(solver, result, expect_separation=False)


@pytest.mark.parametrize("curvature", [0.0, 1e-23, -1e-23, 1e-10])
def test_nearly_linear_distance_roots_are_not_lost(curvature):
    from contact_solver.solvers.zoh_trajectory import quadratic_distance_candidates

    coefficients = np.array([[-2.0, 0.1], [4.0, 0], [curvature, 0], [0.0, 0]])
    candidates = quadratic_distance_candidates(coefficients, 1)
    t, minimum = min(candidates, key=lambda item: item[1])
    assert t == pytest.approx(0.5, abs=1e-10)
    assert minimum == pytest.approx(0.1, abs=1e-12)
    # A repeated stationary root, with no sign change in a derivative partition.
    coefficients[:, 0] = [0.25, -1, 1, 0]
    candidates = quadratic_distance_candidates(coefficients, 1)
    assert min(candidates, key=lambda item: item[1]) == pytest.approx((0.5, 0.1), abs=1e-10)


def test_saved_endpoint_comparison_cli(tmp_path):
    from contact_solver.cli.compare_joint import main

    cfg = config(3)
    config_path, states_path, report_path = (
        tmp_path / "config.yaml",
        tmp_path / "states.npz",
        tmp_path / "report.json",
    )
    cfg.to_yaml(config_path)
    start, goal = crossing(3)
    np.savez(states_path, initial_positions=start, final_positions=goal)
    args = [
        "--config",
        str(config_path),
        "--states",
        str(states_path),
        "--output",
        str(report_path),
    ]
    assert main(args) == 0
    report = json.loads(report_path.read_text())
    assert report["scp"]["system_size"] == 1080
    assert report["sci"]["system_size"] == 8
    assert report["scp"]["converged"] and report["sci"]["converged"]
    cfg.solver.scp_max_iterations = 0
    cfg.to_yaml(config_path)
    assert main(args) == 1
    assert json.loads(report_path.read_text())["scp"]["status"] == "iteration_limit"
    cfg.solver.sci_detection_tol = 0.1
    cfg.to_yaml(config_path)
    with pytest.raises(ValueError, match="equal"):
        main(args)


def test_invalid_qp_state_is_not_accepted(monkeypatch):
    start, goal = crossing(3)

    def invalid_state(self):
        positions = np.broadcast_to(goal[:, None, :], (2, self.K, 3)).copy()
        return (
            self._build_warm_start(positions, np.zeros_like(positions), np.zeros_like(positions)),
            {},
        )

    monkeypatch.setattr(LiftedSCP, "_solve_initial", invalid_state)
    _, result = solve(start, goal)
    assert result["metrics"]["numerical_converged"]
    assert result["metrics"]["status"] == "invalid_solution"
    assert not result["metrics"]["checks"]["state_valid"]
