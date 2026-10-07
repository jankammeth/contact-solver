"""Independent checks of the explicit-vector formulation, not sampled acceptance."""

from itertools import combinations

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from scipy.linalg import solve_banded

from contact_solver import Config, ContactSolver
from contact_solver.cli.vector_sci import smoke_endpoints
from contact_solver.config import ProblemConfig, SolverConfig
from contact_solver.solvers.cubic_utils import distance_candidates
from contact_solver.solvers.contact_solver import RelativeTrajectory
from contact_solver.solvers.matching_numerics import damped_newton
from contact_solver.solvers.vector_matching import (
    MatchingSystem,
    VectorContact,
    blocks_to_banded,
    blocks_to_sparse,
)


def config(d=2, n=2, **options):
    return Config(
        name="vector_test",
        problem=ProblemConfig(
            n_robots=n, dimension=d, time_horizon=5, timestep=0.1, min_distance=1
        ),
        solver=SolverConfig(
            **dict(
                sci_detection_tol=1e-6,
                use_analytical_jacobian=True,
                **options,
            )
        ),
    )


def crossing(d=2):
    start = np.zeros((2, d))
    start[:, :2] = [[-2, -0.2], [2, 0.2]]
    goal = start.copy()
    goal[:, 0] *= -1
    return start, goal


def solve(start, goal, cfg=None, velocities=None, solver_type=ContactSolver):
    solver = solver_type(cfg or config(start.shape[1], len(start)))
    v0, vf = (None, None) if velocities is None else velocities
    solver.set_initial_states(start, v0)
    solver.set_final_states(goal, vf)
    result = solver.generate_trajectories()
    return solver, result


def independent_checks(solver):
    """Check coefficients directly; polynomial squared distances use a separate path."""
    trajectory = solver.trajectory
    c, knots = trajectory.coefficients, trajectory.times
    start = solver.initial_positions.reshape(solver.N, solver.d)
    goal = solver.final_positions.reshape(solver.N, solver.d)
    v0 = solver.initial_velocities.reshape(solver.N, solver.d)
    vf = solver.final_velocities.reshape(solver.N, solver.d)
    np.testing.assert_allclose(trajectory.robots([0, solver.T])[:, 0], start, atol=1e-10)
    np.testing.assert_allclose(trajectory.robots([0, solver.T])[:, 1], goal, atol=1e-10)
    np.testing.assert_allclose(trajectory.robots([0, solver.T], 1)[:, 0], v0, atol=1e-10)
    np.testing.assert_allclose(trajectory.robots([0, solver.T], 1)[:, 1], vf, atol=1e-10)
    for k, contact in enumerate(solver.contacts):
        h = knots[k + 1] - knots[k]
        # C0, C1, C2 and the relative/robot jerk balances, independently of F.
        for derivative in (0, 1, 2):
            left = np.array(
                [
                    [Polynomial(c[k, j, :, axis]).deriv(derivative)(h) for axis in range(solver.d)]
                    for j in range(solver.N - 1)
                ]
            )
            right = c[k + 1, :, derivative] * (1, 1, 2)[derivative]
            np.testing.assert_allclose(left, right, atol=2e-7)
        relative_jump = 6 * (c[k + 1, :, 3] - c[k, :, 3])
        first = relative_jump.sum(axis=0) / solver.N
        robot_jump = np.vstack((first, first - relative_jump))
        expected = np.zeros_like(robot_jump)
        p, q = contact.pair
        expected[p] = contact.mass * solver.R * contact.normal
        expected[q] = -expected[p]
        np.testing.assert_allclose(robot_jump, expected, atol=2e-7)
        positions = trajectory.robots([contact.t_star])[:, 0]
        velocities = trajectory.robots([contact.t_star], 1)[:, 0]
        np.testing.assert_allclose(
            positions[p] - positions[q], solver.R * contact.normal, atol=1e-10
        )
        np.testing.assert_allclose(velocities[p] - velocities[q], contact.velocity, atol=1e-10)
        assert abs(contact.normal @ contact.normal - 1) < 1e-8
        assert abs(contact.normal @ contact.velocity) < 1e-8
        assert contact.mass >= -1e-9
    minimum = np.inf
    for k, h in enumerate(np.diff(knots)):
        for p, q in combinations(range(solver.N), 2):
            relative = c[k, q - 1] - (c[k, p - 1] if p else 0)
            squared = sum((Polynomial(axis) ** 2 for axis in relative.T), Polynomial([0]))
            candidates = [0.0, h]
            candidates.extend(
                root.real
                for root in squared.deriv().roots()
                if abs(root.imag) < 1e-6 and 0 <= root.real <= h
            )
            # Evaluate vector norm to avoid cancellation in squared value near zero.
            minimum = min(
                minimum,
                *(np.linalg.norm([Polynomial(axis)(t) for axis in relative.T]) for t in candidates),
            )
    assert minimum >= solver.R - solver.options.sci_detection_tol - 1e-10
    np.testing.assert_allclose(minimum, trajectory.closest()[2], atol=1e-9)
    return minimum


def acceleration_cost(solver):
    # Two-point Gaussian quadrature integrates the quadratic acceleration cost exactly.
    knots = solver.trajectory.times
    cost = 0.0
    for a, b in zip(knots[:-1], knots[1:], strict=True):
        times = (a + b) / 2 + (b - a) / 2 * np.array([-1, 1]) / np.sqrt(3)
        acc = np.asarray(solver.sample(times=times)[3])
        cost += (b - a) / 2 * np.sum(acc**2)
    return cost


@pytest.mark.parametrize("component", ["normalization", "tangency"])
@pytest.mark.parametrize("error,accepted", [(5e-8, True), (2e-7, False)])
@pytest.mark.parametrize("reaction_constructed", [False, True])
def test_geometry_equations_use_the_matching_tolerance(
    component, error, accepted, reaction_constructed
):
    # A linear relative trajectory has zero acceleration/jerk residuals;
    # only the chosen contact-geometry equation has a nonzero residual.
    cfg = config(sci_root_max_iterations=0)
    solver = ContactSolver(cfg)
    solver.min_gap = cfg.solver.sci_min_contact_gap
    normal = np.array([np.sqrt(1 + error) if component == "normalization" else 1.0, 0.0])
    velocity = np.array([error if component == "tangency" else 0.0, 0.0])
    time = solver.T / 2
    r0 = solver.R * normal - time * velocity
    rf = solver.R * normal + time * velocity
    solver.set_initial_states(np.array([r0, -r0]) / 2, np.array([velocity, -velocity]) / 2)
    solver.set_final_states(np.array([rf, -rf]) / 2, np.array([velocity, -velocity]) / 2)
    contact = VectorContact(
        (0, 1), time, normal, velocity, 0.0, np.empty((0, 2)), np.empty((0, 2))
    )
    system = solver.matching_system([contact])
    x = contact.to_vector()
    trajectory = RelativeTrajectory.from_system(system, x, np.zeros((4, 2)))
    trajectory.reaction_constructed = reaction_constructed
    checks = solver._checks(system, x, trajectory)
    assert checks[component] == pytest.approx(error, rel=1e-7, abs=1e-15)
    assert checks["residual"] == pytest.approx(error, rel=1e-7, abs=1e-15)
    assert checks["valid"] is accepted
    _, info = damped_newton(system, x, np.inf, cfg.solver)
    assert info["success"] is accepted


@pytest.mark.parametrize("d", [2, 3, 5, 10])
@pytest.mark.parametrize("n", [2, 4])
def test_packing_counts_jacobian_and_neighbor_structure(d, n):
    rng = np.random.default_rng(151)
    pairs = [(0, 1), (n - 2, n - 1), (0, n - 1), (0, 1)]
    system = MatchingSystem(pairs, *[rng.normal(size=(n - 1, d)) for _ in range(4)], 1.3, 9, 1e-5)
    x = rng.normal(size=(4, system.b))
    x[:, 0] = [1, 2.7, 5, 7]
    x = x.ravel()
    assert len(x) == 4 * (2 * d * (n - 1) + 2)
    assert system.evaluate(x).shape == x.shape
    for block, pair in zip(x.reshape(4, system.b), pairs, strict=True):
        contact = VectorContact.from_vector(block, pair, n, d)
        np.testing.assert_array_equal(contact.to_vector(), block)
    jac = system.jacobian(x).toarray()
    numeric = np.empty_like(jac)
    for j in range(len(x)):
        step = 1e-5 * max(1, abs(x[j]))
        delta = np.zeros_like(x)
        delta[j] = step
        numeric[:, j] = (system.evaluate(x + delta) - system.evaluate(x - delta)) / (2 * step)
    np.testing.assert_allclose(jac, numeric, rtol=2e-7, atol=1e-7)
    for i in range(4):
        for j in range(4):
            if abs(i - j) > 1:
                assert not np.any(
                    jac[i * system.b : (i + 1) * system.b, j * system.b : (j + 1) * system.b]
                )
    blocks = system.evaluate(x, True)[1]
    width, banded = blocks_to_banded(blocks)
    # Test the actual linear storage/solve on a well-conditioned shifted Jacobian.
    for k in range(4):
        blocks[k, 1] += 1000 * np.eye(system.b)
    width, banded = blocks_to_banded(blocks)
    rhs = rng.normal(size=len(x))
    step = solve_banded((width, width), banded, rhs)
    np.testing.assert_allclose(blocks_to_sparse(blocks) @ step, rhs, atol=1e-12)


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_embedded_and_rotated_planar_solutions(d):
    start, goal = crossing(d)
    reference, result = solve(*crossing())
    embedded, result = solve(start, goal)
    assert result["metrics"]["converged"]
    independent_checks(embedded)
    np.testing.assert_allclose(
        embedded.sample()[1],
        np.pad(np.array(reference.sample()[1]), ((0, 0), (0, 0), (0, d - 2))),
        atol=1e-7,
    )
    q, _ = np.linalg.qr(np.random.default_rng(25).normal(size=(d, d)))
    rotated, result = solve(start @ q, goal @ q)
    assert result["metrics"]["converged"]
    independent_checks(rotated)
    np.testing.assert_allclose(rotated.sample()[1], np.array(embedded.sample()[1]) @ q, atol=1e-7)
    np.testing.assert_allclose(acceleration_cost(rotated), acceleration_cost(reference), rtol=1e-8)


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_no_contact_with_moving_barycenter(d):
    rng = np.random.default_rng(66)
    start = np.zeros((3, d))
    start[:, 0] = [0, 3, 6]
    goal = start + rng.normal(size=d)
    v0 = np.tile(rng.normal(size=d), (3, 1))
    vf = np.tile(rng.normal(size=d), (3, 1))
    solver, result = solve(start, goal, velocities=(v0, vf))
    assert result["metrics"]["converged"]
    assert result["metrics"]["num_contacts"] == 0
    assert (
        result["metrics"]["system_size"] == result["metrics"]["intrinsic_degrees_of_freedom"] == 0
    )
    independent_checks(solver)
    times = np.linspace(0, solver.T, 17)
    from scipy.interpolate import CubicHermiteSpline

    center = CubicHermiteSpline(
        [0, solver.T], [start.mean(0), goal.mean(0)], [v0.mean(0), vf.mean(0)]
    )
    np.testing.assert_allclose(
        np.mean(solver.sample(times=times)[1], axis=0), center(times), atol=1e-12
    )


@pytest.mark.parametrize("d", [3, 5, 10])
def test_genuinely_nonplanar_interacting_fleet(d):
    # Perturb the seed-82 fleet in all extra coordinates, then rotate it.
    start, goal = smoke_endpoints(d, "fleet")
    q, _ = np.linalg.qr(np.random.default_rng(19).normal(size=(d, d)))
    start, goal = start @ q, goal @ q
    assert np.linalg.matrix_rank(np.vstack((start - start[0], goal - start[0]))) >= 3
    cfg = config(d, 3)
    solver, result = solve(start, goal, cfg)
    assert result["metrics"]["converged"]
    assert len(set(c.pair for c in solver.contacts)) == 3
    independent_checks(solver)


@pytest.mark.parametrize("case", ["head_on", "offset", "figure82", "velocity"])
def test_planar_cases_use_one_joint_solver(case):
    cfg = config()
    if case == "figure82":
        rng = np.random.default_rng(82)

        def draw():
            while True:
                points = rng.uniform(-4, 4, (3, 2))
                if all(
                    np.linalg.norm(points[p] - points[q]) > 1.5
                    for p, q in combinations(range(3), 2)
                ):
                    return points

        start, goal = draw(), draw()
        cfg = config(n=3)
        cfg.problem.time_horizon = 6
        cfg.solver.sci_detection_tol = 1e-3
    else:
        start, goal = crossing()
        if case == "head_on":
            start[:, 1] = goal[:, 1] = 0
    velocities = (
        (np.array([[0.2, -0.1], [-0.15, 0.05]]), np.array([[0.1, 0.2], [-0.1, -0.2]]))
        if case == "velocity"
        else None
    )
    vector, result = solve(start, goal, cfg, velocities)
    assert result["metrics"]["converged"]
    independent_checks(vector)


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_degenerate_initialization(d):
    solver = ContactSolver(config(d))
    solver.min_gap = 1e-5
    zeros = np.zeros((1, d))
    for velocity in (np.zeros_like(zeros), np.ones_like(zeros)):
        system = MatchingSystem([], -velocity, velocity, velocity, velocity, 1, 2, 1e-5)
        trajectory = RelativeTrajectory.from_system(system, np.empty(0), np.zeros((4, d)))
        a = solver.initialize_contact((0, 1), 1, trajectory)
        b = solver.initialize_contact((0, 1), 1, trajectory)
        np.testing.assert_array_equal(a.normal, b.normal)
        assert np.linalg.norm(a.normal) == pytest.approx(1)
        assert a.normal @ a.velocity == pytest.approx(0, abs=1e-14)
        assert a.mass == 0


def test_continuous_detection_endpoints_constants_all_coordinates_and_pairs():
    for d in (2, 3, 5, 10):
        c = np.zeros((4, d))
        c[0, -1], c[1, -1] = -2, 4
        candidates = distance_candidates(c, 1)
        assert min(candidates, key=lambda value: value[1]) == pytest.approx((0.5, 0))
        c[1] = 0
        assert distance_candidates(c, 1, 2) == [(2, 2), (3, 2)]
    coefficients = np.zeros((2, 2, 4, 3))
    coefficients[:, 0, 0, 0] = 2
    coefficients[0, 1, 0] = [3, 0, 1]
    coefficients[0, 1, 1] = [-1, 0, -1]
    coefficients[1, 1, 0] = [2, 0, 0]
    coefficients[1, 1, 1] = [1, 0, 1]
    trajectory = RelativeTrajectory(np.array([0, 1, 2]), coefficients, np.zeros((4, 3)))
    assert trajectory.closest() == ((1, 2), 1, 0)  # another pair exactly at a knot


def test_statuses_do_not_accept_sampled_separation_or_unconverged_roots():
    start, goal = crossing()
    solver, result = solve(start, goal, config(sci_min_contact_gap=3))
    assert np.min(np.linalg.norm(np.diff(solver.sample(n=2)[1], axis=0), axis=2)) > 1
    assert result["metrics"]["min_distance"] < 1
    assert result["metrics"]["status"] == "unresolved"
    assert result["metrics"]["convergence_reason"] == "minimum_knot_gap"
    for options, expected_status, expected_reason in (
        ({"sci_min_contact_gap": 3}, "unresolved", "minimum_knot_gap"),
        ({"sci_root_max_iterations": 0}, "root_failure", "iteration_limit"),
    ):
        solver, result = solve(start, goal, config(**options))
        assert not result["metrics"]["converged"]
        assert result["metrics"]["status"] == expected_status
        assert result["metrics"]["convergence_reason"] == expected_reason
    start[1] = start[0]
    _, result = solve(start, goal)
    assert result["metrics"]["status"] == "invalid_solution"
    assert result["metrics"]["convergence_reason"] == "infeasible_endpoint_separation"


def test_negative_mass_is_invalid_even_with_small_matching_residual():
    # A forced contact pulls an already separated constant path inward.
    start = np.array([[0.0, 0.0], [1.5, 0.0]])
    solver, _ = solve(start, start)
    contact = solver.initialize_contact((0, 1), 2.5, solver.trajectory)
    system = solver.matching_system([contact])
    x, info = solver._solve_system(system, contact.to_vector(), np.inf)
    assert info["success"]
    trajectory = RelativeTrajectory.from_system(system, x, solver.trajectory.barycenter)
    checks = solver._checks(system, x, trajectory)
    assert checks["residual"] < solver.options.sci_residual_tol
    assert checks["minimum_mass"] < 0
    assert not checks["valid"]


def test_time_domain_and_jacobian_are_consistent():
    solver, _ = solve(*crossing(), cfg=config(sci_min_contact_gap=1e-5))
    system, x = solver.system, solver.solution.copy()
    for t in (0, solver.T, system.min_gap / 2):
        x[0] = t
        with pytest.raises(ValueError, match="Contact times"):
            system.evaluate(x)
        with pytest.raises(ValueError, match="Contact times"):
            system.jacobian(x)
    # Test time derivatives near the lower admissible gap with one-sided differences.
    x[0] = 2 * system.min_gap
    h = system.min_gap * 1e-4
    delta = np.zeros_like(x)
    delta[0] = h
    numerical = (system.evaluate(x + delta) - system.evaluate(x - delta)) / (2 * h)
    analytical = system.jacobian(x).toarray()[:, 0]
    np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1)


def test_configuration_roundtrip_and_input_validation(tmp_path):
    cfg = config(d=5)
    path = tmp_path / "vector.yaml"
    cfg.to_yaml(path)
    loaded = Config.from_yaml(path)
    assert loaded.problem.dimension == 5
    solver = ContactSolver(loaded)
    with pytest.raises(ValueError, match="endpoint shape"):
        solver.set_initial_states(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        solver.set_initial_states(np.full((2, 5), np.nan))
    with pytest.raises(ValueError, match="dimension"):
        ContactSolver(config(d=1))
    cfg.problem.time_horizon = None
    solver, result = solve(*crossing(5), cfg)
    assert solver.T > 0 and result["metrics"]["converged"]


def test_sparse_fallback_and_cli_paths(monkeypatch, tmp_path):
    import json

    import contact_solver.solvers.matching_numerics as module
    from contact_solver.cli.vector_sci import main

    def singular(*args, **kwargs):
        raise module.LinAlgError("forced singular Newton matrix")

    monkeypatch.setattr(module, "solve_banded", singular)
    solver, result = solve(*crossing(5))
    assert result["metrics"]["converged"]
    assert sum(root["sparse_fallbacks"] for root in result["metrics"]["roots"]) > 0
    independent_checks(solver)
    cfg = config(5)
    config_path, states_path = tmp_path / "config.yaml", tmp_path / "states.npz"
    output_path, trajectory_path = tmp_path / "metrics.json", tmp_path / "trajectory.npz"
    cfg.to_yaml(config_path)
    start, goal = crossing(5)
    np.savez(states_path, initial_positions=start, final_positions=goal)
    args = [
        "--states",
        str(states_path),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--trajectories",
        str(trajectory_path),
    ]
    assert main(args) == 0
    assert json.loads(output_path.read_text())["converged"]
    with np.load(trajectory_path) as data:
        assert data["positions"].shape == (2, 1000, 5)
        assert data["contact_states"].shape == (1, 12)
        assert data["contact_pairs"].shape == (1, 2)
    cfg.solver.sci_root_max_iterations = 0
    cfg.to_yaml(config_path)
    assert main(args) == 1
    assert json.loads(output_path.read_text())["status"] == "root_failure"


@pytest.mark.parametrize("duration", [1e-4, 1, 1000])
def test_all_stationary_polynomial_candidates(duration):
    cubic = Polynomial.fromroots([0.1, 0.5, 0.9])
    coefficients = np.zeros((4, 5))
    coefficients[:, -1] = cubic.coef / duration ** np.arange(4)
    coefficients[0, 1] = 0.25
    candidates = distance_candidates(coefficients, duration)
    for fraction in (0.1, 0.5, 0.9):
        assert any(
            abs(t / duration - fraction) < 1e-8 and abs(distance - 0.25) < 1e-12
            for t, distance in candidates
        )
    assert candidates[0][0] == 0
    assert candidates[-1][0] == duration


def test_repeated_pair_contacts():
    rng = np.random.default_rng(2)
    start, goal = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    cfg = config(3, 3)
    cfg.problem.min_distance = 0.65
    solver, result = solve(start, goal, cfg)
    assert result["metrics"]["converged"]
    assert len(solver.contacts) == 2
    assert solver.contacts[0].pair == solver.contacts[1].pair
    independent_checks(solver)


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_five_robot_smoke_and_direct_endpoints(d):
    start, goal = smoke_endpoints(d, "fleet5")
    cfg = config(d=d, n=5)
    cfg.problem.time_horizon = 6
    solver, result = solve(start, goal, cfg)
    assert result["metrics"]["converged"]
    assert result["metrics"]["num_contacts"] > 0
    independent_checks(solver)
    endpoints = solver.trajectory.endpoint_states()
    for order in (0, 1):
        np.testing.assert_allclose(
            endpoints[order], solver.trajectory.robots([0, solver.T], order), atol=1e-12
        )


def test_fleet20_seed2_removes_negative_mass_contact():
    from pathlib import Path

    with np.load(Path(__file__).parent / "data/fleet20_seed2.npz") as data:
        cfg = config(d=2, n=20, sci_reaction_recovery=True)
        cfg.solver.sci_detection_tol = 1e-3
        cfg.problem.time_horizon = float(data["T"])
        cfg.problem.min_distance = 0.5
        solver, result = solve(data["p0"], data["pf"], cfg, velocities=(data["v0"], data["vf"]))
    metrics = result["metrics"]
    assert metrics["converged"]
    assert metrics["num_contacts"] == 3
    assert len(metrics["removed_contacts"]) == 1
    assert metrics["removed_contacts"][0]["pair"] == (16, 17)
    assert metrics["checks"]["minimum_mass"] > 0
    independent_checks(solver)


def test_contact_removal_reinsertion_stops_at_deadline(monkeypatch):
    from types import SimpleNamespace
    import contact_solver.solvers.contact_solver as module

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    def fake_root(self, system, x, deadline):
        clock.now += 1
        x = x.copy()
        x.reshape(system.m, system.b)[:, 1 + 2 * self.d] = -1
        return x, dict(success=True, message="test_negative_mass", nfev=0)

    monkeypatch.setattr(ContactSolver, "_solve_system", fake_root)
    _, result = solve(*crossing(), cfg=config(timeout=125))
    metrics = result["metrics"]
    assert not metrics["converged"]
    assert metrics["convergence_reason"] == "timeout"
    assert metrics["active_set_updates"] > 120


@pytest.mark.parametrize("seed", [9, 15, 35, 47])
def test_fleet20_reaction_recovery(seed):
    from pathlib import Path

    with np.load(Path(__file__).parent / f"data/fleet20_seed{seed}.npz") as data:
        cfg = config(d=2, n=20, sci_reaction_recovery=True)
        cfg.solver.sci_detection_tol = 1e-3
        cfg.problem.time_horizon = float(data["T"])
        cfg.problem.min_distance = 0.5
        solver, result = solve(data["p0"], data["pf"], cfg, velocities=(data["v0"], data["vf"]))
    assert result["metrics"]["converged"]
    assert any(root["message"] == "reaction_recovery" for root in result["metrics"]["roots"])
    independent_checks(solver)


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_reaction_jacobian_and_coefficient_construction(d):
    from contact_solver.solvers.reaction_matching import ReactionMatching

    rng = np.random.default_rng(23)
    pairs = [(0, 1), (1, 2), (0, 2)]
    system = MatchingSystem(pairs, *[rng.normal(size=(2, d)) for _ in range(4)], 0.5, 8, 1e-5)
    reduced = ReactionMatching(system)
    z = rng.normal(size=(3, d + 2))
    z[:, 0] = [5, 2, 2.001]  # order exchange and very close knots
    z[:, -1] *= 0.01
    w = z.ravel()
    jac = reduced.evaluate(w, True)
    numeric = np.empty_like(jac)
    for j in range(len(w)):
        delta = np.zeros_like(w)
        delta[j] = 1e-6
        numeric[:, j] = (reduced.evaluate(w + delta) - reduced.evaluate(w - delta)) / (2e-6)
    np.testing.assert_allclose(jac, numeric, rtol=2e-6, atol=2e-6)
    times, coeffs, order = reduced.coefficients(w)
    trajectory = RelativeTrajectory(times, coeffs, np.zeros((4, d)), True)
    relative = trajectory.relative([0, 8])
    vel = trajectory.relative([0, 8], 1)
    np.testing.assert_allclose(relative, [system.r0, system.rf], atol=1e-11)
    np.testing.assert_allclose(vel, [system.v0, system.vf], atol=1e-11)
    for k, idx in enumerate(order):
        jump = 6 * (coeffs[k + 1, :, 3] - coeffs[k, :, 3])
        expected = system.R * system.jerk_factors[idx, :, None] * z[idx, -1] * z[idx, 1 : 1 + d]
        np.testing.assert_allclose(jump, expected, atol=1e-12)


def test_reaction_recovery_respects_deadline():
    from contact_solver.solvers.reaction_matching import ReactionMatching

    solver, _ = solve(*crossing())
    reduced = ReactionMatching(solver.system)
    with pytest.raises(TimeoutError):
        reduced.solve(solver.solution, deadline=0, max_evaluations=100)
