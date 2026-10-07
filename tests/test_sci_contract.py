"""Regression tests for the agreed solver/evaluation boundaries."""

from types import SimpleNamespace
import json

import numpy as np
import pytest

from contact_solver import ContactSolver, PrioritizedSCI, PrioritizedSCP
from contact_solver.config import Config, ProblemConfig, SolverConfig
from contact_solver.evaluation import (
    continuous_acceleration_cost,
    discrete_acceleration_cost,
    sample_scp_grid,
)
from contact_solver.solvers.matching_numerics import damped_newton, insertion_time
from contact_solver.solvers.prioritized_sci import _ObstacleContact
from contact_solver.solvers.prioritized_utils import PiecewiseCubicTrajectory
from contact_solver.solvers.contact_solver import RelativeTrajectory


def test_saved_configuration_migrates_only_known_retired_options(tmp_path):
    from omegaconf.errors import ConfigAttributeError, ConfigKeyError

    saved = configuration().to_dict()
    retired = dict(fsolve_xtol=1e-8, root_method="fsolve", vectorize_residuals=True,
                   vectorize_postprocess=True, apply_velocity_bounds=False,
                   sci_geometry_tol=1e-8, sci_max_contacts=30, sci_max_contacts_per_pair=10)
    saved["solver"].update(retired)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(saved))
    migrated = Config.from_yaml(path)
    assert migrated.to_dict() == configuration().to_dict()
    for key in retired:
        assert not hasattr(migrated.solver, key)
        with pytest.raises(ConfigAttributeError):
            migrated.override(**{f"solver.{key}": retired[key]})
    saved["solver"]["misspelled_option"] = 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ConfigKeyError):
        Config.from_yaml(path)


def test_saved_disabled_timeout_uses_finite_default(tmp_path):
    saved = configuration().to_dict()
    saved["solver"]["timeout"] = 0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(saved))
    assert Config.from_yaml(path).solver.timeout == 300


@pytest.mark.parametrize("cls", [ContactSolver, PrioritizedSCI])
@pytest.mark.parametrize("timeout", [0, -1, np.inf, np.nan])
def test_sci_requires_finite_positive_timeout(cls, timeout):
    cfg = configuration()
    cfg.solver.timeout = timeout
    with pytest.raises(ValueError, match="timeout"):
        cls(cfg)


def configuration(n=2):
    return Config(
        name="contract",
        problem=ProblemConfig(n_robots=n, time_horizon=5.0, timestep=0.5, min_distance=0.5),
        solver=SolverConfig(
            sci_detection_tol=1e-6,
            scp_detection_tol=1e-3,
            osqp_eps_abs=1e-7,
            osqp_eps_rel=1e-7,
            scp_convergence_abs=1e-4,
            scp_convergence_rel=1e-4,
        ),
    )


def initialize(cls=ContactSolver, cfg=None):
    solver = cls(cfg or configuration())
    solver.set_initial_states([[0.0, 0.0], [0.0, 3.0]])
    solver.set_final_states([[5.0, 0.0], [5.0, 3.0]])
    return solver


def test_exact_tie_initialization_leaves_nearby_times_unchanged():
    existing = np.array([1.0, 2.0, 3.0])
    original = existing.copy()
    assert 2.0 < insertion_time(2.0, existing, 5.0) <= 2.0 + 5e-6
    for t in (2.0 + 1e-12, np.nextafter(2.0, 3.0), 2.0 - 1e-12):
        assert insertion_time(t, existing, 5.0) == t
    np.testing.assert_array_equal(existing, original)


def test_continuous_success_independent_of_sampling(monkeypatch):
    import contact_solver.solvers.contact_solver as module

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    calls = []
    original_closest = RelativeTrajectory.closest

    def closest(self):
        calls.append(1)
        clock.now += 2.0
        return original_closest(self)

    monkeypatch.setattr(RelativeTrajectory, "closest", closest)
    original_sample = ContactSolver.sample

    def sample(self, *args, **kwargs):
        clock.now += 100.0
        result = original_sample(self, *args, **kwargs)
        # Even an intentionally misleading plotting sample cannot change acceptance.
        result[1][1][:] = result[1][0]
        return result

    monkeypatch.setattr(ContactSolver, "sample", sample)
    result = initialize().generate_trajectories()
    assert result["metrics"]["converged"]
    assert calls == [1]  # No duplicate final full-horizon search.
    timing = result["metrics"]["timing"]
    assert timing["solve_time"] == timing["total_time"] == 2.0
    assert timing["postprocess_time"] == 100.0


@pytest.mark.parametrize("cls", [ContactSolver, PrioritizedSCI])
def test_comparison_uses_exact_scp_grid_and_uniform_weights(cls):
    solver = initialize(cls)
    solver.generate_trajectories()
    evaluated = sample_scp_grid(solver)
    times = np.arange(1, 11) * 0.5
    # Free cubic x(t)=.6t²-.08t³, acceleration=1.2-.48t.
    expected = 2 * 0.5 * np.sum((1.2 - 0.48 * times) ** 2)
    assert discrete_acceleration_cost(evaluated, 0.5) == pytest.approx(expected)
    assert continuous_acceleration_cost(solver) == pytest.approx(4.8)
    assert expected != pytest.approx(4.8)
    assert solver.K == 10  # Display sampling must not overwrite the SCP grid size.


def test_prioritized_detects_violation_at_and_near_existing_contact():
    solver = initialize(PrioritizedSCI)
    solver.T = 1.0
    solver.detection_tol = 0.001
    moving = np.zeros((4, 2))
    moving[0, 0] = -1.0
    moving[1, 0] = 2.0
    robot = PiecewiseCubicTrajectory([(moving, 0.0, 1.0)], 1.0)
    obstacle = PiecewiseCubicTrajectory([(np.zeros((4, 2)), 0.0, 1.0)], 1.0)
    for existing in (0.5, 0.50001):
        contacts = [_ObstacleContact(1, existing, 0.0, 0.0)]
        violations = solver._find_violations(robot, {1: obstacle}, contacts)
        assert any(t == pytest.approx(0.5) and d < 1e-10 for _, t, d in violations)


def test_admissible_newton_does_not_project_outside_root():
    class OutsideRoot:
        m = b = 1
        min_gap = 0.0

        def knot_times(self, x):
            if not 0 < x[0] < 1:
                raise ValueError("outside")
            return np.array([0.0, x[0], 1.0])

        def evaluate(self, x, with_jacobian=False):
            self.knot_times(x)
            residual = x - 2.0  # The mathematical root lies beyond T=1.
            return (
                (residual, np.array([[[[0.0]], [[1.0]], [[0.0]]]])) if with_jacobian else residual
            )

    x, info = damped_newton(OutsideRoot(), np.array([0.5]), np.inf, SolverConfig())
    assert not info["success"]
    assert 0.5 <= x[0] < 1.0  # No snap to the endpoint and no false solved root.


def test_prioritized_removal_reinsertion_stops_at_deadline(monkeypatch):
    import contact_solver.solvers.prioritized_sci as module

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    def repeated_violation(self, trajectory, obstacles, contacts):
        self._last_minimum = 0.0 if obstacles else np.inf
        return [(next(iter(obstacles)), 2.5, 0.0)] if obstacles else []

    def negative_mass(self, p0, v0, pf, vf, contacts, obstacles):
        clock.now += bool(contacts)
        return contacts, dict(success=True, residual=0.0), [-1.0] * len(contacts)

    monkeypatch.setattr(PrioritizedSCI, "_find_violations", repeated_violation)
    monkeypatch.setattr(PrioritizedSCI, "_solve_contact_system", negative_mass)
    cfg = configuration()
    cfg.solver.timeout = 125
    metrics = initialize(PrioritizedSCI, cfg).generate_trajectories()["metrics"]
    assert not metrics["converged"]
    assert metrics["convergence_reason"] == "timeout"
    assert metrics["rank_checks"][-1]["active_set_updates"] > 120


def test_matching_residual_is_checked_even_if_solver_claims_success(monkeypatch):
    def false_success(self, system, x, deadline):
        return x, dict(success=True, message="fake", nfev=0)

    monkeypatch.setattr(ContactSolver, "_solve_system", false_success)
    solver = ContactSolver(configuration())
    solver.set_initial_states([[-2.0, -0.2], [2.0, 0.2]])
    solver.set_final_states([[2.0, -0.2], [-2.0, 0.2]])
    metrics = solver.generate_trajectories()["metrics"]
    assert not metrics["converged"]
    assert metrics["convergence_reason"] == "invalid_matching_solution"
    assert metrics["checks"]["residual"] > solver.options.sci_residual_tol


@pytest.mark.parametrize("enabled", [False, True])
def test_condensed_recovery_default_and_opt_out(monkeypatch, enabled):
    assert SolverConfig().sci_reaction_recovery

    def fail(self, system, x, deadline):
        return x, dict(success=False, message="line_search_failed", nfev=0)

    calls = []

    def recover(*args):
        calls.append(1)
        return None

    monkeypatch.setattr(ContactSolver, "_solve_system", fail)
    monkeypatch.setattr(ContactSolver, "_recover_reactions", recover)
    cfg = configuration()
    cfg.solver.sci_reaction_recovery = enabled
    solver = ContactSolver(cfg)
    solver.set_initial_states([[-2.0, -0.2], [2.0, 0.2]])
    solver.set_final_states([[2.0, -0.2], [-2.0, 0.2]])
    assert not solver.generate_trajectories()["metrics"]["converged"]
    assert bool(calls) == enabled


@pytest.mark.parametrize(
    "cls,module_name", [(PrioritizedSCI, "prioritized_sci"), (PrioritizedSCP, "prioritized_scp")]
)
def test_priority_preprocessing_is_outside_both_timers(monkeypatch, cls, module_name):
    import importlib

    module = importlib.import_module("contact_solver.solvers." + module_name)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    def order(*args):
        clock.now += 200.0
        return [1, 0]

    monkeypatch.setattr(module, "conflict_severity_order", order)
    result = initialize(cls).generate_trajectories()
    assert result["metrics"]["priority_order"] == [1, 0]
    timing = result["metrics"]["timing"]
    assert timing["preprocessing_time"] == 200.0
    assert timing["solve_time"] == timing["total_time"] == 0.0


def test_paired_prioritized_runner_computes_order_once(monkeypatch):
    from contact_solver.cli import sweep_n_prioritized as runner
    from contact_solver.solvers import prioritized_sci, prioritized_scp

    calls = []

    def order(*args):
        calls.append(1)
        return [1, 0]

    def unexpected(*args):
        pytest.fail("solver recomputed the shared order")

    monkeypatch.setattr(runner, "conflict_severity_order", order)
    monkeypatch.setattr(prioritized_sci, "conflict_severity_order", unexpected)
    monkeypatch.setattr(prioritized_scp, "conflict_severity_order", unexpected)
    monkeypatch.setattr(
        runner,
        "generate_random_positions",
        lambda cfg: (np.array([[0.0, 0.0], [0.0, 3.0]]), np.array([[5.0, 0.0], [5.0, 3.0]])),
    )
    result = runner.run_single_experiment(configuration())
    assert calls == [1]
    assert result["priority_order"] == "[1, 0]"
    assert result["prio_sci_success"] and result["prio_scp_success"], result
    assert result["prio_sci_cost"] == pytest.approx(4.896)


def test_finite_difference_ablation_solves_same_matching_problem():
    results = []
    for analytical in (True, False):
        cfg = configuration()
        cfg.solver.use_analytical_jacobian = analytical
        solver = ContactSolver(cfg)
        solver.set_initial_states([[-2.0, -0.2], [2.0, 0.2]])
        solver.set_final_states([[2.0, -0.2], [-2.0, 0.2]])
        result = solver.generate_trajectories()
        assert result["metrics"]["converged"]
        results.append(np.asarray(result["trajectories"]["positions"]))
    np.testing.assert_allclose(*results, atol=1e-7)
