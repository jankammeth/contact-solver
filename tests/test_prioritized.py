"""Tests for PrioritizedSCI and PrioritizedSCP."""

import numpy as np
import pytest

from contact_solver import (
    Config,
    PrioritizedSCI,
    PrioritizedSCP,
    detect_collisions,
)
from contact_solver.config import (
    DynamicsConfig,
    EnvironmentConfig,
    ProblemConfig,
    ScenarioConfig,
    SolverConfig,
    VisualizationConfig,
)
from contact_solver.solvers.prioritized_utils import conflict_severity_order


def _config(n_robots: int = 3, K: int = 50, T: float = 5.0) -> Config:
    return Config(
        name="test_prioritized",
        random_seed=42,
        problem=ProblemConfig(
            n_robots=n_robots,
            time_horizon=T,
            timestep=T / K,
            min_distance=0.5,
            robot_radius=0.25,
            environment=EnvironmentConfig(),
            dynamics=DynamicsConfig(),
            scenario=ScenarioConfig(),
        ),
        solver=SolverConfig(
            scp_detection_tol=1e-3,
            sci_detection_tol=1e-3,
            scp_convergence_abs=1e-2,
            scp_convergence_rel=1e-2,
            osqp_eps_abs=1e-4,
            osqp_eps_rel=0.0,
            scp_max_iterations=20,
            sci_max_contacts=20,
            sci_max_contacts_per_pair=10,
        ),
        visualization=VisualizationConfig(show_plots=False),
    )


def _swap_positions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """N robots on a circle, swapping to antipodal points (head-on collisions)."""
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    R = 4.0
    initial = R * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    final = -initial
    return initial, final


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


class TestConflictSeverityOrder:
    def test_deterministic(self):
        initial, final = _swap_positions(4)
        v0 = np.zeros_like(initial)
        vf = np.zeros_like(final)
        order_a = conflict_severity_order(initial, v0, final, vf, T=5.0, R=0.5)
        order_b = conflict_severity_order(initial, v0, final, vf, T=5.0, R=0.5)
        assert order_a == order_b

    def test_covers_all_robots(self):
        initial, final = _swap_positions(5)
        v0 = np.zeros_like(initial)
        vf = np.zeros_like(final)
        order = conflict_severity_order(initial, v0, final, vf, T=5.0, R=0.5)
        assert sorted(order) == [0, 1, 2, 3, 4]

    def test_no_conflict_no_crash(self):
        # Parallel motion: no collision -> all zero scores, valid order returned
        initial = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
        final = np.array([[5.0, 0.0], [5.0, 2.0], [5.0, 4.0]])
        v0 = np.zeros_like(initial)
        vf = np.zeros_like(final)
        order = conflict_severity_order(initial, v0, final, vf, T=5.0, R=0.5)
        assert sorted(order) == [0, 1, 2]


# ---------------------------------------------------------------------------
#  PrioritizedSCI smoke tests
# ---------------------------------------------------------------------------


class TestPrioritizedSCI:
    def test_solves_head_on_pair(self):
        cfg = _config(n_robots=2)
        initial = np.array([[0.0, 0.0], [3.0, 0.0]])
        final = np.array([[3.0, 0.0], [0.0, 0.0]])

        solver = PrioritizedSCI(cfg, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        assert result["metrics"]["converged"]
        min_d = result["metrics"]["min_distance"]
        assert min_d >= cfg.problem.min_distance - cfg.solver.sci_detection_tol

    def test_swap_three_robots(self):
        cfg = _config(n_robots=3)
        initial, final = _swap_positions(3)

        solver = PrioritizedSCI(cfg, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        assert result["metrics"]["converged"]
        assert result["metrics"]["min_distance"] >= cfg.problem.min_distance - 1e-3

        # Sanity: result dict shape
        assert "trajectories" in result
        for key in ("positions", "velocities", "accelerations"):
            assert key in result["trajectories"]
            assert len(result["trajectories"][key]) == 3

    def test_no_collisions_initial_is_feasible(self):
        # Parallel motion: each robot solves with 0 contacts vs its predecessors
        cfg = _config(n_robots=3)
        initial = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
        final = np.array([[5.0, 0.0], [5.0, 2.0], [5.0, 4.0]])

        solver = PrioritizedSCI(cfg, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        assert result["metrics"]["converged"]
        assert result["metrics"]["num_contacts"] == 0

    def test_metric_keys(self):
        cfg = _config(n_robots=2)
        initial = np.array([[0.0, 0.0], [3.0, 0.0]])
        final = np.array([[3.0, 0.0], [0.0, 0.0]])

        solver = PrioritizedSCI(cfg)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()
        m = result["metrics"]

        for key in (
            "timing", "converged", "convergence_reason",
            "num_contacts", "num_contacts_per_rank",
            "min_distance", "priority_order",
        ):
            assert key in m
        assert len(m["num_contacts_per_rank"]) == 2
        assert len(m["priority_order"]) == 2


# ---------------------------------------------------------------------------
#  PrioritizedSCP smoke tests
# ---------------------------------------------------------------------------


class TestPrioritizedSCP:
    def test_solves_head_on_pair(self):
        cfg = _config(n_robots=2)
        initial = np.array([[0.0, 0.0], [3.0, 0.0]])
        final = np.array([[3.0, 0.0], [0.0, 0.0]])

        solver = PrioritizedSCP(cfg, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        assert result["metrics"]["converged"]
        min_d = result["metrics"]["min_distance"]
        assert min_d >= cfg.problem.min_distance - cfg.solver.scp_detection_tol

    def test_swap_three_robots(self):
        cfg = _config(n_robots=3)
        initial, final = _swap_positions(3)

        solver = PrioritizedSCP(cfg, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        assert result["metrics"]["converged"]
        # Discrete-grid check; allow detection tol
        K = cfg.problem.n_timesteps
        report = detect_collisions(
            result["trajectories"]["positions"],
            cfg.problem.min_distance,
            detection_tol=cfg.solver.scp_detection_tol,
        )
        assert report.all_satisfied

    def test_no_collisions_initial_is_feasible(self):
        cfg = _config(n_robots=3)
        initial = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
        final = np.array([[5.0, 0.0], [5.0, 2.0], [5.0, 4.0]])

        solver = PrioritizedSCP(cfg, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        assert result["metrics"]["converged"]
        assert result["metrics"]["scp_iterations_total"] == 0

    def test_metric_keys(self):
        cfg = _config(n_robots=2)
        initial = np.array([[0.0, 0.0], [3.0, 0.0]])
        final = np.array([[3.0, 0.0], [0.0, 0.0]])

        solver = PrioritizedSCP(cfg)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()
        m = result["metrics"]

        for key in (
            "timing", "converged", "convergence_reason",
            "scp_iterations_total", "scp_iterations_per_rank", "rank_reasons",
            "min_distance", "priority_order",
        ):
            assert key in m
        assert len(m["scp_iterations_per_rank"]) == 2
        assert len(m["priority_order"]) == 2


# ---------------------------------------------------------------------------
#  Cross-solver sanity: both find a feasible joint trajectory on the same input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_robots", [2, 3])
def test_both_solvers_feasible(n_robots):
    cfg = _config(n_robots=n_robots)
    initial, final = _swap_positions(n_robots)

    sci = PrioritizedSCI(cfg)
    sci.set_initial_states(initial)
    sci.set_final_states(final)
    res_sci = sci.generate_trajectories()

    scp = PrioritizedSCP(cfg)
    scp.set_initial_states(initial)
    scp.set_final_states(final)
    res_scp = scp.generate_trajectories()

    assert res_sci["metrics"]["converged"]
    assert res_scp["metrics"]["converged"]

    # Both solvers should agree on the priority order (it's a pure function of
    # the boundary states and only depends on T, R).
    assert res_sci["metrics"]["priority_order"] == res_scp["metrics"]["priority_order"]
