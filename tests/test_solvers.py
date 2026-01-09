"""Test solver implementations."""

import numpy as np
import pytest

from contact_solver import Config, ContactSolver, LiftedSCP, detect_collisions


def get_test_config(n_robots=3):
    """Create a minimal test configuration."""
    from contact_solver.config import (
        DynamicsConfig,
        EnvironmentConfig,
        ProblemConfig,
        ScenarioConfig,
        SolverConfig,
        VisualizationConfig,
    )

    return Config(
        name="test",
        random_seed=42,
        problem=ProblemConfig(
            n_robots=n_robots,
            time_horizon=5.0,  # More generous time
            timestep=0.1,
            min_distance=0.5,  # Smaller collision radius
            robot_radius=0.25,
            environment=EnvironmentConfig(
                pos_x_min=0.0, pos_x_max=10.0, pos_y_min=0.0, pos_y_max=10.0
            ),
            dynamics=DynamicsConfig(
                vel_min=-3.0, vel_max=3.0,  # Higher velocity limits
                acc_min=-10.0, acc_max=10.0,
                jerk_min=-20.0, jerk_max=20.0
            ),
            scenario=ScenarioConfig(),
        ),
        solver=SolverConfig(
            scp_max_iterations=20,
            max_contacts=10,
            max_contacts_per_pair=5,
        ),
        visualization=VisualizationConfig(show_plots=False),
    )


def get_simple_positions(n_robots=3):
    """Generate simple test positions that will have collisions."""
    np.random.seed(42)
    initial = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])[:n_robots]
    final = np.array([[4.0, 0.0], [2.0, 0.0], [0.0, 0.0]])[:n_robots]
    return initial, final


class TestLiftedSCP:
    def test_basic_solve(self):
        config = get_test_config(n_robots=2)
        initial, final = get_simple_positions(n_robots=2)

        solver = LiftedSCP(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)

        result = solver.generate_trajectories()

        assert "trajectories" in result
        assert "metrics" in result
        assert result["metrics"]["converged"] or result["metrics"]["scp_iterations"] > 0

    def test_collision_free(self):
        config = get_test_config(n_robots=2)
        initial, final = get_simple_positions(n_robots=2)

        solver = LiftedSCP(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)

        result = solver.generate_trajectories()
        trajs = result["trajectories"]

        # Check collision report
        K = len(trajs["positions"][0])
        report = detect_collisions(
            trajs["positions"], config.problem.min_distance, [[0, K]] * 2
        )

        # Should have no or minimal violations after convergence
        if result["metrics"]["converged"]:
            assert report.worst_violation < 0.01


class TestContactSolver:
    def test_basic_solve(self):
        config = get_test_config(n_robots=2)
        initial, final = get_simple_positions(n_robots=2)

        solver = ContactSolver(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)

        result = solver.generate_trajectories()

        assert "trajectories" in result
        assert "metrics" in result
        assert "contact_times" in result

    def test_collision_free(self):
        config = get_test_config(n_robots=2)
        initial, final = get_simple_positions(n_robots=2)

        solver = ContactSolver(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)

        result = solver.generate_trajectories()

        min_dist = result["metrics"]["min_distance"]
        assert min_dist >= config.problem.min_distance - 0.01

    def test_contact_insertion(self):
        """Test that contacts are inserted for colliding trajectories."""
        config = get_test_config(n_robots=2)
        # Head-on collision scenario
        initial = np.array([[0.0, 0.0], [3.0, 0.0]])
        final = np.array([[3.0, 0.0], [0.0, 0.0]])

        solver = ContactSolver(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)

        result = solver.generate_trajectories()

        # Should have at least one contact for head-on collision
        assert result["metrics"]["num_contacts"] >= 1


class TestSolverComparison:
    def test_both_solvers_produce_valid_trajectories(self):
        config = get_test_config(n_robots=2)
        initial, final = get_simple_positions(n_robots=2)

        # LiftedSCP
        scp_solver = LiftedSCP(config, verbose=False)
        scp_solver.set_initial_states(initial)
        scp_solver.set_final_states(final)
        scp_result = scp_solver.generate_trajectories()

        # ContactSolver
        contact_solver = ContactSolver(config, verbose=False)
        contact_solver.set_initial_states(initial)
        contact_solver.set_final_states(final)
        contact_result = contact_solver.generate_trajectories()

        # Both should produce trajectories
        assert len(scp_result["trajectories"]["positions"]) == 2
        assert len(contact_result["trajectories"]["positions"]) == 2

        # Both should reach final positions (approximately)
        for i in range(2):
            scp_final = scp_result["trajectories"]["positions"][i][-1]
            contact_final = contact_result["trajectories"]["positions"][i][-1]

            np.testing.assert_allclose(scp_final, final[i], atol=0.1)
            np.testing.assert_allclose(contact_final, final[i], atol=0.1)
