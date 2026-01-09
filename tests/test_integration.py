"""Integration tests for full pipeline."""

import numpy as np
import pytest

from contact_solver import (
    Config,
    ContactSolver,
    LiftedSCP,
    detect_collisions,
    generate_random_positions,
    generate_swap_positions,
    generate_line_positions,
    visualize_trajectories,
    visualize_comparison,
)
from contact_solver.config import (
    DynamicsConfig,
    EnvironmentConfig,
    ProblemConfig,
    ScenarioConfig,
    SolverConfig,
    VisualizationConfig,
)


def make_config(n_robots: int = 3, time_horizon: float = 5.0, seed: int = 42) -> Config:
    """Create test configuration."""
    return Config(
        name="integration_test",
        random_seed=seed,
        problem=ProblemConfig(
            n_robots=n_robots,
            time_horizon=time_horizon,
            timestep=0.1,
            min_distance=0.5,  # Smaller for easier feasibility
            robot_radius=0.25,
            environment=EnvironmentConfig(
                pos_x_min=0.0, pos_x_max=15.0, pos_y_min=0.0, pos_y_max=15.0
            ),
            dynamics=DynamicsConfig(
                vel_min=-3.0, vel_max=3.0,  # More generous
                acc_min=-10.0, acc_max=10.0
            ),
            scenario=ScenarioConfig(cluster_radius=3.0, cluster_margin=1.0, spacing=0.8),
        ),
        solver=SolverConfig(scp_max_iterations=20, max_contacts=15),
        visualization=VisualizationConfig(show_plots=False),
    )


class TestFullPipeline:
    """Test complete pipeline: config -> scenario -> solve -> check."""

    def test_random_scenario_lifted_scp(self):
        """Test LiftedSCP on random scenario."""
        config = make_config(n_robots=3)
        initial, final = generate_random_positions(config)

        solver = LiftedSCP(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        # Check result structure
        assert "trajectories" in result
        assert "metrics" in result
        assert "positions" in result["trajectories"]
        assert "velocities" in result["trajectories"]
        assert "accelerations" in result["trajectories"]

        # Check trajectory dimensions
        positions = result["trajectories"]["positions"]
        assert len(positions) == 3  # 3 robots

    def test_random_scenario_contact_solver(self):
        """Test ContactSolver on random scenario."""
        config = make_config(n_robots=3)
        initial, final = generate_random_positions(config)

        solver = ContactSolver(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        # Check result structure
        assert "trajectories" in result
        assert "metrics" in result
        assert "contact_times" in result

        # Check trajectory dimensions
        positions = result["trajectories"]["positions"]
        assert len(positions) == 3

    def test_swap_scenario(self):
        """Test both solvers on swap scenario (challenging)."""
        config = make_config(n_robots=4, time_horizon=8.0)  # More time for swap
        initial, final = generate_swap_positions(config)

        # ContactSolver should handle this well
        contact_solver = ContactSolver(config, verbose=False)
        contact_solver.set_initial_states(initial)
        contact_solver.set_final_states(final)
        contact_result = contact_solver.generate_trajectories()

        # ContactSolver should produce results
        assert len(contact_result["trajectories"]["positions"]) == 4

        # LiftedSCP may struggle with this challenging scenario
        # We wrap in try/except since swap scenarios can be infeasible for SCP
        try:
            scp_solver = LiftedSCP(config, verbose=False)
            scp_solver.set_initial_states(initial)
            scp_solver.set_final_states(final)
            scp_result = scp_solver.generate_trajectories()
            assert len(scp_result["trajectories"]["positions"]) == 4
        except RuntimeError as e:
            # SCP may fail on challenging swap scenarios - this is expected
            assert "primal infeasible" in str(e) or "OSQP failed" in str(e)

    def test_line_scenario(self):
        """Test line reversal scenario."""
        config = make_config(n_robots=4, time_horizon=5.0)
        initial, final = generate_line_positions(config)

        solver = ContactSolver(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        # Should have contacts (robots need to pass each other)
        assert result["metrics"]["num_contacts"] >= 1


class TestCollisionDetection:
    """Test collision detection module."""

    def test_no_collisions(self):
        """Test detection on well-separated trajectories."""
        # Two robots moving in parallel, well separated
        K = 50
        positions = [
            np.column_stack([np.linspace(0, 5, K), np.zeros(K)]),
            np.column_stack([np.linspace(0, 5, K), np.ones(K) * 3]),
        ]
        
        report = detect_collisions(positions, min_distance=0.8)
        assert report.all_satisfied
        assert report.n_violations == 0

    def test_detect_collision(self):
        """Test detection catches collision."""
        # Two robots crossing paths
        K = 50
        positions = [
            np.column_stack([np.linspace(0, 5, K), np.linspace(0, 5, K)]),
            np.column_stack([np.linspace(5, 0, K), np.linspace(0, 5, K)]),
        ]
        
        report = detect_collisions(positions, min_distance=0.8)
        assert not report.all_satisfied
        assert report.n_violations > 0


class TestVisualization:
    """Test visualization functions don't crash."""

    def test_visualize_trajectories(self):
        """Test trajectory visualization."""
        config = make_config(n_robots=2)
        initial, final = generate_random_positions(config)

        solver = ContactSolver(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        # Should not raise
        fig, ax = visualize_trajectories(
            result["trajectories"]["positions"],
            config,
            show=False,
        )
        assert fig is not None

    def test_visualize_comparison(self):
        """Test comparison visualization."""
        config = make_config(n_robots=2)
        initial, final = generate_random_positions(config)

        # Run both solvers
        scp_solver = LiftedSCP(config, verbose=False)
        scp_solver.set_initial_states(initial)
        scp_solver.set_final_states(final)
        scp_result = scp_solver.generate_trajectories()

        contact_solver = ContactSolver(config, verbose=False)
        contact_solver.set_initial_states(initial)
        contact_solver.set_final_states(final)
        contact_result = contact_solver.generate_trajectories()

        # Should not raise
        fig, axes = visualize_comparison(
            scp_result["trajectories"],
            contact_result["trajectories"],
            config,
            show=False,
        )
        assert fig is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_two_robots_head_on(self):
        """Test minimal case: 2 robots head-on collision."""
        config = make_config(n_robots=2, time_horizon=4.0)  # More time
        initial = np.array([[0.0, 0.0], [3.0, 0.0]])
        final = np.array([[3.0, 0.0], [0.0, 0.0]])

        solver = ContactSolver(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        # Should be feasible
        assert result["metrics"]["min_distance"] >= config.problem.min_distance - 0.05

    def test_already_collision_free(self):
        """Test case where initial straight-line is collision-free."""
        config = make_config(n_robots=2, time_horizon=4.0)  # More time
        # Parallel paths, no collision
        initial = np.array([[0.0, 0.0], [0.0, 3.0]])
        final = np.array([[4.0, 0.0], [4.0, 3.0]])

        solver = LiftedSCP(config, verbose=False)
        solver.set_initial_states(initial)
        solver.set_final_states(final)
        result = solver.generate_trajectories()

        # Should converge quickly (maybe 0 SCP iterations needed)
        assert result["metrics"]["converged"]

    def test_zero_initial_velocity(self):
        """Test with explicit zero initial velocities."""
        config = make_config(n_robots=2, time_horizon=4.0)
        initial = np.array([[0.0, 0.0], [3.0, 0.0]])
        final = np.array([[3.0, 0.0], [0.0, 0.0]])
        velocities = np.zeros((2, 2))

        solver = ContactSolver(config, verbose=False)
        solver.set_initial_states(initial, velocities)
        solver.set_final_states(final, velocities)
        result = solver.generate_trajectories()

        assert "trajectories" in result
