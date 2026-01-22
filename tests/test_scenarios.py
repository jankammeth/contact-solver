"""Test scenario generation."""

import numpy as np
import pytest

from contact_solver import (
    generate_random_positions,
    generate_swap_positions,
    generate_line_positions,
    generate_random_obstacles,
)
from contact_solver.config import (
    Config,
    DynamicsConfig,
    EnvironmentConfig,
    ProblemConfig,
    ScenarioConfig,
    SolverConfig,
    VisualizationConfig,
)


def get_test_config(n_robots=5):
    return Config(
        name="test",
        random_seed=42,
        problem=ProblemConfig(
            n_robots=n_robots,
            time_horizon=5.0,
            timestep=0.2,
            min_distance=0.5,
            robot_radius=0.25,
            environment=EnvironmentConfig(
                pos_x_min=0.0, pos_x_max=20.0, pos_y_min=0.0, pos_y_max=20.0
            ),
            dynamics=DynamicsConfig(),
            scenario=ScenarioConfig(cluster_radius=4.0, cluster_margin=1.5, spacing=0.8),
        ),
        solver=SolverConfig(),
        visualization=VisualizationConfig(show_plots=False),
    )


class TestRandomPositions:
    def test_correct_shape(self):
        config = get_test_config(n_robots=5)
        initial, final = generate_random_positions(config)

        assert initial.shape == (5, 2)
        assert final.shape == (5, 2)

    def test_within_bounds(self):
        config = get_test_config(n_robots=10)
        initial, final = generate_random_positions(config)

        env = config.problem.environment
        assert np.all(initial[:, 0] >= env.pos_x_min)
        assert np.all(initial[:, 0] <= env.pos_x_max)
        assert np.all(initial[:, 1] >= env.pos_y_min)
        assert np.all(initial[:, 1] <= env.pos_y_max)

    def test_reproducible_with_seed(self):
        config = get_test_config()

        initial1, final1 = generate_random_positions(config, seed=123)
        initial2, final2 = generate_random_positions(config, seed=123)

        np.testing.assert_array_equal(initial1, initial2)
        np.testing.assert_array_equal(final1, final2)

    def test_different_with_different_seed(self):
        config = get_test_config()

        initial1, _ = generate_random_positions(config, seed=123)
        initial2, _ = generate_random_positions(config, seed=456)

        assert not np.allclose(initial1, initial2)

    def test_overlapping_clusters(self):
        """Initial and final positions should be in same region (workspace center)."""
        config = get_test_config(n_robots=5)
        initial, final = generate_random_positions(config, seed=42)

        env = config.problem.environment
        cx = (env.pos_x_min + env.pos_x_max) / 2
        cy = (env.pos_y_min + env.pos_y_max) / 2
        center = np.array([cx, cy])

        # Both clusters should be centered near workspace center
        initial_center = np.mean(initial, axis=0)
        final_center = np.mean(final, axis=0)

        assert np.linalg.norm(initial_center - center) < config.problem.scenario.cluster_radius + 1
        assert np.linalg.norm(final_center - center) < config.problem.scenario.cluster_radius + 1


class TestSwapPositions:
    def test_correct_shape(self):
        config = get_test_config(n_robots=4)
        initial, final = generate_swap_positions(config)

        assert initial.shape == (4, 2)
        assert final.shape == (4, 2)

    def test_swap_property(self):
        """Robots should swap to opposite side of circle."""
        config = get_test_config(n_robots=4)
        initial, final = generate_swap_positions(config)

        env = config.problem.environment
        cx = (env.pos_x_min + env.pos_x_max) / 2
        cy = (env.pos_y_min + env.pos_y_max) / 2
        center = np.array([cx, cy])

        # Each robot's final position should be on opposite side of center
        for i in range(4):
            init_vec = initial[i] - center
            final_vec = final[i] - center
            # Vectors should be opposite (dot product negative)
            assert np.dot(init_vec, final_vec) < 0


class TestLinePositions:
    def test_correct_shape(self):
        config = get_test_config(n_robots=5)
        initial, final = generate_line_positions(config)

        assert initial.shape == (5, 2)
        assert final.shape == (5, 2)

    def test_reverse_order(self):
        """Final positions should be reversed order of initial."""
        config = get_test_config(n_robots=5)
        initial, final = generate_line_positions(config)

        np.testing.assert_array_almost_equal(initial, final[::-1])


class TestRandomObstacles:
    def test_correct_shape(self):
        config = get_test_config(n_robots=3)
        obstacles = generate_random_obstacles(config, n_obstacles=5, seed=42)

        assert obstacles.shape == (5, 2)

    def test_within_bounds(self):
        config = get_test_config()
        obstacles = generate_random_obstacles(config, n_obstacles=10, seed=42)

        env = config.problem.environment
        R = config.problem.min_distance
        assert np.all(obstacles[:, 0] >= env.pos_x_min + R)
        assert np.all(obstacles[:, 0] <= env.pos_x_max - R)
        assert np.all(obstacles[:, 1] >= env.pos_y_min + R)
        assert np.all(obstacles[:, 1] <= env.pos_y_max - R)

    def test_avoids_robot_positions(self):
        config = get_test_config(n_robots=3)
        initial, final = generate_random_positions(config, seed=42)
        obstacles = generate_random_obstacles(
            config, n_obstacles=5, seed=42,
            initial_positions=initial, final_positions=final
        )

        R = config.problem.min_distance
        clearance = 2 * R + 0.1

        # Check obstacles don't overlap with robot positions
        for obs in obstacles:
            for pos in initial:
                assert np.linalg.norm(obs - pos) >= clearance - 0.01
            for pos in final:
                assert np.linalg.norm(obs - pos) >= clearance - 0.01

    def test_reproducible_with_seed(self):
        config = get_test_config()

        obs1 = generate_random_obstacles(config, n_obstacles=5, seed=123)
        obs2 = generate_random_obstacles(config, n_obstacles=5, seed=123)

        np.testing.assert_array_equal(obs1, obs2)
