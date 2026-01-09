"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

from contact_solver.config import (
    Config,
    DynamicsConfig,
    EnvironmentConfig,
    ProblemConfig,
    ScenarioConfig,
    SolverConfig,
    VisualizationConfig,
)


@pytest.fixture
def small_config():
    """Small configuration for fast tests."""
    return Config(
        name="test_small",
        random_seed=42,
        problem=ProblemConfig(
            n_robots=2,
            time_horizon=4.0,  # More generous
            timestep=0.1,
            min_distance=0.5,  # Smaller
            robot_radius=0.25,
            environment=EnvironmentConfig(
                pos_x_min=0.0, pos_x_max=10.0, pos_y_min=0.0, pos_y_max=10.0
            ),
            dynamics=DynamicsConfig(
                vel_min=-3.0, vel_max=3.0, acc_min=-10.0, acc_max=10.0
            ),
            scenario=ScenarioConfig(),
        ),
        solver=SolverConfig(scp_max_iterations=20, max_contacts=10),
        visualization=VisualizationConfig(show_plots=False),
    )


@pytest.fixture
def medium_config():
    """Medium configuration for more thorough tests."""
    return Config(
        name="test_medium",
        random_seed=123,
        problem=ProblemConfig(
            n_robots=5,
            time_horizon=6.0,
            timestep=0.2,
            min_distance=0.5,
            robot_radius=0.25,
            environment=EnvironmentConfig(
                pos_x_min=0.0, pos_x_max=20.0, pos_y_min=0.0, pos_y_max=20.0
            ),
            dynamics=DynamicsConfig(
                vel_min=-3.0, vel_max=3.0, acc_min=-10.0, acc_max=10.0
            ),
            scenario=ScenarioConfig(cluster_radius=4.0),
        ),
        solver=SolverConfig(scp_max_iterations=20, max_contacts=15),
        visualization=VisualizationConfig(show_plots=False),
    )


@pytest.fixture
def head_on_positions():
    """Two robots in head-on collision course."""
    initial = np.array([[0.0, 0.0], [3.0, 0.0]])
    final = np.array([[3.0, 0.0], [0.0, 0.0]])
    return initial, final


@pytest.fixture
def parallel_positions():
    """Two robots moving in parallel (no collision)."""
    initial = np.array([[0.0, 0.0], [0.0, 5.0]])
    final = np.array([[5.0, 0.0], [5.0, 5.0]])
    return initial, final


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
