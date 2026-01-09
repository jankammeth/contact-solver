"""Scenario generation for solver comparison."""

import numpy as np

from .config import Config


def generate_random_positions(
    config: Config,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random initial and final positions.

    Generates two clusters of positions (initial and final) with minimum
    spacing between robots.

    Args:
        config: Configuration with problem parameters
        seed: Random seed (overrides config.random_seed if provided)

    Returns:
        Tuple of (initial_positions, final_positions), each (N, 2) arrays
    """
    if seed is not None:
        np.random.seed(seed)
    elif config.random_seed is not None:
        np.random.seed(config.random_seed)

    N = config.problem.n_robots
    env = config.problem.environment
    scenario = config.problem.scenario

    # Workspace bounds
    x_min, x_max = env.pos_x_min, env.pos_x_max
    y_min, y_max = env.pos_y_min, env.pos_y_max

    # Cluster parameters
    radius = scenario.cluster_radius
    margin = scenario.cluster_margin
    spacing = scenario.spacing

    # Generate initial cluster (bottom-left region)
    initial_center = np.array([x_min + margin + radius, y_min + margin + radius])
    initial_positions = _generate_cluster(N, initial_center, radius, spacing)

    # Generate final cluster (top-right region)
    final_center = np.array([x_max - margin - radius, y_max - margin - radius])
    final_positions = _generate_cluster(N, final_center, radius, spacing)

    return initial_positions, final_positions


def _generate_cluster(
    n: int,
    center: np.ndarray,
    radius: float,
    min_spacing: float,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Generate n positions within a circular cluster with minimum spacing."""
    positions = []

    for _ in range(n):
        for attempt in range(max_attempts):
            # Random position within circle
            angle = np.random.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(np.random.uniform(0, 1))
            pos = center + r * np.array([np.cos(angle), np.sin(angle)])

            # Check spacing with existing positions
            valid = True
            for existing in positions:
                if np.linalg.norm(pos - existing) < min_spacing:
                    valid = False
                    break

            if valid:
                positions.append(pos)
                break
        else:
            # Fallback: place on grid if random fails
            idx = len(positions)
            grid_size = int(np.ceil(np.sqrt(n)))
            row, col = idx // grid_size, idx % grid_size
            offset = np.array([col - grid_size / 2, row - grid_size / 2]) * min_spacing
            positions.append(center + offset)

    return np.array(positions)


def generate_swap_positions(
    config: Config,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate positions where robots swap places (head-on collision scenario).

    Args:
        config: Configuration with problem parameters
        seed: Random seed

    Returns:
        Tuple of (initial_positions, final_positions)
    """
    if seed is not None:
        np.random.seed(seed)

    N = config.problem.n_robots
    env = config.problem.environment

    cx = (env.pos_x_min + env.pos_x_max) / 2
    cy = (env.pos_y_min + env.pos_y_max) / 2
    spread = min(env.pos_x_max - env.pos_x_min, env.pos_y_max - env.pos_y_min) * 0.3

    # Place robots in a circle, swap to opposite side
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_positions = np.zeros((N, 2))
    final_positions = np.zeros((N, 2))

    for i, angle in enumerate(angles):
        initial_positions[i] = [cx + spread * np.cos(angle), cy + spread * np.sin(angle)]
        # Swap to opposite side
        final_positions[i] = [cx - spread * np.cos(angle), cy - spread * np.sin(angle)]

    return initial_positions, final_positions


def generate_line_positions(
    config: Config,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate positions on a line that reverse order.

    Args:
        config: Configuration with problem parameters
        seed: Random seed

    Returns:
        Tuple of (initial_positions, final_positions)
    """
    N = config.problem.n_robots
    env = config.problem.environment
    spacing = config.problem.min_distance * 1.5

    cx = (env.pos_x_min + env.pos_x_max) / 2
    cy = (env.pos_y_min + env.pos_y_max) / 2

    # Line of robots
    total_width = (N - 1) * spacing
    x_start = cx - total_width / 2

    initial_positions = np.array([[x_start + i * spacing, cy] for i in range(N)])
    final_positions = initial_positions[::-1].copy()  # Reverse order

    return initial_positions, final_positions
