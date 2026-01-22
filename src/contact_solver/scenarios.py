"""Scenario generation for solver comparison."""

import numpy as np

from .config import Config


def generate_random_positions(
    config: Config,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    elif config.random_seed is not None:
        np.random.seed(config.random_seed)

    N = config.problem.n_robots
    env = config.problem.environment
    scenario = config.problem.scenario

    cx = (env.pos_x_min + env.pos_x_max) / 2
    cy = (env.pos_y_min + env.pos_y_max) / 2
    center = np.array([cx, cy])

    radius = scenario.cluster_radius
    spacing = scenario.spacing

    initial_positions = _generate_cluster(N, center, radius, spacing)
    final_positions = _generate_cluster(N, center, radius, spacing)

    return initial_positions, final_positions


def _generate_cluster(
    n: int,
    center: np.ndarray,
    radius: float,
    min_spacing: float,
    max_attempts: int = 1000,
) -> np.ndarray:
    positions = []

    for _ in range(n):
        for attempt in range(max_attempts):
            angle = np.random.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(np.random.uniform(0, 1))
            pos = center + r * np.array([np.cos(angle), np.sin(angle)])

            valid = True
            for existing in positions:
                if np.linalg.norm(pos - existing) < min_spacing:
                    valid = False
                    break

            if valid:
                positions.append(pos)
                break
        else:
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
    if seed is not None:
        np.random.seed(seed)

    N = config.problem.n_robots
    env = config.problem.environment
    scenario = config.problem.scenario

    cx = (env.pos_x_min + env.pos_x_max) / 2
    cy = (env.pos_y_min + env.pos_y_max) / 2

    radius = scenario.cluster_radius

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_positions = np.zeros((N, 2))
    final_positions = np.zeros((N, 2))

    for i, angle in enumerate(angles):
        initial_positions[i] = [cx + radius * np.cos(angle), cy + radius * np.sin(angle)]
        final_positions[i] = [cx - radius * np.cos(angle), cy - radius * np.sin(angle)]

    return initial_positions, final_positions


def generate_random_obstacles(
    config: Config,
    n_obstacles: int,
    seed: int | None = None,
    initial_positions: np.ndarray | None = None,
    final_positions: np.ndarray | None = None,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    elif config.random_seed is not None:
        np.random.seed(config.random_seed)

    env = config.problem.environment
    R = config.problem.min_distance

    avoid_positions = []
    if initial_positions is not None:
        avoid_positions.extend(initial_positions.tolist())
    if final_positions is not None:
        avoid_positions.extend(final_positions.tolist())

    clearance = 2 * R + 0.1

    obstacles = []
    max_attempts = 1000

    for _ in range(n_obstacles):
        for attempt in range(max_attempts):
            x = np.random.uniform(env.pos_x_min + R, env.pos_x_max - R)
            y = np.random.uniform(env.pos_y_min + R, env.pos_y_max - R)
            pos = np.array([x, y])

            valid = True
            for avoid_pos in avoid_positions:
                if np.linalg.norm(pos - np.array(avoid_pos)) < clearance:
                    valid = False
                    break

            if valid:
                for obs in obstacles:
                    if np.linalg.norm(pos - obs) < 2 * R:
                        valid = False
                        break

            if valid:
                obstacles.append(pos)
                break

    return np.array(obstacles) if obstacles else np.zeros((0, 2))
