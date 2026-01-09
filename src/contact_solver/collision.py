"""KDTree-based collision detection."""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

COLLISION_DETECTION_TOLERANCE = 1e-2


@dataclass
class CollisionReport:
    """Result of collision detection."""

    robot_robot: list[tuple[int, int, int, float]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)

    @property
    def all_satisfied(self) -> bool:
        return len(self.robot_robot) == 0

    @property
    def worst_violation(self) -> float:
        all_dists = [v[3] for v in self.robot_robot]
        return max(0.0, -min(all_dists)) if all_dists else 0.0

    @property
    def n_violations(self) -> int:
        return len(self.robot_robot)


def detect_collisions(
    positions: list[np.ndarray] | np.ndarray,
    min_distance: float,
    robot_timeframes: list[list[int]] | None = None,
) -> CollisionReport:
    """Detect all robot-robot collisions using scipy's cKDTree.

    Args:
        positions: Either (N, K, 2) array or list of (K_i, 2) arrays per robot
        min_distance: Minimum allowed distance
        robot_timeframes: List of [k_start, k_end] per robot (None = full trajectory)

    Returns:
        CollisionReport with all violations
    """
    report = CollisionReport()
    t_start = time.perf_counter()

    # Convert to list format
    if isinstance(positions, np.ndarray):
        positions_list = [positions[i] for i in range(positions.shape[0])]
        is_uniform = True
    else:
        positions_list = positions
        lengths = [len(p) for p in positions]
        is_uniform = len(set(lengths)) == 1 and robot_timeframes is None

    # Detect collisions
    if is_uniform and robot_timeframes is None:
        trajectories = np.array(positions_list)
        report.robot_robot = _detect_all_timesteps(trajectories, min_distance)
    else:
        report.robot_robot = _detect_with_timeframes(
            positions_list, min_distance, robot_timeframes
        )

    report.timing["total"] = time.perf_counter() - t_start
    return report


def _detect_all_timesteps(
    trajectories: np.ndarray,
    min_distance: float,
) -> list[tuple[int, int, int, float]]:
    """Detect collisions across all timesteps for uniform trajectories."""
    N, K, _ = trajectories.shape
    threshold = min_distance - COLLISION_DETECTION_TOLERANCE

    all_violations = []

    for k in range(K):
        positions = trajectories[:, k, :]
        tree = cKDTree(positions)
        pairs = tree.query_pairs(threshold, output_type="ndarray")

        if len(pairs) == 0:
            continue

        pos_i = positions[pairs[:, 0]]
        pos_j = positions[pairs[:, 1]]
        distances = np.linalg.norm(pos_i - pos_j, axis=1)

        for idx in range(len(pairs)):
            i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
            all_violations.append((k, i, j, float(distances[idx])))

    return all_violations


def _detect_with_timeframes(
    positions: list[np.ndarray],
    min_distance: float,
    robot_timeframes: list[list[int]] | None,
) -> list[tuple[int, int, int, float]]:
    """Detect collisions with variable robot timeframes."""
    from collections import defaultdict

    threshold = min_distance - COLLISION_DETECTION_TOLERANCE

    if robot_timeframes is None:
        robot_timeframes = [[0, len(positions[i])] for i in range(len(positions))]

    # Build schedule: timestep -> [(robot_id, k_local), ...]
    schedule = defaultdict(list)
    for i, (start, end) in enumerate(robot_timeframes):
        for k_local in range(end - start):
            schedule[start + k_local].append((i, k_local))

    all_violations = []

    for t in sorted(schedule.keys()):
        active = schedule[t]
        if len(active) < 2:
            continue

        robot_ids = np.array([r for r, _ in active])
        pos_array = np.array([positions[r][k] for r, k in active])

        tree = cKDTree(pos_array)
        pairs = tree.query_pairs(threshold, output_type="ndarray")

        if len(pairs) == 0:
            continue

        pos_i = pos_array[pairs[:, 0]]
        pos_j = pos_array[pairs[:, 1]]
        distances = np.linalg.norm(pos_i - pos_j, axis=1)

        for idx in range(len(pairs)):
            ri = int(robot_ids[pairs[idx, 0]])
            rj = int(robot_ids[pairs[idx, 1]])
            if ri > rj:
                ri, rj = rj, ri
            all_violations.append((t, ri, rj, float(distances[idx])))

    return all_violations
