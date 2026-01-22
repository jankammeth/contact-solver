"""Contact Solver - Multi-robot trajectory optimization comparison."""

from .collision import CollisionReport, detect_collisions
from .config import Config, get_default_config, get_small_config, load_config
from .scenarios import (
    generate_random_positions,
    generate_swap_positions,
    generate_random_obstacles,
)
from .solvers import ContactSolver, LiftedSCP, Solver
from .viz import (
    visualize_trajectories,
    visualize_comparison,
    visualize_pairwise_distances,
    visualize_time_snapshots,
)

__all__ = [
    "Config",
    "load_config",
    "get_default_config",
    "get_small_config",
    "Solver",
    "ContactSolver",
    "LiftedSCP",
    "CollisionReport",
    "detect_collisions",
    "generate_random_positions",
    "generate_swap_positions",
    "generate_random_obstacles",
    "visualize_trajectories",
    "visualize_comparison",
    "visualize_pairwise_distances",
    "visualize_time_snapshots",
]
