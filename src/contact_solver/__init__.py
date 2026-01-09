"""Contact Solver - Contact-based trajectory optimization for multi-robot collision avoidance."""

from .collision import CollisionReport, detect_collisions
from .config import Config, get_default_config, get_small_config, load_config
from .scenarios import generate_random_positions, generate_swap_positions, generate_line_positions
from .solvers import ContactSolver, LiftedSCP, Solver
from .viz import visualize_trajectories, visualize_comparison, visualize_pairwise_distances

__all__ = [
    # Config
    "Config",
    "load_config",
    "get_default_config",
    "get_small_config",
    # Solvers
    "Solver",
    "ContactSolver",
    "LiftedSCP",
    # Collision detection
    "CollisionReport",
    "detect_collisions",
    # Scenario generation
    "generate_random_positions",
    "generate_swap_positions",
    "generate_line_positions",
    # Visualization
    "visualize_trajectories",
    "visualize_comparison",
    "visualize_pairwise_distances",
]
