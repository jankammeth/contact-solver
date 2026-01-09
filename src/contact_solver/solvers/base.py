"""Base solver class for trajectory optimization."""

from abc import ABC, abstractmethod

import numpy as np

from ..config import Config


class Solver(ABC):
    """Abstract base class for path planning solvers."""

    def __init__(self, config: Config):
        self.config = config

        # Problem parameters
        self.N = config.problem.n_robots
        self.h = config.problem.timestep
        self.R = config.problem.min_distance

        # Time horizon can be None (auto-compute in set_final_states)
        self.T = config.problem.time_horizon
        self.K = config.problem.n_timesteps if self.T is not None else None

        # Multiplier for auto time horizon (>1.0 gives more time)
        self.T_MULTIPLIER = 1.0

        # Environment bounds
        self.pos_min = config.problem.environment.pos_min
        self.pos_max = config.problem.environment.pos_max

        # Dynamics limits
        self.vel_min = config.problem.dynamics.vel_min
        self.vel_max = config.problem.dynamics.vel_max
        self.acc_min = config.problem.dynamics.acc_min
        self.acc_max = config.problem.dynamics.acc_max
        self.jerk_min = config.problem.dynamics.jerk_min
        self.jerk_max = config.problem.dynamics.jerk_max

        # State storage
        self.initial_positions = None
        self.initial_velocities = None
        self.final_positions = None
        self.final_velocities = None
        self.trajectories = None

    def set_initial_states(self, positions, velocities=None):
        """Set initial states for all robots."""
        if velocities is None:
            velocities = np.zeros((self.N, 2))

        self.initial_positions = np.asarray(positions).flatten()
        self.initial_velocities = np.asarray(velocities).flatten()

        assert len(self.initial_positions) == len(self.initial_velocities) == 2 * self.N

    def set_final_states(self, positions, velocities=None):
        """Set final states for all robots."""
        if velocities is None:
            velocities = np.zeros((self.N, 2))

        self.final_positions = np.asarray(positions).flatten()
        self.final_velocities = np.asarray(velocities).flatten()

        assert len(self.final_positions) == len(self.final_velocities) == 2 * self.N

        # Auto-compute time horizon if not set
        if self.T is None:
            if self.initial_positions is None:
                raise ValueError("Call set_initial_states() before set_final_states()")
            self.T = self._auto_time_horizon()
            self.K = int(round(self.T / self.h))

    def _auto_time_horizon(self) -> float:
        """Compute time horizon based on max displacement.

        Uses cubic polynomial kinematics: for zero boundary velocities,
        max_velocity = (3/2) * distance / T.
        """
        initial = self.initial_positions.reshape(-1, 2)
        final = self.final_positions.reshape(-1, 2)

        distances = np.linalg.norm(final - initial, axis=1)
        max_distance = distances.max()

        # Minimum time for cubic polynomial with zero boundary velocities
        T = (1.5 * max_distance) / self.vel_max
        T *= self.T_MULTIPLIER

        # Round up to nearest multiple of timestep
        K = int(np.ceil(T / self.h))
        T = K * self.h

        return float(T)

    @abstractmethod
    def generate_trajectories(self, **kwargs):
        """Generate collision-free trajectories. Must be implemented by subclass."""
        pass
