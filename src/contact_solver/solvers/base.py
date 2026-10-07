"""Base solver class for trajectory optimization."""

from abc import ABC, abstractmethod

import numpy as np

from ..config import Config


class Solver(ABC):
    # Opt in to the general-dimensional base endpoint setters. ContactSolver
    # has its own setters. Legacy and prioritized solvers retain planar guards.
    supports_nd_states = False

    def __init__(self, config: Config):
        self.config = config

        self.N = config.problem.n_robots
        self.d = config.problem.dimension
        if not isinstance(self.d, int) or self.d < 2:
            raise ValueError("problem.dimension must be an integer >= 2")
        self.h = config.problem.timestep
        self.R = config.problem.min_distance

        self.T = config.problem.time_horizon
        self.K = config.problem.n_timesteps if self.T is not None else None

        self.T_MULTIPLIER = 1.0

        # Environment limits (None = unconstrained)
        self.pos_min, self.pos_max = config.problem.environment.bounds(self.d)

        # Dynamics limits (None = unconstrained)
        self.vel_min = config.problem.dynamics.vel_min
        self.vel_max = config.problem.dynamics.vel_max
        self.acc_min = config.problem.dynamics.acc_min
        self.acc_max = config.problem.dynamics.acc_max
        self.jerk_min = config.problem.dynamics.jerk_min
        self.jerk_max = config.problem.dynamics.jerk_max

        self.initial_positions = None
        self.initial_velocities = None
        self.final_positions = None
        self.final_velocities = None
        self.trajectories = None

    def set_initial_states(self, positions, velocities=None):
        self.initial_positions, self.initial_velocities = self._endpoint_states(
            positions, velocities
        )

    def set_final_states(self, positions, velocities=None):
        self.final_positions, self.final_velocities = self._endpoint_states(positions, velocities)

        if self.T is None:
            if self.initial_positions is None:
                raise ValueError("Call set_initial_states() before set_final_states()")
            self.T = self._auto_time_horizon()
            self.K = int(round(self.T / self.h))

    def _endpoint_states(self, positions, velocities):
        if self.d != 2 and not self.supports_nd_states:
            raise ValueError(f"{type(self).__name__} supports only dimension=2")
        positions = np.asarray(positions, dtype=float)
        velocities = (
            np.zeros_like(positions) if velocities is None else np.asarray(velocities, dtype=float)
        )
        shapes = ((self.N, self.d), (self.N * self.d,))
        if positions.shape not in shapes or velocities.shape not in shapes:
            raise ValueError(
                f"Expected endpoint positions/velocities of shape ({self.N}, {self.d}) or flattened"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("Endpoint states must be finite")
        return positions.ravel().copy(), velocities.ravel().copy()

    def _auto_time_horizon(self) -> float:
        initial = self.initial_positions.reshape(-1, self.d)
        final = self.final_positions.reshape(-1, self.d)

        distances = np.linalg.norm(final - initial, axis=1)
        max_distance = distances.max()

        v_max = self.vel_max if self.vel_max is not None else 2.0
        T = (1.5 * max_distance) / v_max
        T *= self.T_MULTIPLIER

        K = int(np.ceil(T / self.h))
        T = K * self.h

        return float(T)

    @abstractmethod
    def generate_trajectories(self, **kwargs):
        pass
