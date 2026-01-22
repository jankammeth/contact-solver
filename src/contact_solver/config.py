"""Configuration schema for contact solver experiments."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import MISSING, OmegaConf


@dataclass
class EnvironmentConfig:
    pos_x_min: float = 0.0
    pos_x_max: float = 20.0
    pos_y_min: float = 0.0
    pos_y_max: float = 20.0

    @property
    def pos_min(self) -> np.ndarray:
        return np.array([self.pos_x_min, self.pos_y_min])

    @property
    def pos_max(self) -> np.ndarray:
        return np.array([self.pos_x_max, self.pos_y_max])


@dataclass
class DynamicsConfig:
    vel_min: float = -2.0
    vel_max: float = 2.0
    acc_min: float = -15.0
    acc_max: float = 15.0
    jerk_min: float = -20.0
    jerk_max: float = 20.0


@dataclass
class ScenarioConfig:
    cluster_radius: float = 5.0
    cluster_margin: float = 1.5
    spacing: float = 1.0


@dataclass
class ProblemConfig:
    n_robots: int = MISSING
    time_horizon: float | None = None
    timestep: float = MISSING
    min_distance: float = MISSING
    robot_radius: float = 0.4

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    @property
    def n_timesteps(self) -> int | None:
        if self.time_horizon is None:
            return None
        return int(self.time_horizon / self.timestep)


@dataclass
class SolverConfig:
    scp_tolerance_rel: float = 1e-3
    scp_tolerance_abs: float = 1e-3
    scp_max_iterations: int = 15

    osqp_max_iter: int = 10000
    osqp_eps_abs: float = 1e-3
    osqp_eps_rel: float = 1e-3
    osqp_polish: bool = True
    osqp_warm_start: bool = True
    osqp_verbose: bool = False

    max_contacts: int = 20
    max_contacts_per_pair: int = 10
    apply_velocity_bounds: bool = True


@dataclass
class VisualizationConfig:
    show_plots: bool = True
    save_plots: bool = False
    output_dir: str = "outputs"
    dpi: int = 150
    figsize: tuple[int, int] = (10, 10)


@dataclass
class Config:
    name: str = MISSING
    description: str | None = None
    random_seed: int | None = None

    problem: ProblemConfig = MISSING
    solver: SolverConfig = field(default_factory=SolverConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        schema = OmegaConf.structured(cls)
        yaml_config = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, yaml_config)
        return OmegaConf.to_object(merged)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = OmegaConf.structured(self)
        OmegaConf.save(cfg, path)

    def override(self, **kwargs) -> "Config":
        cfg = OmegaConf.structured(self)
        for key, value in kwargs.items():
            OmegaConf.update(cfg, key, value, merge=False)
        return OmegaConf.to_object(cfg)

    def to_dict(self) -> dict:
        cfg = OmegaConf.structured(self)
        return OmegaConf.to_container(cfg, resolve=True)


def load_config(name: str) -> Config:
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "configs" / f"{name}.yaml",
        Path("configs") / f"{name}.yaml",
    ]

    for path in possible_paths:
        if path.exists():
            return Config.from_yaml(path)

    raise FileNotFoundError(f"Config '{name}' not found. Tried: {possible_paths}")


def get_default_config() -> Config:
    return Config(
        name="default",
        description="Default configuration",
        problem=ProblemConfig(
            n_robots=10,
            time_horizon=10.0,
            timestep=0.2,
            min_distance=0.8,
        ),
    )


def get_small_config() -> Config:
    return Config(
        name="small",
        description="Small configuration for quick testing",
        random_seed=42,
        problem=ProblemConfig(
            n_robots=5,
            time_horizon=5.0,
            timestep=0.2,
            min_distance=0.8,
        ),
    )
