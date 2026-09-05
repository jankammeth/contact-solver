"""Configuration schema for contact solver experiments."""

from dataclasses import dataclass, field
from pathlib import Path

from typing import Optional

import numpy as np
from omegaconf import MISSING, OmegaConf


@dataclass
class EnvironmentConfig:
    pos_x_min: Optional[float] = None
    pos_x_max: Optional[float] = None
    pos_y_min: Optional[float] = None
    pos_y_max: Optional[float] = None

    @property
    def pos_min(self) -> np.ndarray:
        return np.array([
            self.pos_x_min if self.pos_x_min is not None else -np.inf,
            self.pos_y_min if self.pos_y_min is not None else -np.inf,
        ])

    @property
    def pos_max(self) -> np.ndarray:
        return np.array([
            self.pos_x_max if self.pos_x_max is not None else np.inf,
            self.pos_y_max if self.pos_y_max is not None else np.inf,
        ])


@dataclass
class DynamicsConfig:
    vel_min: Optional[float] = None
    vel_max: Optional[float] = None
    acc_min: Optional[float] = None
    acc_max: Optional[float] = None
    jerk_min: Optional[float] = None
    jerk_max: Optional[float] = None


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
    # ── Detection: what counts as a collision ────────────────
    # A pair at distance d violates if d < min_distance - detection_tol.
    # SCP: buffer in discrete-time KDTree check (iteration + post-hoc)
    scp_detection_tol: float = 0.0
    # SCI: threshold for continuous-time violation search; also defines
    #      convergence (solver stops when no violations remain)
    sci_detection_tol: float = 0.0

    # ── Convergence: when has the iterate stabilized ─────────
    # SCP: stop when relative/absolute position change between iterations < this
    scp_convergence_rel: float = 0.0
    scp_convergence_abs: float = 0.0

    # ── Numerical precision: subroutine solve accuracy ───────
    # OSQP: QP subproblem precision inside SCP
    osqp_eps_abs: float = 0.0
    osqp_eps_rel: float = 0.0
    # fsolve: algebraic system precision inside SCI (0 = scipy default ~1.49e-8)
    fsolve_xtol: float = 0.0

    # ── Algorithm limits ─────────────────────────────────────
    scp_max_iterations: int = 15
    osqp_max_iter: int = 10000
    osqp_polish: bool = True
    osqp_warm_start: bool = True
    osqp_verbose: bool = False
    sci_max_contacts: int = 20
    sci_max_contacts_per_pair: int = 10
    sci_min_contact_gap: float = 0.0
    apply_velocity_bounds: bool = True
    timeout: float = 0.0

    # ── PrioritizedSCP-specific options ──────────────────────
    # If True (default, original behavior), the linearized obstacle-constraint
    # matrix passed to OSQP at each SCP iteration of PrioritizedSCP has each
    # row L2-normalized via `normalize_constraint`. If False, the raw matrix
    # is passed.
    #
    # Why this is a toggle: the constraint row at (k, m) has 2-norm
    #     sqrt( sum_{m'=0}^{k} (h^2 * (k+1-m'-0.5))^2 )  ∝  h^2 * k^(3/2)
    # which spans 3+ orders of magnitude across a K=200 trajectory. Row-
    # normalization gives OSQP a unit-norm view of every row but means
    # `osqp_eps_abs` becomes a normalized tolerance whose translation to
    # physical-distance error at a given k is (eps_abs * row_norm_at_k). At
    # late k this physical tolerance can exceed `scp_detection_tol`, causing
    # OSQP-converged iterates to register as infeasible downstream.
    # See the prioritized planning experiment write-up for the A/B evidence.
    #
    # This option does NOT affect the joint solvers (LiftedSCP, ContactSolver)
    # which use the same `normalize_constraint` helper -- those still
    # normalize, since their cost gradient pulls iterates off the constraint
    # boundary and the late-k tolerance issue does not manifest.
    prio_scp_normalize_obstacles: bool = True

    # ── Performance optimizations ────────────────────────────
    # Analytical Jacobian for fsolve / scipy.optimize.root
    use_analytical_jacobian: bool = False
    # Root-finding method: "fsolve", "hybr", "lm"
    root_method: str = "fsolve"
    # Vectorized residual / segment building (numpy arrays instead of dicts)
    vectorize_residuals: bool = False
    # Vectorized post-processing (min_dist computation)
    vectorize_postprocess: bool = False


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
        Path(__file__).parent.parent.parent / "configs" / f"{name}.yaml",
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
