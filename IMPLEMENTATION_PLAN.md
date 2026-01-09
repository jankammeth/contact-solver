# Contact Solver Repository - Implementation Plan

## Goal
A lean repository to compare the **ContactSolver** (analytical piecewise cubic, sequential contact insertion) against **LiftedSCP** (QP-based double-integrator) for multi-robot path planning.

---

## Target Structure

```
contact_solver/
├── pyproject.toml
├── README.md
├── src/contact_solver/
│   ├── __init__.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base.py                  # Solver ABC
│   │   ├── contact_solver.py        # Main contribution
│   │   ├── lifted_scp.py            # QP baseline
│   │   ├── cubic_utils.py           # Shared polynomial math
│   │   ├── velocity_arcs.py         # Velocity bound enforcement
│   │   ├── osqp_utils.py            # QP helpers
│   │   └── constraints/
│   │       ├── __init__.py
│   │       ├── collision.py         # Linearized collision constraints
│   │       └── dynamics.py          # Dynamics constraint builders
│   ├── collision.py                 # KDTree-based detection
│   ├── config.py                    # Flattened OmegaConf config
│   ├── scenarios.py                 # Random position generation
│   └── viz.py                       # Minimal static plotting
├── scripts/
│   ├── compare_solvers.py           # Single-problem comparison
│   └── sweep_compare.py             # Multi-scale sweeps
├── configs/
│   ├── default.yaml
│   ├── small.yaml
│   └── sweep.yaml
└── tests/
    ├── test_contact_solver.py
    └── test_lifted_scp.py
```

---

## Phase 1: Project Skeleton & Configuration

### 1.1 Create project files
- [x] `pyproject.toml` with minimal dependencies:
  - numpy, scipy, matplotlib, osqp, omegaconf, pandas (for sweep results)
- [x] `README.md` with project overview

### 1.2 Flattened Configuration (`config.py`)
Create a simplified config schema (no hierarchical solver params, no wandb):

```python
@dataclass
class ProblemConfig:
    n_robots: int
    time_horizon: float | None  # None = auto
    timestep: float
    min_distance: float
    vel_max: float
    acc_max: float
    jerk_max: float
    pos_min: tuple[float, float]
    pos_max: tuple[float, float]

@dataclass
class SolverConfig:
    scp_tolerance_rel: float = 1e-3
    scp_tolerance_abs: float = 1e-3
    scp_max_iterations: int = 15
    osqp_max_iter: int = 10000
    osqp_eps_abs: float = 1e-3
    osqp_eps_rel: float = 1e-3
    # ContactSolver specific
    max_contacts: int = 20
    max_contacts_per_pair: int = 10
    apply_velocity_bounds: bool = True

@dataclass
class Config:
    name: str
    problem: ProblemConfig
    solver: SolverConfig
    random_seed: int | None = None
```

### 1.3 YAML configs
- [x] `default.yaml` - 10 robots, reasonable defaults
- [x] `small.yaml` - 5 robots for quick testing  
- [x] `sweep.yaml` - base config for scaling sweeps

**Deliverable:** Importable `from contact_solver import Config, load_config`

---

## Phase 2: Core Solver Infrastructure

### 2.1 Base Solver (`solvers/base.py`)
Copy and simplify from `path_planning`:
- [x] Remove hierarchical-specific code
- [x] Keep: `set_initial_states`, `set_final_states`, `generate_trajectories`
- [x] Keep: `_auto_time_horizon()` logic

### 2.2 Constraint Builders (`solvers/constraints/`)
Copy directly (these are clean and modular):
- [x] `dynamics.py` - Keep `build_dynamics_constraints` for double-integrator only
- [x] `collision.py` - Keep `build_collision_constraints`, `normalize_constraint`

### 2.3 OSQP Utilities (`solvers/osqp_utils.py`)
Copy directly:
- [x] `OSQPSettings` dataclass
- [x] `solve_qp()` function with version compatibility

### 2.4 Cubic Utilities (`solvers/cubic_utils.py`)
Copy directly - used by ContactSolver:
- [x] `cubic_coefficients`, `eval_cubic`, `eval_cubic_vel`, `eval_cubic_acc`
- [x] `find_distance_minima`

### 2.5 Velocity Arcs (`solvers/velocity_arcs.py`)
Copy directly - essential for ContactSolver feasibility:
- [x] `apply_velocity_bounds_to_trajectories`
- [x] All supporting functions

**Deliverable:** All solver infrastructure compiles and imports

---

## Phase 3: Solvers

### 3.1 LiftedSCP (`solvers/lifted_scp.py`)
Copy and adapt:
- [x] Update imports to use local modules
- [x] Remove obstacle support (simplify)
- [x] Remove warm_start_trajectories from hierarchical usage
- [x] Keep core SCP loop with collision constraints

### 3.2 ContactSolver (`solvers/contact_solver.py`)
Copy and adapt:
- [x] Update imports
- [x] Keep full sequential contact insertion logic
- [x] Keep velocity arc integration
- [x] Ensure `_Contact` dataclass is included

**Deliverable:** Both solvers run on simple test cases

---

## Phase 4: Supporting Modules

### 4.1 Collision Detection (`collision.py`)
Extract from `spatial_hash_kdtree.py`:
- [x] `CollisionReport` dataclass
- [x] `detect_collisions_kdtree()` - main entry point
- [x] Helper functions for KDTree queries
- [x] Remove obstacle support (or keep minimal)

### 4.2 Scenario Generation (`scenarios.py`)
Minimal random position generation:
- [x] `generate_random_positions(config) -> (initial, final)`
- [x] Simple clustered distribution (no SVG, no complex shapes)
- [x] Ensure minimum spacing between robots

### 4.3 Visualization (`viz.py`)
Extract only what compare_solvers needs:
- [x] `visualize_comparison()` - main comparison plot
- [x] `visualize_trajectories()` - simple trajectory plot
- [x] `visualize_pairwise_distances()` - distance over time
- [x] Remove animation code

**Deliverable:** Full pipeline works: config → scenario → solve → visualize

---

## Phase 5: Comparison Scripts

### 5.1 Single Comparison (`scripts/compare_solvers.py`)
Simplify from original:
- [x] Compare only ContactSolver vs LiftedSCP
- [x] Remove single-integrator branches
- [x] Print: timing, min distance, velocity bounds, cost
- [x] Generate comparison visualization

### 5.2 Sweep Comparison (`scripts/sweep_compare.py`)
Adapt from original:
- [x] Scale N from 2 to 20 (configurable)
- [x] Multiple seeds per N
- [x] Consistent congestion scaling (workspace ~ sqrt(N))
- [x] Output CSV with metrics:
  - solve_time, scp_iterations, osqp_iters
  - min_distance, feasible, cost
- [x] No WandB integration

**Deliverable:** Working comparison pipeline with CSV output

---

## Phase 6: Testing & Polish

### 6.1 Unit Tests
- [x] `test_config.py` - Config loading/overrides
- [x] `test_solver_infrastructure.py` - Constraint builders, utilities
- [x] `test_solvers.py` - Individual solver tests
- [x] `test_scenarios.py` - Scenario generation tests

### 6.2 Integration Tests
- [x] `test_integration.py` - Full pipeline tests
- [x] Test both solvers on same scenarios
- [x] Verify collision-free trajectories
- [x] Test edge cases (2 robots, head-on, parallel)

### 6.3 Documentation
- [x] Complete README with usage examples
- [x] Docstrings on public functions
- [x] Configuration reference in README

### 6.4 Polish
- [x] conftest.py with shared fixtures
- [x] Type hints throughout
- [x] Clean imports in __init__.py

**Deliverable:** All tests pass, comprehensive documentation

---

## Dependencies (pyproject.toml)

```toml
dependencies = [
    "numpy>=1.23",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "osqp>=0.6",
    "omegaconf>=2.2.0",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6", "black>=24"]
```

---

## Files to Copy (with modifications)

| Source | Destination | Modifications |
|--------|-------------|---------------|
| `solvers/base.py` | `solvers/base.py` | Remove hierarchical params |
| `solvers/contact_solver.py` | `solvers/contact_solver.py` | Update imports |
| `solvers/lifted_scp.py` | `solvers/lifted_scp.py` | Update imports, remove obstacles |
| `solvers/cubic_utils.py` | `solvers/cubic_utils.py` | Direct copy |
| `solvers/velocity_arcs.py` | `solvers/velocity_arcs.py` | Direct copy |
| `solvers/osqp_utils.py` | `solvers/osqp_utils.py` | Update imports |
| `solvers/constraints/dynamics.py` | `solvers/constraints/dynamics.py` | Remove SI support |
| `solvers/constraints/collision.py` | `solvers/constraints/collision.py` | Remove SI support |
| `utils/spatial_hash_kdtree.py` | `collision.py` | Rename, simplify |
| `config/schemas.py` | `config.py` | Flatten dramatically |
| `cli/compare_solvers.py` | `scripts/compare_solvers.py` | Remove SI, simplify |
| `cli/sweep_compare_solvers.py` | `scripts/sweep_compare.py` | Remove WandB, simplify |
| `viz/trajectory_viz.py` | `viz.py` | Extract only needed functions |

---

## Estimated Effort

| Phase | Effort | Notes |
|-------|--------|-------|
| Phase 1 | 1 hour | Project setup, config |
| Phase 2 | 2 hours | Constraint builders, utilities |
| Phase 3 | 2 hours | Solver adaptation |
| Phase 4 | 1.5 hours | Supporting modules |
| Phase 5 | 1.5 hours | Comparison scripts |
| Phase 6 | 1 hour | Testing, polish |

**Total: ~9 hours**

---

## Success Criteria

1. `python scripts/compare_solvers.py` runs and produces comparison plot
2. `python scripts/sweep_compare.py --n-max 10 --seeds 5` produces CSV
3. Both solvers produce collision-free trajectories on test cases
4. Repository has < 15 Python files (excluding tests)
5. No dependencies on `path_planning` package
