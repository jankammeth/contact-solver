# Contact Solver

Comparison of two multi-robot trajectory optimization approaches:

- **LiftedSCP**: Sparse joint Sequential Convex Programming for dimension d >= 2,
  with held-acceleration trajectories and continuous-time acceptance checks
- **ContactSolver**: Joint explicit-vector solver for spatial dimension d >= 2,
  with analytical banded Jacobians and continuous-time acceptance checks.
- **PrioritizedSCI / PrioritizedSCP**: Separate sequential algorithms with a shared
  saved priority order, constructed outside their solve timers

See [joint LiftedSCP and paired comparisons](docs/lifted-scp.md) and
[general-dimensional SCI](docs/vector-sci.md) for the APIs, NPZ/YAML CLIs,
tolerances and validation results. Current Section 5.5 blockers are listed in the
[joint validation report](validation/lifted-dimension-validation.md).

## Installation

```bash
pip install -e .
```

Installation builds the continuous collision checker from C source when a C
compiler is available (Xcode Command Line Tools on macOS). Without it, the solver
uses the Python reference checker and warns that native runtimes are unavailable.
`metrics.collision_backend` identifies the checker used. Compilation is outside
the solve timer. Joint and prioritized SCI enable condensed matching recovery by
default; `sci_reaction_recovery=False` disables it for comparisons.

## Usage

```bash
compare-solvers --config small --seed 42
sweep-n --n-min 2 --n-max 10 --seeds 5
```

See `configs/` for configuration options.

## Testing

```bash
pytest tests/ -v
```
