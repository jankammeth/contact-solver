# Contact Solver

Comparison of two multi-robot trajectory optimization approaches:

- **LiftedSCP**: Sparse joint Sequential Convex Programming for dimension d >= 2,
  with held-acceleration trajectories and continuous-time acceptance checks
- **ContactSolver**: Analytical piecewise cubic solver with sequential contact insertion
- **VectorContactSolver**: Joint explicit-vector solver for spatial dimension d >= 2,
  with analytical banded Jacobians and continuous-time acceptance checks

See [joint LiftedSCP and paired comparisons](docs/lifted-scp.md) and
[general-dimensional SCI](docs/vector-sci.md) for the APIs, NPZ/YAML CLIs,
tolerances and validation results. Current Section 5.5 blockers are listed in the
[joint validation report](validation/lifted-dimension-validation.md).

## Installation

```bash
pip install -e .
```

## Usage

```bash
python scripts/compare_solvers.py --config small --seed 42
python scripts/sweep_compare.py --n-min 2 --n-max 10 --seeds 5
```

See `configs/` for configuration options.

## Testing

```bash
pytest tests/ -v
```
