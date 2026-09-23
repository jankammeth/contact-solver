# Contact Solver

Comparison of two multi-robot trajectory optimization approaches:

- **LiftedSCP**: QP-based Sequential Convex Programming
- **ContactSolver**: Analytical piecewise cubic solver with sequential contact insertion
- **VectorContactSolver**: Joint explicit-vector solver for spatial dimension d >= 2,
  with analytical banded Jacobians and continuous-time acceptance checks

See [general-dimensional SCI](docs/vector-sci.md) for the API, NPZ/YAML CLI,
tolerances, validation results, and remaining Section 5.5 blockers.

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
