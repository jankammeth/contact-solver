# General-dimensional joint SCI

`VectorContactSolver` implements the explicit-vector choice in manuscript Sections
4.2–4.4 for `N >= 2` and `d >= 2`. `ContactSolver` remains the planar angle/speed,
eliminated-mass regression reference. No prioritized planner was rewritten.

## Inputs and entry points

Set `problem.dimension` (default 2), and pass finite endpoint positions and optional
velocities of shape `(N, d)` or flattened length `N*d`. Missing velocities are zero.
The prescribed barycenter is the cubic Hermite interpolant of the endpoint averages.
Set a positive horizon for reproducible experiments; automatic horizon selection
uses the old distance-based heuristic generalized to all coordinates.

```python
from contact_solver import Config, VectorContactSolver
from contact_solver.config import ProblemConfig, SolverConfig

config = Config(
    name="example",
    problem=ProblemConfig(n_robots=2, dimension=3, time_horizon=6,
                          timestep=0.1, min_distance=1),
    solver=SolverConfig(sci_detection_tol=1e-6),
)
solver = VectorContactSolver(config)
solver.set_initial_states([[-2, -.2, .1], [2, .2, -.1]])
solver.set_final_states([[2, -.2, .1], [-2, .2, -.1]])
result = solver.generate_trajectories()
assert result["metrics"]["converged"], result["metrics"]
times, positions, velocities, accelerations = solver.sample(n=100)
```

`sample(times=...)` also accepts arbitrary ordered or unordered times within the
horizon. Position, velocity, and acceleration arrays have shape `(N, samples, d)`
when stacked. Sampling is postprocessing and never determines success.
`solver.trajectory` exposes segment coefficients, `solver.contacts` the physical
contact states, and `solver.system.jacobian(solver.solution)` the sparse analytical
Jacobian. `solver.solution` is the packed explicit-vector state.

The `vector-sci` entry point is installed from `pyproject.toml`. From this package
directory it can also be called with `PYTHONPATH=src python -m
contact_solver.cli.vector_sci`:

```sh
vector-sci --states endpoints.npz --config configs/vector_3d.yaml \
  --output metrics.json --trajectories trajectories.npz
vector-sci --smoke --output smoke.json
```

The input NPZ keys are `initial_positions`, `final_positions`, and optionally
`initial_velocities`, `final_velocities`. The output NPZ additionally stores
`times`, `contact_states`, and zero-based `contact_pairs`. The CLI exits 0 only
when every requested vector solve succeeds, otherwise 1; invalid inputs raise an
error. The smoke option runs only 16 cases with at most three robots, plus four
legacy planar comparisons. It is **not** the Section 5.5 campaign.

## Variables, equations, and linear algebra

Indices in code are zero based. A contact block is
`[t, n[d], v[d], alpha, r_j[d], v_j[d], ...]`, with the remaining `j` in increasing
order, excluding the relative coordinate of the second robot in the pair.
The stored relative coordinate `r[j]` means `x[0] - x[j+1]`.
The contact pair satisfies `r_pq = R*n` and `r'_pq = v` by reconstruction.
Neither vector is projected or normalized inside the residual.

For each contact the residual stacks:

1. All relative acceleration jumps, right minus left: `d*(N-1)` equations.
2. All relative jerk jumps minus `alpha*R*c_j*n`: `d*(N-1)` equations,
   where `c_j = sigma_0 - sigma_j`, `sigma_p=1`, and `sigma_q=-1`.
3. `n @ n - 1` and `n @ v`: two equations.

For two robots the relative jerk coefficient is 2; the stored `alpha` is the
pairwise mass, not the doubled relative mass. It is never eliminated.

With `m` contacts, the intrinsic dimension is **`2*m*d*(N-1)`**, while the
implemented square system has **`m*(2*d*(N-1)+2)` unknowns and equations**.
Both counts are returned in metrics. The old planar implementation has
`m*(4*N-5)` unknowns and equations and should not be labeled with either new count.

The analytical derivatives use cubic Hermite endpoint weights and their duration
derivatives, including both neighboring contact times. Only the preceding,
current, and following contact blocks appear in any block row. The root solver
uses damped Newton with `scipy.linalg.solve_banded`, pivoted general-band LU;
there is no dense global Jacobian/factorization in this path. Local blocks and
state maps are dense. The half-bandwidth is at most `2*b-1` for
`b=2*d*(N-1)+2`; band storage grows as `O(m*b^2)` and the band solve as
`O(m*b^3)`. Singular or rejected Newton steps use column-scaled sparse LSMR with
increasing damping. This uses existing SciPy dependencies. It does not establish
large-fleet sparse performance: wide blocks can still be costly.

This root method differs from the planar Powell hybrid implementation. The legacy
`root_method`, `fsolve_xtol`, vectorization flags, and `use_analytical_jacobian`
do not select another algorithm here: the vector solver always uses its
analytical derivatives. LSMR uses `atol=btol=1e-12`; damping attempts are
`0, 1e-5, 1e-3, 1e-1, 1` (the first attempt is a Newton step when available).
The line search uses residual L2 descent, an Armijo factor of `1e-4`, at most
30 halvings, and a `0.99` fraction to the minimum-gap boundary.

## Numerical checks and termination

| Setting | Default | Meaning |
| --- | ---: | --- |
| `sci_residual_tol` | `1e-7` | Absolute infinity norm of unscaled matching equations and endpoint error |
| `sci_geometry_tol` | `1e-8` | Absolute errors in `n@n-1` and `n@v` |
| `sci_mass_tol` | `1e-9` | Accept a mass only if `alpha >= -tol`; masses are not clipped |
| `sci_detection_tol` | `0` | Absolute distance allowance; acceptance requires full-horizon distance `>= R-tol` |
| `sci_min_contact_gap` | `0` | Zero selects `1e-6*T`; a positive value is used as given |
| `sci_root_max_iterations` | `100` | Maximum Newton iterations per insertion |

Use an explicit positive separation allowance (the smoke run uses `1e-6`) rather
than relying on exact floating-point equality. Root iteration stops only when the
unscaled residual is at most the smaller residual/geometry tolerance. All checks
are repeated before success; the solver reports their actual values. Tolerances
are absolute and dimensionful where applicable; badly scaled problems may need
rescaling or different settings.

Contact times remain explicit physical times. They are sorted once after insertion;
pair labels and order remain fixed during each root solve. Both residual and
Jacobian reject out-of-domain times. Feasible line searches enforce the same
minimum gap from each neighbor and both horizon endpoints. No clipping, reordering,
or post-solve time repair changes the returned root. If the worst violation cannot
be inserted because of a contact cap or gap, it remains unresolved.

Contact initialization reads the current relative states, takes `r/||r||`, projects
the relative velocity onto its tangent space, and sets `alpha=0`. For relative
position norm at most `1e-12*max(1,R)`, choose the coordinate axis least aligned with
the relative velocity and orthogonally project it; ties use the first index.
For velocity norm at most `1e-12`, choose the first coordinate axis. This is
deterministic; at a symmetric zero-position encounter it need not preserve the
same solution branch under rotations.

Continuous separation uses normalized segment time and the sum of `r_i*r'_i`
over **all** spatial coordinates. Every pair and every segment contributes both
endpoints and every real stationary candidate; constant-distance segments retain
their endpoints. Near-real polynomial roots use relative imaginary tolerance
`1e-7`; roots within `1e-10` of the normalized interval are clamped to its endpoints.
These are floating-point polynomial checks, not interval-arithmetic certificates.
No insertion exclusion is applied to acceptance, including at other pairs' knots.
The shared legacy cubic helper was generalized and its constant-distance omission
was repaired; the planar matching algorithm itself is unchanged.

| `metrics.status` | Meaning |
| --- | --- |
| `success` | Root/geometry/mass/endpoints valid and full-horizon separation passes |
| `unresolved` | A violating iterate remains; reason identifies global/pair cap, minimum gap, or timeout before insertion |
| `root_failure` | Matching iteration did not converge; reason identifies iteration limit, line-search failure, or root timeout |
| `invalid_solution` | Infeasible endpoint separation or a matching solution failing algebraic/geometry/mass checks |

Unsuccessful outputs retain the last iterate for inspection and set `converged`
to false. `checks.valid` describes matching validity; separation must also pass
for overall success. Local root finding does not guarantee the globally minimizing
branch, existence of a finite contact representation, or convergence.
The timeout is cooperative between numerical operations, not a hard process limit.

Only the unconstrained robot-robot problem is implemented here. Configured finite
position or dynamics bounds are rejected explicitly; obstacle inputs and velocity
arc postprocessing are not supported by this new entry point.

## Validation and Section 5.5 readiness

See [the validation report](../validation/vector-validation.md) and
[raw smoke metrics](../validation/vector-smoke-2026-09-23.json).

The comparison infrastructure still needs work before the 600-instance campaign:

- Joint `LiftedSCP` now supports general dimensions and continuous validation of
  its held-acceleration model; see [its guide](lifted-scp.md) and the
  [current joint validation report](../validation/lifted-dimension-validation.md).
- Scenario generators use circles/discs and `(N,2)` arrays. Implement the stated
  uniform d-ball endpoints with radius `5*(N/2)**(1/d)`, endpoint rejection rules,
  fixed seeds, and identical saved input instances for both solvers.
- Existing `compare_solvers`, `sweep_n`, plotting, and manuscript experiment
  runners have no dimension sweep/schema. Add dimension, initial interacting-pair
  counts, both matching counts, continuous separation, cost, and failure categories.
- Run bounded larger-fleet pilots for conditioning, memory, caps, and timeouts.
  The current small cases establish functionality, not scaling at N=16/32/64.
- Repair the pre-existing test-suite collection/configuration defects before using
  a full green suite as the campaign gate. Check repeated timings and choose common
  tolerances/acceptance rules before making speedup or success-rate claims.

No manuscript prose or benchmark runner is changed by this implementation.
