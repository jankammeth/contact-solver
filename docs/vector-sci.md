# General-dimensional joint SCI

`ContactSolver` implements the explicit-vector choice in manuscript Sections
4.2–4.4 for `N >= 2` and `d >= 2`. Every joint experiment uses this implementation.
The old planar angle/speed implementation and velocity-arc postprocessing have
been removed. Prioritized SCI remains a
separate sequential algorithm and shares the admissible Newton safeguards.

## Inputs and entry points

Set `problem.dimension` (default 2), and pass finite endpoint positions and optional
velocities of shape `(N, d)` or flattened length `N*d`. Missing velocities are zero.
The prescribed barycenter is the cubic Hermite interpolant of the endpoint averages.
Set a positive horizon for reproducible experiments; automatic horizon selection
uses the old distance-based heuristic generalized to all coordinates.

```python
from contact_solver import Config, ContactSolver
from contact_solver.config import ProblemConfig, SolverConfig

config = Config(
    name="example",
    problem=ProblemConfig(n_robots=2, dimension=3, time_horizon=6,
                          timestep=0.1, min_distance=1),
    solver=SolverConfig(sci_detection_tol=1e-6),
)
solver = ContactSolver(config)
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
error. The smoke option runs 20 cases with at most five robots. It is **not** the Section 5.5 campaign.

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

The historical `root_method`, `fsolve_xtol`, `apply_velocity_bounds`, `sci_geometry_tol`,
contact-cap and vectorization fields have been removed from the active configuration schema.
`Config.from_yaml` discards these eight known obsolete fields when loading saved
YAML/JSON experiments; other unknown fields still raise an error. `use_analytical_jacobian=False` explicitly selects central finite
differences for the Jacobian ablation; the matching equations and safeguards
remain the same. LSMR uses `atol=btol=1e-12`; damping attempts are
`0, 1e-5, 1e-3, 1e-1, 1` (the first attempt is a Newton step when available).
The line search uses residual L2 descent, an Armijo factor of `1e-4`, at most
30 halvings, and a `0.99` fraction to the minimum-gap boundary.

## Numerical checks and termination

| Setting | Default | Meaning |
| --- | ---: | --- |
| `sci_residual_tol` | `1e-7` | Absolute infinity norm of unscaled matching equations (including `n@n-1` and `n@v`) and endpoint error |
| `sci_mass_tol` | `1e-9` | Accept a mass only if `alpha >= -tol`; masses are not clipped |
| `sci_detection_tol` | `0` | Absolute distance allowance; acceptance requires full-horizon distance `>= R-tol` |
| `sci_min_contact_gap` | `0` | Zero imposes strict ordering only, with no exclusion neighborhood; a positive value is an explicit admissibility bound |
| `sci_root_max_iterations` | `100` | Maximum Newton iterations per insertion |

Use an explicit positive separation allowance (the smoke run uses `1e-6`) rather
than relying on exact floating-point equality. Root iteration stops only when the
unscaled residual is at most `sci_residual_tol`. All checks
are repeated before success; the solver reports their actual values. Tolerances
are absolute and dimensionful where applicable; badly scaled problems may need
rescaling or different settings.

Contact times remain explicit physical times. They are sorted once after insertion;
pair labels and order remain fixed during each root solve. Both residual and
Jacobian reject out-of-domain times. Feasible line searches enforce the same
minimum gap from each neighbor and both horizon endpoints. No clipping, reordering,
or post-solve time repair changes the returned root. If the worst violation cannot
be inserted because of the minimum gap, it remains unresolved.

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
their endpoints. Stationary roots are isolated by recursively partitioning at derivative roots
and bracketing sign changes on normalized time. Repeated stationary roots and
segment endpoints are included.
These are floating-point polynomial checks, not interval-arithmetic certificates.
No insertion exclusion is applied, including at other pairs' knots. Only if the
new time exactly equals an existing contact time, offset its initial guess by at
most `1e-6*T` (at most one quarter of the adjacent available gap). Existing times
and all noncoincident candidates remain untouched. The new state is then solved
and checked normally; numerical failure remains failure.

| `metrics.status` | Meaning |
| --- | --- |
| `success` | Root/geometry/mass/endpoints valid and full-horizon separation passes |
| `unresolved` | A violating iterate remains; reason identifies minimum gap or timeout before insertion |
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
- Run bounded larger-fleet pilots for conditioning, memory and timeouts.
  The current small cases establish functionality, not scaling at N=16/32/64.
- Repair the pre-existing test-suite collection/configuration defects before using
  a full green suite as the campaign gate. Check repeated timings and choose common
  tolerances/acceptance rules before making speedup or success-rate claims.

All production joint runners now use this solver. Historical saved results still
refer to their recorded source hashes and must be rerun before claiming they
represent this implementation.

### Negative reaction masses and contact removal

The joint solver now removes the most negative contact mass after a converged
matching solve and resolves the remaining contacts. It then repeats the full
continuous separation search, so removing a contact does not bypass collision
acceptance. The positive finite solve timeout bounds insertion/removal cycling. Metrics include
`removed_contacts` (pair, time, mass) and `active_set_updates`. Heavily backtracked
Newton steps switch to the existing regularized fallback rather than stagnating.
The latest 50-seed N=20 validation passes all 50 cases with the recovery below.
Contact removal changes the earlier insertion-only implementation; historical
full-campaign results have not been replaced.

### Condensed reaction recovery

Recovery is **on by default**, matching the implementation used for the current
paper experiments. Set `sci_reaction_recovery=False` to disable it.
If banded Newton stagnates (12 consecutive iterations with less than 1% residual
norm improvement), reaches its iteration limit, or fails its line search, a bounded
Powell-hybrid recovery eliminates the intermediate knot states. The primary solver
and its fast path are unchanged. Explicit zero-iteration settings disable recovery,
and the same wall-clock deadline applies to every residual and Jacobian evaluation.

For zero clamped endpoint data, the scalar Green function is
`G(t,s) = max(t-s,0)^3/6 + s(T-s)^2 t^2/(2T^2) - (T+2s)(T-s)^2 t^3/(6T^3)`.
Adding the prescribed free cubic and the sum of `alpha*R*sigma*n*G` constructs each
robot trajectory. Contact position, unit normal, and tangency determine `m*(d+2)`
unknowns: time, normal components, and mass per contact. The analytic Jacobian is
implemented in `reaction_matching.py` and checked against finite differences in
2, 3, 5 and 10 dimensions. Recovery can exchange contact order; the returned times
are sorted and must satisfy the original endpoint and minimum-gap restrictions.

Local polynomial coefficients are constructed directly from the Green function.
This avoids subtracting nearly equal endpoint states across a very short segment.
Recovered trajectories are checked using their actual coefficient acceleration
and jerk jumps, contact position, normalization, tangency, endpoint states, reaction
signs, and full-horizon pairwise separation. Acceptance tolerances are unchanged.
Negative reactions return to the bounded contact-removal loop. A root-finder success
flag alone is never sufficient for acceptance.

The usual full system remains `m*[2*d*(N-1)+2]`; recovery uses `m*(d+2)` unknowns.
Each recovery entry in `metrics.roots` records its system size and whether contacts
were reordered. Runtime and system-size analyses should distinguish the two paths.
See `validation/fleet20-all50-recovered.md` for the full seed set and limitations.

Prioritized SCI uses the corresponding reduced system for prescribed moving
obstacles, with contact time, angle, and reaction mass as its three unknowns per
contact. It retains the recovered cubic coefficients directly and rechecks the
original matching, endpoint, contact, and tangency residuals. The same deadline,
minimum contact gap, mass-removal loop, and separation tolerance apply.

### Compiled continuous separation search

The package builds `_cubic_minima.c` during installation. Joint and prioritized
SCI batch relative cubic segments, exclude segments using conservative Bézier
control-point bounds, and isolate stationary-distance roots on the remaining
segments. The earliest-contact ablation retains its scalar candidate search.
No sampled clearance is used for acceptance. Without the compiled module, the
Python reference checker searches all segments and emits a performance warning;
`metrics.collision_backend` records `native` or `python`.

## Timing, evaluation, and prioritized comparisons

`timing.solve_time` (also `total_time`) ends after matching validation and the
insertion loop's final continuous separation check. `postprocess_time` measures
sampling; `wall_time` includes it. No sampled distance can turn a failure into
success. Prioritized SCI retains the continuous minimum from each rank's final
search, since later ranks do not change already committed paths.

Paired SCI/SCP costs use `h * sum(||a_i(k*h)||^2)` at exactly `k=1,...,K`, with
the same K, h, and uniform weights for both methods. `continuous_cost` fields
report the exact cubic integral separately; the comparison field remains the
discrete cost. Evaluation lives in `contact_solver.evaluation` outside timers.

Prioritized comparison runners compute the existing sampled severity order once,
save it with preprocessing duration, and pass it to both methods. Solver times
and runtime plots exclude that duration. A standalone prioritized solve may
compute its own order, also outside `solve_time`. The ordering plot now has one
solve-time series per method and no extra planning markers.
