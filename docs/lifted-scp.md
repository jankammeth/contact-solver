# General-dimensional joint LiftedSCP

`LiftedSCP` supports `problem.dimension = d >= 2`, using the existing sparse QP
and OSQP integration. `VectorContactSolver` can solve the same unconstrained
endpoint problems. Prioritized planners and the old `ContactSolver` retain their
planar algorithms; their base endpoint setters reject dimensions other than two.

## Inputs and bounds

Pass finite positions and optional velocities of shape `(N, d)` to
`set_initial_states` and `set_final_states`. Flattened length `N*d` remains accepted;
omitted velocities are zero. Shape and finiteness errors raise `ValueError`.
The solver accepts `N >= 1`, positive finite timestep, horizon and separation radius.

Legacy YAML configurations still default to `dimension: 2`. Position bounds can
use the existing `pos_x_min/max`, `pos_y_min/max` fields, or the new
`problem.environment.position_lower/upper` vectors of length `d`. Do not mix the
two styles. Missing sides are unbounded; vector entries may be infinite. Legacy
x/y bounds constrain only the first two coordinates in higher dimensions.
NaNs, inconsistent lengths and reversed bounds are rejected.

`problem.dynamics.vel_min/max`, `acc_min/max`, and `jerk_min/max` remain scalar
componentwise bounds, applied to every coordinate. They are box bounds, not norm
bounds. As in the old joint QP, configured velocity bounds are included directly;
`apply_velocity_bounds` controls SCI postprocessing, not this QP. The jerk bounds
constrain the discrete proxy `(a[k]-a[k-1])/h` for `k >= 1`. Held acceleration can
jump, so these do not assert bounded classical continuous jerk.

Static obstacle centers are supplied as `LiftedSCP(config,
obstacle_positions=centers)` with finite shape `(M, d)`. Each center uses the same
exclusion distance `problem.min_distance` as robot pairs. These are spherical
exclusions; there is no separate obstacle radius or general obstacle geometry.
An empty obstacle array is accepted. The translated obstacle halfspace is
`eta @ p >= R + eta @ center`.

Compared with the old joint implementation, configured position and discrete
jerk bounds are now enforced (previously ignored), and the missing translation
in obstacle halfspaces is corrected. These fixes can intentionally change planar
results with those constraints. Unconstrained planar regression cases preserve
their trajectories and costs within numerical tolerance.

## Time grid, packing and continuous trajectory

Let `h` be the timestep and `K = round(T/h)`. Require `K >= 1` and
`K*h == T` within `rtol=1e-10, atol=1e-12`; genuinely nonintegral horizons now
raise an error instead of silently truncating. Near-integral floating-point
ratios such as `0.3/0.1` retain all three intervals. Automatic horizons use the
existing distance heuristic, rounded up to this grid, with at least one interval.

There are **`3*d*N*K` variables**, in robot-major, timestep-major order:
`[p[0:d], v[0:d], a[0:d]]`. For zero-based `k`, `p[k]` and `v[k]` are states at
`(k+1)*h`; `a[k]` acts on `[k*h, (k+1)*h)`. The initial state at zero is prescribed,
not an additional optimization block. Dynamics and terminal conditions contribute
`2*d*N*(K+1)` equality rows. Each collision QP adds `K*N*(N-1)/2` pair rows and
`K*N*M` obstacle rows. Bounds add only their enabled rows. Matrices are assembled
as sparse CSC matrices; the acceleration Hessian has `N*K*d` nonzeros. The planar
private `px/py/vx/vy/ax/ay` aliases remain available for existing subclasses.

On interval `k`, with local time `s` in `[0,h]`, the represented trajectory is

```text
p(s) = p_left + s*v_left + 0.5*s^2*a[k]
v(s) = v_left + s*a[k]
```

This is piecewise quadratic position with piecewise constant acceleration.
It is not cubic Hermite interpolation. The continuous object integrates the
returned controls from the prescribed initial state, checks drift from the QP
states, and checks the integrated terminal states. It never repairs QP residuals
by interpolating those states. Acceleration samples at internal knots use the
right interval, and the final endpoint uses the last interval.

`result["trajectories"]` retains the three lists of `N` arrays shaped `(K,d)`;
`result["times"]` is `[h, ..., K*h]`. `solver.trajectory` exposes the integrated
model. `solver.sample(n=1000)` includes both horizon endpoints and returns
`(times, positions, velocities, accelerations)`; `sample(times=...)` accepts
explicit finite times in the horizon. Stacked samples have shape `(N,samples,d)`.

The QP objective remains `sum(a**2)`. The physical acceleration cost returned in
`metrics.acceleration_cost` is `h*sum(a**2)`, equal to the time integral of the
sum of squared acceleration norms. The constant factor does not affect its QP
minimizer. SCI uses its own piecewise cubic trajectory and integrates its affine
accelerations exactly for the same physical cost. Equal endpoints, horizon and
separation tolerances do not make these two trajectory classes identical.

For comparison, SCI's intrinsic degrees of freedom are **`2*m*d*(N-1)`**; its
implemented square system has **`m*(2*d*(N-1)+2)` unknowns and equations**. Neither
is the LiftedSCP QP size. `m` is the final number of contacts.

## Acceptance and statuses

Before reporting success, the solver checks QP dynamics, terminal conditions,
integrated state drift, continuous position/velocity/acceleration bounds, the
discrete jerk proxy, and full-horizon robot-pair and obstacle separation.
Position extrema include interior quadratic critical points. Velocity extrema
occur at interval endpoints. Separation minimizes each quadratic relative
trajectory using the polynomial `r @ r'`, summing all `d` coordinates. Every pair,
obstacle and segment includes both endpoints and constant-distance cases.

The distance derivative has degree at most three in normalized segment time.
Roots are isolated on `[0,1]` using derivative partitions and Brent brackets;
critical points are also retained to include repeated roots. This avoids losing
interior roots when near-zero acceleration produces tiny leading coefficients.
Brent uses absolute tolerance `1e-14` and relative tolerance `4*machine_epsilon`.
These are floating-point checks, not interval-arithmetic certificates.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `scp_detection_tol` | `0` | Distance allowance: full-horizon minimum must be at least `R-tol`, also used by the grid check |
| `scp_validation_tol` | `1e-5` | Absolute componentwise tolerance for unscaled dynamics, terminal error, integrated drift and bounds |
| `scp_convergence_abs` | `0` | Maximum absolute position change for numerical stopping, together with grid feasibility |
| `scp_convergence_rel` | `0` | Relative change is diagnostic only, preserving the old joint stopping rule |
| `osqp_eps_abs/rel` | schema: `0/0` | If both are zero, LiftedSCP alone uses the existing OSQPSettings defaults `1e-5/1e-5`; explicit tolerances are retained |
| `scp_max_iterations` | `15` | Maximum collision QPs after the initial QP |
| `timeout` | `0` | Zero disables; otherwise cooperative wall-time limit in seconds |

Use positive tolerances appropriate to the problem scale. The saved comparison
example uses separation `1e-3`, state validation `1e-5`, absolute change `1e-5`,
and OSQP absolute/relative tolerances `1e-7`. OSQP statuses "solved" and "solved
inaccurate" still require the independent acceptance checks.

| `metrics.status` | Meaning |
| --- | --- |
| `success` | Numerical stopping and all continuous/state checks pass |
| `qp_failure` | A QP failed or returned an invalid vector; `qp_error` gives details |
| `iteration_limit` | Collision QP limit reached without numerical stopping |
| `timeout` | Cooperative time limit reached |
| `invalid_solution` | Numerically converged output fails dynamics, endpoint or integrated-state validation |
| `continuous_infeasible` | Numerically converged output violates continuous separation or bounds |

`metrics.converged` means overall success. `numerical_converged` records SCP
stopping before the continuous audit. `continuous_feasible` records all state,
bound and separation checks for the available iterate, independently of the
stopping reason. `stopping_reason` retains the iterative reason;
`convergence_reason` adds a validation failure when applicable. Limit and QP
failure statuses take precedence even if the last available iterate is feasible.
An initial QP failure returns no trajectory; a later failure retains the last
successful QP iterate. `collision_check` and `grid_min_distance` describe only
the discrete grid and do not determine overall success.

Coincident collision linearizations choose a unit axis least aligned with
relative endpoint travel, projected orthogonal to that travel. Ties use the first
coordinate; zero travel chooses the first axis. Position norms below `1e-6` and
travel norms at most `1e-12` select these fallbacks. This is deterministic but
does not ensure rotation-equivariant branch selection at exact symmetry.

There is no adaptive grid refinement or continuous constraint insertion in this
implementation. A converged but colliding discrete solution is returned explicitly
as infeasible. Timeouts are checked around QPs, not inside OSQP; a QP or final audit
can run beyond the deadline. Hard process limits and SCP refinement are separate
work. No optimality or large-fleet scaling claim follows from the smoke cases.

## Reproducible paired comparisons

Install the package as before or run the module with `PYTHONPATH=src` from this
package directory. The saved-endpoint interface reuses the `vector-sci` NPZ keys:
`initial_positions`, `final_positions`, optional `initial_velocities` and
`final_velocities`. Arrays must match the YAML `N,d`.

```sh
compare-joint --states endpoints.npz --config configs/joint_3d.yaml --output paired.json
PYTHONPATH=src python -m contact_solver.cli.compare_joint \
  --states endpoints.npz --config configs/joint_3d.yaml --output paired.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=src python -m contact_solver.cli.compare_joint --smoke --output smoke.json
```

The interface requires an explicit common horizon and equal SCI/SCP separation
tolerances. Use unconstrained configurations for comparisons: VectorContactSolver
does not support bounds or obstacles. It records endpoints, actual costs, minima,
statuses, system sizes, QP summaries and timing fields. The 16-case smoke report
also records package versions, thread settings and hashes of solver/checker sources.
Its cases have at most three robots and dimensions `2,3,5,10`. It does not launch
the Section 5.5 campaign. Exit status is zero only if both solvers succeed on every
requested case; expected continuous infeasibility still produces exit status one.

`call_wall_time` surrounds `generate_trajectories()` after construction and endpoint
setters. LiftedSCP's `timing.solve_time` includes matrix setup, all QPs and final
continuous validation; `timing.validation_time` is also broken out. Vector SCI's
solve time includes insertion/root work and its validation. Neither call includes
optional sampling, separate cost postprocessing, input loading, construction or
JSON output. OSQP's own `run_time` is a narrower QP-only measure. Compare repeated
call times on fixed saved instances before drawing performance conclusions.

See [the validation report](../validation/lifted-dimension-validation.md) for actual
results, known failures and remaining Section 5.5 preparation.
