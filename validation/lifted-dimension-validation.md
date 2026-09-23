# Joint dimension validation — 2026-09-23

This records the implementation and bounded validation of general-dimensional
joint LiftedSCP. See [the solver guide](../docs/lifted-scp.md) for the API, actual
time-grid convention, tolerances, status meanings and timing boundaries.
No manuscript prose, prioritized algorithm or full campaign runner was changed.
No dependencies were added.

## Verification performed

The targeted suite passed **88 tests in 6.38 s**, with 116 warnings, on macOS
26.5.1 arm64, Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1 and OSQP 1.1.3.
This consists of 39 new LiftedSCP tests, 41 current VectorContactSolver tests and
eight configuration/infrastructure tests. The vector file includes four
pre-existing local five-robot tests, which this change preserves.
To verify that the delivery does not depend on those local SCI edits, the same
four test modules were also run against an isolated snapshot of `2f0b464` plus
only this task's staged changes: **84 passed, 116 warnings in 6.52 s**. That
snapshot contains 37 committed vector tests rather than the 41 local ones.
From the parent repository root, the command was:

```sh
MPLCONFIGDIR=/private/tmp/contact-solver-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=code/contact-solver/src .venv/bin/python -m pytest \
  code/contact-solver/tests/test_lifted_nd.py \
  code/contact-solver/tests/test_vector_solver.py \
  code/contact-solver/tests/test_config.py \
  code/contact-solver/tests/test_solver_infrastructure.py \
  -q --disable-warnings --tb=short --durations=5
```

The new coverage verifies sparse matrix shapes and nonzero counts, variable
ordering and round-trip packing, endpoints, recursively integrated discrete
dynamics, continuous extrema and separation independently of the acceptance
sampler. It covers dimensions 2/3/5/10, embedded and orthogonally rotated planar
encounters with unbounded configurations, genuinely nonplanar interactions
(relative trajectory rank at least three), nonzero endpoint velocities, free
motion in every coordinate, translated spherical obstacles, position and dynamics
bounds, degenerate directions, constant distances and all-pair checking. QP
failure, iteration limit, timeout, invalid state and continuous infeasibility have
separate tested statuses. The saved-endpoint comparison CLI has a round-trip test.

A one-interval swap with constant velocity is separated by two at every grid
node, but has zero pair distance at time 0.5. It now returns
`numerical_converged=true`, `continuous_feasible=false`,
`status=continuous_infeasible`. Equivalent obstacle and interior box-overshoot
cases are also rejected. An initial collision is checked even though time zero
is not an optimization block. Near-linear and repeated polynomial roots have
explicit adversarial tests, including acceleration curvature of order 1e-23.

## Planar reference and pre-existing failures

The four-case [reference fixture](../tests/data/lifted_scp_planar_reference.json)
was captured from the pre-change LiftedSCP at commit `2f0b464`, using an archived
copy outside the working checkout. It includes the solver source SHA256,
endpoints, velocities, selected trajectory states, iterations and costs. Cases
are free motion, an offset encounter, symmetric head-on motion and an encounter
with nonzero endpoint velocities. Regression tolerances are relative cost 2e-5
and absolute state 2e-5. The symmetric head-on case allows a different bypass
branch and compares cost plus independent feasibility. The other trajectories
match their reference snapshots.

Reference costs, also reproduced by the new solver, are 0.2222839678,
1.9384497177, 2.2232105631 and 1.1810170867 respectively. The last case was reported
converged by the old discrete-only acceptance rule. Its unchanged continuous
trajectory has minimum separation 0.9986900206; it correctly fails the common
threshold 0.999 now. Enforcing previously ignored position/jerk bounds and fixing
translated obstacle halfspaces are additional intentional behavior corrections.

The selected legacy suite (`test_config`, `test_solver_infrastructure`,
`test_solvers`, `test_prioritized`) was run before and after editing:

- Before: **18 passed, 9 failed, 34 warnings in 1.55 s**.
- After: **18 passed, the same 9 failed, 34 warnings in 1.52 s**.
- Six failures use the obsolete `SolverConfig(max_contacts=...)` argument.
- Three are existing prioritized SCP feasibility failures:
  `test_swap_three_robots`, `test_no_collisions_initial_is_feasible`, and
  `test_both_solvers_feasible[3]`.

Full-suite collection was checked again: 107 collected tests and two existing
import errors in `test_integration.py` and `test_scenarios.py`, both referencing
missing `generate_line_positions`. Consequently this is not a claim of a green
full repository suite. No new regression was observed in the selected suites.

After final configuration formatting, the configuration/infrastructure tests and
the new YAML/time-grid and saved-comparison tests passed again: **10 passed,
7 warnings in 0.26 s**. Ruff passed on the six changed/new Python implementation
and test modules; Git whitespace checks passed.

## Actual paired smoke results

The [raw JSON](joint-dimension-smoke-2026-09-23.json) contains all 16 runs,
endpoints, individual QP summaries, source hashes, runtime versions and thread
settings. Run from the parent root:

```sh
MPLCONFIGDIR=/private/tmp/contact-solver-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=code/contact-solver/src .venv/bin/python -m contact_solver.cli.compare_joint \
  --smoke --output /private/tmp/joint-dimension-smoke.json
```

The command completed every case and exited **1**, reflecting real continuous
infeasibility rather than an interrupted run. All 16 SCP cases numerically
converged; **11 passed full-horizon validation and five were continuously
infeasible**. All 16 vector SCI cases reported success, with minimum separation
within about 1.5e-10 of one on interacting cases. The largest SCP integrated-grid
state discrepancy was 9.89e-14 or less.

Each pair of solvers used identical endpoint positions and velocities, `T=6`,
`h=0.1`, `R=1`, unbounded coordinates, and common separation allowance 1e-3.
SCP used at most 50 iterations, OSQP limit 20000, absolute/relative QP tolerances
1e-7, absolute position-change tolerance 1e-5, state validation tolerance 1e-5,
and a cooperative timeout of 30 s. The fleet cases have three robots; others
two. Higher-dimensional velocity cases use an independent third-coordinate
endpoint velocity, producing nonplanar relative motion. Higher-dimensional fleet
initializations need not select the same branch as the planar case.

Times below are single `timing.solve_time` observations in milliseconds,
including each solver's continuous audit and excluding optional sampling and
external cost postprocessing. Costs are the same physical time integral of sum
of squared accelerations: exact held-control cost for SCP and exact affine
acceleration cost for SCI. A lower cost for an infeasible run is not a better
feasible solution.

| d | Case | SCP status | SCP iterations | Cost SCP / SCI | SCP minimum distance | Time ms SCP / SCI |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 2 | free | success | 0 | 0.222284 / 0.222222 | 3.000000000 | 10.54 / 0.16 |
| 2 | offset | success | 3 | 1.938450 / 1.937778 | 1.000000000 | 87.60 / 1.40 |
| 2 | velocity | continuous_infeasible | 2 | 1.181017 / 1.180859 | 0.998690021 | 25.36 / 1.44 |
| 2 | fleet | continuous_infeasible | 16 | 9.613303 / 9.619126 | 0.988942060 | 709.10 / 6.91 |
| 3 | free | success | 0 | 0.333426 / 0.333333 | 3.000000000 | 10.30 / 0.12 |
| 3 | offset | success | 3 | 1.938450 / 1.937778 | 1.000000000 | 117.59 / 1.00 |
| 3 | velocity | success | 2 | 1.890462 / 1.889879 | 0.999988987 | 28.17 / 1.15 |
| 3 | fleet | continuous_infeasible | 39 | 9.452121 / 9.450383 | 0.997107077 | 4278.14 / 7.95 |
| 5 | free | success | 0 | 0.555710 / 0.555556 | 3.000000000 | 12.61 / 0.12 |
| 5 | offset | success | 3 | 1.938450 / 1.937778 | 1.000000000 | 178.22 / 1.01 |
| 5 | velocity | success | 2 | 1.890462 / 1.889879 | 0.999988987 | 37.15 / 1.15 |
| 5 | fleet | continuous_infeasible | 19 | 9.314687 / 9.313522 | 0.996997102 | 3485.77 / 8.63 |
| 10 | free | success | 0 | 1.111420 / 1.111111 | 3.000000000 | 18.98 / 0.12 |
| 10 | offset | success | 3 | 1.938450 / 1.937778 | 1.000000000 | 330.84 / 1.06 |
| 10 | velocity | success | 2 | 1.890462 / 1.889879 | 0.999988987 | 59.69 / 1.21 |
| 10 | fleet | continuous_infeasible | 24 | 9.216663 / 9.217332 | 0.988554166 | 2892.28 / 9.61 |

These vector timings include pre-existing/concurrent local SCI performance
changes that were deliberately not edited or included in this task's commit.
The raw report identifies the exact measured solver/checker sources by hash;
checking out this commit alone may not reproduce those timings. They are actual
observations, not controlled speedup estimates or evidence of large-fleet
scaling. Call-wall times, OSQP times and validation times are separately retained
in the JSON. Repeat timings on a fixed clean revision before performance claims.

For `N=2, K=60`, the SCP QP sizes are 720, 1080, 1800 and 3600 variables at
`d=2,3,5,10`; the three-robot sizes are 1080, 1620, 2700 and 5400. These follow
`3*d*N*K`, with no initial-state decision block. SCI's intrinsic count is
`2*m*d*(N-1)` and its implemented square system size is `m*(2*d*(N-1)+2)`;
both SCI counts and contact counts are recorded separately.

## Limitations and work before Section 5.5

- SCP still constrains collisions only at grid states. Five smoke cases demonstrate
  that numerical convergence is insufficient for full-horizon safety. Choose and
  document campaign grid resolution and acceptance thresholds; any adaptive
  refinement is separate work. Position bounds receive a continuous audit but
  their QP inequalities also remain at grid states.
- Piecewise constant acceleration and the discrete jerk proxy differ from SCI's
  continuously matched piecewise cubic trajectory. The common cost and distance
  definitions support comparison without asserting identical admissible classes.
- The new quadratic checker is isolated from the shared cubic helper. An
  adversarial polynomial `r(u)=(-2+4*u+1e-23*u*u, 0.1)` exposed lost interior
  roots in the existing companion-root helper. The SCP checker now handles it;
  hardening and independently validating the generic SCI cubic root routine
  remains follow-up work before treating it as a campaign acceptance oracle.
  Its pre-existing/concurrent edits were preserved.
- No new d-ball campaign generator, 600-instance runner or dimension-aware
  experiment schema was built. Existing scenario generators, comparison/sweep
  commands and plotting largely assume circles/discs and `(N,2)` arrays.
  `compare-joint --states ... --config ...` is the small saved-instance interface.
  Build shared seeded d-ball endpoint data with the manuscript's rejection rules
  and use identical saved instances for both solvers before the campaign.
- Agree on failure categories and timing boundaries, repair the existing suite
  defects, and run bounded repeated larger-fleet pilots for conditioning,
  memory, contact caps and timeouts at the intended fleet sizes. The current
  comparison cases have at most three robots and do not establish Section 5.5
  runtime or success rates. SCI bounds and obstacle comparisons remain unsupported.

The 600-instance benchmark was not launched. Manuscript prose is unchanged.
