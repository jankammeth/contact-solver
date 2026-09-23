# Explicit-vector SCI validation — 2026-09-23

All measurements below are from the local implementation run. The full 600-instance
Section 5.5 benchmark was not run. No manuscript files were edited.

## Tests

From the parent repository:

```sh
MPLCONFIGDIR=/private/tmp/contact-solver-mpl OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=code/contact-solver/src \
.venv/bin/python -m pytest \
  code/contact-solver/tests/test_vector_solver.py \
  code/contact-solver/tests/test_config.py \
  code/contact-solver/tests/test_solver_infrastructure.py -q
```

Final outcome: **45 passed** (37 new vector tests and 8 existing config/infrastructure tests).
Ruff passes for the new modules/tests and changed cubic utilities; Git whitespace checks pass.

Coverage includes:

- Equation counts and packing for N=2/4, d=2/3/5/10, multiple pairs, and four contact blocks.
- Analytical Jacobians against central finite differences, with relative tolerance 2e-7 and absolute tolerance 1e-7; non-neighbor blocks are exactly zero. A separate near-minimum-gap time derivative check passes.
- Banded storage/linear solves and a forced-singular Newton test exercising sparse LSMR fallback.
- Embedded and orthogonally rotated planar encounters in d=2/3/5/10.
- Genuinely nonplanar three-robot fleets in d=3/5/10, with all three pairs interacting; a separate nonplanar case has two contacts for the same pair.
- No-contact trajectories with nonzero endpoint velocities and a prescribed moving barycenter.
- Deterministic zero-relative-position initialization with nonzero and zero relative velocity.
- Coefficient-level C0/C1/C2 checks, robot jerk jumps/mass signs, endpoint positions/velocities, and independent squared-distance polynomial minimization over every pair/segment.
- Endpoints, constant distance, collisions only in the last coordinate, multiple interior stationary minima, other-pair collisions at knots, and short/long segment durations.
- Contact/pair caps (including a residual-valid but still colliding iterate), minimum-gap limits, root-iteration failures, infeasible endpoints, negative masses, and rejection of sample-only success.
- Four planar regression cases: head-on, offset, seed-82 fleet, and nonzero endpoint velocities. Costs agree within relative 2e-5; paths agree within 2e-5 except head-on, where opposite symmetry branches are allowed.
- YAML roundtrip, shape/finite input checks, automatic horizon, NPZ CLI input/output, and nonzero CLI exit status for unresolved output.

## Bounded smoke timings

The committed [raw report](vector-smoke-2026-09-23.json) records all metrics, equation counts, tolerances, root details, costs, package versions, and thread limits.

Run from the parent repository:

```sh
MPLCONFIGDIR=/private/tmp/contact-solver-mpl OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=code/contact-solver/src \
.venv/bin/python -m contact_solver.cli.vector_sci --smoke --output smoke.json
```

Environment: macOS 26.5.1 arm64, Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, OSQP 1.1.3. Requested BLAS/OpenMP thread limits were 1.

All **16/16** vector cases passed. These are single-run solve times in milliseconds, excluding sampled postprocessing. Startup/import time is excluded. No scaling or speedup claim is inferred.

| d | Free, N=2 | Head-on, N=2 | Offset, N=2 | Fleet, N=3 | Fleet contacts | Intrinsic / implemented fleet size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.162 | 2.717 | 2.312 | 9.362 | 3 | 24 / 30 |
| 3 | 0.847 | 2.056 | 1.956 | 10.095 | 3 | 36 / 42 |
| 5 | 0.767 | 1.973 | 1.932 | 11.072 | 3 | 60 / 66 |
| 10 | 0.839 | 2.064 | 2.011 | 11.824 | 3 | 120 / 126 |

The maximum unscaled residual was 4.580e-9. The smallest reported continuous separation was 0.9999999999999998 for R=1, within the 1e-6 distance tolerance. All masses passed the -1e-9 lower bound. The smoke cases used no sparse fallback; fallback behavior is covered separately by the forced-singular test.

Planar smoke comparisons use the same inputs and T=6:

| Case | Vector solve ms | Legacy solve ms | Vector cost | Legacy cost |
| --- | ---: | ---: | ---: | ---: |
| free | 1.162 | 0.098 | 0.2222222222 | 0.2222222222 |
| head_on | 2.717 | 0.981 | 2.222222222 | 2.222222222 |
| offset | 2.312 | 0.946 | 1.937777778 | 1.937777778 |
| fleet | 9.362 | 8.715 | 9.619125863 | 9.619125863 |

Costs use exact two-point Gaussian integration of each segment’s quadratic acceleration norm. The maximum relative planar cost difference in this run was 4.45e-16. The raw report includes independently computed continuous legacy minima; the legacy solver’s own reported minimum remains sampled. The new solver charges additional full-horizon and validity checks to solve time, so these timings alone are not a matched performance experiment.

## Existing suite limitations

Before editing, full-suite collection failed in `test_integration.py` and `test_scenarios.py` because `generate_line_positions` is not exported/implemented. A clean archive of the original submodule HEAD (51f21f1) and the modified tree were also run against the remaining four legacy test files: both produced **18 passed, 9 failed** (1.55 s and 1.62 s respectively). Six failures use obsolete `SolverConfig(max_contacts=...)` arguments; three are existing prioritized SCP feasibility failures. These unrelated issues were left unchanged.

## Remaining scope

The implementation supports the unconstrained robot-robot fleet formulation only; finite position/dynamics bounds are rejected. Root finding is local; singular symmetries, contact clusters, and different branches can remain difficult. Dense local blocks and band width grow with d and N; large-fleet memory/time and robustness remain unmeasured. The polynomial separation test uses floating-point roots, not rigorous interval enclosures.

Before Section 5.5, generalize and validate the planar comparison solver, implement paired uniform d-ball input generation, add dimension-aware experiment/result schemas and interaction counts, repair the existing suite blockers, and run bounded N=16/32/64 pilots with memory/time limits. The [implementation guide](../docs/vector-sci.md) names the affected infrastructure.

Keep the counts distinct: intrinsic degrees of freedom **2md(N−1)**; implemented explicit-vector system **m[2d(N−1)+2]**. The extra two equations per contact impose normalization and tangency.
