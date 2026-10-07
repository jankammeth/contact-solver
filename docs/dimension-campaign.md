# Section 5.5 campaign

`python -m contact_solver.cli.dimension_campaign --output DIR` runs a bounded
pilot by default. It does not silently launch the full campaign. Run from the
parent repository with `PYTHONPATH=code/contact-solver/src` and the existing venv.

## Design and input data

The full design crosses dimensions 2, 3, 5, 10 with fleets 5, 10 and seeds
0–49: 400 paired instances, or 800 solver attempts before repetitions. Initial
and final positions are independently sampled uniformly by volume in a d-ball
of radius `5*(N/2)**(1/d)`, rejecting endpoint pair distances below 0.8. Directions
are normalized Gaussian vectors; normalized radius is `U**(1/d)`. Velocities are
zero. The generator uses `SeedSequence([5501,d,N,seed])` and saves the arrays once.
Both solvers read exactly the same NPZ file.

With `--match-interactions`, the ball radius is multiplied by a calibrated
factor for each `(d,N)`. The uncalibrated design remains the module default;
the manuscript shell script selects calibration and a new output directory.

## Matching interaction density

The target for each fleet is the mean initial interacting-pair count of the
original planar generator. It is estimated on 256 calibration seeds
10000–10255, independently of the evaluation seeds 0–49. Matching the mean
count at fixed fleet size also matches the fraction of interacting pairs.
The planar radius stays unchanged. These targets are new calibration
estimates, not fitted to the previous campaign's 50 evaluation outcomes.

For each higher-dimensional cell, the runner brackets the target by changing
the radius by factors of sqrt(2), then takes six geometric bisection steps.
It keeps the tested radius with the closest mean count. The same calibration
seeds are used at every radius; endpoints are regenerated with spacing 0.8
enforced. No solver runs during calibration. For zero endpoint velocities,
interaction counting minimizes relative straight segments over their shared
monotone cubic progress, avoiding polynomial root searches.

The manifest freezes the reference statistics, selected radii, every search
trial, calibration settings, and a calibration checksum. Resume reuses those
radii and rejects changes to settings, code, environment or calibration.
`--calibration-samples` and `--calibration-seed-start` customize the calibration
seed block; overlap with evaluation seeds is rejected. Failure to bracket the
target or generate spaced endpoints stops preparation with an error.

Before any solver runs, `interaction_validation.json` reports the held-out
mean count and standard error, target mean and standard error, density,
collision-free fraction, and count histogram for each cell. Inspect this
using `--prepare-only`. Validation reports sampling differences; it does not
retune radii or discard evaluation instances. Equal mean density does not
guarantee equal interaction graphs, encounter depths, or solver difficulty.
This is one matched-density experiment, without additional density levels
or conditioning on an exact count distribution.

T=10, R=0.5, K=200, and separation allowance 0.001. Remaining solver settings
come from `configs/sweep_n.yaml`: OSQP absolute tolerance 1e-4, relative tolerance
0, 20 outer iterations, and 10000 QP iterations.
These are the manuscript settings, not the tighter small-smoke settings.
Initial interacting-pair counts use the continuous free cubic and threshold R.

## Running and resuming

```sh
# 8 paired cells, seed 0, two repeats per solver. Short pilot time limits.
MPLCONFIGDIR=/private/tmp/contact-solver-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=code/contact-solver/src \
.venv/bin/python -m contact_solver.cli.dimension_campaign \
  --output runs/numerics/dimension-pilot-n5-n10 --repeats 2 --timeout 15

# Prepare the full design without executing solvers.
PYTHONPATH=code/contact-solver/src .venv/bin/python \
  -m contact_solver.cli.dimension_campaign --full --prepare-only --output NEW_DIR

# Explicit full campaign, default 300-second solver budget.
PYTHONPATH=code/contact-solver/src .venv/bin/python \
  -m contact_solver.cli.dimension_campaign --full --output NEW_DIR
```

Rerun exactly the same command to resume. Completed attempts (including failures)
are retained. Changed source hashes, environment versions, design, inputs or saved
configurations are rejected. Use a new directory for a new revision or settings;
do not silently merge incompatible runs. Repetitions alternate solver order.
`--dimensions`, `--fleet-sizes`, and explicit `--seeds` select bounded subsets.

Each method/repetition is a fresh subprocess with BLAS/OpenMP thread limits 1.
The hard process timeout is the solver budget plus 10 seconds, including import,
setup and reporting. Timed-out processes are killed and recorded as failures.
On Linux, `--memory-gib` applies RLIMIT_AS (default 4 GiB). This macOS environment
cannot apply that cap; the worker explicitly records it as unsupported and reports
peak RSS instead. Attempts run sequentially; there is no concurrent fleet of QPs.
A killed worker cannot report peak RSS or partial numerical diagnostics.

## Results and interpretation

- `manifest.json`: design, source/config hashes, versions, thread limits and,
  when requested, frozen interaction calibration.
- `interaction_validation.json`: held-out interaction statistics for calibrated runs.
- Per-cell `input.npz`, `config.json`, `instance.json`: immutable paired inputs,
  resolved settings, hashes, and initial interacting-pair count.
- Per-attempt `request.json`, `result.json`, `worker.log`: audit trail, status,
  timing, memory, residuals, separation, cost, contacts and recovery diagnostics.
- `attempts.csv`, `summary.json`: all attempt outcomes and paired summaries,
  updated after every attempt. Timeouts, QP failures, continuous infeasibility,
  contact limits, numerical failures and process errors remain distinct.

The common reported `solve_seconds` includes continuous acceptance checks and
excludes optional sampled trajectory postprocessing. Process wall time includes
startup and serialization and is recorded separately. Costs use the exact physical
acceleration integral for each trajectory representation. Runtime medians in the
summary use accepted runs; paired speedups and cost ratios require both methods
to be accepted. Numerical convergence and continuous clearance are also retained
separately. A lower cost on an infeasible trajectory is not a valid improvement.

The contact solver records intrinsic/full matching sizes and each condensed
reaction-recovery size separately. The fallback can reorder/remove contacts; this
is not the original insertion-only implementation. Report both the configuration
and the algorithm revision when interpreting a paper comparison.

The root checker isolates real roots on [0,1] using derivative partitions and
bracketed scalar solves. It fixes the nearly linear counterexample that a
companion eigenproblem missed. It remains floating-point validation, not an
interval-arithmetic certificate. The shared cubic change affects legacy contact
solver timing as well as the new solver.

Fleet size 64 is excluded from the current design. The earlier pilot directory
`runs/numerics/dimension-pilot-20260923` retains its original manifest and raw
measurements; its completed 64-robot runs are historical only. See
`validation/dimension-pilot-n16-n32.md` for the retained pilot results.

## Manuscript campaign entry point

From the parent repository, run `bash scripts/run_dimension_experiment.sh`.
This calibrates radii and runs all 400 paired held-out instances with dimensions
2, 3, 5, 10, fleet sizes 5, 10, seeds 0–49, one repetition and a 300-second
solver budget. The default output is `runs/numerics/dimension-matched-n5-n10`;
the original `dimension-n5-n10-n20` results are retained unchanged.

```sh
# Calibrate, freeze inputs, and inspect held-out interaction counts first.
bash scripts/run_dimension_experiment.sh --prepare-only
# Run/resume solvers on those same frozen inputs.
bash scripts/run_dimension_experiment.sh
```

To run in stages while retaining one dataset and the same frozen manifest:

```sh
# First N=5, then N=10.
bash scripts/run_dimension_experiment.sh --run-fleet-sizes 5
bash scripts/run_dimension_experiment.sh --run-fleet-sizes 10
```

`--run-fleet-sizes` only selects execution; it does not change the design,
calibration, or inputs. Summaries include all completed fleets after either
stage. Run stages sequentially. Omitting the selection resumes the complete
design, skipping saved results. Do not change `--fleet-sizes` to stage a run:
that changes the experiment design and correctly fails manifest validation.

Use `--output NEW_DIR` to select a fresh output directory. Source changes
invalidate resume even for an older uncalibrated campaign: keep its results
for analysis, or use its recorded source revision to resume it.
The lower-level module still defaults to a short seed-0 pilot unless `--full`
is passed. Historical pilot reports retain their original fleet sizes.
