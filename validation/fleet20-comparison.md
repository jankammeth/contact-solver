# Saved fleet benchmark pilot: N=20

Used the original saved input.npz and config.json for fleet-study seeds 0–4,
not regenerated layouts. T=31.622776601683796, R=0.5, tolerance=0.001,
zero endpoint velocities, 30 total contacts and 10 per pair. New vector-only
settings retain defaults. One warm-up per solver, five measured repetitions,
alternating solver order; BLAS/OpenMP thread limits 1. Times are solve time,
excluding sampled postprocessing. Inputs and configurations are recorded in
fleet20-comparison.json alongside source hashes and individual timings.

| Seed | Contacts old/new | Old median s | Vector median s | Valid speedup |
| --- | --- | ---: | ---: | ---: |
| 0 | 6/6 | 0.7500 | 0.3630 | 2.07x |
| 1 | 3/3 | 0.1862 | 0.1254 | 1.48x |
| 2 | 4/4 | 0.6075 | 0.4721 | Failed; exclude |
| 3 | 6/6 | 0.6750 | 0.3385 | 1.99x |
| 4 | 6/6 | 0.7707 | 0.3494 | 2.21x |

Old solver succeeds on all five cases; vector succeeds on four. On successful
pairs, continuous acceleration costs agree to floating-point precision, the
maximum sampled position difference is below 9e-9, and independently computed
continuous minimum separations are 0.5 to floating-point precision.

Seed 2 hits the vector root iteration limit. Residual is 1.94, normalization error
0.147, cost 1.77237 versus 1.55343 for the successful legacy result. Clearance alone
does not make this a valid solution. Do not count its runtime as a speedup.

The current legacy solver was rerun locally: these times are not the historical
paper timings. It shares the updated cubic utilities with the vector solver.
This pilot supports a speed improvement on successful cases but reveals a
robustness regression. Diagnose seed 2 and validate the full seed set before
replacing the manuscript's fleet results. The new implementation is joint only;
prioritized results cannot simply be rerun with it.

## Seed 2 diagnostic correction

The legacy four-contact solution also has a negative reaction mass (-0.00483857)
for zero-based pair (16,17). Thus its reported success is not success under the
vector solver's acceptance criteria. The earlier characterization as solely a
robustness regression was incomplete.

A bounded diagnostic recovered the four-contact root with Powell hybrid, removed
the negative-mass contact, and solved the three-contact system with the existing
vector Newton method. Matching residual: 4.13e-13; smallest mass: 0.00282358;
independently computed continuous separation: 0.5 to floating-point precision;
cost: 1.55303933237 versus 1.55342806993 for the legacy solution.

Reproduce with validation/diagnose_fleet20_seed2.py. Results and trajectory
coefficients are saved in fleet20-seed2-repair.json and fleet20-seed2-repair.npz.
This is an experimental repair, not an automatic production-solver change:
production insertion-only behavior remains unchanged and still fails this case.
Automatic contact deletion needs a deliberate algorithm change, cycle handling,
and broader validation before replacing the experimental results.
