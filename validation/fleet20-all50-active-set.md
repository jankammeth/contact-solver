# All 50 saved N=20 seeds: contact removal

Both runs used the exact saved fleet-study input.npz/config.json for seeds 0–49,
with single-thread BLAS/OpenMP limits. Each is one solve per seed, not repeated timing.
The production comparison solver and manuscript were not changed.

| Version | Passed | Failed seeds |
| --- | ---: | --- |
| Vector before this change | 45/50 | 2, 9, 15, 35, 47 |
| Vector with bounded contact removal | 46/50 | 9, 15, 35, 47 |

No previously passing seed failed. Seed 2 now passes automatically with three
contacts, positive masses, and matching residual 4.07e-13; cost is
1.55303933237 versus the historical 1.55342806993. Its solve time was 0.266 s.
The new run used 30.56 s of total solve time across all 50 seeds; median solve
time among the 46 successes was 0.337 s. Historical runtimes are included for
reference, not treated as a controlled speed comparison.

The fix switches from Newton to the existing regularized sparse least-squares
fallback when Newton would require a step smaller than 1e-3. After a converged
matching solve, it removes the most negative mass below the existing tolerance
and resolves the remaining system. All pairs and segments are checked again;
contacts may be reinserted if needed. Total insertions plus removals are limited
to four times the configured contact cap (at least one update), preventing
unbounded cycling. Existing deadline, gap, contact-cap and acceptance checks remain.

Contact removal occurred on seeds 2, 35 and 47. The latter two reached the root
iteration limit after removal; before the change they failed the sign/validity
checks. Seeds 9 and 15 still fail root convergence. No failed trajectory is accepted.

This improvement is retained, but the method is not yet ready to replace all paper
results: four of fifty instances remain unresolved. The historical experiment
reported 49/50 accepted, using weaker solver acceptance checks; seed 15 failed there.

Files:
- fleet20-all50-active-set.json: all current results, checks, timings and removals.
- fleet20-all50-before-active-set.json: all baseline vector results.
- run_fleet20_all50.py: reproducible runner (optional --baseline-module and --output).

Validation includes the existing vector tests, the saved seed-2 regression with
independent polynomial separation/trajectory checks, and a forced removal/reinsertion
cycle test verifying the update limit.
