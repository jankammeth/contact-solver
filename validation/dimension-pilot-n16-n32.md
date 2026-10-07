# Dimension pilot: fleets 16 and 32

Seed 0; two repetitions per solver and cell. Solver budget 15 seconds; hard process limit 25 seconds. This is a smoke pilot, not the 50-seed experiment.

Raw inputs, source/environment manifest and worker results: `runs/numerics/dimension-pilot-20260923` in the parent repository. That historical run also included N=64; those measurements are excluded here. The runner defaults were subsequently reduced to N=16,32; solver code is unchanged by that scope reduction.

| d | N | Solver | Accepted / attempts | Median accepted seconds | Status counts |
|---|---|---|---|---|---|
| 2 | 16 | SCI | 2/2 | 0.5518 | {'success': 2} |
| 2 | 16 | SCP | 0/2 | — | {'hard_timeout': 2} |
| 2 | 32 | SCI | 0/2 | — | {'root_failure': 2} |
| 2 | 32 | SCP | 0/2 | — | {'hard_timeout': 2} |
| 3 | 16 | SCI | 2/2 | 0.0377 | {'success': 2} |
| 3 | 16 | SCP | 2/2 | 2.0000 | {'success': 2} |
| 3 | 32 | SCI | 2/2 | 0.3489 | {'success': 2} |
| 3 | 32 | SCP | 0/2 | — | {'hard_timeout': 2} |
| 5 | 16 | SCI | 2/2 | 0.0378 | {'success': 2} |
| 5 | 16 | SCP | 2/2 | 2.2316 | {'success': 2} |
| 5 | 32 | SCI | 2/2 | 0.1723 | {'success': 2} |
| 5 | 32 | SCP | 2/2 | 8.2572 | {'success': 2} |
| 10 | 16 | SCI | 2/2 | 0.0417 | {'success': 2} |
| 10 | 16 | SCP | 2/2 | 2.3374 | {'success': 2} |
| 10 | 32 | SCI | 2/2 | 0.1544 | {'success': 2} |
| 10 | 32 | SCP | 2/2 | 8.6932 | {'success': 2} |

SCI passed 14/16 attempts; SCP passed 10/16. SCI at d=2,N=32 exhausted its short solve budget during recovery (reported root_failure). SCP hit the hard process limit at d=2,N=16; d=2,N=32; and d=3,N=32. These are not evidence of convergence within the full 300-second budget.

The five cells where both solvers passed had no initially interacting pairs. Their timing ratios therefore do not establish performance on contact-rich instances. The remaining retained cells had 6, 10 and 1 initially interacting pairs respectively at (d,N)=(2,16),(2,32),(3,32).

Validation before the scope reduction: 121 targeted tests passed, including cubic separation, vector SCI, multidimensional SCP, campaign execution/resume, configuration and solver infrastructure.

The full experiment is now 4 dimensions × 2 fleet sizes × 50 seeds = 400 paired instances (800 solver attempts per repetition). It has not been launched.
