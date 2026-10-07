# Validation overhead comparison — 2026-09-23

Removed duplicate validation and collision searches on unchanged trajectories,
reused endpoint evaluations, and replaced derivative-factor reductions with constants.
A second optimization evaluates endpoint states directly from cubic coefficients and
avoids assembling the empty matching residual. Failed root solves receive fresh checks.
No acceptance condition or tolerance was removed.

Five planar CLI smoke cases, T=6; 10 warm-up solves and 150 measured repetitions
per implementation, rotating execution order. BLAS/OpenMP thread limits are 1.
Times exclude sampled postprocessing and startup. Every measured solve succeeded.
The five-robot case uses seed 0 and has two contacts; the three-robot case uses seed 82.
The five-robot timings were rerun separately after replacing a noninteracting layout.

| Case | Legacy median ms | Original vector ms | Optimized vector ms |
| --- | ---: | ---: | ---: |
| free | 0.0701 | 0.9270 | 0.2175 |
| head_on | 0.8146 | 2.2516 | 1.0516 |
| offset | 0.7799 | 2.0317 | 0.9563 |
| fleet | 8.5752 | 9.0201 | 6.7375 |
| fleet5 | 13.9951 | 9.7641 | 6.7776 |

The JSON contains medians and 10th/90th percentiles. Reproduce with
`validation/compare_vector_overhead.py ORIGINAL_VECTOR_MODULE.py OUTPUT.json`
using the source PYTHONPATH and thread limits from vector-validation.md; optional
trailing case names update an existing report. The original module is the pre-optimization
vector_contact_solver.py, loaded alongside the current implementation.

49 tests passed across test_vector_solver.py, test_config.py, and
test_solver_infrastructure.py. All 20 smoke cases passed, including five robots
in dimensions 2, 3, 5, and 10. Ruff and whitespace checks passed.

The no-contact case remains above 0.1 ms. An isolated warmed run measured 0.168 ms;
the interleaved comparison table includes normal runtime variability.
Further reductions would require optimizing setup, allocation, and endpoint/collision
validation; these checks are still included. The legacy solver retains fewer acceptance
checks, so this is not an equal-validation workload comparison. Small cases do not
establish large-fleet scaling.

## Further free-path optimization

The next pass constructs the free relative cubic without empty knot-state assembly,
skips empty residual bookkeeping, combines barycenter reductions, and avoids separate
endpoint-separation diagnosis after a successful continuous clearance check.
Exactly constant relative position now returns both endpoints directly without
constructing the stationary-distance polynomial. This shared helper also benefits
the legacy solver. Endpoint reconstruction checks remain inside solve time.

An isolated warmed 300-run measurement gave a vector median of **0.0913 ms**
(10th/90th percentiles 0.0893/0.1008 ms). The 150-run interleaved comparison gave
**0.1230 ms** (0.0962/0.2394 ms): sub-0.1 ms is achievable in the warmed case,
not a latency guarantee. The free smoke case has exactly constant relative position;
other free trajectories may require polynomial roots.

| Case | Legacy ms | Previous vector ms | Final vector ms |
| --- | ---: | ---: | ---: |
| Free | 0.0450 | 0.2630 | 0.1230 |
| Head-on | 0.8244 | 1.1660 | 0.9587 |
| Offset | 0.7787 | 0.9825 | 0.8795 |
| Three robots | 8.3755 | 6.5297 | 6.4430 |
| Five robots | 13.9069 | 6.8210 | 6.6974 |

Raw results: vector-free-path-comparison.json. Same warm-up, repetition count and
thread settings as above. The previous vector module was saved immediately before
this free-path pass; all versions in this run share the updated cubic utilities.
All measured solves succeeded. All 49 validation tests, Ruff and whitespace checks
passed after the final edits.
