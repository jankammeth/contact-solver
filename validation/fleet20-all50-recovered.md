# Remaining N=20 failures resolved

The final implementation passes **50/50** original saved fleet-study instances
(seeds 0–49), versus 45/50 before contact removal and 46/50 after removal alone.
All use the saved inputs and per-attempt configurations. Tolerances are unchanged.

## Diagnosis and fix

- **Seed 9:** the fixed contact order obstructed the matching solve; reaction
  recovery allows the contact order to change. A negative-mass contact is then
  removed, and further collision detection/insertion produces a valid solution.
- **Seed 15:** the full knot-state Newton system stalls near clustered contacts.
  Eliminating intermediate states gives a well-conditioned enough small recovery
  system to converge, after which the insertion loop finishes normally.
- **Seed 35:** after removing a negative reaction, the remaining full matching
  solve stalls with repeated regularized steps. Condensed recovery resolves it.
- **Seed 47:** after removal/reinsertion, very close contacts cause poor conditioning
  and cancellation in endpoint-based Hermite reconstruction. Direct reaction-based
  coefficients retain the small-segment accuracy needed for matching checks.

Finite-difference probes confirmed the original Jacobian on failing states; this
was not a missing derivative. Merely increasing the iteration budget did not
resolve all failures. The implemented fallback uses the same force-balance model,
an analytical Jacobian, and existing SciPy dependencies. It does not relax validation.

## Final single-run measurements

| Seed | Contacts | Solve seconds | Maximum checked residual |
| --- | ---: | ---: | ---: |
| 9 | 13 | 3.563 | 7.25e-9 |
| 15 | 8 | 0.623 | 7.22e-12 |
| 35 | 7 | 1.992 | 6.17e-9 |
| 47 | 6 | 0.577 | 1.91e-12 |

Across all 50: total solve time 26.485 s; median 0.364 s; maximum checked residual
9.66e-9; minimum full-horizon separation 0.49920153, above the required 0.499
(R=0.5, absolute tolerance 0.001). All accepted masses satisfy the existing sign
check. Runtime excludes sampled postprocessing. This is one run per seed with
single-thread BLAS/OpenMP limits, not a repeated speedup study.

The final four regression cases also pass independent coefficient-level endpoint,
continuity, jump, contact and polynomial-separation checks. The focused suite is
**60 passed**, including recovery Jacobians in d=2/3/5/10, a close-knot/reordered
coefficient test, deadline enforcement, contact-removal cycling, and prior tests.
Ruff and whitespace checks pass.

Reproduce from the parent repository:

```sh
MPLCONFIGDIR=/private/tmp/contact-solver-mpl OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=code/contact-solver/src .venv/bin/python \
code/contact-solver/validation/run_fleet20_all50.py \
--output code/contact-solver/validation/fleet20-all50-recovered.json
```

Raw results: fleet20-all50-recovered.json. The baseline and intermediate reports
are retained as historical validation snapshots. No manuscript data or text has
been replaced. This validates the 50 planar N=20 cases, not every experimental
setting, a global optimum, or a rigorous interval separation certificate.
