#!/usr/bin/env python3
"""Sweep over discretization K for fixed N.

Compares LiftedSCP at varying discretization granularity against the
continuous-time ContactSolver (SCI). For each eligible seed (inherited
from a prior sweep_n run with both solvers succeeded and homotopy classes
agreed at K=100), SCI is solved once and resampled at every K in the
sweep, while SCP is solved fresh at each K.

The SCP solver tolerances are held fixed at the K=100 values from
configs/sweep_n.yaml; the sweep measures how SCP's cost converges to
SCI's reference and how its runtime grows with K, all on the same
problem instances.

Append mode (--append) extends an existing CSV with new K values or new
seeds, running only the missing (N, seed, K) cells. SCI is re-solved per
seed because solver state is not cached on disk, but no SCP work is
duplicated.
"""

import argparse
import datetime as _dt
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contact_solver import (
    ContactSolver,
    LiftedSCP,
    generate_random_positions,
    load_config,
)
from contact_solver.config import Config
from contact_solver.cli._sweep_utils import (
    build_scaled_config,
    compare_braid_words,
    compute_braid_word,
    compute_metrics,
    compute_scp_min_dist_continuous,
    load_eligible_seeds,
    run_solver_with_timeout,
)


def run_seed_K_sweep(
    base_config: Config,
    N: int,
    seed: int,
    K_values: list[int],
    timeout: float,
    K_sci_reference: int,
) -> list[dict]:
    """For a single seed, solve SCI once and run SCP across the given K values.

    SCI is continuous-time, so its solve does not depend on K. The
    `K_sci_reference` argument only controls the timestep field of the
    config passed to ContactSolver (irrelevant to the solver itself);
    set it to max(K_values) so resampling at any K is interpolation, not
    extrapolation past the natural sample grid.

    In append mode, the caller is responsible for filtering K_values to
    only those missing from the existing CSV for this seed.

    Returns one result dict per K in K_values, schema matching sweep_n.
    """
    # ── Solve SCI once (K-independent) ──────────────────────────
    config_sci = build_scaled_config(base_config, N, seed, K=K_sci_reference)
    np.random.seed(config_sci.random_seed)
    initial, final = generate_random_positions(config_sci)

    sci_solver = ContactSolver(config_sci, verbose=False)
    sci_solver.set_initial_states(initial)
    sci_solver.set_final_states(final)

    try:
        sci_result, sci_timeout = run_solver_with_timeout(sci_solver, timeout)
    except Exception as e:
        # Catastrophic SCI failure: emit one row per K marked as failed
        rows = []
        for K in K_values:
            rows.append({
                "N": N, "seed": seed, "K": K,
                "contact_success": False,
                "contact_reason": "error",
                "contact_error": str(e),
            })
        return rows

    if sci_timeout or not sci_result["metrics"]["converged"]:
        reason = "timeout" if sci_timeout else sci_result["metrics"]["convergence_reason"]
        rows = []
        for K in K_values:
            rows.append({
                "N": N, "seed": seed, "K": K,
                "contact_success": False,
                "contact_reason": reason,
                "contact_time": timeout if sci_timeout else sci_result["metrics"]["timing"]["total_time"],
            })
        return rows

    sci_time = sci_result["metrics"]["timing"]["total_time"]
    sci_num_contacts = sci_result["metrics"]["num_contacts"]
    sci_max_per_pair = sci_result["metrics"]["max_contacts_single_pair"]

    # Continuous-time SCI cost (K-independent)
    T = config_sci.problem.time_horizon
    sci_acc_cont = sci_result["trajectories"]["accelerations"]
    dt_cont = T / (len(sci_acc_cont[0]) - 1)
    sci_cost_continuous = sum(
        np.trapezoid(np.sum(a**2, axis=1), dx=dt_cont) for a in sci_acc_cont
    )

    # ── SCP across K, SCI resampled per K ───────────────────────
    rows = []
    for K in K_values:
        config_K = build_scaled_config(base_config, N, seed, K=K)

        # Paranoia: confirm seed + N alone determine scenario, independent of K
        np.random.seed(config_K.random_seed)
        init_K, final_K = generate_random_positions(config_K)
        assert np.allclose(init_K, initial), (
            f"Seed reproducibility broken at N={N} seed={seed} K={K}: "
            "initial positions differ from K_sci_reference build"
        )
        assert np.allclose(final_K, final), (
            f"Seed reproducibility broken at N={N} seed={seed} K={K}: "
            "final positions differ from K_sci_reference build"
        )

        row = {"N": N, "seed": seed, "K": K}

        # ── SCI evaluated at this K's grid ─────────────────────
        h_K = config_K.problem.timestep
        scp_times = np.arange(1, K + 1) * h_K
        _, sci_pos_K, _, sci_acc_K = sci_solver.sample(times=scp_times)
        sci_traj_K = {"positions": sci_pos_K, "accelerations": sci_acc_K}
        sci_metrics_K = compute_metrics(sci_traj_K, config_K)

        row.update({
            "contact_success": True,
            "contact_reason": "converged",
            "contact_time": sci_time,                       # constant in K
            "contact_contacts": sci_num_contacts,
            "contact_max_contacts_single_pair": sci_max_per_pair,
            "contact_cost": sci_metrics_K["acceleration_cost"],   # K-dependent re-integration
            "contact_cost_continuous": sci_cost_continuous,       # K-independent
            "contact_min_dist": sci_metrics_K["min_distance"],
        })

        # ── SCP at this K ──────────────────────────────────────
        scp_solver = LiftedSCP(config_K, verbose=False)
        scp_solver.set_initial_states(initial)
        scp_solver.set_final_states(final)

        try:
            scp_result, scp_timeout = run_solver_with_timeout(scp_solver, timeout)
        except Exception as e:
            row.update({
                "scp_success": False,
                "scp_reason": "error",
                "scp_time": np.nan,
                "scp_error": str(e),
            })
            rows.append(row)
            continue

        if scp_timeout:
            row.update({
                "scp_success": False,
                "scp_reason": "timeout",
                "scp_time": timeout,
            })
            rows.append(row)
            continue

        scp_metrics = compute_metrics(scp_result["trajectories"], config_K)
        reason = scp_result["metrics"]["convergence_reason"]
        collision_free = scp_result["metrics"]["collision_check"]["all_satisfied"]
        if reason == "collision_free_initial":
            reason = "converged"
        if reason == "converged" and not collision_free:
            reason = "infeasible"
        success = reason == "converged"

        row.update({
            "scp_success": success,
            "scp_reason": reason,
            "scp_time": scp_result["metrics"]["timing"]["total_time"],
            "scp_iterations": scp_result["metrics"]["scp_iterations"],
            "scp_cost": scp_metrics["acceleration_cost"],
            "scp_min_dist": scp_metrics["min_distance"],
            "scp_worst_violation": scp_result["metrics"]["collision_check"]["worst_violation"],
        })

        # Continuous-time min distance under SCP's own piecewise-constant-
        # acceleration integrator.  Recorded for every SCP run (success or
        # not) so the panel can show inter-sample violations even on solutions
        # SCP itself reported as feasible at its grid points.
        try:
            scp_pos_full = scp_result["trajectories"]["positions"]
            scp_vel_full = scp_result["trajectories"]["velocities"]
            scp_acc_full = scp_result["trajectories"]["accelerations"]
            scp_min_dist_continuous = compute_scp_min_dist_continuous(
                scp_pos_full, scp_vel_full, scp_acc_full, h_K,
            )
            row["scp_min_dist_continuous"] = float(scp_min_dist_continuous)
        except Exception as e:
            row["scp_min_dist_continuous"] = np.nan
            row["scp_min_dist_continuous_error"] = str(e)

        # ── Cross-solver metrics (only when SCP succeeded) ─────
        if success:
            scp_pos = scp_result["trajectories"]["positions"]
            sum_sq = 0.0
            max_dev = 0.0
            n_pts = 0
            for i in range(N):
                diff = sci_pos_K[i] - scp_pos[i]
                norms = np.linalg.norm(diff, axis=1)
                sum_sq += np.sum(norms**2)
                max_dev = max(max_dev, np.max(norms))
                n_pts += len(norms)
            row["position_rmse"] = float(np.sqrt(sum_sq / n_pts))
            row["max_position_deviation"] = float(max_dev)

            # Homotopy check at this K
            braid_scp = compute_braid_word(scp_pos)
            braid_sci = compute_braid_word(sci_pos_K)
            cmp_ = compare_braid_words(braid_scp, braid_sci)
            row.update({
                "same_homotopy_class": cmp_["same_homotopy_class"],
                "n_crossings_scp": cmp_["n_crossings_scp"],
                "n_crossings_contact": cmp_["n_crossings_contact"],
            })

        rows.append(row)

    return rows


def _compute_missing_work(
    existing_df: pd.DataFrame | None,
    N: int,
    eligible_seeds: list[int],
    K_values: list[int],
) -> dict[int, list[int]]:
    """Return {seed: [K values still to run]} given the existing CSV.

    The working seed set is the union of `eligible_seeds` (from current
    sweep_n filtering) and seeds already present in the CSV for this N.
    For each seed in the union, any K from `K_values` not present is
    considered missing and gets scheduled.

    Returns a dict, sorted by seed, containing only seeds with non-empty
    missing-K lists.
    """
    csv_seeds: set[int] = set()
    done: set[tuple[int, int]] = set()  # (seed, K) tuples already recorded

    if existing_df is not None and len(existing_df) > 0:
        df_N = existing_df[existing_df["N"] == N]
        for _, r in df_N[["seed", "K"]].iterrows():
            csv_seeds.add(int(r["seed"]))
            done.add((int(r["seed"]), int(r["K"])))

    seed_set = sorted(set(eligible_seeds) | csv_seeds)

    missing: dict[int, list[int]] = {}
    for seed in seed_set:
        todo = [K for K in K_values if (seed, K) not in done]
        if todo:
            missing[seed] = todo
    return missing


def _load_existing_meta(meta_path: Path) -> dict | None:
    """Load existing meta JSON if present, else None."""
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def _write_meta(
    meta_path: Path,
    *,
    N: int,
    K_values_full: list[int],
    K_sci_reference: int,
    eligible_seeds: list[int],
    seeds_in_csv: list[int],
    source_sweep_n_csv: str,
    homotopy_filter_applied: bool,
    timeout: float,
    base_config_name: str,
    append_mode: bool,
    prior_meta: dict | None,
) -> None:
    """Write the sweep_K meta JSON, merging with prior meta if appending."""
    full_seed_set = sorted(set(eligible_seeds) | set(seeds_in_csv))
    meta = {
        "N": N,
        "K_values": sorted(set(K_values_full)),
        "K_sci_reference": K_sci_reference,
        "eligible_seeds": full_seed_set,
        "n_seeds": len(full_seed_set),
        "source_sweep_n_csv": source_sweep_n_csv,
        "homotopy_filter_applied": homotopy_filter_applied,
        "timeout": timeout,
        "base_config": base_config_name,
    }

    history = []
    if prior_meta is not None and "append_history" in prior_meta:
        history = list(prior_meta["append_history"])

    if append_mode:
        history.append({
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "K_values_requested": sorted(set(K_values_full)),
        })
    if history:
        meta["append_history"] = history

    meta_path.write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Sweep LiftedSCP over discretization K for fixed N, "
                    "comparing against continuous-time ContactSolver.",
    )
    parser.add_argument("--n", type=int, required=True,
                        help="Fixed robot count for the sweep.")
    parser.add_argument("--K-values", type=int, nargs="+",
                        default=[25, 50, 100, 200, 400, 800],
                        help="K values to sweep (log-spaced recommended). "
                             "In --append mode, only missing K values relative "
                             "to the existing CSV are run.")
    parser.add_argument("--K-sci-reference", type=int, default=None,
                        help="K used only to derive timestep for the SCI solve "
                             "(SCI itself is continuous-time). "
                             "Defaults to max of (current --K-values "
                             "∪ K values already in CSV in append mode).")
    parser.add_argument("--sweep-n-csv", type=str,
                        default="0_sweep_n/run_K100/sweep_n_results.csv",
                        help="Source CSV for seed eligibility filtering. "
                             "Defaults to the K=100 run snapshot.")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Optional cap on number of eligible seeds. "
                             "Ignored in --append mode (the union of CSV and "
                             "eligible seeds is used).")
    parser.add_argument("--no-homotopy-filter", action="store_true",
                        help="Disable homotopy-agreement filter "
                             "(still requires both solvers succeed).")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-solver timeout in seconds.")
    parser.add_argument("--output", type=str,
                        default="0_sweep_K/sweep_K_results.csv")
    parser.add_argument("--config", type=str, default="sweep_n",
                        help="Base config name (reuse sweep_n by default).")
    parser.add_argument("--append", action="store_true",
                        help="Extend an existing CSV: load it, identify "
                             "(N, seed, K) cells already present, and run only "
                             "the missing ones. Existing rows are preserved.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load base config ───────────────────────────────────────
    try:
        base_config = load_config(args.config)
    except FileNotFoundError:
        from contact_solver import get_small_config
        base_config = get_small_config()

    # ── Seed eligibility filter ────────────────────────────────
    eligible = load_eligible_seeds(
        Path(args.sweep_n_csv),
        N=args.n,
        require_homotopy_agreement=not args.no_homotopy_filter,
    )

    # ── Append-mode setup ──────────────────────────────────────
    existing_df: pd.DataFrame | None = None
    existing_K: list[int] = []
    existing_seeds: list[int] = []
    prior_meta: dict | None = None

    if args.append:
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            if len(existing_df) > 0:
                # Validate N consistency
                csv_N_values = existing_df["N"].dropna().unique().astype(int).tolist()
                if csv_N_values and (len(csv_N_values) > 1 or csv_N_values[0] != args.n):
                    raise SystemExit(
                        f"--append: existing CSV has N={csv_N_values}, "
                        f"refusing to append N={args.n}."
                    )
                df_N = existing_df[existing_df["N"] == args.n]
                existing_K = sorted(df_N["K"].dropna().unique().astype(int).tolist())
                existing_seeds = sorted(df_N["seed"].dropna().unique().astype(int).tolist())
            else:
                existing_df = None  # treat empty CSV as missing
        prior_meta = _load_existing_meta(
            output_path.with_name(f"{output_path.stem}_meta.json")
        )
        if args.seeds is not None:
            print("--append: ignoring --seeds (using union of CSV and eligible seeds).")
    else:
        if args.seeds is not None:
            eligible = eligible[: args.seeds]

    if not eligible and existing_df is None:
        raise SystemExit(
            f"No eligible seeds for N={args.n} in {args.sweep_n_csv} "
            f"(homotopy_filter={'off' if args.no_homotopy_filter else 'on'})"
        )

    # ── Determine K_sci_reference ──────────────────────────────
    K_values_full = sorted(set(args.K_values) | set(existing_K))
    if args.K_sci_reference is not None:
        K_sci_reference = args.K_sci_reference
    else:
        K_sci_reference = max(K_values_full)

    # ── Compute the work to do ─────────────────────────────────
    missing = _compute_missing_work(
        existing_df, args.n, eligible, args.K_values,
    )
    seeds_to_run = sorted(missing.keys())
    n_cells_to_run = sum(len(v) for v in missing.values())
    n_cells_skipped = (
        (len(existing_K) * len(existing_seeds)) if args.append else 0
    )

    # ── Save config + metadata ─────────────────────────────────
    config_output = output_path.with_name(f"{output_path.stem}_config.yaml")
    if not (args.append and config_output.exists()):
        # Preserve the snapshot from the original run; only write fresh on
        # non-append or when missing.
        base_config.to_yaml(config_output)

    meta_output = output_path.with_name(f"{output_path.stem}_meta.json")
    _write_meta(
        meta_output,
        N=args.n,
        K_values_full=K_values_full,
        K_sci_reference=K_sci_reference,
        eligible_seeds=eligible,
        seeds_in_csv=existing_seeds,
        source_sweep_n_csv=str(args.sweep_n_csv),
        homotopy_filter_applied=not args.no_homotopy_filter,
        timeout=args.timeout,
        base_config_name=args.config,
        append_mode=args.append,
        prior_meta=prior_meta,
    )

    print("=" * 60)
    print(f"Sweep K: LiftedSCP discretization sensitivity"
          f"{' [APPEND]' if args.append else ''}")
    print("=" * 60)
    print(f"N             : {args.n}")
    print(f"K values (req): {sorted(set(args.K_values))}")
    if args.append:
        print(f"K values (csv): {existing_K}")
        print(f"K values (all): {K_values_full}")
    print(f"K_sci_ref     : {K_sci_reference}")
    if args.append:
        print(f"Seeds in CSV  : {len(existing_seeds)}")
        print(f"Eligible seeds: {len(eligible)}")
        print(f"Seeds to run  : {len(seeds_to_run)}")
        print(f"Cells to run  : {n_cells_to_run}  (skipping {n_cells_skipped} already done)")
    else:
        print(f"Eligible seeds: {len(eligible)}")
        print(f"  → {eligible}")
    print(f"Homotopy filt.: {'OFF' if args.no_homotopy_filter else 'ON'}")
    print(f"Timeout       : {args.timeout}s")
    print(f"Output        : {args.output}")
    print(f"Config        : {config_output}")
    print(f"Metadata      : {meta_output}")
    print()

    if not missing:
        print("Nothing to do — every (seed, K) cell is already in the CSV.")
        return

    # ── Main sweep loop ────────────────────────────────────────
    # Start from the existing CSV's rows in append mode, then append new ones.
    all_rows: list[dict] = []
    if existing_df is not None and len(existing_df) > 0:
        all_rows = existing_df.to_dict(orient="records")

    start = time.time()

    for i_seed, seed in enumerate(seeds_to_run):
        K_for_this_seed = missing[seed]
        print(f"\n[{i_seed + 1}/{len(seeds_to_run)}] seed={seed}  "
              f"running K={K_for_this_seed}")

        rows = run_seed_K_sweep(
            base_config=base_config,
            N=args.n,
            seed=seed,
            K_values=K_for_this_seed,
            timeout=args.timeout,
            K_sci_reference=K_sci_reference,
        )
        all_rows.extend(rows)

        # Per-K status lines for this seed
        for r in rows:
            K = r["K"]
            scp_ok = r.get("scp_success", False)
            scp_t = r.get("scp_time", float("nan"))
            sci_ok = r.get("contact_success", False)
            scp_cost = r.get("scp_cost")
            sci_cost = r.get("contact_cost")
            if scp_cost is not None and sci_cost is not None and sci_cost > 0:
                gap = (scp_cost - sci_cost) / sci_cost
                gap_str = f"gap={gap:+.4f}"
            else:
                gap_str = "gap=NA"

            line = (
                f"  K={K:4d}: "
                f"SCI={'✓' if sci_ok else '✗'} "
                f"SCP={'✓' if scp_ok else '✗'} ({scp_t:.2f}s)  {gap_str}"
            )
            if not scp_ok:
                line += f"  [{r.get('scp_reason', '?')}]"
            print(line)

        # Intermediate save: sort by seed, K for a stable, human-readable file
        df_save = pd.DataFrame(all_rows)
        if "seed" in df_save.columns and "K" in df_save.columns:
            df_save = df_save.sort_values(["seed", "K"]).reset_index(drop=True)
        df_save.to_csv(args.output, index=False)

    elapsed = time.time() - start

    # ── Summary ────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 60)
    print("Summary (entire CSV, including pre-existing rows)")
    print("=" * 60)

    for K in K_values_full:
        sub = df[df["K"] == K]
        n_total = len(sub)
        scp_ok = int(sub["scp_success"].sum()) if "scp_success" in sub else 0
        sci_ok = int(sub["contact_success"].sum()) if "contact_success" in sub else 0

        both = sub[(sub.get("scp_success", False) == True)
                   & (sub.get("contact_success", False) == True)]
        if len(both) > 0 and "scp_cost" in both and "contact_cost" in both:
            gaps = ((both["scp_cost"] - both["contact_cost"])
                    / both["contact_cost"]).dropna()
            scp_med_t = both["scp_time"].median()
            sci_med_t = both["contact_time"].median()
            print(
                f"K = {K:4d}  "
                f"SCP {scp_ok}/{n_total}  SCI {sci_ok}/{n_total}  "
                f"median gap = {gaps.median():+.4f}  "
                f"(IQR {gaps.quantile(0.25):+.4f} … {gaps.quantile(0.75):+.4f})  "
                f"SCP {scp_med_t:.3f}s  SCI {sci_med_t:.3f}s"
            )

            scp_reasons = sub.loc[sub["scp_success"] == False, "scp_reason"]
            if len(scp_reasons) > 0:
                counts = Counter(scp_reasons)
                breakdown = ", ".join(f"{r}: {c}" for r, c in counts.items())
                print(f"            SCP failures: {breakdown}")
        else:
            print(f"K = {K:4d}  SCP {scp_ok}/{n_total}  SCI {sci_ok}/{n_total}")

    print(f"\nResults saved to: {args.output}")
    print(f"Total time: {elapsed:.1f}s  ({n_cells_to_run} cells executed)")


if __name__ == "__main__":
    main()
