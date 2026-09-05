#!/usr/bin/env python3
"""Pocket probe, Phase B0: active-set structure inside the pocket.

Axis family r(0) = (dA, 0), r(T) = (-dB, 0), zero boundary velocities,
depths inside the symmetric 2-3 graze anchor d_g = 1.3447448 (F13
quartic). Structure-agnostic LiftedSCP solve, then the three questions:

    - geometry:  does the near-active set form ONE interval (arc), TWO
                 intervals (lift-off), or isolated points (3+ contacts)?
                 (active_width_classifier + active interval at fixed eps
                 + max interior gap inside the active interval)
    - duals:     atoms vs density under K-refinement: an atom keeps O(1)
                 mass at ~1 node as K grows; an arc density carries mass
                 ~ Lambda h per node. Junction signature of the
                 atom-arc-atom ansatz = O(1) edge masses bracketing an
                 interior density whose per-node mass shrinks ~ 1/K.
    - energy:    J_nd per (depth, K), for later comparison against the
                 Phase A high-precision BVP.

Reuses phase_diagram end to end: run_scp_class (OSQP retry, timeout,
duals), via_point_guess (upper-bypass initialization: axis data is
exactly mirror-degenerate, the guess pins the upper branch; F7 gives
the tie exactly, so classifying one branch loses nothing), and the
regime classifiers. New here: the depth x K campaign loop, per-node
profile CSVs, edge/interior dual-mass accounting, refinement overlays.

Usage (from contact-solver/):
    python -m contact_solver.cli.pocket_probe --quick
    python -m contact_solver.cli.pocket_probe
    python -m contact_solver.cli.pocket_probe --d 1.34 1.20 --K 200 1000
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from contact_solver import load_config
from contact_solver.cli.phase_diagram import (
    R_ND,
    run_scp_class,
    via_point_guess,
)
from contact_solver.regime import (
    _connected_clusters,
    active_width_classifier,
    dual_mass_classifier,
    fuse_regime,
)

D_GRAZE = 1.3447448104892225  # symmetric 2-3 graze (F13 quartic root)

DEFAULT_DEPTHS = [1.34, 1.32, 1.30, 1.27, 1.24, 1.20, 1.15, 1.10, 1.05]
DEFAULT_K = [200, 1000, 4000]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def eps_ladder(det_tol: float, n: int = 10) -> np.ndarray:
    """Width-classifier eps grid with the floor at ~3x the solver slack
    (below that, widths measure tolerance noise, not geometry)."""
    lo = np.log10(3.0 * det_tol)
    return np.logspace(lo, -1.0, n)


def active_interval(times, dist, eps):
    """Largest connected {d <= R + eps} component: (t_lo, t_hi, width),
    plus the max interior gap max(d - R) INSIDE that component (the
    lift-off detector: a genuine two-arc solution keeps an interior
    gap that does not shrink with K), and the number of components."""
    mask = dist <= R_ND + eps
    comps = _connected_clusters(mask)
    if not comps:
        return dict(t_lo=np.nan, t_hi=np.nan, width=0.0,
                    interior_gap=np.nan, n_components=0)
    s, e = max(comps, key=lambda c: c[1] - c[0])
    interior_gap = float(np.max(dist[s:e + 1]) - R_ND) if e > s else 0.0
    return dict(t_lo=float(times[s]), t_hi=float(times[e]),
                width=float(times[e] - times[s]),
                interior_gap=interior_gap, n_components=len(comps))


def dual_cluster_stats(times, duals, n_edge: int = 3):
    """Edge/interior mass accounting on the heaviest dual cluster.

    Returns per-node-resolution quantities: total mass, mass in the
    n_edge outermost nodes at each end (junction-atom detector: stays
    O(1) under refinement), interior mass and max interior node mass
    (density detector: max node mass shrinks ~ 1/K)."""
    lam = np.clip(np.asarray(duals, float), 0.0, None)
    if lam.max() < 1e-12:
        return None
    active = lam > 1e-4 * lam.max()
    comps = _connected_clusters(active)
    if not comps:
        return None
    s, e = max(comps, key=lambda c: float(lam[c[0]:c[1] + 1].sum()))
    seg = lam[s:e + 1]
    n = len(seg)
    ne = min(n_edge, n)
    mass = float(seg.sum())
    left = float(seg[:ne].sum())
    right = float(seg[-ne:].sum())
    has_interior = n > 2 * ne
    return dict(
        dm_t_lo=float(times[s]), dm_t_hi=float(times[e]), dm_n_nodes=n,
        dm_mass=mass, dm_edge_left=left, dm_edge_right=right,
        dm_interior=float(mass - left - right) if has_interior else 0.0,
        dm_interior_max_node=float(seg[ne:-ne].max()) if has_interior else 0.0,
        dm_max_share=float(seg.max() / mass),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_case(sol, rec, det_tol, path: Path, title: str):
    """3-panel diagnostic: trajectory, log gap profile, dual profile."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(title, fontsize=10)

    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(R_ND * np.cos(th), R_ND * np.sin(th), color="#EEE", zorder=0)
    ax.plot(R_ND * np.cos(th), R_ND * np.sin(th), "k--", lw=0.8)
    ax.plot(sol["rel"][:, 0], sol["rel"][:, 1], lw=1.6, color="#F58518")
    ax.plot(*sol["rel"][0], "g^", ms=8)
    ax.plot(*sol["rel"][-1], "rv", ms=8)
    ax.set_aspect("equal")
    ax.set_title("relative trajectory r(t)")

    ax = axes[1]
    gap = np.maximum(sol["dist"] - R_ND, 1e-9)
    ax.semilogy(sol["times"], gap, lw=1.0, color="#4C78A8")
    ax.axhline(det_tol, color="k", ls="--", lw=0.8, label="detection tol")
    if np.isfinite(rec.get("t_lo", np.nan)):
        ax.axvspan(rec["t_lo"], rec["t_hi"], color="#F58518", alpha=0.15,
                   label="active interval")
    ax.set_xlabel("t/T")
    ax.set_ylabel("d(t) - R")
    ax.legend(fontsize=7)
    ax.set_title(f"gap  [{rec['regime']}]  interior gap "
                 f"{rec.get('interior_gap', np.nan):.1e}")

    ax = axes[2]
    if sol["duals"] is not None:
        lam = np.clip(sol["duals"], 0, None)
        ax.plot(sol["times"], lam, lw=0.9, color="#4C78A8")
        for key, x in (("dm_t_lo", rec.get("dm_t_lo")),
                       ("dm_t_hi", rec.get("dm_t_hi"))):
            if x is not None and np.isfinite(x):
                ax.axvline(x, color="#F58518", lw=0.7, alpha=0.6)
        ax.set_title(
            f"duals: edges {rec.get('dm_edge_left', 0):.3g}/"
            f"{rec.get('dm_edge_right', 0):.3g}  interior "
            f"{rec.get('dm_interior', 0):.3g}  share "
            f"{rec.get('dm_max_share', np.nan):.2f}")
    else:
        ax.text(0.5, 0.5, "no duals", ha="center", transform=ax.transAxes)
    ax.set_xlabel("t/T")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_refinement(depth_key, sols, out_path: Path):
    """Per-depth K-refinement overlay: per-node dual mass (atoms stay
    O(1), density shrinks ~ 1/K) and density estimate lam * K (density
    collapses onto one curve, atoms grow ~ K)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.suptitle(f"dual-mass refinement, {depth_key}", fontsize=10)
    for K, sol in sorted(sols.items()):
        if sol["duals"] is None:
            continue
        lam = np.clip(sol["duals"], 0, None)
        axes[0].plot(sol["times"], lam, lw=0.9, label=f"K={K}")
        axes[1].plot(sol["times"], lam * K, lw=0.9, label=f"K={K}")
    axes[0].set_ylabel("per-node dual mass")
    axes[1].set_ylabel(r"density estimate $\lambda \cdot K$")
    for ax in axes:
        ax.set_xlabel("t/T")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_summary(df: pd.DataFrame, out_path: Path):
    """Active-interval width vs depth at the finest K per depth."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fin = df.loc[df.groupby("dA")["K"].idxmax()]
    fin = fin.sort_values("dA")
    ax.plot(fin["dA"], fin["width"], "o-", color="#4C78A8", ms=5)
    ax.axvline(D_GRAZE, color="k", ls="--", lw=0.8,
               label=r"$d_g$ (graze quartic)")
    ax.set_xlabel(r"$d_A$")
    ax.set_ylabel("active-interval width (t/T)")
    ax.legend(fontsize=8)
    ax.set_title("arc width vs depth (finest K)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Pocket probe (Phase B0)")
    ap.add_argument("--config", type=str, default="pocket")
    ap.add_argument("--d", type=float, nargs="+", default=DEFAULT_DEPTHS,
                    help="axis depths dA (default: pocket ladder "
                         f"{DEFAULT_DEPTHS}; graze at {D_GRAZE:.4f})")
    ap.add_argument("--dB", type=float, nargs="+", default=None,
                    help="dB values for an asymmetric axis slice: a "
                         "single value is broadcast over --d (dA list); "
                         "a list of the same length as --d is paired "
                         "elementwise; default: dB = dA (symmetric "
                         "family). For a fixed-dA slice give one --d "
                         "and a --dB list.")
    ap.add_argument("--K", type=int, nargs="+", default=DEFAULT_K,
                    help=f"K refinement ladder (default {DEFAULT_K})")
    ap.add_argument("--output-dir", type=str, default="0_pocket/run1")
    ap.add_argument("--quick", action="store_true",
                    help="sanity run: d = 1.30, K = 200 1000")
    args = ap.parse_args()

    if args.quick:
        args.d, args.K = [1.30], [200, 1000]
        args.output_dir = "0_pocket/quick"

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: could not load config '{args.config}': {e}")
        sys.exit(2)

    out = Path(args.output_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "profiles").mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out / "config_used.yaml")

    det_tol = cfg.solver.scp_detection_tol
    T = cfg.problem.time_horizon
    eps_rel = eps_ladder(det_tol)
    Ks = sorted(args.K)
    z = np.zeros(2)

    # ── resolve (dA, dB) pairs
    if args.dB is None:
        pairs = [(dA, dA) for dA in args.d]
    elif len(args.dB) == 1:
        pairs = [(dA, args.dB[0]) for dA in args.d]
    elif len(args.d) == 1:
        pairs = [(args.d[0], dB) for dB in args.dB]
    elif len(args.dB) == len(args.d):
        pairs = list(zip(args.d, args.dB))
    else:
        print("ERROR: --dB must be length 1, or match --d, or --d must "
              "be a single value")
        sys.exit(2)

    n_total = len(pairs) * len(Ks)
    print(f"pocket probe: {len(pairs)} depths x {len(Ks)} K levels "
          f"-> {n_total} solves -> {out}")
    print(f"  graze anchor d_g = {D_GRAZE:.10f}; depths beyond it are "
          f"pocket points")

    rows = []
    t_start = time.time()
    n_done = 0
    for dA, dB in pairs:
        depth_key = f"dA{dA:g}_dB{dB:g}"
        r0, rT = np.array([dA, 0.0]), np.array([-dB, 0.0])
        rel_prev = None
        sols = {}
        for K in Ks:
            n_done += 1
            guess = rel_prev if rel_prev is not None \
                else via_point_guess(r0, rT, +1, K)
            diag = {}
            t0 = time.time()
            sol = run_scp_class(cfg, r0, z, rT, z, +1, K,
                                rel_guess=guess, diag=diag)
            dt = time.time() - t0
            if sol is None:
                print(f"  [{n_done}/{n_total}] {depth_key} K={K}: "
                      f"FAILED ({diag})  t={dt:6.1f}s")
                rows.append(dict(dA=dA, dB=dB, K=K, regime="scp_failed",
                                 fail="; ".join(f"{k}={v}"
                                                for k, v in diag.items())))
                pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
                continue
            rel_prev = sol["rel"]
            sols[K] = sol

            # ── classifiers + pocket diagnostics
            wcl = active_width_classifier(sol["times"], sol["dist"], R_ND,
                                          eps_rel=eps_rel)
            dcl = (dual_mass_classifier(sol["times"], sol["duals"])
                   if sol["duals"] is not None else [])
            regime = fuse_regime(wcl, dcl)
            ai = active_interval(sol["times"], sol["dist"], 3.0 * det_tol)
            ds = (dual_cluster_stats(sol["times"], sol["duals"])
                  if sol["duals"] is not None else None)

            rec = dict(
                dA=dA, dB=dB, K=K, regime=regime,
                cost_solve=sol["cost"], cost_nd=sol["cost"] * T**3,
                min_distance=sol["min_distance"],
                strict=sol.get("strict", True),
                eps_scale=sol.get("eps_scale", 1.0),
                dua_res=(sol["dual_quality"]["dua_res"]
                         if sol["dual_quality"] else np.nan),
                width_slopes=";".join(f"{c.slope:.2f}" for c in wcl),
                width_labels=";".join(c.label for c in wcl),
                dual_shares=";".join(f"{c.max_share:.2f}" for c in dcl),
                time_s=dt, fail="",
                **ai, **(ds or {}),
            )
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

            # ── per-node profile CSV (t, gap, dual) — the raw material
            # for any later analysis
            prof = pd.DataFrame({
                "t": sol["times"],
                "x": sol["rel"][:, 0], "y": sol["rel"][:, 1],
                "gap": sol["dist"] - R_ND,
                "dual": (np.clip(sol["duals"], 0, None)
                         if sol["duals"] is not None
                         else np.full(K, np.nan)),
            })
            prof.to_csv(out / "profiles" / f"{depth_key}_K{K}.csv",
                        index=False)

            plot_case(sol, rec, det_tol,
                      out / "plots" / f"{depth_key}_K{K}.png",
                      title=(f"{depth_key} K={K}: {regime}  "
                             f"J_nd={rec['cost_nd']:.6g}  "
                             f"width={ai['width']:.4f}  "
                             f"components={ai['n_components']}"))

            print(f"  [{n_done}/{n_total}] {depth_key} K={K}: "
                  f"{regime:<12} width={ai['width']:.4f} "
                  f"comps={ai['n_components']} "
                  f"edges={rec.get('dm_edge_left', 0):.3g}/"
                  f"{rec.get('dm_edge_right', 0):.3g} "
                  f"int={rec.get('dm_interior', 0):.3g} "
                  f"t={dt:6.1f}s")

        if sols:
            plot_refinement(depth_key, sols,
                            out / "plots" / f"refine_{depth_key}.png")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    ok = df[df["regime"] != "scp_failed"]
    if len(ok):
        plot_summary(ok, out / "plots" / "summary_width_vs_depth.png")

    print("\n" + "=" * 64)
    print(f"done: {n_done} solves in {(time.time() - t_start) / 60:.1f} min")
    if len(ok):
        piv = ok.pivot_table(index=["dA", "dB"], columns="K",
                             values="regime", aggfunc="first")
        print("regime labels (rows: dA, dB; cols: K):")
        print(piv.to_string())
    print(f"results: {out / 'results.csv'}")
    print(f"plots:   {out / 'plots'}")


if __name__ == "__main__":
    main()
