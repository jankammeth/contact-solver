#!/usr/bin/env python3
"""Phase-diagram campaign for the two-robot active-set regimes (G4).

Grid search over the nondimensionalized boundary conditions of the
two-robot problem (R = 1, T = 1, r(0) on the +x axis). Reuses the
existing infrastructure end to end:

    - config:    configs/phase_diagram.yaml via load_config
                 (edit the YAML, not this file, to change tolerances)
    - solvers:   LiftedSCP (structure-agnostic PRIMARY) and
                 ContactSolver (SCI, cross-check), as in compare-solvers
    - execution: run_solver_with_timeout, compute_metrics
                 from cli/_sweep_utils

New in this module (and only this):
    - nondimensional BC grid generators (the "normalization")
    - homotopy-class guard: LiftedSCP initialized from a left/right
      bypass path (LiftedSCPWithGuess), winner classified
    - active-set regime detection (contact_solver.regime)
    - diagnostic plots per instance + aggregate phase diagram

Usage:
    phase-diagram --grid 7 --K 200 1000 --plot-instances 5
    phase-diagram --smoke        # 4 hand-built verification instances
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contact_solver import LiftedSCP, load_config
from contact_solver.cli._sweep_utils import (
    compute_metrics,
    run_solver_with_timeout,
)
from contact_solver.regime import (
    active_width_classifier,
    dual_mass_classifier,
    fuse_regime,
    is_counterexample,
    winding_class,
)

R_ND = 1.0
T_ND = 1.0

C_REGIME = {
    "0": "#DDDDDD",
    "1": "#4C78A8",
    "2": "#59A14F",
    "arc": "#F58518",
}
C_FLAG = "#E15759"


# ---------------------------------------------------------------------------
# Normalization: nondimensional BC grids (new)
# ---------------------------------------------------------------------------


def split_bcs(r0, v0, rT, vT):
    """Relative BCs -> absolute two-robot BCs (symmetric split)."""
    p0 = np.array([+0.5 * np.asarray(r0), -0.5 * np.asarray(r0)])
    pf = np.array([+0.5 * np.asarray(rT), -0.5 * np.asarray(rT)])
    w0 = np.array([+0.5 * np.asarray(v0), -0.5 * np.asarray(v0)])
    wf = np.array([+0.5 * np.asarray(vT), -0.5 * np.asarray(vT)])
    return p0, w0, pf, wf


def layer1_instances(grid: int, d0_list=(1.5,), rT_box=4.0):
    """Zero-velocity slice: r(0) = (d0, 0); rT on a grid restricted to
    the upper half plane (reflection symmetry), skipping infeasible rT
    inside the disk. The y = 0 row is shifted to y = 0.05: exactly
    mirror-symmetric data makes the QP dual degenerate (see
    run_scp_class docstring); the tie line is measure-zero and is
    traced from just off-axis instead."""
    for d0 in d0_list:
        for x in np.linspace(-rT_box, rT_box, grid):
            for y in np.linspace(0.0, rT_box, max(grid // 2 + 1, 2)):
                if y == 0.0:
                    y = 0.05
                rT = np.array([x, y])
                if np.linalg.norm(rT) < R_ND * 1.05:
                    continue
                yield (np.array([d0, 0.0]), np.zeros(2), rT, np.zeros(2))


def layer2_instances(grid: int, d0: float, rT, lam_max: float,
                     alpha0: float = 0.0, alphaT: float = 0.0):
    """Tangential-velocity slice at fixed geometry: r(0) = (d0, 0),
    r(T) = rT; sweep signed magnitudes (lam0, lamT) on a grid along
    fixed directions. alpha0/alphaT (radians) tilt the directions off
    the tangent: 0 = purely tangential, negative = inward (toward the
    disk), positive = outward. dir = cos(a)*that + sin(a)*rhat.
    Both lambdas are NONDIMENSIONAL velocities (converted to solve
    units inside classify_instance). No reflection reduction: fixing
    rT off the axis breaks the mirror symmetry."""
    rT = np.asarray(rT, float)
    rhat0 = np.array([1.0, 0.0])          # r(0) on the +x axis
    that0 = np.array([0.0, 1.0])
    rhatT = rT / np.linalg.norm(rT)
    thatT = np.array([-rhatT[1], rhatT[0]])
    dir0 = np.cos(alpha0) * that0 + np.sin(alpha0) * rhat0
    dirT = np.cos(alphaT) * thatT + np.sin(alphaT) * rhatT
    for lam0 in np.linspace(-lam_max, lam_max, grid):
        for lamT in np.linspace(-lam_max, lam_max, grid):
            yield (np.array([d0, 0.0]), lam0 * dir0, rT.copy(), lamT * dirT)


def smoke_instances():
    """Hand-built verification cases with (loosely) known regimes."""
    z = np.zeros(2)
    return [
        ("A_no_contact", np.array([2.0, 0.0]), z, np.array([2.0, 2.0]), z),
        ("B_single_contact", np.array([2.0, 0.0]), z, np.array([-2.0, 0.9]), z),
        ("C_antipodal", np.array([2.0, 0.0]), z, np.array([-2.0, 0.0]), z),
        ("D_tight_swap", np.array([1.2, 0.0]), z, np.array([-1.2, 0.05]), z),
    ]


def augment_candidates(df_prev: pd.DataFrame, M: int, explore_frac: float = 0.15,
                       rT_box: float = 4.0, min_sep: float = 0.02,
                       seed: int = 0):
    """Adaptive refinement instances via Delaunay boundary-edge bisection.

    Every Delaunay edge whose endpoints carry different regime labels
    crosses a regime boundary; the budget M is spent on the midpoints
    of the longest such edges (jittered), so sampling concentrates on
    all under-resolved boundary stretches at once and uniform regions
    receive nothing. A small exploration fraction goes to uniform
    random points as insurance against missed regions (the analytic
    prefilter makes any landing in the provably-0 region free).

    scp_failed rows are label-less (excluded); "ambiguous..." is its
    own label so ambiguity boundaries refine until they resolve.
    """
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    z = np.zeros(2)
    d0 = float(df_prev["r0x"].iloc[0])
    labeled = df_prev[df_prev["regime_scp"].astype(str) != "scp_failed"]
    pts = labeled[["rTx", "rTy"]].to_numpy(dtype=float)
    labels = labeled["regime_scp"].astype(str).to_numpy()

    def admissible(p, existing):
        if np.linalg.norm(p) < R_ND * 1.05 or p[1] < 0.0:
            return False
        if abs(p[0]) > rT_box or p[1] > rT_box:
            return False
        return np.min(np.linalg.norm(existing - p[None, :], axis=1)) >= min_sep

    cands = []
    existing = pts.copy()

    n_explore = int(round(explore_frac * M))
    n_interior = int(round(0.25 * M))
    n_boundary = M - n_explore - n_interior

    # ── boundary-edge bisection
    if len(pts) >= 4:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                i, j = sorted((simplex[a], simplex[(a + 1) % 3]))
                edges.add((i, j))
        boundary_edges = [
            (np.linalg.norm(pts[i] - pts[j]), i, j)
            for i, j in edges
            if labels[i] != labels[j]
            and labels[i] != "0" and labels[j] != "0"
            # 0<->contact edges are excluded: at zero velocities the
            # 0-boundary is the chord-tangency locus, known in closed
            # form (drawn analytically in the phase plot); the budget
            # goes entirely to boundaries between contact regimes.
        ]
        boundary_edges.sort(reverse=True)  # longest first
        e_idx = 0
        while len(cands) < n_boundary and e_idx < len(boundary_edges):
            length, i, j = boundary_edges[e_idx]
            e_idx += 1
            mid = 0.5 * (pts[i] + pts[j])
            p = mid + rng.normal(0.0, length / 6.0, 2)
            if not admissible(p, existing):
                p = mid  # fall back to the exact midpoint
                if not admissible(p, existing):
                    continue
            cands.append(p)
            existing = np.vstack([existing, p])

    # ── interior of the rarest contact region: uniform points whose
    # nearest labeled sample carries that label (confirms the region
    # interior and, for "arc", supplies G3-usable instances).
    contact_labels = [l for l in np.unique(labels) if l != "0"]
    if contact_labels and n_interior > 0:
        from scipy.spatial import cKDTree
        rarest = min(contact_labels, key=lambda l: (labels == l).sum())
        tree = cKDTree(pts)
        added, tries = 0, 0
        while added < n_interior and tries < 200 * n_interior:
            tries += 1
            p = rng.uniform([-rT_box, 0.0], [rT_box, rT_box])
            if not admissible(p, existing):
                continue
            _, i_near = tree.query(p)
            if labels[i_near] != rarest:
                continue
            cands.append(p)
            existing = np.vstack([existing, p])
            added += 1

    # ── uniform exploration for the remainder of the budget
    tries = 0
    while len(cands) < M and tries < 100 * M:
        tries += 1
        p = rng.uniform([-rT_box, 0.0], [rT_box, rT_box])
        if admissible(p, existing):
            cands.append(p)
            existing = np.vstack([existing, p])

    return [(f"A_{n:05d}", np.array([d0, 0.0]), z, np.asarray(p), z)
            for n, p in enumerate(cands)]


# ---------------------------------------------------------------------------
# Homotopy-class guard (new): side-selecting initial guess for LiftedSCP
# ---------------------------------------------------------------------------


def loop_guess(r0, rT, w: int, K: int, r_min: float = 1.3):
    """Initial relative path in winding class w: angle interpolated
    along the lifted arc theta0 -> theta0 + principal + 2 pi w, radius
    interpolated |r0| -> |rT| with a floor of r_min * R away from the
    endpoints (keeps the seed clear of the disk). w = 0 reduces to a
    near-direct sweep; w = -1 / +1 are the loop classes (E-G2-a)."""
    r0, rT = np.asarray(r0, float), np.asarray(rT, float)
    th0 = np.arctan2(r0[1], r0[0])
    dth = np.arctan2(rT[1], rT[0]) - th0
    dth = np.arctan2(np.sin(dth), np.cos(dth)) + 2.0 * np.pi * w
    s = np.linspace(0.0, 1.0, K)
    th = th0 + s * dth
    rad = (1 - s) * np.linalg.norm(r0) + s * np.linalg.norm(rT)
    bump = np.clip(r_min * R_ND - rad, 0.0, None) * np.sin(np.pi * s)
    rad = rad + bump
    return np.stack([rad * np.cos(th), rad * np.sin(th)], axis=1)


def via_point_guess(r0, rT, side: int, K: int, bypass_r: float = 1.5) -> np.ndarray:
    """(K, 2) relative-trajectory guess routing around the disk.

    Samples the straight chord; wherever the chord enters the circle of
    radius bypass_r * R, that span is replaced by angular interpolation
    along the circle, traversed counterclockwise for side = +1 and
    clockwise for side = -1.
    """
    r0, rT = np.asarray(r0, float), np.asarray(rT, float)
    rb = bypass_r * R_ND
    ts = np.linspace(0.0, 1.0, K)
    chord = r0[None, :] * (1 - ts[:, None]) + rT[None, :] * ts[:, None]
    d = np.linalg.norm(chord, axis=1)
    inside = d < rb
    if not inside.any():
        return chord
    i0 = int(np.argmax(inside))
    i1 = int(len(ts) - 1 - np.argmax(inside[::-1]))
    a_in = np.arctan2(chord[max(i0 - 1, 0), 1], chord[max(i0 - 1, 0), 0])
    a_out = np.arctan2(chord[min(i1 + 1, K - 1), 1],
                       chord[min(i1 + 1, K - 1), 0])
    dth = a_out - a_in
    if side > 0:
        dth = dth % (2 * np.pi) or 2 * np.pi
    else:
        dth = dth % (2 * np.pi) - 2 * np.pi
        if dth == 0.0:
            dth = -2 * np.pi
    guess = chord.copy()
    span = np.arange(i0, i1 + 1)
    frac = (span - (i0 - 1)) / (i1 + 1 - (i0 - 1))
    ang = a_in + frac * dth
    guess[span, 0] = rb * np.cos(ang)
    guess[span, 1] = rb * np.sin(ang)
    return guess


class LiftedSCPWithGuess(LiftedSCP):
    """LiftedSCP whose first linearization point is a supplied relative
    path (used for the homotopy-class guard; force_scp_iterations keeps
    the constrained loop running even when the guess is feasible)."""

    def set_relative_guess(self, rel_guess: np.ndarray):
        self._rel_guess = rel_guess
        self.force_scp_iterations = True

    def _solve_initial(self):
        rel = self._rel_guess  # (K, 2)
        x = np.zeros(self.total_vars)
        pos = {0: +0.5 * rel, 1: -0.5 * rel}
        for i in range(self.N):
            p = pos[i]
            v = np.gradient(p, self.h, axis=0)
            a = np.gradient(v, self.h, axis=0)
            for k in range(self.K):
                idx = self._get_var_indices(i, k)
                x[idx["px"]], x[idx["py"]] = p[k]
                x[idx["vx"]], x[idx["vy"]] = v[k]
                x[idx["ax"]], x[idx["ay"]] = a[k]
        return x, {"status": "guess"}


# ---------------------------------------------------------------------------
# Per-instance protocol (reuses config/timeout/metrics infrastructure)
# ---------------------------------------------------------------------------


def _config_at_K(base_config, K: int):
    T = base_config.problem.time_horizon
    return base_config.override(**{"problem.timestep": T / K})


def chord_clearance(r0, rT) -> float:
    """Min distance from the origin to the segment [r0, rT].

    For zero boundary velocities the unconstrained relative minimizer
    traces exactly this segment (cubic time-parameterization of the
    straight chord), so chord_clearance >= R proves regime 0 without
    solving anything.
    """
    r0, rT = np.asarray(r0, float), np.asarray(rT, float)
    d = rT - r0
    denom = float(d @ d)
    s = 0.0 if denom < 1e-14 else float(np.clip(-(r0 @ d) / denom, 0.0, 1.0))
    return float(np.linalg.norm(r0 + s * d))


def run_scp_class(base_config, r0, v0, rT, vT, side, K, rel_guess=None,
                  diag=None):
    """Solve one homotopy class with SCP; None on failure/timeout.

    Single solve at the config settings, exactly as compare-solvers
    runs LiftedSCP; on an OSQP max-iter failure (degenerate-adjacent
    instances), one retry at 10x eps. A run that exhausts SCP
    iterations with a collision-free final iterate is accepted
    (strict=False): the classifiers read geometry and duals, not the
    convergence bit.

    diag: optional dict, mutated with last-attempt failure info
    ("why", "osqp", "min_d", "scp_reason") for logging.
    """
    p0, w0, pf, wf = split_bcs(r0, v0, rT, vT)
    if rel_guess is None:
        guess = via_point_guess(r0, rT, side, K)
    else:
        ts_new = np.linspace(0.0, 1.0, K)
        ts_old = np.linspace(0.0, 1.0, len(rel_guess))
        guess = np.stack(
            [np.interp(ts_new, ts_old, rel_guess[:, c]) for c in (0, 1)],
            axis=1,
        )

    for eps_scale in (1.0, 10.0):
        cfg = _config_at_K(base_config, K)
        if eps_scale != 1.0:
            cfg = cfg.override(**{
                "solver.osqp_eps_abs": cfg.solver.osqp_eps_abs * eps_scale,
                "solver.osqp_eps_rel": cfg.solver.osqp_eps_rel * eps_scale,
            })
        solver = LiftedSCPWithGuess(cfg)
        solver.set_initial_states(p0, w0)
        solver.set_final_states(pf, wf)
        solver.set_relative_guess(guess)
        try:
            out, timed_out = run_solver_with_timeout(
                solver, cfg.solver.timeout or 120.0
            )
        except RuntimeError as e:
            if diag is not None:
                diag.update(why="osqp", osqp=str(e)[:120])
            continue  # OSQP failure -> one retry at 10x eps
        if timed_out or out is None:
            if diag is not None:
                diag.update(why="timeout")
            return None
        traj = out["trajectories"]
        m = compute_metrics(traj, cfg)
        strict = bool(out["metrics"]["converged"])
        feasible = m["min_distance"] >= R_ND - cfg.solver.scp_detection_tol
        if not (strict or feasible):
            if diag is not None:
                diag.update(
                    why="infeasible_stall",
                    min_d=float(m["min_distance"]),
                    scp_reason=out["metrics"]["convergence_reason"],
                )
            continue
        rel = traj["positions"][0] - traj["positions"][1]
        # normalized time t/T (classification is invariant under the
        # time rescaling; the solve runs at the config horizon)
        times = np.arange(K) / K
        w, _ = winding_class(rel)
        duals, dual_q = solver.get_collision_duals()
        return {
            "times": times,
            "rel": rel,
            "dist": np.linalg.norm(rel, axis=1),
            "cost": float(m["acceleration_cost"]),
            "min_distance": float(m["min_distance"]),
            "winding": w,
            "converged": True,
            "strict": strict,
            "duals": None if duals is None else duals[0],
            "dual_quality": dual_q,
            "eps_scale": eps_scale,
        }
    return None


def classify_instance(base_config, r0, v0, rT, vT, K_levels=(200,),
                      allow_two=False, forced_guess=None):
    """SCP-only protocol for one instance. Returns (record, artifacts).

    v0/vT are NONDIMENSIONAL boundary velocities (r/R per t/T); the
    solve runs at the config horizon T with R = 1, so they are
    converted by the factor R/T before entering the solver.

    forced_guess: optional (K, 2) relative path; when given, the class
    guard is skipped and the single solve is initialized from it
    (classifies THAT homotopy class, not the direct-class minimizer;
    method='forced_class' in the record).
    """
    rec = {"r0x": r0[0], "r0y": r0[1], "rTx": rT[0], "rTy": rT[1],
           "v0x": v0[0], "v0y": v0[1], "vTx": vT[0], "vTy": vT[1],
           "method": "solved"}
    K_hi = max(K_levels)
    T = base_config.problem.time_horizon
    v0_s = np.asarray(v0, float) * R_ND / T
    vT_s = np.asarray(vT, float) * R_ND / T

    # ── analytic prefilter (zero velocities): if the straight chord
    # clears the disk, the unconstrained minimizer is feasible, hence
    # optimal -> regime 0 is exact, no solve needed.
    if (forced_guess is None
            and np.allclose(v0, 0.0) and np.allclose(vT, 0.0)
            and chord_clearance(r0, rT) >= R_ND):
        delta = np.asarray(rT, float) - np.asarray(r0, float)
        rec.update(method="analytic", regime_scp="0", flag="",
                   cost=6.0 * float(delta @ delta) / T**3,
                   winding=0, side_gap=0.0, K_used=0, strict=True,
                   min_distance=chord_clearance(r0, rT), eps_scale=0.0)
        return rec, None

    # ── class guard: both direct classes, exactly two solves — unless a
    # forced guess pins the class.
    diags = {+1: {}, -1: {}}
    if forced_guess is not None:
        rec["method"] = "forced_class"
        coarse = {+1: run_scp_class(base_config, r0, v0_s, rT, vT_s, +1,
                                    min(K_levels), rel_guess=forced_guess,
                                    diag=diags[+1]),
                  -1: None}
    else:
        coarse = {s: run_scp_class(base_config, r0, v0_s, rT, vT_s, s,
                                   min(K_levels), diag=diags[s])
                  for s in (+1, -1)}
    ok = [s for s in (+1, -1) if coarse[s] and coarse[s]["converged"]]
    if not ok:
        rec.update(regime_scp="scp_failed", flag="scp_failed",
                   cost=np.nan, winding=0,
                   fail_p1="; ".join(f"{k}={v}" for k, v in diags[+1].items()),
                   fail_m1="; ".join(f"{k}={v}" for k, v in diags[-1].items()))
        return rec, None
    best_side = min(ok, key=lambda s: coarse[s]["cost"])
    rec["side_gap"] = (abs(coarse[+1]["cost"] - coarse[-1]["cost"])
                       if len(ok) == 2 else np.nan)

    # class-unresolved guard: if an excluded (non-usable) side still
    # holds a clearly cheaper trajectory, the classified class may be
    # the wrong one -> flag rather than silently classify.
    unresolved = any(
        coarse[s] is not None and not coarse[s]["converged"]
        and coarse[s]["cost"] < 0.98 * coarse[best_side]["cost"]
        for s in (+1, -1)
    )

    # coarse-to-fine continuation: the high-K solve is warm-started from
    # the interpolated coarse solution (ADMM iteration counts explode at
    # high K from a crude guess; from the near-optimal coarse solution
    # they don't). Skipped entirely when only one K level is given.
    if K_hi == min(K_levels):
        scp, rec["K_used"] = coarse[best_side], K_hi
    else:
        scp = run_scp_class(base_config, r0, v0_s, rT, vT_s, best_side, K_hi,
                            rel_guess=coarse[best_side]["rel"])
        if scp is None or not scp["converged"]:
            scp, rec["K_used"] = coarse[best_side], min(K_levels)
        else:
            rec["K_used"] = K_hi
    rec["cost"], rec["winding"] = scp["cost"], scp["winding"]
    rec["min_distance"] = scp["min_distance"]
    rec["eps_scale"] = scp.get("eps_scale", 1.0)
    rec["strict"] = scp.get("strict", True)

    # ── SCP classifiers
    wcl = active_width_classifier(scp["times"], scp["dist"], R_ND)
    dcl = (dual_mass_classifier(scp["times"], scp["duals"])
           if scp["duals"] is not None else [])
    rec["regime_scp"] = fuse_regime(wcl, dcl)
    rec["width_slopes"] = ";".join(f"{c.slope:.2f}" for c in wcl)
    rec["dual_shares"] = ";".join(f"{c.max_share:.2f}" for c in dcl)
    rec["dua_res"] = (scp["dual_quality"]["dua_res"]
                      if scp["dual_quality"] else np.nan)
    rec["osqp_status"] = (scp["dual_quality"]["status"]
                          if scp["dual_quality"] else "")

    if unresolved:
        rec["flag"] = "class_unresolved"
    elif is_counterexample(rec["regime_scp"], allow_two=allow_two):
        rec["flag"] = "counterexample"
    elif rec["regime_scp"].startswith("ambiguous"):
        rec["flag"] = "ambiguous"
    else:
        rec["flag"] = ""
    art = {"scp": scp, "width": wcl, "dual": dcl,
           "coarse": coarse, "best_side": best_side}
    return rec, art


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_instance(rec, art, path: Path, title: str):
    """4-panel diagnostic: trajectory, distance, width scaling, duals."""
    scp = art["scp"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(title, fontsize=10)

    ax = axes[0, 0]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(R_ND * np.cos(th), R_ND * np.sin(th), color="#EEE", zorder=0)
    ax.plot(R_ND * np.cos(th), R_ND * np.sin(th), "k--", lw=0.8)
    for side, sol in art["coarse"].items():
        if sol is None:
            continue
        style = (dict(lw=2.0, color="#F58518") if side == art["best_side"]
                 else dict(lw=0.9, color="#BBB"))
        ax.plot(sol["rel"][:, 0], sol["rel"][:, 1], **style,
                label=f"side {side:+d} (J={sol['cost']:.3g}, w={sol['winding']})")
    for c in art["width"]:
        i = np.argmin(np.abs(scp["times"] - c.t_center))
        ax.plot(*scp["rel"][i], "o", ms=6,
                color=C_REGIME["1"] if c.label == "point" else C_FLAG)
    ax.plot(*scp["rel"][0], "g^", ms=8)
    ax.plot(*scp["rel"][-1], "rv", ms=8)
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    ax.set_title("relative trajectory r(t)")

    ax = axes[0, 1]
    ax.plot(scp["times"], scp["dist"], lw=1.2, color="#4C78A8")
    ax.axhline(R_ND, color="k", ls="--", lw=0.8)
    for c in art["width"]:
        ax.axvline(c.t_center, color="#F58518", lw=0.7, alpha=0.6)
        ax.text(c.t_center, ax.get_ylim()[1], c.label, fontsize=7,
                rotation=90, va="top")
    ax.set_ylim(R_ND * 0.95, None)
    ax.set_title(f"d(t)  [SCP={rec['regime_scp']}"
                 f"  strict={rec.get('strict', True)}]")
    ax.set_xlabel("t")

    ax = axes[1, 0]
    for c in art["width"]:
        pos = c.widths > 0
        if pos.any():
            ax.loglog(c.eps[pos], c.widths[pos], "o-", ms=4,
                      label=f"t={c.t_center:.2f} slope={c.slope:.2f} → {c.label}")
    eps_ref = np.logspace(-2.5, -1.0, 8) * R_ND
    ax.loglog(eps_ref, 0.5 * np.sqrt(eps_ref), "k--", lw=0.7,
              label="slope 1/2 ref")
    ax.legend(fontsize=7)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"active width $w(\varepsilon)$")
    ax.set_title("active-width scaling")

    ax = axes[1, 1]
    if scp["duals"] is not None:
        lam = np.clip(scp["duals"], 0, None)
        ax.plot(scp["times"], lam, lw=0.9, color="#4C78A8")
        # shade the physical event span(s); annotate per-detector votes
        # (the FUSED label is in the figure title; width and dual are
        # individual votes that fuse_regime combines)
        for wc in art["width"]:
            ax.axvspan(wc.t_lo, wc.t_hi,
                       color=C_REGIME.get("arc" if wc.label == "arc" else "1"),
                       alpha=0.12)
            ax.text(wc.t_center, 0.98 * (lam.max() if lam.max() > 0 else 1.0),
                    f"width vote: {wc.label} ({wc.slope:.2f})",
                    fontsize=7, ha="center", va="top")
        # sub-clusters of mu inside an event: junction atoms / interior density
        for c in art["dual"]:
            ax.axvline(c.t_center, color="#F58518", lw=0.6, alpha=0.5)
            role = "atom" if c.label == "atom" else (
                "junction" if c.max_share >= 0.1 else "interior")
            ax.text(c.t_center, 0.75 * (lam.max() if lam.max() > 0 else 1.0),
                    f"dual vote: {c.label}\n({role}, {c.max_share:.2f})",
                    fontsize=6, ha="center", va="top", color="#666")
        ax.set_title(f"collision duals ≈ μ  (dua_res={rec['dua_res']:.1e})")
    else:
        ax.text(0.5, 0.5, "no collision QP solved", ha="center",
                transform=ax.transAxes)
    ax.set_xlabel("t")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_phase_diagram(df: pd.DataFrame, out_path: Path):
    """Layer-1 slice(s): per d0, two panels — (left) rT samples colored
    by regime, (right) shaded regions via nearest-sample lookup."""
    from matplotlib.colors import ListedColormap

    d0s = sorted(df["r0x"].unique())
    fig, axes = plt.subplots(len(d0s), 2, figsize=(10.5, 4.8 * len(d0s)),
                             squeeze=False)

    def draw_geometry(ax, d0):
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(R_ND * np.cos(th), R_ND * np.sin(th), "k--", lw=0.8, zorder=3)
        # analytic 0-boundary (zero velocities): regime 0 iff the chord
        # [r0, rT] clears the disk -> two tangent half-lines from r(0).
        if d0 > R_ND:
            beta = np.arccos(R_ND / d0)
            for sgn in (+1, -1):
                q = R_ND * np.array([np.cos(sgn * beta), np.sin(sgn * beta)])
                u = q - np.array([d0, 0.0])
                u = u / np.linalg.norm(u)
                far = q + 12.0 * u
                ax.plot([q[0], far[0]], [q[1], far[1]],
                        color="#777777", lw=0.9, zorder=3)
        ax.plot([d0], [0], "k*", ms=12, zorder=4, label=r"$r(0)$")
        ax.set_aspect("equal")
        ax.set_xlim(-4.3, 4.3)
        ax.set_ylim(-4.3, 4.3)

    for row, d0 in enumerate(d0s):
        sub = df[df["r0x"] == d0]
        # reflection symmetry across the x-axis: upper half computed,
        # full plane displayed via the mirror image.
        pts = sub[["rTx", "rTy"]].to_numpy(dtype=float)
        labels = sub["regime_scp"].astype(str).to_numpy()
        keep = labels != "scp_failed"
        pts_full = np.vstack([pts[keep], pts[keep] * [1, -1]])
        labels_full = np.concatenate([labels[keep], labels[keep]])

        # ── left: samples
        ax = axes[row][0]
        draw_geometry(ax, d0)
        for regime, color in C_REGIME.items():
            m = labels_full == regime
            if not m.any():
                continue
            ax.scatter(pts_full[m, 0], pts_full[m, 1], c=color, s=18,
                       label=regime, edgecolors="none", zorder=2)
        m_other = ~np.isin(labels_full, list(C_REGIME.keys()))
        if m_other.any():
            ax.scatter(pts_full[m_other, 0], pts_full[m_other, 1], c=C_FLAG,
                       s=28, marker="x", label="flag/other", zorder=2)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"$\\|r(0)\\| = {d0:g}$ — samples")

        # ── right: shaded regions (linear one-hot interpolation, argmax)
        ax = axes[row][1]
        if len(pts_full):
            from scipy.interpolate import griddata

            order = list(C_REGIME.keys()) + ["other"]
            colors = list(C_REGIME.values()) + [C_FLAG]
            lab_ids = np.array([
                order.index(l) if l in C_REGIME else len(order) - 1
                for l in labels_full
            ])
            g = np.linspace(-4.3, 4.3, 480)
            X, Y = np.meshgrid(g, g)
            grid_pts = np.column_stack([X.ravel(), Y.ravel()])
            # linear interpolation of each class indicator over the
            # Delaunay triangulation, then argmax: boundaries become
            # piecewise-linear curves halfway between opposing samples
            # (and thin regions bounded by their own samples survive,
            # unlike k-NN majority smoothing). Nearest-neighbor fill
            # outside the convex hull.
            fields = []
            for c in range(len(order)):
                onehot = (lab_ids == c).astype(float)
                f = griddata(pts_full, onehot, grid_pts, method="linear")
                f_nn = griddata(pts_full, onehot, grid_pts, method="nearest")
                fields.append(np.where(np.isnan(f), f_nn, f))
            Z = np.argmax(np.stack(fields, axis=0), axis=0)
            Z = Z.reshape(X.shape).astype(float)
            Z[np.hypot(X, Y) < R_ND] = np.nan  # mask the disk interior
            ax.pcolormesh(X, Y, Z, cmap=ListedColormap(colors),
                          vmin=-0.5, vmax=len(order) - 0.5, alpha=0.6,
                          shading="auto", zorder=1)
        draw_geometry(ax, d0)
        ax.set_title(f"$\\|r(0)\\| = {d0:g}$ — regions")

    fig.suptitle("Two-robot regime phase diagram (Layer 1, zero velocities; "
                 "SCP-primary labels)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_phase_diagram_layer2(df: pd.DataFrame, out_path: Path,
                              d0: float, rT, alpha0: float = 0.0,
                              alphaT: float = 0.0):
    """Velocity-plane slice at fixed geometry: (lam0, lamT) colored by
    regime; same two panels as Layer 1 (samples + shaded regions)."""
    from matplotlib.colors import ListedColormap
    from scipy.interpolate import griddata

    rT = np.asarray(rT, float)
    rhatT = rT / np.linalg.norm(rT)
    thatT = np.array([-rhatT[1], rhatT[0]])
    dir0 = (np.cos(alpha0) * np.array([0.0, 1.0])
            + np.sin(alpha0) * np.array([1.0, 0.0]))
    dirT = np.cos(alphaT) * thatT + np.sin(alphaT) * rhatT
    lam0 = (df["v0x"].to_numpy(dtype=float) * dir0[0]
            + df["v0y"].to_numpy(dtype=float) * dir0[1])
    lamT = (df["vTx"].to_numpy(dtype=float) * dirT[0]
            + df["vTy"].to_numpy(dtype=float) * dirT[1])
    labels = df["regime_scp"].astype(str).to_numpy()
    keep = labels != "scp_failed"
    pts = np.column_stack([lam0, lamT])[keep]
    labels = labels[keep]
    lim = 1.05 * np.abs(pts).max() if len(pts) else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), squeeze=False)

    ax = axes[0][0]
    for regime, color in C_REGIME.items():
        m = labels == regime
        if not m.any():
            continue
        ax.scatter(pts[m, 0], pts[m, 1], c=color, s=18, label=regime,
                   edgecolors="none", zorder=2)
    m_other = ~np.isin(labels, list(C_REGIME.keys()))
    if m_other.any():
        ax.scatter(pts[m_other, 0], pts[m_other, 1], c=C_FLAG, s=28,
                   marker="x", label="flag/other", zorder=2)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel(r"$\lambda_0$ (tangential, nondim.)")
    ax.set_ylabel(r"$\lambda_T$ (tangential, nondim.)")
    ax.set_aspect("equal")
    ax.set_title("samples")

    ax = axes[0][1]
    if len(pts) >= 4:
        order = list(C_REGIME.keys()) + ["other"]
        colors = list(C_REGIME.values()) + [C_FLAG]
        lab_ids = np.array([
            order.index(l) if l in C_REGIME else len(order) - 1
            for l in labels
        ])
        g = np.linspace(-lim, lim, 480)
        X, Y = np.meshgrid(g, g)
        grid_pts = np.column_stack([X.ravel(), Y.ravel()])
        fields = []
        for c in range(len(order)):
            onehot = (lab_ids == c).astype(float)
            f = griddata(pts, onehot, grid_pts, method="linear")
            f_nn = griddata(pts, onehot, grid_pts, method="nearest")
            fields.append(np.where(np.isnan(f), f_nn, f))
        Z = np.argmax(np.stack(fields, axis=0), axis=0).reshape(X.shape)
        ax.pcolormesh(X, Y, Z.astype(float), cmap=ListedColormap(colors),
                      vmin=-0.5, vmax=len(order) - 0.5, alpha=0.6,
                      shading="auto", zorder=1)
    ax.set_xlabel(r"$\lambda_0$")
    ax.set_ylabel(r"$\lambda_T$")
    ax.set_aspect("equal")
    ax.set_title("regions")

    fig.suptitle(f"Layer 2 — velocity slice ($d_0$={d0:g}, "
                 f"$r_T$=({rT[0]:g}, {rT[1]:g}), "
                 f"$\\alpha_0$={np.degrees(alpha0):g}°, "
                 f"$\\alpha_T$={np.degrees(alphaT):g}°)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="G4 phase-diagram campaign")
    ap.add_argument("--config", type=str, default="phase_diagram",
                    help="Config name (loads configs/phase_diagram.yaml)")
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--d0", type=float, nargs="+", default=[1.5])
    ap.add_argument("--K", type=int, nargs="+", default=[200],
                    help="SCP K levels (max = classification level)")
    ap.add_argument("--plot-instances", type=int, default=5)
    ap.add_argument("--output-dir", type=str, default="0_phase_diagram/run1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--layer2", action="store_true",
                    help="tangential-velocity slice at fixed geometry: "
                         "grid over (lam0, lamT); regime 2 is legitimate")
    ap.add_argument("--rT", type=float, nargs=2, default=[-1.5, 0.05],
                    help="fixed endpoint for --layer2")
    ap.add_argument("--lam-max", type=float, default=4.0,
                    help="velocity-magnitude range for --layer2 "
                         "(nondimensional)")
    ap.add_argument("--alpha", type=float, nargs=2, default=[0.0, 0.0],
                    metavar=("A0", "AT"),
                    help="velocity direction tilt in degrees for --layer2: "
                         "0 = tangential, negative = inward toward the "
                         "disk, positive = outward")
    ap.add_argument("--single", type=float, nargs=8, default=None,
                    metavar=("R0X", "R0Y", "RTX", "RTY",
                             "V0X", "V0Y", "VTX", "VTY"),
                    help="classify one hand-built instance (nondimensional "
                         "boundary values; velocities in r/R per t/T) and "
                         "save its diagnostic panel")
    ap.add_argument("--winding", type=int, default=None,
                    help="with --single: also solve the winding-w class "
                         "from a loop initial guess and classify it "
                         "alongside the direct-class best")
    ap.add_argument("--augment", type=int, default=None, metavar="M",
                    help="adaptive refinement: add M points concentrated "
                         "on the regime boundaries (Delaunay boundary-edge "
                         "bisection) of a previous run; requires "
                         "--from-results")
    ap.add_argument("--from-results", type=str, default=None,
                    help="path to the previous results.csv to augment")
    args = ap.parse_args()

    try:
        base_config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: could not load config '{args.config}': {e}")
        sys.exit(2)

    out = Path(args.output_dir)
    (out / "plots" / "instances").mkdir(parents=True, exist_ok=True)
    base_config.to_yaml(out / "config_used.yaml")

    df_prev = None
    if args.augment:
        if args.layer2:
            print("ERROR: --augment is not supported for --layer2 yet")
            sys.exit(2)
        if not args.from_results:
            print("ERROR: --augment M requires --from-results <results.csv>")
            sys.exit(2)
        df_prev = pd.read_csv(args.from_results)
        instances = augment_candidates(df_prev, M=args.augment)
        print(f"augmenting {args.from_results}: {len(instances)} adaptive "
              f"samples (boundary bisection + exploration)")
    elif args.smoke:
        instances = list(smoke_instances())
    elif args.single:
        s = args.single
        r0s, rTs = np.array(s[0:2]), np.array(s[2:4])
        v0s, vTs = np.array(s[4:6]), np.array(s[6:8])
        instances = [("SINGLE_direct", r0s, v0s, rTs, vTs)]
        if args.winding is not None:
            instances.append((f"SINGLE_w{args.winding:+d}",
                              r0s, v0s, rTs, vTs))
    elif args.layer2:
        a0, aT = np.radians(args.alpha[0]), np.radians(args.alpha[1])
        instances = [(f"L2_{i:05d}", r0, v0, rT, vT)
                     for i, (r0, v0, rT, vT) in enumerate(
                         layer2_instances(args.grid, args.d0[0], args.rT,
                                          args.lam_max, a0, aT))]
    else:
        instances = [(f"L1_{i:05d}", r0, v0, rT, vT)
                     for i, (r0, v0, rT, vT) in enumerate(
                         layer1_instances(args.grid, tuple(args.d0)))]

    print(f"{len(instances)} instances -> {out}")
    rows = []
    t0 = time.time()
    for n_done, (name, r0, v0, rT, vT) in enumerate(instances):
        t_inst = time.time()
        forced = None
        if name.startswith("SINGLE_w"):
            forced = loop_guess(r0, rT, args.winding, min(args.K))
        rec, art = classify_instance(base_config, r0, v0, rT, vT,
                                     K_levels=tuple(args.K),
                                     allow_two=args.layer2 or bool(args.single),
                                     forced_guess=forced)
        dt_inst = time.time() - t_inst
        rec["id"] = name
        rows.append(rec)
        flagged = bool(rec.get("flag"))
        if art is not None and (flagged or n_done < args.plot_instances
                                or args.smoke):
            plot_instance(rec, art, out / "plots" / "instances" / f"{name}.png",
                          title=(f"{name}: r0={np.round(r0, 2)} "
                                 f"rT={np.round(rT, 2)} "
                                 f"SCP={rec['regime_scp']}"
                                 f"{'  ⚑ ' + rec['flag'] if flagged else ''}"))
        print(f"  [{n_done + 1}/{len(instances)}] {name}: "
              f"SCP={rec['regime_scp']:<12} "
              f"{('⚑ ' + rec['flag']) if flagged else '':<18} "
              f"t={dt_inst:6.2f}s")
        if (n_done + 1) % 10 == 0 or n_done + 1 == len(instances):
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    if args.layer2:
        plot_phase_diagram_layer2(df, out / "plots" / "phase_diagram_layer2.png",
                                  d0=args.d0[0], rT=args.rT,
                                  alpha0=np.radians(args.alpha[0]),
                                  alphaT=np.radians(args.alpha[1]))
    elif args.augment and df_prev is not None:
        df_all = pd.concat([df_prev, df], ignore_index=True)
        df_all.to_csv(out / "results_combined.csv", index=False)
        plot_phase_diagram(df_all, out / "plots" / "phase_diagram_layer1.png")
    elif not args.smoke and not args.single:
        plot_phase_diagram(df, out / "plots" / "phase_diagram_layer1.png")

    n_flag = int((df["flag"] != "").sum())
    print("\n" + "=" * 60)
    print(f"done: {len(df)} instances in {(time.time() - t0) / 60:.1f} min, "
          f"flags {n_flag}")
    print(df["regime_scp"].value_counts().to_string())
    print(f"results: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
