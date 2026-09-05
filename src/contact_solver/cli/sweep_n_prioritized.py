#!/usr/bin/env python3
"""Sweep comparison of PrioritizedSCI vs PrioritizedSCP across robot counts.

Mirrors sweep_n.py one-to-one, with the two solvers being the prioritized
counterparts of the joint solvers there:

    sweep_n:                sweep_n_prioritized:
      LiftedSCP    (scp_*)    PrioritizedSCP  (prio_scp_*)
      ContactSolver(contact_*) PrioritizedSCI (prio_sci_*)

Cost / runtime ratios are computed directly between the two prioritized
solvers; there is no joint baseline run.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contact_solver import (
    PrioritizedSCI,
    PrioritizedSCP,
    generate_random_positions,
    load_config,
)
from contact_solver.config import Config
from contact_solver.cli._sweep_utils import (
    build_scaled_config,
    compare_braid_words,
    compute_braid_word,
    compute_metrics,
    run_solver_with_timeout,
)


APPLE_STUDIO_WIDTH_PX = 5120
APPLE_STUDIO_HEIGHT_PX = 2880
PLOT_DPI = 200


# ── Colors (same palette as sweep_n) ─────────────────────────
C_SCP = "#4C78A8"      # Prio SCP — blue
C_SCI = "#F58518"      # Prio SCI — orange


def _maximize_figure_window() -> None:
    """Best-effort fullscreen for interactive backends."""
    manager = plt.get_current_fig_manager()

    if hasattr(manager, "full_screen_toggle"):
        try:
            manager.full_screen_toggle()
            return
        except Exception:
            pass

    window = getattr(manager, "window", None)
    if window is None:
        return

    for method_name in ("showMaximized", "state", "wm_attributes"):
        method = getattr(window, method_name, None)
        if method is None:
            continue
        try:
            if method_name == "state":
                method("zoomed")
            elif method_name == "wm_attributes":
                method("-zoomed", True)
            else:
                method()
            return
        except Exception:
            continue


def _paired_boxplot(ax, n_values, scp_col, sci_col, df, **kwargs):
    """Side-by-side boxplots for the two prioritized solvers."""
    scp_data = [
        df.loc[(df["N"] == n) & (df["prio_scp_success"] == True), scp_col].dropna().to_numpy()
        for n in n_values
    ]
    sci_data = [
        df.loc[(df["N"] == n) & (df["prio_sci_success"] == True), sci_col].dropna().to_numpy()
        for n in n_values
    ]

    valid, labels = [], []
    for i, n in enumerate(n_values):
        if len(scp_data[i]) > 0 and len(sci_data[i]) > 0:
            valid.append((scp_data[i], sci_data[i]))
            labels.append(str(n))

    if not valid:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    pos = np.arange(1, len(valid) + 1)
    off = 0.18

    bp_scp = ax.boxplot(
        [p[0] for p in valid], positions=pos - off, widths=0.32,
        patch_artist=True, manage_ticks=False,
    )
    bp_sci = ax.boxplot(
        [p[1] for p in valid], positions=pos + off, widths=0.32,
        patch_artist=True, manage_ticks=False,
    )
    for p in bp_scp["boxes"]:
        p.set_facecolor(C_SCP)
        p.set_alpha(0.7)
    for p in bp_sci["boxes"]:
        p.set_facecolor(C_SCI)
        p.set_alpha(0.7)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.legend(
        [bp_scp["boxes"][0], bp_sci["boxes"][0]],
        ["Prio SCP", "Prio SCI"],
        loc=kwargs.get("legend_loc", "upper left"),
        fontsize=8,
    )


def create_runtime_dashboard(df: pd.DataFrame, plot_output: str, show_plot: bool = False) -> None:
    """Create and save a 4x3 dashboard mirroring sweep_n's layout."""
    width_in = APPLE_STUDIO_WIDTH_PX / PLOT_DPI
    height_in = APPLE_STUDIO_HEIGHT_PX / PLOT_DPI * 4 / 3
    fig, axes = plt.subplots(4, 3, figsize=(width_in, height_in), dpi=PLOT_DPI)
    fig.suptitle("Prioritized Sweep Comparison Dashboard", fontsize=22)

    n_values = sorted(df["N"].dropna().unique().astype(int).tolist())

    # ── Panel 1: Runtime boxplot ─────────────────────────────
    ax = axes[0, 0]
    _paired_boxplot(ax, n_values, "prio_scp_time", "prio_sci_time", df)
    ax.set_yscale("log")
    ax.set_title("Runtime")
    ax.set_xlabel("N")
    ax.set_ylabel("Time [s]")
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Runtime median + IQR ────────────────────────
    ax = axes[0, 1]
    grouped_iqr = (
        df.loc[(df["prio_scp_success"] == True) | (df["prio_sci_success"] == True)]
        .groupby("N", as_index=False)
        .agg(
            scp_median=("prio_scp_time", lambda x: x[df.loc[x.index, "prio_scp_success"] == True].median()),
            scp_q25=("prio_scp_time", lambda x: x[df.loc[x.index, "prio_scp_success"] == True].quantile(0.25)),
            scp_q75=("prio_scp_time", lambda x: x[df.loc[x.index, "prio_scp_success"] == True].quantile(0.75)),
            scp_q025=("prio_scp_time", lambda x: x[df.loc[x.index, "prio_scp_success"] == True].quantile(0.025)),
            scp_q975=("prio_scp_time", lambda x: x[df.loc[x.index, "prio_scp_success"] == True].quantile(0.975)),
            sci_median=("prio_sci_time", lambda x: x[df.loc[x.index, "prio_sci_success"] == True].median()),
            sci_q25=("prio_sci_time", lambda x: x[df.loc[x.index, "prio_sci_success"] == True].quantile(0.25)),
            sci_q75=("prio_sci_time", lambda x: x[df.loc[x.index, "prio_sci_success"] == True].quantile(0.75)),
            sci_q025=("prio_sci_time", lambda x: x[df.loc[x.index, "prio_sci_success"] == True].quantile(0.025)),
            sci_q975=("prio_sci_time", lambda x: x[df.loc[x.index, "prio_sci_success"] == True].quantile(0.975)),
        )
        .sort_values("N")
    )

    if len(grouped_iqr) > 0:
        x = grouped_iqr["N"].to_numpy(dtype=float)
        for col, color, marker, label in [
            ("scp", C_SCP, "o", "Prio SCP"),
            ("sci", C_SCI, "s", "Prio SCI"),
        ]:
            median = grouped_iqr[f"{col}_median"].to_numpy(dtype=float)
            q25 = grouped_iqr[f"{col}_q25"].to_numpy(dtype=float)
            q75 = grouped_iqr[f"{col}_q75"].to_numpy(dtype=float)
            q025 = grouped_iqr[f"{col}_q025"].to_numpy(dtype=float)
            q975 = grouped_iqr[f"{col}_q975"].to_numpy(dtype=float)
            ax.plot(x, median, color=color, lw=2.2, marker=marker, label=label)
            ax.fill_between(x, q025, q975, color=color, alpha=0.08)
            ax.fill_between(x, q25, q75, color=color, alpha=0.18)
        ax.legend(fontsize=8)

    ax.set_yscale("log")
    ax.set_title("Runtime (Median + IQR)")
    ax.set_xlabel("N")
    ax.set_ylabel("Time [s]")
    ax.grid(True, alpha=0.25)

    # ── Panel 3: Speedup diverging bars (Prio SCP / Prio SCI) ─
    ax = axes[0, 2]
    grouped_speedup = (
        df.loc[(df["prio_scp_success"] == True) & (df["prio_sci_success"] == True)]
        .groupby("N", as_index=False)
        .agg(
            scp_median=("prio_scp_time", "median"),
            sci_median=("prio_sci_time", "median"),
        )
        .sort_values("N")
    )

    if len(grouped_speedup) > 0:
        x = np.arange(len(grouped_speedup))
        n_labels = grouped_speedup["N"].astype(int).astype(str).tolist()
        speedup_pct = (
            grouped_speedup["scp_median"].to_numpy(dtype=float)
            / grouped_speedup["sci_median"].to_numpy(dtype=float)
            - 1.0
        ) * 100.0
        colors = ["#59A14F" if value >= 0 else "#E15759" for value in speedup_pct]

        ax.bar(x, speedup_pct, color=colors, alpha=0.85, width=0.65)
        ax.axhline(0.0, color="black", lw=1.2, alpha=0.8)

        max_abs = float(np.max(np.abs(speedup_pct))) if len(speedup_pct) > 0 else 0.0
        pad = max(5.0, 0.1 * max_abs)
        limit_pad = max(8.0, 0.2 * max_abs)
        ax.set_ylim(-(max_abs + limit_pad), max_abs + limit_pad)
        ax.set_xticks(x)
        ax.set_xticklabels(n_labels)

        for xi, value in zip(x, speedup_pct, strict=True):
            ratio = (
                grouped_speedup["scp_median"].to_numpy(dtype=float)[xi]
                / grouped_speedup["sci_median"].to_numpy(dtype=float)[xi]
            )
            factor = ratio if ratio >= 1.0 else 1.0 / ratio
            label = f"{factor:.2f}x"
            if value >= 0:
                y_text = value + pad * 0.2
                va = "bottom"
            else:
                y_text = value - pad * 0.2
                va = "top"
            ax.text(xi, y_text, label, ha="center", va=va, fontsize=8)

    ax.set_title("Median Speedup of Prio SCI vs Prio SCP")
    ax.set_xlabel("N")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(False, axis="y")

    # ── Panel 4: Success/failure stacked bars ────────────────
    ax = axes[1, 0]
    x_arr = np.arange(len(n_values), dtype=float)
    bar_w = 0.34

    failure_colors = {
        "success": "#59A14F",
        "max_iterations": "#8FAADC",
        "rank_max_iterations": "#8FAADC",
        "rank_osqp_failed": "#E15759",
        "joint_violation_after_ranks_converged": "#B4C7E7",
        "osqp_failed": "#E15759",
        "infeasible": "#B4C7E7",
        "timeout": "#D6E4F0",
        "max_contacts": "#F4B183",
        "max_contacts_per_pair": "#F8CBAD",
        "error": "#BAB0AC",
    }

    scp_fail_reasons_all = sorted(
        set(df.loc[df.get("prio_scp_success", False) == False, "prio_scp_reason"].dropna().unique())
    ) if "prio_scp_reason" in df else []
    sci_fail_reasons_all = sorted(
        set(df.loc[df.get("prio_sci_success", False) == False, "prio_sci_reason"].dropna().unique())
    ) if "prio_sci_reason" in df else []

    def plot_outcome_stack(prefix: str, positions: np.ndarray, hatch: str, reasons: list[str]):
        success_rates = []
        success_vals = []
        for n in n_values:
            sub = df[df["N"] == n]
            total = len(sub)
            success = sub[f"{prefix}_success"].sum() if total > 0 else 0
            success_rate = success / total if total > 0 else 0.0
            success_rates.append(success_rate)
            success_vals.append(success_rate)

        success_vals = np.array(success_vals, dtype=float)
        ax.bar(
            positions,
            success_vals,
            bar_w,
            color=failure_colors["success"],
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
        )

        bottom = success_vals.copy()
        for reason in reasons:
            vals = []
            for n in n_values:
                sub = df[df["N"] == n]
                total = len(sub)
                if total == 0:
                    vals.append(0.0)
                else:
                    count = ((sub[f"{prefix}_success"] == False) & (sub[f"{prefix}_reason"] == reason)).sum()
                    vals.append(count / total)
            vals = np.array(vals, dtype=float)
            if np.any(vals > 0):
                ax.bar(
                    positions,
                    vals,
                    bar_w,
                    bottom=bottom,
                    color=failure_colors.get(reason, "#cccccc"),
                    edgecolor="black",
                    linewidth=0.8,
                    hatch=hatch,
                )
                bottom += vals

        for xpos, rate in zip(positions, success_rates, strict=True):
            ax.text(xpos, 1.02, f"{rate:.0%}", ha="center", va="bottom", fontsize=8)

    plot_outcome_stack("prio_scp", x_arr - bar_w / 2, "", scp_fail_reasons_all)
    plot_outcome_stack("prio_sci", x_arr + bar_w / 2, "//", sci_fail_reasons_all)

    outcome_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=failure_colors["success"], edgecolor="black"),
    ]
    outcome_labels = ["success"]
    for reason in scp_fail_reasons_all + sci_fail_reasons_all:
        if reason not in outcome_labels:
            outcome_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=failure_colors.get(reason, "#cccccc"), edgecolor="black")
            )
            outcome_labels.append(reason)

    solver_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//"),
    ]

    ax.legend(
        solver_handles + outcome_handles,
        ["Prio SCP", "Prio SCI"] + outcome_labels,
        fontsize=7,
        loc="upper right",
        ncol=2,
    )
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks(x_arr)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_title("Outcome Breakdown")
    ax.set_xlabel("N")
    ax.set_ylabel("Fraction of runs")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 5: Cost ratio bars (Prio SCI / Prio SCP) ───────
    ax = axes[1, 1]
    median_ratio = []
    valid_ratio_labels = []
    valid_positions = []
    for i, n in enumerate(n_values):
        sub = df[df["N"] == n].dropna(subset=["prio_scp_cost", "prio_sci_cost"])
        ratio = (sub["prio_sci_cost"] / sub["prio_scp_cost"]).to_numpy()
        if len(ratio) > 0:
            median_ratio.append(float(np.median(ratio)))
            valid_ratio_labels.append(str(n))
            valid_positions.append(i)

    if median_ratio:
        x = np.array(valid_positions, dtype=float)
        ratio_vals = np.array(median_ratio, dtype=float)
        ratio_dev = (ratio_vals - 1.0) * 100.0

        ax.bar(x, ratio_dev, color=C_SCI, alpha=0.85, width=0.65)
        ax.axhline(0.0, color="black", lw=1.2, alpha=0.8)

        max_abs = float(np.max(np.abs(ratio_dev))) if len(ratio_dev) > 0 else 0.0
        pad = max(5.0, 0.1 * max_abs)
        limit_pad = max(8.0, 0.2 * max_abs)
        ax.set_ylim(-(max_abs + limit_pad), max_abs + limit_pad)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_ratio_labels)

        for xi, value, ratio in zip(x, ratio_dev, ratio_vals, strict=True):
            if value >= 0:
                y_text = value + pad * 0.2
                va = "bottom"
            else:
                y_text = value - pad * 0.2
                va = "top"
            ax.text(xi, y_text, f"{ratio:.2f}x", ha="center", va=va, fontsize=8)

    ax.set_title("Median Cost Ratio (Prio SCI / Prio SCP)")
    ax.set_xlabel("N")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(False, axis="y")

    # ── Panel 6: Minimum pairwise distance (zoomed to R) ─────
    R = 0.5  # from config
    det_tol = 1e-3  # from config

    ax = axes[1, 2]

    if "prio_scp_min_dist" in df and "prio_sci_min_dist" in df:
        scp_data, sci_data, d_pos, d_labels = [], [], [], []
        for i, n in enumerate(n_values):
            sub = df[df["N"] == n]
            s = sub["prio_scp_min_dist"].dropna().to_numpy()
            c = sub["prio_sci_min_dist"].dropna().to_numpy()
            if len(s) > 0 and len(c) > 0:
                scp_data.append(s)
                sci_data.append(c)
                d_pos.append(i + 1)
                d_labels.append(str(n))

        if scp_data:
            show_above = R + det_tol * 5
            for i_n, (pos, s_arr, c_arr) in enumerate(zip(d_pos, scp_data, sci_data)):
                s_show = s_arr[s_arr < show_above]
                c_show = c_arr[c_arr < show_above]
                jitter_s = np.random.default_rng(i_n).uniform(-0.2, -0.04, size=len(s_show))
                jitter_c = np.random.default_rng(i_n + 100).uniform(0.04, 0.2, size=len(c_show))
                ax.scatter(pos + jitter_s, s_show, s=18, color=C_SCP, alpha=0.8,
                           edgecolors="black", linewidths=0.3, zorder=3)
                ax.scatter(pos + jitter_c, c_show, s=18, color=C_SCI, alpha=0.8,
                           edgecolors="black", linewidths=0.3, zorder=3)

            ax.axhline(R, color="red", ls="-", lw=1.0, alpha=0.7, label="$R$")
            ax.axhline(R - det_tol, color="red", ls=":", lw=0.8, alpha=0.5, label="$R - \\epsilon$")
            ax.axhspan(R - det_tol, R, color="red", alpha=0.08)
            ax.set_ylim(R - det_tol * 4, R + det_tol * 6)
            ax.set_xticks(d_pos)
            ax.set_xticklabels(d_labels)
            ax.legend(
                [plt.Rectangle((0, 0), 1, 1, facecolor=C_SCP, edgecolor="black"),
                 plt.Rectangle((0, 0), 1, 1, facecolor=C_SCI, edgecolor="black"),
                 plt.Line2D([0], [0], color="red", ls="-"),
                 plt.Line2D([0], [0], color="red", ls=":")],
                ["Prio SCP", "Prio SCI", "$R$", "$R - \\epsilon$"],
                fontsize=6, loc="upper right",
            )
    ax.set_title("Min. Pairwise Distance (zoom)")
    ax.set_xlabel("N")
    ax.set_ylabel("$d_{\\min}$ [m]")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 7: Contacts (Prio SCI) ─────────────────────────
    # PrioritizedSCI does not report a per-pair contact maximum (each rank
    # is one robot vs many obstacles, not a symmetric pairing), so the
    # max-per-pair half-violin from sweep_n's contact panel has no
    # counterpart here. Panel intentionally left blank.
    ax = axes[2, 0]
    ax.set_title("Contact Counts (Prio SCI)")
    ax.set_xlabel("N")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 8: SCP iterations (Prio SCP, summed across ranks) ──
    ax = axes[2, 1]
    if "prio_scp_iterations_total" in df:
        iter_data, iter_pos, iter_labels = [], [], []
        for i, n in enumerate(n_values):
            vals = df.loc[df["N"] == n, "prio_scp_iterations_total"].dropna().to_numpy()
            if len(vals) > 0:
                iter_data.append(vals)
                iter_pos.append(i + 1)
                iter_labels.append(str(n))
        if iter_data:
            bp = ax.boxplot(
                iter_data, positions=iter_pos, widths=0.5,
                patch_artist=True, manage_ticks=False,
            )
            for p in bp["boxes"]:
                p.set_facecolor(C_SCP)
                p.set_alpha(0.7)
            ax.set_xticks(iter_pos)
            ax.set_xticklabels(iter_labels)
    ax.set_title("Total SCP Iterations (Prio SCP)")
    ax.set_xlabel("N")
    ax.set_ylabel("Iterations (sum over ranks)")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 9: Position RMSE + Max Deviation ───────────────
    ax = axes[2, 2]
    if "position_rmse" in df and "max_position_deviation" in df:
        rmse_data, maxd_data, d_pos, d_labels = [], [], [], []
        for i, n in enumerate(n_values):
            sub = df[df["N"] == n]
            r = sub["position_rmse"].dropna().to_numpy()
            m = sub["max_position_deviation"].dropna().to_numpy()
            if len(r) > 0:
                rmse_data.append(r)
                maxd_data.append(m)
                d_pos.append(i + 1)
                d_labels.append(str(n))
        if rmse_data:
            off = 0.18
            bp1 = ax.boxplot(
                rmse_data, positions=np.array(d_pos) - off, widths=0.32,
                patch_artist=True, manage_ticks=False,
            )
            bp2 = ax.boxplot(
                maxd_data, positions=np.array(d_pos) + off, widths=0.32,
                patch_artist=True, manage_ticks=False,
            )
            for p in bp1["boxes"]:
                p.set_facecolor("#59A14F")
                p.set_alpha(0.7)
            for p in bp2["boxes"]:
                p.set_facecolor("#E15759")
                p.set_alpha(0.7)
            ax.set_xticks(d_pos)
            ax.set_xticklabels(d_labels)
            ax.legend(
                [bp1["boxes"][0], bp2["boxes"][0]],
                ["Position RMSE", "Max deviation"],
                fontsize=7,
            )
    ax.set_title("Trajectory Similarity (Prio SCI vs Prio SCP)")
    ax.set_xlabel("N")
    ax.set_ylabel("Distance [m]")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 10: Homotopy class agreement ───────────────────
    ax = axes[3, 0]
    if "same_homotopy_class" in df:
        agree_rates = []
        bar_labels = []
        bar_pos = []
        for i, n in enumerate(n_values):
            sub = df[df["N"] == n]["same_homotopy_class"].dropna()
            if len(sub) > 0:
                agree_rates.append(sub.mean() * 100)
                bar_labels.append(str(n))
                bar_pos.append(i)

        if agree_rates:
            colors = ["#59A14F" if r > 50 else "#E15759" for r in agree_rates]
            ax.bar(bar_pos, agree_rates, color=colors, alpha=0.85, width=0.65)
            ax.axhline(100, color="#59A14F", ls="--", lw=0.8, alpha=0.4)
            ax.set_ylim(0, 110)
            ax.set_xticks(bar_pos)
            ax.set_xticklabels(bar_labels)
            for xp, rate in zip(bar_pos, agree_rates):
                ax.text(xp, rate + 2, f"{rate:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_title("Same Homotopy Class")
    ax.set_xlabel("N")
    ax.set_ylabel("Agreement [%]")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 11: Number of crossings ────────────────────────
    ax = axes[3, 1]
    if "n_crossings_prio_scp" in df and "n_crossings_prio_sci" in df:
        _paired_boxplot(ax, n_values, "n_crossings_prio_scp", "n_crossings_prio_sci", df)
    ax.set_title("Braid Crossings")
    ax.set_xlabel("N")
    ax.set_ylabel("Number of crossings")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 12: Homotopy class vs RMSE ─────────────────────
    ax = axes[3, 2]
    if "same_homotopy_class" in df and "position_rmse" in df:
        both_cols = df.dropna(subset=["same_homotopy_class", "position_rmse"])
        same = both_cols[both_cols["same_homotopy_class"] == True]["position_rmse"].to_numpy()
        diff = both_cols[both_cols["same_homotopy_class"] == False]["position_rmse"].to_numpy()

        box_data = []
        box_labels = []
        if len(same) > 0:
            box_data.append(same)
            box_labels.append(f"Same\n(n={len(same)})")
        if len(diff) > 0:
            box_data.append(diff)
            box_labels.append(f"Different\n(n={len(diff)})")
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
            colors_box = ["#59A14F", "#E15759"]
            for patch, c in zip(bp["boxes"], colors_box):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
    ax.set_title("RMSE by Homotopy Class")
    ax.set_ylabel("Position RMSE [m]")
    ax.set_yscale("log", nonpositive="clip")
    ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_output, dpi=PLOT_DPI)
    print(f"Dashboard plot saved to: {plot_output}")

    if show_plot:
        _maximize_figure_window()
        plt.show()
    else:
        plt.close(fig)


def run_single_experiment(config: Config, timeout: float = 300.0) -> dict:
    """Run PrioritizedSCP and PrioritizedSCI on a single configuration."""
    if config.solver.timeout > 0:
        timeout = config.solver.timeout

    np.random.seed(config.random_seed)
    initial, final = generate_random_positions(config)

    results = {
        "N": config.problem.n_robots,
        "seed": config.random_seed,
        "K": config.problem.n_timesteps,
    }

    K = config.problem.n_timesteps
    h = config.problem.timestep

    scp_traj_K = None
    sci_traj_K = None

    # ── PrioritizedSCP ───────────────────────────────────────
    try:
        scp_solver = PrioritizedSCP(config, verbose=False)
        scp_solver.set_initial_states(initial)
        scp_solver.set_final_states(final)
        scp_result, scp_timeout = run_solver_with_timeout(scp_solver, timeout)

        if scp_timeout:
            results.update({
                "prio_scp_success": False,
                "prio_scp_reason": "timeout",
                "prio_scp_time": timeout,
            })
        else:
            success = scp_result["metrics"]["converged"]
            scp_traj_K = scp_result["trajectories"] if success else None

            scp_cost_metrics = compute_metrics(scp_result["trajectories"], config)
            per_rank_t = scp_result["metrics"]["timing"].get("per_rank_time", [])

            results.update({
                "prio_scp_success": success,
                "prio_scp_reason": scp_result["metrics"]["convergence_reason"],
                "prio_scp_time": scp_result["metrics"]["timing"]["total_time"],
                "prio_scp_iterations_total": scp_result["metrics"]["scp_iterations_total"],
                "prio_scp_iterations_per_rank": json.dumps(
                    list(scp_result["metrics"].get("scp_iterations_per_rank", []))
                ),
                "prio_scp_time_per_rank": json.dumps([float(t) for t in per_rank_t]),
                "prio_scp_rank_reasons": json.dumps(
                    list(scp_result["metrics"].get("rank_reasons", []))
                ),
                "prio_scp_cost": scp_cost_metrics["acceleration_cost"],
                "prio_scp_min_dist": scp_cost_metrics["min_distance"],
            })
    except Exception as e:
        results.update({
            "prio_scp_success": False,
            "prio_scp_reason": "error",
            "prio_scp_time": np.nan,
            "prio_scp_error": str(e),
        })

    # ── PrioritizedSCI ───────────────────────────────────────
    try:
        sci_solver = PrioritizedSCI(config, verbose=False)
        sci_solver.set_initial_states(initial)
        sci_solver.set_final_states(final)
        sci_result, sci_timeout = run_solver_with_timeout(sci_solver, timeout)

        if sci_timeout:
            results.update({
                "prio_sci_success": False,
                "prio_sci_reason": "timeout",
                "prio_sci_time": timeout,
            })
        else:
            success = sci_result["metrics"]["converged"]
            traj_dense = sci_result["trajectories"]

            # Resample SCI's dense trajectory at SCP's K-point grid for fair
            # cost / similarity comparison (cf. sweep_n's contact sampling)
            n_dense = len(traj_dense["positions"][0])
            grid_idx = np.linspace(0, n_dense - 1, K).astype(int)
            sci_traj_K = {
                "positions": [p[grid_idx] for p in traj_dense["positions"]],
                "velocities": [v[grid_idx] for v in traj_dense["velocities"]],
                "accelerations": [a[grid_idx] for a in traj_dense["accelerations"]],
            }
            sci_cost_metrics = compute_metrics(sci_traj_K, config)

            # Continuous cost: trapezoidal integral over the dense grid
            T = config.problem.time_horizon
            dt_cont = T / (n_dense - 1)
            cost_continuous = sum(
                np.trapezoid(np.sum(a ** 2, axis=1), dx=dt_cont)
                for a in traj_dense["accelerations"]
            )

            per_rank_t = sci_result["metrics"]["timing"].get("per_rank_time", [])

            # NB: PrioritizedSCI has no max_contacts_single_pair metric --
            # per-pair counts are not meaningful in prioritized planning
            # (each rank is one robot vs many obstacles).
            results.update({
                "prio_sci_success": success,
                "prio_sci_reason": sci_result["metrics"]["convergence_reason"],
                "prio_sci_time": sci_result["metrics"]["timing"]["total_time"],
                "prio_sci_contacts": sci_result["metrics"]["num_contacts"],
                "prio_sci_contacts_per_rank": json.dumps(
                    list(sci_result["metrics"].get("num_contacts_per_rank", []))
                ),
                "prio_sci_time_per_rank": json.dumps([float(t) for t in per_rank_t]),
                "prio_sci_cost": sci_cost_metrics["acceleration_cost"],
                "prio_sci_cost_continuous": cost_continuous,
                "prio_sci_min_dist": sci_cost_metrics["min_distance"],
            })
    except Exception as e:
        results.update({
            "prio_sci_success": False,
            "prio_sci_reason": "error",
            "prio_sci_time": np.nan,
            "prio_sci_error": str(e),
        })

    # ── Cross-solver metrics (only when both succeed) ────────
    if scp_traj_K is not None and sci_traj_K is not None:
        scp_pos = scp_traj_K["positions"]
        sci_pos = sci_traj_K["positions"]
        N = len(scp_pos)

        sum_sq = 0.0
        max_dev = 0.0
        n_points = 0
        for i in range(N):
            diff = sci_pos[i] - scp_pos[i]
            norms = np.linalg.norm(diff, axis=1)
            sum_sq += np.sum(norms ** 2)
            max_dev = max(max_dev, np.max(norms))
            n_points += len(norms)

        results.update({
            "position_rmse": float(np.sqrt(sum_sq / n_points)),
            "max_position_deviation": float(max_dev),
        })

        # Braid word comparison (homotopy class)
        # NB: compare_braid_words returns the keys n_crossings_scp /
        # n_crossings_contact positionally — arg-a is the "scp side", arg-b
        # is the "contact side". We pass Prio SCP first, Prio SCI second.
        braid_scp = compute_braid_word(scp_pos)
        braid_sci = compute_braid_word(sci_pos)
        braid_cmp = compare_braid_words(braid_scp, braid_sci)
        results.update({
            "same_homotopy_class": braid_cmp["same_homotopy_class"],
            "n_crossings_prio_scp": braid_cmp["n_crossings_scp"],
            "n_crossings_prio_sci": braid_cmp["n_crossings_contact"],
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sweep comparison: Prio SCP vs Prio SCI across robot counts"
    )
    parser.add_argument("--n-min", type=int, default=2, help="Minimum robots")
    parser.add_argument("--n-max", type=int, default=20, help="Maximum robots")
    parser.add_argument("--n-step", type=int, default=2, help="Robot count step")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds per N")
    parser.add_argument("--timesteps", type=int, default=200, help="Number of timesteps (K)")
    parser.add_argument("--timeout", type=float, default=300, help="Solver timeout (seconds)")
    parser.add_argument(
        "--output",
        type=str,
        default="0_sweep_n_prioritized/sweep_n_prioritized_results.csv",
        help="Output CSV",
    )
    parser.add_argument("--plot-output", type=str, default=None,
                        help="Output dashboard image path")
    parser.add_argument("--no-plot", action="store_true", help="Disable dashboard plotting")
    parser.add_argument("--config", type=str, default="sweep_n_prioritized",
                        help="Base config name (loads sweep_n_prioritized.yaml by default)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load config — fail loudly if the named YAML can't be found ──
    try:
        base_config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"\nERROR: could not load config '{args.config}'.")
        print(f"  {e}")
        print(f"  Working directory: {Path.cwd()}")
        print(f"  Pass --config <name> to use a different YAML from configs/.")
        sys.exit(2)

    # Confirm at the top of the run what is actually loaded, so any mismatch
    # between expected and actual settings is immediately visible.
    prio_norm = getattr(base_config.solver, "prio_scp_normalize_obstacles", True)
    print(f"Loaded config: name='{base_config.name}', "
          f"osqp_eps_abs={base_config.solver.osqp_eps_abs}, "
          f"scp_detection_tol={base_config.solver.scp_detection_tol}, "
          f"prio_scp_normalize_obstacles={prio_norm}")

    N_values = list(range(args.n_min, args.n_max + 1, args.n_step))
    seeds = list(range(args.seeds))

    # Save config and sweep parameters alongside results
    config_output = output_path.with_name(f"{output_path.stem}_config.yaml")
    base_config.to_yaml(config_output)
    sweep_meta = {
        "n_min": args.n_min, "n_max": args.n_max, "n_step": args.n_step,
        "seeds": args.seeds, "timesteps": args.timesteps, "timeout": args.timeout,
        "base_config": args.config,
    }
    meta_output = output_path.with_name(f"{output_path.stem}_meta.json")
    meta_output.write_text(json.dumps(sweep_meta, indent=2))
    print(f"Config saved to: {config_output}")
    print(f"Sweep metadata saved to: {meta_output}")

    print("=" * 60)
    print("Sweep Comparison: Prio SCP vs Prio SCI")
    print("=" * 60)
    print(f"N values:    {N_values}")
    print(f"Seeds per N: {args.seeds}")
    print(f"Timesteps K: {args.timesteps}")
    print(f"Timeout:     {args.timeout}s")
    print(f"Output:      {args.output}")
    print()

    all_results = []
    total_experiments = len(N_values) * len(seeds)
    completed = 0
    start_time = time.time()

    for N in N_values:
        print(f"\nN = {N} robots:")
        for seed in seeds:
            config = build_scaled_config(base_config, N, seed, args.timesteps)
            result = run_single_experiment(config, args.timeout)
            all_results.append(result)
            completed += 1

            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total_experiments - completed) if completed > 0 else 0.0

            scp_ok = result.get("prio_scp_success", False)
            sci_ok = result.get("prio_sci_success", False)
            scp_time = result.get("prio_scp_time", np.nan)
            sci_time = result.get("prio_sci_time", np.nan)

            status = (
                f"  seed={seed}: "
                f"SCP={'✓' if scp_ok else '✗'} ({scp_time:.2f}s)"
            )
            if not scp_ok:
                status += f" [{result.get('prio_scp_reason', '?')}]"
            status += (
                f", SCI={'✓' if sci_ok else '✗'} ({sci_time:.2f}s)"
            )
            if not sci_ok:
                status += f" [{result.get('prio_sci_reason', '?')}]"
            status += f"  [ETA {eta / 60:.1f}min]"
            print(status)

        # Save intermediate results after each N block
        pd.DataFrame(all_results).to_csv(args.output, index=False)

    df = pd.DataFrame(all_results)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for N in N_values:
        sub = df[df["N"] == N]
        scp_times = sub.loc[sub["prio_scp_success"] == True, "prio_scp_time"]
        sci_times = sub.loc[sub["prio_sci_success"] == True, "prio_sci_time"]
        scp_success = sub["prio_scp_success"].sum()
        sci_success = sub["prio_sci_success"].sum()
        n_total = len(sub)

        print(f"\nN = {N}:")
        if len(scp_times) > 0:
            print(f"  Prio SCP: {scp_times.mean():.3f}s ± {scp_times.std():.3f}s, "
                  f"success: {scp_success}/{n_total}")
        else:
            print(f"  Prio SCP: no successes, {n_total} runs")

        scp_reasons = sub.loc[sub["prio_scp_success"] == False, "prio_scp_reason"]
        if len(scp_reasons) > 0:
            counts = Counter(scp_reasons)
            breakdown = ", ".join(f"{r}: {c}" for r, c in counts.items())
            print(f"    failures: {breakdown}")

        if len(sci_times) > 0:
            print(f"  Prio SCI: {sci_times.mean():.3f}s ± {sci_times.std():.3f}s, "
                  f"success: {sci_success}/{n_total}")
        else:
            print(f"  Prio SCI: no successes, {n_total} runs")

        sci_reasons = sub.loc[sub["prio_sci_success"] == False, "prio_sci_reason"]
        if len(sci_reasons) > 0:
            counts = Counter(sci_reasons)
            breakdown = ", ".join(f"{r}: {c}" for r, c in counts.items())
            print(f"    failures: {breakdown}")

        if len(scp_times) > 0 and len(sci_times) > 0:
            speedup = scp_times.mean() / sci_times.mean()
            print(f"  Speedup (SCP/SCI): {speedup:.2f}x")

        # Cost ratio (SCI / SCP) summary on paired successes
        paired = sub.dropna(subset=["prio_scp_cost", "prio_sci_cost"])
        if len(paired) > 0:
            ratio = (paired["prio_sci_cost"] / paired["prio_scp_cost"]).to_numpy()
            print(f"  Cost ratio SCI/SCP: median={float(np.median(ratio)):.4f}, "
                  f"mean={float(np.mean(ratio)):.4f}  (n={len(ratio)})")

        if "same_homotopy_class" in sub.columns:
            agree = sub["same_homotopy_class"].dropna()
            if len(agree) > 0:
                print(f"  Same homotopy class: "
                      f"{agree.mean() * 100:.1f}% ({int(agree.sum())}/{len(agree)})")

    print(f"\nResults saved to: {args.output}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} min")

    if not args.no_plot:
        plot_output = args.plot_output
        if plot_output is None:
            plot_output = str(output_path.with_name(f"{output_path.stem}_dashboard.png"))
        create_runtime_dashboard(df, plot_output=plot_output, show_plot=False)


if __name__ == "__main__":
    main()
