#!/usr/bin/env python3
"""Sweep comparison of ContactSolver vs LiftedSCP across varying robot counts."""

import argparse
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contact_solver import (
    ContactSolver,
    LiftedSCP,
    generate_random_positions,
    load_config,
)
from contact_solver.config import Config


APPLE_STUDIO_WIDTH_PX = 5120
APPLE_STUDIO_HEIGHT_PX = 2880
PLOT_DPI = 200


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


# ── Colors ───────────────────────────────────────────────────
C_SCP = "#4C78A8"
C_CONTACT = "#F58518"
C_SCP_LIGHT = "#4C78A8"
C_CONTACT_LIGHT = "#F58518"


def _paired_boxplot(ax, n_values, scp_col, contact_col, df, **kwargs):
    """Side-by-side boxplots for two solvers."""
    scp_data = [
        df.loc[(df["N"] == n) & (df["scp_success"] == True), scp_col].dropna().to_numpy()
        for n in n_values
    ]
    contact_data = [
        df.loc[(df["N"] == n) & (df["contact_success"] == True), contact_col].dropna().to_numpy()
        for n in n_values
    ]

    valid, labels = [], []
    for i, n in enumerate(n_values):
        if len(scp_data[i]) > 0 and len(contact_data[i]) > 0:
            valid.append((scp_data[i], contact_data[i]))
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
    bp_con = ax.boxplot(
        [p[1] for p in valid], positions=pos + off, widths=0.32,
        patch_artist=True, manage_ticks=False,
    )
    for p in bp_scp["boxes"]:
        p.set_facecolor(C_SCP)
        p.set_alpha(0.7)
    for p in bp_con["boxes"]:
        p.set_facecolor(C_CONTACT)
        p.set_alpha(0.7)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.legend(
        [bp_scp["boxes"][0], bp_con["boxes"][0]],
        ["LiftedSCP", "ContactSolver"],
        loc=kwargs.get("legend_loc", "upper left"),
        fontsize=8,
    )


def create_runtime_dashboard(df: pd.DataFrame, plot_output: str, show_plot: bool) -> None:
    """Create and save a dashboard."""
    width_in = APPLE_STUDIO_WIDTH_PX / PLOT_DPI
    height_in = APPLE_STUDIO_HEIGHT_PX / PLOT_DPI * 4 / 3
    fig, axes = plt.subplots(4, 3, figsize=(width_in, height_in), dpi=PLOT_DPI)
    fig.suptitle("Sweep Comparison Dashboard", fontsize=22)

    n_values = sorted(df["N"].dropna().unique().astype(int).tolist())

    # ── Panel 1: Runtime boxplot ─────────────────────────────
    ax = axes[0, 0]
    _paired_boxplot(ax, n_values, "scp_time", "contact_time", df)
    ax.set_yscale("log")
    ax.set_title("Runtime")
    ax.set_xlabel("N")
    ax.set_ylabel("Time [s]")
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Runtime median + IQR ────────────────────────
    ax = axes[0, 1]
    grouped_iqr = (
        df.loc[(df["scp_success"] == True) | (df["contact_success"] == True)]
        .groupby("N", as_index=False)
        .agg(
            scp_median=("scp_time", lambda x: x[df.loc[x.index, "scp_success"] == True].median()),
            scp_q25=("scp_time", lambda x: x[df.loc[x.index, "scp_success"] == True].quantile(0.25)),
            scp_q75=("scp_time", lambda x: x[df.loc[x.index, "scp_success"] == True].quantile(0.75)),
            scp_q025=("scp_time", lambda x: x[df.loc[x.index, "scp_success"] == True].quantile(0.025)),
            scp_q975=("scp_time", lambda x: x[df.loc[x.index, "scp_success"] == True].quantile(0.975)),
            contact_median=("contact_time", lambda x: x[df.loc[x.index, "contact_success"] == True].median()),
            contact_q25=("contact_time", lambda x: x[df.loc[x.index, "contact_success"] == True].quantile(0.25)),
            contact_q75=("contact_time", lambda x: x[df.loc[x.index, "contact_success"] == True].quantile(0.75)),
            contact_q025=("contact_time", lambda x: x[df.loc[x.index, "contact_success"] == True].quantile(0.025)),
            contact_q975=("contact_time", lambda x: x[df.loc[x.index, "contact_success"] == True].quantile(0.975)),
        )
        .sort_values("N")
    )

    if len(grouped_iqr) > 0:
        x = grouped_iqr["N"].to_numpy(dtype=float)
        for col, color, marker, label in [
            ("scp", C_SCP, "o", "LiftedSCP"),
            ("contact", C_CONTACT, "s", "ContactSolver"),
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

    # ── Panel 3: Speedup diverging bars ──────────────────────
    ax = axes[0, 2]
    grouped_speedup = (
        df.loc[(df["scp_success"] == True) & (df["contact_success"] == True)]
        .groupby("N", as_index=False)
        .agg(
            scp_median=("scp_time", "median"),
            contact_median=("contact_time", "median"),
        )
        .sort_values("N")
    )

    if len(grouped_speedup) > 0:
        x = np.arange(len(grouped_speedup))
        n_labels = grouped_speedup["N"].astype(int).astype(str).tolist()
        speedup_pct = (
            grouped_speedup["scp_median"].to_numpy(dtype=float)
            / grouped_speedup["contact_median"].to_numpy(dtype=float)
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
            ratio = grouped_speedup["scp_median"].to_numpy(dtype=float)[xi] / grouped_speedup["contact_median"].to_numpy(dtype=float)[xi]
            factor = ratio if ratio >= 1.0 else 1.0 / ratio
            label = f"{factor:.2f}x"
            if value >= 0:
                y_text = value + pad * 0.2
                va = "bottom"
            else:
                y_text = value - pad * 0.2
                va = "top"
            ax.text(xi, y_text, label, ha="center", va=va, fontsize=8)

    ax.set_title("Median Speedup vs ContactSolver")
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
        "infeasible": "#B4C7E7",
        "timeout": "#D6E4F0",
        "max_contacts": "#F4B183",
        "max_contacts_per_pair": "#F8CBAD",
        "error": "#BAB0AC",
    }

    scp_fail_reasons_all = sorted(
        set(df.loc[df.get("scp_success", False) == False, "scp_reason"].dropna().unique())
    ) if "scp_reason" in df else []
    contact_fail_reasons_all = sorted(
        set(df.loc[df.get("contact_success", False) == False, "contact_reason"].dropna().unique())
    ) if "contact_reason" in df else []

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

    plot_outcome_stack("scp", x_arr - bar_w / 2, "", scp_fail_reasons_all)
    plot_outcome_stack("contact", x_arr + bar_w / 2, "//", contact_fail_reasons_all)

    outcome_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=failure_colors["success"], edgecolor="black"),
    ]
    outcome_labels = ["success"]
    for reason in scp_fail_reasons_all + contact_fail_reasons_all:
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
        ["LiftedSCP", "ContactSolver"] + outcome_labels,
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

    # ── Panel 5: Cost ratio bars ─────────────────────────────
    ax = axes[1, 1]
    median_ratio = []
    valid_ratio_labels = []
    valid_positions = []
    for i, n in enumerate(n_values):
        sub = df[df["N"] == n].dropna(subset=["scp_cost", "contact_cost"])
        ratio = (sub["contact_cost"] / sub["scp_cost"]).to_numpy()
        if len(ratio) > 0:
            median_ratio.append(float(np.median(ratio)))
            valid_ratio_labels.append(str(n))
            valid_positions.append(i)

    if median_ratio:
        x = np.array(valid_positions, dtype=float)
        ratio_vals = np.array(median_ratio, dtype=float)
        ratio_dev = (ratio_vals - 1.0) * 100.0

        ax.bar(x, ratio_dev, color=C_CONTACT, alpha=0.85, width=0.65)
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

    ax.set_title("Median Cost Ratio (Contact / SCP)")
    ax.set_xlabel("N")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(False, axis="y")

    # ── Panel 6: Minimum pairwise distance (zoomed to R) ─────
    R = 0.5  # from config
    det_tol = 1e-3  # from config

    ax = axes[1, 2]

    if "scp_min_dist" in df and "contact_min_dist" in df:
        scp_data, contact_data, d_pos, d_labels = [], [], [], []
        for i, n in enumerate(n_values):
            sub = df[df["N"] == n]
            s = sub["scp_min_dist"].dropna().to_numpy()
            c = sub["contact_min_dist"].dropna().to_numpy()
            if len(s) > 0 and len(c) > 0:
                scp_data.append(s)
                contact_data.append(c)
                d_pos.append(i + 1)
                d_labels.append(str(n))

        if scp_data:
            # Only show points below R + small margin
            show_above = R + det_tol * 5
            for i_n, (pos, s_arr, c_arr) in enumerate(zip(d_pos, scp_data, contact_data)):
                s_show = s_arr[s_arr < show_above]
                c_show = c_arr[c_arr < show_above]
                jitter_s = np.random.default_rng(i_n).uniform(-0.2, -0.04, size=len(s_show))
                jitter_c = np.random.default_rng(i_n + 100).uniform(0.04, 0.2, size=len(c_show))
                ax.scatter(pos + jitter_s, s_show, s=18, color=C_SCP, alpha=0.8, edgecolors="black", linewidths=0.3, zorder=3)
                ax.scatter(pos + jitter_c, c_show, s=18, color=C_CONTACT, alpha=0.8, edgecolors="black", linewidths=0.3, zorder=3)

            ax.axhline(R, color="red", ls="-", lw=1.0, alpha=0.7, label="$R$")
            ax.axhline(R - det_tol, color="red", ls=":", lw=0.8, alpha=0.5, label=f"$R - \\epsilon$")
            ax.axhspan(R - det_tol, R, color="red", alpha=0.08)
            ax.set_ylim(R - det_tol * 4, R + det_tol * 6)
            ax.set_xticks(d_pos)
            ax.set_xticklabels(d_labels)
            ax.legend(
                [plt.Rectangle((0, 0), 1, 1, facecolor=C_SCP, edgecolor="black"),
                 plt.Rectangle((0, 0), 1, 1, facecolor=C_CONTACT, edgecolor="black"),
                 plt.Line2D([0], [0], color="red", ls="-"),
                 plt.Line2D([0], [0], color="red", ls=":")],
                ["LiftedSCP", "ContactSolver", "$R$", "$R - \\epsilon$"],
                fontsize=6, loc="upper right",
            )
    ax.set_title("Min. Pairwise Distance (zoom)")
    ax.set_xlabel("N")
    ax.set_ylabel("$d_{\\min}$ [m]")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 7: Contacts (total + max per pair) ─────────────
    ax = axes[2, 0]
    if "contact_contacts" in df and "contact_max_contacts_single_pair" in df:
        total_data, max_pair_data, c_labels, c_pos = [], [], [], []
        for i, n in enumerate(n_values):
            sub = df.loc[df["N"] == n]
            t = sub["contact_contacts"].dropna().to_numpy()
            m = sub["contact_max_contacts_single_pair"].dropna().to_numpy()
            if len(t) > 0:
                total_data.append(t)
                max_pair_data.append(m)
                c_pos.append(i + 1)
                c_labels.append(str(n))

        if total_data:
            vp_total = ax.violinplot(
                total_data, positions=np.array(c_pos), widths=0.7,
                showmeans=False, showmedians=True, showextrema=False,
            )
            vp_pair = ax.violinplot(
                max_pair_data, positions=np.array(c_pos), widths=0.7,
                showmeans=False, showmedians=True, showextrema=False,
            )

            for body, pos in zip(vp_total["bodies"], c_pos, strict=True):
                verts = body.get_paths()[0].vertices
                verts[:, 0] = np.maximum(verts[:, 0], pos)
                body.set_facecolor(C_CONTACT)
                body.set_edgecolor("black")
                body.set_alpha(0.75)

            for body, pos in zip(vp_pair["bodies"], c_pos, strict=True):
                verts = body.get_paths()[0].vertices
                verts[:, 0] = np.minimum(verts[:, 0], pos)
                body.set_facecolor("#E45756")
                body.set_edgecolor("black")
                body.set_alpha(0.75)

            vp_total["cmedians"].set_color("black")
            vp_total["cmedians"].set_linewidth(1.2)
            vp_pair["cmedians"].set_color("black")
            vp_pair["cmedians"].set_linewidth(1.2)

            ax.set_xticks(c_pos)
            ax.set_xticklabels(c_labels)

            # Limit lines from config
            ax.axhline(30, color=C_CONTACT, ls="--", lw=1, alpha=0.5, label="max_contacts (30)")
            ax.axhline(10, color="#E45756", ls="--", lw=1, alpha=0.5, label="max_per_pair (10)")
            ax.legend(
                [
                 plt.Rectangle((0, 0), 1, 1, facecolor=C_CONTACT, edgecolor="black"),
                 plt.Rectangle((0, 0), 1, 1, facecolor="#E45756", edgecolor="black"),
                 plt.Line2D([0], [0], color=C_CONTACT, ls="--"),
                 plt.Line2D([0], [0], color="#E45756", ls="--")],
                ["Total contacts", "Max per pair", "sci_max_contacts", "sci_max_contacts_per_pair"],
                fontsize=7,
            )
    ax.set_title("Contact Counts")
    ax.set_xlabel("N")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 8: SCP iterations ──────────────────────────────
    ax = axes[2, 1]
    if "scp_iterations" in df:
        iter_data, iter_pos, iter_labels = [], [], []
        for i, n in enumerate(n_values):
            vals = df.loc[df["N"] == n, "scp_iterations"].dropna().to_numpy()
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
            ax.axhline(20, color=C_SCP, ls="--", lw=1, alpha=0.5, label="scp_max_iterations (20)")
            ax.legend(fontsize=7)
    ax.set_title("SCP Iterations")
    ax.set_xlabel("N")
    ax.set_ylabel("Iterations")
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
    ax.set_title("Trajectory Similarity")
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
    if "n_crossings_scp" in df and "n_crossings_contact" in df:
        _paired_boxplot(ax, n_values, "n_crossings_scp", "n_crossings_contact", df)
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


def compute_scale_factor(N: int, base_N: int = 2) -> float:
    """Compute scaling factor for N robots relative to base_N."""
    return math.sqrt(N / base_N)


def build_scaled_config(base_config: Config, N: int, seed: int, K: int = 100) -> Config:
    """Build a config with parameters scaled for N robots."""
    scale = compute_scale_factor(N, base_N=2)

    # Scale workspace and time with sqrt(N) to maintain congestion
    base_time = 10.0
    base_cluster_radius = 5.0

    time_horizon = base_time * scale
    cluster_radius = base_cluster_radius * scale
    timestep = time_horizon / K

    return base_config.override(**{
        "random_seed": seed,
        "problem.n_robots": N,
        "problem.time_horizon": time_horizon,
        "problem.timestep": timestep,
        "problem.scenario.cluster_radius": cluster_radius,
        "visualization.show_plots": False,
    })


# ── Braid word extraction and comparison ──────────────────────

def compute_braid_word(positions: list[np.ndarray], axis: int = 0, min_sep: float = 0.0) -> list[tuple[int, int, int]]:
    """Extract the braid word from a multi-robot trajectory.

    Projects robot positions onto a reference axis and records crossing events
    (when two robots swap their order along that axis).

    Args:
        positions: List of N arrays of shape (K, 2), one per robot.
        axis: Projection axis (0=x, 1=y).
        min_sep: Minimum perpendicular separation to register a crossing.
            Crossings where robots are closer than this in the perpendicular
            direction are ignored (likely numerical noise).

    Returns:
        List of (i, j, sign) tuples, where i < j are robot indices and
        sign is +1 or -1 indicating the crossing direction.
        +1: robot i overtakes robot j (moves ahead in projection)
        -1: robot j overtakes robot i
    """
    N = len(positions)
    K = len(positions[0])

    # Project onto axis
    proj = np.array([positions[i][:, axis] for i in range(N)])  # (N, K)

    crossings = []
    for k in range(1, K):
        for i in range(N):
            for j in range(i + 1, N):
                prev_diff = proj[i, k - 1] - proj[j, k - 1]
                curr_diff = proj[i, k] - proj[j, k]

                # Crossing: sign change in the difference
                if prev_diff * curr_diff < 0:
                    # Determine crossing sign from the perpendicular axis
                    perp = 1 - axis
                    # Midpoint perpendicular positions at crossing time
                    mid_perp_i = 0.5 * (positions[i][k - 1, perp] + positions[i][k, perp])
                    mid_perp_j = 0.5 * (positions[j][k - 1, perp] + positions[j][k, perp])

                    # Skip crossings with insufficient perpendicular separation
                    perp_sep = abs(mid_perp_i - mid_perp_j)
                    if perp_sep < min_sep:
                        continue

                    # sign: which side does i pass j?
                    sign = 1 if mid_perp_i > mid_perp_j else -1
                    crossings.append((i, j, sign))

    return crossings


def normalize_braid_word(
    crossings: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Normalize a braid word by applying far-commutativity.

    Sorts commuting generators (those involving disjoint robot pairs)
    into a canonical order so that independent crossings in different
    temporal orders produce the same normalized word.
    """
    word = list(crossings)
    changed = True
    while changed:
        changed = False
        for idx in range(len(word) - 1):
            (i1, j1, s1) = word[idx]
            (i2, j2, s2) = word[idx + 1]
            # Two crossings commute if they involve disjoint pairs
            pair1 = {i1, j1}
            pair2 = {i2, j2}
            if pair1.isdisjoint(pair2):
                # Canonical order: sort by (min robot index, max robot index)
                if (i1, j1) > (i2, j2):
                    word[idx], word[idx + 1] = word[idx + 1], word[idx]
                    changed = True
    return word


def compare_braid_words(
    crossings_a: list[tuple[int, int, int]],
    crossings_b: list[tuple[int, int, int]],
) -> dict:
    """Compare two braid words and return comparison metrics.

    Uses multiset comparison of (pair, sign) tuples rather than ordered
    braid word comparison. This is robust to temporal reordering of
    nearly-simultaneous crossings (a discretization artifact), while
    correctly detecting genuine homotopy class differences (sign flips).
    """
    # Multiset comparison: same crossings with same signs, ignoring order
    from collections import Counter
    multiset_a = Counter(crossings_a)
    multiset_b = Counter(crossings_b)
    same_homotopy = multiset_a == multiset_b

    return {
        "same_homotopy_class": same_homotopy,
        "n_crossings_scp": len(crossings_a),
        "n_crossings_contact": len(crossings_b),
    }


def create_homotopy_disagreement_plot(
    df: pd.DataFrame, base_config: Config, plot_output: str, show_plot: bool = False
) -> None:
    """Create overlay plots for cases where solvers found different homotopy classes."""
    if "same_homotopy_class" not in df:
        return

    disagree = df[df["same_homotopy_class"] == False][["N", "seed", "scp_cost", "contact_cost"]]
    if len(disagree) == 0:
        print("No homotopy disagreements to plot.")
        return

    cases = disagree.values.tolist()
    print(f"Plotting {len(cases)} homotopy disagreement cases...")

    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 1, figsize=(6, 5 * n_cases), squeeze=False)

    cmap = plt.cm.tab10

    for row, (N, seed, scp_cost, contact_cost) in enumerate(cases):
        N, seed = int(N), int(seed)
        config = build_scaled_config(base_config, N, seed, K=100)

        np.random.seed(config.random_seed)
        initial, final = generate_random_positions(config)

        # Run SCP
        scp = LiftedSCP(config, verbose=False)
        scp.set_initial_states(initial)
        scp.set_final_states(final)
        scp_result = scp.generate_trajectories()
        scp_pos = scp_result["trajectories"]["positions"]

        # Run ContactSolver
        contact = ContactSolver(config, verbose=False)
        contact.set_initial_states(initial)
        contact.set_final_states(final)
        contact_result = contact.generate_trajectories()
        K = config.problem.n_timesteps
        h = config.problem.timestep
        scp_times = np.arange(1, K + 1) * h
        _, contact_pos, _, _ = contact.sample(times=scp_times)

        colors = [cmap(i % 10) for i in range(N)]
        cost_ratio = contact_cost / scp_cost if scp_cost > 0 else float("nan")

        ax = axes[row, 0]
        for i in range(N):
            ax.plot(scp_pos[i][:, 0], scp_pos[i][:, 1],
                    color=colors[i], lw=1.0, alpha=0.6, ls="-")
            ax.plot(contact_pos[i][:, 0], contact_pos[i][:, 1],
                    color=colors[i], lw=1.0, alpha=0.6, ls="--")
        ax.scatter(initial[:, 0], initial[:, 1], c="black", s=20, zorder=5, marker="o")
        ax.scatter(final[:, 0], final[:, 1], c="black", s=20, zorder=5, marker="s")
        ax.set_aspect("equal")
        ax.set_title(
            f"N={N}, seed={seed}    "
            f"$J_{{\\mathrm{{SCP}}}}$={scp_cost:.2f}  "
            f"$J_{{\\mathrm{{SCI}}}}$={contact_cost:.2f}  "
            f"(ratio={cost_ratio:.3f})",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)

        # Legend on first panel only
        if row == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color="gray", lw=1, ls="-", label="LiftedSCP"),
                Line2D([0], [0], color="gray", lw=1, ls="--", label="ContactSolver"),
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

        print(f"  N={N}, seed={seed}  cost ratio={cost_ratio:.3f}")

    fig.suptitle("Homotopy Class Disagreements", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_output, dpi=200, bbox_inches="tight")
    print(f"Homotopy disagreement plot saved to: {plot_output}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def compute_metrics(trajectories: dict, config: Config) -> dict:
    """Compute trajectory metrics."""
    positions = trajectories["positions"]
    accelerations = trajectories["accelerations"]

    N = len(positions)
    K = len(positions[0])
    h = config.problem.timestep

    # Min pairwise distance
    min_dist = np.inf
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i][k] - positions[j][k])
                min_dist = min(min_dist, dist)

    # Acceleration cost: h * sum ||a[k]||^2  (left Riemann sum,
    # consistent with the piecewise-constant acceleration model in the QP)
    cost = 0.0
    for a in accelerations:
        cost += h * np.sum(a**2)

    return {
        "min_distance": min_dist,
        "acceleration_cost": cost,
    }


def compute_cross_solver_metrics(
    scp_traj: dict, contact_traj: dict, config: Config,
) -> dict:
    """Compute metrics comparing both solvers' trajectories."""
    scp_pos = scp_traj["positions"]
    contact_pos = contact_traj["positions"]
    N = len(scp_pos)

    # Resample contact solver to SCP timesteps (both have K points)
    K_scp = len(scp_pos[0])
    K_contact = len(contact_pos[0])

    # Interpolate contact solver positions to SCP timestep grid
    if K_scp != K_contact:
        t_scp = np.linspace(0, 1, K_scp)
        t_contact = np.linspace(0, 1, K_contact)
        contact_pos_resampled = []
        for i in range(N):
            resampled = np.zeros((K_scp, 2))
            for d in range(2):
                resampled[:, d] = np.interp(t_scp, t_contact, contact_pos[i][:, d])
            contact_pos_resampled.append(resampled)
    else:
        contact_pos_resampled = contact_pos

    # Position RMSE and max deviation
    sum_sq = 0.0
    max_dev = 0.0
    n_points = 0
    for i in range(N):
        diff = contact_pos_resampled[i] - scp_pos[i]
        norms = np.linalg.norm(diff, axis=1)
        sum_sq += np.sum(norms**2)
        max_dev = max(max_dev, np.max(norms))
        n_points += len(norms)

    position_rmse = np.sqrt(sum_sq / n_points)

    return {
        "position_rmse": position_rmse,
        "max_position_deviation": max_dev,
    }


def run_solver_with_timeout(solver, timeout: float):
    """Run solver with timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(solver.generate_trajectories)
        try:
            result = future.result(timeout=timeout)
            return result, False
        except FuturesTimeoutError:
            return None, True


def run_single_experiment(config: Config, timeout: float = 300.0) -> dict:
    """Run both solvers on a single configuration."""
    # Use config timeout if set, otherwise use the passed value
    if config.solver.timeout > 0:
        timeout = config.solver.timeout

    np.random.seed(config.random_seed)
    initial, final = generate_random_positions(config)

    results = {
        "N": config.problem.n_robots,
        "seed": config.random_seed,
        "K": config.problem.n_timesteps,
    }

    scp_traj = None
    contact_traj = None
    contact_traj_K = None

    # ── LiftedSCP ────────────────────────────────────────────
    try:
        scp_solver = LiftedSCP(config, verbose=False)
        scp_solver.set_initial_states(initial)
        scp_solver.set_final_states(final)

        scp_result, scp_timeout = run_solver_with_timeout(scp_solver, timeout)

        if scp_timeout:
            results.update({
                "scp_success": False,
                "scp_reason": "timeout",
                "scp_time": timeout,
            })
        else:
            scp_metrics = compute_metrics(scp_result["trajectories"], config)
            reason = scp_result["metrics"]["convergence_reason"]
            collision_free = scp_result["metrics"]["collision_check"]["all_satisfied"]

            # Remap: collision_free_initial → converged
            if reason == "collision_free_initial":
                reason = "converged"
            # If iterate converged but collisions remain → infeasible
            if reason == "converged" and not collision_free:
                reason = "infeasible"

            success = reason == "converged"
            scp_traj = scp_result["trajectories"] if success else None

            results.update({
                "scp_success": success,
                "scp_reason": reason,
                "scp_time": scp_result["metrics"]["timing"]["total_time"],
                "scp_iterations": scp_result["metrics"]["scp_iterations"],
                "scp_cost": scp_metrics["acceleration_cost"],
                "scp_min_dist": scp_metrics["min_distance"],
                "scp_worst_violation": scp_result["metrics"]["collision_check"]["worst_violation"],
            })

            # Save debug log for failed runs
            if not success and "_debug_log" in scp_result["metrics"]:
                import json
                debug_dir = Path("0_sweep_n/debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                N = config.problem.n_robots
                seed = config.random_seed
                debug_file = debug_dir / f"scp_fail_N{N}_seed{seed}.json"
                debug_data = {
                    "N": N, "seed": seed, "K": config.problem.n_timesteps,
                    "R": config.problem.min_distance,
                    "detection_tol": config.solver.scp_detection_tol,
                    "osqp_eps_abs": config.solver.osqp_eps_abs,
                    "osqp_eps_rel": config.solver.osqp_eps_rel,
                    "scp_convergence_rel": config.solver.scp_convergence_rel,
                    "scp_convergence_abs": config.solver.scp_convergence_abs,
                    "reason": reason,
                    "iterations": scp_result["metrics"]["_debug_log"],
                }
                debug_file.write_text(json.dumps(debug_data, indent=2))
                print(f"    Debug log saved: {debug_file}")
    except Exception as e:
        results.update({
            "scp_success": False,
            "scp_reason": "infeasible",
            "scp_time": np.nan,
            "scp_error": str(e),
        })

    # ── ContactSolver ────────────────────────────────────────
    try:
        contact_solver = ContactSolver(config, verbose=False)
        contact_solver.set_initial_states(initial)
        contact_solver.set_final_states(final)

        contact_result, contact_timeout = run_solver_with_timeout(contact_solver, timeout)

        if contact_timeout:
            results.update({
                "contact_success": False,
                "contact_reason": "timeout",
                "contact_time": timeout,
            })
        else:
            reason = contact_result["metrics"]["convergence_reason"]
            success = contact_result["metrics"]["converged"]

            contact_traj = contact_result["trajectories"] if success else None

            # Discrete cost: evaluate continuous trajectories at SCP's K timesteps
            K = config.problem.n_timesteps
            h = config.problem.timestep
            scp_times = np.arange(1, K + 1) * h  # SCP grid: h, 2h, ..., Kh=T
            _, pos_K, _, acc_K = contact_solver.sample(times=scp_times)
            contact_traj_K = {"positions": pos_K, "accelerations": acc_K}
            contact_metrics_discrete = compute_metrics(contact_traj_K, config)

            # Continuous cost: trapezoidal integral over dense 1000-point grid
            contact_acc = contact_result["trajectories"]["accelerations"]
            T = config.problem.time_horizon
            dt_cont = T / (len(contact_acc[0]) - 1)
            cost_continuous = sum(np.trapezoid(np.sum(a**2, axis=1), dx=dt_cont) for a in contact_acc)

            results.update({
                "contact_success": success,
                "contact_reason": reason,
                "contact_time": contact_result["metrics"]["timing"]["total_time"],
                "contact_contacts": contact_result["metrics"]["num_contacts"],
                "contact_max_contacts_single_pair": contact_result["metrics"]["max_contacts_single_pair"],
                "contact_cost": contact_metrics_discrete["acceleration_cost"],
                "contact_cost_continuous": cost_continuous,
                "contact_min_dist": contact_metrics_discrete["min_distance"],
            })
    except Exception as e:
        results.update({
            "contact_success": False,
            "contact_reason": "error",
            "contact_time": np.nan,
            "contact_error": str(e),
        })

    # ── Cross-solver metrics (only when both succeed) ────────
    if scp_traj is not None and contact_traj_K is not None:
        # Use K-point resampled trajectories (same grid as SCP, no interpolation)
        scp_pos = scp_traj["positions"]
        contact_pos_K = contact_traj_K["positions"]
        N = len(scp_pos)

        sum_sq = 0.0
        max_dev = 0.0
        n_points = 0
        for i in range(N):
            diff = contact_pos_K[i] - scp_pos[i]
            norms = np.linalg.norm(diff, axis=1)
            sum_sq += np.sum(norms**2)
            max_dev = max(max_dev, np.max(norms))
            n_points += len(norms)

        results.update({
            "position_rmse": np.sqrt(sum_sq / n_points),
            "max_position_deviation": max_dev,
        })

        # Braid word comparison (homotopy class)
        braid_scp = compute_braid_word(scp_pos)
        braid_contact = compute_braid_word(contact_pos_K)
        braid_cmp = compare_braid_words(braid_scp, braid_contact)
        results.update({
            "same_homotopy_class": braid_cmp["same_homotopy_class"],
            "n_crossings_scp": braid_cmp["n_crossings_scp"],
            "n_crossings_contact": braid_cmp["n_crossings_contact"],
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep comparison across robot counts")
    parser.add_argument("--n-min", type=int, default=2, help="Minimum robots")
    parser.add_argument("--n-max", type=int, default=20, help="Maximum robots")
    parser.add_argument("--n-step", type=int, default=2, help="Robot count step")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds per N")
    parser.add_argument("--timesteps", type=int, default=100, help="Number of timesteps")
    parser.add_argument("--timeout", type=float, default=300, help="Solver timeout (seconds)")
    parser.add_argument("--output", type=str, default="0_sweep_n/sweep_n_results.csv", help="Output CSV")
    parser.add_argument("--plot-output", type=str, default=None, help="Output dashboard image path")
    parser.add_argument("--no-plot", action="store_true", help="Disable dashboard plotting")
    parser.add_argument("--config", type=str, default="sweep_n", help="Base config name")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load base config
    try:
        base_config = load_config(args.config)
    except FileNotFoundError:
        from contact_solver import get_small_config
        base_config = get_small_config()

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
    import json
    meta_output = output_path.with_name(f"{output_path.stem}_meta.json")
    meta_output.write_text(json.dumps(sweep_meta, indent=2))
    print(f"Config saved to: {config_output}")
    print(f"Sweep metadata saved to: {meta_output}")

    print("=" * 60)
    print("Sweep Comparison: ContactSolver vs LiftedSCP")
    print("=" * 60)
    print(f"N values: {N_values}")
    print(f"Seeds per N: {args.seeds}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Timeout: {args.timeout}s")
    print(f"Output: {args.output}")
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
            eta = (elapsed / completed) * (total_experiments - completed)

            # Status indicator
            scp_ok = result.get("scp_success", False)
            contact_ok = result.get("contact_success", False)
            scp_time = result.get("scp_time", np.nan)
            contact_time = result.get("contact_time", np.nan)

            status = f"  seed={seed}: SCP={'✓' if scp_ok else '✗'} ({scp_time:.2f}s)"
            if not scp_ok:
                status += f" [{result.get('scp_reason', '?')}]"
            status += f", Contact={'✓' if contact_ok else '✗'} ({contact_time:.2f}s)"
            if not contact_ok:
                status += f" [{result.get('contact_reason', '?')}]"
            print(status)

        # Save intermediate results
        df = pd.DataFrame(all_results)
        df.to_csv(args.output, index=False)

    # Final summary
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for N in N_values:
        subset = df[df["N"] == N]

        scp_times = subset.loc[subset["scp_success"] == True, "scp_time"]
        contact_times = subset.loc[subset["contact_success"] == True, "contact_time"]
        scp_success = subset["scp_success"].sum()
        contact_success = subset["contact_success"].sum()
        n_total = len(subset)

        print(f"\nN = {N}:")
        if len(scp_times) > 0:
            print(f"  LiftedSCP:     {scp_times.mean():.3f}s ± {scp_times.std():.3f}s, success: {scp_success}/{n_total}")
        else:
            print(f"  LiftedSCP:     no successes, {n_total} runs")

        # SCP failure breakdown
        scp_reasons = subset.loc[subset["scp_success"] == False, "scp_reason"]
        if len(scp_reasons) > 0:
            counts = Counter(scp_reasons)
            breakdown = ", ".join(f"{r}: {c}" for r, c in counts.items())
            print(f"    failures: {breakdown}")

        if len(contact_times) > 0:
            print(f"  ContactSolver: {contact_times.mean():.3f}s ± {contact_times.std():.3f}s, success: {contact_success}/{n_total}")
        else:
            print(f"  ContactSolver: no successes, {n_total} runs")

        # Contact failure breakdown
        contact_reasons = subset.loc[subset["contact_success"] == False, "contact_reason"]
        if len(contact_reasons) > 0:
            counts = Counter(contact_reasons)
            breakdown = ", ".join(f"{r}: {c}" for r, c in counts.items())
            print(f"    failures: {breakdown}")

        if len(scp_times) > 0 and len(contact_times) > 0:
            speedup = scp_times.mean() / contact_times.mean()
            print(f"  Speedup: {speedup:.2f}x")

    print(f"\nResults saved to: {args.output}")
    print(f"Total time: {time.time() - start_time:.1f}s")

    if not args.no_plot:
        plot_output = args.plot_output
        if plot_output is None:
            output_path = Path(args.output)
            plot_output = str(output_path.with_name(f"{output_path.stem}_dashboard.png"))
        create_runtime_dashboard(df, plot_output=plot_output, show_plot=False)

        # Homotopy disagreement overlay plots
        homotopy_plot = str(Path(args.output).with_name(f"{Path(args.output).stem}_homotopy_disagreements.pdf"))
        create_homotopy_disagreement_plot(df, base_config, homotopy_plot, show_plot=False)


if __name__ == "__main__":
    main()
