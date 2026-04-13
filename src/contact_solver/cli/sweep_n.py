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
    scp_data = [df.loc[df["N"] == n, scp_col].dropna().to_numpy() for n in n_values]
    contact_data = [df.loc[df["N"] == n, contact_col].dropna().to_numpy() for n in n_values]

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
    """Create and save a 3x3 dashboard."""
    width_in = APPLE_STUDIO_WIDTH_PX / PLOT_DPI
    height_in = APPLE_STUDIO_HEIGHT_PX / PLOT_DPI
    fig, axes = plt.subplots(3, 3, figsize=(width_in, height_in), dpi=PLOT_DPI)
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

    # ── Panel 2: Runtime mean ± std ──────────────────────────
    ax = axes[0, 1]
    grouped = (
        df.groupby("N", as_index=False)
        .agg(
            scp_mean=("scp_time", "mean"),
            scp_std=("scp_time", "std"),
            contact_mean=("contact_time", "mean"),
            contact_std=("contact_time", "std"),
        )
        .sort_values("N")
    )
    if len(grouped) > 0:
        x = grouped["N"].to_numpy(dtype=float)
        for col, color, marker, label in [
            ("scp", C_SCP, "o", "LiftedSCP"),
            ("contact", C_CONTACT, "s", "ContactSolver"),
        ]:
            m = grouped[f"{col}_mean"].to_numpy(dtype=float)
            s = np.nan_to_num(grouped[f"{col}_std"].to_numpy(dtype=float), nan=0.0)
            ax.plot(x, m, color=color, lw=2.2, marker=marker, label=label)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.18)
        ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.set_title("Runtime (Mean ± Std)")
    ax.set_xlabel("N")
    ax.set_ylabel("Time [s]")
    ax.grid(True, alpha=0.25)

    # ── Panel 3: Runtime median + IQR ────────────────────────
    ax = axes[0, 2]
    grouped_iqr = (
        df.groupby("N", as_index=False)
        .agg(
            scp_median=("scp_time", "median"),
            scp_q25=("scp_time", lambda x: x.quantile(0.25)),
            scp_q75=("scp_time", lambda x: x.quantile(0.75)),
            contact_median=("contact_time", "median"),
            contact_q25=("contact_time", lambda x: x.quantile(0.25)),
            contact_q75=("contact_time", lambda x: x.quantile(0.75)),
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
            ax.plot(x, median, color=color, lw=2.2, marker=marker, label=label)
            ax.fill_between(x, q25, q75, color=color, alpha=0.18)
        ax.legend(fontsize=8)

    ax.set_yscale("log")
    ax.set_title("Runtime (Median + IQR)")
    ax.set_xlabel("N")
    ax.set_ylabel("Time [s]")
    ax.grid(True, alpha=0.25)

    # ── Panel 4: Success rate + failure breakdown ────────────
    ax = axes[1, 0]
    scp_success_rate, contact_success_rate = [], []
    # Collect failure reason categories
    scp_fail_reasons_all = set()
    contact_fail_reasons_all = set()
    for n in n_values:
        sub = df[df["N"] == n]
        scp_success_rate.append(sub["scp_success"].mean() if "scp_success" in sub else 0)
        contact_success_rate.append(sub["contact_success"].mean() if "contact_success" in sub else 0)
        if "scp_reason" in sub:
            scp_fail_reasons_all.update(sub.loc[sub["scp_success"] == False, "scp_reason"].dropna().unique())
        if "contact_reason" in sub:
            contact_fail_reasons_all.update(sub.loc[sub["contact_success"] == False, "contact_reason"].dropna().unique())

    x_arr = np.array(n_values, dtype=float)
    ax.plot(x_arr, scp_success_rate, color=C_SCP, lw=2.2, marker="o", label="LiftedSCP")
    ax.plot(x_arr, contact_success_rate, color=C_CONTACT, lw=2.2, marker="s", label="ContactSolver")

    # Stacked failure bars (thin, below the line plot)
    bar_w = 0.35 * (x_arr[1] - x_arr[0]) if len(x_arr) > 1 else 0.5
    fail_colors_scp = {"max_iterations": "#8FAADC", "infeasible": "#B4C7E7", "timeout": "#D6E4F0"}
    fail_colors_contact = {"max_contacts": "#F4B183", "max_contacts_per_pair": "#F8CBAD", "timeout": "#FCE4D6"}

    ax2 = ax.twinx()
    for solver, reasons, colors, x_off in [
        ("scp", sorted(scp_fail_reasons_all), fail_colors_scp, -bar_w / 2),
        ("contact", sorted(contact_fail_reasons_all), fail_colors_contact, bar_w / 2),
    ]:
        bottom = np.zeros(len(n_values))
        for reason in reasons:
            counts = []
            for n in n_values:
                sub = df[df["N"] == n]
                col = f"{solver}_reason"
                if col in sub:
                    counts.append(((sub[col] == reason) & (sub[f"{solver}_success"] == False)).sum())
                else:
                    counts.append(0)
            counts = np.array(counts, dtype=float)
            if counts.sum() > 0:
                ax2.bar(
                    x_arr + x_off, counts, bar_w * 0.9, bottom=bottom,
                    color=colors.get(reason, "#cccccc"), alpha=0.6, label=reason,
                )
                bottom += counts

    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Success Rate & Failures")
    ax.set_xlabel("N")
    ax.set_ylabel("Success rate")
    ax2.set_ylabel("Failure count")
    # Combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 5: Cost ratio (contact / scp) ──────────────────
    ax = axes[1, 1]
    for i, n in enumerate(n_values):
        sub = df[df["N"] == n].dropna(subset=["scp_cost", "contact_cost"])
        ratio = (sub["contact_cost"] / sub["scp_cost"]).to_numpy()
        if len(ratio) > 0:
            ax.scatter(
                np.full(len(ratio), n) + np.random.uniform(-0.3, 0.3, len(ratio)),
                ratio, s=12, alpha=0.5, color=C_CONTACT, edgecolors="none",
            )
    ax.axhline(1.0, color="black", ls="-", lw=1.2, alpha=0.8)
    ax.set_title("Cost Ratio (Contact / SCP)")
    ax.set_xlabel("N")
    ax.set_ylabel("Cost ratio")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 6: Min distance / R ────────────────────────────
    ax = axes[1, 2]
    for i, n in enumerate(n_values):
        sub = df[df["N"] == n]
        R = 0.5  # min_distance from config
        for col, color, x_off in [("scp_min_dist", C_SCP, -0.15), ("contact_min_dist", C_CONTACT, 0.15)]:
            vals = (sub[col].dropna() / R).to_numpy()
            if len(vals) > 0:
                jitter = np.random.uniform(-0.08, 0.08, len(vals))
                ax.scatter(
                    np.full(len(vals), n) + x_off + jitter,
                    vals, s=12, alpha=0.5, color=color, edgecolors="none",
                )
    ax.axhline(1.0, color="black", ls="-", lw=1.2, alpha=0.8, label="d = R")
    ax.legend(
        [plt.Line2D([0], [0], color=C_SCP, marker="o", ls="", ms=5),
         plt.Line2D([0], [0], color=C_CONTACT, marker="o", ls="", ms=5),
         plt.Line2D([0], [0], color="black", ls="-", lw=1.2)],
        ["LiftedSCP", "ContactSolver", "d = R"],
        fontsize=7,
    )
    ax.set_title("Min Distance / R")
    ax.set_xlabel("N")
    ax.set_ylabel("min_dist / R")
    ax.grid(True, alpha=0.25, axis="y")

    # ── Panel 7: Contacts (total + max per pair) ─────────────
    ax = axes[2, 0]
    if "contact_contacts" in df and "contact_max_contacts_single_pair" in df:
        total_data, max_pair_data, c_labels, c_pos = [], [], [], []
        for i, n in enumerate(n_values):
            sub = df.loc[(df["N"] == n) & (df["contact_success"] == True)]
            t = sub["contact_contacts"].dropna().to_numpy()
            m = sub["contact_max_contacts_single_pair"].dropna().to_numpy()
            if len(t) > 0:
                total_data.append(t)
                max_pair_data.append(m)
                c_pos.append(i + 1)
                c_labels.append(str(n))

        if total_data:
            off = 0.18
            bp1 = ax.boxplot(
                total_data, positions=np.array(c_pos) - off, widths=0.32,
                patch_artist=True, manage_ticks=False,
            )
            bp2 = ax.boxplot(
                max_pair_data, positions=np.array(c_pos) + off, widths=0.32,
                patch_artist=True, manage_ticks=False,
            )
            for p in bp1["boxes"]:
                p.set_facecolor(C_CONTACT)
                p.set_alpha(0.7)
            for p in bp2["boxes"]:
                p.set_facecolor("#E45756")
                p.set_alpha(0.7)
            ax.set_xticks(c_pos)
            ax.set_xticklabels(c_labels)

            # Limit lines from config
            ax.axhline(30, color=C_CONTACT, ls="--", lw=1, alpha=0.5, label="max_contacts (30)")
            ax.axhline(10, color="#E45756", ls="--", lw=1, alpha=0.5, label="max_per_pair (10)")
            ax.legend(
                [bp1["boxes"][0], bp2["boxes"][0],
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


def compute_metrics(trajectories: dict, config: Config) -> dict:
    """Compute trajectory metrics."""
    positions = trajectories["positions"]
    accelerations = trajectories["accelerations"]

    N = len(positions)
    K = len(positions[0])
    dt = config.problem.time_horizon / (K - 1) if K > 1 else config.problem.timestep

    # Min pairwise distance
    min_dist = np.inf
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i][k] - positions[j][k])
                min_dist = min(min_dist, dist)

    # Acceleration cost
    cost = 0.0
    for a in accelerations:
        a_sq = np.sum(a**2, axis=1)
        cost += np.trapezoid(a_sq, dx=dt)

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
                debug_dir = Path("sweep_n/debug")
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
            contact_metrics = compute_metrics(contact_result["trajectories"], config)
            reason = contact_result["metrics"]["convergence_reason"]
            success = contact_result["metrics"]["converged"]

            contact_traj = contact_result["trajectories"] if success else None

            results.update({
                "contact_success": success,
                "contact_reason": reason,
                "contact_time": contact_result["metrics"]["timing"]["total_time"],
                "contact_contacts": contact_result["metrics"]["num_contacts"],
                "contact_max_contacts_single_pair": contact_result["metrics"]["max_contacts_single_pair"],
                "contact_cost": contact_metrics["acceleration_cost"],
                "contact_min_dist": contact_metrics["min_distance"],
            })
    except Exception as e:
        results.update({
            "contact_success": False,
            "contact_reason": "error",
            "contact_time": np.nan,
            "contact_error": str(e),
        })

    # ── Cross-solver metrics (only when both succeed) ────────
    if scp_traj is not None and contact_traj is not None:
        cross = compute_cross_solver_metrics(scp_traj, contact_traj, config)
        results.update({
            "position_rmse": cross["position_rmse"],
            "max_position_deviation": cross["max_position_deviation"],
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
    parser.add_argument("--output", type=str, default="sweep_n/sweep_n_results.csv", help="Output CSV")
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


if __name__ == "__main__":
    main()
