#!/usr/bin/env python3
"""Dashboard for sweep_K results (LiftedSCP vs ContactSolver across K)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


APPLE_STUDIO_WIDTH_PX = 5120
APPLE_STUDIO_HEIGHT_PX = 2880
PLOT_DPI = 200

# Colors match sweep_n
C_SCP = "#4C78A8"
C_CONTACT = "#F58518"

# Per-experiment constants (must match configs/sweep_n.yaml).
R_DEFAULT = 0.5
DETECTION_TOL_DEFAULT = 1e-3


def _placeholder(ax, title: str) -> None:
    """Draw an empty placeholder panel."""
    ax.text(
        0.5, 0.5, "(placeholder)",
        ha="center", va="center", transform=ax.transAxes,
        color="gray", fontsize=10, style="italic",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def _grouped_stats(df: pd.DataFrame, group: str, col: str, mask=None) -> pd.DataFrame:
    """Median + IQR + 95% CI per group, on rows passing `mask`."""
    sub = df if mask is None else df.loc[mask]
    sub = sub.dropna(subset=[col, group])
    grouped = sub.groupby(group)[col]
    out = pd.DataFrame({
        "median": grouped.median(),
        "q25": grouped.quantile(0.25),
        "q75": grouped.quantile(0.75),
        "q025": grouped.quantile(0.025),
        "q975": grouped.quantile(0.975),
        "n": grouped.count(),
    }).reset_index().sort_values(group)
    return out


def _panel_runtime(ax, df: pd.DataFrame) -> None:
    """Runtime (median + IQR + 95% CI) vs K, both solvers."""
    K_values = sorted(df["K"].dropna().unique().astype(int).tolist())

    scp = _grouped_stats(df, "K", "scp_time", mask=df.get("scp_success") == True)
    sci = _grouped_stats(df, "K", "contact_time", mask=df.get("contact_success") == True)

    for stats, color, marker, label in [
        (scp, C_SCP, "o", "LiftedSCP"),
        (sci, C_CONTACT, "s", "ContactSolver"),
    ]:
        if len(stats) == 0:
            continue
        x = stats["K"].to_numpy(dtype=float)
        # 95% CI (lighter)
        ax.fill_between(x, stats["q025"], stats["q975"], color=color, alpha=0.10, linewidth=0)
        # IQR (darker)
        ax.fill_between(x, stats["q25"], stats["q75"], color=color, alpha=0.18, linewidth=0)
        # Median line
        ax.plot(x, stats["median"], color=color, marker=marker, lw=2.0, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$K$")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime vs $K$")
    ax.set_xticks(K_values)
    ax.get_xaxis().set_major_formatter(
        plt.matplotlib.ticker.ScalarFormatter()
    )
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="best")


def _panel_cost_gap(ax, df: pd.DataFrame) -> None:
    """Relative cost gap (J_SCP - J_SCI) / J_SCI vs K."""
    K_values = sorted(df["K"].dropna().unique().astype(int).tolist())

    both_ok = df.loc[
        (df.get("scp_success") == True)
        & (df.get("contact_success") == True)
    ].copy()
    both_ok = both_ok.dropna(subset=["scp_cost", "contact_cost"])
    both_ok = both_ok[both_ok["contact_cost"] > 0]
    both_ok["gap"] = (both_ok["scp_cost"] - both_ok["contact_cost"]) / both_ok["contact_cost"]

    stats = _grouped_stats(both_ok, "K", "gap")
    if len(stats) > 0:
        x = stats["K"].to_numpy(dtype=float)
        # 95% CI (lighter)
        ax.fill_between(x, stats["q025"], stats["q975"], color=C_SCP, alpha=0.10,
                        linewidth=0)
        # IQR (darker)
        ax.fill_between(x, stats["q25"], stats["q75"], color=C_SCP, alpha=0.18,
                        linewidth=0, label="IQR")
        # Median line
        ax.plot(x, stats["median"], color=C_SCP, marker="o", lw=2.0,
                label="median gap")

    ax.axhline(0.0, color="black", lw=1.0, alpha=0.6, ls="--",
               label="SCI baseline")
    ax.set_xscale("log")
    ax.set_xlabel("$K$")
    ax.set_ylabel(r"$(J_\mathrm{SCP} - J_\mathrm{SCI}) / J_\mathrm{SCI}$")
    ax.set_title("Cost gap vs $K$")
    ax.set_xticks(K_values)
    ax.get_xaxis().set_major_formatter(
        plt.matplotlib.ticker.ScalarFormatter()
    )
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="best")


def _panel_scp_violation(
    ax,
    df: pd.DataFrame,
    R: float = R_DEFAULT,
) -> None:
    """Relative inter-sample collision violation of the SCP trajectory vs K.

    Plots (R - d_min_continuous) / R, where d_min_continuous is the minimum
    pairwise distance along SCP's interpolated trajectory (using SCP's own
    piecewise-constant-acceleration integrator between grid points).

    Interpretation:
        0   = safe (disks just touch or are separated)
        1   = robot centers coincide (full overlap)
        0.8 = 80% overlap

    SCP enforces the collision constraint only at its K grid points;
    between samples, the implied continuous-time trajectory can dip below R.
    """
    K_values = sorted(df["K"].dropna().unique().astype(int).tolist())
    if not K_values:
        ax.set_title("SCP inter-sample violation vs $K$")
        return

    succ = df.loc[df.get("scp_success") == True].copy()

    has_continuous = (
        "scp_min_dist_continuous" in succ.columns
        and succ["scp_min_dist_continuous"].notna().any()
    )

    if has_continuous:
        succ["rel_violation"] = (
            (R - succ["scp_min_dist_continuous"]) / R
        ).clip(lower=0.0)
        stats = _grouped_stats(succ, "K", "rel_violation")
        if len(stats) > 0:
            x = stats["K"].to_numpy(dtype=float)
            # 95% CI (lighter)
            ax.fill_between(x, stats["q025"], stats["q975"],
                            color=C_SCP, alpha=0.10, linewidth=0)
            # IQR (darker)
            ax.fill_between(x, stats["q25"], stats["q75"],
                            color=C_SCP, alpha=0.18, linewidth=0, label="IQR")
            # Median line
            ax.plot(x, stats["median"], color=C_SCP, marker="o", lw=2.0,
                    label="median")
    else:
        ax.text(
            0.5, 0.5,
            "scp_min_dist_continuous not in CSV\n— re-run sweep-K to populate",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="gray", style="italic",
        )

    ax.axhline(0.0, color="black", lw=1.0, alpha=0.6, ls="--",
               label="no violation")
    ax.axhline(1.0, color="red", lw=1.0, alpha=0.5, ls=":",
               label="full overlap")
    ax.set_xscale("log")
    ax.set_xlabel("$K$")
    ax.set_ylabel(r"$(R - d_\mathrm{min}^\mathrm{cont}) / R$")
    ax.set_title("SCP inter-sample violation vs $K$")
    ax.set_xticks(K_values)
    ax.get_xaxis().set_major_formatter(
        plt.matplotlib.ticker.ScalarFormatter()
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25, which="both")
    if has_continuous:
        ax.legend(fontsize=8, loc="best")


def _panel_speedup(ax, df: pd.DataFrame) -> None:
    """Per-seed speedup ratio t_SCP / t_SCI vs K.

    Values > 1 mean SCP is slower than SCI; values < 1 mean SCP is faster.
    The horizontal reference at y=1 marks SCI parity; the curve's crossing
    of that line is the K* at which SCP first matches SCI's runtime.

    Ratios are computed per (seed, K), then aggregated, so the IQR captures
    instance-to-instance variation in the runtime ratio (not in raw runtimes).
    """
    K_values = sorted(df["K"].dropna().unique().astype(int).tolist())

    both_ok = df.loc[
        (df.get("scp_success") == True)
        & (df.get("contact_success") == True)
    ].copy()
    both_ok = both_ok.dropna(subset=["scp_time", "contact_time"])
    both_ok = both_ok[both_ok["contact_time"] > 0]
    both_ok["ratio"] = both_ok["scp_time"] / both_ok["contact_time"]

    stats = _grouped_stats(both_ok, "K", "ratio")
    if len(stats) > 0:
        x = stats["K"].to_numpy(dtype=float)
        # 95% CI (lighter)
        ax.fill_between(x, stats["q025"], stats["q975"],
                        color=C_SCP, alpha=0.10, linewidth=0)
        # IQR (darker)
        ax.fill_between(x, stats["q25"], stats["q75"],
                        color=C_SCP, alpha=0.18, linewidth=0, label="IQR")
        # Median line
        ax.plot(x, stats["median"], color=C_SCP, marker="o", lw=2.0,
                label="median")

    ax.axhline(1.0, color="black", lw=1.0, alpha=0.6, ls="--",
               label="SCI parity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$K$")
    ax.set_ylabel(r"$t_\mathrm{SCP} / t_\mathrm{SCI}$")
    ax.set_title("Speedup vs $K$")
    ax.set_xticks(K_values)
    ax.get_xaxis().set_major_formatter(
        plt.matplotlib.ticker.ScalarFormatter()
    )
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="best")


def create_sweep_K_dashboard(
    df: pd.DataFrame,
    plot_output: str,
    show_plot: bool = False,
) -> None:
    """Create and save the sweep_K dashboard.

    Layout (2x3):
        [ Runtime vs K ]    [ Cost gap vs K ]    [ SCP inter-sample violation ]
        [ Speedup vs K ]    [ placeholder    ]    [ placeholder ]
    """
    width_in = APPLE_STUDIO_WIDTH_PX / PLOT_DPI
    height_in = APPLE_STUDIO_HEIGHT_PX / PLOT_DPI

    fig, axes = plt.subplots(2, 3, figsize=(width_in, height_in), dpi=PLOT_DPI)

    n_seeds = df.groupby("K").size().min() if "K" in df else "?"
    N_unique = df["N"].dropna().unique().astype(int).tolist() if "N" in df else []
    N_str = f"N={N_unique[0]}" if len(N_unique) == 1 else f"N={N_unique}"
    fig.suptitle(f"sweep_K dashboard — {N_str}, {n_seeds} seeds per K", fontsize=18)

    # ── Filled panels ──────────────────────────────────────────
    _panel_runtime(axes[0, 0], df)
    _panel_cost_gap(axes[0, 1], df)
    _panel_scp_violation(axes[0, 2], df)
    _panel_speedup(axes[1, 0], df)

    # ── Placeholders ───────────────────────────────────────────
    _placeholder(axes[1, 1], "Position RMSE vs $K$")
    _placeholder(axes[1, 2], "Outcome breakdown vs $K$")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_output, dpi=PLOT_DPI)
    print(f"Dashboard saved to: {plot_output}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    """Regenerate the sweep_K dashboard from an existing CSV."""
    parser = argparse.ArgumentParser(description="Generate sweep_K dashboard from CSV results")
    parser.add_argument("--input", type=str, default="0_sweep_K/sweep_K_results.csv",
                        help="Path to input sweep_K results CSV (default: 0_sweep_K/sweep_K_results.csv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output PNG dashboard (default: same dir as input with _dashboard suffix)")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Determine output path
    if args.output:
        plot_output = Path(args.output)
    else:
        # Default: put dashboard next to input CSV with _dashboard suffix
        plot_output = input_path.parent / (input_path.stem + "_dashboard.png")
    
    plot_output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    create_sweep_K_dashboard(df, plot_output=str(plot_output), show_plot=False)


if __name__ == "__main__":
    main()
