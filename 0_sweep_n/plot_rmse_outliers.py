#!/usr/bin/env python3
"""Plot trajectory overlays for same-homotopy-class RMSE outliers."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contact_solver import ContactSolver, LiftedSCP, generate_random_positions, load_config
from contact_solver.cli._sweep_utils import build_scaled_config


def find_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Find RMSE outliers in same-homotopy-class cases using 1.5×IQR rule."""
    same = df[df['same_homotopy_class'] == True].dropna(subset=['position_rmse'])
    
    q1 = same['position_rmse'].quantile(0.25)
    q3 = same['position_rmse'].quantile(0.75)
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    
    outliers = same[same['position_rmse'] > fence].copy()
    outliers = outliers.sort_values('position_rmse', ascending=False)
    
    print(f"Outlier detection: Q1={q1:.2e}, Q3={q3:.2e}, IQR={iqr:.2e}, fence={fence:.2e}")
    print(f"Found {len(outliers)} outliers\n")
    
    return outliers


def run_and_plot_case(ax, base_config, N, seed, rmse, max_dev, K=200):
    """Run both solvers and plot trajectory overlay on given axis."""
    config = build_scaled_config(base_config, N, seed, K=K)
    
    np.random.seed(config.random_seed)
    initial, final = generate_random_positions(config)
    
    # Run LiftedSCP
    scp = LiftedSCP(config, verbose=False)
    scp.set_initial_states(initial)
    scp.set_final_states(final)
    scp_result = scp.generate_trajectories()
    scp_pos = scp_result["trajectories"]["positions"]
    
    # Run ContactSolver and sample at SCP grid
    contact = ContactSolver(config, verbose=False)
    contact.set_initial_states(initial)
    contact.set_final_states(final)
    contact_result = contact.generate_trajectories()
    
    h = config.problem.timestep
    scp_times = np.arange(1, K + 1) * h
    _, contact_pos, _, _ = contact.sample(times=scp_times)
    
    # Plot trajectories
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(N)]
    
    for i in range(N):
        # SCP: solid lines
        ax.plot(scp_pos[i][:, 0], scp_pos[i][:, 1],
                color=colors[i], lw=1.0, alpha=0.6, ls="-")
        # ContactSolver: dashed lines
        ax.plot(contact_pos[i][:, 0], contact_pos[i][:, 1],
                color=colors[i], lw=1.0, alpha=0.6, ls="--")
    
    # Initial/final positions
    ax.scatter(initial[:, 0], initial[:, 1], c="black", s=25, zorder=5, marker="o", edgecolors='white', linewidths=0.5)
    ax.scatter(final[:, 0], final[:, 1], c="black", s=25, zorder=5, marker="s", edgecolors='white', linewidths=0.5)
    
    ax.set_aspect("equal")
    ax.set_title(
        f"N={N}, seed={seed}\n"
        f"RMSE={rmse:.2e}, max_dev={max_dev:.2e}",
        fontsize=8
    )
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.15)


def main():
    parser = argparse.ArgumentParser(description="Plot RMSE outlier trajectories")
    parser.add_argument("--input", type=str, default="0_sweep_n/run_K200/sweep_n_results.csv",
                        help="Input sweep results CSV")
    parser.add_argument("--output", type=str, default="0_sweep_n/run_K200/rmse_outliers.png",
                        help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()
    
    # Load data and find outliers
    df = pd.read_csv(args.input)
    outliers = find_outliers(df)
    
    if len(outliers) == 0:
        print("No outliers found.")
        return
    
    # Load base config
    base_config = load_config("sweep_n")
    
    # Create figure: 4 columns, enough rows for all cases
    n_cases = len(outliers)
    n_cols = 4
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), squeeze=False)
    
    # Plot each outlier
    for idx, (_, row) in enumerate(outliers.iterrows()):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        
        N = int(row['N'])
        seed = int(row['seed'])
        rmse = row['position_rmse']
        max_dev = row['max_position_deviation']
        
        print(f"[{idx+1}/{n_cases}] Plotting N={N}, seed={seed}, RMSE={rmse:.2e}...", flush=True)
        
        try:
            run_and_plot_case(ax, base_config, N, seed, rmse, max_dev, K=200)
            print(f"  ✓ Complete", flush=True)
        except Exception as e:
            print(f"  ✗ Error: {e}", flush=True)
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:50]}", 
                   ha="center", va="center", transform=ax.transAxes, fontsize=6)
            ax.set_title(f"N={N}, seed={seed} (FAILED)", fontsize=8, color='red')
        
        # Add legend to first panel only
        if idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color="gray", lw=1.5, ls="-", label="LiftedSCP"),
                Line2D([0], [0], color="gray", lw=1.5, ls="--", label="ContactSolver"),
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc="upper right")
    
    # Hide unused subplots
    for idx in range(n_cases, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis('off')
    
    fig.suptitle("Same Homotopy Class — High RMSE Outliers (K=200)", fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved {n_cases} outlier plots to: {output_path}")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
