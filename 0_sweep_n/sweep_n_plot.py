#!/usr/bin/env python3
"""Regenerate the sweep dashboard for a single run folder.

Usage:
    python 0_sweep_n/sweep_n_plot.py --folder run_K100
    python 0_sweep_n/sweep_n_plot.py --folder run_K200
"""

import argparse
from pathlib import Path

import pandas as pd

from contact_solver.cli.sweep_n import create_runtime_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate sweep_n dashboard for a run folder.")
    parser.add_argument("--folder", type=str, required=True,
                        help="Run folder under 0_sweep_n/ (e.g. run_K100, run_K200).")
    args = parser.parse_args()

    run_dir = Path("0_sweep_n") / args.folder
    csv_path = run_dir / "sweep_n_results.csv"
    plot_output = run_dir / "sweep_n_results_dashboard.png"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    create_runtime_dashboard(df, plot_output=str(plot_output), show_plot=False)


if __name__ == "__main__":
    main()
