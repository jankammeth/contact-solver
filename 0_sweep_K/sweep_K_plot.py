#!/usr/bin/env python3
"""Regenerate sweep_K dashboard plots from an existing CSV file."""

from pathlib import Path

import pandas as pd

from contact_solver.cli.sweep_K_plot import create_sweep_K_dashboard


def main() -> None:
    input_path = Path("0_sweep_K/sweep_K_results.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    plot_output = Path("0_sweep_K/sweep_K_results_dashboard.png")
    plot_output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    create_sweep_K_dashboard(df, plot_output=str(plot_output), show_plot=False)


if __name__ == "__main__":
    main()
