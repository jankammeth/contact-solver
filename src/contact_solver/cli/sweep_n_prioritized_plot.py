#!/usr/bin/env python3
"""Regenerate prioritized sweep dashboard from existing CSV results."""

import argparse
from pathlib import Path

import pandas as pd

from .sweep_n_prioritized import create_runtime_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prioritized sweep dashboard from CSV results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="0_sweep_n_prioritized/sweep_n_prioritized_results.csv",
        help="Path to input prioritized sweep results CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output PNG dashboard (default: alongside input with _dashboard suffix)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if args.output:
        plot_output = Path(args.output)
    else:
        plot_output = input_path.parent / (input_path.stem + "_dashboard.png")

    plot_output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    create_runtime_dashboard(df, plot_output=str(plot_output), show_plot=False)


if __name__ == "__main__":
    main()
