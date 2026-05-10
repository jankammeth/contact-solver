#!/usr/bin/env python3
"""Generate 2D scenario configuration plots for the paper.

Creates a figure with multiple panels, each showing a scenario's start/end
positions connected by thin lines, with collision radii and sampling disk.

Usage:
    configurations
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from contact_solver.config import Config, ProblemConfig
from contact_solver.scenarios import generate_random_positions


# ── Define scenarios: (N, seed) pairs ────────────────────────
SCENARIOS = [
    (5, 42),
    (10, 7),
    (20, 14),
]

# ── Style ────────────────────────────────────────────────────
C_START = "#4C78A8"
C_END = "#F58518"
LINE_WIDTH = 0.6
LINE_ALPHA = 0.35
MARKER_SIZE = 3.5
RADIUS_ALPHA = 0.12
RADIUS_EDGE_ALPHA = 0.3
RADIUS_LW = 0.4

# ── Problem parameters ───────────────────────────────────────
MIN_DISTANCE = 0.5
ROBOT_RADIUS = 0.25
CLUSTER_RADIUS = 5.0
SPACING = 0.8


def make_config(n_robots: int) -> Config:
    problem = ProblemConfig(
        n_robots=n_robots,
        time_horizon=10.0,
        timestep=0.1,
        min_distance=MIN_DISTANCE,
        robot_radius=ROBOT_RADIUS,
    )
    problem.scenario.cluster_radius = CLUSTER_RADIUS
    problem.scenario.spacing = SPACING
    return Config(name="config_plot", problem=problem)


def plot_scenario(ax, n_robots: int, seed: int):
    cfg = make_config(n_robots)
    initial, final = generate_random_positions(cfg, seed=seed)
    R = MIN_DISTANCE / 2  # collision radius per robot

    # Connecting lines
    for i in range(n_robots):
        ax.plot(
            [initial[i, 0], final[i, 0]],
            [initial[i, 1], final[i, 1]],
            color="#888888", lw=LINE_WIDTH, alpha=LINE_ALPHA, zorder=1,
        )

    # Collision radii (circles)
    for i in range(n_robots):
        ax.add_patch(plt.Circle(
            initial[i], R, facecolor=C_START, edgecolor=C_START,
            alpha=RADIUS_ALPHA, linewidth=RADIUS_LW, zorder=2,
        ))
        ax.add_patch(plt.Circle(
            final[i], R, facecolor=C_END, edgecolor=C_END,
            alpha=RADIUS_ALPHA, linewidth=RADIUS_LW, zorder=2,
        ))

    # Markers
    ax.scatter(
        initial[:, 0], initial[:, 1],
        marker="o", s=MARKER_SIZE**2, color=C_START,
        edgecolors="none", zorder=3, label="$p_0$",
    )
    ax.scatter(
        final[:, 0], final[:, 1],
        marker="s", s=MARKER_SIZE**2, color=C_END,
        edgecolors="none", zorder=3, label="$p_f$",
    )

    # Info text
    info = f"$N={n_robots}$,  $R={MIN_DISTANCE}$\,m,  $r_{{\mathrm{{cluster}}}}={CLUSTER_RADIUS}$\,m"
    ax.set_title(info, fontsize=9)

    # Sampling disk
    ax.add_patch(plt.Circle(
        (0, 0), CLUSTER_RADIUS, facecolor="none", edgecolor="#AAAAAA",
        linestyle="--", linewidth=0.6, zorder=0,
    ))

    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.12, linewidth=0.4)


def main():
    out_dir = Path("0_configurations")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_scenarios = len(SCENARIOS)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(4 * n_scenarios, 4))
    if n_scenarios == 1:
        axes = [axes]

    for ax, (n, seed) in zip(axes, SCENARIOS):
        plot_scenario(ax, n, seed)

    # Shared legend from first axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = out_dir / "configurations.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
