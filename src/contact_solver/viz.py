"""Visualization utilities for solver comparison."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from .config import Config


# Plot style configuration
PLOT_CONFIG = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def apply_plot_style():
    """Apply consistent plot styling."""
    plt.rcParams.update(PLOT_CONFIG)


def generate_colors(n: int) -> list:
    """Generate n distinct colors from a colormap."""
    cmap = plt.cm.tab10
    return [cmap(i % 10) for i in range(n)]


def visualize_trajectories(
    positions: list[np.ndarray],
    config: Config,
    title: str = "Trajectories",
    show: bool | None = None,
):
    """Visualize 2D trajectories.

    Args:
        positions: List of (K, 2) position arrays per robot
        config: Configuration
        title: Plot title
        show: Whether to show plot (default: from config)
    """
    apply_plot_style()

    N = len(positions)
    env = config.problem.environment
    robot_radius = config.problem.robot_radius
    min_distance = config.problem.min_distance

    colors = generate_colors(N)

    fig, ax = plt.subplots(figsize=config.visualization.figsize)
    ax.set_aspect("equal")

    # Workspace boundary
    ax.add_patch(
        Rectangle(
            (env.pos_x_min, env.pos_y_min),
            env.pos_x_max - env.pos_x_min,
            env.pos_y_max - env.pos_y_min,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            alpha=0.7,
        )
    )

    # Plot trajectories
    for i in range(N):
        color = colors[i]
        pos = positions[i]

        # Trajectory line
        ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.5, alpha=0.8)

        # Start marker (circle)
        ax.scatter(
            pos[0, 0], pos[0, 1], color=color, marker="o", s=80, edgecolors="black", linewidths=0.5
        )
        ax.add_patch(Circle(pos[0], robot_radius, color=color, alpha=0.3))

        # End marker (square)
        ax.scatter(
            pos[-1, 0],
            pos[-1, 1],
            color=color,
            marker="s",
            s=80,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlim(env.pos_x_min - 1, env.pos_x_max + 1)
    ax.set_ylim(env.pos_y_min - 1, env.pos_y_max + 1)
    ax.set_xlabel(r"$p_x$ [m]")
    ax.set_ylabel(r"$p_y$ [m]")
    ax.set_title(title)

    plt.tight_layout()

    if show is None:
        show = config.visualization.show_plots

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax


def visualize_comparison(
    scp_trajectories: dict,
    contact_trajectories: dict,
    config: Config,
    contact_times: list[float] | None = None,
    title: str = "LiftedSCP vs ContactSolver",
    show: bool | None = None,
):
    """Visualize comparison between SCP and ContactSolver trajectories.

    Args:
        scp_trajectories: Dict with 'positions', 'velocities', 'accelerations' from SCP
        contact_trajectories: Dict with same keys from ContactSolver
        config: Configuration
        contact_times: Optional list of contact times to mark
        title: Plot title
        show: Whether to show plot
    """
    apply_plot_style()

    scp_pos = scp_trajectories["positions"]
    scp_vel = scp_trajectories["velocities"]
    scp_acc = scp_trajectories["accelerations"]

    cont_pos = contact_trajectories["positions"]
    cont_vel = contact_trajectories["velocities"]
    cont_acc = contact_trajectories["accelerations"]

    N = len(scp_pos)
    K_scp = len(scp_pos[0])
    K_cont = len(cont_pos[0])

    h = config.problem.timestep
    T = K_scp * h

    times_scp = np.linspace(0, T, K_scp)
    times_cont = np.linspace(0, T, K_cont)

    min_distance = config.problem.min_distance
    colors = generate_colors(N)

    fig, axes = plt.subplots(4, 2, figsize=(14, 10))

    # Row 0: Positions
    for col, (label, comp) in enumerate([(r"$p_x$ [m]", 0), (r"$p_y$ [m]", 1)]):
        ax = axes[0, col]
        for i in range(N):
            ax.plot(times_scp, scp_pos[i][:, comp], color=colors[i], linewidth=1.5, linestyle="-")
            ax.plot(
                times_cont, cont_pos[i][:, comp], color=colors[i], linewidth=1.5, linestyle="--"
            )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_title(title)

    # Row 1: Velocities
    for col, (label, comp) in enumerate([(r"$v_x$ [m/s]", 0), (r"$v_y$ [m/s]", 1)]):
        ax = axes[1, col]
        for i in range(N):
            ax.plot(times_scp, scp_vel[i][:, comp], color=colors[i], linewidth=1.5, linestyle="-")
            ax.plot(
                times_cont, cont_vel[i][:, comp], color=colors[i], linewidth=1.5, linestyle="--"
            )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # Row 2: Accelerations
    for col, (label, comp) in enumerate([(r"$a_x$ [m/s²]", 0), (r"$a_y$ [m/s²]", 1)]):
        ax = axes[2, col]
        for i in range(N):
            ax.plot(times_scp, scp_acc[i][:, comp], color=colors[i], linewidth=1.5, linestyle="-")
            ax.plot(
                times_cont, cont_acc[i][:, comp], color=colors[i], linewidth=1.5, linestyle="--"
            )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # Row 3, Col 0: Pairwise distances
    ax = axes[3, 0]

    def compute_distances(positions):
        N = len(positions)
        K = len(positions[0])
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        distances = np.zeros((K, len(pairs)))
        for idx, (i, j) in enumerate(pairs):
            for k in range(K):
                distances[k, idx] = np.linalg.norm(positions[i][k] - positions[j][k])
        return distances

    scp_dist = compute_distances(scp_pos)
    cont_dist = compute_distances(cont_pos)

    for idx in range(scp_dist.shape[1]):
        ax.plot(times_scp, scp_dist[:, idx], color="steelblue", linewidth=1, alpha=0.7)
        ax.plot(times_cont, cont_dist[:, idx], color="steelblue", linewidth=1, alpha=0.4, linestyle="--")

    ax.axhline(y=min_distance, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    if contact_times:
        for t in contact_times:
            ax.axvline(x=t, color="green", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$d$ [m]")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Row 3, Col 1: 2D trajectories
    ax = axes[3, 1]
    ax.set_aspect("equal")

    for i in range(N):
        color = colors[i]
        ax.plot(scp_pos[i][:, 0], scp_pos[i][:, 1], color=color, linewidth=1.5, linestyle="-")
        ax.plot(cont_pos[i][:, 0], cont_pos[i][:, 1], color=color, linewidth=1.5, linestyle="--", alpha=0.6)

        ax.scatter(scp_pos[i][0, 0], scp_pos[i][0, 1], color=color, marker="o", s=50, edgecolors="black", linewidths=0.5)
        ax.scatter(scp_pos[i][-1, 0], scp_pos[i][-1, 1], color=color, marker="s", s=50, edgecolors="black", linewidths=0.5)

    ax.set_xlabel(r"$p_x$ [m]")
    ax.set_ylabel(r"$p_y$ [m]")
    ax.grid(True, alpha=0.3)

    # Add legend to first plot
    axes[0, 0].legend(
        [
            plt.Line2D([0], [0], color="black", linestyle="-"),
            plt.Line2D([0], [0], color="black", linestyle="--"),
        ],
        ["LiftedSCP", "ContactSolver"],
        loc="upper right",
        fontsize=8,
    )

    plt.tight_layout()

    if show is None:
        show = config.visualization.show_plots

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


def visualize_pairwise_distances(
    positions: list[np.ndarray],
    config: Config,
    title: str = "Pairwise Distances",
    show: bool | None = None,
):
    """Visualize pairwise distances over time.

    Args:
        positions: List of (K, 2) position arrays per robot
        config: Configuration
        title: Plot title
        show: Whether to show plot
    """
    apply_plot_style()

    N = len(positions)
    K = len(positions[0])
    h = config.problem.timestep
    times = np.arange(K) * h
    min_distance = config.problem.min_distance

    fig, ax = plt.subplots(figsize=(10, 5))

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    colors = generate_colors(len(pairs))

    for idx, (i, j) in enumerate(pairs):
        distances = np.array([np.linalg.norm(positions[i][k] - positions[j][k]) for k in range(K)])
        ax.plot(times, distances, color=colors[idx], linewidth=1.5, label=f"({i}, {j})")

    ax.axhline(y=min_distance, color="red", linestyle="--", linewidth=2, label=f"$R$ = {min_distance}")

    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"Distance [m]")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show is None:
        show = config.visualization.show_plots

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax
