"""Visualization utilities for solver comparison."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from .config import Config


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
    plt.rcParams.update(PLOT_CONFIG)


def generate_colors(n: int, colormap: str = "gnuplot", start: float = 0.1, end: float = 0.9) -> list:
    if n == 0:
        return []
    
    cmap = plt.cm.get_cmap(colormap)
    
    if n == 1:
        values = [0.5]
    else:
        values = np.linspace(start, end, n)
    
    return [cmap(val)[:3] for val in values]


def _draw_obstacles(ax, obstacle_positions: np.ndarray, radius: float):
    for obs in obstacle_positions:
        circle = Circle(
            obs, radius,
            facecolor="gray",
            edgecolor="black",
            linewidth=1,
            alpha=0.5,
            zorder=1,
        )
        ax.add_patch(circle)


def _env_bounds(env, positions: list[np.ndarray] | None = None, margin: float = 1.0):
    """Get plot bounds from env config, falling back to trajectory data."""
    if env.pos_x_min is not None and env.pos_x_max is not None:
        x_min, x_max = env.pos_x_min, env.pos_x_max
    elif positions is not None:
        all_x = np.concatenate([p[:, 0] for p in positions])
        x_min, x_max = all_x.min() - margin, all_x.max() + margin
    else:
        x_min, x_max = -10.0, 10.0

    if env.pos_y_min is not None and env.pos_y_max is not None:
        y_min, y_max = env.pos_y_min, env.pos_y_max
    elif positions is not None:
        all_y = np.concatenate([p[:, 1] for p in positions])
        y_min, y_max = all_y.min() - margin, all_y.max() + margin
    else:
        y_min, y_max = -10.0, 10.0

    return x_min, x_max, y_min, y_max


def _has_env_bounds(env) -> bool:
    """Check if environment has explicit position limits."""
    return all(v is not None for v in [env.pos_x_min, env.pos_x_max, env.pos_y_min, env.pos_y_max])


def visualize_trajectories(
    positions: list[np.ndarray],
    config: Config,
    obstacle_positions: np.ndarray | None = None,
    title: str = "Trajectories",
    show: bool | None = None,
):
    apply_plot_style()

    N = len(positions)
    env = config.problem.environment
    robot_radius = config.problem.robot_radius
    min_distance = config.problem.min_distance

    colors = generate_colors(N)

    fig, ax = plt.subplots(figsize=config.visualization.figsize)
    ax.set_aspect("equal")

    x_min, x_max, y_min, y_max = _env_bounds(env, positions)

    if _has_env_bounds(env):
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
                linestyle="--",
                alpha=0.7,
            )
        )

    if obstacle_positions is not None and len(obstacle_positions) > 0:
        _draw_obstacles(ax, obstacle_positions, min_distance)

    for i in range(N):
        color = colors[i]
        pos = positions[i]

        ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.5, alpha=0.8, zorder=2)

        ax.scatter(
            pos[0, 0], pos[0, 1], color=color, marker="o", s=80,
            edgecolors="black", linewidths=0.5, zorder=4
        )
        ax.add_patch(Circle(pos[0], robot_radius, color=color, alpha=0.3, zorder=3))

        ax.scatter(
            pos[-1, 0], pos[-1, 1], color=color, marker="s", s=80,
            edgecolors="black", linewidths=0.5, zorder=4
        )

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
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


def visualize_time_snapshots(
    positions: list[np.ndarray],
    config: Config,
    n_snapshots: int = 5,
    obstacle_positions: np.ndarray | None = None,
    title: str = "Time Snapshots",
    show: bool | None = None,
):
    apply_plot_style()

    N = len(positions)
    K = len(positions[0])
    env = config.problem.environment
    robot_radius = config.problem.robot_radius
    min_distance = config.problem.min_distance
    timestep = config.problem.timestep

    colors = generate_colors(N)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))
    if n_snapshots == 1:
        axes = [axes]

    frame_indices = np.linspace(0, K - 1, n_snapshots, dtype=int)

    for f_idx, (ax, frame_idx) in enumerate(zip(axes, frame_indices)):
        t = frame_idx * timestep

        ax.set_aspect("equal")
        x_min, x_max, y_min, y_max = _env_bounds(env, positions, margin=0.5)
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_title(rf"$t = {t:.2f}$ s")

        ax.set_xticks([])
        ax.set_yticks([])

        if _has_env_bounds(env):
            ax.add_patch(
                Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=1,
                    edgecolor="black",
                    facecolor="none",
                    linestyle="--",
                    alpha=0.5,
                )
            )

        if obstacle_positions is not None and len(obstacle_positions) > 0:
            _draw_obstacles(ax, obstacle_positions, min_distance)

        for i in range(N):
            color = colors[i]
            pos = positions[i]
            current_pos = pos[frame_idx]

            if frame_idx > 0:
                history = pos[:frame_idx + 1]
                ax.plot(
                    history[:, 0], history[:, 1],
                    color=color, linewidth=1, alpha=0.5, zorder=2
                )

            safety_circle = Circle(
                current_pos, min_distance,
                facecolor=color, edgecolor="none",
                alpha=0.1, zorder=2
            )
            ax.add_patch(safety_circle)

            robot_circle = Circle(
                current_pos, robot_radius,
                facecolor=color, edgecolor="black",
                linewidth=0.5, alpha=0.8, zorder=3
            )
            ax.add_patch(robot_circle)

            if f_idx == 0:
                ax.scatter(
                    pos[0, 0], pos[0, 1],
                    marker="o", s=30, color="white",
                    edgecolors=color, linewidths=1, zorder=4
                )

            if f_idx == n_snapshots - 1:
                ax.scatter(
                    pos[-1, 0], pos[-1, 1],
                    marker="s", s=30, color="white",
                    edgecolors=color, linewidths=1, zorder=4
                )

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if show is None:
        show = config.visualization.show_plots

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


def visualize_comparison(
    scp_trajectories: dict,
    contact_trajectories: dict,
    config: Config,
    contact_times: list[float] | None = None,
    obstacle_positions: np.ndarray | None = None,
    title: str = "LiftedSCP vs ContactSolver",
    show: bool | None = None,
):
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
    T = config.problem.time_horizon
    min_distance = config.problem.min_distance

    times_scp = h * np.arange(1, K_scp + 1)
    times_cont = np.linspace(0, T, K_cont)

    colors = generate_colors(N)

    fig, axes = plt.subplots(6, 2, figsize=(14, 14))

    def compute_jerk(accelerations: list[np.ndarray], dt: float) -> list[np.ndarray]:
        jerks = []
        for acc in accelerations:
            if len(acc) < 2:
                jerks.append(np.zeros((0, acc.shape[1])))
            else:
                jerks.append(np.diff(acc, axis=0) / dt)
        return jerks

    dt_cont = T / max(K_cont - 1, 1)
    scp_jerk = compute_jerk(scp_acc, h)
    cont_jerk = compute_jerk(cont_acc, dt_cont)
    times_scp_jerk = times_scp[1:]
    times_cont_jerk = times_cont[1:]

    scp_snap = compute_jerk(scp_jerk, h)
    cont_snap = compute_jerk(cont_jerk, dt_cont)
    times_scp_snap = times_scp_jerk[1:]
    times_cont_snap = times_cont_jerk[1:]

    for col, (label, comp) in enumerate([(r"$p_x$ [m]", 0), (r"$p_y$ [m]", 1)]):
        ax = axes[0, col]
        for i in range(N):
            ax.plot(times_scp, scp_pos[i][:, comp], color=colors[i], linewidth=1.5,
                    linestyle="-", marker=".", markersize=2, alpha=0.8)
            ax.plot(times_cont, cont_pos[i][:, comp], color=colors[i], linewidth=1.5,
                    linestyle="--", alpha=0.6)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_title(title)

    for col, (label, comp) in enumerate([(r"$v_x$ [m/s]", 0), (r"$v_y$ [m/s]", 1)]):
        ax = axes[1, col]
        for i in range(N):
            ax.plot(times_scp, scp_vel[i][:, comp], color=colors[i], linewidth=1.5,
                    linestyle="-", marker=".", markersize=2, alpha=0.8)
            ax.plot(times_cont, cont_vel[i][:, comp], color=colors[i], linewidth=1.5,
                    linestyle="--", alpha=0.6)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    for col, (label, comp) in enumerate([(r"$a_x$ [m/s²]", 0), (r"$a_y$ [m/s²]", 1)]):
        ax = axes[2, col]
        for i in range(N):
            ax.plot(times_scp, scp_acc[i][:, comp], color=colors[i], linewidth=1.5,
                    linestyle="-", marker=".", markersize=2, alpha=0.8)
            ax.plot(times_cont, cont_acc[i][:, comp], color=colors[i], linewidth=1.5,
                    linestyle="--", alpha=0.6)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    for col, (label, comp) in enumerate([(r"$j_x$ [m/s³]", 0), (r"$j_y$ [m/s³]", 1)]):
        ax = axes[3, col]
        for i in range(N):
            ax.plot(times_scp_jerk, scp_jerk[i][:, comp], color=colors[i], linewidth=1.5,
                    linestyle="-", marker=".", markersize=2, alpha=0.8)
            ax.plot(times_cont_jerk, cont_jerk[i][:, comp], color=colors[i], linewidth=1.5,
                    linestyle="--", alpha=0.6)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    for col, (label, comp) in enumerate([(r"$s_x$ [m/s⁴]", 0), (r"$s_y$ [m/s⁴]", 1)]):
        ax = axes[4, col]
        for i in range(N):
            ax.plot(times_scp_snap, scp_snap[i][:, comp], color=colors[i], linewidth=1.5,
                    linestyle="-", marker=".", markersize=2, alpha=0.8)
            ax.plot(times_cont_snap, cont_snap[i][:, comp], color=colors[i], linewidth=1.5,
                    linestyle="--", alpha=0.6)

        scp_snap_vals = [snap[:, comp] for snap in scp_snap if snap.size > 0]
        if scp_snap_vals:
            y = np.concatenate(scp_snap_vals)
            y_min = float(np.min(y))
            y_max = float(np.max(y))
            if y_max > y_min:
                pad = 0.1 * (y_max - y_min)
            else:
                pad = max(1e-3, 0.1 * max(abs(y_min), abs(y_max), 1.0))
            ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    ax = axes[5, 0]

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
        ax.plot(times_scp, scp_dist[:, idx], color="steelblue", linewidth=1.2,
                linestyle="-", alpha=0.7)
        ax.plot(times_cont, cont_dist[:, idx], color="steelblue", linewidth=1.2,
                linestyle="--", alpha=0.4)

    ax.axhline(y=min_distance, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    if contact_times:
        for t in contact_times:
            ax.axvline(x=t, color="green", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$d$ [m]")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[5, 1]
    ax.set_aspect("equal")

    if obstacle_positions is not None and len(obstacle_positions) > 0:
        _draw_obstacles(ax, obstacle_positions, min_distance)

    for i in range(N):
        color = colors[i]
        ax.plot(scp_pos[i][:, 0], scp_pos[i][:, 1], color=color, linewidth=1.5,
                linestyle="-", alpha=0.8)
        ax.plot(cont_pos[i][:, 0], cont_pos[i][:, 1], color=color, linewidth=1.5,
                linestyle="--", alpha=0.5)

        ax.scatter(scp_pos[i][0, 0], scp_pos[i][0, 1], color=color, marker="o", s=50,
                   edgecolors="black", linewidths=0.5)
        ax.scatter(scp_pos[i][-1, 0], scp_pos[i][-1, 1], color=color, marker="s", s=50,
                   edgecolors="black", linewidths=0.5)

    ax.set_xlabel(r"$p_x$ [m]")
    ax.set_ylabel(r"$p_y$ [m]")
    ax.grid(True, alpha=0.3)

    axes[0, 0].legend(
        [
            plt.Line2D([0], [0], color="black", linestyle="-", marker=".", markersize=3),
            plt.Line2D([0], [0], color="black", linestyle="--"),
        ],
        ["LiftedSCP (discrete)", "ContactSolver (continuous)"],
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
