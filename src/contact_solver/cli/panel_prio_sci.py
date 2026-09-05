#!/usr/bin/env python3
"""Per-rank panel plot for the Prioritized SCI solver.

For a given seed and N (default 8), runs PrioritizedSCI rank-by-rank and
produces an N-panel figure where panel k shows the solver state after rank
k has been inserted:

    - all robots' start (●) and goal (■) markers are drawn in every panel
    - robots committed in earlier ranks (0..k-1) are drawn as faded trails
    - the robot just inserted at rank k is drawn prominently in its color
    - the panel title reports the rank's robot index and convergence status

This is intended as a paper figure tool: run it across seeds to find one
that produces a visually clean N=8 sequence, then style it later via janz.

Usage (after `pip install -e .`):

    panel-prio-sci --seed 42
    panel-prio-sci --seed 42 --n 8 --output 0_panel_prio_sci/seed_42.png
    panel-prio-sci --seed-range 0:100  # batch-render seeds 0..99 to scan
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow `python -m` and direct script invocation from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contact_solver import generate_random_positions, load_config
from contact_solver.cli._sweep_utils import build_scaled_config
from contact_solver.config import Config
from contact_solver.solvers.prioritized_sci import PrioritizedSCI
from contact_solver.solvers.prioritized_utils import (
    PiecewiseCubicTrajectory,
    conflict_severity_order,
)


# ──────────────────────────────────────────────────────────────────────────
#  Rank-by-rank driver — replicates PrioritizedSCI.generate_trajectories()
#  but yields snapshots after every rank instead of returning only the final
#  result. Keeps the solver itself untouched.
# ──────────────────────────────────────────────────────────────────────────

def run_prio_sci_per_rank(
    solver: PrioritizedSCI,
) -> tuple[list[int], list[dict[int, PiecewiseCubicTrajectory]], list[tuple[int, bool, str]]]:
    """Run PrioritizedSCI and return (order, snapshots, rank_info).

    snapshots[k] is the dict {robot_id: PiecewiseCubicTrajectory} after the
    k-th rank has been inserted (so snapshots[0] has one robot, snapshots[-1]
    has all N). rank_info[k] is (robot_id, converged_ok, reason_string).
    """
    p0 = solver.initial_positions.reshape(solver.N, 2)
    pf = solver.final_positions.reshape(solver.N, 2)
    v0 = (
        solver.initial_velocities.reshape(solver.N, 2)
        if solver.initial_velocities is not None
        else np.zeros((solver.N, 2))
    )
    vf = (
        solver.final_velocities.reshape(solver.N, 2)
        if solver.final_velocities is not None
        else np.zeros((solver.N, 2))
    )

    order = conflict_severity_order(p0, v0, pf, vf, solver.T, solver.R)

    trajectories: dict[int, PiecewiseCubicTrajectory] = {}
    snapshots: list[dict[int, PiecewiseCubicTrajectory]] = []
    rank_info: list[tuple[int, bool, str]] = []

    for robot_i in order:
        obstacles = dict(trajectories)  # snapshot of priors at this rank
        traj_i, n_contacts, _max_per_pair, ok = solver._solve_single_robot(
            p0[robot_i], v0[robot_i], pf[robot_i], vf[robot_i],
            obstacles, verbose=False,
        )
        trajectories[robot_i] = traj_i

        # Reason matches the solver's own taxonomy: either it converged (all
        # detected violations resolved within max_contacts) or it hit
        # max_iterations on the contact-insertion loop.
        reason = "converged" if ok else "max_iterations"
        rank_info.append((robot_i, ok, reason))
        snapshots.append(dict(trajectories))  # copy: solver may mutate later

    return order, snapshots, rank_info


# ──────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────

def _sample(traj: PiecewiseCubicTrajectory, n_samples: int = 200) -> np.ndarray:
    ts = np.linspace(0.0, traj.T, n_samples)
    return np.array([traj.eval_pos(t) for t in ts])


def plot_panels(
    p0: np.ndarray,
    pf: np.ndarray,
    order: list[int],
    snapshots: list[dict[int, PiecewiseCubicTrajectory]],
    rank_info: list[tuple[int, bool, str]],
    output_path: Path,
    title_prefix: str = "",
) -> None:
    """Render the N-panel figure to output_path."""
    N = len(order)

    # Layout: aim for ~4 columns to match the screenshot's "two rows of four"
    # for N=8. For other N, ceil-divide.
    n_cols = min(4, N)
    n_rows = int(np.ceil(N / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False
    )

    # Color per robot, indexed by rank (so rank 0 is purple/dark, rank N-1
    # is yellow/light — viridis matches the screenshot).
    cmap = plt.get_cmap("viridis")
    rank_colors = {
        order[k]: cmap(k / max(N - 1, 1)) for k in range(N)
    }

    # Shared axis bounds — pad slightly beyond start/goal markers so faded
    # trails don't get clipped.
    all_pts = np.vstack([p0, pf])
    pad = 1.5
    x_min, x_max = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    y_min, y_max = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    span = max(x_max - x_min, y_max - y_min)
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    x_min, x_max = cx - span / 2, cx + span / 2
    y_min, y_max = cy - span / 2, cy + span / 2

    for k in range(N):
        ax = axes.flat[k]
        snap = snapshots[k]
        current_robot = order[k]
        _, ok, reason = rank_info[k]

        # Start/goal markers for ALL robots, in every panel.
        ax.scatter(p0[:, 0], p0[:, 1], marker="o", s=28, color="black",
                   zorder=3)
        ax.scatter(pf[:, 0], pf[:, 1], marker="s", s=28, color="black",
                   zorder=3)

        # Trajectories: previously-placed faded, current bold.
        for rank_k, robot_id in enumerate(order[: k + 1]):
            traj = snap[robot_id]
            pts = _sample(traj)
            color = rank_colors[robot_id]
            if robot_id == current_robot:
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.5,
                        alpha=1.0, zorder=2)
            else:
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.0,
                        alpha=0.45, zorder=1)

        # Title — green check for ok, red cross + reason otherwise. The
        # screenshot uses red text for failed ranks; mirror that.
        mark = "\u2713" if ok else "\u2717"
        title = f"rank {k} (robot {current_robot}): {mark} {reason}"
        ax.set_title(title, fontsize=10, color="black" if ok else "red")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    # Hide unused axes (e.g. N=10 with n_cols=4 leaves 2 empties on row 3).
    for k in range(N, n_rows * n_cols):
        axes.flat[k].set_visible(False)

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
#  CLI entry
# ──────────────────────────────────────────────────────────────────────────

def _render_one(base_config: Config, N: int, seed: int, K: int,
                output_path: Path) -> tuple[bool, float, int]:
    """Solve and render a single (N, seed). Returns (all_ranks_ok, wall, n_fail)."""
    config = build_scaled_config(base_config, N=N, seed=seed, K=K)

    np.random.seed(config.random_seed)
    initial, final = generate_random_positions(config)

    solver = PrioritizedSCI(config, verbose=False)
    solver.set_initial_states(initial)
    solver.set_final_states(final)

    t0 = time.perf_counter()
    order, snapshots, rank_info = run_prio_sci_per_rank(solver)
    wall = time.perf_counter() - t0

    p0 = solver.initial_positions.reshape(N, 2)
    pf = solver.final_positions.reshape(N, 2)

    n_fail = sum(1 for _, ok, _ in rank_info if not ok)
    all_ok = n_fail == 0

    title = f"PrioritizedSCI — N={N}, seed={seed}, t={wall:.2f}s"
    plot_panels(p0, pf, order, snapshots, rank_info, output_path, title)

    return all_ok, wall, n_fail


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-rank panels of PrioritizedSCI for paper figures."
    )
    parser.add_argument("--n", type=int, default=8,
                        help="Number of robots (default: 8)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--seed-range", type=str, default=None,
                        help="Render a batch: 'start:stop' (stop exclusive), "
                             "e.g. '0:50' for seeds 0..49.")
    parser.add_argument("--radius", type=float, default=None,
                        help="Collision radius R in meters (overrides "
                             "problem.min_distance from the config; scenario "
                             "spacing is scaled proportionally). Default: "
                             "config value (0.5 for sweep_n_prioritized).")
    parser.add_argument("--timesteps", type=int, default=200,
                        help="K for cost / similarity grid (does not affect "
                             "SCI itself; default 200)")
    parser.add_argument("--config", type=str, default="sweep_n_prioritized",
                        help="Base config name in configs/ (default: "
                             "sweep_n_prioritized)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (single-seed mode). Default: "
                             "0_panel_prio_sci/n{N}_seed{S}.png")
    parser.add_argument("--output-dir", type=str,
                        default="0_panel_prio_sci",
                        help="Output directory (batch mode; default: "
                             "0_panel_prio_sci)")
    args = parser.parse_args()

    try:
        base_config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: could not load config '{args.config}': {e}")
        sys.exit(2)

    # Optional collision-radius override. Scale the scenario spacing by the
    # same ratio so random start/goal positions cannot spawn inside R.
    if args.radius is not None:
        ratio = args.radius / base_config.problem.min_distance
        base_config = base_config.override(**{
            "problem.min_distance": args.radius,
            "problem.scenario.spacing":
                base_config.problem.scenario.spacing * ratio,
        })
        print(f"Radius override: R={args.radius} "
              f"(spacing scaled x{ratio:.2f})")

    # ── Batch mode: --seed-range ─────────────────────────────────────────
    if args.seed_range is not None:
        try:
            start_s, stop_s = args.seed_range.split(":")
            start, stop = int(start_s), int(stop_s)
        except ValueError:
            print(f"ERROR: --seed-range must be 'start:stop', got "
                  f"'{args.seed_range}'")
            sys.exit(2)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Batch: rendering N={args.n}, seeds {start}..{stop - 1}")
        print(f"  output dir: {out_dir}")
        summary: list[tuple[int, bool, float, int]] = []
        for seed in range(start, stop):
            out = out_dir / f"n{args.n}_seed{seed:04d}.png"
            ok, wall, n_fail = _render_one(
                base_config, args.n, seed, args.timesteps, out,
            )
            summary.append((seed, ok, wall, n_fail))
            mark = "\u2713" if ok else f"\u2717 ({n_fail} failed ranks)"
            print(f"  seed={seed:4d}: {mark}  [{wall:.2f}s]  -> {out.name}")

        ok_seeds = [s for s, ok, _, _ in summary if ok]
        print(f"\nFully-converged seeds ({len(ok_seeds)}/{len(summary)}): "
              f"{ok_seeds}")
        return

    # ── Single-seed mode ─────────────────────────────────────────────────
    if args.output is None:
        out_path = Path(args.output_dir) / f"n{args.n}_seed{args.seed:04d}.png"
    else:
        out_path = Path(args.output)

    ok, wall, n_fail = _render_one(
        base_config, args.n, args.seed, args.timesteps, out_path,
    )
    mark = "\u2713 all ranks converged" if ok else f"\u2717 {n_fail} ranks failed"
    print(f"N={args.n}, seed={args.seed}: {mark}  [{wall:.2f}s]")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
