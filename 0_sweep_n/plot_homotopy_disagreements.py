#!/usr/bin/env python3
"""Plot overlay of SCP and ContactSolver trajectories for homotopy disagreement cases."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contact_solver import ContactSolver, LiftedSCP, generate_random_positions, load_config
from contact_solver.cli.sweep_n import build_scaled_config, compute_braid_word, compare_braid_words


def plot_case(ax_scp, ax_contact, ax_overlay, config, seed, case_label):
    """Run both solvers and plot trajectories."""
    np.random.seed(config.random_seed)
    initial, final = generate_random_positions(config)
    N = config.problem.n_robots
    R = config.problem.min_distance

    # Run SCP
    scp = LiftedSCP(config, verbose=False)
    scp.set_initial_states(initial)
    scp.set_final_states(final)
    scp_result = scp.generate_trajectories()
    scp_pos = scp_result["trajectories"]["positions"]

    # Run ContactSolver
    contact = ContactSolver(config, verbose=False)
    contact.set_initial_states(initial)
    contact.set_final_states(final)
    contact_result = contact.generate_trajectories()

    # Sample at SCP grid
    K = config.problem.n_timesteps
    h = config.problem.timestep
    scp_times = np.arange(1, K + 1) * h
    _, contact_pos, _, _ = contact.sample(times=scp_times)

    # Braid comparison
    braid_scp = compute_braid_word(scp_pos)
    braid_contact = compute_braid_word(contact_pos)
    cmp = compare_braid_words(braid_scp, braid_contact)

    # Colors for robots
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(N)]

    for ax, pos, title in [
        (ax_scp, scp_pos, "LiftedSCP"),
        (ax_contact, contact_pos, "ContactSolver"),
    ]:
        for i in range(N):
            ax.plot(pos[i][:, 0], pos[i][:, 1], color=colors[i], lw=0.8, alpha=0.7)
        ax.scatter(initial[:, 0], initial[:, 1], c="black", s=15, zorder=5, marker="o")
        ax.scatter(final[:, 0], final[:, 1], c="black", s=15, zorder=5, marker="s")
        ax.set_aspect("equal")
        ax.set_title(f"{title}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.15)

    # Overlay
    for i in range(N):
        ax_overlay.plot(scp_pos[i][:, 0], scp_pos[i][:, 1],
                        color=colors[i], lw=0.8, alpha=0.5, ls="-")
        ax_overlay.plot(contact_pos[i][:, 0], contact_pos[i][:, 1],
                        color=colors[i], lw=0.8, alpha=0.5, ls="--")
    ax_overlay.scatter(initial[:, 0], initial[:, 1], c="black", s=15, zorder=5, marker="o")
    ax_overlay.scatter(final[:, 0], final[:, 1], c="black", s=15, zorder=5, marker="s")
    ax_overlay.set_aspect("equal")
    ax_overlay.set_title(f"Overlay  ({case_label})", fontsize=8)
    ax_overlay.tick_params(labelsize=6)
    ax_overlay.grid(True, alpha=0.15)


def main():
    # Load the CSV to find disagreement cases
    csv_path = Path("0_sweep_n/sweep_n_results.csv")
    df = pd.read_csv(csv_path)
    disagree = df[df["same_homotopy_class"] == False][["N", "seed"]].values.tolist()

    if not disagree:
        print("No homotopy disagreements found.")
        return

    print(f"Found {len(disagree)} disagreement cases")

    try:
        base_config = load_config("sweep_n")
    except FileNotFoundError:
        from contact_solver import get_small_config
        base_config = get_small_config()

    n_cases = len(disagree)
    fig, axes = plt.subplots(n_cases, 3, figsize=(14, 4 * n_cases))
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    for row, (N, seed) in enumerate(disagree):
        N, seed = int(N), int(seed)
        config = build_scaled_config(base_config, N, seed, K=100)
        label = f"N={N}, seed={seed}"
        print(f"  Plotting {label}...")
        plot_case(axes[row, 0], axes[row, 1], axes[row, 2], config, seed, label)

    fig.suptitle("Homotopy Class Disagreements: SCP vs ContactSolver", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path("0_sweep_n") / "homotopy_disagreements.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
