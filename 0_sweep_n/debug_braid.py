#!/usr/bin/env python3
"""Debug braid word disagreements: print exact braid words and perpendicular separations."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contact_solver import ContactSolver, LiftedSCP, generate_random_positions, load_config
from contact_solver.cli.sweep_n import (
    build_scaled_config,
    compute_braid_word,
    normalize_braid_word,
)


def analyze_case(base_config, N, seed):
    config = build_scaled_config(base_config, N, seed, K=100)
    np.random.seed(config.random_seed)
    initial, final = generate_random_positions(config)
    R = config.problem.min_distance

    # Run SCP
    scp = LiftedSCP(config, verbose=False)
    scp.set_initial_states(initial)
    scp.set_final_states(final)
    scp_pos = scp.generate_trajectories()["trajectories"]["positions"]

    # Run ContactSolver, sample at SCP grid
    contact = ContactSolver(config, verbose=False)
    contact.set_initial_states(initial)
    contact.set_final_states(final)
    contact.generate_trajectories()
    K = config.problem.n_timesteps
    h = config.problem.timestep
    _, contact_pos, _, _ = contact.sample(times=np.arange(1, K + 1) * h)

    # Braid words (no min_sep filtering)
    braid_scp = compute_braid_word(scp_pos)
    braid_contact = compute_braid_word(contact_pos)
    norm_scp = normalize_braid_word(braid_scp)
    norm_contact = normalize_braid_word(braid_contact)

    print(f"\n{'=' * 70}")
    print(f"N={N}, seed={seed}, R={R}")
    print(f"  Raw crossings: SCP={len(braid_scp)}, Contact={len(braid_contact)}")
    print(f"  Normalized:    SCP={len(norm_scp)}, Contact={len(norm_contact)}")
    print()

    # Print normalized words side by side
    max_len = max(len(norm_scp), len(norm_contact))
    print(f"  {'idx':>4s}  {'SCP':>20s}  {'Contact':>20s}  {'Match':>5s}")
    print(f"  {'-'*55}")
    for idx in range(max_len):
        s = norm_scp[idx] if idx < len(norm_scp) else None
        c = norm_contact[idx] if idx < len(norm_contact) else None
        match = "  ✓" if s == c else "  ✗"
        print(f"  {idx:4d}  {str(s):>20s}  {str(c):>20s}  {match}")

    # For each crossing in the SCP braid word, compute perpendicular separation
    print(f"\n  Perpendicular separations at crossings (R={R}):")
    proj_scp = np.array([scp_pos[r][:, 0] for r in range(len(scp_pos))])

    for ci, (i, j, sign) in enumerate(braid_scp):
        for k in range(1, K):
            prev = proj_scp[i, k - 1] - proj_scp[j, k - 1]
            curr = proj_scp[i, k] - proj_scp[j, k]
            if prev * curr < 0:
                perp_i = 0.5 * (scp_pos[i][k - 1, 1] + scp_pos[i][k, 1])
                perp_j = 0.5 * (scp_pos[j][k - 1, 1] + scp_pos[j][k, 1])
                sep = abs(perp_i - perp_j)
                flag = " <-- below R!" if sep < R else ""
                print(f"    SCP crossing {ci}: pair=({i},{j}) sign={sign:+d} k={k} perp_sep={sep:.4f}{flag}")
                break

    # Same for contact
    proj_con = np.array([contact_pos[r][:, 0] for r in range(len(contact_pos))])
    for ci, (i, j, sign) in enumerate(braid_contact):
        for k in range(1, K):
            prev = proj_con[i, k - 1] - proj_con[j, k - 1]
            curr = proj_con[i, k] - proj_con[j, k]
            if prev * curr < 0:
                perp_i = 0.5 * (contact_pos[i][k - 1, 1] + contact_pos[i][k, 1])
                perp_j = 0.5 * (contact_pos[j][k - 1, 1] + contact_pos[j][k, 1])
                sep = abs(perp_i - perp_j)
                flag = " <-- below R!" if sep < R else ""
                print(f"    CON crossing {ci}: pair=({i},{j}) sign={sign:+d} k={k} perp_sep={sep:.4f}{flag}")
                break


def main():
    base_config = load_config("sweep_n")
    cases = [(6, 18), (6, 37), (6, 41), (6, 44), (8, 3), (8, 23), (8, 30)]
    for N, seed in cases:
        analyze_case(base_config, N, seed)


if __name__ == "__main__":
    main()
