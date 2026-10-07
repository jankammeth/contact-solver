#!/usr/bin/env python3
"""Compare ContactSolver vs LiftedSCP on a single problem instance."""

import argparse
import sys
from pathlib import Path

import numpy as np

from contact_solver.evaluation import (
    continuous_acceleration_cost,
    discrete_acceleration_cost,
    sample_scp_grid,
)

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contact_solver import (
    LiftedSCP,
    ContactSolver,
    generate_random_positions,
    generate_swap_positions,
    load_config,
    visualize_comparison,
)



def compute_max_velocity(velocities: list[np.ndarray]) -> float:
    """Compute max component-wise velocity."""
    return max(np.max(np.abs(v)) for v in velocities)


def compute_min_distance(positions: list[np.ndarray]) -> float:
    """Compute minimum pairwise distance."""
    N = len(positions)
    K = len(positions[0])
    min_dist = np.inf
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i][k] - positions[j][k])
                min_dist = min(min_dist, dist)
    return min_dist


def main():
    parser = argparse.ArgumentParser(description="Compare ContactSolver vs LiftedSCP")
    parser.add_argument("--config", default="small", help="Config name or path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--scenario", choices=["random", "swap"], default="random")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        from contact_solver import get_small_config
        config = get_small_config()

    # Override config with command-line args
    overrides = {"visualization.show_plots": not args.no_plot}
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    config = config.override(**overrides)

    # Generate positions (uses config.random_seed)
    if args.scenario == "swap":
        initial, final = generate_swap_positions(config)
    else:
        initial, final = generate_random_positions(config)

    N = config.problem.n_robots
    h = config.problem.timestep
    T = config.problem.time_horizon
    R = config.problem.min_distance
    v_max = config.problem.dynamics.vel_max

    print("=" * 60)
    print("Solver Comparison: ContactSolver vs LiftedSCP")
    print("=" * 60)
    print(f"\nConfiguration: {config.name}")
    print(f"  N = {N} robots")
    print(f"  T = {T}s, h = {h}s, K = {config.problem.n_timesteps}")
    print(f"  R = {R}m (min distance)")
    print(f"  v_max = {v_max} m/s")

    # ==================== LiftedSCP ====================
    print("\n" + "-" * 40)
    print("LiftedSCP (QP-based)")
    print("-" * 40)

    scp_solver = LiftedSCP(config, verbose=args.verbose)
    scp_solver.set_initial_states(initial)
    scp_solver.set_final_states(final)
    scp_result = scp_solver.generate_trajectories()

    scp_traj = scp_result["trajectories"]
    scp_metrics = scp_result["metrics"]

    scp_time = scp_metrics["timing"]["total_time"]
    scp_iters = scp_metrics["scp_iterations"]
    scp_min_dist = compute_min_distance(scp_traj["positions"])
    scp_max_vel = compute_max_velocity(scp_traj["velocities"])
    scp_cost = discrete_acceleration_cost(scp_traj, h)

    print(f"  Time: {scp_time:.4f}s")
    print(f"  SCP iterations: {scp_iters}")
    print(f"  Converged: {scp_metrics['converged']} ({scp_metrics['convergence_reason']})")
    print(f"  Min distance: {scp_min_dist:.4f}m {'✓' if scp_min_dist >= R - 0.01 else '✗'}")
    if v_max is not None:
        print(f"  Max velocity: {scp_max_vel:.4f} m/s {'✓' if scp_max_vel <= v_max + 0.01 else '✗'}")
    else:
        print(f"  Max velocity: {scp_max_vel:.4f} m/s (unconstrained)")
    print(f"  Acceleration cost: {scp_cost:.4f}")

    # ==================== ContactSolver ====================
    print("\n" + "-" * 40)
    print("ContactSolver (Analytical)")
    print("-" * 40)

    contact_solver = ContactSolver(config, verbose=args.verbose)
    contact_solver.set_initial_states(initial)
    contact_solver.set_final_states(final)
    contact_result = contact_solver.generate_trajectories()

    contact_traj = contact_result["trajectories"]
    contact_metrics = contact_result["metrics"]
    contact_times = contact_result.get("contact_times", [])

    contact_time = contact_metrics["timing"]["total_time"]
    contact_min_dist = contact_metrics["min_distance"]
    contact_max_vel = compute_max_velocity(contact_traj["velocities"])
    contact_cost = discrete_acceleration_cost(sample_scp_grid(contact_solver, scp_solver.K, scp_solver.h), scp_solver.h)
    contact_cost_continuous = continuous_acceleration_cost(contact_solver)
    num_contacts = contact_metrics["num_contacts"]

    print(f"  Time: {contact_time:.4f}s")
    print(f"  Contacts: {num_contacts}")
    print(f"  Converged: {contact_metrics['converged']} ({contact_metrics['convergence_reason']})")
    print(f"  Min distance: {contact_min_dist:.4f}m {'✓' if contact_min_dist >= R - 0.01 else '✗'}")
    if v_max is not None:
        print(f"  Max velocity: {contact_max_vel:.4f} m/s {'✓' if contact_max_vel <= v_max + 0.01 else '✗'}")
    else:
        print(f"  Max velocity: {contact_max_vel:.4f} m/s (unconstrained)")
    print(f"  Acceleration cost: {contact_cost:.4f}")
    print(f"  Continuous acceleration cost (separate): {contact_cost_continuous:.4f}")

    # ==================== Comparison ====================
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    speedup = scp_time / contact_time if contact_time > 0 else float("inf")
    cost_diff = (contact_cost / scp_cost - 1) * 100 if scp_cost > 0 else 0

    print("\nTiming:")
    print(f"  LiftedSCP:     {scp_time:.4f}s")
    print(f"  ContactSolver: {contact_time:.4f}s")
    if speedup >= 1:
        print(f"  → ContactSolver is {speedup:.1f}x faster")
    else:
        print(f"  → LiftedSCP is {1/speedup:.1f}x faster")

    print(f"\nFeasibility (R = {R}m):")
    print(f"  LiftedSCP:     {scp_min_dist:.4f}m {'✓' if scp_min_dist >= R - 0.01 else '✗'}")
    print(f"  ContactSolver: {contact_min_dist:.4f}m {'✓' if contact_min_dist >= R - 0.01 else '✗'}")

    print("\nAcceleration Cost:")
    print(f"  LiftedSCP:     {scp_cost:.4f}")
    print(f"  ContactSolver: {contact_cost:.4f} ({cost_diff:+.1f}%)")

    # ==================== Visualization ====================
    if not args.no_plot:
        print("\nGenerating visualization...")
        visualize_comparison(
            scp_traj,
            contact_traj,
            config,
            contact_times=contact_times,
            title=f"LiftedSCP vs ContactSolver ({N} robots)",
        )


if __name__ == "__main__":
    main()
