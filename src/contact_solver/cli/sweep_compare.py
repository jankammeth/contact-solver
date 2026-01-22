#!/usr/bin/env python3
"""Sweep comparison of ContactSolver vs LiftedSCP across varying robot counts."""

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contact_solver import (
    ContactSolver,
    LiftedSCP,
    generate_random_positions,
    load_config,
)
from contact_solver.config import Config


def compute_scale_factor(N: int, base_N: int = 2) -> float:
    """Compute scaling factor for N robots relative to base_N."""
    return math.sqrt(N / base_N)


def build_scaled_config(base_config: Config, N: int, seed: int, K: int = 100) -> Config:
    """Build a config with parameters scaled for N robots."""
    scale = compute_scale_factor(N, base_N=2)

    # Scale workspace and time with sqrt(N) to maintain congestion
    base_time = 10.0
    base_workspace = 25.0
    base_cluster_radius = 5.0

    time_horizon = base_time * scale
    workspace = base_workspace * scale
    cluster_radius = base_cluster_radius * scale
    timestep = time_horizon / K

    return base_config.override(**{
        "random_seed": seed,
        "problem.n_robots": N,
        "problem.time_horizon": time_horizon,
        "problem.timestep": timestep,
        "problem.environment.pos_x_min": 0.0,
        "problem.environment.pos_x_max": workspace,
        "problem.environment.pos_y_min": 0.0,
        "problem.environment.pos_y_max": workspace,
        "problem.scenario.cluster_radius": cluster_radius,
        "visualization.show_plots": False,
    })


def compute_metrics(trajectories: dict, config: Config) -> dict:
    """Compute trajectory metrics."""
    positions = trajectories["positions"]
    velocities = trajectories["velocities"]
    accelerations = trajectories["accelerations"]

    N = len(positions)
    K = len(positions[0])
    dt = config.problem.time_horizon / (K - 1) if K > 1 else config.problem.timestep

    # Min pairwise distance
    min_dist = np.inf
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i][k] - positions[j][k])
                min_dist = min(min_dist, dist)

    # Max velocity
    max_vel = max(np.max(np.abs(v)) for v in velocities)

    # Acceleration cost
    cost = 0.0
    for a in accelerations:
        a_sq = np.sum(a**2, axis=1)
        cost += np.trapezoid(a_sq, dx=dt)

    return {
        "min_distance": min_dist,
        "max_velocity": max_vel,
        "acceleration_cost": cost,
    }


def run_solver_with_timeout(solver, timeout: float):
    """Run solver with timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(solver.generate_trajectories)
        try:
            result = future.result(timeout=timeout)
            return result, False
        except FuturesTimeoutError:
            return None, True


def run_single_experiment(config: Config, timeout: float = 300.0) -> dict:
    """Run both solvers on a single configuration."""
    np.random.seed(config.random_seed)
    initial, final = generate_random_positions(config)

    results = {
        "N": config.problem.n_robots,
        "seed": config.random_seed,
        "K": config.problem.n_timesteps,
    }

    # LiftedSCP
    try:
        scp_solver = LiftedSCP(config, verbose=False)
        scp_solver.set_initial_states(initial)
        scp_solver.set_final_states(final)

        scp_result, scp_timeout = run_solver_with_timeout(scp_solver, timeout)

        if scp_timeout:
            results.update({
                "scp_time": timeout,
                "scp_converged": False,
                "scp_timeout": True,
            })
        else:
            scp_metrics = compute_metrics(scp_result["trajectories"], config)
            results.update({
                "scp_time": scp_result["metrics"]["timing"]["total_time"],
                "scp_iterations": scp_result["metrics"]["scp_iterations"],
                "scp_converged": scp_result["metrics"]["converged"],
                "scp_min_dist": scp_metrics["min_distance"],
                "scp_max_vel": scp_metrics["max_velocity"],
                "scp_cost": scp_metrics["acceleration_cost"],
                "scp_timeout": False,
            })
    except Exception as e:
        results.update({
            "scp_time": np.nan,
            "scp_converged": False,
            "scp_error": str(e),
        })

    # ContactSolver
    try:
        contact_solver = ContactSolver(config, verbose=False)
        contact_solver.set_initial_states(initial)
        contact_solver.set_final_states(final)

        contact_result, contact_timeout = run_solver_with_timeout(contact_solver, timeout)

        if contact_timeout:
            results.update({
                "contact_time": timeout,
                "contact_converged": False,
                "contact_timeout": True,
            })
        else:
            contact_metrics = compute_metrics(contact_result["trajectories"], config)
            results.update({
                "contact_time": contact_result["metrics"]["timing"]["total_time"],
                "contact_contacts": contact_result["metrics"]["num_contacts"],
                "contact_converged": contact_result["metrics"]["converged"],
                "contact_min_dist": contact_metrics["min_distance"],
                "contact_max_vel": contact_metrics["max_velocity"],
                "contact_cost": contact_metrics["acceleration_cost"],
                "contact_timeout": False,
            })
    except Exception as e:
        results.update({
            "contact_time": np.nan,
            "contact_converged": False,
            "contact_error": str(e),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep comparison across robot counts")
    parser.add_argument("--n-min", type=int, default=2, help="Minimum robots")
    parser.add_argument("--n-max", type=int, default=20, help="Maximum robots")
    parser.add_argument("--n-step", type=int, default=2, help="Robot count step")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per N")
    parser.add_argument("--timesteps", type=int, default=100, help="Number of timesteps")
    parser.add_argument("--timeout", type=float, default=300, help="Solver timeout (seconds)")
    parser.add_argument("--output", type=str, default="sweep_results.csv", help="Output CSV")
    parser.add_argument("--config", type=str, default="sweep", help="Base config name")
    args = parser.parse_args()

    # Load base config
    try:
        base_config = load_config(args.config)
    except FileNotFoundError:
        from contact_solver import get_small_config
        base_config = get_small_config()

    N_values = list(range(args.n_min, args.n_max + 1, args.n_step))
    seeds = list(range(args.seeds))

    print("=" * 60)
    print("Sweep Comparison: ContactSolver vs LiftedSCP")
    print("=" * 60)
    print(f"N values: {N_values}")
    print(f"Seeds per N: {args.seeds}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Timeout: {args.timeout}s")
    print(f"Output: {args.output}")
    print()

    all_results = []
    total_experiments = len(N_values) * len(seeds)
    completed = 0

    start_time = time.time()

    for N in N_values:
        print(f"\nN = {N} robots:")

        for seed in seeds:
            config = build_scaled_config(base_config, N, seed, args.timesteps)

            result = run_single_experiment(config, args.timeout)
            all_results.append(result)

            completed += 1
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total_experiments - completed)

            # Status indicator
            scp_ok = result.get("scp_converged", False)
            contact_ok = result.get("contact_converged", False)
            scp_time = result.get("scp_time", np.nan)
            contact_time = result.get("contact_time", np.nan)

            status = f"  seed={seed}: SCP={'✓' if scp_ok else '✗'} ({scp_time:.2f}s), "
            status += f"Contact={'✓' if contact_ok else '✗'} ({contact_time:.2f}s)"
            print(status)

        # Save intermediate results
        df = pd.DataFrame(all_results)
        df.to_csv(args.output, index=False)

    # Final summary
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for N in N_values:
        subset = df[df["N"] == N]

        scp_times = subset["scp_time"].dropna()
        contact_times = subset["contact_time"].dropna()
        scp_success = subset["scp_converged"].sum()
        contact_success = subset["contact_converged"].sum()

        print(f"\nN = {N}:")
        print(f"  LiftedSCP:     {scp_times.mean():.3f}s ± {scp_times.std():.3f}s, success: {scp_success}/{len(subset)}")
        print(f"  ContactSolver: {contact_times.mean():.3f}s ± {contact_times.std():.3f}s, success: {contact_success}/{len(subset)}")

        if len(scp_times) > 0 and len(contact_times) > 0:
            speedup = scp_times.mean() / contact_times.mean()
            print(f"  Speedup: {speedup:.2f}x")

    print(f"\nResults saved to: {args.output}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
