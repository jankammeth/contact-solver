"""Compare the joint solvers on saved N-dimensional endpoints or 16 small smoke cases."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from contact_solver import Config, LiftedSCP, ContactSolver
from contact_solver.cli.vector_sci import smoke_endpoints
from contact_solver.config import ProblemConfig, SolverConfig
from contact_solver.evaluation import (
    continuous_acceleration_cost,
    discrete_acceleration_cost,
    sample_scp_grid,
)


def compare_instance(config, states):
    if config.problem.time_horizon is None:
        raise ValueError("Paired comparison requires an explicit common time_horizon")
    if config.solver.scp_detection_tol != config.solver.sci_detection_tol:
        raise ValueError("Use equal scp_detection_tol and sci_detection_tol for paired comparisons")
    records = {}
    for name, cls in (("scp", LiftedSCP), ("sci", ContactSolver)):
        solver = cls(config)
        solver.set_initial_states(states["initial_positions"], states.get("initial_velocities"))
        solver.set_final_states(states["final_positions"], states.get("final_velocities"))
        started = time.perf_counter()
        result = solver.generate_trajectories()
        call_time = time.perf_counter() - started
        metrics = result["metrics"]
        kept = (
            "status",
            "converged",
            "convergence_reason",
            "numerical_converged",
            "continuous_feasible",
            "stopping_reason",
            "dimension",
            "n_timesteps",
            "system_size",
            "vars_per_timestep",
            "intrinsic_degrees_of_freedom",
            "num_contacts",
            "min_distance",
            "grid_min_distance",
            "closest_time",
            "closest_pair",
            "checks",
            "tolerances",
            "timing",
            "scp_iterations",
            "matching_seconds",
            "roots",
            "qp_error",
        )
        record = {key: metrics[key] for key in kept if key in metrics}
        record["call_wall_time"] = call_time
        evaluated = result["trajectories"] if name == "scp" else sample_scp_grid(solver, config.problem.n_timesteps)
        record["acceleration_cost"] = discrete_acceleration_cost(evaluated, solver.h)
        record["cost_grid"] = dict(rule="right endpoints: h, 2h, ..., K*h", K=config.problem.n_timesteps, weight=solver.h)
        if name == "sci":
            record["continuous_acceleration_cost"] = continuous_acceleration_cost(solver)
        if name == "scp":
            solves = [
                metrics["osqp_metrics"]["initial_solve"],
                *metrics["osqp_metrics"]["iterations"],
            ]
            record["qp_solves"] = [
                {key: info.get(key) for key in ("status", "iter", "pri_res", "dua_res", "run_time")}
                for info in solves
            ]
        records[name] = record
    return dict(
        dimension=config.problem.dimension,
        n_robots=config.problem.n_robots,
        horizon=config.problem.time_horizon,
        timestep=config.problem.timestep,
        endpoints={key: np.asarray(value).tolist() for key, value in states.items()},
        **records,
    )


def smoke_report():
    records = []
    for d in (2, 3, 5, 10):
        for case in ("free", "offset", "velocity", "fleet"):
            start, goal = smoke_endpoints(d, "offset" if case == "velocity" else case)
            v0, vf = np.zeros_like(start), np.zeros_like(start)
            if case == "velocity":
                if d == 2:
                    v0[:] = [[0.2, -0.1], [-0.15, 0.05]]
                    vf[:] = [[0.1, 0.2], [-0.1, -0.2]]
                else:
                    v0[:, 2], vf[:, 2] = [0.15, -0.1], [-0.1, 0.2]
            config = Config(
                name=f"joint_smoke_{d}_{case}",
                problem=ProblemConfig(
                    n_robots=len(start), dimension=d, time_horizon=6, timestep=0.1, min_distance=1
                ),
                solver=SolverConfig(
                    scp_detection_tol=1e-3,
                    sci_detection_tol=1e-3,
                    scp_convergence_abs=1e-5,
                    osqp_eps_abs=1e-7,
                    osqp_eps_rel=1e-7,
                    scp_max_iterations=50,
                    osqp_max_iter=20000,
                    timeout=30,
                ),
            )
            record = compare_instance(
                config,
                dict(
                    initial_positions=start,
                    final_positions=goal,
                    initial_velocities=v0,
                    final_velocities=vf,
                ),
            )
            record["case"] = case
            records.append(record)
    # Include hashes of the actual solver/checker sources, including local changes.
    root = Path(__file__).parents[1]
    paths = (
        "solvers/lifted_scp.py",
        "solvers/zoh_trajectory.py",
        "solvers/contact_solver.py",
        "solvers/vector_matching.py",
        "solvers/cubic_utils.py",
    )
    return dict(
        environment=dict(
            python=sys.version,
            platform=platform.platform(),
            packages={name: version(name) for name in ("numpy", "scipy", "osqp")},
            thread_limits={
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
            source_sha256={
                name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths
            },
        ),
        records=records,
    )


def json_values(value):
    """Keep reports standards-compliant: an absent minimum is null, never Infinity."""
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_values(item) for item in value]
    if isinstance(value, np.generic):
        return json_values(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--states", type=Path, help="NPZ with initial/final positions and optional velocities"
    )
    source.add_argument("--smoke", action="store_true", help="Only 16 small cases, N <= 3")
    parser.add_argument("--config", type=Path, help="YAML config required with --states")
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    args = parser.parse_args(argv)
    if args.smoke:
        if args.config:
            parser.error("--config applies only to --states")
        report = smoke_report()
        successful = all(
            row["scp"]["converged"] and row["sci"]["converged"] for row in report["records"]
        )
    else:
        if args.config is None:
            parser.error("--states requires --config")
        config = Config.from_yaml(args.config)
        with np.load(args.states, allow_pickle=False) as data:
            states = {
                key: data[key]
                for key in (
                    "initial_positions",
                    "final_positions",
                    "initial_velocities",
                    "final_velocities",
                )
                if key in data
            }
        report = compare_instance(config, states)
        successful = report["scp"]["converged"] and report["sci"]["converged"]
    encoded = json.dumps(json_values(report), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())
