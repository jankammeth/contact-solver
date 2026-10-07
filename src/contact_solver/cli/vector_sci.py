"""Run explicit-vector SCI on NPZ endpoint data, or a bounded 20-case smoke suite."""

import argparse
import json
import os
import platform
import sys
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np

from contact_solver import Config, ContactSolver
from contact_solver.config import ProblemConfig, SolverConfig
from contact_solver.evaluation import continuous_acceleration_cost


def smoke_endpoints(d, case):
    """Small deterministic cases; the fleet extends the manuscript's seed 82."""
    if case in ("fleet", "fleet5"):
        n = 5 if case == "fleet5" else 3
        rng = np.random.default_rng(0 if n == 5 else 82)

        def draw():
            while True:
                points = rng.uniform(-4, 4, (n, 2))
                if all(
                    np.linalg.norm(points[p] - points[q]) > 1.5
                    for p, q in combinations(range(n), 2)
                ):
                    return points

        start, goal = draw(), draw()
        rng = np.random.default_rng(17)
        start = np.column_stack((start, rng.normal(scale=0.1, size=(n, d - 2))))
        goal = np.column_stack((goal, rng.normal(scale=0.1, size=(n, d - 2))))
        return start, goal
    start = np.zeros((2, d))
    if case == "free":
        start[1, 0] = 3
        return start, start + 1
    start[:, :2] = [[-2, -0.2], [2, 0.2]]
    if case == "head_on":
        start[:, 1] = 0
    goal = start.copy()
    goal[:, 0] *= -1
    return start, goal


def smoke_report():
    records = []
    for d in (2, 3, 5, 10):
        for case in ("free", "head_on", "offset", "fleet", "fleet5"):
            start, goal = smoke_endpoints(d, case)
            cfg = Config(
                name=f"smoke_{case}_{d}",
                problem=ProblemConfig(
                    n_robots=len(start), dimension=d, time_horizon=6, timestep=0.1, min_distance=1
                ),
                solver=SolverConfig(
                    sci_detection_tol=1e-6,
                    use_analytical_jacobian=True,
                ),
            )
            solver = ContactSolver(cfg)
            solver.set_initial_states(start)
            solver.set_final_states(goal)
            metrics = solver.generate_trajectories()["metrics"]
            record = dict(case=case, dimension=d, n_robots=len(start), vector=metrics)
            record["vector_acceleration_cost"] = continuous_acceleration_cost(solver)
            records.append(record)
    return dict(
        environment=dict(
            python=sys.version,
            platform=platform.platform(),
            packages={name: version(name) for name in ("numpy", "scipy", "osqp")},
            thread_limits={
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        ),
        records=records,
    )



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smoke", action="store_true", help="Only 20 small cases, N <= 5")
    source.add_argument(
        "--states",
        type=Path,
        help="NPZ with initial_positions/final_positions and optional velocities",
    )
    parser.add_argument("--config", type=Path, help="YAML Config, required with --states")
    parser.add_argument("--output", type=Path, help="JSON metrics path; otherwise print JSON")
    parser.add_argument(
        "--trajectories", type=Path, help="Optional sampled NPZ output for a single solve"
    )
    args = parser.parse_args(argv)
    if args.smoke:
        if args.config or args.trajectories:
            parser.error("--config/--trajectories apply only to --states")
        report = smoke_report()
        success = all(record["vector"]["converged"] for record in report["records"])
    else:
        if args.config is None:
            parser.error("--states requires --config")
        solver = ContactSolver(Config.from_yaml(args.config))
        with np.load(args.states, allow_pickle=False) as data:
            solver.set_initial_states(data["initial_positions"], data.get("initial_velocities"))
            solver.set_final_states(data["final_positions"], data.get("final_velocities"))
        result = solver.generate_trajectories()
        report = result["metrics"]
        success = report["converged"]
        if args.trajectories:
            args.trajectories.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.trajectories,
                times=result["times"],
                **result["trajectories"],
                contact_states=solver.solution.reshape(len(solver.contacts), solver.system.b),
                contact_pairs=np.array([c.pair for c in solver.contacts], dtype=int).reshape(-1, 2),
            )
    encoded = json.dumps(
        report,
        indent=2,
        allow_nan=False,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    else:
        print(encoded)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
