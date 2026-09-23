"""Run explicit-vector SCI on NPZ endpoint data, or a bounded 16-case smoke suite."""

import argparse
import json
import os
import platform
import sys
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np

from contact_solver import Config, ContactSolver, VectorContactSolver
from contact_solver.config import ProblemConfig, SolverConfig


def smoke_endpoints(d, case):
    """Small deterministic cases; the fleet extends the manuscript's seed 82."""
    if case == "fleet":
        rng = np.random.default_rng(82)

        def draw():
            while True:
                points = rng.uniform(-4, 4, (3, 2))
                if all(
                    np.linalg.norm(points[p] - points[q]) > 1.5
                    for p, q in combinations(range(3), 2)
                ):
                    return points

        start, goal = draw(), draw()
        rng = np.random.default_rng(17)
        start = np.column_stack((start, rng.normal(scale=0.1, size=(3, d - 2))))
        goal = np.column_stack((goal, rng.normal(scale=0.1, size=(3, d - 2))))
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
        for case in ("free", "head_on", "offset", "fleet"):
            start, goal = smoke_endpoints(d, case)
            cfg = Config(
                name=f"smoke_{case}_{d}",
                problem=ProblemConfig(
                    n_robots=len(start), dimension=d, time_horizon=6, timestep=0.1, min_distance=1
                ),
                solver=SolverConfig(
                    sci_detection_tol=1e-6,
                    use_analytical_jacobian=True,
                    apply_velocity_bounds=False,
                ),
            )
            solver = VectorContactSolver(cfg)
            solver.set_initial_states(start)
            solver.set_final_states(goal)
            metrics = solver.generate_trajectories()["metrics"]
            record = dict(case=case, dimension=d, n_robots=len(start), vector=metrics)
            record["vector_acceleration_cost"] = acceleration_cost(solver, solver.trajectory.times)
            if d == 2:
                planar = ContactSolver(cfg)
                planar.set_initial_states(start)
                planar.set_final_states(goal)
                record["legacy_planar"] = planar.generate_trajectories()["metrics"]
                knots = [0] + [segment[2] for segment in planar._segments[0]]
                record["legacy_acceleration_cost"] = acceleration_cost(planar, knots)
                # The planar metrics' minimum is sampled; compute a continuous one too.
                from contact_solver.solvers.cubic_utils import distance_candidates

                minima = []
                for s, (_, a, b) in enumerate(planar._segments[0]):
                    for p, q in combinations(range(len(start)), 2):
                        coefficients = np.array(planar._segments[q - 1][s][0])
                        if p:
                            coefficients -= np.array(planar._segments[p - 1][s][0])
                        minima.extend(
                            distance for _, distance in distance_candidates(coefficients, b - a)
                        )
                record["legacy_continuous_min_distance"] = min(minima)
                # Legacy reports infinity for absent obstacles; keep JSON standards compliant.
                del record["legacy_planar"]["min_obstacle_distance"]
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


def acceleration_cost(solver, knots):
    """Integrate the quadratic acceleration cost exactly by two-point Gauss quadrature."""
    cost = 0.0
    for a, b in zip(knots[:-1], knots[1:], strict=True):
        times = (a + b) / 2 + (b - a) / (2 * np.sqrt(3)) * np.array([-1, 1])
        cost += (b - a) / 2 * np.sum(np.asarray(solver.sample(times=times)[3]) ** 2)
    return float(cost)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smoke", action="store_true", help="Only 16 small cases, N <= 3")
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
        solver = VectorContactSolver(Config.from_yaml(args.config))
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
