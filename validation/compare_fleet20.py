"""Paired pilot on saved fleet-study inputs; run from the parent repository."""

import argparse
import importlib.util
import sys
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

from contact_solver import Config, ContactSolver
from contact_solver.evaluation import continuous_acceleration_cost


def continuous_minimum(solver, vector):
    if vector:
        times = solver.trajectory.times
        coefficients = solver.trajectory.coefficients
    else:
        times = np.array([0] + [s[2] for s in solver._segments[0]])
        coefficients = np.array(
            [
                [solver._segments[j][k][0] for j in range(solver.N - 1)]
                for k in range(len(times) - 1)
            ]
        )
    best = np.inf
    for k, h in enumerate(np.diff(times)):
        for p in range(solver.N):
            for q in range(p + 1, solver.N):
                c = coefficients[k, q - 1].copy()
                if p:
                    c -= coefficients[k, p - 1]
                c = c * np.array([1, h, h * h, h * h * h])[:, None]
                squared = sum((Polynomial(c[:, axis]) ** 2 for axis in range(2)), Polynomial([0]))
                candidates = [0.0, 1.0]
                for root in squared.deriv().roots():
                    if abs(root.imag) < 1e-7 and 0 <= root.real <= 1:
                        candidates.append(float(root.real))
                # Evaluate the vector, avoiding cancellation in squared polynomial values.
                best = min(
                    best, *(np.linalg.norm(sum(c[i] * u**i for i in range(4))) for u in candidates)
                )
    return float(best), continuous_acceleration_cost(solver, knots=times)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--legacy-module", required=True,
                    help="Path to a saved historical planar solver module")
args = parser.parse_args()
name = "contact_solver.solvers.saved_planar_baseline"
spec = importlib.util.spec_from_file_location(name, args.legacy_module)
legacy_module = importlib.util.module_from_spec(spec)
sys.modules[name] = legacy_module
spec.loader.exec_module(legacy_module)


rows = {
    int(r["seed"]): r
    for r in csv.DictReader(open("data/numerics/fleet/tables/attempts.csv"))
    if r["N"] == "20" and r["method"] == "sci"
}
report = {
    "description": "Saved fleet N=20, seeds 0-4, one warmup then five alternating timed runs; single-thread limits; solve time excludes sampled postprocessing.",
    "records": [],
}
output = Path("code/contact-solver/validation/fleet20-comparison.json")
for seed in range(5):
    row = rows[seed]
    path = Path(row["instance_file"])
    cfg = Config.from_yaml(path.with_name("config.json"))
    with np.load(path) as data:
        arrays = {key: data[key].copy() for key in ("p0", "v0", "pf", "vf")}
    solvers = {}
    for name, cls in [("legacy", legacy_module.ContactSolver), ("vector", ContactSolver)]:
        s = cls(cfg)
        s.set_initial_states(arrays["p0"], arrays["v0"])
        s.set_final_states(arrays["pf"], arrays["vf"])
        solvers[name] = s
        s.generate_trajectories()
    timings = {name: [] for name in solvers}
    metrics = {}
    for repeat in range(5):
        for name in ("legacy", "vector") if repeat % 2 == 0 else ("vector", "legacy"):
            result = solvers[name].generate_trajectories()["metrics"]
            timings[name].append(result["timing"]["solve_time"])
            metrics[name] = result
    record = {
        "seed": seed,
        "instance_id": row["instance_id"],
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config": cfg.to_dict(),
        "historical_runtime": float(row["reported_runtime"]),
    }
    for name, s in solvers.items():
        minimum, cost = continuous_minimum(s, name == "vector")
        record[name] = {
            "times_s": timings[name],
            "median_s": float(np.median(timings[name])),
            "success": bool(metrics[name]["converged"]),
            "reason": metrics[name]["convergence_reason"],
            "contacts": metrics[name]["num_contacts"],
            "continuous_minimum": minimum,
            "cost": cost,
            "matching_seconds": metrics[name].get("matching_seconds"),
            "checks": metrics[name].get("checks"),
        }
    samples = np.linspace(0, cfg.problem.time_horizon, 1001)
    record["max_position_difference"] = float(
        np.max(
            np.linalg.norm(
                np.asarray(solvers["legacy"].sample(times=samples)[1])
                - np.asarray(solvers["vector"].sample(times=samples)[1]),
                axis=-1,
            )
        )
    )
    record["speedup"] = (
        record["legacy"]["median_s"] / record["vector"]["median_s"]
        if record["legacy"]["success"] and record["vector"]["success"]
        else None
    )
    report["records"].append(record)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in record.items() if k not in ("config", "input_sha256")}),
        flush=True,
    )
