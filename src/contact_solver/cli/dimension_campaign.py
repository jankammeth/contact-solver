"""Reproducible Section 5.5 inputs, isolated attempts, and paired summaries.

Default execution is a bounded pilot, never the full 600-instance campaign.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from contact_solver import Config, LiftedSCP, ContactSolver
from contact_solver.cli.compare_joint import json_values
from contact_solver.evaluation import (
    continuous_acceleration_cost,
    discrete_acceleration_cost,
    sample_scp_grid,
)
from contact_solver.solvers.cubic_utils import cubic_coefficients, distance_candidates

THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_values(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoints(d, n, seed, spacing=0.8, radius_scale=1.0):
    rng = np.random.default_rng(np.random.SeedSequence([5501, d, n, seed]))
    radius = radius_scale * 5 * (n / 2) ** (1 / d)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Sampling radius must be finite and positive")

    def draw():
        points = []
        for _ in range(100000):
            direction = rng.normal(size=d)
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            point = radius * rng.random() ** (1 / d) * direction / norm
            if all(np.linalg.norm(point - other) >= spacing for other in points):
                points.append(point)
                if len(points) == n:
                    return np.array(points)
        raise RuntimeError("Endpoint rejection limit exceeded")

    return dict(
        initial_positions=draw(),
        final_positions=draw(),
        initial_velocities=np.zeros((n, d)),
        final_velocities=np.zeros((n, d)),
    )


def interacting_pairs(states, horizon=10.0, radius=0.5):
    # Zero endpoint velocities give every robot the same monotone cubic progress.
    # Minimize each relative segment exactly, without polynomial root finding.
    if all(np.all(states[k] == 0) for k in ("initial_velocities", "final_velocities")):
        initial, final = states["initial_positions"], states["final_positions"]
        p, q = np.triu_indices(len(initial), 1)
        a = initial[p] - initial[q]
        b = final[p] - final[q] - a
        bb = np.sum(b * b, axis=1)
        progress = np.divide(-np.sum(a * b, axis=1), bb, out=np.zeros_like(bb), where=bb > 0)
        closest = a + np.clip(progress, 0, 1)[:, None] * b
        return int(np.count_nonzero(np.linalg.norm(closest, axis=1) < radius))
    coeffs = np.array(
        cubic_coefficients(
            states["initial_positions"],
            states["initial_velocities"],
            states["final_positions"],
            states["final_velocities"],
            horizon,
        )
    )
    n = coeffs.shape[1]
    return sum(
        min(v for _, v in distance_candidates(coeffs[:, p] - coeffs[:, q], horizon)) < radius
        for p in range(n)
        for q in range(p + 1, n)
    )


def campaign_config(d, n, seed, timeout, radius_scale=1.0):
    base = Path(__file__).resolve().parents[3] / "configs/sweep_n.yaml"
    cfg = Config.from_yaml(base)
    return cfg.override(
        **{
            "random_seed": seed,
            "problem.dimension": d,
            "problem.n_robots": n,
            "problem.time_horizon": 10.0,
            "problem.timestep": 0.05,
            "problem.scenario.cluster_radius": radius_scale * 5 * (n / 2) ** (1 / d),
            "solver.timeout": timeout,
        }
    )


def interaction_statistics(counts, n):
    counts = np.asarray(counts, dtype=float)
    mean = float(np.mean(counts))
    return dict(
        samples=len(counts),
        mean_pairs=mean,
        standard_error=float(np.std(counts, ddof=1) / np.sqrt(len(counts)))
        if len(counts) > 1
        else None,
        pair_density=mean / (n * (n - 1) / 2),
        collision_free_fraction=float(np.mean(counts == 0)),
        histogram={str(int(k)): int(np.sum(counts == k)) for k in np.unique(counts)},
    )


def calibrate_interactions(design):
    """Match each cell to a planar reference using only disjoint calibration seeds."""
    settings = design["interaction_calibration"]
    seeds = range(settings["seed_start"], settings["seed_start"] + settings["samples"])
    if set(seeds).intersection(design["seeds"]):
        raise ValueError("Calibration and evaluation seeds must be disjoint")
    cells = {}
    for n in design["fleet_sizes"]:

        def evaluate(d, scale, n=n):
            counts = [
                interacting_pairs(
                    endpoints(d, n, s, spacing=design["spacing"], radius_scale=scale),
                    radius=design["R"],
                )
                for s in seeds
            ]
            return dict(radius_scale=scale, **interaction_statistics(counts, n))

        reference = evaluate(2, 1.0)
        target = reference["mean_pairs"]
        if target == 0:
            raise ValueError(
                f"No planar calibration interactions for N={n}; increase calibration samples"
            )
        for d in design["dimensions"]:
            trials = [reference if d == 2 else evaluate(d, 1.0)]
            if d != 2:
                # Bracket by factors of sqrt(2), then refine in log radius.
                # Endpoint rejection can break strict samplewise monotonicity;
                # retain the closest observed candidate, not just the last one.
                scale = 1.0
                shrink = trials[0]["mean_pairs"] < target
                for _ in range(12):
                    scale *= 2 ** (-0.5 if shrink else 0.5)
                    try:
                        trial = evaluate(d, scale)
                    except RuntimeError as error:
                        raise ValueError(
                            f"Cannot calibrate d={d}, N={n} at radius scale {scale:g}: {error}"
                        ) from error
                    trials.append(trial)
                    if (trial["mean_pairs"] >= target) == shrink:
                        break
                else:
                    raise ValueError(f"Could not bracket interaction target for d={d}, N={n}")
                lo, hi = sorted([trials[-2]["radius_scale"], trials[-1]["radius_scale"]])
                for _ in range(6):
                    scale = float(np.sqrt(lo * hi))
                    trial = evaluate(d, scale)
                    trials.append(trial)
                    if trial["mean_pairs"] >= target:
                        lo = scale
                    else:
                        hi = scale
            best = min(trials, key=lambda t: abs(t["mean_pairs"] - target))
            cells[f"d{d}_n{n}"] = dict(
                dimension=d,
                n_robots=n,
                radius=best["radius_scale"] * 5 * (n / 2) ** (1 / d),
                radius_scale=best["radius_scale"],
                reference=reference,
                achieved=best,
                trials=trials,
            )
            print(
                f"Calibration d={d}, N={n}: radius={cells[f'd{d}_n{n}']['radius']:.4g}, "
                f"mean pairs={best['mean_pairs']:.3f}, target={target:.3f}",
                flush=True,
            )
    return cells


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix in (".py", ".c", ".so", ".pyd", ".dylib"))
    hashes = {str(p.relative_to(root)): file_hash(p) for p in paths}
    hashes["configs/sweep_n.yaml"] = file_hash(root.parents[1] / "configs/sweep_n.yaml")
    for name in ("setup.py", "pyproject.toml"):
        hashes[name] = file_hash(root.parents[1] / name)
    return dict(
        sources=hashes,
        packages={p: version(p) for p in ("numpy", "scipy", "osqp", "omegaconf")},
        python=sys.version,
        platform=platform.platform(),
        thread_limits={k: "1" for k in THREADS},
    )


def prepare(root, design):
    root = Path(root)
    signature = dict(design=design, environment=source_manifest())
    manifest = root / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if {k: saved.get(k) for k in signature} != signature:
            raise ValueError(
                "Existing manifest differs: use a new output directory for changed code/settings"
            )
        calibration = saved.get("interaction_calibration", {})
        if bool(calibration) != ("interaction_calibration" in design):
            raise ValueError("Existing manifest has missing or unexpected calibration")
        if calibration and saved.get("calibration_sha256") != calibration_hash(calibration):
            raise ValueError("Saved calibration changed")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Refusing to adopt a nonempty directory without a manifest")
        calibration = calibrate_interactions(design) if "interaction_calibration" in design else {}
        if calibration:
            signature.update(
                interaction_calibration=calibration,
                calibration_sha256=calibration_hash(calibration),
            )
        atomic_json(manifest, signature)
    cases = []
    for d in design["dimensions"]:
        for n in design["fleet_sizes"]:
            scale = calibration[f"d{d}_n{n}"]["radius_scale"] if calibration else 1.0
            for seed in design["seeds"]:
                case = root / f"d{d}_n{n}_seed{seed}"
                case.mkdir(exist_ok=True)
                path = case / "input.npz"
                metadata = case / "instance.json"
                cfgpath = case / "config.json"
                if metadata.exists():
                    info = json.loads(metadata.read_text())
                    if (
                        file_hash(path) != info["input_sha256"]
                        or file_hash(cfgpath) != info["config_sha256"]
                    ):
                        raise ValueError(f"Saved input/config changed: {case}")
                else:
                    states = endpoints(d, n, seed, spacing=design["spacing"], radius_scale=scale)
                    np.savez(path, **states)
                    atomic_json(
                        cfgpath, campaign_config(d, n, seed, design["timeout"], scale).to_dict()
                    )
                    info = dict(
                        dimension=d,
                        n_robots=n,
                        seed=seed,
                        radius=scale * 5 * (n / 2) ** (1 / d),
                        initial_interacting_pairs=interacting_pairs(states, radius=design["R"]),
                        input_sha256=file_hash(path),
                        config_sha256=file_hash(cfgpath),
                    )
                    atomic_json(metadata, info)
                cases.append(case)
    if calibration:
        validation = []
        for key, cell in calibration.items():
            counts = [
                json.loads((case / "instance.json").read_text())["initial_interacting_pairs"]
                for case in cases
                if case.name.startswith(key + "_seed")
            ]
            stats = interaction_statistics(counts, cell["n_robots"])
            validation.append(
                dict(
                    dimension=cell["dimension"],
                    n_robots=cell["n_robots"],
                    radius=cell["radius"],
                    target_mean_pairs=cell["reference"]["mean_pairs"],
                    target_standard_error=cell["reference"]["standard_error"],
                    mean_pair_difference=stats["mean_pairs"] - cell["reference"]["mean_pairs"],
                    **stats,
                )
            )
        atomic_json(root / "interaction_validation.json", validation)
        for row in validation:
            print(
                f"Held-out d={row['dimension']}, N={row['n_robots']}: "
                f"mean pairs={row['mean_pairs']:.3f}, target={row['target_mean_pairs']:.3f}, "
                f"free fraction={row['collision_free_fraction']:.0%}",
                flush=True,
            )
    return cases


def calibration_hash(calibration):
    return hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    case = Path(request["case"])
    cfg = Config.from_yaml(case / "config.json")
    method = request["method"]
    memory_limit = "unsupported on this platform"
    if sys.platform.startswith("linux"):
        cap = int(request["memory_gib"] * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        memory_limit = f"RLIMIT_AS {cap} bytes"
    started = time.perf_counter()
    record = dict(method=method, memory_limit=memory_limit)
    try:
        solver = (ContactSolver if method == "sci" else LiftedSCP)(cfg)
        with np.load(case / "input.npz") as states:
            solver.set_initial_states(states["initial_positions"], states["initial_velocities"])
            solver.set_final_states(states["final_positions"], states["final_velocities"])
        result = solver.generate_trajectories()
        metrics = result["metrics"]
        feasible = bool(metrics["converged"])
        numerical = bool(
            metrics.get("numerical_converged", metrics.get("checks", {}).get("valid", False))
        )
        distance = metrics.get("min_distance")
        clearance = bool(
            distance is not None
            and np.isfinite(distance)
            and distance >= cfg.problem.min_distance - cfg.solver.sci_detection_tol
        )
        evaluated = sample_scp_grid(solver, cfg.problem.n_timesteps) if method == "sci" else result["trajectories"]
        cost = discrete_acceleration_cost(evaluated, solver.h)
        if method == "sci":
            record["continuous_cost"] = continuous_acceleration_cost(solver)
        record["cost_grid"] = dict(K=cfg.problem.n_timesteps, weight=solver.h, rule="right endpoints")
        recoveries = [
            r for r in metrics.get("roots", []) if r.get("message") == "reaction_recovery"
        ]
        record.update(
            status=metrics.get("status", "success" if feasible else "unresolved"),
            numerical_converged=numerical,
            continuous_feasible=clearance,
            accepted=feasible,
            solve_seconds=metrics["timing"]["solve_time"],
            cost=cost,
            min_distance=distance,
            eta=(
                max(0.0, cfg.problem.min_distance - distance) / cfg.problem.min_distance
                if distance is not None and np.isfinite(distance)
                else None
            ),
            contacts=metrics.get("num_contacts"),
            full_system_size=metrics.get("system_size"),
            intrinsic_degrees_of_freedom=metrics.get("intrinsic_degrees_of_freedom"),
            recovery_calls=len(recoveries),
            recovery_system_sizes=[r["system_size"] for r in recoveries],
            metrics={
                key: metrics[key]
                for key in (
                    "convergence_reason",
                    "stopping_reason",
                    "checks",
                    "tolerances",
                    "timing",
                    "roots",
                    "removed_contacts",
                    "active_set_updates",
                    "scp_iterations",
                    "grid_min_distance",
                    "qp_error",
                )
                if key in metrics
            },
        )
    except MemoryError:
        record.update(status="memory_limit", accepted=False, error="MemoryError")
    except Exception as error:
        record.update(status="error", accepted=False, error=repr(error))
    record["call_wall_seconds"] = time.perf_counter() - started
    record["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == "darwin" else 1024
    )
    atomic_json(request["result"], record)


def execute(case, method, repeat, hard_timeout, memory_gib):
    attempt = case / f"{method}_repeat{repeat}"
    attempt.mkdir(exist_ok=True)
    output = attempt / "result.json"
    if output.exists():
        return json.loads(output.read_text()), True
    request = attempt / "request.json"
    atomic_json(
        request,
        dict(
            case=str(case.resolve()),
            method=method,
            result=str(output.resolve()),
            memory_gib=memory_gib,
        ),
    )
    env = dict(os.environ, **{k: "1" for k in THREADS})
    started = time.perf_counter()
    with (attempt / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "contact_solver.cli.dimension_campaign",
                "--worker",
                str(request.resolve()),
            ],
            stdout=log,
            stderr=log,
            env=env,
        )
        try:
            process.wait(timeout=hard_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            atomic_json(
                output,
                dict(
                    method=method,
                    status="hard_timeout",
                    accepted=False,
                    hard_timeout_seconds=hard_timeout,
                ),
            )
    if not output.exists():
        atomic_json(
            output,
            dict(
                method=method, status="worker_exit", accepted=False, returncode=process.returncode
            ),
        )
    record = json.loads(output.read_text())
    record["process_wall_seconds"] = time.perf_counter() - started
    record["returncode"] = process.returncode
    atomic_json(output, record)
    return record, False


def summarize(root, cases, repeats):
    rows = []
    paired = []
    for case in cases:
        info = json.loads((case / "instance.json").read_text())
        for repeat in range(repeats):
            results = {}
            for method in ("sci", "scp"):
                p = case / f"{method}_repeat{repeat}" / "result.json"
                if not p.exists():
                    continue
                record = json.loads(p.read_text())
                results[method] = record
                row = {**info, "repeat": repeat, "method": method}
                row.update(
                    {
                        k: record.get(k)
                        for k in (
                            "status",
                            "accepted",
                            "numerical_converged",
                            "continuous_feasible",
                            "solve_seconds",
                            "process_wall_seconds",
                            "peak_rss_mib",
                            "cost",
                            "eta",
                            "contacts",
                            "full_system_size",
                            "intrinsic_degrees_of_freedom",
                            "recovery_calls",
                            "recovery_system_sizes",
                        )
                    }
                )
                rows.append(row)
            if len(results) == 2:
                valid = all(r["accepted"] for r in results.values())
                paired.append(
                    dict(
                        dimension=info["dimension"],
                        n_robots=info["n_robots"],
                        seed=info["seed"],
                        repeat=repeat,
                        initial_interacting_pairs=info["initial_interacting_pairs"],
                        both_accepted=valid,
                        speedup=(
                            results["scp"]["solve_seconds"] / results["sci"]["solve_seconds"]
                            if valid
                            else None
                        ),
                        cost_ratio=(
                            results["scp"]["cost"] / results["sci"]["cost"]
                            if valid and results["sci"]["cost"] > 0
                            else None
                        ),
                    )
                )
    groups = []
    for d, n, method in sorted({(r["dimension"], r["n_robots"], r["method"]) for r in rows}):
        subset = [r for r in rows if (r["dimension"], r["n_robots"], r["method"]) == (d, n, method)]
        accepted = [r for r in subset if r["accepted"]]
        groups.append(
            dict(
                dimension=d,
                n_robots=n,
                method=method,
                attempts=len(subset),
                accepted=len(accepted),
                numerical_converged=sum(bool(r["numerical_converged"]) for r in subset),
                median_success_seconds=float(np.median([r["solve_seconds"] for r in accepted]))
                if accepted
                else None,
                statuses={
                    s: sum(r["status"] == s for r in subset)
                    for s in sorted({r["status"] for r in subset})
                },
            )
        )
    atomic_json(root / "summary.json", dict(groups=groups, paired=paired))
    if rows:
        with (root / "attempts.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full", action="store_true", help="Explicitly request all 50 seeds per cell"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 3, 5, 10])
    parser.add_argument("--fleet-sizes", type=int, nargs="+", default=[5, 10])
    parser.add_argument(
        "--run-fleet-sizes",
        type=int,
        nargs="+",
        help="Execute only these fleets from the frozen design; other fleets can run later",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--memory-gib", type=float, default=4.0)
    parser.add_argument(
        "--match-interactions",
        action="store_true",
        help="Calibrate radii to the planar mean interaction count for each fleet",
    )
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--calibration-seed-start", type=int, default=10000)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not args.output:
        parser.error("--output is required")
    if args.full and args.seeds:
        parser.error("--full and --seeds are mutually exclusive")
    seeds = list(range(50)) if args.full else (args.seeds or [0])
    timeout = args.timeout if args.timeout is not None else (300.0 if args.full else 15.0)
    if (
        min(args.dimensions) < 2
        or min(args.fleet_sizes) < 2
        or min(seeds) < 0
        or args.repeats < 1
        or not np.isfinite(timeout)
        or timeout <= 0
        or args.memory_gib <= 0
        or not np.isfinite(args.memory_gib)
        or args.calibration_samples < 2
        or args.calibration_seed_start < 0
    ):
        parser.error("Invalid dimensions, fleet sizes, seeds, repeats, timeout or memory limit")
    if any(
        len(values) != len(set(values)) for values in (args.dimensions, args.fleet_sizes, seeds)
    ):
        parser.error("Duplicate cells/seeds are not allowed")
    if args.run_fleet_sizes is not None and (
        len(args.run_fleet_sizes) != len(set(args.run_fleet_sizes))
        or not set(args.run_fleet_sizes).issubset(args.fleet_sizes)
    ):
        parser.error("--run-fleet-sizes must be a unique subset of --fleet-sizes")
    design = dict(
        dimensions=args.dimensions,
        fleet_sizes=args.fleet_sizes,
        seeds=seeds,
        repeats=args.repeats,
        timeout=timeout,
        hard_timeout=timeout + 10,
        memory_gib=args.memory_gib,
        horizon=10.0,
        K=200,
        R=0.5,
        spacing=0.8,
        epsilon=0.001,
        mode="full" if args.full else "pilot",
    )
    if args.match_interactions:
        calibration_seeds = range(
            args.calibration_seed_start, args.calibration_seed_start + args.calibration_samples
        )
        if set(seeds).intersection(calibration_seeds):
            parser.error("Calibration and evaluation seeds must be disjoint")
        design["interaction_calibration"] = dict(
            samples=args.calibration_samples,
            seed_start=args.calibration_seed_start,
            method="planar-mean-radius-v1",
        )
    cases = prepare(args.output, design)
    print(
        f"Prepared {len(cases)} paired instances; {len(cases) * 2 * args.repeats} solver attempts",
        flush=True,
    )
    if args.prepare_only:
        return
    selected = [
        case
        for case in cases
        if args.run_fleet_sizes is None
        or json.loads((case / "instance.json").read_text())["n_robots"] in args.run_fleet_sizes
    ]
    print(
        f"Executing {len(selected)} paired instances; summaries retain all completed fleets",
        flush=True,
    )
    for case in selected:
        for repeat in range(args.repeats):
            for method in ("sci", "scp") if repeat % 2 == 0 else ("scp", "sci"):
                record, cached = execute(
                    case, method, repeat, design["hard_timeout"], args.memory_gib
                )
                print(
                    f"{case.name} {method} repeat={repeat}: {record['status']}"
                    + (" [cached]" if cached else ""),
                    flush=True,
                )
                summarize(args.output, cases, args.repeats)
    print(json.dumps(summarize(args.output, cases, args.repeats), indent=2), flush=True)


if __name__ == "__main__":
    main()
