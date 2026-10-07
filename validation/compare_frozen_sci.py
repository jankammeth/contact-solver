"""Bounded integration check against the frozen implementation used in the paper.

Run with the paper's virtual environment. No SCP solves or publication data writes.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CASES = {
    "fleet": [
        "e91cfb1e676d609567c4",
        "5ccd83c21e28a2d6f453",
        "91f6ac7426541cc65298",
        "d26efa9f6d9ba47ccd02",
        "2ca30d7e72a235a38e1d",
    ],
    "prioritized": ["N20-seed00", "N14-seed35", "N18-seed28", "N18-seed42", "N14-seed40"],
    "dimension": ["d3_n10_seed45", "d10_n10_seed20"],
    "ablations": ["ba28536ee06e0cb320b6", "07a182a5652db70cf7ef", "8159b0beb90b03e1b057"],
}


def worker(args):
    import numpy as np

    source = args.snapshot / "source" if args.worker == "frozen" else ROOT
    sys.path.insert(0, str(source / "code/contact-solver/src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    if args.worker == "frozen":
        sys.path.insert(0, str(source / "optimized"))
        from activate import install

        install()

    from ablation_metrics import measure, segments_for

    from contact_solver.config import Config
    from contact_solver.solvers.contact_solver import ContactSolver
    from contact_solver.solvers.prioritized_sci import PrioritizedSCI

    rows, samples = [], {}
    for study, cases in CASES.items():
        for case in cases:
            folder = args.snapshot / study / "attempts" / case
            old = json.loads((folder / "result.json").read_text())
            baseline = old["baseline"]
            group = baseline.get("group", "joint")
            config = Config.from_yaml(folder / "config.json")
            times = []
            for _ in range(args.repeats):
                solver = (
                    PrioritizedSCI(config, priority_order=old["priority_order"])
                    if group == "prioritized"
                    else ContactSolver(
                        config,
                        contact_selection=(
                            "earliest" if baseline.get("variant") == "earliest" else "deepest"
                        ),
                    )
                )
                with np.load(folder / "input.npz") as data:
                    solver.set_initial_states(data["p0"], data["v0"])
                    solver.set_final_states(data["pf"], data["vf"])
                solver.K = int(baseline["K"])
                result = solver.generate_trajectories()
                metrics = result["metrics"]
                times.append(metrics["timing"]["solve_time"])
            paths = segments_for(solver, result, group, "sci")
            measured = measure(paths, solver.R, solver.T)
            samples[case] = np.array(result["trajectories"]["positions"])
            residual = (
                max(r["residual"] for r in metrics["rank_checks"])
                if group == "prioritized"
                else metrics["checks"]["residual"]
            )
            row = dict(
                study=study,
                case=case,
                accepted=bool(metrics["converged"]),
                archived_accepted=old["accepted"],
                contacts=metrics["num_contacts"],
                reason=metrics["convergence_reason"],
                residual=residual,
                archived_cost=old["cost"],
                archived_clearance=old["r_min_full"],
                seconds=times,
                median_seconds=statistics.median(times),
                backend=metrics.get("collision_backend", "frozen_native"),
                **measured,
            )
            rows.append(row)
            print(
                args.worker,
                study,
                case,
                row["accepted"],
                round(row["median_seconds"], 5),
                flush=True,
            )
    (args.output / f"{args.worker}.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez_compressed(args.output / f"{args.worker}.npz", **samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=["frozen", "integrated"])
    args = parser.parse_args()
    args.snapshot, args.output = args.snapshot.resolve(), args.output.resolve()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    for study, cases in CASES.items():
        for case in cases:
            for name in ("input.npz", "config.json", "result.json"):
                if not (args.snapshot / study / "attempts" / case / name).is_file():
                    parser.error(f"Missing {study}/{case}/{name}")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        dict.fromkeys(
            (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ),
            "1",
        )
    )
    if args.worker:
        worker(args)
        return
    for mode in ("frozen", "integrated"):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--snapshot",
                str(args.snapshot),
                "--output",
                str(args.output),
                "--repeats",
                str(args.repeats),
                "--worker",
                mode,
            ],
            check=True,
            timeout=180,
        )
    import numpy as np

    old = json.loads((args.output / "frozen.json").read_text())
    new = json.loads((args.output / "integrated.json").read_text())
    comparisons = []
    with (
        np.load(args.output / "frozen.npz") as old_paths,
        np.load(args.output / "integrated.npz") as new_paths,
    ):
        for a, b in zip(old, new, strict=True):
            error = float(np.max(np.abs(old_paths[a["case"]] - new_paths[b["case"]])))
            passed = (
                a["case"] == b["case"]
                and a["accepted"] == b["accepted"] == b["archived_accepted"]
                and a["contacts"] == b["contacts"]
                and a["reason"] == b["reason"]
                and np.isclose(a["cost"], b["cost"], rtol=1e-10, atol=1e-10)
                and np.isclose(b["cost"], b["archived_cost"], rtol=1e-10, atol=1e-10)
                and abs(a["r_min_full"] - b["r_min_full"]) < 1e-9
                and np.isclose(a["residual"], b["residual"], rtol=1e-8, atol=1e-10)
                and error < 1e-9
                and b["backend"] == "native"
            )
            comparisons.append(
                dict(
                    study=a["study"],
                    case=a["case"],
                    passed=bool(passed),
                    accepted=b["accepted"],
                    maximum_position_difference=error,
                    runtime_ratio=b["median_seconds"] / a["median_seconds"],
                    frozen_seconds=a["median_seconds"],
                    integrated_seconds=b["median_seconds"],
                )
            )
    report = dict(
        cases=len(comparisons),
        passed=sum(r["passed"] for r in comparisons),
        repeats=args.repeats,
        median_runtime_ratio=statistics.median(r["runtime_ratio"] for r in comparisons),
        maximum_position_difference=max(r["maximum_position_difference"] for r in comparisons),
        comparisons=comparisons,
    )
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))
    if report["passed"] != report["cases"]:
        raise SystemExit(
            "Integration differs from the measured implementation; inspect comparison.json"
        )


if __name__ == "__main__":
    main()
