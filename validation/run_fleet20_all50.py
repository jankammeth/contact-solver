import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from contact_solver import Config, ContactSolver
from contact_solver.evaluation import continuous_acceleration_cost

parser = argparse.ArgumentParser()
parser.add_argument("--baseline-module")
parser.add_argument(
    "--output", default="code/contact-solver/validation/fleet20-all50-active-set.json"
)
args = parser.parse_args()
solver_class = ContactSolver
if args.baseline_module:
    name = "contact_solver.solvers.vector_baseline"
    spec = importlib.util.spec_from_file_location(name, args.baseline_module)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    solver_class = getattr(module, "ContactSolver", None) or module.VectorContactSolver
rows = {
    int(r["seed"]): r
    for r in csv.DictReader(open("data/numerics/fleet/tables/attempts.csv"))
    if r["N"] == "20" and r["method"] == "sci"
}
out = Path(args.output)
records = []
for seed in range(50):
    row = rows[seed]
    p = Path(row["instance_file"])
    s = solver_class(Config.from_yaml(p.with_name("config.json")))
    with np.load(p) as a:
        s.set_initial_states(a["p0"], a["v0"])
        s.set_final_states(a["pf"], a["vf"])
    try:
        r = s.generate_trajectories()["metrics"]
        cost = continuous_acceleration_cost(s)
        rec = dict(
            seed=seed,
            success=r["converged"],
            reason=r["convergence_reason"],
            seconds=r["timing"]["solve_time"],
            contacts=r["num_contacts"],
            removed=r.get("removed_contacts", []),
            checks=r["checks"],
            min_distance=r["min_distance"],
            cost=cost,
            old_cost=float(row["cost"]),
            old_seconds=float(row["reported_runtime"]),
        )
    except Exception as e:
        rec = dict(seed=seed, success=False, error=repr(e))
    records.append(rec)
    out.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(rec), flush=True)
