import importlib.util
import json
import sys

import numpy as np

from contact_solver import Config, ContactSolver
from contact_solver.cli.vector_sci import smoke_endpoints
from contact_solver.config import ProblemConfig, SolverConfig

name = "contact_solver.solvers.vector_before"
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
spec.loader.exec_module(mod)
report = json.loads(open(sys.argv[2]).read()) if len(sys.argv) > 3 else {}
for case in (sys.argv[3:] or ("free", "head_on", "offset", "fleet", "fleet5")):
    a, b = smoke_endpoints(2, case)
    cfg = Config(
        name="timing",
        problem=ProblemConfig(
            n_robots=len(a), dimension=2, time_horizon=6, timestep=0.1, min_distance=1
        ),
        solver=SolverConfig(
            sci_detection_tol=1e-6, use_analytical_jacobian=True
        ),
    )
    solvers = {}
    for label, cls in [
        ("before", getattr(mod, "ContactSolver", None) or mod.VectorContactSolver),
        ("after", ContactSolver),
    ]:
        s = cls(cfg)
        s.set_initial_states(a)
        s.set_final_states(b)
        solvers[label] = s
    for _ in range(10):
        for s in solvers.values():
            s.generate_trajectories()
    vals = {k: [] for k in solvers}
    for i in range(150):
        labels = list(solvers)
        for label in labels[i % len(labels) :] + labels[: i % len(labels)]:
            result = solvers[label].generate_trajectories()["metrics"]
            assert result["converged"], (case, label)
            vals[label].append(result["timing"]["solve_time"] * 1000)
    report[case] = {
        k: {
            "median_ms": float(np.median(v)),
            "p10_ms": float(np.percentile(v, 10)),
            "p90_ms": float(np.percentile(v, 90)),
        }
        for k, v in vals.items()
    }
    print(case, json.dumps(report[case]), flush=True)
open(sys.argv[2], "w").write(json.dumps(report, indent=2) + "\n")
