"""Bounded diagnostic: recover seed 2 by removing a negative-mass contact.

This is an experimental active-set repair, not a change to the production solver.
Run from the parent repository with PYTHONPATH=code/contact-solver/src.
"""

import json
import time
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import root

from contact_solver import Config, ContactSolver
from contact_solver.evaluation import continuous_acceleration_cost
from contact_solver.solvers.cubic_utils import cubic_coefficients
from contact_solver.solvers.contact_solver import RelativeTrajectory
from contact_solver.solvers.vector_matching import blocks_to_sparse

folder = Path("runs/numerics/fleet-restart/attempts/7b23a0d3339500da9391")
solver = ContactSolver(Config.from_yaml(folder / "config.json"))
with np.load(folder / "input.npz") as data:
    solver.set_initial_states(data["p0"], data["v0"])
    solver.set_final_states(data["pf"], data["vf"])
original = solver._solve_system
starts = []


def capture(system, x, deadline):
    starts.append((system, x.copy()))
    return original(system, x, deadline)


solver._solve_system = capture
baseline = solver.generate_trajectories()["metrics"]
system, initial = starts[-1]


def residual(x):
    try:
        return system.evaluate(x)
    except ValueError:
        return np.full_like(x, 1e10)


# Independently recover the root of the stalled four-contact system.
solution = root(
    residual,
    initial,
    jac=lambda x: blocks_to_sparse(system.evaluate(x, True)[1]).toarray(),
    method="hybr",
    options={"maxfev": 1000},
)
assert np.max(np.abs(residual(solution.x))) < solver.options.sci_residual_tol
contacts = system.contacts(solution.x)
masses = solution.x.reshape(system.m, system.b)[:, 1 + 2 * solver.d]
removed = [
    list(c.pair)
    for c, mass in zip(contacts, masses, strict=True)
    if mass < -solver.options.sci_mass_tol
]
contacts = [
    c for c, mass in zip(contacts, masses, strict=True) if mass >= -solver.options.sci_mass_tol
]
reduced = solver.matching_system(contacts)
x = np.concatenate([c.to_vector() for c in contacts])
x, info = original(reduced, x, time.perf_counter() + 30)
assert info["success"]
arrays = np.array(
    (
        solver.initial_positions,
        solver.initial_velocities,
        solver.final_positions,
        solver.final_velocities,
    )
).reshape(4, solver.N, solver.d)
barycenter = np.array(cubic_coefficients(*arrays.mean(axis=1), solver.T))
trajectory = RelativeTrajectory.from_system(reduced, x, barycenter)
checks = solver._checks(reduced, x, trajectory)
closest = trajectory.closest()
assert checks["valid"] and closest[2] >= solver.R - solver.options.sci_detection_tol
solver.trajectory = trajectory
cost = continuous_acceleration_cost(solver)
# Separate polynomial implementation checks every pair and segment.

minimum = np.inf
for k, h in enumerate(np.diff(trajectory.times)):
    for p in range(solver.N):
        for q in range(p + 1, solver.N):
            c = trajectory.coefficients[k, q - 1].copy()
            if p:
                c -= trajectory.coefficients[k, p - 1]
            c *= np.array([1, h, h**2, h**3])[:, None]
            squared = sum(
                (Polynomial(c[:, axis]) ** 2 for axis in range(solver.d)), Polynomial([0])
            )
            times = [0.0, 1.0] + [
                float(z.real)
                for z in squared.deriv().roots()
                if abs(z.imag) < 1e-7 and 0 <= z.real <= 1
            ]
            minimum = min(
                minimum, *(np.linalg.norm(sum(c[i] * t**i for i in range(4))) for t in times)
            )
assert minimum >= solver.R - solver.options.sci_detection_tol
report = dict(
    seed=2,
    production_status=baseline["status"],
    four_contact_masses=masses.tolist(),
    removed_pairs_zero_based=removed,
    repaired_contacts=len(contacts),
    root=info,
    checks=checks,
    continuous_minimum=float(minimum),
    cost=cost,
    historical_cost=1.5534280699302707,
)
output = Path("code/contact-solver/validation/fleet20-seed2-repair.json")
output.write_text(json.dumps(report, indent=2) + "\n")
np.savez(
    output.with_suffix(".npz"),
    times=trajectory.times,
    coefficients=trajectory.coefficients,
    barycenter=trajectory.barycenter,
    contact_states=x,
    pairs=reduced.pairs,
)
print(json.dumps(report, indent=2))
