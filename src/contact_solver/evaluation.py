"""Trajectory evaluation outside solver timers.

Paired costs use the SCP right-endpoint grid and its weight h for both methods.
Continuous SCI costs are reported separately, including the explicitly labelled
integrated-cost gap alongside the same-grid gap in the discretization study.
"""

import numpy as np


def sample_scp_grid(solver, K=None, h=None):
    K = solver.K if K is None else K
    h = solver.h if h is None else h
    times = np.arange(1, K + 1) * h
    _, positions, velocities, accelerations = solver.sample(times=times)
    return dict(positions=positions, velocities=velocities, accelerations=accelerations)


def discrete_acceleration_cost(trajectories, h):
    return float(h * np.sum(np.asarray(trajectories["accelerations"]) ** 2))


def continuous_acceleration_cost(solver, *, knots=None):
    """Exact cubic-path integral; explicit knots support saved baseline adapters."""
    if knots is None:
        if hasattr(solver, "trajectory"):
            knots = solver.trajectory.times
        else:
            knots = sorted(
                {
                    t
                    for path in solver._trajectories.values()
                    for _, a, b in path.segments
                    for t in (a, b)
                }
            )
    cost = 0.0
    for a, b in zip(knots[:-1], knots[1:], strict=True):
        times = (a + b) / 2 + (b - a) / (2 * np.sqrt(3)) * np.array([-1.0, 1.0])
        cost += (b - a) / 2 * np.sum(np.asarray(solver.sample(times=times)[3]) ** 2)
    return float(cost)
