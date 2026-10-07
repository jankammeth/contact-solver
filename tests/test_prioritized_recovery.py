"""Recovery must retain the computed trajectory and the original acceptance bounds."""

from pathlib import Path

import numpy as np
import pytest

from contact_solver.config import Config, ProblemConfig, SolverConfig
from contact_solver.solvers import prioritized_sci as module
from contact_solver.solvers.prioritized_sci import PrioritizedSCI


def solver():
    horizon = 10 * np.sqrt(14 / 2)
    cfg = Config(
        name="prioritized14-seed35",
        problem=ProblemConfig(
            n_robots=14, time_horizon=horizon, timestep=horizon / 200, min_distance=0.5
        ),
        solver=SolverConfig(sci_detection_tol=1e-3),
    )
    result = PrioritizedSCI(cfg, priority_order=[2, 12, 5, 1, 6, 0, 11, 3, 4, 7, 8, 9, 10, 13])
    with np.load(Path(__file__).parent / "data/prioritized14_seed35.npz") as data:
        result.set_initial_states(data["p0"], data["v0"])
        result.set_final_states(data["pf"], data["vf"])
    return result


def test_saved_stagnation_case_recovers_and_reuse_is_safe():
    instance = solver()
    first = instance.generate_trajectories()
    metrics = first["metrics"]
    assert metrics["converged"]
    assert any(r["message"] == "reaction_recovery" for r in metrics["roots"])
    assert max(r["residual"] for r in metrics["rank_checks"]) <= 1e-7
    assert min(r["minimum_mass"] for r in metrics["rank_checks"]) >= -1e-9
    assert metrics["min_distance"] >= 0.499
    for i, path in instance._trajectories.items():
        np.testing.assert_allclose(
            path.eval_pos(0), instance.initial_positions.reshape(-1, 2)[i], atol=1e-7
        )
        np.testing.assert_allclose(
            path.eval_pos(instance.T), instance.final_positions.reshape(-1, 2)[i], atol=1e-7
        )
    second = instance.generate_trajectories()
    assert second["metrics"]["converged"]
    np.testing.assert_allclose(
        first["trajectories"]["positions"], second["trajectories"]["positions"], atol=1e-12
    )


@pytest.mark.parametrize(
    "enabled,iterations,message,expected",
    [
        (True, 100, "stagnation", True),
        (False, 100, "stagnation", False),
        (True, 0, "iteration_limit", False),
        (True, 100, "timeout", False),
    ],
)
def test_recovery_obeys_option_iteration_limit_and_timeout(
    monkeypatch, enabled, iterations, message, expected
):
    instance = solver()
    instance.options.sci_reaction_recovery = enabled
    instance.options.sci_root_max_iterations = iterations
    calls = []

    def failed(system, x, deadline, options):
        return x, dict(success=False, message=message, nfev=0)

    def recovery(self, *args):
        calls.append(1)
        return args[-3:]

    monkeypatch.setattr(module, "damped_newton", failed)
    monkeypatch.setattr(PrioritizedSCI, "_recover_reactions", recovery)
    metrics = instance.generate_trajectories()["metrics"]
    assert not metrics["converged"]
    assert bool(calls) == expected
