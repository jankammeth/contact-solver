"""Process deadlines must stop work and preserve completed solver state."""

from types import SimpleNamespace
import os
import time

import numpy as np
import pytest

from contact_solver import ContactSolver, LiftedSCP, PrioritizedSCI, PrioritizedSCP
from contact_solver.cli._sweep_utils import run_solver_with_timeout
from contact_solver.config import Config, ProblemConfig, SolverConfig
from contact_solver.evaluation import continuous_acceleration_cost, sample_scp_grid


class SleepingSolver:
    def __init__(self, marker):
        self.marker = marker
        self.config = SimpleNamespace(solver=SimpleNamespace(timeout=300))

    def generate_trajectories(self):
        self.marker.write_text(str(os.getpid()))
        time.sleep(30)
        pytest.fail("Timed-out worker continued")


class FailingSolver(SleepingSolver):
    def generate_trajectories(self):
        raise ValueError("deliberate worker failure")


def test_timeout_kills_and_reaps_worker(tmp_path):
    marker = tmp_path / "pid"
    result, timed_out = run_solver_with_timeout(SleepingSolver(marker), 0.2)
    assert result is None and timed_out
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


def test_worker_error_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="deliberate worker failure"):
        run_solver_with_timeout(FailingSolver(tmp_path / "unused"), 5)


@pytest.mark.parametrize("cls", [ContactSolver, LiftedSCP, PrioritizedSCI, PrioritizedSCP])
def test_completed_state_survives_process_for_evaluation(cls):
    cfg = Config(
        name="process_smoke",
        problem=ProblemConfig(n_robots=2, time_horizon=5, timestep=0.5, min_distance=0.5),
        solver=SolverConfig(sci_detection_tol=1e-6, scp_detection_tol=1e-3,
                            scp_convergence_abs=1e-4, scp_convergence_rel=1e-4,
                            osqp_eps_abs=1e-7, osqp_eps_rel=1e-7),
    )
    solver = cls(cfg)
    solver.set_initial_states([[0., 0.], [0., 3.]])
    solver.set_final_states([[5., 0.], [5., 3.]])
    result, timed_out = run_solver_with_timeout(solver, 5)
    assert not timed_out and result["metrics"]["converged"]
    if cls in (ContactSolver, PrioritizedSCI):
        assert continuous_acceleration_cost(solver) == pytest.approx(4.8)
        assert np.isfinite(sample_scp_grid(solver)["positions"]).all()
    elif cls is LiftedSCP:
        assert np.isfinite(solver.trajectories["positions"]).all()
    else:
        assert np.isfinite(result["trajectories"]["positions"]).all()
