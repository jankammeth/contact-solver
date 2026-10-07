import json
import subprocess
import sys

import numpy as np
import pytest

from contact_solver.cli.dimension_campaign import (
    calibrate_interactions,
    campaign_config,
    endpoints,
    execute,
    interacting_pairs,
    prepare,
    summarize,
)
from contact_solver.solvers.cubic_utils import cubic_coefficients, distance_candidates


def design():
    return dict(
        dimensions=[3],
        fleet_sizes=[2],
        seeds=[0],
        repeats=1,
        timeout=3.0,
        hard_timeout=13.0,
        memory_gib=4.0,
        horizon=10.0,
        K=200,
        R=0.5,
        spacing=0.8,
        epsilon=0.001,
        mode="pilot",
    )


def test_default_dimension_campaign_uses_only_plotted_fleets(monkeypatch, tmp_path):
    from contact_solver.cli import dimension_campaign as campaign

    def inspect_design(root, selected):
        assert selected["fleet_sizes"] == [5, 10]
        raise RuntimeError("design inspected")

    monkeypatch.setattr(campaign, "prepare", inspect_design)
    monkeypatch.setattr(sys, "argv", ["dimension_campaign", "--prepare-only", "--output", str(tmp_path)])
    with pytest.raises(RuntimeError, match="design inspected"):
        campaign.main()


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_uniform_ball_reproducibility_and_spacing(d):
    states = endpoints(d, 32, 7)
    radius = 5 * 16 ** (1 / d)
    repeated = endpoints(d, 32, 7)
    for name in states:
        np.testing.assert_array_equal(states[name], repeated[name])
    for name in ("initial_positions", "final_positions"):
        p = states[name]
        assert p.shape == (32, d)
        assert np.max(np.linalg.norm(p, axis=1)) <= radius
        distances = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
        assert min(distances[np.triu_indices(32, 1)]) >= 0.8
    np.testing.assert_array_equal(states["initial_velocities"], 0)
    many = endpoints(d, 256, 8, spacing=0)["initial_positions"]
    scaled = np.linalg.norm(many, axis=1) / (5 * 128 ** (1 / d))
    assert abs(np.mean(scaled**d) - 0.5) < 0.08


def test_paper_settings():
    cfg = campaign_config(10, 32, 0, 15)
    assert cfg.problem.n_timesteps == 200
    assert cfg.problem.time_horizon == 10
    assert cfg.solver.sci_detection_tol == cfg.solver.scp_detection_tol == 0.001
    assert cfg.solver.osqp_eps_abs == 1e-4
    assert cfg.solver.scp_max_iterations == 20


def test_manifest_resume_and_mutation_rejection(tmp_path):
    d = design()
    cases = prepare(tmp_path, d)
    assert prepare(tmp_path, d) == cases
    with pytest.raises(ValueError, match="manifest differs"):
        prepare(tmp_path, {**d, "timeout": 4.0})
    (cases[0] / "input.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input/config changed"):
        prepare(tmp_path, d)


def test_worker_and_cached_resume(tmp_path):
    cases = prepare(tmp_path, design())
    result, cached = execute(cases[0], "sci", 0, 13.0, 4.0)
    assert not cached
    assert result["accepted"]
    assert result["numerical_converged"] and result["continuous_feasible"]
    assert result["peak_rss_mib"] > 0
    again, cached = execute(cases[0], "sci", 0, 13.0, 4.0)
    assert cached and again == result
    groups = summarize(tmp_path, cases, 1)
    assert groups[0]["accepted"] == 1


def test_hard_timeout_is_saved_as_failure(tmp_path, monkeypatch):
    cases = prepare(tmp_path, design())

    class Process:
        returncode = -9
        killed = False

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("test worker", timeout)

        def kill(self):
            self.killed = True

    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: process)
    result, _ = execute(cases[0], "scp", 0, 0.01, 4.0)
    assert process.killed and not result["accepted"]
    assert result["status"] == "hard_timeout"
    summary = summarize(tmp_path, cases, 1)
    assert summary[0]["median_success_seconds"] is None
    assert json.loads((tmp_path / "summary.json").read_text())["paired"] == []


@pytest.mark.parametrize("d", [2, 3, 5, 10])
def test_fast_interaction_count_matches_cubic_checker(d):
    for seed in range(5):
        states = endpoints(d, 5, seed, radius_scale=0.25)
        coeffs = np.array(
            cubic_coefficients(
                states["initial_positions"],
                states["initial_velocities"],
                states["final_positions"],
                states["final_velocities"],
                10.0,
            )
        )
        expected = sum(
            min(v for _, v in distance_candidates(coeffs[:, p] - coeffs[:, q], 10.0)) < 0.5
            for p in range(5)
            for q in range(p + 1, 5)
        )
        assert interacting_pairs(states) == expected


def test_stationary_pairs_and_strict_threshold():
    positions = np.array([[0.0, 0.0], [0.5, 0.0], [0.75, 0.0]])
    states = dict(
        initial_positions=positions,
        final_positions=positions,
        initial_velocities=np.zeros_like(positions),
        final_velocities=np.zeros_like(positions),
    )
    assert interacting_pairs(states) == 1


def test_calibration_freezes_radii_and_validates_held_out_inputs(tmp_path, monkeypatch):
    import contact_solver.cli.dimension_campaign as campaign

    d = {
        **design(),
        "dimensions": [2, 3],
        "fleet_sizes": [5],
        "seeds": [0, 1, 2],
        "interaction_calibration": dict(
            samples=32, seed_start=10000, method="planar-mean-radius-v1"
        ),
    }
    cases = prepare(tmp_path, d)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    cells = manifest["interaction_calibration"]
    assert cells["d2_n5"]["radius_scale"] == 1
    cell = cells["d3_n5"]
    assert 0 < cell["radius_scale"] < 1
    assert abs(cell["achieved"]["mean_pairs"] - cell["reference"]["mean_pairs"]) <= 1 / 32
    for case in cases:
        info = json.loads((case / "instance.json").read_text())
        cfg = json.loads((case / "config.json").read_text())
        radius = cells[f"d{info['dimension']}_n5"]["radius"]
        assert info["radius"] == cfg["problem"]["scenario"]["cluster_radius"] == radius
        with np.load(case / "input.npz") as states:
            p = states["initial_positions"]
            assert np.max(np.linalg.norm(p, axis=1)) <= radius
            assert (
                np.min(np.linalg.norm(p[:, None] - p[None, :], axis=-1)[np.triu_indices(5, 1)])
                >= 0.8
            )
            assert interacting_pairs(states) == info["initial_interacting_pairs"]
    validation = json.loads((tmp_path / "interaction_validation.json").read_text())
    assert all(row["samples"] == 3 for row in validation)
    assert all(sum(row["histogram"].values()) == 3 for row in validation)

    def no_recalibration(*args):
        pytest.fail("Resume must use frozen radii")

    monkeypatch.setattr(campaign, "calibrate_interactions", no_recalibration)
    assert prepare(tmp_path, d) == cases
    cell["radius_scale"] *= 0.9
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="calibration changed"):
        prepare(tmp_path, d)


def test_calibration_rejects_seed_overlap():
    d = {**design(), "interaction_calibration": dict(samples=16, seed_start=0)}
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_interactions(d)


def test_staged_execution_augments_same_manifest_and_summary(tmp_path, monkeypatch):
    import contact_solver.cli.dimension_campaign as campaign

    calls = []

    def fake_execute(case, method, repeat, hard_timeout, memory_gib):
        calls.append((case.name, method))
        result = dict(
            method=method,
            accepted=True,
            status="success",
            numerical_converged=True,
            solve_seconds=1.0,
            cost=1.0,
        )
        campaign.atomic_json(case / f"{method}_repeat{repeat}" / "result.json", result)
        return result, False

    monkeypatch.setattr(campaign, "execute", fake_execute)
    base = [
        "dimension_campaign",
        "--output",
        str(tmp_path),
        "--dimensions",
        "3",
        "--fleet-sizes",
        "5",
        "10",
        "20",
        "--seeds",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", base + ["--run-fleet-sizes", "5", "10"])
    campaign.main()
    manifest = (tmp_path / "manifest.json").read_bytes()
    first_results = {p: p.read_bytes() for p in tmp_path.glob("d*/*/result.json")}
    assert len(first_results) == 4
    assert not list((tmp_path / "d3_n20_seed0").glob("*/result.json"))
    calls.clear()
    monkeypatch.setattr(sys, "argv", base + ["--run-fleet-sizes", "20"])
    campaign.main()
    assert calls == [("d3_n20_seed0", "sci"), ("d3_n20_seed0", "scp")]
    assert (tmp_path / "manifest.json").read_bytes() == manifest
    assert all(p.read_bytes() == contents for p, contents in first_results.items())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["paired"]) == 3
    assert {g["n_robots"] for g in summary["groups"]} == {5, 10, 20}
    monkeypatch.setattr(sys, "argv", base + ["--run-fleet-sizes", "30"])
    with pytest.raises(SystemExit):
        campaign.main()
