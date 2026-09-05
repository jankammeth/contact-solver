"""Shared helpers for sweep experiments (sweep_n, sweep_K, ...).

This module factors out logic common to the multi-robot sweep CLIs so that
adding new sweeps does not require copy-pasting the scenario builder, the
braid-word machinery, or the trajectory metric computations.
"""

from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import numpy as np
import pandas as pd

from contact_solver.config import Config


# ──────────────────────────────────────────────────────────────────────────────
# Scenario scaling
# ──────────────────────────────────────────────────────────────────────────────

def compute_scale_factor(N: int, base_N: int = 2) -> float:
    """Compute scaling factor for N robots relative to base_N."""
    return math.sqrt(N / base_N)


def build_scaled_config(base_config: Config, N: int, seed: int, K: int = 100) -> Config:
    """Build a config with workspace and time scaled by sqrt(N/2).

    Scaling rule (cf. Section IV of the paper):
        time_horizon   = base_time          * sqrt(N/2)
        cluster_radius = base_cluster_radius * sqrt(N/2)
        timestep       = time_horizon / K
    """
    scale = compute_scale_factor(N, base_N=2)

    base_time = 10.0
    base_cluster_radius = 5.0

    time_horizon = base_time * scale
    cluster_radius = base_cluster_radius * scale
    timestep = time_horizon / K

    return base_config.override(**{
        "random_seed": seed,
        "problem.n_robots": N,
        "problem.time_horizon": time_horizon,
        "problem.timestep": timestep,
        "problem.scenario.cluster_radius": cluster_radius,
        "visualization.show_plots": False,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Solver execution with timeout
# ──────────────────────────────────────────────────────────────────────────────

def run_solver_with_timeout(solver, timeout: float):
    """Run solver.generate_trajectories() with a soft timeout.

    Returns (result_dict, timed_out_bool).  On timeout, the worker thread is
    not killed; partial state cannot be recovered without invasive changes.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(solver.generate_trajectories)
        try:
            return future.result(timeout=timeout), False
        except FuturesTimeoutError:
            return None, True


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(trajectories: dict, config: Config) -> dict:
    """Compute trajectory metrics (min pairwise distance + discrete cost).

    Cost is a left Riemann sum h * sum ||a[k]||^2, consistent with the
    piecewise-constant acceleration model used in the SCP QP.
    """
    positions = trajectories["positions"]
    accelerations = trajectories["accelerations"]

    N = len(positions)
    K = len(positions[0])
    h = config.problem.timestep

    # Min pairwise distance
    min_dist = np.inf
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i][k] - positions[j][k])
                min_dist = min(min_dist, dist)

    # Acceleration cost
    cost = 0.0
    for a in accelerations:
        cost += h * np.sum(a**2)

    return {
        "min_distance": min_dist,
        "acceleration_cost": cost,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Continuous-time min distance for the SCP discretization
# ──────────────────────────────────────────────────────────────────────────────
#
# LiftedSCP's dynamics constraints (cf. lifted_scp._build_dynamics_constraints)
# enforce piecewise-constant acceleration on each interval [t_{k-1}, t_k], with
# acceleration value a[k] (the right-endpoint variable).  The unique continuous-
# time interpretation consistent with this discretization is:
#
#     p(t_{k-1} + tau) = p[k-1] + tau * v[k-1] + 0.5 * tau^2 * a[k],
#     tau in [0, h]
#
# This is what a low-level controller integrating the SCP solution would track.
# The inter-sample minimum pairwise distance is therefore a lower bound on the
# actual continuous-time safety margin --- SCP only enforces the constraint at
# the grid points.

def _pair_min_dist_on_interval(
    p_i: np.ndarray, v_i: np.ndarray, a_i: np.ndarray,
    p_j: np.ndarray, v_j: np.ndarray, a_j: np.ndarray,
    h: float,
) -> float:
    """Minimum ||p_i(tau) - p_j(tau)|| on tau in [0, h] under the SCP integrator.

    Each component of (p_i - p_j) is quadratic in tau:
        d(tau) = (p_i - p_j) + tau * (v_i - v_j) + 0.5 * tau^2 * (a_i - a_j)
    so ||d(tau)||^2 is a quartic in tau.  Its derivative is a cubic, whose roots
    on [0, h] are the candidate stationary points; the minimum is over those
    and the two endpoints.
    """
    # Component coefficients: d(tau) = c0 + c1*tau + c2*tau^2
    c0 = p_i - p_j
    c1 = v_i - v_j
    c2 = 0.5 * (a_i - a_j)

    # ||d(tau)||^2 = sum_d (c0d + c1d*tau + c2d*tau^2)^2
    # Expand to polynomial in tau (degree 4):
    #   q0 = c0.c0
    #   q1 = 2 c0.c1
    #   q2 = c1.c1 + 2 c0.c2
    #   q3 = 2 c1.c2
    #   q4 = c2.c2
    q0 = float(np.dot(c0, c0))
    q1 = 2.0 * float(np.dot(c0, c1))
    q2 = float(np.dot(c1, c1)) + 2.0 * float(np.dot(c0, c2))
    q3 = 2.0 * float(np.dot(c1, c2))
    q4 = float(np.dot(c2, c2))

    # Derivative: q'(tau) = q1 + 2 q2 tau + 3 q3 tau^2 + 4 q4 tau^3
    coeffs = [4.0 * q4, 3.0 * q3, 2.0 * q2, q1]  # numpy convention: highest first

    candidates = [0.0, h]
    # numpy.roots requires the leading coefficient to be non-zero; otherwise
    # the derivative is of lower degree and we drop terms accordingly.
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-15:
        coeffs = coeffs[1:]
    if len(coeffs) > 1:
        try:
            roots = np.roots(coeffs)
            for r in roots:
                if abs(r.imag) < 1e-9:
                    rr = float(r.real)
                    if 0.0 < rr < h:
                        candidates.append(rr)
        except np.linalg.LinAlgError:
            pass  # fall back to endpoints

    best_sq = float("inf")
    for tau in candidates:
        val = q0 + q1 * tau + q2 * tau * tau + q3 * tau ** 3 + q4 * tau ** 4
        if val < best_sq:
            best_sq = val

    # Numerical safety: clip tiny negatives from floating-point noise
    return math.sqrt(max(best_sq, 0.0))


def compute_scp_min_dist_continuous(
    positions: list[np.ndarray],
    velocities: list[np.ndarray],
    accelerations: list[np.ndarray],
    h: float,
) -> float:
    """Continuous-time minimum pairwise distance of an SCP solution.

    Reconstructs the trajectory between grid points using SCP's own integrator
    (piecewise-constant acceleration, with a[k] applied on [t_{k-1}, t_k]) and
    finds the minimum pairwise distance over the continuous time interval.

    Args:
        positions: list of N arrays of shape (K, 2).
        velocities: list of N arrays of shape (K, 2).
        accelerations: list of N arrays of shape (K, 2).
        h: timestep (seconds).

    Returns:
        Minimum pairwise distance over t in [0, T].
    """
    N = len(positions)
    K = len(positions[0])

    if N < 2 or K < 2:
        return float("inf")

    min_dist = float("inf")
    # Each interval [k-1, k] uses position/velocity at k-1 as the initial
    # condition and acceleration at k for the ballistic step.
    for k in range(1, K):
        for i in range(N):
            for j in range(i + 1, N):
                d = _pair_min_dist_on_interval(
                    positions[i][k - 1], velocities[i][k - 1], accelerations[i][k],
                    positions[j][k - 1], velocities[j][k - 1], accelerations[j][k],
                    h,
                )
                if d < min_dist:
                    min_dist = d
    return min_dist


# ──────────────────────────────────────────────────────────────────────────────
# Braid word extraction and comparison (homotopy class)
# ──────────────────────────────────────────────────────────────────────────────

def compute_braid_word(
    positions: list[np.ndarray],
    axis: int = 0,
    min_sep: float = 0.0,
) -> list[tuple[int, int, int]]:
    """Extract the braid word from a multi-robot trajectory.

    Projects robot positions onto a reference axis and records crossing events
    (when two robots swap their order along that axis).

    Args:
        positions: List of N arrays of shape (K, 2), one per robot.
        axis: Projection axis (0=x, 1=y).
        min_sep: Minimum perpendicular separation to register a crossing.
            Crossings where robots are closer than this in the perpendicular
            direction are ignored (likely numerical noise).

    Returns:
        List of (i, j, sign) tuples, where i < j are robot indices and
        sign is +1 or -1 indicating the crossing direction.
    """
    N = len(positions)
    K = len(positions[0])

    proj = np.array([positions[i][:, axis] for i in range(N)])  # (N, K)

    crossings = []
    for k in range(1, K):
        for i in range(N):
            for j in range(i + 1, N):
                prev_diff = proj[i, k - 1] - proj[j, k - 1]
                curr_diff = proj[i, k] - proj[j, k]

                if prev_diff * curr_diff < 0:
                    perp = 1 - axis
                    mid_perp_i = 0.5 * (positions[i][k - 1, perp] + positions[i][k, perp])
                    mid_perp_j = 0.5 * (positions[j][k - 1, perp] + positions[j][k, perp])

                    perp_sep = abs(mid_perp_i - mid_perp_j)
                    if perp_sep < min_sep:
                        continue

                    sign = 1 if mid_perp_i > mid_perp_j else -1
                    crossings.append((i, j, sign))

    return crossings


def normalize_braid_word(
    crossings: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Normalize a braid word by applying far-commutativity."""
    word = list(crossings)
    changed = True
    while changed:
        changed = False
        for idx in range(len(word) - 1):
            (i1, j1, _s1) = word[idx]
            (i2, j2, _s2) = word[idx + 1]
            pair1 = {i1, j1}
            pair2 = {i2, j2}
            if pair1.isdisjoint(pair2):
                if (i1, j1) > (i2, j2):
                    word[idx], word[idx + 1] = word[idx + 1], word[idx]
                    changed = True
    return word


def compare_braid_words(
    crossings_a: list[tuple[int, int, int]],
    crossings_b: list[tuple[int, int, int]],
) -> dict:
    """Compare two braid words via multiset comparison of (pair, sign) tuples.

    Robust to temporal reordering of nearly-simultaneous crossings while
    correctly detecting genuine homotopy-class differences (sign flips).
    """
    multiset_a = Counter(crossings_a)
    multiset_b = Counter(crossings_b)
    same_homotopy = multiset_a == multiset_b

    return {
        "same_homotopy_class": same_homotopy,
        "n_crossings_scp": len(crossings_a),
        "n_crossings_contact": len(crossings_b),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Seed eligibility filtering (used by sweep_K)
# ──────────────────────────────────────────────────────────────────────────────

def load_eligible_seeds(
    sweep_n_csv: Path | str,
    N: int,
    require_homotopy_agreement: bool = True,
) -> list[int]:
    """Return seeds where both solvers succeeded at this N in a prior sweep_n run.

    Args:
        sweep_n_csv: Path to sweep_n_results.csv.
        N: Fleet size to filter on.
        require_homotopy_agreement: If True, additionally require that the two
            solvers agreed on homotopy class at K=100.

    Returns:
        Sorted list of seeds passing the filter.
    """
    csv_path = Path(sweep_n_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"sweep_n CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Coerce booleans (CSVs may store them as strings)
    for col in ("scp_success", "contact_success", "same_homotopy_class"):
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: str(v).strip().lower() == "true"
                if not isinstance(v, bool) else v
            )

    mask = (df["N"] == N) & (df["scp_success"] == True) & (df["contact_success"] == True)
    if require_homotopy_agreement:
        if "same_homotopy_class" not in df.columns:
            raise ValueError(
                f"'same_homotopy_class' column missing from {csv_path}; "
                "cannot filter on homotopy agreement."
            )
        mask &= (df["same_homotopy_class"] == True)

    seeds = sorted(df.loc[mask, "seed"].astype(int).unique().tolist())
    return seeds
