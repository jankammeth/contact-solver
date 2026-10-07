"""Admissible damped Newton shared by joint and prioritized matching."""

import time

import numpy as np
from scipy.linalg import LinAlgError, solve_banded
from scipy.sparse.linalg import lsmr

from .vector_matching import blocks_to_banded, blocks_to_sparse


def insertion_time(candidate, existing, horizon):
    """Offset only an exactly coincident new INITIAL guess, never a solved time."""
    existing = np.asarray(existing, dtype=float)
    if not np.any(existing == candidate):
        return candidate
    knots = np.sort(np.r_[0.0, existing, horizon])
    # Prefer the right side; stay between neighboring existing knots.
    for direction in (1, -1):
        neighbors = knots[knots > candidate] if direction == 1 else knots[knots < candidate]
        if not len(neighbors):
            continue
        gap = (neighbors.min() - candidate) if direction == 1 else (candidate - neighbors.max())
        offset = min(1e-6 * horizon, gap / 4)
        trial = candidate + direction * offset
        if 0 < trial < horizon and trial != candidate and not np.any(existing == trial):
            return float(trial)
    raise ValueError("No representable interior initial guess for coincident contacts")


def finite_difference_blocks(system, x):
    """Central differences respecting the time domain, for the Jacobian ablation."""
    residual = system.evaluate(x)
    calls = 1
    blocks = np.zeros((system.m, 3, system.b, system.b))
    gaps = np.diff(system.knot_times(x))
    for j in range(len(x)):
        col_block, column = divmod(j, system.b)
        h = np.cbrt(np.finfo(float).eps) * max(1.0, abs(x[j]))
        if column == 0:
            h = min(h, 0.25 * (min(gaps[col_block : col_block + 2]) - system.min_gap))
        if h <= 0 or x[j] + h == x[j] or x[j] - h == x[j]:
            raise ValueError("No admissible finite-difference step")
        delta = np.zeros_like(x)
        delta[j] = h
        derivative = ((system.evaluate(x + delta) - system.evaluate(x - delta)) / (2 * h)).reshape(
            system.m, system.b
        )
        calls += 2
        for row in range(max(0, col_block - 1), min(system.m, col_block + 2)):
            blocks[row, col_block - row + 1, :, column] = derivative[row]
    return residual, blocks, calls


def damped_newton(system, x, deadline, options):
    """Feasible damped Newton, with no post-solve change to contact times."""
    root_tolerance = options.sci_residual_tol
    nfev, fallbacks = 0, 0
    message = "iteration_limit"
    previous_norm = np.inf
    stagnant_iterations = 0
    for iteration in range(options.sci_root_max_iterations + 1):
        try:
            if options.use_analytical_jacobian and getattr(system, "analytical_jacobian", True):
                residual, blocks = system.evaluate(x, True)
                nfev += 1
            else:
                residual, blocks, calls = finite_difference_blocks(system, x)
                nfev += calls
        except (ValueError, FloatingPointError):
            message = "invalid_matching_evaluation"
            break
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(blocks)):
            message = "nonfinite_matching_evaluation"
            break
        if np.max(np.abs(residual)) <= root_tolerance:
            return x, dict(
                success=True,
                message="residual_satisfied",
                iterations=iteration,
                nfev=nfev,
                sparse_fallbacks=fallbacks,
            )
        if time.perf_counter() >= deadline:
            message = "timeout"
            break
        current_norm = np.linalg.norm(residual)
        stagnant_iterations = stagnant_iterations + 1 if current_norm >= 0.99 * previous_norm else 0
        previous_norm = current_norm
        if stagnant_iterations >= 12:
            message = "stagnation"
            break
        if iteration == options.sci_root_max_iterations:
            break
        bandwidth, band = blocks_to_banded(blocks)
        steps = []
        try:
            newton = solve_banded((bandwidth, bandwidth), band, -residual)
            if np.all(np.isfinite(newton)):
                steps.append(newton)
        except (LinAlgError, ValueError):
            pass
        # Lazy fallback: sparse LSMR also handles rotational null directions.
        accepted = False
        jacobian = None
        for attempt in range(5):
            if attempt < len(steps):
                step = steps[attempt]
            else:
                if jacobian is None:
                    jacobian = blocks_to_sparse(blocks)
                column_norm = np.sqrt(np.asarray(jacobian.power(2).sum(axis=0)).ravel())
                scale = 1 / np.maximum(column_norm, 1e-12)
                scaled = jacobian.multiply(scale).tocsr()
                damping = (0.0, 1e-5, 1e-3, 1e-1, 1.0)[attempt]
                step = (
                    scale
                    * lsmr(
                        scaled,
                        -residual,
                        damp=damping,
                        atol=1e-12,
                        btol=1e-12,
                        maxiter=max(100, 4 * len(x)),
                    )[0]
                )
                fallbacks += 1
            time_step = np.concatenate(([0], step.reshape(system.m, system.b)[:, 0], [0]))
            gaps = np.diff(system.knot_times(x))
            gap_steps = np.diff(time_step)
            decreasing = gap_steps < 0
            length = 1.0
            if np.any(decreasing):
                length = min(
                    length,
                    0.99 * np.min((gaps[decreasing] - system.min_gap) / -gap_steps[decreasing]),
                )
            for _ in range(30):
                # Tiny Newton steps can stagnate despite strict descent.
                # Try the regularized direction before exhausting iterations.
                if attempt < len(steps) and length < 1e-3:
                    break
                if length <= 1e-14 or time.perf_counter() >= deadline:
                    break
                candidate = x + length * step
                try:
                    next_residual = system.evaluate(candidate)
                    nfev += 1
                except ValueError:
                    length *= 0.5
                    continue
                if np.all(np.isfinite(next_residual)) and np.linalg.norm(next_residual) < (
                    1 - 1e-4 * length
                ) * np.linalg.norm(residual):
                    x = candidate
                    accepted = True
                    break
                length *= 0.5
            if accepted:
                break
        if not accepted:
            message = "timeout" if time.perf_counter() >= deadline else "line_search_failed"
            break
    return x, dict(
        success=False,
        message=message,
        iterations=iteration,
        nfev=nfev,
        sparse_fallbacks=fallbacks,
    )
