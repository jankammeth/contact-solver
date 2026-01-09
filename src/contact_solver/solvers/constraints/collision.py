"""Linearized collision constraints for trajectory optimization."""

import numpy as np
import scipy.sparse as sp

from .dynamics import Constraint


def normalize_constraint(constraint: Constraint) -> Constraint:
    """Normalize each constraint row to have unit coefficient norm."""
    if constraint.matrix.shape[0] == 0:
        return constraint

    matrix_csr = constraint.matrix.tocsr()
    row_norms = np.sqrt(np.array(matrix_csr.power(2).sum(axis=1)).flatten())
    row_norms = np.maximum(row_norms, 1e-10)

    scale_factors = 1.0 / row_norms
    D_inv = sp.diags(scale_factors, format="csr")

    normalized_matrix = (D_inv @ matrix_csr).tocsc()
    normalized_lower = constraint.lower * scale_factors
    normalized_upper = constraint.upper * scale_factors

    # Handle infinities
    normalized_lower = np.where(np.isinf(constraint.lower), constraint.lower, normalized_lower)
    normalized_upper = np.where(np.isinf(constraint.upper), constraint.upper, normalized_upper)

    return Constraint(normalized_matrix, normalized_lower, normalized_upper)


def build_collision_constraints(
    prev_positions: list[np.ndarray],
    robot_timeframes: list[list[int]],
    decision_var_offsets: np.ndarray,
    h: float,
    R: float,
    safety_margin: float,
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    total_decision_vars: int,
) -> Constraint:
    """Build linearized collision avoidance constraints for double-integrator.

    Args:
        prev_positions: List of (K_i, 2) position arrays from previous iteration
        robot_timeframes: List of [k_start, k_end] for each robot
        decision_var_offsets: Cumulative offsets into decision variable vector
        h: Time step
        R: Minimum distance
        safety_margin: Additional margin for constraints
        initial_positions: Flat array of initial positions (2*N,)
        initial_velocities: Flat array of initial velocities (2*N,)
        total_decision_vars: Total number of decision variables

    Returns:
        Combined collision constraint
    """
    N = len(robot_timeframes)
    K_per_robot = [t[1] - t[0] for t in robot_timeframes]
    start_times = [t[0] for t in robot_timeframes]

    init_pos = initial_positions.reshape(N, 2)
    init_vel = initial_velocities.reshape(N, 2)

    rows, cols, vals, rhs_list = [], [], [], []
    row_idx = 0

    for i in range(N):
        for j in range(i + 1, N):
            start_i, end_i = start_times[i], start_times[i] + K_per_robot[i]
            start_j, end_j = start_times[j], start_times[j] + K_per_robot[j]

            overlap_start = max(start_i, start_j)
            overlap_end = min(end_i, end_j)

            for k_global in range(overlap_start, overlap_end):
                k_i = k_global - start_times[i]
                k_j = k_global - start_times[j]

                pi_prev, pj_prev = prev_positions[i][k_i], prev_positions[j][k_j]
                eta, dist = _linearization_direction(pi_prev, pj_prev)

                # Robot i coefficients (positive)
                _add_integration_coeffs(
                    rows, cols, vals, row_idx, k_i, h, decision_var_offsets[i], eta, 1.0
                )
                # Robot j coefficients (negative)
                _add_integration_coeffs(
                    rows, cols, vals, row_idx, k_j, h, decision_var_offsets[j], eta, -1.0
                )

                # RHS for double integrator
                init_pos_contrib = eta @ (init_pos[i] - init_pos[j])
                init_vel_contrib = eta @ (
                    init_vel[i] * ((k_i + 1) * h) - init_vel[j] * ((k_j + 1) * h)
                )
                linearization_term = eta @ (pi_prev - pj_prev) - dist
                rhs_list.append(
                    (R + safety_margin) + linearization_term - (init_pos_contrib + init_vel_contrib)
                )
                row_idx += 1

    constraint = _coo_to_constraint(rows, cols, vals, rhs_list, total_decision_vars)
    return normalize_constraint(constraint)


def _linearization_direction(pos_a: np.ndarray, pos_b: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute unit vector from b to a, or random direction if coincident."""
    diff = pos_a - pos_b
    dist = np.hypot(diff[0], diff[1])

    if dist < 1e-6:
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        eta = np.array([np.cos(angle), np.sin(angle)])
    else:
        eta = diff / dist

    return eta, dist


def _add_integration_coeffs(
    rows: list,
    cols: list,
    vals: list,
    row_idx: int,
    k: int,
    h: float,
    base_offset: int,
    eta: np.ndarray,
    sign: float,
) -> None:
    """Add sparse coefficients for position integration (double-integrator).

    p[k] = p0 + (k+1)*h*v0 + h^2 * sum_{m=0}^{k} (k+1-m-0.5) * a[m]
    """
    if k < 0:
        return

    m_indices = np.arange(k + 1)
    weights = (h * h) * (k + 1 - m_indices - 0.5)

    rows.extend([row_idx] * (k + 1))
    cols.extend(base_offset + 2 * m_indices)
    vals.extend(sign * eta[0] * weights)

    rows.extend([row_idx] * (k + 1))
    cols.extend(base_offset + 2 * m_indices + 1)
    vals.extend(sign * eta[1] * weights)


def _coo_to_constraint(rows, cols, vals, rhs_list, total_decision_vars) -> Constraint:
    """Assemble sparse constraint from builders."""
    n = len(rhs_list)
    if n == 0:
        return Constraint(sp.csc_matrix((0, total_decision_vars)), np.array([]), np.array([]))

    matrix = sp.coo_matrix((vals, (rows, cols)), shape=(n, total_decision_vars)).tocsc()
    return Constraint(matrix, np.array(rhs_list), np.full(n, np.inf))
