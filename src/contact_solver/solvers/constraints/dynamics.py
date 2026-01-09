"""Constraint matrix builders for double-integrator trajectory optimization."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Constraint:
    """Constraint specification: l <= matrix @ x <= u"""

    matrix: sp.csc_matrix
    lower: np.ndarray
    upper: np.ndarray

    def __iter__(self):
        return iter((self.matrix, self.lower, self.upper))


# ============================================================================
# 1D Constraint Blocks
# ============================================================================


def jerk_block(K: int, h: float) -> sp.csc_matrix:
    """Jerk constraint matrix: (a[k+1] - a[k]) / h"""
    rows, cols, vals = [], [], []
    for k in range(K - 1):
        rows += [k, k]
        cols += [k, k + 1]
        vals += [-1.0 / h, 1.0 / h]
    return sp.coo_matrix((vals, (rows, cols)), shape=(K - 1, K)).tocsc()


def accel_block(K: int) -> sp.csc_matrix:
    """Acceleration constraint matrix: identity"""
    return sp.eye(K, format="csc")


def vel_block(K: int, h: float) -> sp.csc_matrix:
    """Velocity integration matrix: v[k] = v0 + h * sum(a[0:k])"""
    diags = [np.ones(K - d) for d in range(K)]
    offsets = [-d for d in range(K)]
    return sp.diags(diags, offsets, shape=(K, K), format="csc") * h


def pos_block(K: int, h: float) -> sp.csc_matrix:
    """Position integration matrix: p[k] = p0 + (k+1)*h*v0 + weighted_sum(a)"""
    diags_S = [h * h * (d + 0.5) for d in range(K)]
    return sp.diags(
        [np.full(K - d, diags_S[d]) for d in range(K)],
        [-d for d in range(K)],
        shape=(K, K),
        format="csc",
    )


# ============================================================================
# Multi-Dimensional Extension
# ============================================================================


def to_nd(matrix_1d: sp.csc_matrix, dims: int) -> sp.csc_matrix:
    """Extend 1D matrix to N dimensions via Kronecker product."""
    I_dims = sp.eye(dims, format="csc")
    return sp.kron(matrix_1d, I_dims, format="csc")


# ============================================================================
# Multi-Robot
# ============================================================================


def stack_robots(per_robot_constraints: list[Constraint]) -> Constraint:
    """Stack per-robot constraints into block-diagonal multi-robot constraint."""
    matrices = [c.matrix for c in per_robot_constraints]
    matrix_multi = sp.block_diag(matrices, format="csc")

    lower_multi = np.concatenate([c.lower for c in per_robot_constraints])
    upper_multi = np.concatenate([c.upper for c in per_robot_constraints])

    return Constraint(matrix_multi, lower_multi, upper_multi)


# ============================================================================
# API
# ============================================================================


def build_dynamics_constraints(
    K_i: list[int],
    h: float,
    initial_states: dict[str, np.ndarray],
    final_states: dict[str, np.ndarray],
    box_limits: dict[str, float | np.ndarray],
    dims: int = 2,
    free_final_velocity: bool = False,
    normalize: bool = False,
) -> dict[str, Constraint]:
    """Build all dynamics constraints for multi-robot trajectory optimization.

    Args:
        K_i: List of timesteps per robot
        h: Time step duration
        initial_states: Dict with 'position' (N, dims), 'velocity' (N, dims)
        final_states: Dict with 'position' (N, dims), 'velocity' (N, dims)
        box_limits: Dict with jerk/acc/vel/pos min/max values
        dims: Number of spatial dimensions
        free_final_velocity: If True, don't enforce final velocity equality
        normalize: If True, normalize constraint rows

    Returns:
        Dict mapping constraint type to Constraint: 'jerk', 'acceleration', 'velocity', 'position'
    """
    N = len(K_i)
    p0 = initial_states["position"]
    v0 = initial_states["velocity"]
    pf = final_states["position"]
    vf = final_states["velocity"]

    # Extract limits
    jerk_min = box_limits.get("jerk_min", -np.inf)
    jerk_max = box_limits.get("jerk_max", np.inf)
    acc_min = box_limits.get("acc_min", -np.inf)
    acc_max = box_limits.get("acc_max", np.inf)
    vel_min = box_limits.get("vel_min", -np.inf)
    vel_max = box_limits.get("vel_max", np.inf)
    pos_min = box_limits.get("pos_min", np.array([-np.inf] * dims))
    pos_max = box_limits.get("pos_max", np.array([np.inf] * dims))

    jerk_constraints = []
    accel_constraints = []
    vel_constraints = []
    pos_constraints = []

    for i in range(N):
        K = K_i[i]

        # Jerk
        J = to_nd(jerk_block(K, h), dims)
        jerk_lower = np.full(J.shape[0], jerk_min)
        jerk_upper = np.full(J.shape[0], jerk_max)
        jerk_constraints.append(Constraint(J, jerk_lower, jerk_upper))

        # Acceleration
        A = to_nd(accel_block(K), dims)
        accel_lower = np.full(A.shape[0], acc_min)
        accel_upper = np.full(A.shape[0], acc_max)
        accel_constraints.append(Constraint(A, accel_lower, accel_upper))

        # Velocity
        T = to_nd(vel_block(K, h), dims)
        vel_lower = np.empty(dims * K)
        vel_upper = np.empty(dims * K)
        for k in range(K):
            for d in range(dims):
                idx = dims * k + d
                vel_lower[idx] = vel_min - v0[i, d]
                vel_upper[idx] = vel_max - v0[i, d]

        # Final velocity equality
        if not free_final_velocity:
            for d in range(dims):
                idx = dims * (K - 1) + d
                vel_lower[idx] = vf[i, d] - v0[i, d]
                vel_upper[idx] = vf[i, d] - v0[i, d]

        vel_constraints.append(Constraint(T, vel_lower, vel_upper))

        # Position
        S = to_nd(pos_block(K, h), dims)
        pos_lower = np.empty(dims * K)
        pos_upper = np.empty(dims * K)
        for k in range(K):
            for d in range(dims):
                idx = dims * k + d
                offset = p0[i, d] + h * (k + 1) * v0[i, d]
                pos_lower[idx] = pos_min[d] - offset
                pos_upper[idx] = pos_max[d] - offset

        # Final position equality
        for d in range(dims):
            idx = dims * (K - 1) + d
            pos_lower[idx] = pf[i, d] - p0[i, d] - h * K * v0[i, d]
            pos_upper[idx] = pf[i, d] - p0[i, d] - h * K * v0[i, d]

        pos_constraints.append(Constraint(S, pos_lower, pos_upper))

    result = {
        "jerk": stack_robots(jerk_constraints),
        "acceleration": stack_robots(accel_constraints),
        "velocity": stack_robots(vel_constraints),
        "position": stack_robots(pos_constraints),
    }

    if normalize:
        from .collision import normalize_constraint

        result = {k: normalize_constraint(v) for k, v in result.items()}

    return result


def integration_matrices(K: int, h: float, dims: int = 2) -> dict[str, sp.csc_matrix]:
    """Extract integration matrices without bounds."""
    T = vel_block(K, h)
    S = pos_block(K, h)
    J = jerk_block(K, h)
    J_multi = to_nd(J, dims)
    return {"T": T, "S": S, "J": J, "J_multi": J_multi}
