"""Constraint builders for double-integrator dynamics."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Constraint:
    matrix: sp.csc_matrix
    lower: np.ndarray
    upper: np.ndarray

    def __iter__(self):
        return iter((self.matrix, self.lower, self.upper))


def jerk_block(K: int, h: float) -> sp.csc_matrix:
    rows, cols, vals = [], [], []
    for k in range(K - 1):
        rows += [k, k]
        cols += [k, k + 1]
        vals += [-1.0 / h, 1.0 / h]
    return sp.coo_matrix((vals, (rows, cols)), shape=(K - 1, K)).tocsc()


def accel_block(K: int) -> sp.csc_matrix:
    return sp.eye(K, format="csc")


def vel_block(K: int, h: float) -> sp.csc_matrix:
    diags = [np.ones(K - d) for d in range(K)]
    offsets = [-d for d in range(K)]
    return sp.diags(diags, offsets, shape=(K, K), format="csc") * h


def pos_block(K: int, h: float) -> sp.csc_matrix:
    diags_S = [h * h * (d + 0.5) for d in range(K)]
    return sp.diags(
        [np.full(K - d, diags_S[d]) for d in range(K)],
        [-d for d in range(K)],
        shape=(K, K),
        format="csc",
    )


def to_nd(matrix_1d: sp.csc_matrix, dims: int) -> sp.csc_matrix:
    I_dims = sp.eye(dims, format="csc")
    return sp.kron(matrix_1d, I_dims, format="csc")


def stack_robots(per_robot_constraints: list[Constraint]) -> Constraint:
    matrices = [c.matrix for c in per_robot_constraints]
    matrix_multi = sp.block_diag(matrices, format="csc")
    lower_multi = np.concatenate([c.lower for c in per_robot_constraints])
    upper_multi = np.concatenate([c.upper for c in per_robot_constraints])
    return Constraint(matrix_multi, lower_multi, upper_multi)


def build_dynamics_constraints(
    K_i: list[int],
    h: float,
    initial_states: dict[str, np.ndarray],
    final_states: dict[str, np.ndarray],
    box_limits: dict[str, float | np.ndarray | None],
    dims: int = 2,
    free_final_velocity: bool = False,
    normalize: bool = False,
) -> dict[str, Constraint]:
    N = len(K_i)
    p0 = initial_states["position"]
    v0 = initial_states["velocity"]
    pf = final_states["position"]
    vf = final_states["velocity"]

    jerk_min = box_limits.get("jerk_min")
    jerk_max = box_limits.get("jerk_max")
    acc_min = box_limits.get("acc_min")
    acc_max = box_limits.get("acc_max")
    vel_min = box_limits.get("vel_min")
    vel_max = box_limits.get("vel_max")
    pos_min = box_limits.get("pos_min")
    pos_max = box_limits.get("pos_max")

    has_jerk = jerk_min is not None or jerk_max is not None
    has_accel = acc_min is not None or acc_max is not None
    # Velocity and position blocks are always needed for terminal constraints

    result = {}

    # Jerk constraints: skip entirely if unconstrained
    if has_jerk:
        jerk_constraints = []
        jl = jerk_min if jerk_min is not None else -np.inf
        ju = jerk_max if jerk_max is not None else np.inf
        for i in range(N):
            K = K_i[i]
            J = to_nd(jerk_block(K, h), dims)
            jerk_constraints.append(Constraint(J, np.full(J.shape[0], jl), np.full(J.shape[0], ju)))
        result["jerk"] = stack_robots(jerk_constraints)

    # Acceleration constraints: skip entirely if unconstrained
    if has_accel:
        accel_constraints = []
        al = acc_min if acc_min is not None else -np.inf
        au = acc_max if acc_max is not None else np.inf
        for i in range(N):
            K = K_i[i]
            A = to_nd(accel_block(K), dims)
            accel_constraints.append(Constraint(A, np.full(A.shape[0], al), np.full(A.shape[0], au)))
        result["acceleration"] = stack_robots(accel_constraints)

    # Velocity constraints: always needed (carries terminal velocity)
    vel_constraints = []
    vl = vel_min if vel_min is not None else -np.inf
    vu = vel_max if vel_max is not None else np.inf
    for i in range(N):
        K = K_i[i]
        T = to_nd(vel_block(K, h), dims)
        vel_lower = np.full(dims * K, vl)
        vel_upper = np.full(dims * K, vu)
        # Shift bounds by initial velocity
        for k in range(K):
            for d in range(dims):
                idx = dims * k + d
                vel_lower[idx] -= v0[i, d]
                vel_upper[idx] -= v0[i, d]
        # Terminal velocity equality
        if not free_final_velocity:
            for d in range(dims):
                idx = dims * (K - 1) + d
                vel_lower[idx] = vf[i, d] - v0[i, d]
                vel_upper[idx] = vf[i, d] - v0[i, d]
        vel_constraints.append(Constraint(T, vel_lower, vel_upper))
    result["velocity"] = stack_robots(vel_constraints)

    # Position constraints: always needed (carries terminal position)
    pos_constraints = []
    # Handle per-dimension position limits
    if pos_min is not None:
        pl = np.asarray(pos_min)
    else:
        pl = np.full(dims, -np.inf)
    if pos_max is not None:
        pu = np.asarray(pos_max)
    else:
        pu = np.full(dims, np.inf)
    for i in range(N):
        K = K_i[i]
        S = to_nd(pos_block(K, h), dims)
        pos_lower = np.empty(dims * K)
        pos_upper = np.empty(dims * K)
        for k in range(K):
            for d in range(dims):
                idx = dims * k + d
                offset = p0[i, d] + h * (k + 1) * v0[i, d]
                pos_lower[idx] = pl[d] - offset
                pos_upper[idx] = pu[d] - offset
        # Terminal position equality
        for d in range(dims):
            idx = dims * (K - 1) + d
            pos_lower[idx] = pf[i, d] - p0[i, d] - h * K * v0[i, d]
            pos_upper[idx] = pf[i, d] - p0[i, d] - h * K * v0[i, d]
        pos_constraints.append(Constraint(S, pos_lower, pos_upper))
    result["position"] = stack_robots(pos_constraints)

    if normalize:
        from .collision import normalize_constraint
        result = {k: normalize_constraint(v) for k, v in result.items()}

    return result


def integration_matrices(K: int, h: float, dims: int = 2) -> dict[str, sp.csc_matrix]:
    T = vel_block(K, h)
    S = pos_block(K, h)
    J = jerk_block(K, h)
    J_multi = to_nd(J, dims)
    return {"T": T, "S": S, "J": J, "J_multi": J_multi}
