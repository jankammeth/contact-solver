"""Velocity arc insertion for enforcing component-wise velocity bounds.

When velocity bounds are violated, this module inserts velocity arcs
(constant velocity segments) using a closed-form solution that minimizes
acceleration cost.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ComponentArc:
    """Represents a velocity arc for a single component (x or y)."""

    component: int  # 0 for x, 1 for y
    t1: float  # Entry time into saturation (global time)
    t2: float  # Exit time from saturation (global time)
    v_sat: float  # Saturation velocity (+/- v_max)
    p1: float  # Position at arc entry
    p2: float  # Position at arc exit


def eval_1d_cubic(
    p0: float, v0: float, pf: float, vf: float, T: float, t: float
) -> tuple[float, float, float]:
    """Evaluate 1D cubic trajectory at time t. Returns (position, velocity, acceleration)."""
    if T < 1e-10:
        return p0, v0, 0.0

    c = (3 * (pf - p0) - (2 * v0 + vf) * T) / T**2
    d = (-2 * (pf - p0) + (v0 + vf) * T) / T**3

    pos = p0 + v0 * t + c * t**2 + d * t**3
    vel = v0 + 2 * c * t + 3 * d * t**2
    acc = 2 * c + 6 * d * t

    return pos, vel, acc


def compute_max_speed_component(
    p0: float, v0: float, pf: float, vf: float, T: float
) -> tuple[float, float, float]:
    """Compute maximum speed for a 1D cubic trajectory."""
    if T < 1e-10:
        return abs(v0), 0.0, v0

    c = (3 * (pf - p0) - (2 * v0 + vf) * T) / T**2
    d = (-2 * (pf - p0) + (v0 + vf) * T) / T**3

    max_speed = 0.0
    t_max = 0.0
    v_at_max = v0

    candidates = [0.0, T]

    if abs(d) > 1e-12:
        t_crit = -c / (3 * d)
        if 0 < t_crit < T:
            candidates.append(t_crit)

    for t in candidates:
        _, vel, _ = eval_1d_cubic(p0, v0, pf, vf, T, t)
        if abs(vel) > max_speed:
            max_speed = abs(vel)
            t_max = t
            v_at_max = vel

    return max_speed, t_max, v_at_max


def compute_velocity_arc_params(
    p0: float, v0: float, pf: float, vf: float, T: float, v_max: float, tol: float = 1e-6
) -> tuple[float, float, float, float, float] | None:
    """Compute velocity arc parameters using closed-form solution."""
    max_speed, t_max, v_at_max = compute_max_speed_component(p0, v0, pf, vf, T)

    if max_speed <= v_max + tol:
        return None

    v_sat = np.sign(v_at_max) * v_max
    delta_p = pf - p0

    alpha = v_sat - v0
    beta = v_sat - vf
    R = 3 * (delta_p - v_sat * T)

    if abs(alpha) < tol and abs(beta) < tol:
        return None
    elif abs(alpha) < tol:
        T1 = 0.0
        T3 = -R / beta
    elif abs(beta) < tol:
        T1 = -R / alpha
        T3 = 0.0
    else:
        denom = np.abs(alpha) ** 1.5 + np.abs(beta) ** 1.5
        T1 = -R * np.sqrt(np.abs(alpha)) / denom * np.sign(alpha)
        T3 = -R * np.sqrt(np.abs(beta)) / denom * np.sign(beta)

    T2 = T - T1 - T3

    if T1 < -tol or T3 < -tol or T2 < -tol:
        return None

    T1 = max(0.0, T1)
    T3 = max(0.0, T3)

    p1 = p0 + T1 * (v0 + 2 * v_sat) / 3
    p2 = pf - T3 * (2 * v_sat + vf) / 3

    return T1, T3, v_sat, p1, p2


def sample_component_trajectory(
    p0: float,
    v0: float,
    pf: float,
    vf: float,
    T: float,
    arc: ComponentArc | None,
    times: np.ndarray,
    t_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a 1D trajectory with optional velocity arc."""
    n = len(times)
    positions = np.zeros(n)
    velocities = np.zeros(n)
    accelerations = np.zeros(n)

    t_start = t_offset
    t_end = t_offset + T

    for i, t_global in enumerate(times):
        if t_global < t_start - 1e-10 or t_global > t_end + 1e-10:
            continue

        t_local = t_global - t_start

        if arc is None:
            positions[i], velocities[i], accelerations[i] = eval_1d_cubic(
                p0, v0, pf, vf, T, t_local
            )
        else:
            T1 = arc.t1 - t_offset
            T3 = t_end - arc.t2

            if T1 > 1e-10 and t_local <= T1:
                c1 = (arc.v_sat - v0) / T1
                d1 = (v0 - arc.v_sat) / (3 * T1**2)
                positions[i] = p0 + v0 * t_local + c1 * t_local**2 + d1 * t_local**3
                velocities[i] = v0 + 2 * c1 * t_local + 3 * d1 * t_local**2
                accelerations[i] = 2 * c1 + 6 * d1 * t_local
            elif T3 > 1e-10 and t_global >= arc.t2:
                t_local3 = t_global - arc.t2
                d3 = (vf - arc.v_sat) / (3 * T3**2)
                positions[i] = arc.p2 + arc.v_sat * t_local3 + d3 * t_local3**3
                velocities[i] = arc.v_sat + 3 * d3 * t_local3**2
                accelerations[i] = 6 * d3 * t_local3
            else:
                t_cruise = t_local - T1 if T1 > 1e-10 else t_local
                positions[i] = arc.p1 + arc.v_sat * t_cruise
                velocities[i] = arc.v_sat
                accelerations[i] = 0.0

    return positions, velocities, accelerations


def apply_velocity_bounds_to_trajectories(
    times: np.ndarray,
    positions: list[np.ndarray],
    velocities: list[np.ndarray],
    accelerations: list[np.ndarray],
    knot_times: list[float],
    v_max: float,
    verbose: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
    """Apply component-wise velocity bounds to sampled trajectories."""
    N = len(positions)
    T_total = times[-1]

    segment_bounds = [0.0] + sorted(knot_times) + [T_total]
    velocity_arc_times = []

    new_positions = []
    new_velocities = []
    new_accelerations = []

    for robot_idx in range(N):
        if verbose:
            print(f"  Robot {robot_idx}:")

        robot_pos = np.zeros_like(positions[robot_idx])
        robot_vel = np.zeros_like(velocities[robot_idx])
        robot_acc = np.zeros_like(accelerations[robot_idx])

        for seg_idx in range(len(segment_bounds) - 1):
            t_start = segment_bounds[seg_idx]
            t_end = segment_bounds[seg_idx + 1]
            T_seg = t_end - t_start

            if T_seg < 1e-10:
                continue

            start_idx = np.searchsorted(times, t_start)
            end_idx = np.searchsorted(times, t_end)
            if end_idx >= len(times):
                end_idx = len(times) - 1

            p0 = positions[robot_idx][start_idx].copy()
            v0 = velocities[robot_idx][start_idx].copy()
            pf = positions[robot_idx][end_idx].copy()
            vf = velocities[robot_idx][end_idx].copy()

            for comp in range(2):
                comp_name = "x" if comp == 0 else "y"

                arc_params = compute_velocity_arc_params(
                    p0[comp], v0[comp], pf[comp], vf[comp], T_seg, v_max
                )

                if arc_params is None:
                    arc = None
                    if verbose:
                        max_speed, _, _ = compute_max_speed_component(
                            p0[comp], v0[comp], pf[comp], vf[comp], T_seg
                        )
                        print(f"    Segment {seg_idx}, {comp_name}: no arc (max_v={max_speed:.3f})")
                else:
                    T1, T3, v_sat, p1, p2 = arc_params
                    t1 = t_start + T1
                    t2 = t_end - T3

                    arc = ComponentArc(
                        component=comp, t1=t1, t2=t2, v_sat=v_sat, p1=p1, p2=p2
                    )

                    velocity_arc_times.extend([t1, t2])

                    if verbose:
                        T2 = T_seg - T1 - T3
                        print(
                            f"    Segment {seg_idx}, {comp_name}: arc at v={v_sat:.2f}, "
                            f"T1={T1:.3f}, T2={T2:.3f}, T3={T3:.3f}"
                        )

                pos_comp, vel_comp, acc_comp = sample_component_trajectory(
                    p0[comp], v0[comp], pf[comp], vf[comp], T_seg, arc, times, t_start
                )

                mask = (times >= t_start - 1e-10) & (times <= t_end + 1e-10)
                robot_pos[mask, comp] = pos_comp[mask]
                robot_vel[mask, comp] = vel_comp[mask]
                robot_acc[mask, comp] = acc_comp[mask]

        new_positions.append(robot_pos)
        new_velocities.append(robot_vel)
        new_accelerations.append(robot_acc)

    return new_positions, new_velocities, new_accelerations, sorted(set(velocity_arc_times))
