"""General-dimensional joint sequential contact insertion with explicit masses."""

import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import LinAlgError, solve_banded
from scipy.sparse.linalg import lsmr

from .base import Solver
from .cubic_utils import cubic_coefficients, distance_candidates
from .vector_matching import MatchingSystem, VectorContact, blocks_to_banded, blocks_to_sparse


@dataclass
class RelativeTrajectory:
    times: np.ndarray
    coefficients: np.ndarray  # segment, relative coordinate, polynomial power, spatial coordinate
    barycenter: np.ndarray  # polynomial power, spatial coordinate

    @classmethod
    def from_system(cls, system, x, barycenter):
        times, r, v = system.states(x)
        h = np.diff(times)[:, None, None]
        coefficients = np.stack(
            (
                r[:-1],
                v[:-1],
                (3 * (r[1:] - r[:-1]) - (2 * v[:-1] + v[1:]) * h) / h**2,
                (-2 * (r[1:] - r[:-1]) + (v[:-1] + v[1:]) * h) / h**3,
            ),
            axis=2,
        )
        return cls(times, coefficients, barycenter)

    def relative(self, times, derivative=0):
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or not np.all(np.isfinite(times))
            or np.any(times < self.times[0])
            or np.any(times > self.times[-1])
        ):
            raise ValueError("Sample times must be a finite 1-D array within [0, T]")
        if derivative not in (0, 1, 2, 3):
            raise ValueError("Supported derivative orders are 0 through 3")
        segments = np.clip(
            np.searchsorted(self.times, times, side="right") - 1, 0, len(self.times) - 2
        )
        local = times - self.times[segments]
        c = self.coefficients[segments]
        result = np.zeros_like(c[:, :, 0])
        for power in range(3, derivative - 1, -1):
            factor = np.prod(np.arange(power - derivative + 1, power + 1))
            result = result * local[:, None, None] + factor * c[:, :, power]
        return result

    def robots(self, times, derivative=0):
        relative = self.relative(times, derivative)
        center = np.zeros((len(times), self.barycenter.shape[1]))
        for power in range(3, derivative - 1, -1):
            factor = np.prod(np.arange(power - derivative + 1, power + 1))
            center = center * np.asarray(times)[:, None] + factor * self.barycenter[power]
        n = relative.shape[1] + 1
        first = center + relative.sum(axis=1) / n
        return np.concatenate((first[:, None], first[:, None] - relative), axis=1).transpose(
            1, 0, 2
        )

    def closest(self):
        """All robot pairs on every segment, including every knot and endpoint."""
        n = self.coefficients.shape[1] + 1
        best = (None, 0.0, np.inf)
        for s, h in enumerate(np.diff(self.times)):
            for p, q in combinations(range(n), 2):
                c = self.coefficients[s, q - 1].copy()
                if p:
                    c -= self.coefficients[s, p - 1]
                for t, distance in distance_candidates(c, h, self.times[s]):
                    if distance < best[2]:
                        best = ((p, q), t, distance)
        return best


class VectorContactSolver(Solver):
    """Joint solver for the unconstrained fleet problem in dimension d >= 2.

    The legacy ContactSolver remains the planar angle/speed reference. This
    solver always supplies its analytical Jacobian, uses banded damped Newton
    steps, and falls back to sparse damped least squares for singular systems.
    Success requires algebraic/geometry/mass checks and continuous separation.
    """

    def __init__(self, config, verbose=False, max_contacts=None, max_contacts_per_pair=None):
        super().__init__(config)
        self.d = config.problem.dimension
        if not isinstance(self.d, int) or self.d < 2 or self.N < 2:
            raise ValueError("VectorContactSolver requires integer dimension >= 2 and N >= 2")
        self.verbose = verbose
        self.max_contacts = config.solver.sci_max_contacts if max_contacts is None else max_contacts
        self.max_contacts_per_pair = (
            config.solver.sci_max_contacts_per_pair
            if max_contacts_per_pair is None
            else max_contacts_per_pair
        )
        self.options = config.solver
        if (
            not np.isfinite(self.R)
            or self.R <= 0
            or not np.isfinite(self.h)
            or self.h <= 0
            or min(self.max_contacts, self.max_contacts_per_pair) < 0
        ):
            raise ValueError("Radius/timestep must be positive and contact caps nonnegative")
        tolerances = (
            self.options.sci_residual_tol,
            self.options.sci_geometry_tol,
            self.options.sci_mass_tol,
            self.options.sci_detection_tol,
            self.options.sci_min_contact_gap,
        )
        if not np.all(np.isfinite(tolerances)) or min(tolerances) < 0 or min(tolerances[:2]) <= 0:
            raise ValueError("Invalid vector solver tolerances")
        if self.options.sci_root_max_iterations < 0:
            raise ValueError("Root iteration limit must be nonnegative")
        # These bounds require additional junction types, absent from Sections 4.2-4.4.
        if (
            np.any(np.isfinite(self.pos_min))
            or np.any(np.isfinite(self.pos_max))
            or any(
                value is not None
                for value in (
                    self.vel_min,
                    self.vel_max,
                    self.acc_min,
                    self.acc_max,
                    self.jerk_min,
                    self.jerk_max,
                )
            )
        ):
            raise ValueError("VectorContactSolver supports the unconstrained fleet problem only")

    def _endpoint(self, positions, velocities):
        positions = np.asarray(positions, dtype=float)
        if positions.shape not in ((self.N, self.d), (self.N * self.d,)):
            raise ValueError(
                f"Expected endpoint shape ({self.N}, {self.d}) or a flattened equivalent"
            )
        velocities = (
            np.zeros_like(positions) if velocities is None else np.asarray(velocities, dtype=float)
        )
        if velocities.size != positions.size or velocities.shape not in (
            (self.N, self.d),
            (self.N * self.d,),
        ):
            raise ValueError("Velocity shape must match the endpoint dimension")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("Endpoint states must be finite")
        return positions.ravel().copy(), velocities.ravel().copy()

    def set_initial_states(self, positions, velocities=None):
        self.initial_positions, self.initial_velocities = self._endpoint(positions, velocities)

    def set_final_states(self, positions, velocities=None):
        self.final_positions, self.final_velocities = self._endpoint(positions, velocities)
        if self.T is None:
            if self.initial_positions is None:
                raise ValueError("Call set_initial_states before automatic horizon selection")
            distance = np.linalg.norm(
                (self.final_positions - self.initial_positions).reshape(self.N, self.d), axis=1
            ).max()
            self.T = max(self.h, np.ceil(0.75 * distance / self.h) * self.h)
            self.K = int(round(self.T / self.h))

    def matching_system(self, contacts):
        arrays = [
            a.reshape(self.N, self.d)
            for a in (
                self.initial_positions,
                self.initial_velocities,
                self.final_positions,
                self.final_velocities,
            )
        ]
        relative = [a[0] - a[1:] for a in arrays]
        return MatchingSystem([c.pair for c in contacts], *relative, self.R, self.T, self.min_gap)

    def initialize_contact(self, pair, contact_time, trajectory):
        r = trajectory.relative([contact_time])[0]
        v = trajectory.relative([contact_time], 1)[0]
        p, q = pair
        relative_position = r[q - 1] - (r[p - 1] if p else 0)
        relative_velocity = v[q - 1] - (v[p - 1] if p else 0)
        distance = np.linalg.norm(relative_position)
        if distance > 1e-12 * max(1.0, self.R):
            normal = relative_position / distance
        else:
            speed = np.linalg.norm(relative_velocity)
            normal = np.zeros(self.d)
            if speed > 1e-12:
                direction = relative_velocity / speed
                normal[np.argmin(np.abs(direction))] = 1
                normal -= (normal @ direction) * direction
                normal /= np.linalg.norm(normal)
            else:
                normal[0] = 1
        velocity = relative_velocity - (normal @ relative_velocity) * normal
        others = [j for j in range(self.N - 1) if j != q - 1]
        return VectorContact(
            pair, contact_time, normal, velocity, 0.0, r[others].copy(), v[others].copy()
        )

    def _solve_system(self, system, x, deadline):
        """Feasible damped Newton, with no post-solve change to contact times."""
        root_tolerance = min(self.options.sci_residual_tol, self.options.sci_geometry_tol)
        nfev, fallbacks = 0, 0
        message = "iteration_limit"
        for iteration in range(self.options.sci_root_max_iterations + 1):
            residual, blocks = system.evaluate(x, True)
            nfev += 1
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
            if iteration == self.options.sci_root_max_iterations:
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
                        0.99 * np.min((gaps[decreasing] - self.min_gap) / -gap_steps[decreasing]),
                    )
                for _ in range(30):
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

    def _checks(self, system, x, trajectory):
        z = system.unpack(x)
        residual = system.evaluate(x).reshape(system.m, system.b)
        md = (self.N - 1) * self.d

        def norm(a):
            return float(np.max(np.abs(a))) if a.size else 0.0

        endpoint_error = max(
            norm(
                trajectory.robots([0, self.T], order)[:, endpoint] - target.reshape(self.N, self.d)
            )
            for order, targets in (
                (0, (self.initial_positions, self.final_positions)),
                (1, (self.initial_velocities, self.final_velocities)),
            )
            for endpoint, target in enumerate(targets)
        )
        checks = dict(
            residual=norm(residual),
            acceleration_jump=norm(residual[:, :md]),
            jerk_balance=norm(residual[:, md : 2 * md]),
            normalization=norm(residual[:, -2]),
            tangency=norm(residual[:, -1]),
            endpoint_error=endpoint_error,
            minimum_mass=float(np.min(z[:, 1 + 2 * self.d])) if len(z) else 0.0,
            minimum_knot_gap=float(np.min(np.diff(trajectory.times))),
        )
        checks["valid"] = bool(
            checks["residual"] <= self.options.sci_residual_tol
            and max(checks["normalization"], checks["tangency"]) <= self.options.sci_geometry_tol
            and endpoint_error <= self.options.sci_residual_tol
            and checks["minimum_mass"] >= -self.options.sci_mass_tol
            and checks["minimum_knot_gap"] >= self.min_gap
        )
        return checks

    def generate_trajectories(self, verbose=None):
        started = time.perf_counter()
        verbose = self.verbose if verbose is None else verbose
        if self.initial_positions is None or self.final_positions is None:
            raise ValueError("Set initial and final states before solving")
        if self.T is None or not np.isfinite(self.T) or self.T <= 0:
            raise ValueError("Time horizon must be positive and finite")
        self.min_gap = self.options.sci_min_contact_gap or 1e-6 * self.T
        barycenter = np.array(
            cubic_coefficients(
                *[
                    a.reshape(self.N, self.d).mean(axis=0)
                    for a in (
                        self.initial_positions,
                        self.initial_velocities,
                        self.final_positions,
                        self.final_velocities,
                    )
                ],
                self.T,
            )
        )
        self.contacts = []
        x = np.empty(0)
        system = self.matching_system(self.contacts)
        trajectory = RelativeTrajectory.from_system(system, x, barycenter)
        deadline = started + self.options.timeout if self.options.timeout > 0 else np.inf
        roots = []
        matching_seconds = 0.0
        status, reason = "success", "converged"
        while True:
            pair, contact_time, distance = trajectory.closest()
            checks = self._checks(system, x, trajectory)
            if not checks["valid"]:
                status, reason = "invalid_solution", "invalid_matching_solution"
                break
            endpoint_positions = trajectory.robots([0, self.T])
            if any(
                np.min(np.linalg.norm(endpoint_positions[p] - endpoint_positions[q], axis=1))
                < self.R - self.options.sci_detection_tol
                for p, q in combinations(range(self.N), 2)
            ):
                status, reason = "invalid_solution", "infeasible_endpoint_separation"
                break
            if distance >= self.R - self.options.sci_detection_tol:
                break
            if len(self.contacts) >= self.max_contacts:
                status, reason = "unresolved", "max_contacts"
                break
            if Counter(c.pair for c in self.contacts)[pair] >= self.max_contacts_per_pair:
                status, reason = "unresolved", "max_contacts_per_pair"
                break
            if np.min(np.abs(trajectory.times - contact_time)) <= self.min_gap:
                status, reason = "unresolved", "minimum_knot_gap"
                break
            if time.perf_counter() >= deadline:
                status, reason = "unresolved", "timeout"
                break
            self.contacts.append(self.initialize_contact(pair, contact_time, trajectory))
            self.contacts.sort(key=lambda c: c.t_star)
            system = self.matching_system(self.contacts)
            x = np.concatenate([c.to_vector() for c in self.contacts])
            root_started = time.perf_counter()
            x, root_info = self._solve_system(system, x, deadline)
            matching_seconds += time.perf_counter() - root_started
            roots.append(root_info)
            self.contacts = system.contacts(x)
            trajectory = RelativeTrajectory.from_system(system, x, barycenter)
            if verbose:
                print(f"VectorContactSolver: m={len(self.contacts)}, root={root_info['message']}")
            if not root_info["success"]:
                status, reason = "root_failure", root_info["message"]
                break
        self.system, self.solution, self.trajectory = system, x, trajectory
        checks = self._checks(system, x, trajectory)
        closest = trajectory.closest()
        solve_time = time.perf_counter() - started
        times, positions, velocities, accelerations = self.sample()
        self.K = len(times)
        self.trajectories = dict(
            positions=positions, velocities=velocities, accelerations=accelerations
        )
        m = len(self.contacts)
        return dict(
            trajectories=self.trajectories,
            times=times,
            contact_times=[c.t_star for c in self.contacts],
            contacts=self.contacts,
            arc_times=[],
            metrics=dict(
                converged=status == "success",
                status=status,
                convergence_reason=reason,
                dimension=self.d,
                num_contacts=m,
                min_distance=closest[2],
                closest_pair=closest[0],
                closest_time=closest[1],
                checks=checks,
                roots=roots,
                intrinsic_degrees_of_freedom=2 * m * self.d * (self.N - 1),
                system_size=m * (2 * self.d * (self.N - 1) + 2),
                max_contacts_single_pair=max(
                    Counter(c.pair for c in self.contacts).values(), default=0
                ),
                matching_seconds=matching_seconds,
                matching_solves=len(roots),
                nfev=sum(root["nfev"] for root in roots),
                linear_solver="banded Newton / sparse LSMR fallback",
                tolerances=dict(
                    residual=self.options.sci_residual_tol,
                    geometry=self.options.sci_geometry_tol,
                    mass=self.options.sci_mass_tol,
                    separation=self.options.sci_detection_tol,
                    minimum_knot_gap=self.min_gap,
                ),
                timing=dict(
                    total_time=solve_time,
                    solve_time=solve_time,
                    postprocess_time=time.perf_counter() - started - solve_time,
                    wall_time=time.perf_counter() - started,
                ),
            ),
        )

    def sample(self, n=1000, times=None):
        if not hasattr(self, "trajectory"):
            raise ValueError("Call generate_trajectories before sample")
        times = np.linspace(0, self.T, n) if times is None else np.asarray(times, dtype=float)
        return (times, *[list(self.trajectory.robots(times, order)) for order in (0, 1, 2)])
