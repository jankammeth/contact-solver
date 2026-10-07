"""General-dimensional joint sequential contact insertion with explicit masses."""

import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .base import Solver
from .collision_search import COLLISION_BACKEND, minima
from .cubic_utils import cubic_coefficients, distance_candidates
from .matching_numerics import damped_newton, insertion_time
from .reaction_matching import ReactionMatching
from .vector_matching import MatchingSystem, VectorContact


@dataclass
class RelativeTrajectory:
    times: np.ndarray
    coefficients: np.ndarray  # segment, relative coordinate, polynomial power, spatial coordinate
    barycenter: np.ndarray  # polynomial power, spatial coordinate

    reaction_constructed: bool = False

    @classmethod
    def from_system(cls, system, x, barycenter):
        if not system.m:
            # No contact states or matching maps are needed for one free cubic.
            if x.size or system.T < system.min_gap:
                raise ValueError("Invalid empty contact state or minimum gap")
            coefficients = np.array(
                cubic_coefficients(system.r0, system.v0, system.rf, system.vf, system.T)
            ).transpose(1, 0, 2)[None]
            return cls(np.array([0.0, system.T]), coefficients, barycenter)
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
            factor = ((1, 1, 1, 1), (0, 1, 2, 3), (0, 0, 2, 6), (0, 0, 0, 6))[derivative][power]
            result = result * local[:, None, None] + factor * c[:, :, power]
        return result

    def robots(self, times, derivative=0):
        relative = self.relative(times, derivative)
        center = np.zeros((len(times), self.barycenter.shape[1]))
        for power in range(3, derivative - 1, -1):
            factor = ((1, 1, 1, 1), (0, 1, 2, 3), (0, 0, 2, 6), (0, 0, 0, 6))[derivative][power]
            center = center * np.asarray(times)[:, None] + factor * self.barycenter[power]
        n = relative.shape[1] + 1
        first = center + relative.sum(axis=1) / n
        return np.concatenate((first[:, None], first[:, None] - relative), axis=1).transpose(
            1, 0, 2
        )

    def endpoint_states(self):
        """Evaluate both endpoint states directly from the stored cubic coefficients."""
        h = self.times[-1] - self.times[-2]
        a, b, c, d = self.coefficients[-1].transpose(1, 0, 2)
        relative = np.stack(
            (
                np.stack((self.coefficients[0, :, 0], ((d * h + c) * h + b) * h + a)),
                np.stack((self.coefficients[0, :, 1], (3 * d * h + 2 * c) * h + b)),
            )
        )
        t = self.times[-1]
        a, b, c, d = self.barycenter
        center = np.array(((a, ((d * t + c) * t + b) * t + a), (b, (3 * d * t + 2 * c) * t + b)))
        first = center + relative.sum(axis=2) / (relative.shape[2] + 1)
        return np.concatenate((first[:, :, None], first[:, :, None] - relative), axis=2).transpose(
            0, 2, 1, 3
        )

    def first_violation(self, threshold):
        """Earliest violating stationary candidate, for the insertion ablation."""
        best = (None, np.inf, np.inf)
        n = self.coefficients.shape[1] + 1
        for s, h in enumerate(np.diff(self.times)):
            for p, q in combinations(range(n), 2):
                c = self.coefficients[s, q - 1].copy()
                if p:
                    c -= self.coefficients[s, p - 1]
                for t, distance in distance_candidates(c, h, self.times[s]):
                    if distance < threshold and t < best[1]:
                        best = ((p, q), t, distance)
        return best

    def closest(self):
        """Global pairwise minimum, pruning only intervals that cannot improve it."""
        n = self.coefficients.shape[1] + 1
        p, q = np.triu_indices(n, 1)
        c = self.coefficients[:, q - 1].copy()
        mask = p > 0
        c[:, mask] -= self.coefficients[:, p[mask] - 1]
        durations = np.repeat(np.diff(self.times), len(p))
        offsets = np.repeat(self.times[:-1], len(p))
        times, distances = minima(c.reshape(-1, 4, c.shape[-1]), durations)
        k = int(np.argmin(distances))
        return (
            (int(p[k % len(p)]), int(q[k % len(p)])),
            float(offsets[k] + times[k] * durations[k]),
            float(distances[k]),
        )


class ContactSolver(Solver):
    """Joint solver for the unconstrained fleet problem in dimension d >= 2.

    All joint experiments use this implementation. Banded damped Newton uses
    an analytical Jacobian (finite differences only for an explicit ablation),
    with sparse damped least squares for singular systems.
    Success requires algebraic/geometry/mass checks and continuous separation.
    """

    def __init__(
        self,
        config,
        verbose=False,
        contact_selection="deepest",
    ):
        super().__init__(config)
        self.d = config.problem.dimension
        if not isinstance(self.d, int) or self.d < 2 or self.N < 2:
            raise ValueError("ContactSolver requires integer dimension >= 2 and N >= 2")
        if contact_selection not in ("deepest", "earliest"):
            raise ValueError("contact_selection must be deepest or earliest")
        self.contact_selection = contact_selection
        self.verbose = verbose
        self.options = config.solver
        if (
            not np.isfinite(self.R)
            or self.R <= 0
            or not np.isfinite(self.h)
            or self.h <= 0
        ):
            raise ValueError("Radius and timestep must be positive")
        if not np.isfinite(self.options.timeout) or self.options.timeout <= 0:
            raise ValueError("Solve timeout must be positive and finite")
        tolerances = (
            self.options.sci_residual_tol,
            self.options.sci_mass_tol,
            self.options.sci_detection_tol,
            self.options.sci_min_contact_gap,
        )
        if not np.all(np.isfinite(tolerances)) or min(tolerances) < 0 or tolerances[0] <= 0:
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
            raise ValueError("ContactSolver supports the unconstrained fleet problem only")

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
        return damped_newton(system, x, deadline, self.options)

    def _recover_reactions(self, system, initial, barycenter, deadline):
        reduced = ReactionMatching(system)
        try:
            result = reduced.solve(initial, deadline, max(100, 20 * len(initial)))
            times, coefficients, order = reduced.coefficients(result.x)
            if (
                not result.success
                or not np.all(np.isfinite(result.x))
                or np.min(np.diff(times)) <= 0
                or np.min(np.diff(times)) < self.min_gap
            ):
                return None
            trajectory = RelativeTrajectory(times, coefficients, barycenter, True)
            states = result.x.reshape(system.m, self.d + 2)
            relative = trajectory.relative(times[1:-1])
            velocities = trajectory.relative(times[1:-1], 1)
            contacts = []
            for k, index in enumerate(order):
                p, q = system.pairs[index]
                others = [j for j in range(self.N - 1) if j != q - 1]
                velocity = velocities[k, q - 1] - (velocities[k, p - 1] if p else 0)
                contacts.append(
                    VectorContact(
                        (p, q),
                        states[index, 0],
                        states[index, 1 : 1 + self.d],
                        velocity,
                        states[index, -1],
                        relative[k, others],
                        velocities[k, others],
                    )
                )
            recovered = self.matching_system(contacts)
            x = np.concatenate([c.to_vector() for c in contacts])
            checks = self._checks(recovered, x, trajectory)
            # Negative masses are handled by the active-set loop, not accepted.
            if (
                checks["residual"] > self.options.sci_residual_tol
                or checks["endpoint_error"] > self.options.sci_residual_tol
            ):
                return None
            info = dict(
                success=True,
                message="reaction_recovery",
                termination_status=int(result.status),
                termination_message=str(result.message),
                nfev=result.nfev,
                iterations=None,
                sparse_fallbacks=0,
                reordered=bool(np.any(order != np.arange(system.m))),
                system_size=system.m * (self.d + 2),
            )
            return recovered, x, trajectory, info
        except (TimeoutError, ValueError, FloatingPointError):
            return None

    def _checks(self, system, x, trajectory, endpoint_states=None):
        z = system.unpack(x) if system.m else x.reshape(0, system.b)
        md = (self.N - 1) * self.d
        if trajectory.reaction_constructed:
            residual = np.zeros((system.m, system.b))
            h = np.diff(trajectory.times)[:-1, None, None]
            c = trajectory.coefficients
            residual[:, :md] = (2 * c[1:, :, 2] - 2 * c[:-1, :, 2] - 6 * c[:-1, :, 3] * h).reshape(
                system.m, md
            )
            jump = 6 * (c[1:, :, 3] - c[:-1, :, 3])
            mass_normal = z[:, 1 + 2 * self.d, None] * z[:, 1 : 1 + self.d]
            residual[:, md : 2 * md] = (
                jump - self.R * system.jerk_factors[:, :, None] * mass_normal[:, None]
            ).reshape(system.m, md)
            residual[:, -2] = np.sum(z[:, 1 : 1 + self.d] ** 2, axis=1) - 1
            residual[:, -1] = np.sum(
                z[:, 1 : 1 + self.d] * z[:, 1 + self.d : 1 + 2 * self.d], axis=1
            )
            r = trajectory.relative(trajectory.times[1:-1])
            contact_error = max(
                np.max(
                    np.abs(r[k, q - 1] - (r[k, p - 1] if p else 0) - self.R * z[k, 1 : 1 + self.d])
                )
                for k, (p, q) in enumerate(system.pairs)
            )
        else:
            residual = (
                system.evaluate(x).reshape(system.m, system.b)
                if system.m
                else np.empty((0, system.b))
            )
            contact_error = 0.0

        def norm(a):
            return float(np.max(np.abs(a))) if a.size else 0.0

        if endpoint_states is None:
            endpoint_states = trajectory.endpoint_states()
        endpoint_error = max(
            norm(states[:, endpoint] - target.reshape(self.N, self.d))
            for states, targets in zip(
                endpoint_states,
                (
                    (self.initial_positions, self.final_positions),
                    (self.initial_velocities, self.final_velocities),
                ),
                strict=True,
            )
            for endpoint, target in enumerate(targets)
        )
        checks = dict(
            residual=max(norm(residual), contact_error),
            contact_position_error=contact_error,
            acceleration_jump=norm(residual[:, :md]),
            jerk_balance=norm(residual[:, md : 2 * md]),
            normalization=norm(residual[:, -2]),
            tangency=norm(residual[:, -1]),
            endpoint_error=endpoint_error,
            minimum_mass=float(np.min(z[:, 1 + 2 * self.d])) if len(z) else 0.0,
            minimum_knot_gap=(float(np.min(np.diff(trajectory.times))) if system.m else self.T),
        )
        checks["valid"] = bool(
            checks["residual"] <= self.options.sci_residual_tol
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
        self.min_gap = self.options.sci_min_contact_gap
        endpoint_arrays = np.array(
            (
                self.initial_positions,
                self.initial_velocities,
                self.final_positions,
                self.final_velocities,
            )
        ).reshape(4, self.N, self.d)
        barycenter = np.array(cubic_coefficients(*endpoint_arrays.mean(axis=1), self.T))
        self.contacts = []
        x = np.empty(0)
        system = self.matching_system(self.contacts)
        trajectory = RelativeTrajectory.from_system(system, x, barycenter)
        deadline = started + self.options.timeout
        roots = []
        matching_seconds = 0.0
        removed_contacts = []
        insertion_offsets = []
        active_set_updates = 0
        status, reason = "success", "converged"
        while True:
            closest = trajectory.closest()
            pair, contact_time, distance = closest
            endpoint_states = trajectory.endpoint_states()
            checks = self._checks(system, x, trajectory, endpoint_states)
            if not checks["valid"]:
                status, reason = "invalid_solution", "invalid_matching_solution"
                break
            # The continuous search includes endpoints; only diagnose endpoint
            # infeasibility separately when a violation actually exists.
            if distance >= self.R - self.options.sci_detection_tol:
                break
            endpoint_positions = endpoint_states[0]
            if any(
                np.min(np.linalg.norm(endpoint_positions[p] - endpoint_positions[q], axis=1))
                < self.R - self.options.sci_detection_tol
                for p, q in combinations(range(self.N), 2)
            ):
                status, reason = "invalid_solution", "infeasible_endpoint_separation"
                break
            if self.contact_selection == "earliest":
                pair, contact_time, _ = trajectory.first_violation(
                    self.R - self.options.sci_detection_tol
                )
            try:
                initial_time = insertion_time(
                    contact_time, [c.t_star for c in self.contacts], self.T
                )
            except ValueError:
                status, reason = "unresolved", "coincident_initialization_failed"
                break
            if initial_time != contact_time:
                insertion_offsets.append(
                    dict(pair=pair, collision_time=contact_time, initial_time=initial_time)
                )
            if (
                not 0 < initial_time < self.T
                or np.min(np.abs(trajectory.times - initial_time)) < self.min_gap
            ):
                status, reason = "unresolved", "minimum_knot_gap"
                break
            if time.perf_counter() >= deadline:
                status, reason = "unresolved", "timeout"
                break
            active_set_updates += 1
            self.contacts.append(self.initialize_contact(pair, initial_time, trajectory))
            self.contacts.sort(key=lambda c: c.t_star)
            system = self.matching_system(self.contacts)
            x = np.concatenate([c.to_vector() for c in self.contacts])
            root_started = time.perf_counter()
            while True:
                recovered_trajectory = None
                if system.m:
                    initial = x.copy()
                    x, root_info = self._solve_system(system, x, deadline)
                    roots.append(root_info)
                    if (
                        self.options.sci_reaction_recovery
                        and not root_info["success"]
                        and root_info["message"] != "timeout"
                        and self.options.sci_root_max_iterations > 0
                    ):
                        recovered = self._recover_reactions(system, initial, barycenter, deadline)
                        if recovered is not None:
                            system, x, recovered_trajectory, root_info = recovered
                            roots.append(root_info)
                else:
                    root_info = dict(success=True, message="empty_active_set")
                self.contacts = system.contacts(x)
                if not root_info["success"] or not system.m:
                    break
                masses = x.reshape(system.m, system.b)[:, 1 + 2 * self.d]
                index = int(np.argmin(masses))
                if masses[index] >= -self.options.sci_mass_tol:
                    break
                if time.perf_counter() >= deadline:
                    root_info = dict(success=False, message="timeout")
                    break
                active_set_updates += 1
                removed = self.contacts.pop(index)
                removed_contacts.append(
                    dict(pair=removed.pair, time=removed.t_star, mass=float(masses[index]))
                )
                system = self.matching_system(self.contacts)
                x = (
                    np.concatenate([c.to_vector() for c in self.contacts])
                    if self.contacts
                    else np.empty(0)
                )
            matching_seconds += time.perf_counter() - root_started
            trajectory = (
                recovered_trajectory
                if recovered_trajectory is not None
                else RelativeTrajectory.from_system(system, x, barycenter)
            )
            if verbose:
                print(f"ContactSolver: m={len(self.contacts)}, root={root_info['message']}")
            if not root_info["success"]:
                status, reason = "root_failure", root_info["message"]
                # The failed solve changed the trajectory since the last loop checks.
                checks = self._checks(system, x, trajectory)
                closest = trajectory.closest()
                break
        self.system, self.solution, self.trajectory = system, x, trajectory
        # All other exits retain the checks of this exact trajectory from the loop.
        solve_time = time.perf_counter() - started
        times, positions, velocities, accelerations = self.sample()
        self.trajectories = dict(
            positions=positions, velocities=velocities, accelerations=accelerations
        )
        m = len(self.contacts)
        return dict(
            trajectories=self.trajectories,
            times=times,
            contact_times=[c.t_star for c in self.contacts],
            contacts=self.contacts,
            metrics=dict(
                converged=status == "success",
                collision_backend=COLLISION_BACKEND,
                status=status,
                convergence_reason=reason,
                dimension=self.d,
                num_contacts=m,
                min_distance=closest[2],
                closest_pair=closest[0],
                closest_time=closest[1],
                checks=checks,
                roots=roots,
                reaction_recovery_enabled=self.options.sci_reaction_recovery,
                contact_selection=self.contact_selection,
                analytical_jacobian=self.options.use_analytical_jacobian,
                removed_contacts=removed_contacts,
                insertion_offsets=insertion_offsets,
                active_set_updates=active_set_updates,
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
