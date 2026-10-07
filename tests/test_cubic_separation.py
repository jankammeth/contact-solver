"""Adversarial continuous-time checks independent of companion roots."""

import numpy as np
import pytest

from contact_solver.solvers.cubic_utils import (
    cubic_coefficients, distance_candidates, eval_cubic, eval_cubic_vel, unit_interval_candidates,
)
from contact_solver.solvers.prioritized_utils import PiecewiseCubicTrajectory


@pytest.mark.parametrize("duration", [1e-12, 1e-11, 1e-9, 1.0])
def test_short_cubic_reaches_its_endpoint(duration):
    p0, pf = np.array([-1., 0.]), np.array([1., 0.])
    zero = np.zeros(2)
    coefficients = cubic_coefficients(p0, zero, pf, zero, duration)
    np.testing.assert_allclose(eval_cubic(coefficients, duration), pf, atol=1e-14)
    # Normalized derivative checks avoid subtracting speeds of order 1/T.
    np.testing.assert_allclose(eval_cubic_vel(coefficients, duration) * duration, zero, atol=1e-14)
    assert min(d for _, d in distance_candidates(coefficients, duration)) < 1e-12


@pytest.mark.parametrize("duration", [1e-110, 1e-12, 1.0])
def test_short_cubic_preserves_nonzero_endpoint_velocities(duration):
    # T**3 underflows for the shortest interval, but its coefficients are finite.
    p0, pf = np.zeros(2), np.array([2., -1.]) * duration
    v0, vf = np.array([1., 2.]), np.array([-1., 3.])
    coefficients = cubic_coefficients(p0, v0, pf, vf, duration)
    np.testing.assert_allclose(eval_cubic(coefficients, duration) / duration, pf / duration, atol=1e-14)
    np.testing.assert_allclose(eval_cubic_vel(coefficients, 0), v0, atol=1e-14)
    np.testing.assert_allclose(eval_cubic_vel(coefficients, duration), vf, atol=1e-14)


@pytest.mark.parametrize("duration", [0., -1., np.nan, np.inf, 1e-200])
def test_invalid_or_unrepresentable_cubic_fails_explicitly(duration):
    with pytest.raises(ValueError, match="Cubic"):
        cubic_coefficients(np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2), duration)


def test_sampling_selects_the_correct_short_segment():
    # A tolerance on segment selection used to extrapolate the first segment.
    zero, one = np.zeros(2), np.ones(2)
    duration = 1e-12
    first = cubic_coefficients(zero, zero, zero, zero, duration)
    second = cubic_coefficients(zero, zero, one, zero, duration)
    path = PiecewiseCubicTrajectory([(first, 0., duration), (second, duration, 2*duration)], 2*duration)
    np.testing.assert_allclose(path.eval_pos(1.5*duration), .5*one)
    np.testing.assert_allclose(path.eval_pos(2*duration), one)


@pytest.mark.parametrize("curvature", [0.0, 1e-23, -1e-23, 1e-12])
@pytest.mark.parametrize("duration", [1e-6, 1.0, 1e6])
def test_nearly_linear_collision(curvature, duration):
    c = np.array([[-2.0, 0.1], [4.0, 0.0], [curvature, 0.0], [0.0, 0.0]])
    c /= np.array([1, duration, duration**2, duration**3])[:, None]
    t, distance = min(distance_candidates(c, duration, 2.0), key=lambda item: item[1])
    assert distance == pytest.approx(0.1, abs=1e-12)
    assert (t - 2.0) / duration == pytest.approx(0.5, abs=1e-9)


def test_repeated_roots_and_cubic_multiple_minima():
    roots = unit_interval_candidates(
        np.polynomial.polynomial.polyfromroots([0.2, 0.2, 0.5, 0.8, 0.8])
    )
    for target in [0.2, 0.5, 0.8]:
        assert min(abs(np.array(roots) - target)) < 1e-7
    c = np.zeros((4, 3))
    c[:, 2] = np.polynomial.polynomial.polyfromroots([0.2, 0.5, 0.8])
    candidates = distance_candidates(c, 1.0)
    for target in [0.2, 0.5, 0.8]:
        assert min(distance for t, distance in candidates if abs(t - target) < 1e-7) < 1e-10


def test_random_cubics_against_dense_upper_bound():
    rng = np.random.default_rng(78)
    grid = np.linspace(0, 1, 10001)
    for _ in range(30):
        c = rng.normal(size=(4, 5))
        reference = np.linalg.norm(sum(c[k] * grid[:, None] ** k for k in range(4)), axis=1).min()
        minimum = min(v for _, v in distance_candidates(c, 1.0))
        assert minimum <= reference + 1e-11
