"""Compare the compiled, pruned search with independent scalar isolation."""

import numpy as np
import pytest

from contact_solver.solvers import collision_search as search
from contact_solver.solvers.cubic_utils import distance_candidates


@pytest.mark.parametrize("dimension", [2, 3, 5, 10])
@pytest.mark.parametrize("duration", [1e-6, 1.0, 1e6])
def test_batch_and_pruning_preserve_continuous_minimum(dimension, duration):
    rng = np.random.default_rng(27)
    c = rng.normal(size=(25, 4, dimension))
    # Most intervals are far away and should be prunable.
    c[5:, 0] += 30
    c /= np.array([1, duration, duration**2, duration**3])[None, :, None]
    reference = np.array([min(d for _, d in distance_candidates(v, duration)) for v in c])
    times, full = search.minima(c, np.full(len(c), duration), pruning=False)
    np.testing.assert_allclose(full, reference, rtol=1e-12, atol=1e-12)
    times, pruned = search.minima(c, np.full(len(c), duration))
    assert pruned.min() == pytest.approx(reference.min(), rel=1e-12, abs=1e-12)
    assert times[np.argmin(pruned)] >= 0
    assert times[np.argmin(pruned)] <= 1
    if search.COLLISION_BACKEND == "native":
        assert np.isinf(pruned).any()


@pytest.mark.parametrize("roots", [[0, 0, 1], [0.2, 0.2, 0.8], [0.2, 0.5, 0.8]])
def test_repeated_roots_and_endpoint_minima(roots):
    c = np.zeros((1, 4, 2))
    c[0, :, 0] = np.polynomial.polynomial.polyfromroots(roots)
    _, distances = search.minima(c, [1])
    assert distances[0] < 1e-12


@pytest.mark.parametrize("curvature", [0, 1e-23, -1e-23, 1e-12])
def test_nearly_linear_and_constant_distance(curvature):
    c = np.array(
        [[[-2, 0.1], [4, 0], [curvature, 0], [0, 0]], [[1, 0], [0, 0], [0, 0], [0, 0]]], dtype=float
    )
    times, distances = search.minima(c, [1, 1], pruning=False)
    np.testing.assert_allclose(distances, [0.1, 1], atol=1e-12)
    assert times[0] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("duration", [0, 1e-12])
def test_zero_and_short_intervals(duration):
    c = np.zeros((1, 4, 2))
    c[0, 0, 0] = -1
    c[0, 1, 0] = 2 / duration if duration else 2
    _, distances = search.minima(c, [duration])
    assert distances[0] == pytest.approx(0 if duration else 1, abs=1e-12)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1])
def test_invalid_duration_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        search.minima(np.zeros((1, 4, 2)), [bad])


def test_native_buffer_sizes_and_types_checked():
    if search._cubic_minima is None:
        pytest.skip("No C compiler available")
    for c in (np.zeros((1, 4, 2), dtype=np.float32), np.zeros((1, 4, 1))):
        with pytest.raises(ValueError):
            search._cubic_minima.minima(c, 1, 2, True, np.empty(1), np.empty(1))


def test_python_fallback_retains_continuous_search(monkeypatch):
    monkeypatch.setattr(search, "_cubic_minima", None)
    monkeypatch.setattr(search, "_warned_fallback", False)
    c = np.zeros((1, 4, 2))
    c[0, :, 0] = np.polynomial.polynomial.polyfromroots([0.2, 0.5, 0.8])
    with pytest.warns(RuntimeWarning, match="slower Python"):
        _, distances = search.minima(c, [1])
    assert distances[0] < 1e-12
