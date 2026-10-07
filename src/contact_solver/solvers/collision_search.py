"""Batched continuous minima, with the scalar isolator as a portable fallback."""

import warnings

import numpy as np

from .cubic_utils import distance_candidates

try:
    from . import _cubic_minima
except ImportError:
    _cubic_minima = None

COLLISION_BACKEND = "native" if _cubic_minima is not None else "python"
_warned_fallback = False


def normalized(coefficients, durations):
    """Map each local cubic to [0, 1] without forming duration cubed."""
    result = np.array(coefficients, dtype=float, copy=True)
    durations = np.asarray(durations, dtype=float)
    if (
        result.ndim != 3
        or result.shape[1] != 4
        or result.shape[2] < 1
        or durations.shape != (len(result),)
    ):
        raise ValueError("Expected (batch, 4, d) coefficients and one duration per cubic")
    if not np.isfinite(result).all() or not np.isfinite(durations).all() or np.any(durations < 0):
        raise ValueError("Expected finite coefficients and nonnegative finite durations")
    with np.errstate(over="raise", invalid="raise"):
        try:
            for power in range(1, 4):
                result[:, power:] *= durations[:, None, None]
        except FloatingPointError as exc:
            raise ValueError("Normalized cubic exceeds floating-point range") from exc
    return np.ascontiguousarray(result)


def minima(coefficients, durations, pruning=True):
    """Return normalized times and distances; pruned entries have distance +inf.

    With pruning, only the global minimum of the batch is guaranteed. Every
    surviving segment is checked at its endpoints and stationary candidates.
    The scalar fallback checks all segments, without sampled acceptance.
    """
    global _warned_fallback
    coefficients = normalized(coefficients, durations)
    times = np.empty(len(coefficients))
    distances = np.empty(len(coefficients))
    if _cubic_minima is not None:
        _cubic_minima.minima(
            coefficients, len(coefficients), coefficients.shape[-1], pruning, times, distances
        )
    else:
        if not _warned_fallback:
            warnings.warn(
                "Compiled collision checker unavailable; using the slower Python checker. "
                "Reinstall contact-solver with a C compiler to reproduce native runtimes.",
                RuntimeWarning,
                stacklevel=2,
            )
            _warned_fallback = True
        for k, cubic in enumerate(coefficients):
            times[k], distances[k] = min(distance_candidates(cubic, 1.0), key=lambda v: v[1])
    return times, distances
