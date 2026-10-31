"""ABC building blocks as plain numpy functions on arrays (Karaboga & Basturk, 2007)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def neighbour_value(x_ij: float, x_kj: float, phi: float) -> float:
    """v_ij = x_ij + phi * (x_ij - x_kj), phi in [-1, 1], clipped to [0, 1] like the original ABC."""
    v = x_ij + phi * (x_ij - x_kj)
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def fitness(f: FloatArray) -> FloatArray:
    """Fitness of minimisation values: 1 / (1 + f) for f >= 0, 1 + |f| otherwise; +inf gives 0."""
    f = np.asarray(f, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore"):
        return np.where(f >= 0, 1.0 / (1.0 + np.maximum(f, 0.0)), 1.0 + np.abs(f))


def fitness_scalar(f: float) -> float:
    """`fitness` for one value without numpy overhead."""
    if f >= 0:
        return 0.0 if f == math.inf else 1.0 / (1.0 + f)
    return 1.0 + abs(f)


def onlooker_probabilities(fit: FloatArray) -> FloatArray:
    """p_i = 0.9 * fit_i / max(fit) + 0.1; all-zero fitness gives equal probabilities."""
    top = float(np.max(fit))
    if top <= 0.0:
        return np.ones_like(fit)
    return 0.9 * (fit / top) + 0.1
