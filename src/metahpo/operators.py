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


def pso_velocity(
    v: FloatArray,
    x: FloatArray,
    p: FloatArray,
    g: FloatArray,
    inertia: float,
    cognitive: float,
    social: float,
    r1: FloatArray,
    r2: FloatArray,
    max_velocity: float,
) -> FloatArray:
    """PSO velocity of Shi & Eberhart (1998): w * v + c1 * r1 * (p - x) + c2 * r2 * (g - x), clipped to
    [-max_velocity, max_velocity] coordinate by coordinate."""
    raw = inertia * v + cognitive * r1 * (p - x) + social * r2 * (g - x)
    return np.asarray(np.clip(raw, -max_velocity, max_velocity), dtype=np.float64)


def pso_absorb(x: FloatArray, v: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Absorbing walls: positions clipped to [0, 1], and the velocity of every clipped coordinate set to 0."""
    outside = (x < 0.0) | (x > 1.0)
    clipped = np.asarray(np.clip(x, 0.0, 1.0), dtype=np.float64)
    return clipped, np.asarray(np.where(outside, 0.0, v), dtype=np.float64)


def constriction(phi1: float, phi2: float) -> float:
    """Constriction coefficient of Clerc & Kennedy (2002): chi = 2 / |2 - phi - sqrt(phi^2 - 4 phi)| with
    phi = phi1 + phi2, defined for finite phi > 4; constriction(2.05, 2.05) = 0.7298..."""
    phi = phi1 + phi2
    if not (math.isfinite(phi) and phi > 4.0):
        raise ValueError(f"constriction needs phi1 + phi2 > 4, got {phi}")
    return 2.0 / abs(2.0 - phi - math.sqrt(phi * phi - 4.0 * phi))
