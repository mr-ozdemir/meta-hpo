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


def levy_scale(z: FloatArray) -> FloatArray:
    """Levy(0, 1) draws from standard normal draws z: 1 / z^2 (Zhang, Lim & Liu 2023, equation 3); the caller
    excludes z == 0."""
    z = np.asarray(z, dtype=np.float64)
    return np.asarray(1.0 / (z * z), dtype=np.float64)


def cauchy_scale(u: FloatArray) -> FloatArray:
    """Cauchy(0, 1) draws from uniform draws u in [0, 1): tan(pi * (u - 0.5)) (equation 5)."""
    u = np.asarray(u, dtype=np.float64)
    return np.asarray(np.tan(np.pi * (u - 0.5)), dtype=np.float64)


def correlation(a: FloatArray, b: FloatArray, eps: float = 1e-12) -> float:
    """Pearson correlation of two coordinate vectors (MATLAB corr2, equation 1); 0 when the denominator is at
    most eps, which covers a constant vector and d = 1; clipped to [-1, 1]."""
    ac = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    bc = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    denom = math.sqrt(float(np.sum(ac * ac)) * float(np.sum(bc * bc)))
    if denom <= eps:
        return 0.0
    return min(1.0, max(-1.0, float(np.sum(ac * bc)) / denom))


def repair(z: FloatArray, reference: FloatArray) -> FloatArray:
    """NaN or infinite coordinates of z replaced by those of reference, the rest clipped to [0, 1]."""
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(z)
    clipped = np.clip(np.where(finite, z, 0.0), 0.0, 1.0)
    return np.asarray(np.where(finite, clipped, reference), dtype=np.float64)


def small_difference(a: float, b: float, eps: float = 1e-12) -> bool:
    """abs(a - b) <= eps * max(abs(a), abs(b)): a relative test only, so scaling a and b changes nothing."""
    return abs(a - b) <= eps * max(abs(a), abs(b))


def secant_point(a: FloatArray, b: FloatArray, h_a: float, h_b: float) -> FloatArray:
    """Secant step at level 0 through (a, h_a) and (b, h_b): a - h_a * (a - b) / (h_a - h_b) (equation 12)."""
    return np.asarray(a - h_a * (a - b) / (h_a - h_b), dtype=np.float64)


def iqi_point(a: FloatArray, b: FloatArray, c: FloatArray, h_a: float, h_b: float, h_c: float) -> FloatArray:
    """Inverse quadratic interpolation at level 0 through (a, h_a), (b, h_b) and (c, h_c) (equation 11)."""
    w_a = h_b * h_c / ((h_a - h_b) * (h_a - h_c))
    w_b = h_a * h_c / ((h_b - h_a) * (h_b - h_c))
    w_c = h_a * h_b / ((h_c - h_a) * (h_c - h_b))
    return np.asarray(w_a * a + w_b * b + w_c * c, dtype=np.float64)


def steffensen_step(a: FloatArray, h: float, h_probe: float) -> FloatArray:
    """Steffensen step a - h^2 / (h_probe - h); h at a, h_probe at the probe a + h (equations 13-14)."""
    return np.asarray(a - h * h / (h_probe - h), dtype=np.float64)


def itp_point(
    left: float,
    right: float,
    h_left: float,
    h_right: float,
    k: int,
    eps: float = 1e-3,
    kappa1: float = 0.2,
    kappa2: float = 2.0,
    n0: int = 1,
) -> float:
    """One ITP step of Oliveira & Takahashi (2020) on [left, right] (equations 15-22): regula falsi, truncated
    towards the midpoint by kappa1 (right - left)^kappa2, projected into the midpoint's minmax radius
    eps 2^(n_half + n0 - k) - (right - left) / 2 with n_half = ceil(log2(1 / (2 eps))). Equal h values give
    the midpoint; no sign condition is needed for a single step."""
    mid = (left + right) / 2.0
    xf = mid if small_difference(h_left, h_right) else (right * h_left - left * h_right) / (h_left - h_right)
    sigma = 1.0 if mid - xf > 0 else -1.0 if mid - xf < 0 else 0.0
    xt = xf + sigma * min(kappa1 * (right - left) ** kappa2, abs(mid - xf))
    n_half = max(0, math.ceil(math.log2(1.0 / (2.0 * eps))))
    radius = max(0.0, eps * 2.0 ** (n_half + n0 - k) - (right - left) / 2.0)
    return float(mid - sigma * min(radius, abs(xt - mid)))


def move_two_leaders(x: FloatArray, g: FloatArray, s: FloatArray, levy: FloatArray) -> FloatArray:
    """Operation 1 (equation 2): x + L * ((s + g) / 2 - x)."""
    return np.asarray(x + levy * ((s + g) / 2.0 - x), dtype=np.float64)


def move_cauchy(x: FloatArray, center: FloatArray, cauchy: FloatArray) -> FloatArray:
    """Operation 2 (equation 4): x + Ca * (center - x) with center = (nbest + gbest) / 2."""
    return np.asarray(x + cauchy * (center - x), dtype=np.float64)


def move_gauss(center: FloatArray, spread: FloatArray, normal: FloatArray) -> FloatArray:
    """Operation 3 (equations 6-7): center + |nbest - gbest| * N(0, 1)."""
    return np.asarray(center + spread * normal, dtype=np.float64)


def move_mixed(
    x: FloatArray,
    center: FloatArray,
    spread: FloatArray,
    cauchy: FloatArray,
    normal: FloatArray,
    pick: NDArray[np.bool_],
) -> FloatArray:
    """Operation 4 (equation 8): operation 2 where pick is True, operation 3 elsewhere."""
    moved = np.where(pick, move_cauchy(x, center, cauchy), move_gauss(center, spread, normal))
    return np.asarray(moved, dtype=np.float64)


def move_levy_difference(
    x: FloatArray,
    center: FloatArray,
    x_q: FloatArray,
    x_r: FloatArray,
    levy: FloatArray,
    cross: NDArray[np.bool_],
) -> FloatArray:
    """Operation 5 (equations 9-10): center + L * (x_q - x_r) where cross is True, x elsewhere."""
    return np.asarray(np.where(cross, center + levy * (x_q - x_r), x), dtype=np.float64)
