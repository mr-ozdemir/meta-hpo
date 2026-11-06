"""Unit tests of the BBPSO building blocks in operators.py: samplers, correlation, repair, root steps, moves.

Oracles: the docstrings of operators.levy_scale, cauchy_scale, correlation, repair, small_difference,
secant_point, iqi_point, steffensen_step, itp_point, move_two_leaders, move_cauchy, move_gauss, move_mixed
and move_levy_difference (Zhang, Lim & Liu 2023, equations 1-12, 13-14 and 15-22; Oliveira & Takahashi 2020
for ITP). Expected values are computed by hand; distribution checks use a seeded numpy Generator.

Input space: Float64 arrays and numbers of any size, which is unbounded.
Subset: Literal vectors with exact results, the boundary cases named in each docstring (d = 1, constant
vectors, equal h values, z on the walls, NaN and inf coordinates), 100,000 seeded draws for the two
samplers, and the bracketed root of cos t = t for the ITP step.
"""

import math

import numpy as np
import pytest

from metahpo import operators

DRAWS = 100_000


DIST_SEED = 0


def _itp_bracket(f, steps):
    """Runs `steps` ITP steps on [0, 1] for f with a sign change, updating the bracket by sign; returns the
    final bracket midpoint and the number of steps until the bracket is at most 2e-3 wide."""
    left, right, h_left, h_right = 0.0, 1.0, f(0.0), f(1.0)
    used = 0
    for k in range(steps):
        if right - left <= 2e-3:
            break
        t = operators.itp_point(left, right, h_left, h_right, k)
        h = f(t)
        if (h > 0) == (h_left > 0):
            left, h_left = t, h
        else:
            right, h_right = t, h
        used = k + 1
    return (left + right) / 2.0, used


def _x2m2(x):
    """x^2 - 2, whose positive root is sqrt(2)."""
    return x * x - 2.0


def test_levy_scale_should_be_one_over_z_squared():
    """Given z = 2 and z = -0.5, when the Levy scale is computed, then it is 0.25 and 4.

    Oracle: levy_scale docstring (1 / z^2).
    """
    expected = [0.25, 4.0]

    scale = operators.levy_scale(np.array([2.0, -0.5]))

    assert scale.tolist() == expected


def test_levy_scale_should_follow_the_levy_0_1_law_on_normal_draws():
    """Given 100,000 seeded standard normal draws, when they are turned into Levy draws, then the median is
    close to 2.198 and the share above 1 close to 0.683.

    Oracle: levy_scale docstring: 1 / Z^2 with Z ~ N(0, 1) is Levy(0, 1), whose median is 1 / 0.6745^2 and
    whose P(L > 1) equals P(|Z| < 1) = 0.6827.
    """
    rng = np.random.default_rng(DIST_SEED)

    draws = operators.levy_scale(rng.standard_normal(DRAWS))

    assert float(np.median(draws)) == pytest.approx(2.198, abs=0.05)
    assert float(np.mean(draws > 1.0)) == pytest.approx(0.6827, abs=0.005)


def test_cauchy_scale_should_map_the_quartiles_to_minus_one_zero_and_one():
    """Given u = 0.25, 0.5 and 0.75, when the Cauchy scale is computed, then it is -1, 0 and 1.

    Oracle: cauchy_scale docstring (tan(pi (u - 0.5))).
    """
    expected = [-1.0, 0.0, 1.0]

    scale = operators.cauchy_scale(np.array([0.25, 0.5, 0.75]))

    assert np.allclose(scale, expected, rtol=0.0, atol=1e-12)


def test_cauchy_scale_should_follow_the_cauchy_0_1_law_on_uniform_draws():
    """Given 100,000 seeded uniform draws, when they are turned into Cauchy draws, then about half have a
    magnitude above 1.

    Oracle: cauchy_scale docstring; for Cauchy(0, 1), P(|Ca| > 1) = 0.5.
    """
    rng = np.random.default_rng(DIST_SEED)

    draws = operators.cauchy_scale(rng.random(DRAWS))

    assert float(np.mean(np.abs(draws) > 1.0)) == pytest.approx(0.5, abs=0.005)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param([0.0, 1.0], [1.0, 0.0], -1.0, id="opposite"),
        pytest.param([0.0, 1.0], [0.0, 2.0], 1.0, id="same-direction"),
        pytest.param([0.3], [0.7], 0.0, id="d-1"),
        pytest.param([0.5, 0.5, 0.5], [0.1, 0.9, 0.4], 0.0, id="constant-vector"),
        pytest.param([0.0, 0.5, 1.0], [0.0, 1.0, 0.5], 0.5, id="partial"),
    ],
)
def test_correlation_should_give_the_pearson_coefficient_or_zero_without_spread(a, b, expected):
    """Given two coordinate vectors, when their correlation is computed, then it is the Pearson coefficient,
    or 0 for d = 1 and for a constant vector.

    Oracle: correlation docstring (MATLAB corr2; 0 when the denominator is at most eps).
    """
    value = operators.correlation(np.array(a), np.array(b))

    assert value == pytest.approx(expected, abs=1e-12)


def test_correlation_should_be_symmetric():
    """Given two random vectors, when the correlation is computed both ways, then the values agree.

    Oracle: correlation docstring; the Pearson coefficient is symmetric.
    """
    rng = np.random.default_rng(DIST_SEED)
    a, b = rng.random(7), rng.random(7)

    forward = operators.correlation(a, b)

    assert forward == pytest.approx(operators.correlation(b, a), abs=1e-15)


def test_repair_should_replace_non_finite_coordinates_and_clip_the_rest():
    """Given NaN, inf, -0.5, 1.5 and 0.3 with a reference of 0.75, when repaired, then the result is 0.75,
    0.75, 0, 1 and 0.3.

    Oracle: repair docstring.
    """
    expected = [0.75, 0.75, 0.0, 1.0, 0.3]

    repaired = operators.repair(np.array([math.nan, math.inf, -0.5, 1.5, 0.3]), np.full(5, 0.75))

    assert repaired.tolist() == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param(1e-20, 1.1e-20, False, id="tiny-but-apart"),
        pytest.param(1e-20, 1e-20 * (1 + 1e-13), True, id="tiny-and-close"),
        pytest.param(0.0, 0.0, True, id="both-zero"),
        pytest.param(0.0, 1e-300, False, id="zero-and-tiny"),
        pytest.param(5.0, 5.0 + 1e-12, True, id="large-and-close"),
    ],
)
def test_small_difference_should_be_a_relative_test_only(a, b, expected):
    """Given pairs at very different scales, when small_difference is asked, then only the relative gap
    counts.

    Oracle: small_difference docstring (abs(a - b) <= eps max(|a|, |b|), relative only).
    """
    result = operators.small_difference(a, b)

    assert result is expected


def test_secant_point_should_hit_the_root_of_a_linear_function():
    """Given h(x) = 4x - 1 at a = 0 and b = 1, when the secant point is computed, then it is the root 0.25.

    Oracle: secant_point docstring (equation 12).
    """
    expected = [0.25]

    point = operators.secant_point(np.array([0.0]), np.array([1.0]), -1.0, 3.0)

    assert point.tolist() == expected


def test_iqi_point_should_hit_the_root_when_x_is_quadratic_in_h():
    """Given h(x) = sqrt(4x) - 1 at x = 0.0625, 0.5625 and 1, when the IQI point is computed, then it is the
    root 0.25, because x = (h + 1)^2 / 4 is quadratic in h.

    Oracle: iqi_point docstring (equation 11, inverse quadratic interpolation).
    """
    expected = 0.25

    point = operators.iqi_point(np.array([0.0625]), np.array([0.5625]), np.array([1.0]), -0.5, 0.5, 1.0)

    assert float(point[0]) == pytest.approx(expected, abs=1e-12)


def test_secant_and_iqi_points_should_not_change_when_every_h_is_scaled_by_1e_minus_20():
    """Given the secant and IQI inputs above, when every h is multiplied by 1e-20, then both points stay the
    same.

    Oracle: both formulas are ratios of h values, which the bbpso module relies on for scale invariance.
    """
    a, b, c = np.array([0.0625]), np.array([0.5625]), np.array([1.0])

    scaled = (
        operators.secant_point(a, b, -1e-20, 3e-20),
        operators.iqi_point(a, b, c, -0.5e-20, 0.5e-20, 1e-20),
    )

    plain = (operators.secant_point(a, b, -1.0, 3.0), operators.iqi_point(a, b, c, -0.5, 0.5, 1.0))
    assert np.allclose(scaled, plain, rtol=1e-12, atol=0.0)


def test_steffensen_step_should_move_towards_the_root_of_x_squared_minus_two():
    """Given h(x) = x^2 - 2 at a = 1.5 and its probe a + h(a) = 1.75, when the step is computed, then it is
    within 0.1 of sqrt(2).

    Oracle: steffensen_step docstring (equations 13-14).
    """
    step = operators.steffensen_step(np.array([1.5]), _x2m2(1.5), _x2m2(1.5 + _x2m2(1.5)))

    assert abs(float(step[0]) - math.sqrt(2.0)) < 0.1


@pytest.mark.parametrize(
    ("h_left", "h_right", "expected"),
    [
        pytest.param(0.10, 0.12, -0.024, id="no-sign-change-beyond-left"),
        pytest.param(0.10, 1.0, 0.0888888888888889, id="no-sign-change-near-left"),
        pytest.param(0.3, 0.3, 0.5, id="equal-h-midpoint"),
        pytest.param(-1.0, 1.0, 0.5, id="symmetric-bracket"),
    ],
)
def test_itp_point_should_take_one_step_without_a_sign_condition(h_left, h_right, expected):
    """Given h at the ends of [0, 1], when one ITP step is taken, then t is the documented point: the regula
    falsi estimate truncated by 0.2 and projected into the radius 0.524 of the midpoint.

    Oracle: itp_point docstring (equations 15-22 with eps 1e-3, kappa1 0.2, kappa2 2, n0 1; equal h values
    give the midpoint). (0.10, 0.12): regula falsi -5, truncated to -4.8, projected to 0.5 - 0.524.
    """
    t = operators.itp_point(0.0, 1.0, h_left, h_right, 0)

    assert t == pytest.approx(expected, abs=1e-12)


def test_itp_point_should_find_a_bracketed_root_within_the_minmax_number_of_steps():
    """Given cos t - t on [0, 1], when ITP steps update the bracket by sign, then the bracket shrinks to
    2e-3 within n_half + n0 = 10 steps around the root 0.739085.

    Oracle: itp_point docstring and Oliveira & Takahashi (2020): ITP keeps the minmax bound of bisection.
    """
    root, steps = _itp_bracket(lambda t: math.cos(t) - t, 20)

    assert root == pytest.approx(0.739085, abs=1e-3)
    assert steps <= 10


def test_move_two_leaders_should_step_by_levy_times_the_gap_to_the_leaders_midpoint():
    """Given x = 0.2, leaders 0.4 and 0.8 (midpoint 0.6) and L = 2, when operation 1 moves x, then it lands
    at 1.0.

    Oracle: move_two_leaders docstring (equation 2).
    """
    expected = [1.0]

    moved = operators.move_two_leaders(np.array([0.2]), np.array([0.8]), np.array([0.4]), np.array([2.0]))

    assert np.allclose(moved, expected, rtol=0.0, atol=1e-12)


def test_move_cauchy_and_move_gauss_should_follow_their_formulas():
    """Given x = 0.25, center 0.5, Ca = -1, spread 0.25 and N = 2, when operations 2 and 3 move, then they
    give 0 and 1.

    Oracle: move_cauchy (x + Ca (center - x)) and move_gauss (center + spread N) docstrings.
    """
    x, center = np.array([0.25]), np.array([0.5])

    cauchy = operators.move_cauchy(x, center, np.array([-1.0]))

    gauss = operators.move_gauss(center, np.array([0.25]), np.array([2.0]))
    assert (cauchy.tolist(), gauss.tolist()) == ([0.0], [1.0])


def test_move_gauss_should_return_the_center_when_the_spread_is_zero():
    """Given a zero spread, when operation 3 moves, then the result is the center for any normal draw.

    Oracle: move_gauss docstring; the standard deviation |nbest - gbest| is 0 when both guides coincide.
    """
    center = np.array([0.3, 0.7])

    moved = operators.move_gauss(center, np.zeros(2), np.array([5.0, -3.0]))

    assert moved.tolist() == center.tolist()


def test_move_mixed_should_pick_the_cauchy_or_gauss_move_per_coordinate():
    """Given pick = (True, False), when operation 4 moves, then coordinate 0 is the Cauchy move and
    coordinate 1 the Gaussian move.

    Oracle: move_mixed docstring (equation 8).
    """
    x, center, spread = np.array([0.25, 0.25]), np.array([0.5, 0.5]), np.array([0.25, 0.25])
    cauchy, normal = np.array([-1.0, -1.0]), np.array([2.0, 2.0])

    moved = operators.move_mixed(x, center, spread, cauchy, normal, np.array([True, False]))

    assert moved.tolist() == [0.0, 1.0]


def test_move_levy_difference_should_keep_x_where_the_crossover_does_not_pick_the_donor():
    """Given cross = (True, False), center 0.5, L = 0.5 and x_q - x_r = 0.25, when operation 5 moves, then
    coordinate 0 is 0.625 and coordinate 1 keeps x.

    Oracle: move_levy_difference docstring (equations 9-10).
    """
    x = np.array([0.1, 0.1])

    moved = operators.move_levy_difference(
        x, np.full(2, 0.5), np.full(2, 0.75), np.full(2, 0.5), np.full(2, 0.5), np.array([True, False])
    )

    assert moved.tolist() == [0.625, 0.1]
