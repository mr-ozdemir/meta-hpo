"""Boundary tests of the ABC operators: both sides of every threshold in operators.py.

Oracle: the operators docstrings (neighbour_value clipped to [0, 1]; fitness 1 / (1 + f) for f >= 0 and
1 + |f| otherwise; onlooker p = 0.9 * fit / max(fit) + 0.1 with all-zero fitness giving equal
probabilities), the original hyp_optimizer ABC.py and Karaboga & Basturk (2007).

Inputs are dyadic (sums of powers of two), so every expected value is an exact binary float computed by
hand and the exact matcher is the right one. TINY = 2**-53 is half an ulp of 1.0, the smallest step that
stays representable next to 0.5 and 1.0; at 0 the closest floats are the subnormals -5e-324 and 5e-324
(SMALLEST_SUBNORMAL).

Thresholds covered: neighbour_value clip at 0 and at 1 (closest float outside, exactly on, closest float
inside); the fitness branch f >= 0 (a small negative value, 0.0 and -0.0); the onlooker guard top <= 0 on
its not-taken side with the smallest top in the normal float range (its taken side, all-zero fitness, is
already in tests/test_abc.py).

Not in this module: onlooker_probabilities with a subnormal top (for example [0.0, 5e-324]) returns
[0.1, 1.1] instead of the formula value [0.1, 1.0]. That is reported as a library defect, so no test here
asserts it. The infeasible-requirement inventory (P-T4-07) is in the module docstring of
test_abc_boundaries.py.

Input space: Three float64 values for neighbour_value and float64 arrays of any length for fitness and
onlooker_probabilities.
Subset: Both sides of every threshold in operators.py.
"""

import numpy as np
import pytest

from metahpo.operators import fitness, fitness_scalar, neighbour_value, onlooker_probabilities

TINY = 2.0**-53
SMALLEST_SUBNORMAL = 5e-324
"""math.nextafter(0.0, 1.0): the closest floats to 0 are -5e-324 and 5e-324, and with phi = 1 the raw value
2 * x_ij - x_kj is exact in subnormal arithmetic."""
SMALL_NEGATIVE = -(2.0**-20)
PHI_ONE = 1.0


@pytest.mark.parametrize(
    ("x_ij", "x_kj", "expected"),
    [
        pytest.param(
            0.0, SMALLEST_SUBNORMAL, 0.0, id="raw-minus-smallest-subnormal-just-below-zero-is-clipped"
        ),
        pytest.param(0.5, 1.0, 0.0, id="raw-exactly-zero-stays-zero"),
        pytest.param(
            SMALLEST_SUBNORMAL,
            SMALLEST_SUBNORMAL,
            SMALLEST_SUBNORMAL,
            id="raw-smallest-subnormal-just-above-zero-is-kept",
        ),
        pytest.param(0.5, TINY, 1.0 - TINY, id="raw-one-minus-2-to-minus-53-just-below-one-is-kept"),
        pytest.param(0.5, 0.0, 1.0, id="raw-exactly-one-stays-one"),
        pytest.param(1.0 - TINY, 1.0 - 4 * TINY, 1.0, id="raw-next-float-above-one-is-clipped"),
    ],
)
def test_neighbour_value_should_clip_only_outside_the_unit_interval_when_raw_value_is_near_an_end(
    x_ij, x_kj, expected
):
    """Given x_ij, x_kj and phi = 1 that put the raw value near 0 or 1, when neighbour_value runs, then it
    clips only outside.

    Oracle: neighbour_value docstring 'v_ij = x_ij + phi * (x_ij - x_kj) ... clipped to [0, 1]' and the bound
    clipping of the original ABC.py. Hand calculation with phi = 1: raw = 2 * x_ij - x_kj, which gives
    -5e-324, 0, 5e-324, 1 - 2**-53, 1 and 1 + 2**-52 (the next float above 1), all exact in binary64. Each
    clip threshold is approached from both sides by the closest representable values (P-T1-40, P-T1-42).
    Postcondition: none (pure function).
    """
    phi = PHI_ONE

    result = neighbour_value(x_ij, x_kj, phi)

    assert result == expected


FITNESS_BRANCH_CASES = [
    pytest.param(SMALL_NEGATIVE, 1.0 + 2.0**-20, id="small-negative-takes-one-plus-abs-branch"),
    pytest.param(0.0, 1.0, id="zero-takes-one-over-one-plus-f-branch"),
    pytest.param(-0.0, 1.0, id="negative-zero-counts-as-zero"),
]


@pytest.mark.parametrize(("f", "expected"), FITNESS_BRANCH_CASES)
def test_fitness_scalar_should_switch_branch_exactly_at_zero_when_f_is_on_either_side(f, expected):
    """Given f slightly below 0, at 0.0 or at -0.0, when fitness_scalar runs, then the branch for that side
    gives the value.

    Oracle: fitness docstring '1 / (1 + f) for f >= 0, 1 + |f| otherwise' (Karaboga & Basturk 2007, original
    ABC.py calculate_fitness). Both formulas equal 1 - f to first order at 0 and differ by about f**2, so the
    negative input closest to 0 that still tells them apart needs f**2 above the resolution of 1.0:
    f = -2**-20 gives 1 + 2**-20 here, while 1 / (1 + f) would give about 1 + 2**-20 + 2**-40. 0.0 and -0.0
    give 1 / 1; -0.0 >= 0 holds in IEEE 754, an implicit boundary of the language (P-T1-42).
    Postcondition: none (pure function).
    """

    result = fitness_scalar(f)

    assert result == expected


@pytest.mark.parametrize(("f", "expected"), FITNESS_BRANCH_CASES)
def test_fitness_should_switch_branch_exactly_at_zero_when_an_array_element_is_on_either_side(f, expected):
    """Given a one-element array with f slightly below 0, at 0.0 or at -0.0, when fitness runs, then that
    branch gives the value.

    Oracle: fitness docstring (same formula and hand calculation as the fitness_scalar cases); the array form
    has its own branch site (np.where), so it is checked separately. Postcondition: none (pure function).
    """
    values = np.array([f])

    result = fitness(values)

    assert result.tolist() == [expected]


def test_onlooker_probabilities_should_use_the_formula_when_the_largest_fitness_is_tiny_but_positive():
    """Given fitness [0, 2**-1000], when onlooker_probabilities runs, then the formula gives [0.1, 1.0], not
    the all-zero ones.

    Oracle: onlooker_probabilities docstring 'p_i = 0.9 * fit_i / max(fit) + 0.1; all-zero fitness gives equal
    probabilities'. 2**-1000 is a positive top close to 0 that is still a normal float, so it is the
    not-taken side of the guard top <= 0 (P-T1-40). Hand calculation: 0.9 * 0 / top + 0.1 = 0.1 and
    0.9 * top / top + 0.1 = 1.0. Postcondition: none (pure function).
    """
    fit = np.array([0.0, 2.0**-1000])

    result = onlooker_probabilities(fit)

    assert result.tolist() == [0.1, 1.0]
