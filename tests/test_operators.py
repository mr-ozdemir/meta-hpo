"""Literal-input tests of the ABC operator formulas in metahpo.operators.

Oracles: Karaboga & Basturk (2007), the original hyp_optimizer ABC.py (calculate_neighbour_solution,
calculate_fitness, calculate_probabilities) and the docstrings of metahpo.operators. Every expected value is a
hand-computed literal; each input is chosen so that the listed mutants of the formula give a different result.

Unspecified, therefore not asserted: neighbour_value for phi outside [-1, 1]; fitness and fitness_scalar for
NaN and -inf (the optimizer maps non-finite scores to +inf before calling them); onlooker_probabilities for
negative fitness values.

Equivalent mutants, not claimed as kills: '<' -> '<=' and '>' -> '>=' in the clip of neighbour_value; '>=' ->
'>' at f == 0 and 1 + abs(f) -> 1 - f for f < 0 in fitness/fitness_scalar; removing the explicit '+inf gives
0' special case of fitness_scalar, because 1.0 / (1.0 + inf) is already 0.0; dropping np.maximum in the array
fitness (1.0 / (1.0 + np.maximum(f, 0.0)) -> 1.0 / (1.0 + f)), because np.where discards the unselected branch
and np.errstate silences its division warnings.

Input space: Three float64 values for neighbour_value and float64 arrays of any length for fitness and
onlooker_probabilities.
Subset: Discriminating literal inputs for each formula.
"""

import math

import numpy as np
import pytest

from metahpo.operators import fitness, fitness_scalar, neighbour_value, onlooker_probabilities

NEIGHBOUR_IN_RANGE_CASES = [
    pytest.param(0.5, 0.25, 0.5, 0.625, id="dyadic-kills-div-phi-minus-to-plus-and-wrong-base"),
    pytest.param(0.5, 0.2, 0.5, pytest.approx(0.65), id="non-dyadic-kills-minus-to-plus"),
    pytest.param(0.5, 0.25, -0.5, 0.375, id="negative-phi-kills-abs-phi"),
    pytest.param(0.75, 0.25, -1.0, 0.25, id="phi-minus-one-edge-returns-x_kj"),
    pytest.param(0.75, 0.25, 0.0, 0.75, id="phi-zero-returns-x_ij-kills-div-phi"),
]

NEIGHBOUR_CLIP_CASES = [
    pytest.param(0.9, 0.1, 1.0, 1.0, id="clips-above-one-raw-1.7-kills-missing-upper-clip"),
    pytest.param(0.1, 0.9, 1.0, 0.0, id="clips-below-zero-raw-minus-0.7-kills-missing-lower-clip"),
]

FITNESS_CASES = [
    pytest.param(1.0, 0.5, id="positive-unit-kills-1-over-2-plus-f"),
    pytest.param(3.0, 0.25, id="positive-kills-f-over-1-plus-f-and-1-over-1-plus-f-squared"),
    pytest.param(-2.0, 3.0, id="negative-branch-1-plus-abs-kills-1-over-1-plus-abs-and-1-plus-f"),
    pytest.param(0.0, 1.0, id="branch-boundary-zero"),
    pytest.param(math.inf, 0.0, id="plus-inf-implicit-bound-is-zero"),
]

ONLOOKER_CASES = [
    pytest.param(np.array([2.0, 1.0]), [1.0, 0.55], id="max-not-one-kills-fit-over-sum-and-missing-division"),
    pytest.param(np.zeros(2), [1.0, 1.0], id="all-zero-gives-ones-kills-top-lt-zero-zeros-like-and-half"),
    pytest.param(np.array([3.0, 3.0]), [1.0, 1.0], id="tied-nonzero-gives-ones-consistent-with-all-zero-row"),
    pytest.param(np.array([0.0, 4.0]), [0.1, 1.0], id="zero-fitness-floor-is-0.1-kills-missing-plus-0.1"),
]


@pytest.mark.parametrize(("x_ij", "x_kj", "phi", "expected"), NEIGHBOUR_IN_RANGE_CASES)
def test_neighbour_value_should_follow_x_ij_plus_phi_times_difference_when_result_is_in_range(
    x_ij, x_kj, phi, expected
):
    """Given an in-range neighbour, when neighbour_value runs, then it returns x_ij + phi * (x_ij - x_kj).

    Oracle: Karaboga & Basturk (2007) v_ij = x_ij + phi_ij (x_ij - x_kj), phi in [-1, 1]; ABC.py
    calculate_neighbour_solution; the neighbour_value docstring. Hand calculation for (0.5, 0.25, 0.5):
    0.5 + 0.5 * 0.25 = 0.625, while x_ij + x_kj gives 0.875, '/phi' gives 1.0, '-phi' or base x_kj give 0.375.
    For (0.5, 0.2, 0.5) the '-' -> '+' mutant gives 0.85; for phi = -0.5 abs(phi) gives 0.625; phi = -1 is the
    lower edge of the documented phi range and reduces to x_kj; phi = 0 is neutral and '/phi' raises.
    (0.5, 0.2, 1.0) is deliberately not used because it cannot tell '*phi' from '/phi'. Postcondition: none
    (pure function).
    """

    result = neighbour_value(x_ij, x_kj, phi)

    assert result == expected


@pytest.mark.parametrize(("x_ij", "x_kj", "phi", "expected"), NEIGHBOUR_CLIP_CASES)
def test_neighbour_value_should_clip_to_unit_interval_when_raw_value_leaves_it(x_ij, x_kj, phi, expected):
    """Given a raw neighbour outside [0, 1], when neighbour_value runs, then it returns the nearer bound.

    Oracle: ABC.py calculate_neighbour_solution clips to LOWER_BOUND/UPPER_BOUND, and the neighbour_value
    docstring says 'clipped to [0, 1]'. Hand calculation: 0.9 + 1.0 * 0.8 = 1.7 -> 1.0 and
    0.1 + 1.0 * (-0.8) = -0.7 -> 0.0. A missing upper clip gives 1.7, a missing lower clip gives -0.7, and
    swapped clip constants give the bounds the wrong way round. '<' -> '<=' and '>' -> '>=' in the clip are
    equivalent mutants and are not claimed. Postcondition: none (pure function).
    """

    result = neighbour_value(x_ij, x_kj, phi)

    assert result == expected


@pytest.mark.parametrize(("f", "expected"), FITNESS_CASES)
def test_fitness_scalar_should_follow_original_fitness_formula_on_each_branch(f, expected):
    """Given one minimisation value, when fitness_scalar runs, then it returns 1/(1+f) for f >= 0 else 1+|f|.

    Oracle: ABC.py calculate_fitness (1 / (fun + 1) if fun >= 0 else 1 + abs(fun)), Karaboga & Basturk (2007),
    and the fitness docstring '+inf gives 0'. 3.0 is decisive: f/(1+f) gives 0.75 and 1/(1+f*f) gives 0.1
    there, while f/(1+f) survives f = 1.0. 1.0 kills 1/(2+f) (1/3); -2.0 kills 1/(1+abs(f)) (1/3) and 1+f
    (-1.0). 0.0 is the branch boundary (P-T1-40) and +inf the implicit bound (P-T1-42). Postcondition: none.
    """

    result = fitness_scalar(f)

    assert result == expected


@pytest.mark.parametrize(("f", "expected"), FITNESS_CASES)
def test_fitness_should_follow_original_fitness_formula_on_each_branch_for_one_element_array(f, expected):
    """Given a one-element array, when fitness runs, then its element equals the same literal as
    fitness_scalar.

    Oracle: ABC.py calculate_fitness and the fitness docstring; the rows and their mutant reasoning are the
    same table as the fitness_scalar test, so the array path is checked against the same hand-computed
    literals (a wrong +inf case in the array version would give nan or a nonzero value). Postcondition: none.
    """

    result = fitness(np.array([f]))

    assert result.tolist() == [expected]


def test_fitness_should_select_the_branch_per_element_when_one_array_mixes_both_branches_and_inf():
    """Given positive, negative, zero and +inf values in one array, when fitness runs, then each element is
    right.

    Oracle: ABC.py calculate_fitness applied element by element, hand-computed as [0.5, 0.25, 3.0, 1.0, 0.0].
    The mixed vector is decisive for array-only mutants that take the branch condition from one element
    (np.where(f.flat[0] >= 0, ...)) or from the whole array (np.where(np.all(f >= 0), ...)) instead of per
    element: on a one-element array both agree with the per-element rule, so only this test catches them.
    Site: the np.where condition of fitness. It adds 3.0 to the existing [0, 1, -2, inf] case. The comparison
    with fitness_scalar only checks that both paths agree; the literal list is the oracle.
    Postcondition: none (pure function).
    """
    values = np.array([1.0, 3.0, -2.0, 0.0, math.inf])

    result = fitness(values)

    assert result.shape == (5,)
    assert result.tolist() == [0.5, 0.25, 3.0, 1.0, 0.0]
    assert result.tolist() == list(map(fitness_scalar, values.tolist()))


@pytest.mark.parametrize(("fit", "expected"), ONLOOKER_CASES)
def test_onlooker_probabilities_should_follow_point_nine_fit_over_max_plus_point_one(fit, expected):
    """Given food-source fitness values, when onlooker_probabilities runs, then p_i = 0.9 * fit_i / max + 0.1.

    Oracle: ABC.py calculate_probabilities (0.9 * (fitness / max_fit)) + 0.1 and the docstring 'all-zero
    fitness gives equal probabilities'. [2, 1] is used instead of [0.5, 1.0] because max == 1 hides the
    division: on [2, 1] fit/sum gives [0.667, 0.333], 0.9*fit/sum+0.1 gives [0.7, 0.4], dropping '/max' or
    using min gives [1.9, 1.0], dropping '+0.1' gives [0.9, 0.45], swapping 0.9 and 0.1 gives [1.0, 0.95]. [0,
    4] puts the weakest source exactly on the 0.1 floor and kills a dropped '+0.1'.
    All-zero fitness: the original ABC.py divides by max_fit = 0 there (nan), and the docstring only requires
    equal probabilities. The literal 1.0 comes from the library spec 'all-zero -> ones' (also asserted by the
    repository's own tests/test_abc.py) and is the value the formula gives whenever all fitness values tie:
    every source is then a maximum source, 0.9 * fit_i / max = 0.9 and p_i = 1.0, as the [3, 3] row checks by
    hand (0.9 * 3 / 3 + 0.1 = 1.0). The all-zero row is the other side of the 'top <= 0' branch and kills
    'top <= 0' -> 'top < 0' (nan), ones_like -> zeros_like ([0, 0]) and full_like(fit, 0.5) ([0.5, 0.5]).
    Postcondition: none (pure function).
    """

    result = onlooker_probabilities(fit)

    assert result.shape == (2,)
    assert result.tolist() == pytest.approx(expected, abs=1e-12)
