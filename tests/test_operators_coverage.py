"""Branch-coverage tests of metahpo.operators, derived from the control flow graphs below.

Control flow graphs (course notation: nodes are the real line numbers of src/metahpo/operators.py,
straight-line code is merged, every decision has one T and one F edge, s and e are entry and exit; a
conditional expression on one line is split into the decision nodes a, b, ...):

neighbour_value, operators.py 14-16
    s   -> 15
    15  -> 16a            v = x_ij + phi * (x_ij - x_kj)
    16a -T-> 16x -> e     v < 0.0: return 0.0
    16a -F-> 16b
    16b -T-> 16y -> e     v > 1.0: return 1.0
    16b -F-> 16z -> e     return v

fitness, operators.py 19-23
    s   -> 21-22 -> 23    asarray, errstate
    23  -T-> 23x -> e     np.where condition f >= 0 for an element: 1 / (1 + max(f, 0))
    23  -F-> 23y -> e     element with f < 0: 1 + |f|

fitness_scalar, operators.py 26-30
    s   -> 28
    28  -T-> 29           f >= 0
    28  -F-> 30 -> e      return 1 + |f|
    29  -T-> 29x -> e     f == inf: return 0.0
    29  -F-> 29y -> e     return 1 / (1 + f)

onlooker_probabilities, operators.py 33-38
    s   -> 35 -> 36       top = max(fit)
    36  -T-> 37 -> e      top <= 0: ones
    36  -F-> 38 -> e      0.9 * fit / top + 0.1

Test requirements (branch coverage, one per T/F edge; for the vectorised np.where the two element classes are
the edges, and OP-7 is the +inf element inside the T class):
    OP-1   16a T   v < 0, clipped to 0.0
    OP-2   16a F   v >= 0
    OP-3   16b T   v > 1, clipped to 1.0
    OP-4   16b F   v <= 1, v returned
    OP-5   23 T    finite element f >= 0
    OP-6   23 F    element f < 0
    OP-7   23 T    element f = +inf, gives 0.0
    OP-8   28 T    f >= 0
    OP-9   28 F    f < 0
    OP-10  29 T    f == inf
    OP-11  29 F    finite f >= 0
    OP-12  36 T    top <= 0, all-zero fitness gives ones
    OP-13  36 F    top > 0

Coverage level: 13 of 13 feasible branch requirements (100 percent), all by this module. Every requirement is
feasible.

Infeasible or unspecified, therefore not asserted:
    - the top < 0 part of OP-12: fitness values are never negative, and the docstring specifies only the
      all-zero case;
    - fitness(NaN) and fitness_scalar(NaN): no spec source states the result (the optimizer maps NaN to +inf
      before calling them).
Note on OP-10: 1.0 / (1.0 + inf) is already 0.0 in IEEE arithmetic, so the inf guard on line 29 is
behaviourally redundant; the test executes the edge, but no output can tell the two edges apart.

Box types (P-T4-08): white-box tests are designed from the edges above with crafted inputs; black-box tests
are designed from the docstrings, the original hyp_optimizer ABC.py and Karaboga & Basturk (2007). Every unit
has at least one black-box test.

Input space: Three float64 values for neighbour_value and float64 arrays of any length for fitness and
onlooker_probabilities.
Subset: One case per branch of the control flow graphs.
"""

import math

import numpy as np
import pytest

from metahpo.operators import fitness, fitness_scalar, neighbour_value, onlooker_probabilities

GRID_SEED = 0
GRID_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_PHIS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _neighbour_values_on_grid(phis):
    """neighbour_value for every (x_ij, x_kj) of GRID_VALUES and every phi of `phis`, in grid order."""
    return [neighbour_value(x_ij, x_kj, phi) for x_ij in GRID_VALUES for x_kj in GRID_VALUES for phi in phis]


def _grid_values_and_positions(phi, position):
    """(neighbour_value(x_ij, x_kj, phi), (x_ij, x_kj)[position]) lists over every pair of GRID_VALUES."""
    pairs = [(x_ij, x_kj) for x_ij in GRID_VALUES for x_kj in GRID_VALUES]
    return [neighbour_value(x_ij, x_kj, phi) for x_ij, x_kj in pairs], [pair[position] for pair in pairs]


def _scalar_fitness_of_each(values):
    """fitness_scalar applied to each value of `values` one by one."""
    return [fitness_scalar(v) for v in values]


@pytest.mark.parametrize(
    ("x_ij", "x_kj", "phi", "expected"),
    [
        pytest.param(0.25, 0.875, 0.5, 0.0, id="OP-1-raw-minus-0.0625-clips-to-0"),
        pytest.param(0.75, 0.125, 0.5, 1.0, id="OP-3-raw-1.0625-clips-to-1"),
    ],
)
def test_neighbour_value_should_clip_to_the_violated_bound_when_crafted_input_leaves_the_cube(
    x_ij, x_kj, phi, expected
):
    """A raw neighbour value outside [0, 1] is replaced by the bound it crosses.

    Given crafted inputs whose raw value is binary-exact: (0.25, 0.875, 0.5) gives v = -0.0625, forcing 16a T,
    and (0.75, 0.125, 0.5) gives v = 1.0625, forcing 16a F then 16b T.
    When neighbour_value is called.
    Then it returns exactly 0.0 or 1.0; postcondition: none (pure function).
    Covers: OP-1, OP-2, OP-3
    Box: white
    Source: operators.neighbour_value docstring ('clipped to [0, 1] like the original ABC'); ABC.py 100-103.
    """
    result = neighbour_value(x_ij, x_kj, phi)

    assert result == expected


@pytest.mark.parametrize(
    ("x_ij", "x_kj", "phi", "expected"),
    [
        pytest.param(0.25, 0.75, -1.0, 0.75, id="OP-2-OP-4-strictly-inside"),
        pytest.param(0.5, 1.0, 1.0, 0.0, id="OP-2-raw-exactly-0-is-not-clipped"),
        pytest.param(0.5, 0.0, 1.0, 1.0, id="OP-4-raw-exactly-1-is-not-clipped"),
        pytest.param(0.3, 0.7, 0.0, 0.3, id="OP-2-OP-4-phi-0-returns-x_ij"),
    ],
)
def test_neighbour_value_should_return_the_raw_value_when_crafted_input_stays_in_the_closed_cube(
    x_ij, x_kj, phi, expected
):
    """A raw neighbour value inside [0, 1], ends included, is returned unchanged.

    Given crafted inputs: (0.25, 0.75, -1.0) gives v = 0.75 strictly inside (16a F, 16b F); (0.5, 1.0, 1.0)
    gives v = 0.0 exactly, so 'v < 0' is false on the boundary (16a F); (0.5, 0.0, 1.0) gives v = 1.0 exactly,
    so 'v > 1' is false on the boundary (16b F); phi = 0 gives v = x_ij.
    When neighbour_value is called.
    Then it returns v; postcondition: none (pure function).
    Covers: OP-2, OP-4
    Box: white
    Source: operators.neighbour_value docstring (v_ij = x_ij + phi * (x_ij - x_kj), clipped to [0, 1]).
    """
    result = neighbour_value(x_ij, x_kj, phi)

    assert result == expected


def test_neighbour_value_should_stay_in_the_unit_interval_and_reach_both_bounds_when_inputs_span_a_grid():
    """Over a 5 x 5 x 5 grid of positions and phi values every neighbour value lies in [0, 1].

    Given x_ij and x_kj in {0, 0.25, 0.5, 0.75, 1} and phi in {-1, -0.5, 0, 0.5, 1} (125 triples, chosen so
    that raw values fall below 0, above 1, on both ends and inside).
    When neighbour_value is called on every triple.
    Then the results stay in [0, 1] and both bounds are reached; the clip edges themselves are forced by the
    white-box tests (0.0 and 1.0 also occur without clipping, for x_ij = 0 or 1 with phi = 0).
    Covers: OP-1, OP-2, OP-3, OP-4
    Box: black
    Source: Karaboga & Basturk (2007) neighbour equation with phi in [-1, 1]; ABC.py 98-103 (bound clamping).
    """
    results = _neighbour_values_on_grid(GRID_PHIS)

    assert min(results) == 0.0
    assert max(results) == 1.0


@pytest.mark.parametrize(
    ("phi", "position"),
    [
        pytest.param(0.0, 0, id="phi-0-returns-x_ij"),
        pytest.param(-1.0, 1, id="phi-minus-1-returns-x_kj"),
    ],
)
def test_neighbour_value_should_return_the_fixed_point_of_the_karaboga_equation_when_phi_is_0_or_minus_1(
    phi, position
):
    """The neighbour equation has fixed points: phi = 0 gives x_ij and phi = -1 gives x_kj.

    Given every pair (x_ij, x_kj) of the grid {0, 0.25, 0.5, 0.75, 1} (all binary-exact, so no rounding).
    When neighbour_value(x_ij, x_kj, phi) is called with phi = 0 or phi = -1.
    Then the result equals x_ij (phi = 0) or x_kj (phi = -1, since x_ij - (x_ij - x_kj) = x_kj), which lies in
    [0, 1], so the clip is never taken.
    Covers: OP-2, OP-4
    Box: black
    Source: Karaboga & Basturk (2007) v_ij = x_ij + phi_ij * (x_ij - x_kj); hand algebra.
    """
    results, expected = _grid_values_and_positions(phi, position)

    assert results == expected


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        pytest.param(0.0, 1.0, id="boundary-zero-takes-the-nonnegative-branch"),
        pytest.param(3.0, 0.25, id="typical-positive"),
        pytest.param(0.25, 0.8, id="fractional-positive"),
    ],
)
def test_fitness_scalar_should_return_one_over_one_plus_f_when_f_is_finite_and_nonnegative(f, expected):
    """A finite non-negative value takes the 1 / (1 + f) branch.

    Given f = 0.0 (the boundary of 'f >= 0': the 1 + |f| branch would also give 1.0, so this case pins the
    boundary value itself), f = 3.0 or f = 0.25 (finite and >= 0, forcing 28 T then 29 F).
    When fitness_scalar(f) is called.
    Then it returns 1.0, 0.25 or 0.8 exactly (hand computed); postcondition: none (pure function).
    Covers: OP-8, OP-11
    Box: white
    Source: ABC.py 46-47 (1 / (fun + 1) for fun >= 0); operators.fitness docstring.
    """
    result = fitness_scalar(f)

    assert result == expected


def test_fitness_scalar_should_return_zero_when_f_is_positive_infinity():
    """+inf takes the inf guard and gives fitness 0.0.

    Given f = +inf (>= 0 and == inf, forcing 28 T then 29 T).
    When fitness_scalar(math.inf) is called.
    Then it returns exactly 0.0; postcondition: none (pure function).
    Covers: OP-8, OP-10
    Box: white
    Source: operators.fitness docstring ('+inf gives 0'), referenced by the fitness_scalar docstring.
    """
    result = fitness_scalar(math.inf)

    assert result == 0.0


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        pytest.param(-0.5, 1.5, id="fractional-negative"),
        pytest.param(-2.5, 3.5, id="negative"),
    ],
)
def test_fitness_scalar_should_return_one_plus_abs_f_when_f_is_negative(f, expected):
    """A negative value takes the 1 + |f| branch.

    Given f = -0.5 or f = -2.5 (forcing 28 F).
    When fitness_scalar(f) is called.
    Then it returns 1.5 or 3.5 exactly; postcondition: none (pure function).
    Covers: OP-9
    Box: white
    Source: ABC.py 48-49 (1 + abs(fun)); operators.fitness docstring.
    """
    result = fitness_scalar(f)

    assert result == expected


def test_fitness_should_apply_the_original_formula_to_one_element_of_each_input_class():
    """fitness maps each element by its class: non-negative, negative and +inf.

    Given the array [0.25, 3.0, -0.5, -1000.0, +inf]: two finite non-negative elements (23 T), two negative
    elements (23 F) and +inf (23 T, divide guarded by errstate).
    When fitness(arr) is called.
    Then the result is exactly [0.8, 0.25, 1.5, 1001.0, 0.0] (hand computed), float64, shape (5,).
    Covers: OP-5, OP-6, OP-7
    Box: black
    Source: ABC.py 44-50 (calculate_fitness); operators.fitness docstring ('+inf gives 0').
    """
    arr = np.array([0.25, 3.0, -0.5, -1000.0, math.inf])

    result = fitness(arr)

    assert result.tolist() == [0.8, 0.25, 1.5, 1001.0, 0.0]
    assert result.dtype == np.float64
    assert result.shape == (5,)


def test_fitness_should_equal_fitness_scalar_elementwise_when_values_cover_both_signs_and_infinity():
    """The vectorised and the scalar fitness agree exactly on every element.

    Given 1000 values drawn with np.random.default_rng(GRID_SEED).uniform(-10, 10) (both sign classes in
    volume) with +inf appended.
    When fitness(arr) is compared with fitness_scalar(v) for each element v.
    Then every pair is exactly equal (relational oracle: fitness_scalar is 'fitness for one value').
    Covers: OP-5, OP-6, OP-7, OP-8, OP-9, OP-10, OP-11
    Box: black
    Source: operators.fitness_scalar docstring ('`fitness` for one value without numpy overhead').
    """
    arr = np.append(np.random.default_rng(GRID_SEED).uniform(-10.0, 10.0, 1000), math.inf)

    vectorised = fitness(arr).tolist()

    assert vectorised == _scalar_fitness_of_each(arr.tolist())


@pytest.mark.parametrize(
    ("fit", "expected"),
    [
        pytest.param([0.2, 0.4, 0.8], [0.325, 0.55, 1.0], id="maximum-0.8-shows-the-division-by-max"),
        pytest.param([0.5, 1.0], [0.55, 1.0], id="maximum-1.0-last-element"),
    ],
)
def test_onlooker_probabilities_should_scale_by_the_maximum_when_the_maximum_fitness_is_positive(
    fit, expected
):
    """With a positive maximum fitness every probability is 0.9 * fit / max + 0.1.

    Given fit = [0.2, 0.4, 0.8] (the maximum is 0.8, not 1.0, so a missing division by max would be visible:
    the last value would be 0.82) or fit = [0.5, 1.0]; top > 0 forces 36 F.
    When onlooker_probabilities(fit) is called.
    Then the result is the hand-computed vector within 1e-12, and the element at the maximum fitness is
    exactly 1.0 (0.9 * max / max + 0.1); postcondition: none (pure function).
    Covers: OP-13
    Box: black
    Source: ABC.py 124-127 (prob = 0.9 * fitness / max_fit + 0.1); operators.onlooker_probabilities docstring.
    """
    arr = np.array(fit)

    result = onlooker_probabilities(arr)

    np.testing.assert_allclose(result, expected, rtol=1e-12)
    assert result[int(np.argmax(arr))] == 1.0


def test_onlooker_probabilities_should_give_equal_ones_when_every_fitness_is_zero():
    """All-zero fitness gives equal probabilities instead of dividing by a zero maximum.

    Given fit = np.zeros(4): the maximum is 0, the boundary of 'top <= 0', crafted to force 36 T (a '<'
    comparison would take 36 F and divide 0 by 0).
    When onlooker_probabilities(fit) is called.
    Then the result is exactly [1.0, 1.0, 1.0, 1.0]; postcondition: none (pure function).
    Covers: OP-12
    Box: white
    Source: operators.onlooker_probabilities docstring ('all-zero fitness gives equal probabilities'); task
    spec ('all-zero -> ones').
    """
    fit = np.zeros(4)

    result = onlooker_probabilities(fit)

    assert result.tolist() == [1.0, 1.0, 1.0, 1.0]
