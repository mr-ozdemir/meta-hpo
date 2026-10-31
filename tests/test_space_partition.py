"""Equivalence classes, strict partitions and exhaustive small domains of Float, Int, Categorical and Space.

Oracles: README search-space table (Float real numbers in [low, high]; Int integers low..high inclusive;
Categorical one of choices), the docstrings of Int and Categorical ('every integer / choice gets an equal
share of the unit interval'), Parameter.decode ('unit values in [0, 1]'), Space ('dimension d = number of
parameters'), Space.rows / Space.to_dicts ('params ... with Python types') and Space.move ('a point
differing from a decoded point only in coordinate j'). Every expected value below is derived by hand from
these sentences: linear Float low + u * (high - low), log Float exp(ln low + u * (ln high - ln low)), equal
share index floor(u * n) on [k/n, (k+1)/n).

The class tables of the other units of this work item are in the module docstrings of
test_operators_partition.py (fitness, onlooker probabilities), test_abc_partition.py (ABCOptimizer),
test_study_partition.py (Trial, Study) and test_facade_partition.py (optimize).

Class table (feature, viewpoint, class id, definition, representative, expected behaviour, covering test)

Float, construction validity
  F1  low < high, linear     Float(0, 10)            valid
      test_space_to_dicts_should_interpolate_linearly_when_float_is_linear
  F2  low == high            Float(1, 1)             ValueError
      tests/test_space.py::test_invalid_parameters_are_rejected (existing)
  F3  low > high             Float(2, 1)             ValueError
      test_float_should_raise_value_error_when_bounds_are_invalid
  F4  log, low > 0           Float(1, 100, log)      valid
      test_space_to_dicts_should_interpolate_in_log_space_when_float_is_log
  F5  log, low <= 0          Float(0, 1, log) existing; Float(-1, 1, log)   ValueError
      test_float_should_raise_value_error_when_bounds_are_invalid
Float, decoding (interior values, not boundaries)
  F1  u = 0.3 -> 3.0, u = 0.7 -> 7.0;  F4  u = 0.25 -> sqrt(10), u = 0.75 -> 10 * sqrt(10)

Int, construction validity
  I1  low < high, linear     Int(0, 9)               valid
      test_space_to_dicts_should_give_the_equal_share_integer_when_int_has_several_values
  I2  low == high            Int(2, 2)               valid, always low
      test_space_to_dicts_should_always_give_low_when_int_has_one_value
  I3  low > high             Int(3, 2)               ValueError
  I4  log, low <= 0          Int(0, 5, log), Int(-1, 5, log)   ValueError
      test_int_should_raise_value_error_when_bounds_are_invalid
  I5  log, low > 0           Int(1, 100, log)        valid, value inside low..high (share unspecified)
      test_space_to_dicts_should_stay_inside_the_inclusive_range_when_int_is_log
Int, decoding partition (strict)
  class k = [k/4, (k+1)/4) for k = 0..2 and [3/4, 1] for k = 3 on Int(0, 3); edges k/4, the double just
  below k/4, 0.0 and 1.0; midpoints (k + 0.5)/4 as interior values
      test_space_should_decode_each_grid_point_to_its_own_integer_by_rows_and_move_when_int_is_0_to_3
Int, exhaustive small domains (midpoint grid of 7 points per value)
  Int(0, 1), Int(-2, 2), Int(2, 2)
      test_space_to_dicts_should_give_each_integer_seven_times_on_the_midpoint_grid_when_domain_is_small

Categorical, construction validity
  C1  no choice              Categorical([])         ValueError
      tests/test_space.py::test_invalid_parameters_are_rejected (existing)
  C2  one choice             Categorical(['only'])   always 'only'
      test_space_to_dicts_should_always_give_the_choice_when_categorical_has_one_choice
  C3  several strings        Categorical(['a', 'b', 'c'])   choice floor(3u)
      test_space_to_dicts_should_give_the_equal_share_choice_when_categorical_has_string_choices
  C4  non-string objects     (object(), None, 7, object())  the same objects (identity)
      test_space_to_dicts_should_return_the_same_choice_objects_when_choices_are_not_strings
      test_optimize_should_pass_the_same_choice_objects_when_choices_are_not_strings
Categorical, decoding partition (strict)
  class j = [j/4, (j+1)/4) for j = 0..2 and [3/4, 1] for j = 3 on four choices
      test_space_should_decode_each_grid_point_to_its_own_choice_by_rows_and_move_when_categorical_has_4
Categorical, exhaustive small domains: sizes 1 to 5, distinct object() choices, 7 midpoints per choice
      test_space_to_dicts_should_give_each_choice_seven_times_on_the_midpoint_grid_when_size_is_1_to_5

Space, construction validity
  S1  empty mapping          Space({})               ValueError
  S2  value not a Parameter  Space({'x': (0, 1)}) existing; Space({'x': Float}) (the class itself)  TypeError
      test_space_should_reject_the_mapping_when_it_is_empty_or_holds_a_non_parameter
  S3  valid mixed mapping    Float, Int, Categorical   dim == 3, Python types
      test_space_should_report_one_dimension_per_parameter_and_python_types_when_mapping_is_mixed
Space, output types and identity: S3 (type(v) is float / int), C4 (identity)
Space, size (number of distinct parameter sets)
  Z1  no Float, Int only           Int(2, 2), Int(-2, 2), Int(1, 64, log)   high - low + 1
  Z2  no Float, Categorical only   Categorical(['only'])                     number of choices
  Z3  no Float, several columns    Int(1, 3) x 2 choices, Int(0, 9) x Int(1, 2) x 3 choices   product
      test_space_size_should_be_the_product_of_int_widths_and_choice_counts_when_the_space_has_no_float
  Z4  some Float                   Float alone, log Float, Float first or after discrete columns   inf
      test_space_size_should_be_infinite_when_any_parameter_is_a_float

Edge and corner cases: k/4 and the double just below it, u = 0.0 and u = 1.0, low == high for Int, one
choice; the corner where several edges meet (Int(2, 2) with one choice under the cache) is in
test_facade_partition.py.

Covered (exhaustive claims are limited to the enumerated grids): the 13 grid points of Int(0, 3) and of
four choices; the midpoint grid of 7 points per value for Int(0, 1), Int(-2, 2), Int(2, 2) and for
Categorical sizes 1 to 5. A pass shows agreement with the stated spec only: the equal share of the Int and
Categorical docstrings. It is no proof for every u.

Not covered: u outside [0, 1] (no class, the optimizers never produce it); the share of log Int values; any
grid boundary j/n with n not a power of two (j/n is not exact as a double, so the side is ambiguous);
duplicate choices; Float continuity between the sampled points; neighbour_value (tests/test_abc.py).

Requirement issues (P-T1-38), not asserted:
  - Duplicate choices (['a', 'a', 'b']) put one value into two classes; behaviour unspecified.
  - The equal-share intervals [k/n, (k+1)/n) leave u = 1.0 in no class although Parameter.decode accepts
    [0, 1]. The table closes the top class, which the README inclusive range and the existing
    tests/test_space.py (Int(3, 7) at 1.0 gives 7) agree with.
  - The Int docstring promises an equal share for every Int, but log=True gives unequal shares
    (reported as a defect, see docs/test-izlenebilirlik.md).
  - A str passed as choices is split into characters; unspecified.
  - A log Float decodes u = 1.0 slightly above high (Float(1e-4, 1e-2, log=True) gives
    0.010000000000000004), outside the README range [low, high]; reported as a defect, so the log
    class F4 is tested at interior u only.

Input space: Every row of d unit-cube coordinates in [0, 1] for every parameter kind and bound, which is
unbounded.
Subset: One representative per equivalence class, strict partitions and exhaustive small domains.
"""

import math
from collections import Counter

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space, optimize

ANY_SEED = 0
GRID_POINTS_PER_VALUE = 7
UNUSED_FLOAT_U = 0.25
BASE_U = 0.5


def _column(params, name):
    return [p[name] for p in params]


def _midpoint_grid(n_values):
    size = GRID_POINTS_PER_VALUE * n_values
    return ((np.arange(size) + 0.5) / size).reshape(-1, 1)


def _distinct_objects(n):
    return tuple(object() for _ in range(n))


def _decode_one_value(parameter, u):
    space = Space({"v": parameter})
    (params,) = space.to_dicts(np.array([[u]]))
    return params["v"]


def _rows_and_move(space, u):
    base_keys, base_params = space.rows(np.array([[BASE_U, UNUSED_FLOAT_U]]))
    full_keys, full_params = space.rows(np.array([[u, UNUSED_FLOAT_U]]))
    moved = space.move(base_keys[0], base_params[0], 0, u)
    return (full_keys[0], full_params[0]), moved


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param({"low": 2.0, "high": 1.0}, id="F3-low-above-high"),
        pytest.param({"low": -1.0, "high": 1.0, "log": True}, id="F5-log-negative-low-interior-of-invalid"),
    ],
)
def test_float_should_raise_value_error_when_bounds_are_invalid(bounds):
    """Given Float bounds of an invalid class, when Float is built, then ValueError is raised.

    Oracle: task spec 'invalid parameters raise ValueError'; README Float values in [low, high] need
    low < high, and a log scale needs low > 0. F2 (low == high) and F5 with low == 0 are already in
    tests/test_space.py, so these cases add the other side of low < high and the interior of low <= 0.
    """
    with pytest.raises(ValueError):
        Float(**bounds)


@pytest.mark.parametrize(
    ("u", "expected"),
    [
        pytest.param(0.3, 3.0, id="F1-interior-u0.3"),
        pytest.param(0.7, 7.0, id="F1-interior-u0.7"),
    ],
)
def test_space_to_dicts_should_interpolate_linearly_when_float_is_linear(u, expected):
    """Given Float(0, 10) and an interior u, when to_dicts decodes it, then it returns 10u as a Python float.

    Oracle: README Float values in [low, high] with the unit-cube mapping low + u * (high - low); hand
    calculation 0 + 0.3 * 10 = 3.0 and 0 + 0.7 * 10 = 7.0. approx because the product is float arithmetic.
    Postcondition: none (pure function).
    """
    parameter = Float(0.0, 10.0)

    value = _decode_one_value(parameter, u)

    assert value == pytest.approx(expected, rel=1e-12)
    assert type(value) is float


@pytest.mark.parametrize(
    ("u", "expected"),
    [
        pytest.param(0.25, math.sqrt(10.0), id="F4-interior-u0.25-gives-100-to-the-quarter"),
        pytest.param(0.75, 10.0 * math.sqrt(10.0), id="F4-interior-u0.75-gives-100-to-three-quarters"),
    ],
)
def test_space_to_dicts_should_interpolate_in_log_space_when_float_is_log(u, expected):
    """Given Float(1, 100, log=True) and an interior u, when to_dicts decodes it, then it returns 100 ** u.

    Oracle: log scale means equal steps of ln value: exp(ln 1 + u * (ln 100 - ln 1)) = 100 ** u, so
    0.25 gives sqrt(10) and 0.75 gives 10 * sqrt(10) by hand. u = 0.5 is avoided because
    tests/test_space.py already checks the geometric midpoint. Postcondition: none (pure function).
    """
    parameter = Float(1.0, 100.0, log=True)

    value = _decode_one_value(parameter, u)

    assert value == pytest.approx(expected, rel=1e-12)
    assert type(value) is float


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param({"low": 3, "high": 2}, id="I3-low-above-high"),
        pytest.param({"low": 0, "high": 5, "log": True}, id="I4-log-low-zero-edge"),
        pytest.param({"low": -1, "high": 5, "log": True}, id="I4-log-low-negative-interior"),
    ],
)
def test_int_should_raise_value_error_when_bounds_are_invalid(bounds):
    """Given Int bounds of an invalid class, when Int is built, then ValueError is raised.

    Oracle: task spec 'invalid parameters raise ValueError'; README integers low..high inclusive (empty when
    low > high) and log scale undefined for low <= 0. Postcondition: none (construction fails).
    """
    with pytest.raises(ValueError):
        Int(**bounds)


@pytest.mark.parametrize(
    ("u", "expected"),
    [
        pytest.param(0.15, 1, id="I1-interior-u0.15-floor-1.5"),
        pytest.param(0.55, 5, id="I1-interior-u0.55-floor-5.5"),
    ],
)
def test_space_to_dicts_should_give_the_equal_share_integer_when_int_has_several_values(u, expected):
    """Given Int(0, 9) and an interior u, when to_dicts decodes it, then it gives floor(10u) as a Python int.

    Oracle: Int docstring (every integer an equal share, so value low + floor(u * 10)); hand calculation
    floor(1.5) = 1 and floor(5.5) = 5; both u lie in the middle of a share, not on k/10.
    Postcondition: none (pure function).
    """
    parameter = Int(0, 9)

    value = _decode_one_value(parameter, u)

    assert value == expected
    assert type(value) is int


@pytest.mark.parametrize(
    "u",
    [
        pytest.param(0.0, id="I2-u-lower-edge"),
        pytest.param(0.5, id="I2-u-interior"),
        pytest.param(math.nextafter(1.0, 0.0), id="I2-u-just-below-one"),
        pytest.param(1.0, id="I2-u-upper-edge"),
    ],
)
def test_space_to_dicts_should_always_give_low_when_int_has_one_value(u):
    """Given Int(2, 2), when to_dicts decodes any listed u, then it returns the only integer 2 as Python int.

    Oracle: README integers low..high inclusive (low == high leaves exactly one value) and the Int docstring
    (that value owns the whole unit interval). Postcondition: none (pure function).
    """
    parameter = Int(2, 2)

    value = _decode_one_value(parameter, u)

    assert value == 2
    assert type(value) is int


@pytest.mark.parametrize(
    "u",
    [
        pytest.param(0.25, id="I5-interior-u0.25"),
        pytest.param(0.5, id="I5-interior-u0.5"),
        pytest.param(0.75, id="I5-interior-u0.75"),
    ],
)
def test_space_to_dicts_should_stay_inside_the_inclusive_range_when_int_is_log(u):
    """Given Int(1, 100, log=True), when to_dicts decodes an interior u, then it gives a Python int in 1..100.

    Oracle: README integers low..high inclusive. Relational oracle only: the share of each value under
    log=True is not specified, so no exact value is asserted. Postcondition: none (pure function).
    """
    parameter = Int(1, 100, log=True)

    value = _decode_one_value(parameter, u)

    assert 1 <= value <= 100
    assert type(value) is int


STRICT_INT_0_3_GRID = [
    pytest.param(0.0, 0, id="u0-lower-edge-gives-0"),
    pytest.param(0.125, 0, id="u0.125-midpoint-gives-0"),
    pytest.param(math.nextafter(0.25, 0.0), 0, id="just-below-1of4-gives-0"),
    pytest.param(0.25, 1, id="u1of4-gives-1"),
    pytest.param(0.375, 1, id="u0.375-midpoint-gives-1"),
    pytest.param(math.nextafter(0.5, 0.0), 1, id="just-below-2of4-gives-1"),
    pytest.param(0.5, 2, id="u2of4-gives-2"),
    pytest.param(0.625, 2, id="u0.625-midpoint-gives-2"),
    pytest.param(math.nextafter(0.75, 0.0), 2, id="just-below-3of4-gives-2"),
    pytest.param(0.75, 3, id="u3of4-gives-3"),
    pytest.param(0.875, 3, id="u0.875-midpoint-gives-3"),
    pytest.param(math.nextafter(1.0, 0.0), 3, id="just-below-one-gives-3"),
    pytest.param(1.0, 3, id="u1-upper-edge-closed-top-class-gives-3"),
]


@pytest.mark.parametrize(("u", "expected"), STRICT_INT_0_3_GRID)
def test_space_should_decode_each_grid_point_to_its_own_integer_by_rows_and_move_when_int_is_0_to_3(
    u, expected
):
    """Given Space({'k': Int(0, 3), 'o': Float(0, 1)}), when rows and move decode u, then both give k.

    Oracle: Int docstring (equal share: k owns [k/4, (k+1)/4)), README inclusive range (1.0 belongs to the
    top value) and the Space.move docstring (the moved point equals the decoded point that differs only in
    coordinate j, so key and params must equal those of rows). n = 4 is a power of two, so k/4 and u * 4
    are exact doubles and the side of every boundary is decided by the spec, not by rounding. The double
    just below k/4 is the discriminating input against an off-by-one or a round instead of floor; the two
    paths are separate decoders (sites: grouped numpy decode and Int.decode_one).
    Postcondition: none (pure functions).
    """
    space = Space({"k": Int(0, 3), "o": Float(0.0, 1.0)})

    decoded, moved = _rows_and_move(space, u)

    assert decoded[1]["k"] == expected
    assert moved == decoded


STRICT_CATEGORICAL_4_GRID = [
    pytest.param(0.0, "a", id="u0-lower-edge-gives-a"),
    pytest.param(0.125, "a", id="u0.125-midpoint-gives-a"),
    pytest.param(math.nextafter(0.25, 0.0), "a", id="just-below-1of4-gives-a"),
    pytest.param(0.25, "b", id="u1of4-gives-b"),
    pytest.param(0.375, "b", id="u0.375-midpoint-gives-b"),
    pytest.param(math.nextafter(0.5, 0.0), "b", id="just-below-2of4-gives-b"),
    pytest.param(0.5, "c", id="u2of4-gives-c"),
    pytest.param(0.625, "c", id="u0.625-midpoint-gives-c"),
    pytest.param(math.nextafter(0.75, 0.0), "c", id="just-below-3of4-gives-c"),
    pytest.param(0.75, "d", id="u3of4-gives-d"),
    pytest.param(0.875, "d", id="u0.875-midpoint-gives-d"),
    pytest.param(math.nextafter(1.0, 0.0), "d", id="just-below-one-gives-d"),
    pytest.param(1.0, "d", id="u1-upper-edge-closed-top-class-gives-d"),
]


@pytest.mark.parametrize(("u", "expected"), STRICT_CATEGORICAL_4_GRID)
def test_space_should_decode_each_grid_point_to_its_own_choice_by_rows_and_move_when_categorical_has_4(
    u, expected
):
    """Given four choices a..d and a Float, when u is decoded by rows and by move, then both give the choice.

    Oracle: Categorical docstring (equal share: choice j owns [j/4, (j+1)/4)), README 'one of choices' (1.0
    must still give a choice, the top one) and the Space.move docstring (same key and params as rows).
    Four choices keep j/4 exact; the double just below j/4 guards against an off-by-one in the index
    (sites: grouped categorical decode and Categorical.decode_one). Postcondition: none (pure functions).
    """
    space = Space({"c": Categorical(["a", "b", "c", "d"]), "o": Float(0.0, 1.0)})

    decoded, moved = _rows_and_move(space, u)

    assert decoded[1]["c"] == expected
    assert moved == decoded


@pytest.mark.parametrize(
    ("low", "high"),
    [
        pytest.param(0, 1, id="int_0_1-two-values"),
        pytest.param(-2, 2, id="int_minus2_2-negative-low-five-values"),
        pytest.param(2, 2, id="int_2_2-one-value"),
    ],
)
def test_space_to_dicts_should_give_each_integer_seven_times_on_the_midpoint_grid_when_domain_is_small(
    low, high
):
    """Given a small Int domain, when its midpoint grid is decoded, then each integer occurs 7 times.

    Oracle: Int docstring equal share (the spec assumption this exhaustive check depends on). The grid
    (i + 0.5) / (7n) keeps every point away from a share boundary, so rounding cannot move it. The claim is
    limited to this grid; Int(3, 7) is left out because tests/test_space.py already counts it.
    Postcondition: none (pure function).
    """
    space = Space({"v": Int(low, high)})
    grid = _midpoint_grid(high - low + 1)

    counts = Counter(_column(space.to_dicts(grid), "v"))

    assert counts == dict.fromkeys(range(low, high + 1), GRID_POINTS_PER_VALUE)


@pytest.mark.parametrize("n_choices", [pytest.param(n, id=f"cat_size_{n}") for n in range(1, 6)])
def test_space_to_dicts_should_give_each_choice_seven_times_on_the_midpoint_grid_when_size_is_1_to_5(
    n_choices,
):
    """Given n distinct object() choices, when their midpoint grid is decoded, then each occurs 7 times.

    Oracle: Categorical docstring equal share (the spec assumption this exhaustive check depends on).
    Counting by identity means equal-comparing objects cannot hide a wrong choice. The claim is limited to
    sizes 1 to 5 on this grid. Postcondition: none (pure function).
    """
    choices = _distinct_objects(n_choices)
    space = Space({"c": Categorical(choices)})
    grid = _midpoint_grid(n_choices)

    counts = Counter(map(id, _column(space.to_dicts(grid), "c")))

    assert counts == dict.fromkeys(map(id, choices), GRID_POINTS_PER_VALUE)


@pytest.mark.parametrize(
    "u",
    [
        pytest.param(0.0, id="C2-u-lower-edge"),
        pytest.param(0.5, id="C2-u-interior"),
        pytest.param(1.0, id="C2-u-upper-edge"),
    ],
)
def test_space_to_dicts_should_always_give_the_choice_when_categorical_has_one_choice(u):
    """Given Categorical(['only']), when to_dicts decodes any listed u, then it returns 'only'.

    Oracle: README 'one of choices' with a single choice and the Categorical docstring (it owns the whole
    unit interval). Postcondition: none (pure function).
    """
    parameter = Categorical(["only"])

    value = _decode_one_value(parameter, u)

    assert value == "only"


@pytest.mark.parametrize(
    ("u", "expected"),
    [
        pytest.param(0.1, "a", id="C3-interior-u0.1-floor-0.3"),
        pytest.param(0.5, "b", id="C3-interior-u0.5-floor-1.5"),
        pytest.param(0.9, "c", id="C3-interior-u0.9-floor-2.7"),
    ],
)
def test_space_to_dicts_should_give_the_equal_share_choice_when_categorical_has_string_choices(u, expected):
    """Given three string choices and an interior u, when to_dicts decodes it, then it gives choice floor(3u).

    Oracle: Categorical docstring (equal share, index floor(u * 3)); hand calculation floor(0.3) = 0,
    floor(1.5) = 1, floor(2.7) = 2. Postcondition: none (pure function).
    """
    parameter = Categorical(["a", "b", "c"])

    value = _decode_one_value(parameter, u)

    assert value == expected


FIRST_OBJECT = object()
LAST_OBJECT = object()
NON_STRING_CHOICES = (FIRST_OBJECT, None, 7, LAST_OBJECT)


@pytest.mark.parametrize(
    ("u", "expected"),
    [
        pytest.param(0.1, FIRST_OBJECT, id="C4-plain-object"),
        pytest.param(0.3, None, id="C4-none"),
        pytest.param(0.6, 7, id="C4-int"),
        pytest.param(0.9, LAST_OBJECT, id="C4-second-plain-object"),
    ],
)
def test_space_to_dicts_should_return_the_same_choice_objects_when_choices_are_not_strings(u, expected):
    """Given choices (object(), None, 7, object()), when to_dicts decodes u, then it returns that very object.

    Oracle: README 'one of choices' and the Categorical docstring (index floor(4u): 0.1 -> 0, 0.3 -> 1,
    0.6 -> 2, 0.9 -> 3). Identity ('is') is the matcher because a choice must be the user's object, not an
    equal copy or a converted value. Postcondition: none (pure function).
    """
    parameter = Categorical(NON_STRING_CHOICES)

    value = _decode_one_value(parameter, u)

    assert value is expected


def test_optimize_should_pass_the_same_choice_objects_when_choices_are_not_strings():
    """Given non-string choices, when optimize runs 30 uncached evaluations, then each is a given choice.

    Oracle: README 'one of choices'. 30 evaluations with the default colony (5 initial points) reach the
    employed and onlooker phases, whose candidates are decoded by Space.move instead of rows, so both
    decode paths hand the objective the user's objects (identity, compared by id). Relational oracle,
    seed-independent; ANY_SEED only makes the run reproducible. Postcondition: none beyond the study.
    """
    space = Space({"c": Categorical(NON_STRING_CHOICES), "x": Float(0.0, 1.0)})
    seen = []

    def spy_objective(params):
        seen.append(params["c"])
        return params["x"]

    optimize(spy_objective, space, max_evals=30, seed=ANY_SEED, cache=False)

    assert len(seen) == 30
    assert set(map(id, seen)) <= set(map(id, NON_STRING_CHOICES))


@pytest.mark.parametrize(
    ("mapping", "error"),
    [
        pytest.param({}, ValueError, id="S1-empty-mapping"),
        pytest.param({"x": Float}, TypeError, id="S2-parameter-class-instead-of-instance"),
    ],
)
def test_space_should_reject_the_mapping_when_it_is_empty_or_holds_a_non_parameter(mapping, error):
    """Given an empty mapping or a value that is no parameter instance, when Space is built, then it raises.

    Oracle: task spec (invalid parameters raise ValueError/TypeError) and the Space docstring (an ordered set
    of named parameters, d = number of parameters, so d = 0 is not a space). The class Float itself is the
    discriminating input: it is related to a parameter but is not one. Space({'x': (0, 1)}) is already in
    tests/test_space.py. Postcondition: none (construction fails).
    """
    with pytest.raises(error):
        Space(mapping)


def test_space_should_report_one_dimension_per_parameter_and_python_types_when_mapping_is_mixed():
    """Given a Float, an Int and a Categorical, when dim and to_dicts are read, then d = 3 with Python types.

    Oracle: Space docstring (d = number of parameters), rows/to_dicts docstring (Python types), hand
    calculation 0 + 0.3 * 10 = 3.0, floor(0.55 * 10) = 5, floor(0.75 * 2) = 1 -> 'q'. type(...) is
    float / int is the matcher because a numpy scalar would pass isinstance(v, float).
    Postcondition: none (pure functions).
    """
    space = Space({"a": Float(0.0, 10.0), "n": Int(0, 9), "c": Categorical(["p", "q"])})

    (params,) = space.to_dicts(np.array([[0.3, 0.55, 0.75]]))

    assert space.dim == 3
    assert params == {"a": pytest.approx(3.0, rel=1e-12), "n": 5, "c": "q"}
    assert (type(params["a"]), type(params["n"])) == (float, int)


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        pytest.param({"n": Int(2, 2)}, 1, id="Z1-one-value-int"),
        pytest.param({"n": Int(-2, 2)}, 5, id="Z1-inclusive-int-width"),
        pytest.param({"n": Int(1, 64, log=True)}, 64, id="Z1-log-int-counts-every-integer"),
        pytest.param({"c": Categorical(["only"])}, 1, id="Z2-one-choice"),
        pytest.param({"n": Int(1, 3), "c": Categorical(["p", "q"])}, 6, id="Z3-int-times-two-choices"),
        pytest.param(
            {"n": Int(0, 9), "m": Int(1, 2), "c": Categorical(["x", "y", "z"])}, 60, id="Z3-product-of-three"
        ),
    ],
)
def test_space_size_should_be_the_product_of_int_widths_and_choice_counts_when_the_space_has_no_float(
    mapping, expected
):
    """Given a Space of Int and Categorical parameters only, when size is read, then it is the product of the
    inclusive Int widths high - low + 1 and the numbers of choices.

    Oracle: Space.size docstring ('Number of distinct parameter sets'), README (Int integers low..high
    inclusive; Categorical one of choices). Hand calculation: 1, 5, 64, 1, 3 * 2 = 6 and 10 * 2 * 3 = 60.
    Int(2, 2) and one choice are the smallest factors; the log Int shows that the scale does not change the
    count; 60 needs every factor, so a size that skipped a column or used high - low would differ.
    """
    space = Space(mapping)

    size = space.size

    assert size == expected


@pytest.mark.parametrize(
    "mapping",
    [
        pytest.param({"x": Float(0.0, 1.0)}, id="Z4-float-alone"),
        pytest.param({"x": Float(1e-4, 1e-2, log=True)}, id="Z4-log-float"),
        pytest.param({"x": Float(0.0, 1.0), "n": Int(1, 3)}, id="Z4-float-first"),
        pytest.param(
            {"n": Int(1, 3), "c": Categorical(["p"]), "x": Float(0.0, 1.0)}, id="Z4-float-after-discrete"
        ),
    ],
)
def test_space_size_should_be_infinite_when_any_parameter_is_a_float(mapping):
    """Given a Space with at least one Float, alone or before or after Int and Categorical parameters, when
    size is read, then it is math.inf.

    Oracle: Space.size docstring ('infinite when the space has a Float') and README (Float real numbers in
    [low, high]). The Float after two discrete columns shows that a finite partial product does not leak
    out; the Float first shows that the Float decides wherever it stands.
    """
    space = Space(mapping)

    size = space.size

    assert size == math.inf
