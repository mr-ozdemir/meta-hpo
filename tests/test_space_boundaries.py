"""Boundary tests of the search space, including the equal-share boundaries of Int.

Boundary tests of the search space: explicit, implicit and code-level edges of every parameter kind.

Oracles: the README search-space table (Float: real numbers in [low, high]; Int: integers low..high,
inclusive; Categorical: one of choices; optimizers search the unit cube [0, 1]^d), the Int and Categorical
docstrings (every value gets an equal share of the unit interval), Parameter.decode 'Unit values in [0, 1]',
Space.sample 'n uniform points of the unit cube', Space.rows (keys are raw rows, params have Python types)
and Space.move (a point differing from a decoded point only in coordinate j).

Inputs: TOP = nextafter(1.0, 0.0) is the largest value rng.random() returns, so u = 0.0 and u = TOP are the
real ends of sampling. u = 1.0 is in the documented domain [0, 1] and ABC reaches it through neighbour
clipping. START_U = 0.5 is the start row of every move; the expected value never depends on it.

Code-level boundaries (P-T1-44) driven here: the u = 1.0 clamp of Int and Categorical in decode and in the
one-coordinate move, the log-Int clip to [low, high] at u = 0.0 (rint(0.5) = 0) and at u = 1.0
(rint(high + 0.5)), the float-only rows branch and the per-kind column grouping of Space.decode.

Unspecified, therefore not asserted:
- which integer an exact share edge k / n belongs to (only the values just below and just above it are
  fixed by the equal-share rule; at the edge itself only agreement of all decode paths is asserted);
- which integer a log-scaled Int gives near an end beyond being inside low..high;
- Float or Int with infinite or NaN bounds, and Int ranges wider than 2**53 (not exactly representable);
- the exact float a log-scaled Float gives at u = 0.0 (only the linear Float is exactly low there).

Not in this module: Float decoding outside [low, high] at u = 1.0 (linear and log) and at u = 0.0 (log).
That is reported as a library defect, so no test here asserts it. The infeasible-requirement inventory
(P-T4-07) is in the module docstring of test_abc_boundaries.py.

Share-boundary tests of Int decoding through every public decode path.

Oracle: the Int docstring 'Integers low..high inclusive; every integer gets an equal share of the unit
interval', the Space.decode docstring 'integers as whole numbers' and the Space.move docstring (a point
differing in one coordinate, decoded the same way). For Int(0, 2) the shares are thirds, so the expected
integers are derived by hand from u * 3.

Unspecified, therefore not asserted: the decoded value at u exactly equal to 1/3 or 2/3, where floating-point
representation makes the side ambiguous.

Input space: Every row of d unit-cube coordinates in [0, 1] for every parameter kind and bound, which is
unbounded.
Subset: U = 0, u = 1.0, the equal-share boundaries and the log ends of every parameter kind.
"""

import math

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space

TOP = math.nextafter(1.0, 0.0)


START_U = 0.5


ANY_SEED = 0


def _value_via_rows(parameter, u):
    """The Python value of a one-parameter Space at u, through the full decode of Space.rows."""
    space = Space({"p": parameter})
    _, rows = space.rows(np.array([[u]]))
    return rows[0]["p"]


def _value_via_move(parameter, u):
    """The Python value of a one-parameter Space at u, through Space.move from the row decoded at START_U."""
    space = Space({"p": parameter})
    keys, rows = space.rows(np.array([[START_U]]))
    _, params = space.move(keys[0], rows[0], 0, u)
    return params["p"]


def _types(mapping):
    """The list of value types of a params dict, in parameter order."""
    return [type(v) for v in mapping.values()]


DECODE_PATHS = [
    pytest.param(_value_via_rows, id="rows-full-decode"),
    pytest.param(_value_via_move, id="move-one-coordinate"),
]


def test_float_should_decode_both_unit_ends_to_its_bounds_when_bounds_are_one_ulp_apart():
    """Given Float(1.0, nextafter(1.0, 2.0)), when decode runs at u = 0 and u = 1, then it gives exactly low
    and high.

    Oracle: README 'real numbers in [low, high]' and the Float rule low < high. The one-ulp gap is the
    smallest valid side of that rule (P-T1-40); its invalid partner Float(1.0, 1.0) is in tests/test_space.py.
    Hand calculation: 1.0 + 0 * ulp = 1.0 and 1.0 + 1 * ulp = high, both exact. Postcondition: none (pure).
    """
    high = math.nextafter(1.0, 2.0)
    parameter = Float(1.0, high)

    result = parameter.decode(np.array([0.0, 1.0]))

    assert result.tolist() == [1.0, high]


@pytest.mark.parametrize("value_at", DECODE_PATHS)
@pytest.mark.parametrize(
    ("parameter", "u", "expected"),
    [
        pytest.param(Float(-3.0, 7.0), 0.0, -3.0, id="float-lowest-u-gives-low"),
        pytest.param(Int(0, 3), 0.0, 0, id="int-lowest-u-gives-low"),
        pytest.param(Int(0, 3), TOP, 3, id="int-largest-sampled-u-gives-high"),
        pytest.param(Int(0, 3), 1.0, 3, id="int-u-one-is-clamped-to-high"),
        pytest.param(Categorical(["x", "y", "z"]), 0.0, "x", id="categorical-lowest-u-gives-first"),
        pytest.param(Categorical(["x", "y", "z"]), TOP, "z", id="categorical-largest-sampled-u-gives-last"),
        pytest.param(Categorical(["x", "y", "z"]), 1.0, "z", id="categorical-u-one-is-clamped-to-last"),
    ],
)
def test_space_should_map_the_ends_of_the_unit_interval_to_the_end_values_when_u_is_zero_top_or_one(
    value_at, parameter, u, expected
):
    """Given a parameter and an end of [0, 1], when it is decoded by rows or by move, then the end value comes
    out with its type.

    Oracle: README table and the equal-share docstrings: the first share starts at 0, the last share holds
    TOP, and u = 1.0, the closed end of the documented domain, has no later share. Linear Float: low + 0 =
    low. These are the implicit ends of the sampler (P-T1-42) and the code-level u = 1.0 clamp (P-T1-44),
    which the scalar move path implements separately from the array decode. Postcondition: none (pure).
    """
    result = value_at(parameter, u)

    assert (result, type(result)) == (expected, type(expected))


@pytest.mark.parametrize("value_at", DECODE_PATHS)
@pytest.mark.parametrize(
    ("parameter", "u"),
    [
        pytest.param(Int(1, 3, log=True), 0.0, id="log-int-low-one-at-u-zero-rint-half-is-zero"),
        pytest.param(Int(1, 3, log=True), 1.0, id="log-int-at-u-one-rint-three-and-a-half-is-four"),
        pytest.param(Int(1, 1000, log=True), 0.0, id="wide-log-int-at-u-zero"),
        pytest.param(Int(1, 1000, log=True), TOP, id="wide-log-int-at-largest-sampled-u"),
        pytest.param(Int(1, 1000, log=True), 1.0, id="wide-log-int-at-u-one"),
    ],
)
def test_space_should_keep_log_int_values_inside_low_high_when_u_is_at_an_end(value_at, parameter, u):
    """Given a log-scaled Int and an end of [0, 1], when it is decoded by rows or by move, then the value is
    an int in low..high.

    Oracle: README 'integers low..high, inclusive'. Site: the log-Int clip. With low = 1 the log scale starts
    at log(0.5), so u = 0.0 gives exp(log(0.5)) = 0.5 and round-half-even gives 0; with high = 3 it ends at
    log(3.5) and u = 1.0 gives 4. Without the clip both are out of range, so these inputs are decisive
    (P-T1-44). Which integer is chosen inside the range is unspecified. Postcondition: none (pure).
    """
    result = value_at(parameter, u)

    assert type(result) is int
    assert parameter.low <= result <= parameter.high


@pytest.mark.parametrize("value_at", DECODE_PATHS)
@pytest.mark.parametrize(
    ("u", "expected"),
    [
        pytest.param(math.nextafter(0.25, 0.0), 0, id="just-below-first-quarter"),
        pytest.param(math.nextafter(0.25, 1.0), 1, id="just-above-first-quarter"),
        pytest.param(math.nextafter(0.5, 0.0), 1, id="just-below-half"),
        pytest.param(math.nextafter(0.5, 1.0), 2, id="just-above-half"),
        pytest.param(math.nextafter(0.75, 0.0), 2, id="just-below-third-quarter"),
        pytest.param(math.nextafter(0.75, 1.0), 3, id="just-above-third-quarter"),
    ],
)
def test_int_should_switch_value_at_each_share_edge_when_u_is_one_ulp_away_from_it(value_at, u, expected):
    """Given Int(0, 3) and u one ulp below or above a share edge, when it is decoded by rows or by move, then
    the share's integer comes out.

    Oracle: Int docstring 'every integer gets an equal share of the unit interval': four integers give shares
    of width 0.25, so the edges are 0.25, 0.5 and 0.75, and the closest floats on each side belong to the
    lower and the upper share (P-T1-40). Postcondition: none (pure).
    """
    parameter = Int(0, 3)

    result = value_at(parameter, u)

    assert (result, type(result)) == (expected, int)


@pytest.mark.parametrize(
    ("edge", "neighbours"),
    [
        pytest.param(0.25, {0, 1}, id="first-quarter"),
        pytest.param(0.5, {1, 2}, id="half"),
        pytest.param(0.75, {2, 3}, id="third-quarter"),
    ],
)
def test_int_should_give_the_same_integer_in_every_decode_path_when_u_is_exactly_on_a_share_edge(
    edge, neighbours
):
    """Given Int(0, 3) and u exactly on a share edge, when Int.decode, rows and move decode it, then all agree
    on a neighbouring integer.

    Oracle: Space.move docstring (a point differing in coordinate j decodes like the full decode) and
    Space.rows (one decode); which side owns the exact edge is unspecified, so only agreement and membership
    in the two neighbouring integers are asserted. The edges are exact binary floats (P-T1-40). Postcondition:
    none.
    """
    parameter = Int(0, 3)

    observed = (
        int(parameter.decode(np.array([edge]))[0]),
        _value_via_rows(parameter, edge),
        _value_via_move(parameter, edge),
    )

    assert len(set(observed)) == 1
    assert observed[0] in neighbours


@pytest.mark.parametrize("value_at", DECODE_PATHS)
@pytest.mark.parametrize(
    ("u", "expected"),
    [
        pytest.param(0.0, 0, id="lowest-u-gives-zero"),
        pytest.param(TOP, 2**53 - 1, id="largest-sampled-u-gives-two-to-53-minus-one"),
        pytest.param(1.0, 2**53, id="u-one-gives-two-to-53"),
    ],
)
def test_int_should_return_exact_python_ints_when_the_range_reaches_two_to_the_53(value_at, u, expected):
    """Given Int(0, 2**53), when u = 0, TOP or 1 is decoded by rows or by move, then the exact Python int of
    that share comes out.

    Oracle: README 'integers low..high, inclusive', Int equal-share docstring and Space.rows 'Python types'.
    2**53 is the largest integer range float64 holds exactly, an implicit boundary of the hardware (P-T1-42,
    P-T2-33). Hand calculation: there are 2**53 + 1 shares; TOP = 1 - 2**-53 lies below the last share,
    which starts at 1 - 1 / (2**53 + 1), so it is in share 2**53 - 1. Postcondition: none (pure).
    """
    parameter = Int(0, 2**53)

    result = value_at(parameter, u)

    assert (result, type(result)) == (expected, int)


@pytest.mark.parametrize("value_at", DECODE_PATHS)
@pytest.mark.parametrize("u", [0.0, 0.5, TOP, 1.0], ids=["u-zero", "u-half", "u-top", "u-one"])
def test_int_should_always_give_its_only_value_when_low_equals_high(value_at, u):
    """Given Int(5, 5), when any u in [0, 1] is decoded by rows or by move, then the value is 5.

    Oracle: README 'integers low..high, inclusive' with a one-element range, the smallest valid Int
    (P-T2-33); its invalid neighbour low > high is in the error tests. Postcondition: none (pure).
    """
    parameter = Int(5, 5)

    result = value_at(parameter, u)

    assert (result, type(result)) == (5, int)


@pytest.mark.parametrize("value_at", DECODE_PATHS)
@pytest.mark.parametrize("u", [0.0, 0.5, TOP, 1.0], ids=["u-zero", "u-half", "u-top", "u-one"])
def test_categorical_should_always_give_its_only_choice_when_it_has_one_choice(value_at, u):
    """Given Categorical(['only']), when any u in [0, 1] is decoded by rows or by move, then the choice is
    'only'.

    Oracle: README 'one of choices' with a one-element collection (P-T2-33); the empty collection is rejected
    in tests/test_space.py. Postcondition: none (pure).
    """
    parameter = Categorical(["only"])

    result = value_at(parameter, u)

    assert result == "only"


NONE_CHOICE_OBJECT = object()


@pytest.mark.parametrize("value_at", DECODE_PATHS)
@pytest.mark.parametrize(
    ("u", "expected"),
    [
        pytest.param(0.2, None, id="first-share-gives-none"),
        pytest.param(0.8, NONE_CHOICE_OBJECT, id="second-share-gives-the-object"),
    ],
)
def test_categorical_should_return_the_choice_object_itself_when_choices_are_none_and_an_object(
    value_at, u, expected
):
    """Given Categorical([None, obj]), when u in either share is decoded by rows or by move, then that very
    object comes out.

    Oracle: README 'one of choices'. None is the null special value and a bare object has no value equality,
    so only identity proves the choice itself is returned (P-T2-33). 0.2 and 0.8 are inside the two shares.
    Postcondition: none (pure).
    """
    parameter = Categorical([None, NONE_CHOICE_OBJECT])

    result = value_at(parameter, u)

    assert result is expected


def test_float_should_stay_inside_its_bounds_when_u_is_the_largest_sampled_value():
    """Given Float(-3.0, 7.0), when decode runs at TOP, then the value is inside [-3.0, 7.0].

    Oracle: README 'real numbers in [low, high]'. TOP is the implicit upper end of rng.random() (P-T1-42);
    the exact float is rounding-dependent, so a range oracle is used. Postcondition: none (pure).
    """
    parameter = Float(-3.0, 7.0)

    (result,) = parameter.decode(np.array([TOP])).tolist()

    assert -3.0 <= result <= 7.0


@pytest.mark.parametrize("value_at", DECODE_PATHS)
@pytest.mark.parametrize("u", [0.0, TOP, 1.0], ids=["u-zero", "u-top", "u-one"])
def test_log_float_should_stay_inside_its_bounds_when_the_range_spans_24_decades(value_at, u):
    """Given Float(1e-12, 1e12, log=True), when an end of [0, 1] is decoded by rows or by move, then 1e-12 <=
    value <= 1e12.

    Oracle: README 'real numbers in [low, high]'. 24 decades stress the rounding of log and exp, an implicit
    boundary of floating point (P-T1-42). Postcondition: none (pure).
    """
    parameter = Float(1e-12, 1e12, log=True)

    result = value_at(parameter, u)

    assert type(result) is float
    assert 1e-12 <= result <= 1e12


MIXED_SPACE = {
    "f": Float(-2.0, 3.0),
    "lf": Float(1e-3, 10.0, log=True),
    "i": Int(-4, 9),
    "li": Int(2, 500, log=True),
    "c": Categorical(["r", "s", None, 7]),
}


MIXED_BASE_U = [0.37, 0.61, 0.13, 0.88, 0.42]


LOG_INT_ROUNDS_UP_U = 0.3


EDGE_US = [0.0, TOP, 1.0, 0.25, 0.5, 0.75, LOG_INT_ROUNDS_UP_U]


EDGE_IDS = [
    "u-zero",
    "u-top",
    "u-one",
    "u-quarter",
    "u-half",
    "u-three-quarters",
    "u-log-int-fraction-above-half",
]


@pytest.mark.parametrize("u", EDGE_US, ids=EDGE_IDS)
@pytest.mark.parametrize("j", range(len(MIXED_SPACE)), ids=list(MIXED_SPACE))
def test_space_move_should_equal_the_full_decode_when_one_coordinate_is_set_to_an_edge_value(j, u):
    """Given a mixed five-kind Space and a decoded base row, when move sets column j to an edge u, then key
    and params equal rows() of that point.

    Oracle: Space.move docstring '(key, params) of a point differing from a decoded point only in coordinate
    j' and Space.rows. Site: the scalar decode used by move is separate code from the grouped array decode, so
    the ends 0, TOP, 1 and the share edges 0.25, 0.5, 0.75 are its code-level boundaries (P-T1-44). The base
    row is an arbitrary interior point. LOG_INT_ROUNDS_UP_U = 0.3 is the rounding boundary of the move path
    for 'li': on the log scale from log(1.5) to log(500.5) it gives exp(...) of about 8.57, whose fractional
    part is above 0.5, so rounding to the nearest integer (9) and cutting the fraction off (8) differ there.
    Postcondition: none (pure).
    """
    space = Space(MIXED_SPACE)
    base = np.array([MIXED_BASE_U])
    keys, rows = space.rows(base)
    point = base.copy()
    point[0, j] = u
    expected_keys, expected_rows = space.rows(point)

    key, params = space.move(keys[0], rows[0], j, u)

    assert key == expected_keys[0]
    assert params == expected_rows[0]
    assert _types(params) == _types(expected_rows[0])


def test_space_rows_should_return_python_floats_equal_to_the_key_when_every_parameter_is_a_float():
    """Given a Space with only Float parameters, when rows runs, then the params are Python floats and equal
    the raw key.

    Oracle: Space.rows docstring (a key is the raw row; params have Python types); for floats the raw value is
    the parameter value. Site: the float-only branch of rows, which skips the int/choice conversion
    (P-T1-44). Hand calculation: 0 + 0.25 * 1 = 0.25 and 10 + 0.5 * 10 = 15.0. Postcondition: none (pure).
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(10.0, 20.0)})

    keys, rows = space.rows(np.array([[0.25, 0.5]]))

    assert rows == [{"a": 0.25, "b": 15.0}]
    assert _types(rows[0]) == [float, float]
    assert keys == [tuple(rows[0].values())]


def test_space_to_dicts_should_decode_each_column_by_its_own_kind_when_kinds_are_interleaved():
    """Given a Space ordered Float, Int, Float, Categorical, Int, when to_dicts runs, then each column has its
    own kind's value and type.

    Oracle: README table and the space module docstring (columns of the same kind are decoded together, so
    interleaving tests the column grouping, P-T1-44). Hand calculation: 0.3 -> 0.3; floor(0.55 * 10) = 5;
    10 + 0.8 * 10 = 18.0; floor(0.7 * 2) = 1 -> 'y'; -3 + floor(0.1 * 7) = -3. Postcondition: none (pure).
    """
    space = Space(
        {
            "a": Float(0.0, 1.0),
            "b": Int(0, 9),
            "c": Float(10.0, 20.0),
            "d": Categorical(["x", "y"]),
            "e": Int(-3, 3),
        }
    )

    (params,) = space.to_dicts(np.array([[0.3, 0.55, 0.8, 0.7, 0.1]]))

    assert params == {"a": 0.3, "b": 5, "c": 18.0, "d": "y", "e": -3}
    assert _types(params) == [float, int, float, str, int]


def test_space_should_have_dimension_one_when_it_has_one_parameter():
    """Given one Float parameter, when a Space is built from it, then its dim is 1.

    Oracle: Space docstring 'dimension d = number of parameters'. One element is the minimal valid collection
    (P-T2-33); the empty mapping is rejected in tests/test_space.py. Postcondition: none.
    """
    parameters = {"a": Float(0.0, 1.0)}

    space = Space(parameters)

    assert space.dim == 1


def test_space_sample_should_return_one_column_in_the_unit_interval_when_the_space_has_one_parameter():
    """Given a one-parameter Space and a seeded generator, when sample(rng, 5) runs, then the points are a
    (5, 1) array in [0, 1).

    Oracle: Space.sample 'n uniform points of the unit cube, shape (n, d)' with d = 1, the minimal valid
    collection (P-T2-33). ANY_SEED is fixed only for determinism, the assertions hold for every seed. Both
    assertions are properties of the one returned array. Postcondition: none (the generator advances).
    """
    space = Space({"a": Float(0.0, 1.0)})
    rng = np.random.default_rng(ANY_SEED)

    points = space.sample(rng, 5)

    assert points.shape == (5, 1)
    assert bool(np.all((points >= 0.0) & (points < 1.0)))


def test_space_to_dicts_should_give_the_same_dict_when_parameter_order_and_columns_are_permuted_together():
    """Given the same two parameters declared in both orders, when to_dicts gets the columns in matching
    order, then the dicts are equal.

    Oracle: Space docstring 'an ordered set of named parameters': a name keeps its parameter whatever its
    position, so a permutation of declaration and columns must not change the result (P-T2-33). Hand
    calculation: a = 0 + 0.3 * 10 = 3.0, b = floor(0.7 * 3) = 2 -> 'r'. Postcondition: none (pure).
    """
    a, b = Float(0.0, 10.0), Categorical(["p", "q", "r"])
    first, second = Space({"a": a, "b": b}), Space({"b": b, "a": a})

    result = (first.to_dicts(np.array([[0.3, 0.7]])), second.to_dicts(np.array([[0.7, 0.3]])))

    assert result[0] == result[1]
    assert result[0] == [{"a": 3.0, "b": "r"}]


INT_SHARE_CASES = [
    pytest.param(0.0, 0, id="domain-lower-edge"),
    pytest.param(0.3333, 0, id="just-below-first-share-boundary-kills-round-instead-of-floor"),
    pytest.param(0.3334, 1, id="just-above-first-share-boundary-kills-missing-plus-one-in-width"),
    pytest.param(0.6666, 1, id="just-below-second-share-boundary"),
    pytest.param(0.6667, 2, id="just-above-second-share-boundary"),
    pytest.param(0.9999, 2, id="top-share"),
    pytest.param(1.0, 2, id="domain-upper-edge-kills-missing-clip"),
]


SHARE_START_U = 0.0


@pytest.mark.parametrize(("u", "expected"), INT_SHARE_CASES)
def test_int_decode_should_map_u_to_its_equal_share_integer_when_u_straddles_a_share_boundary(u, expected):
    """Given Int(0, 2) and a unit value near a third, when Int.decode runs, then it returns floor(3u) clipped.

    Oracle: Int docstring (equal share per integer). 0.3333/0.3334 and 0.6666/0.6667 lie on both sides of a
    share boundary (P-T1-40), 0.0 and 1.0 are the implicit domain edges (P-T1-42). floor -> round gives 1 at
    0.3333, (high - low + 1) -> (high - low) gives 0 at 0.3334, and a missing clip gives 3 at 1.0.
    Site: array decode of Int. Postcondition: none (pure function).
    """
    parameter = Int(0, 2)

    result = parameter.decode(np.array([u]))

    assert result.tolist() == [expected]


@pytest.mark.parametrize(("u", "expected"), INT_SHARE_CASES)
def test_space_decode_should_give_the_share_integer_as_whole_float_when_u_straddles_a_share_boundary(
    u, expected
):
    """Given Space({'x': Int(0, 2)}), when Space.decode runs on one point, then the raw value is that integer.

    Oracle: Int docstring and Space.decode docstring 'integers as whole numbers'; the same hand-derived
    thirds as the Int.decode test. Site: the grouped decode of Space, which can differ from Int.decode.
    Postcondition: none (pure function).
    """
    space = Space({"x": Int(0, 2)})

    result = space.decode(np.array([[u]]))

    assert result.tolist() == [[float(expected)]]


@pytest.mark.parametrize(("u", "expected"), INT_SHARE_CASES)
def test_space_to_dicts_should_give_the_share_integer_as_python_int_when_u_straddles_a_share_boundary(
    u, expected
):
    """Given Space({'x': Int(0, 2)}), when to_dicts runs on one point, then x is that integer as a Python int.

    Oracle: Int docstring and the to_dicts docstring (the dictionaries the user's function receives); the
    same hand-derived thirds as the Int.decode test. Postcondition: none (pure function).
    """
    space = Space({"x": Int(0, 2)})

    (params,) = space.to_dicts(np.array([[u]]))

    assert params == {"x": expected}
    assert type(params["x"]) is int


@pytest.mark.parametrize(("u", "expected"), INT_SHARE_CASES)
def test_space_move_should_decode_the_new_coordinate_to_its_share_integer_when_u_straddles_a_share_boundary(
    u, expected
):
    """Given a decoded row of Space({'x': Int(0, 2)}), when move sets coordinate 0 to u, then x is that
    integer.

    Oracle: Space.move docstring (the point differs only in coordinate j and is decoded like a full decode)
    and the Int docstring. Site: the scalar one-coordinate decode, which is separate from the array decode,
    so a mutant there is only caught here. The start row has x = 0, so u = 0.0 checks a no-change move.
    Postcondition: the returned key holds the same whole-number raw value.
    """
    space = Space({"x": Int(0, 2)})
    keys, rows = space.rows(np.array([[SHARE_START_U]]))

    new_key, new_params = space.move(keys[0], rows[0], 0, u)

    assert new_params == {"x": expected}
    assert type(new_params["x"]) is int
    assert new_key == (float(expected),)
