"""Coverage-driven tests of the search space: statements, branches and the stdlib coverage measurement.

Space tests added after the stdlib coverage measurement: Int bounds, log Int, empty Space, Categorical, move.

They close the red lines and one-sided branches of space.py found by the baseline trace and branch probe:
lines 101 and 103 with all four Int.__post_init__ edges, the log branches of _decode_int (lines 31-32),
Int.kind, Float.decode_one (lines 83-84) and Int.decode_one (lines 114-115), the direct Categorical.decode
(line 140) and the empty-mapping check of Space (line 155, edge 154L).

Oracles: the README search-space table (Int: integers low..high inclusive, log option), the Int and
Categorical docstrings (every integer / choice gets an equal share of the unit interval), the Space.move
docstring ('(key, params) of a point differing from a decoded point only in coordinate j') with Space.rows as
the reference decoding, and the task specification (invalid parameters raise ValueError).

Unspecified, therefore not asserted: the exact rounding of log-scaled integers inside the range (only
bounds, ends and monotonicity are checked), and bit-exact equality of log Float values between move and rows
(math.exp and numpy exp may differ in the last bit, so those are compared with a relative tolerance).

Unit tests of Space and its parameter types (space.py) for statements the repository tests do not reach.

Grouped by method: Float.decode_one and Int.decode_one through Space.move (log scale), Int.decode (log
scale ends, single-value range), Int.__post_init__, Categorical.decode, Space.__init__, Space.to_dicts.

Unspecified, not asserted: the exact share of each integer under Int(log=True).
Known library defect, kept out of this module: a log-scaled Float can decode one float step above high
(README states [low, high]; Float decode and decode_one have no clip), e.g. Float(1e-4, 1e-2, log=True)
at u=1.0 gives 0.010000000000000004.
Excluded from statement coverage: the abstract Parameter methods, because Parameter is not public API.

Branch-coverage tests of Space.decode (with _decode_float and _decode_int) and Space.rows.

Control flow graphs (course notation: nodes are the real line numbers of src/metahpo/space.py, straight-line
code is merged, every decision has one T and one F edge, s and e are entry and exit; loops use the for
template, a loop header is T when it enters the body and F when the items are exhausted):

Space.decode, space.py 183-195
    s   -> 185-186 -> 187     u as a 2-d float array, raw = empty_like(u)
    187 -T-> 188 -> 189       for: next kind group (its block of columns)
    187 -F-> 195 -> e         for: groups exhausted, return raw
    189 -T-> 190 -> 187       float or float_log: _decode_float
    189 -F-> 191
    191 -T-> 192 -> 187       int or int_log: _decode_int
    191 -F-> 194 -> 187       categorical: _decode_categorical

_decode_float, space.py 22-26
    s   -> 23
    23  -T-> 24-25 -> e       log: exp of the linear map of the logs
    23  -F-> 26 -> e          linear map

_decode_int, space.py 29-35
    s   -> 30
    30  -T-> 31-32 -> 35      log: round on the log scale widened by 0.5
    30  -F-> 34 -> 35         low + floor(u * (high - low + 1))
    35  -> e                  clip to [low, high]

Space.rows, space.py 197-212
    s   -> 203-204 -> 205     raw = decode(u).tolist(), keys = tuples of raw
    205 -T-> 206              not plain: an Int or Categorical column exists
    205 -F-> 211-212 -> e     params straight from raw
    206 -T-> 207              for r: next row
    206 -F-> 211-212 -> e     rows exhausted
    207 -T-> 208 -> 207       for j in Int columns: int(r[j])
    207 -F-> 209
    209 -T-> 210 -> 209       for j in Categorical columns: choices[int(r[j])]
    209 -F-> 206

Test requirements (branch coverage, one per T/F edge):
    SP-DEC-1  187 T     SP-DEC-2  187 F     SP-DEC-3  189 T     SP-DEC-4  189 F     SP-DEC-5  191 T
    SP-DEC-6  191 F     SP-DEC-7   23 T     SP-DEC-8   23 F     SP-DEC-9   30 T     SP-DEC-10  30 F
    SP-ROWS-1 205 T     SP-ROWS-2 205 F     SP-ROWS-3 206 T     SP-ROWS-4 206 F     SP-ROWS-5 207 T
    SP-ROWS-6 207 F     SP-ROWS-7 209 T     SP-ROWS-8 209 F

Coverage level: 18 of 18 feasible branch requirements (100 percent), all by this module.

Infeasible: SP-DEC-2 on the first check of line 187, because Space rejects an empty mapping (space.py 154-155)
and so has at least one group; SP-DEC-2 is covered at loop exit. Out of this module's scope: Space.__init__,
the Parameter classes, Space.move and Space.sample (other work items test them).

Unspecified, therefore not asserted: the exact interior values of a log-scaled Int (the README and the Int
docstring give only 'integers low..high, inclusive'; only the log-scale property is checked, as a wide band
around the geometric middle), and decode of an empty (0, d) array.

Box types (P-T4-08): white-box tests force one edge each with a single-kind space; black-box tests check the
rows and decode docstrings and the README on mixed spaces.

Input space: Every row of d unit-cube coordinates in [0, 1] for every parameter kind and bound, which is
unbounded.
Subset: One scenario per statement or branch that the coverage measurement and the branch inventory found
unreached.
"""

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space

MIXED_SPACE_PARAMETERS = {
    "f": Float(-2.0, 3.0),
    "fl": Float(1e-4, 1e-1, log=True),
    "i": Int(-3, 7),
    "il": Int(2, 300, log=True),
    "c": Categorical(["x", "y", "z"]),
}


START_POINT = [0.21, 0.43, 0.65, 0.37, 0.59]


MOVE_TARGETS = [
    pytest.param(0.0, id="u_lower_edge"),
    pytest.param(0.3, id="u_interior"),
    pytest.param(0.5, id="u_middle"),
    pytest.param(0.999, id="u_just_below_top"),
    pytest.param(1.0, id="u_upper_edge_needs_clip"),
]


def _types(params):
    """The exact type of every value of a params dict."""
    return {name: type(value) for name, value in params.items()}


def _types_of_rows(rows):
    """_types of every params dict in a list."""
    return [_types(params) for params in rows]


def _start_row(space):
    """(key, params) of START_POINT decoded by Space.rows."""
    (key,), (params,) = space.rows(np.array([START_POINT]))
    return key, params


def _rows_with_coordinate(space, j, u):
    """Reference (key, params): Space.rows of START_POINT with coordinate j set to u."""
    moved_x = np.array([START_POINT])
    moved_x[0, j] = u
    (key,), (params,) = space.rows(moved_x)
    return key, params


@pytest.mark.parametrize(
    ("low", "high", "log"),
    [
        pytest.param(5, 4, False, id="low_above_high"),
        pytest.param(0, 5, True, id="log_with_low_zero_boundary"),
        pytest.param(-1, 5, True, id="log_with_negative_low"),
    ],
)
def test_int_should_raise_value_error_when_bounds_or_log_low_are_invalid(low, high, log):
    """Given low > high, or log=True with low <= 0, when Int is built, then ValueError is raised.

    Oracle: README 'integers low..high, inclusive' (needs low <= high), a log scale needs positive values,
    and the task specification 'invalid parameters raise ValueError'. Inputs: one invalid value per case;
    low = 0 is the boundary of the log condition. Sites: space lines 101 and 103.
    """
    with pytest.raises(ValueError):
        Int(low, high, log=log)


def test_int_should_decode_the_unit_interval_ends_to_low_and_high_when_log_low_is_one():
    """Given Int(1, 5, log=True), when u = 0 and u = 1 are decoded, then 1 and 5.

    Oracle: README 'integers low..high, inclusive' with the log option; the ends of the unit interval map to
    the ends of the range. Input: low = 1 is the valid side of the log boundary low > 0 (site: the fourth
    Int.__post_init__ edge). Postcondition: none, decode is pure.
    """
    parameter = Int(1, 5, log=True)

    values = parameter.decode(np.array([0.0, 1.0]))

    assert values.tolist() == [1, 5]


def test_int_should_decode_every_u_to_its_only_value_when_low_equals_high():
    """Given Int(5, 5), when u from 0 to 1 is decoded, then every value is 5.

    Oracle: README 'integers low..high, inclusive' and the Int docstring (the only integer owns the whole
    unit interval). Input: low == high is the valid boundary of low <= high; u = 1.0 checks the upper clip.
    """
    parameter = Int(5, 5)

    values = parameter.decode(np.array([0.0, 0.5, 0.999, 1.0]))

    assert values.tolist() == [5, 5, 5, 5]


def test_int_should_decode_to_non_decreasing_integers_from_low_to_high_when_log_scaled():
    """Given Int(1, 1000, log=True), when a 1001-point grid of u is decoded, then integers rise from 1 to
    1000.

    Oracle: README Int row (integers low..high inclusive, log option); a log scale is monotone. Inputs: the
    grid includes u = 0 and u = 1, where the unrounded log value lies half a unit outside the range. Sites:
    _decode_int log branch (space lines 31-32).
    """
    parameter = Int(1, 1000, log=True)

    values = parameter.decode(np.linspace(0.0, 1.0, 1001))

    assert values.dtype.kind == "i"
    assert (int(values[0]), int(values[-1])) == (1, 1000)
    assert (int(values.min()), int(values.max())) == (1, 1000)
    assert bool(np.all(np.diff(values) >= 0)) is True


def test_space_to_dicts_should_give_python_ints_at_low_and_high_when_int_is_log_scaled():
    """Given Space({'n': Int(1, 1000, log=True)}), when u = 0 and u = 1 go through to_dicts, then int 1 and
    1000.

    Oracle: README (the space maps each point to typed parameters; Int gives integers) and the Space.rows
    docstring 'params ... with Python types'. Site: the int_log kind of the grouped decoder (Int.kind log
    side).
    """
    space = Space({"n": Int(1, 1000, log=True)})

    params = space.to_dicts(np.array([[0.0], [1.0]]))

    assert params == [{"n": 1}, {"n": 1000}]
    assert _types_of_rows(params) == [{"n": int}, {"n": int}]


def test_space_should_raise_value_error_when_the_mapping_is_empty():
    """Given an empty mapping, when Space is built, then ValueError is raised.

    Oracle: task specification 'invalid parameters raise ValueError'; a space of dimension 0 has no point to
    search. Input: the empty mapping is the zero class. Site: space line 155 (edge 154L).
    """
    with pytest.raises(ValueError):
        Space({})


def test_categorical_decode_should_give_the_index_of_the_equal_share_and_the_last_index_for_u_one():
    """Given Categorical(['a', 'b', 'c']), when u = 0, 0.34, 0.67, 1.0 are decoded, then indices 0, 1, 2, 2.

    Oracle: Categorical docstring 'every choice gets an equal share of the unit interval' (shares [0, 1/3),
    [1/3, 2/3), [2/3, 1]). Inputs: 0.34 and 0.67 sit just above the share boundaries, u = 1.0 must stay on the
    last choice. Site: direct Categorical.decode (space line 140). Postcondition: none, decode is pure.
    """
    parameter = Categorical(["a", "b", "c"])

    indices = parameter.decode(np.array([0.0, 0.34, 0.67, 1.0]))

    assert indices.tolist() == [0, 1, 2, 2]


@pytest.mark.parametrize("u", MOVE_TARGETS)
@pytest.mark.parametrize(
    "column",
    [pytest.param("i", id="int"), pytest.param("il", id="int_log"), pytest.param("c", id="categorical")],
)
def test_space_move_should_equal_rows_of_the_moved_point_exactly_when_the_column_is_int_or_categorical(
    column, u
):
    """Given a point of a mixed space, when one Int, log Int or Categorical coordinate moves to u, then move
    equals rows.

    Oracle: Space.move docstring (key and params of the point differing only in coordinate j), with Space.rows
    as the reference decoding; integer and category results are exact. Inputs: the unit edges (u = 1.0 needs
    the clip or clamp), a value just below the top and interior values. Sites: Int.decode_one log branch
    (space lines 114-115) and the other decode_one branches.
    """
    space = Space(MIXED_SPACE_PARAMETERS)
    key, params = _start_row(space)
    j = list(MIXED_SPACE_PARAMETERS).index(column)
    reference_key, reference_params = _rows_with_coordinate(space, j, u)

    moved_key, moved_params = space.move(key, params, j, u)

    assert moved_key == reference_key
    assert moved_params == reference_params
    assert _types(moved_params) == _types(reference_params)


@pytest.mark.parametrize("u", MOVE_TARGETS)
@pytest.mark.parametrize("column", [pytest.param("f", id="float"), pytest.param("fl", id="float_log")])
def test_space_move_should_equal_rows_of_the_moved_point_within_rounding_when_the_column_is_float(column, u):
    """Given a point of a mixed space, when one Float or log Float coordinate moves to u, then move matches
    rows.

    Oracle: Space.move docstring with Space.rows as the reference. Tolerance: relative 1e-12, because the
    one-value path uses math.exp and the vectorised path numpy exp (unspecified last-bit difference). Sites:
    Float.decode_one log branch (space lines 83-84).
    """
    space = Space(MIXED_SPACE_PARAMETERS)
    key, params = _start_row(space)
    j = list(MIXED_SPACE_PARAMETERS).index(column)
    reference_key, reference_params = _rows_with_coordinate(space, j, u)

    moved_key, moved_params = space.move(key, params, j, u)

    assert moved_key == pytest.approx(reference_key, rel=1e-12)
    assert moved_params == pytest.approx(reference_params, rel=1e-12)
    assert _types(moved_params) == _types(reference_params)


GRID_POINTS = 201


START_U = 0.5


def _unit_grid(points):
    """A fresh grid of `points` evenly spaced values over the closed unit interval [0, 1]."""
    return np.linspace(0.0, 1.0, points)


def _moved_and_decoded(space, name, grid):
    """For every u of grid: the key and the params value of Space.move from the decoded START_U point to u in
    coordinate 0 (parameter `name`), and the key and params value of Space.rows for u decoded from scratch,
    as four lists."""
    keys, params = space.rows(np.array([[START_U]]))
    moved = [space.move(keys[0], params[0], 0, float(u)) for u in grid]
    full_keys, full_params = space.rows(grid.reshape(-1, 1))
    return (
        [k for k, _ in moved],
        [p[name] for _, p in moved],
        full_keys,
        [p[name] for p in full_params],
    )


def _python_types(dicts, names):
    """Per dict, the tuple of the Python types of the values under `names`."""
    return [tuple(type(d[n]) for n in names) for d in dicts]


def test_space_move_should_match_full_decode_when_float_is_log_scaled():
    """Given Space({'lr': Float(1e-3, 1.0, log=True)}), when move() sets coordinate 0 to each u of a 201-point
    grid, then key and params agree with rows() of the same point and every moved value lies in [1e-3, 1.0].

    Oracle: Space.move docstring, move gives the (key, params) of the point that differs only in coordinate j,
    so it must agree with a full decode of that point. Inputs: the grid includes both ends; the start point
    u=0.5 differs from most grid points, and at u=0.5 a linear decode gives 0.5005 instead of about 0.0316
    (site: Float.decode_one log branch). Tolerance: math.log/exp and numpy log/exp may differ in the last bit.
    Bound oracle: README table 'Float(low, high) real numbers in [low, high]'; on this grid every decode stays
    inside the bounds (the out-of-bounds defect shows only for other ranges, see the module docstring).
    """
    space = Space({"lr": Float(1e-3, 1.0, log=True)})

    moved_keys, moved_values, full_keys, full_values = _moved_and_decoded(
        space, "lr", _unit_grid(GRID_POINTS)
    )

    np.testing.assert_allclose(moved_keys, full_keys, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(moved_values, full_values, rtol=1e-12, atol=0.0)
    assert min(moved_values) >= 1e-3
    assert max(moved_values) <= 1.0


def test_space_move_should_match_full_decode_when_int_is_log_scaled():
    """Given Space({'n': Int(1, 100, log=True)}), when move() sets coordinate 0 to each u of a 201-point grid,
    then key and params equal rows() of the same point exactly, and params hold Python ints.

    Oracle: Space.move docstring (agreement with a full decode); Space.rows docstring 'params ... with Python
    types'; integer decodes are exact. Inputs: the grid covers both ends and many rounding points, so a
    truncating instead of rounding single decode, or a vectorised decode with other share edges, is detected
    (sites: Int.decode_one log branch, log branch of the vectorised integer decode).
    """
    space = Space({"n": Int(1, 100, log=True)})

    moved_keys, moved_values, full_keys, full_values = _moved_and_decoded(space, "n", _unit_grid(GRID_POINTS))

    assert moved_keys == full_keys
    assert moved_values == full_values
    assert set(map(type, moved_values)) == {int}


def test_int_decode_should_map_unit_ends_to_low_and_high_when_log_scaled():
    """Given Int(1, 100, log=True), when decode([0.0, 1.0]) is called, then it returns [1, 100].

    Oracle: README table 'integers low..high, inclusive': both bounds are values of the parameter and u=0 and
    u=1 are the extreme points of the unit interval. Inputs: the two ends of [0, 1]. Postcondition: none.
    """
    param = Int(1, 100, log=True)

    values = param.decode(np.array([0.0, 1.0]))

    np.testing.assert_array_equal(values, np.array([1, 100]))


def test_int_decode_should_always_return_the_value_when_low_equals_high():
    """Given Int(3, 3), when decode([0.0, 0.5, 1.0]) is called, then every value is 3.

    Oracle: README 'integers low..high, inclusive'; with low == high, 3 is the only value. Inputs: low == high
    is the valid side of the low <= high boundary; u=1.0 is the closed upper end of the unit interval.
    """
    param = Int(3, 3)

    values = param.decode(np.array([0.0, 0.5, 1.0]))

    np.testing.assert_array_equal(values, np.array([3, 3, 3]))


def test_int_init_should_raise_value_error_when_low_is_above_high():
    """Given low = high + 1, when Int(4, 3) is built, then ValueError is raised.

    Oracle: README 'integers low..high, inclusive' (an empty range has no value) and task specification
    'Invalid parameters raise ValueError'. Input: one above the valid boundary low == high, decisive against
    a bound that allows low == high + 1 (site: Int.__post_init__ low/high check -> raise).
    """
    with pytest.raises(ValueError):
        Int(4, 3)


def test_int_init_should_raise_value_error_when_log_scaled_and_low_is_zero():
    """Given log=True and low=0, when Int(0, 10, log=True) is built, then ValueError is raised.

    Oracle: task specification 'Invalid parameters raise ValueError' and the Int message 'a log-scaled Int
    needs low > 0'. Input: 0 is just below the smallest valid low, 1; the valid side Int(1, 100, log=True)
    is used by the log tests above (site: Int.__post_init__ log check -> raise).
    """
    with pytest.raises(ValueError):
        Int(0, 10, log=True)


def test_categorical_decode_should_return_integer_indices_with_equal_shares():
    """Given Categorical(['a', 'b', 'c']), when decode is called on values around the share edges 1/3 and 2/3,
    then it returns the indices [0, 0, 1, 1, 2, 2] as integers.

    Oracle: Categorical docstring 'every choice gets an equal share of the unit interval' and the decode
    docstring 'categories as indices'. Inputs: 0.0 is the lower end, 0.33/0.34 and 0.66/0.67 lie just below
    and above the two share edges, and 1.0 is the closed upper end, which maps to the last index
    (site: Categorical.decode).
    """
    param = Categorical(["a", "b", "c"])

    indices = param.decode(np.array([0.0, 0.33, 0.34, 0.66, 0.67, 1.0]))

    assert np.issubdtype(indices.dtype, np.integer)
    np.testing.assert_array_equal(indices, np.array([0, 0, 1, 1, 2, 2]))


def test_space_to_dicts_should_return_one_dict_per_row_in_row_order():
    """Given a Float/Int/Categorical space and three different rows, when to_dicts() is called, then it
    returns three dicts in row order with float, int and choice values.

    Oracle: Space.to_dicts docstring (points (n, d) to the parameter dictionaries) and Space.rows docstring
    'with Python types'; hand calculation: Float(0, 10) gives 0, 5, 10; Int(1, 5) gives 1, 1 + floor(2.5) = 3
    and 5; Categorical(['a', 'b']) gives index 0, floor(1.0) = 1 and 1. Inputs: three distinct rows, so
    repeating the first row or returning only one dict is detected (site: Space.rows dict construction,
    Space.to_dicts). One parameter per type is needed to check all three Python types.
    """
    space = Space({"f": Float(0.0, 10.0), "n": Int(1, 5), "c": Categorical(["a", "b"])})

    dicts = space.to_dicts(np.array([[0.0] * 3, [0.5] * 3, [1.0] * 3]))

    assert dicts == [
        {"f": 0.0, "n": 1, "c": "a"},
        {"f": 5.0, "n": 3, "c": "b"},
        {"f": 10.0, "n": 5, "c": "b"},
    ]
    assert _python_types(dicts, ["f", "n"]) == [(float, int)] * 3


ROWS_SEED = 0


CORNERS = np.array([[0.0], [0.5], [1.0]])


def _all_types(values):
    """The set of exact types of `values` (type(v), so a numpy scalar is not taken for a Python one)."""
    return {type(v) for v in values}


def _param_values(params):
    """Every value of every params dict, in row order."""
    return [v for p in params for v in p.values()]


def _key_parts(keys):
    """Every component of every key, in row order."""
    return [part for key in keys for part in key]


def _column(params, name):
    """The value of parameter `name` in every params dict, in row order."""
    return [p[name] for p in params]


def _params_as_float_tuples(params, names):
    """Each params dict as a tuple of float(value) in the order of `names`."""
    return [tuple(float(p[name]) for name in names) for p in params]


def _decode_rows_as_tuples(space, u):
    """space.decode(u) as a list of row tuples, the documented form of a key."""
    return [tuple(row) for row in space.decode(u).tolist()]


def _equal_keys_iff_equal_params(keys, params):
    """For every pair of rows: (keys equal) == (params equal); returns the pairs where the iff fails."""
    n = len(keys)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return [(a, b) for a, b in pairs if (keys[a] == keys[b]) != (params[a] == params[b])]


def _count_equal_key_pairs_from_distinct_u(keys, u):
    """Number of row pairs whose keys are equal although their unit-cube rows differ."""
    n = len(keys)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return sum(keys[a] == keys[b] and not np.array_equal(u[a], u[b]) for a, b in pairs)


def _columns_decoded_alone(space, u):
    """Each column j of u decoded by a one-parameter Space holding only parameter j, stacked as (n, d)."""
    singles = [Space({name: p}) for name, p in zip(space.names, space.parameters, strict=True)]
    return np.hstack([single.decode(u[:, [j]]) for j, single in enumerate(singles)])


def test_space_rows_should_give_python_float_params_when_every_parameter_is_a_float():
    """An all-Float space returns params of exact type float and keys equal to the decoded rows.

    Given Space({'a': Float(0, 1), 'b': Float(1e-3, 1, log=True)}) (no Int or Categorical column, so the
    space is plain and 205 F is forced) and 5 rows drawn with default_rng(ROWS_SEED).
    When rows(u) is called.
    Then every params value has type float (np.float64 would fail, it only subclasses float), and the keys are
    the rows of decode(u) as tuples.
    Covers: SP-ROWS-2
    Box: white
    Source: Space.rows docstring ('params ... with Python types'; 'a key is the hashable raw row').
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(1e-3, 1.0, log=True)})
    u = np.random.default_rng(ROWS_SEED).random((5, 2))

    keys, params = space.rows(u)

    assert _all_types(_param_values(params)) == {float}
    assert keys == _decode_rows_as_tuples(space, u)


def test_space_rows_should_give_python_int_params_and_whole_float_keys_when_every_parameter_is_an_int():
    """An Int-only space returns int params and keys made of whole floats.

    Given Space({'n': Int(1, 5), 'm': Int(2, 64, log=True)}) and crafted rows [[0.0, 0.0], [0.5, 0.5],
    [1.0, 1.0]] (Int columns only: 205 T, the Int loop runs, the Categorical loop is empty).
    When rows(u) is called.
    Then every params value has type int, every key component has type float, and each key holds the same
    numbers as its params (the key is the raw row, the params its Python form).
    Covers: SP-ROWS-1, SP-ROWS-3, SP-ROWS-4, SP-ROWS-5, SP-ROWS-6, SP-ROWS-8
    Box: white
    Source: Space.rows docstring; Space.decode docstring ('integers as whole numbers'); README ('integers
    low..high, inclusive').
    """
    space = Space({"n": Int(1, 5), "m": Int(2, 64, log=True)})
    u = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    keys, params = space.rows(u)

    assert _all_types(_param_values(params)) == {int}
    assert _all_types(_key_parts(keys)) == {float}
    assert keys == _params_as_float_tuples(params, ("n", "m"))


def test_space_rows_should_give_the_choice_objects_and_float_index_keys_when_every_parameter_is_categorical():
    """A Categorical-only space returns the choice objects themselves and the float choice indices as keys.

    Given Space({'c': Categorical(['relu', 'tanh'])}) and crafted rows [[0.1], [0.6], [1.0]] (205 T, the Int
    loop is empty, the Categorical loop runs); u = 1.0 is the upper corner and must give the last choice.
    When rows(u) is called.
    Then the params are the choices 'relu', 'tanh', 'tanh' (the very objects of the declaration) and the keys
    are (0.0,), (1.0,), (1.0,) with float components.
    Covers: SP-ROWS-1, SP-ROWS-3, SP-ROWS-4, SP-ROWS-6, SP-ROWS-7, SP-ROWS-8
    Box: white
    Source: Space.rows docstring; Space.decode docstring ('categories as indices'); Categorical docstring.
    """
    choices = ["relu", "tanh"]
    space = Space({"c": Categorical(choices)})
    u = np.array([[0.1], [0.6], [1.0]])

    keys, params = space.rows(u)

    assert _column(params, "c") == ["relu", "tanh", "tanh"]
    assert params[0]["c"] is choices[0]
    assert keys == [(0.0,), (1.0,), (1.0,)]
    assert _all_types(_key_parts(keys)) == {float}


def test_space_rows_should_give_equal_keys_exactly_when_params_are_equal_for_a_mixed_space():
    """Equal keys mean equal parameters and unequal keys mean unequal parameters.

    Given an interleaved Space(Float, Int, Categorical, Float), 300 rows drawn with default_rng(ROWS_SEED) and
    three crafted rows: two that differ only inside one Int bucket (0.10 and 0.11 both give Int(1, 5) = 1, so
    their keys must be equal although u differs) and one that differs only in the category.
    When rows(u) is called.
    Then the keys are the decode rows as tuples, for every pair of rows keys are equal exactly when params are
    equal, and at least one pair has equal keys from different u rows (so the 'equal' side is not vacuous).
    Covers: SP-ROWS-1, SP-ROWS-3, SP-ROWS-4, SP-ROWS-5, SP-ROWS-6, SP-ROWS-7, SP-ROWS-8
    Box: black
    Source: Space.rows docstring ('equal keys mean the user's function sees equal parameters'); Candidate
    docstring (key is the decoded identity used for caching).
    """
    space = Space(
        {"f1": Float(0.0, 1.0), "i": Int(1, 5), "c": Categorical(["a", "b", "c"]), "f2": Float(0.0, 1.0)}
    )
    crafted = np.array([[0.3, 0.10, 0.5, 0.7], [0.3, 0.11, 0.5, 0.7], [0.3, 0.10, 0.9, 0.7]])
    u = np.vstack([np.random.default_rng(ROWS_SEED).random((300, 4)), crafted])

    keys, params = space.rows(u)

    assert keys == _decode_rows_as_tuples(space, u)
    assert _equal_keys_iff_equal_params(keys, params) == []
    assert _count_equal_key_pairs_from_distinct_u(keys, u) >= 1


@pytest.mark.parametrize(
    ("parameter", "expected"),
    [
        pytest.param(Float(-1.0, 3.0), [-1.0, 1.0, 3.0], id="SP-DEC-3-SP-DEC-8-linear-float"),
        pytest.param(Int(0, 4), [0.0, 2.0, 4.0], id="SP-DEC-4-SP-DEC-5-SP-DEC-10-linear-int"),
        pytest.param(Categorical(["w", "x", "y", "z"]), [0.0, 2.0, 3.0], id="SP-DEC-4-SP-DEC-6-categorical"),
    ],
)
def test_space_decode_should_follow_the_branch_of_the_parameter_kind_when_the_space_has_one_parameter(
    parameter, expected
):
    """A one-parameter space decodes the corners 0, 0.5 and 1 by its kind's rule.

    Given a one-parameter space of one kind (forcing one group: 187 T once, then 187 F) and u = [0, 0.5, 1].
    When decode(u) is called.
    Then Float(-1, 3) gives [-1, 1, 3] (189 T, 23 F); Int(0, 4) gives [0, 2, 4] (189 F, 191 T, 30 F; u = 1 is
    clipped to high); Categorical of 4 gives indices [0, 2, 3] (189 F, 191 F; u = 1 maps to the last index).
    All values are hand computed and exact.
    Covers: SP-DEC-1, SP-DEC-2, SP-DEC-3, SP-DEC-4, SP-DEC-5, SP-DEC-6, SP-DEC-8, SP-DEC-10
    Box: white
    Source: README Search space table; Int docstring ('low..high inclusive; every integer gets an equal
    share'); Categorical docstring; Space.decode docstring ('categories as indices').
    """
    space = Space({"p": parameter})

    raw = space.decode(CORNERS)

    assert raw.ravel().tolist() == expected


def test_space_decode_should_map_the_corners_to_the_bounds_when_the_float_is_log_scaled():
    """A log-scaled Float maps 0, 0.5 and 1 to low, the geometric middle and high.

    Given Space({'lr': Float(1e-3, 1e1, log=True)}) and u = [0, 0.5, 1] (189 T, 23 T).
    When decode(u) is called.
    Then the values are 1e-3, 1e-1 and 1e1 within a relative 1e-12 (real float arithmetic of exp and log).
    Covers: SP-DEC-1, SP-DEC-2, SP-DEC-3, SP-DEC-7
    Box: white
    Source: README Search space table (Float(1e-4, 1e-2, log=True) example: real numbers in [low, high] on
    a log scale).
    """
    space = Space({"lr": Float(1e-3, 1e1, log=True)})

    raw = space.decode(CORNERS)

    np.testing.assert_allclose(raw.ravel(), [1e-3, 1e-1, 1e1], rtol=1e-12)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        pytest.param(1, 7, id="narrow-range"),
        pytest.param(2, 256, id="wide-range"),
    ],
)
def test_space_decode_should_give_whole_numbers_inside_inclusive_bounds_when_the_int_is_log_scaled(low, high):
    """A log-scaled Int reaches both inclusive bounds at the cube corners and stays whole and inside them.

    Given Space({'n': Int(low, high, log=True)}) (189 F, 191 T, 30 T), the corners 0 and 1 and 500 values
    drawn with default_rng(ROWS_SEED).
    When decode is called.
    Then u = 0 gives low and u = 1 gives high, and every drawn value is a whole number in [low, high].
    Interior values are unspecified and not asserted.
    Covers: SP-DEC-1, SP-DEC-2, SP-DEC-4, SP-DEC-5, SP-DEC-9
    Box: black
    Source: README ('integers low..high, inclusive'); Space.decode docstring ('integers as whole numbers').
    """
    space = Space({"n": Int(low, high, log=True)})
    drawn = np.random.default_rng(ROWS_SEED).random((500, 1))

    corners = space.decode(np.array([[0.0], [1.0]])).ravel().tolist()
    values = space.decode(drawn).ravel()

    assert corners == [low, high]
    np.testing.assert_array_equal(values, np.round(values))
    assert values.min() >= low
    assert values.max() <= high


def test_space_decode_should_put_the_cube_middle_near_the_geometric_middle_when_the_int_is_log_scaled():
    """A log-scaled Int puts the cube middle near the geometric middle, not the linear one.

    Given Space({'n': Int(1, 10000, log=True)}) (189 F, 191 T, 30 T): the wide range makes the two scales far
    apart, the geometric middle sqrt(1 * 10000) = 100 against the linear middle about 5000.
    When u = 0.5 is decoded.
    Then the value lies in [50, 150], a band around the geometric middle wide enough for any rounding or
    bound-widening detail (unspecified); a linear mapping (30 F taken by mistake) would give about 5000.
    Covers: SP-DEC-1, SP-DEC-2, SP-DEC-4, SP-DEC-5, SP-DEC-9
    Box: black
    Source: Int signature (log=False flag, as in README Int(low, high, log=False)); space.py module docstring
    ('log integer' kind); README log example Float(1e-4, 1e-2, log=True); hand calculation.
    """
    space = Space({"n": Int(1, 10000, log=True)})

    value = space.decode(np.array([[0.5]])).item()

    assert 50.0 <= value <= 150.0


def test_space_decode_should_give_each_column_its_own_parameter_value_when_kinds_are_interleaved():
    """Decoding kinds together must not move or mix columns: column j equals parameter j decoded alone.

    Given an interleaved Space(Float, Int(log), Categorical, Float(log), Int, Float, Int) (five kinds, so
    187 T five times and every kind edge); the linear Float and the linear Int kinds each hold two columns
    with very different bounds (Float(-1, 3) and Float(10, 20), Int(0, 4) and Int(100, 300)), so a column
    moved or reordered inside its kind group decodes to values outside its own range; 200 rows drawn with
    default_rng(ROWS_SEED) plus an all-0 and an all-1 row.
    When decode(u) is called.
    Then the result equals, column by column, the decode of a one-parameter Space holding only that
    parameter (relational oracle: grouping by kind is an internal speed-up), within a relative 1e-12 for the
    float columns.
    Covers: SP-DEC-1, SP-DEC-2, SP-DEC-3, SP-DEC-4, SP-DEC-5, SP-DEC-6, SP-DEC-7, SP-DEC-8, SP-DEC-9,
    SP-DEC-10
    Box: black
    Source: space.py module docstring ('parameters of the same kind are decoded together ... one numpy
    expression per kind'); Space docstring ('an ordered set of named parameters').
    """
    space = Space(
        {
            "a": Float(-1.0, 3.0),
            "b": Int(1, 9, log=True),
            "c": Categorical([10, 20, 30]),
            "d": Float(1e-3, 1.0, log=True),
            "e": Int(0, 4),
            "f": Float(10.0, 20.0),
            "g": Int(100, 300),
        }
    )
    u = np.vstack([np.random.default_rng(ROWS_SEED).random((200, 7)), np.zeros(7), np.ones(7)])

    raw = space.decode(u)

    np.testing.assert_allclose(raw, _columns_decoded_alone(space, u), rtol=1e-12)
