"""Space.move against Space.rows: the one-coordinate decode the ABC neighbours rely on.

ABCOptimizer builds every neighbour's params and key with Space.move instead of decoding the whole row
again (abc.py module docstring: "only that coordinate is decoded again; the rest of the decoded parameters
are copied from the source"). A wrong move never raises; it only shows up later as a candidate whose params
do not match its x. These tests interrogate move's own return value, so such a defect fails right at the
method that holds it.

Oracle: Space.move docstring ("(key, params) of a point differing from a decoded point only in coordinate
j"), Space.rows docstring (keys are raw rows, params carry Python types), README search space table (Int
low..high inclusive, Categorical one of choices) and the Int/Categorical docstrings (every value gets an
equal share of the unit interval, so u = 1.0 lies in the last share). The expected value of every case is
rows() of the moved point, which is the documented full decode, never move's own formula.

Inputs: one Space holding all five decode kinds (float, float_log, int, int_log, categorical). SOURCE is a
literal point inside a share of every column. Every column gets the common class set: both closed ends 0.0
and 1.0, points 1e-12 inside each end, 0.5 and RANDOM_COUNT seeded uniforms (RANDOM_U_SEED is arbitrary and
fixed for reproducibility). Discrete columns also get both sides of every share edge (k/n and k/n - 1e-12),
and the log Int column both sides of several rounding edges (value m + 0.5), where a different rounding
rule than rows() would show. Since 0.0 and 1.0 decode to different values in every column, each column has
cases where move must really change the value.

Unspecified, not asserted:
- bit-exact agreement for log-scaled Float: move uses math.log/math.exp and rows numpy, so that column is
  compared with a relative tolerance of LOG_REL_TOL;
- what move returns for u outside [0, 1]; only the unit interval is used here.

Input space: Every row of d unit-cube coordinates in [0, 1] for every parameter kind and bound, which is
unbounded.
Subset: Every coordinate of small mixed spaces at edge and interior unit values, compared with rows().
"""

import math

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space

SOURCE = (0.3, 0.6, 0.5, 0.4, 0.1)
RANDOM_U_SEED = 1
RANDOM_COUNT = 20
EDGE_OFFSET = 1e-12
LOG_REL_TOL = 1e-12
INT_LOW, INT_HIGH = 3, 7
INT_LOG_LOW, INT_LOG_HIGH = 1, 100
CHOICES = ("a", "b", "c")
INT_LOG_ROUNDING_VALUES = (1, 9, 50, 99)

COMMON_US = [0.0, EDGE_OFFSET, 0.5, 1.0 - EDGE_OFFSET, 1.0] + np.random.default_rng(RANDOM_U_SEED).random(
    RANDOM_COUNT
).tolist()


def _share_edges(n):
    """Both sides of every share edge k/n of a discrete column with n values."""
    return [k / n for k in range(n + 1)] + [k / n - EDGE_OFFSET for k in range(1, n + 1)]


def _int_log_rounding_edges():
    """Both sides of the unit value that decodes to m + 0.5, the rounding edge between m and m + 1."""
    lo, hi = math.log(INT_LOG_LOW - 0.5), math.log(INT_LOG_HIGH + 0.5)
    edges = [(math.log(m + 0.5) - lo) / (hi - lo) for m in INT_LOG_ROUNDING_VALUES]
    return [e - EDGE_OFFSET for e in edges] + [e + EDGE_OFFSET for e in edges]


def _cases(column, kind, extra_us):
    """One case per distinct u of the common set plus the column's own edges."""
    return [pytest.param(column, u, id=f"{kind}-u{u!r}") for u in sorted(set(COMMON_US + extra_us))]


EXACT_CASES = (
    _cases("f", "float", [])
    + _cases("i", "int", _share_edges(INT_HIGH - INT_LOW + 1))
    + _cases("il", "int_log", _int_log_rounding_edges())
    + _cases("c", "categorical", _share_edges(len(CHOICES)))
)


def _five_kind_space():
    """A fresh Space with one column of every decode kind, in a fixed order."""
    return Space(
        {
            "f": Float(-2.0, 3.0),
            "fl": Float(1e-4, 1e-1, log=True),
            "i": Int(INT_LOW, INT_HIGH),
            "il": Int(INT_LOG_LOW, INT_LOG_HIGH, log=True),
            "c": Categorical(list(CHOICES)),
        }
    )


def _decoded_source(space):
    """The (key, params) that rows() gives for SOURCE."""
    keys, params = space.rows(np.array([SOURCE]))
    return keys[0], params[0]


def _rows_of_moved_point(space, j, u):
    """The (key, params) that rows() gives for SOURCE with coordinate j replaced by u: the expected value."""
    moved = np.array([SOURCE])
    moved[0, j] = u
    keys, params = space.rows(moved)
    return keys[0], params[0]


def _typed(params):
    """Each value paired with its exact type, so that 3 == 3.0 or a numpy scalar does not pass as equal."""
    return {name: (type(value), value) for name, value in params.items()}


def _key_without(key, j):
    """The key without element j."""
    return key[:j] + key[j + 1 :]


def _params_without(params, name):
    """The params without entry name."""
    return {n: v for n, v in params.items() if n != name}


@pytest.mark.parametrize(("column", "u"), EXACT_CASES)
def test_space_move_should_equal_rows_of_the_moved_point_when_the_column_decodes_exactly(column, u):
    """Given SOURCE decoded by rows(), when move() changes one exactly decoded column to u, then key and
    params equal rows() of the moved point.

    Oracle: Space.move and Space.rows docstrings; rows() of the moved point is the expected value. Inputs:
    see the module docstring (ends, near-ends, share edges, int_log rounding edges, seeded uniforms). A move
    that writes a neighbouring column, ignores u, keeps the old key, rounds differently at an edge, or
    returns a numpy scalar or a category index instead of the choice fails the comparison (site:
    Parameter.decode_one and to_python behind Space.move). Postcondition checked: the whole key and the whole
    params dict, values and exact types.
    """
    space = _five_kind_space()
    j = space.names.index(column)
    source_key, source_params = _decoded_source(space)
    expected_key, expected_params = _rows_of_moved_point(space, j, u)

    key, params = space.move(source_key, source_params, j, u)

    assert key == expected_key
    assert _typed(params) == _typed(expected_params)


@pytest.mark.parametrize("u", [pytest.param(u, id=f"float_log-u{u!r}") for u in COMMON_US])
def test_space_move_should_match_rows_within_rounding_when_the_column_is_log_float(u):
    """Given SOURCE decoded by rows(), when move() changes the log Float column to u, then it matches
    rows() of the moved point to LOG_REL_TOL.

    Oracle: Space.move and Space.rows docstrings. The two decodes use different log/exp implementations,
    so only this column gets a tight relative tolerance; every other column of key and params, copied from
    the source, must be exactly equal, and the moved value must be a Python float.
    """
    space = _five_kind_space()
    j = space.names.index("fl")
    source_key, source_params = _decoded_source(space)
    expected_key, expected_params = _rows_of_moved_point(space, j, u)

    key, params = space.move(source_key, source_params, j, u)

    assert _key_without(key, j) == _key_without(expected_key, j)
    assert key[j] == pytest.approx(expected_key[j], rel=LOG_REL_TOL)
    assert _typed(_params_without(params, "fl")) == _typed(_params_without(expected_params, "fl"))
    assert type(params["fl"]) is float
    assert params["fl"] == pytest.approx(expected_params["fl"], rel=LOG_REL_TOL)


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        pytest.param("i", INT_HIGH, id="int-high-inclusive"),
        pytest.param("il", INT_LOG_HIGH, id="int_log-high-inclusive"),
        pytest.param("c", CHOICES[-1], id="categorical-last-choice"),
    ],
)
def test_space_move_should_decode_the_upper_end_of_the_unit_interval_to_the_last_value(column, expected):
    """Given SOURCE, when move() sets a discrete column to exactly 1.0, then the value is high or the last
    choice.

    Oracle: README search space table (Int low..high inclusive, Categorical one of choices) and the
    Int/Categorical docstrings (equal shares, so the closed end 1.0 belongs to the last share). 1.0 is the
    discriminating boundary: floor(1.0 * n) is n, one past the last value, so a move without the clamp gives
    high + 1 or an IndexError (site: Int.decode_one / Categorical.decode_one clamp).
    """
    space = _five_kind_space()
    j = space.names.index(column)
    source_key, source_params = _decoded_source(space)

    key, params = space.move(source_key, source_params, j, 1.0)

    assert params[column] == expected
    assert type(params[column]) is type(expected)


def test_space_move_should_leave_the_source_params_unchanged_when_it_changes_a_value():
    """Given SOURCE decoded to category 'a', when move() sets the categorical column to 1.0, then the
    source params dict still holds 'a'.

    Oracle (derived): abc.py module docstring, the rest of a neighbour's params is "copied from the source",
    and ABCOptimizer keeps the source's params for later neighbours, so move must not write into them.
    Input: SOURCE[4] = 0.1 lies in the first share and 1.0 in the last, so the moved value really differs
    and an in-place write would show. Postcondition: the returned dict is a new object holding 'c'.
    """
    space = _five_kind_space()
    j = space.names.index("c")
    source_key, source_params = _decoded_source(space)
    snapshot = dict(source_params)

    key, params = space.move(source_key, source_params, j, 1.0)

    assert source_params == snapshot
    assert params["c"] == CHOICES[-1]
    assert params is not source_params
