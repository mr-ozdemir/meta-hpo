"""Hand-written Space examples behind the generated properties of test_fuzz_grammar.py.

The grammar-derived tests compare move with rows and keys with decode on random declarations; these examples
pin the same properties to literal expected values computed by hand from the spec, so the generated tests
are not the only check of Space.move and Space.rows keys.

Oracles: Space.move docstring ("(key, params) of a point differing from a decoded point only in coordinate
j"), Space.rows docstring (a key is the hashable raw row: equal keys mean the user's function sees equal
parameters; params have Python types), Space.decode docstring (integers as whole numbers, categories as
indices), and the Int and Categorical docstrings (every value gets an equal share of the unit interval, so
bucket k of n values is [k/n, (k+1)/n)). Float(0.0, 10.0) decodes 0.5 to 0 + 0.5 * 10 = 5.0 exactly.

Unspecified, therefore not asserted: whether move leaves the params dict it receives unchanged.

Input space: Every row of d unit-cube coordinates in [0, 1] for every parameter kind and bound, which is
unbounded.
Subset: Three hand-written declarations behind the generated properties.
"""

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space

SOURCE_U = 0.1


@pytest.mark.parametrize(
    ("u_j", "expected"),
    [
        pytest.param(0.24999, 0, id="just-below-first-edge"),
        pytest.param(0.25, 1, id="first-edge"),
        pytest.param(0.5, 2, id="middle-edge"),
        pytest.param(0.74999, 2, id="just-below-last-edge"),
        pytest.param(0.75, 3, id="last-edge"),
    ],
)
def test_space_move_should_give_the_int_of_the_equal_share_bucket_that_u_j_falls_in(u_j, expected):
    """Given Space({'a': Int(0, 3)}) decoded at SOURCE_U (value 0), when move sets coordinate 0 to u_j, then
    the key holds the whole float and params the Python int of bucket floor(4 * u_j).

    Oracle: Int docstring, four values with shares of width 0.25: 0.5 lies in [0.5, 0.75), value 2. Both
    sides of the edges 0.25 and 0.75 are used (site: Int.decode_one floor). Postcondition: none checked
    beyond the returned pair.
    """
    space = Space({"a": Int(0, 3)})
    keys, params = space.rows(np.array([[SOURCE_U]]))

    key, moved = space.move(keys[0], params[0], 0, u_j)

    assert (key, moved) == ((float(expected),), {"a": expected})
    assert type(moved["a"]) is int


def test_space_move_should_change_only_coordinate_j_when_the_space_has_other_coordinates():
    """Given Space({'x': Float(0.0, 10.0), 'k': Categorical(['a', 'b', 'c'])}) decoded at (0.5, 0.1), which is
    (5.0, 'a'), when move sets coordinate 1 to 0.9, then only k changes, to 'c' with raw index 2.0.

    Oracle: Space.move docstring; Categorical equal shares, 0.9 lies in [2/3, 1), index 2; Float linear
    0 + 0.5 * 10 = 5.0. A move that rewrites or drops the untouched coordinate fails (site: Space.move key
    splice and params copy).
    """
    space = Space({"x": Float(0.0, 10.0), "k": Categorical(["a", "b", "c"])})
    keys, params = space.rows(np.array([[0.5, 0.1]]))

    key, moved = space.move(keys[0], params[0], 1, 0.9)

    assert (key, moved) == ((5.0, 2.0), {"x": 5.0, "k": "c"})


def test_space_rows_should_give_equal_keys_when_different_points_decode_to_equal_params():
    """Given Space({'n': Int(1, 3), 'k': Categorical(['a', 'b'])}), when rows decodes (0.0, 0.0), (0.3, 0.4)
    and (0.5, 0.75), then the first two points get the same key and params and the third a different one.

    Oracle: Space.rows docstring (equal keys mean equal parameters, keys are hashable) and the equal shares:
    Int(1, 3) maps 0.0 and 0.3 (0.9 < 1) to 1 and 0.5 (1.5) to 2; Categorical(['a', 'b']) maps 0.0 and 0.4
    to 'a' and 0.75 to 'b'. The two different points with equal decodes are decisive: a key built from u
    instead of the decoded row would differ (site: Space.rows keys).
    """
    space = Space({"n": Int(1, 3), "k": Categorical(["a", "b"])})
    u = np.array([[0.0, 0.0], [0.3, 0.4], [0.5, 0.75]])

    keys, params = space.rows(u)

    assert keys == [(1.0, 0.0), (1.0, 0.0), (2.0, 1.0)]
    assert params == [{"n": 1, "k": "a"}, {"n": 1, "k": "a"}, {"n": 2, "k": "b"}]
    assert len(set(keys)) == 2
