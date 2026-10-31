"""Space.rows() for every mix of column kinds: no, one and several Int and Categorical columns.

Oracles: the Space.rows docstring (a key is the hashable raw row; params are the dictionaries the user's
function receives, with Python types), the Space.decode docstring and task specification (integers as whole
floats, categories as indices) and the README parameter table (Int low..high inclusive with an equal share
of [0, 1] per integer, Categorical one of choices with an equal share per choice). Expected values are
hand-computed from the equal shares: Int(1, 4) has shares [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1];
Categorical with n choices maps u to index floor(u * n). All u values are dyadic, so every Float decode is
exact.

The cases run the Int and Categorical conversion loops of rows() 0 times (Float only), once and several
times. Expected types are checked with type(v) so that numpy scalars, and floats in place of ints, fail; the
mixed choices [3, 5.5, None] show that the original choice object comes back, not a converted value.

Unspecified, therefore not asserted: rows() of zero points (shape (0, d)) and tell([], []).

Input space: Every row of d unit-cube coordinates in [0, 1] for every parameter kind and bound, which is
unbounded.
Subset: Spaces with no, one and several Int and Categorical columns.
"""

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space

ROWS_CASES = [
    pytest.param(
        {"a": Float(0.0, 1.0), "b": Float(-1.0, 1.0)},
        [[0.25, 0.5], [0.5, 0.75], [0.0, 0.25]],
        [(0.25, 0.0), (0.5, 0.5), (0.0, -0.5)],
        [{"a": 0.25, "b": 0.0}, {"a": 0.5, "b": 0.5}, {"a": 0.0, "b": -0.5}],
        [[float, float]] * 3,
        id="floats-only-no-int-or-categorical-column",
    ),
    pytest.param(
        {"n": Int(1, 4), "a": Float(0.0, 1.0)},
        [[0.0, 0.25], [0.5, 0.5], [0.875, 0.0]],
        [(1.0, 0.25), (3.0, 0.5), (4.0, 0.0)],
        [{"n": 1, "a": 0.25}, {"n": 3, "a": 0.5}, {"n": 4, "a": 0.0}],
        [[int, float]] * 3,
        id="one-int-column",
    ),
    pytest.param(
        {"c": Categorical(["relu", "tanh"]), "a": Float(0.0, 1.0)},
        [[0.25, 0.25], [0.75, 0.5], [0.5, 0.0]],
        [(0.0, 0.25), (1.0, 0.5), (1.0, 0.0)],
        [{"c": "relu", "a": 0.25}, {"c": "tanh", "a": 0.5}, {"c": "tanh", "a": 0.0}],
        [[str, float]] * 3,
        id="one-categorical-column",
    ),
    pytest.param(
        {
            "n": Int(1, 4),
            "m": Int(10, 13),
            "c": Categorical(["relu", "tanh"]),
            "k": Categorical([3, 5.5, None]),
            "a": Float(0.0, 1.0),
        },
        [[0.0, 0.25, 0.25, 0.0, 0.25], [0.5, 0.5, 0.75, 0.5, 0.5], [0.875, 0.875, 0.5, 0.875, 0.0]],
        [(1.0, 11.0, 0.0, 0.0, 0.25), (3.0, 12.0, 1.0, 1.0, 0.5), (4.0, 13.0, 1.0, 2.0, 0.0)],
        [
            {"n": 1, "m": 11, "c": "relu", "k": 3, "a": 0.25},
            {"n": 3, "m": 12, "c": "tanh", "k": 5.5, "a": 0.5},
            {"n": 4, "m": 13, "c": "tanh", "k": None, "a": 0.0},
        ],
        [[int, int, str, int, float], [int, int, str, float, float], [int, int, str, type(None), float]],
        id="two-int-two-categorical-one-float-mixed-choice-types",
    ),
]


def _value_types(params):
    return [[type(v) for v in p.values()] for p in params]


def _key_element_types(keys):
    return [[type(v) for v in k] for k in keys]


@pytest.mark.parametrize(
    ("parameters", "u", "expected_keys", "expected_params", "expected_types"), ROWS_CASES
)
def test_space_rows_should_give_raw_float_keys_and_python_typed_params_for_any_mix_of_columns(
    parameters, u, expected_keys, expected_params, expected_types
):
    """Given three unit points, when rows() decodes them, then keys are raw floats and params Python types.

    Oracle: Space.rows and Space.decode docstrings, the task specification and the README table; expected
    values are hand-computed from the equal shares (module docstring). Postcondition: none, rows() is pure.
    """
    space = Space(parameters)

    keys, params = space.rows(np.array(u))

    assert keys == expected_keys
    assert params == expected_params
    assert _value_types(params) == expected_types
    assert _key_element_types(keys) == [[float] * space.dim] * len(u)
