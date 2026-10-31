"""Specification gaps of the search space that tests/test_space.py leaves open.

Every test adds an equivalence class or boundary that test_space.py does not exercise; the class is named in
the test docstring. Oracles: README.md (Search space table and the unit-cube sentence), CHANGELOG.md and the
docstrings of metahpo.space. Expected values are hand-computed from those sentences.

Unspecified and therefore not asserted: how a log-scaled Int distributes the unit interval over its integers
(equal share is promised for linear Int only), whether log decoding is monotone, and whether Space.move
leaves its params argument untouched.

Input space: Every row of d unit-cube coordinates in [0, 1] for every parameter kind and bound, which is
unbounded.
Subset: One test per specification gap that test_space.py leaves open.
"""

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space

ANY_SEED = 0
MIXED_POINT = np.array([0.3, 0.6, 0.1, 0.8, 0.45, 0.7])


def _mixed_space():
    """Six parameters of all five kinds; the two linear floats sit in non-adjacent columns 0 and 5."""
    return Space(
        {
            "a": Float(-1.0, 1.0),
            "b": Float(1e-4, 1e-2, log=True),
            "c": Int(3, 9),
            "d": Int(1, 1000, log=True),
            "e": Categorical(["x", "y", "z"]),
            "f": Float(0.0, 10.0),
        }
    )


def _with_coordinate(x, j, u_j):
    """Copy of the unit point x with coordinate j set to u_j."""
    moved = x.copy()
    moved[j] = u_j
    return moved


@pytest.mark.parametrize(
    "make_invalid",
    [
        pytest.param(lambda: Float(2.0, 1.0), id="float_low_above_high"),
        pytest.param(lambda: Int(5, 4), id="int_low_above_high"),
        pytest.param(lambda: Int(0, 10, log=True), id="log_int_low_not_positive"),
        pytest.param(lambda: Space({}), id="space_without_parameters"),
    ],
)
def test_parameter_should_raise_value_error_when_arguments_are_invalid(make_invalid):
    """Given one invalid constructor call per case, when it runs, then ValueError is raised.

    Oracle: computed spec 'Invalid parameters raise ValueError/TypeError'. The classes are new beyond
    test_space.py, which covers only Float low == high, the log Float branch, an empty Categorical and a
    non-parameter value: here low > high for Float and Int, the log Int branch and an empty Space.
    """
    with pytest.raises(ValueError):
        make_invalid()


def test_int_should_decode_every_unit_value_to_low_when_low_equals_high():
    """Given Int(5, 5), when u = 0, 0.5 and 1 are decoded, then every value is 5.

    Oracle: README 'integers `low..high`, inclusive' with low == high leaves the single integer 5. Boundary:
    the one-value range, the valid side of the low <= high check, which test_space.py (Int(3, 7)) misses;
    u = 1 is the upper edge where the floor would step past high without the clip.
    """
    u = np.array([0.0, 0.5, 1.0])

    decoded = Int(5, 5).decode(u)

    assert decoded.tolist() == [5, 5, 5]


def test_int_log_should_decode_to_integers_inside_the_inclusive_bounds():
    """Given Int(1, 1000, log=True) and a fine u-grid on [0, 1], when decoded, then every value is an integer
    of 1..1000 and both ends occur.

    Oracle: README 'integers `low..high`, inclusive' and Space.decode 'integers as whole numbers'. Class: the
    log-scaled integer kind, which no test in test_space.py touches. u = 0 maps near 0.5 and u = 1 near
    1000.5, so both ends need the rounding and the clip. The grid of 10001 points is fine enough to hit
    both ends.
    Postcondition: none, decode is a pure function.
    """
    u = np.linspace(0.0, 1.0, 10001)

    decoded = Int(1, 1000, log=True).decode(u)

    assert decoded.dtype.kind == "i"
    assert decoded.min() == 1
    assert decoded.max() == 1000


@pytest.mark.parametrize(
    "n_choices", [pytest.param(1, id="single_choice"), pytest.param(3, id="three_choices")]
)
def test_categorical_should_give_every_choice_an_equal_share(n_choices):
    """Given Categorical with n choices and 3000 grid midpoints of [0, 1], when decoded, then every index
    occurs 3000 // n times.

    Oracle: Categorical docstring 'every choice gets an equal share of the unit interval.' Classes: the
    single-choice boundary and equal share itself, which test_space.py only spot-checks at three points.
    3000 is divisible by 1 and 3, so the expected counts are exact.
    """
    n_points = 3000
    u = (np.arange(n_points) + 0.5) / n_points

    counts = np.bincount(Categorical(list(range(n_choices))).decode(u), minlength=n_choices)

    assert counts.tolist() == [n_points // n_choices] * n_choices


@pytest.mark.parametrize(
    ("mapping", "expected_dim"),
    [
        pytest.param({"x": Float(0.0, 1.0)}, 1, id="one_parameter"),
        pytest.param({"a": Float(0.0, 1.0), "b": Int(1, 3), "c": Categorical("pq")}, 3, id="three_kinds"),
    ],
)
def test_space_dim_should_equal_number_of_parameters(mapping, expected_dim):
    """Given a Space of one or three parameters, when dim is read, then it equals the parameter count.

    Oracle: Space docstring 'dimension d = number of parameters.' No existing test reads dim.
    """
    space = Space(mapping)

    dim = space.dim

    assert dim == expected_dim


@pytest.mark.parametrize("n_points", [pytest.param(1, id="single_row"), pytest.param(7, id="several_rows")])
def test_space_sample_should_return_n_points_of_the_half_open_unit_cube(n_points):
    """Given the six-parameter space and a seeded generator, when n points are sampled, then the shape is
    (n, 6) and every value lies in [0, 1).

    Oracle: Space.sample docstring 'n uniform points of the unit cube, shape (n, d).', README 'Optimizers
    search the unit cube `[0, 1]^d`' and the computed spec '(n, d) in [0,1)'. n = 1 is the single-row
    boundary. The seed only fixes the draw; no drawn value is asserted.
    """
    space = _mixed_space()

    sample = space.sample(np.random.default_rng(ANY_SEED), n_points)

    assert sample.shape == (n_points, space.dim)
    assert sample.min() >= 0.0
    assert sample.max() < 1.0


@pytest.mark.parametrize("column", [pytest.param(j, id=f"column_{j}") for j in range(6)])
def test_space_decode_should_match_the_decode_of_each_parameter(column):
    """Given the six-parameter space, 50 seeded points plus the all-0 and all-1 corners, when the space
    decodes them, then each column equals that parameter's own decode.

    Oracle: CHANGELOG 'decoded from the unit cube in vectorised numpy' and Space.decode 'integers as whole
    numbers, categories as indices.' Class: grouped decoding of mixed kinds, including one kind spread over
    the non-adjacent columns 0 and 5, which no existing test touches. Int and Categorical columns compare
    exactly with the integer arrays of Int.decode / Categorical.decode; float columns allow 1e-12 relative
    error for the log arithmetic.
    """
    space = _mixed_space()
    u = np.vstack([space.sample(np.random.default_rng(ANY_SEED), 50), np.zeros(6), np.ones(6)])

    raw = space.decode(u)

    np.testing.assert_allclose(
        raw[:, column], space.parameters[column].decode(u[:, column]), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize(
    "u_j",
    [
        pytest.param(0.0, id="lower_edge"),
        pytest.param(0.5, id="interior"),
        pytest.param(0.999999, id="just_below_upper_edge"),
        pytest.param(1.0, id="upper_edge"),
    ],
)
@pytest.mark.parametrize("column", [pytest.param(j, id=f"column_{j}") for j in range(6)])
def test_space_move_should_match_the_full_decode_of_the_moved_point(column, u_j):
    """Given a decoded point of the six-parameter space, when coordinate j is moved to u_j, then the key and
    params equal those of rows() on the moved point.

    Oracle: Space.move docstring '(key, params) of a point differing from a decoded point only in coordinate
    j'. Boundaries: u_j = 1 is where Int and Categorical need their clip, 0.999999 just below it, 0 the lower
    edge. Exact equality is required because the cache compares keys exactly.
    """
    space = _mixed_space()
    (key,), (params,) = space.rows(MIXED_POINT[None])
    expected_keys, expected_params = space.rows(_with_coordinate(MIXED_POINT, column, u_j)[None])

    moved_key, moved_params = space.move(key, params, column, u_j)

    assert moved_key == expected_keys[0]
    assert moved_params == expected_params[0]


def test_space_rows_should_return_exact_python_types():
    """Given a space of a log Float, an Int and two Categoricals, when rows() decodes one point, then the
    float is exactly float, the integer exactly int and each category is the choice object itself.

    Oracle: Space.rows docstring 'params are the dictionaries the user's function receives, with Python
    types.' Class: exact types, stricter than the isinstance check of test_space.py, which a numpy scalar
    such as np.float64 would pass. Hand calculation: units = 16 + floor(0.5 * 241) = 136, act index
    floor(0.1 * 2) = 0, k index floor(0.4 * 3) = 1.
    """
    relu = "relu"
    space = Space(
        {
            "lr": Float(1e-4, 1e-2, log=True),
            "units": Int(16, 256),
            "act": Categorical([relu, "tanh"]),
            "k": Categorical([3, 5, 7]),
        }
    )

    _, (params,) = space.rows(np.array([[0.3, 0.5, 0.1, 0.4]]))

    assert type(params["lr"]) is float
    assert type(params["units"]) is int
    assert params["units"] == 136
    assert params["act"] is relu
    assert type(params["k"]) is int
    assert params["k"] == 5


@pytest.mark.parametrize(
    ("second_point", "same_cells"),
    [
        pytest.param([0.3, 0.5001, 0.2], True, id="same_int_and_category_cells"),
        pytest.param([0.3, 0.505, 0.2], False, id="next_int_cell"),
    ],
)
def test_space_rows_should_give_equal_keys_exactly_when_points_share_every_cell(second_point, same_cells):
    """Given two unit points with the same float coordinate, when rows() decodes them, then the keys are
    equal hashable tuples exactly when both points fall into the same Int and Categorical cells.

    Oracle: Space.rows docstring 'A key is the hashable raw row: equal keys mean the user's function sees
    equal parameters.' Hand calculation for Int(16, 256): floor(0.5 * 241) = floor(0.5001 * 241) = 120 but
    floor(0.505 * 241) = 121; the category index is floor(0.1 * 2) = floor(0.2 * 2) = 0. Class: key identity
    of two distinct unit points, never asserted in test_space.py; both sides of the iff are cases.
    """
    space = Space({"lr": Float(0.0, 1.0), "units": Int(16, 256), "act": Categorical(["relu", "tanh"])})

    keys, params = space.rows(np.array([[0.3, 0.5, 0.1], second_point]))

    assert type(keys[0]) is tuple
    assert (len({keys[0], keys[1]}) == 1) is same_cells
    assert (params[0] == params[1]) is same_cells
