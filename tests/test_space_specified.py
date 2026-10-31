"""Resolved ambiguities and unspecified behaviour of the search space (space.py).

Each test below pins one question that a plain reading of the API leaves open, and its docstring names the
spec sentence that answers it. Expected values are hand-computed literals.

Resolved here:
- Is Int's high included even at the largest unit value sample() can return (1 - 2**-53, since
  rng.random lies in [0, 1)), and on every decode path? README 'integers low..high, inclusive' and the Int
  docstring 'Integers low..high inclusive; every integer gets an equal share of the unit interval'.
  u = 1.0 and the equal shares are already covered by tests/test_space.py::
  test_int_gives_every_value_an_equal_share and are not repeated.
- Index or value? Space.decode docstring 'categories as indices'; Space.rows docstring 'A key is the hashable
  raw row' and 'params are the dictionaries the user's function receives, with Python types'. The index is
  0-based only by inference from the decode/choices correspondence (index 2 is choices[2]); no sentence says
  so. tests/test_space.py::test_categorical_maps_to_choices already pins the choice side of to_dicts.

Derived elsewhere, not stated by a README or docstring sentence (accepted at integration with the oracle
each test cites, so they are not listed as unspecified below):
- move leaves the source params unchanged: abc.py module docstring 'the rest of the decoded parameters are
  copied from the source' (abc-protocol-invariants, test_space_move.py::
  test_space_move_should_leave_the_source_params_unchanged_when_it_changes_a_value).
- decode, rows and to_dicts leave the u array unchanged: rule P-T3-29 side-effect check, and no docstring
  gives licence to change the input (fuzz-strings-valid-generation, test_space_fuzz.py::
  test_space_should_leave_the_mapping_choice_lists_and_u_unchanged_when_built_and_decoded).

Unit values avoid exactly 1/3 and 2/3: 2/3 * 3 rounds to 2.0 in binary64, so such inputs would pin float
rounding rather than the spec.

Unspecified behaviour. No spec source (README, CHANGELOG, docstrings, the original ABC.py, Karaboga &
Basturk 2007) states the following, so no test asserts it. 'Observed in 0.1.0.dev0' notes are not
guaranteed.
- UNSPECIFIED: Space.decode, rows, to_dicts and move with unit values outside [0, 1]. Observed in 0.1.0.dev0:
  Float extrapolates (Float(0, 10) gives -5.0 at u = -0.5 and 15.0 at u = 1.5), Int clips to low and high on
  both sides, Categorical clips only above; below 0 it gives a negative index that to_dicts maps through
  Python negative indexing (u = -0.5 on three choices gives index -2.0 and the second choice), and far below
  0 an IndexError.
- UNSPECIFIED: duplicate Categorical choices. Observed in 0.1.0.dev0: Categorical(['a', 'a']) is accepted;
  the keys (0.0,) and (1.0,) differ while both params are {'c': 'a'}, so the cache treats them as different
  parameter sets.
- UNSPECIFIED: non-integer Int bounds. Observed in 0.1.0.dev0: Int(1.5, 3.2) is accepted; at u = 0 the key
  holds 1.5 (not a whole number) and params hold 1, which is below low.
- UNSPECIFIED: whether the params dicts returned by rows, to_dicts and move are fresh objects, or may be
  shared with dicts handed out earlier or kept by the optimizer, so that editing one changes another.
  Observed in 0.1.0.dev0: rows and to_dicts build new dicts on every call (the same u gives different dict
  objects) and move returns a new dict; the study module notes what happens when the user's function edits
  the dict it receives.
- UNSPECIFIED: decode results for unit values exactly on a share edge (k / n computed in floating point).

Input space: Every row of d unit-cube coordinates in [0, 1] for every parameter kind and bound, which is
unbounded.
Subset: One case per resolved ambiguity; unspecified behaviour is listed, not asserted.
"""

import numpy as np
import pytest

from metahpo import Categorical, Float, Int, Space

TOP_SAMPLE = 1 - 2**-53
INT_HIGH_CASES = [
    pytest.param(0.66, 2, id="just-below-the-2-3-edge-gives-2"),
    pytest.param(0.67, 3, id="just-above-the-2-3-edge-gives-high"),
    pytest.param(TOP_SAMPLE, 3, id="largest-sampled-value-gives-high"),
]


def _one_to_three():
    """Space with one Int(1, 3): three equal shares [0, 1/3), [1/3, 2/3) and [2/3, 1)."""
    return Space({"n": Int(1, 3)})


@pytest.mark.parametrize("u, expected", INT_HIGH_CASES)
def test_space_rows_should_give_int_high_only_when_u_is_in_the_top_share(u, expected):
    """Given Int(1, 3), when rows decodes u, then the top share [2/3, 1) gives high = 3 and 0.66, just below
    it, gives 2 (the negative partner).

    Oracle: README 'integers low..high, inclusive' and the Int docstring (equal shares put the 2|3 edge at
    2/3); Space.decode docstring 'integers as whole numbers' (key holds 3.0); Space.rows docstring 'with
    Python types' (params hold the int 3). Inputs: 0.66 / 0.67 straddle the 2|3 edge, and TOP_SAMPLE is the
    largest value sample() can return, so an exclusive-high decode such as low + floor(u * (high - low))
    gives 2 there (site: Int decode share count). Postcondition: none, rows is pure.
    """
    space = _one_to_three()

    keys, params = space.rows(np.array([[u]]))

    assert keys == [(float(expected),)]
    assert params == [{"n": expected}]
    assert type(params[0]["n"]) is int


@pytest.mark.parametrize("u, expected", INT_HIGH_CASES)
def test_space_to_dicts_should_give_int_high_only_when_u_is_in_the_top_share(u, expected):
    """Given Int(1, 3), when to_dicts decodes u, then the top share [2/3, 1) gives the Python int 3 and 0.66,
    just below it, gives 2 (the negative partner).

    Oracle: README 'integers low..high, inclusive', the Int docstring (equal shares) and the Space.to_dicts
    docstring 'the parameter dictionaries that the user's function receives'. to_dicts is a public method
    of its own, so it is checked directly rather than through rows. Inputs: as for rows; TOP_SAMPLE is where
    an exclusive-high decode gives 2 (site: Int decode share count). Postcondition: none, to_dicts is pure.
    """
    space = _one_to_three()

    params = space.to_dicts(np.array([[u]]))

    assert params == [{"n": expected}]
    assert type(params[0]["n"]) is int


@pytest.mark.parametrize(
    "u_j",
    [
        pytest.param(0.67, id="just-above-the-2-3-edge"),
        pytest.param(TOP_SAMPLE, id="largest-sampled-value"),
    ],
)
def test_space_move_should_include_int_high_when_u_j_is_in_the_top_share(u_j):
    """Given the decoded point n = 1 of Int(1, 3), when move sets coordinate 0 to u_j in the top share, then
    the result is key (3.0,) and params {'n': 3} with a Python int.

    Oracle: as for rows (README 'inclusive', Int docstring) plus the Space.move docstring: the result is the
    (key, params) of the point that differs only in coordinate j, so it equals what rows gives for u_j.
    Input: move uses its own one-value decode (site: Int.decode_one), so the inclusive high is checked on
    that path separately. Postcondition: none, move is pure.
    """
    space = _one_to_three()
    (start_key,), (start_params,) = space.rows(np.array([[0.1]]))

    key, params = space.move(start_key, start_params, 0, u_j)

    assert (key, params) == ((3.0,), {"n": 3})
    assert type(params["n"]) is int


def test_space_rows_should_put_category_indices_in_keys_and_choices_in_params():
    """Given Categorical(['a', 'b', 'c']) and one u per share (0.1, 0.5, 0.9), when decode and rows run,
    then decode and the keys hold the indices 0, 1, 2 and params hold the choices 'a', 'b', 'c'.

    Oracle: Space.decode docstring 'categories as indices'; Space.rows docstring 'A key is the hashable raw
    row' and 'params are the dictionaries the user's function receives'; Categorical docstring 'every choice
    gets an equal share' (shares of width 1/3). Input: string choices differ from their indices in type and
    value, so a decode that returned choices, or params that held indices, fails (site: rows categorical
    mapping). Postcondition: none, both calls are pure.
    """
    space = Space({"c": Categorical(["a", "b", "c"])})
    u = np.array([[0.1], [0.5], [0.9]])

    raw = space.decode(u)
    keys, params = space.rows(u)

    assert raw.tolist() == [[0.0], [1.0], [2.0]]
    assert keys == [(0.0,), (1.0,), (2.0,)]
    assert params == [{"c": "a"}, {"c": "b"}, {"c": "c"}]


def test_space_move_should_put_category_index_in_key_and_choice_in_params_when_moving_a_categorical():
    """Given the decoded point x = 0.25, c = 'a' of a Float plus Categorical(['a', 'b', 'c']) space, when move
    sets the categorical coordinate to 0.9, then the key holds index 2.0, params hold 'c' and x is unchanged.

    Oracle: Space.move docstring (a point differing only in coordinate j) with the Space.rows docstring (keys
    are raw rows, params carry the choices). Input: the categorical is column 1, so writing the choice into
    the key, the index into params, or touching column 0 all fail (site: Space.move key splice and
    Categorical.to_python). Postcondition: none, move is pure.
    """
    space = Space({"x": Float(0.0, 1.0), "c": Categorical(["a", "b", "c"])})
    (start_key,), (start_params,) = space.rows(np.array([[0.25, 0.1]]))

    key, params = space.move(start_key, start_params, 1, 0.9)

    assert (key, params) == ((0.25, 2.0), {"x": 0.25, "c": "c"})
