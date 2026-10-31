"""Regression tests: pinned seeded outcomes, runtime budget and the recorded fault history.

Characterisation and regression checks of the whole metahpo pipeline.

Status of this file
-------------------
This is a characterisation (regression) suite. It is kept separate from the spec tests in test_space.py and
test_abc.py. It records what the code does now, not what the code is meant to do: a pinned value is not
proof that the value is right, because a recorded output knows nothing about intended behaviour (the
oracle problem). The spec tests decide correctness; this file only says that behaviour did not change.

Kinds of check
--------------
1. Version-independent pinned literals. Space.decode, Space.rows and Space.to_dicts are run on a
   hand-written unit-cube matrix with no random number generator, and the results are compared with literal
   values derived by hand. Integer and category columns are compared exactly; every column that goes through
   log or exp is compared with pytest.approx, because the last bit of exp may differ between platforms.
2. Same-run characterisations. Two code paths are run inside one process and compared with each other: an
   ask/tell loop against optimize(), and optimize() against itself with the same seed.

Why no seeded golden master
---------------------------
Seed-derived literals, such as the first candidate parameters or the final best_score of a seeded run, are
deliberately not written into this file. The numpy Generator stream is not promised to be stable across
numpy versions, and the CI matrix (Python 3.10 to 3.14 on Ubuntu, 3.14 on Windows and macOS, numpy>=1.23
unpinned) would turn such literals into environment-dependent expectations (P-T1-33). Expected values must
also come from a specification, not from observed output (P-T8-04). Seeded runs are therefore only compared
with other runs in the same process.

What this file cannot catch
---------------------------
A same-run characterisation compares two paths that share the same code. A deterministic change that both
paths share, for example a different fitness formula or a different neighbour move, changes both sides in the
same way and passes here. The spec tests (test_abc.py and the operator tests) cover that case.

When a test here fails
----------------------
A failure means behaviour changed. Decide whether the change was intended. Update a pinned literal only
together with a deliberate change that is documented in CHANGELOG.md, never just to make the test pass.

Order
-----
Tests are defined from the most local unit to the most integrated one, so that a run stopped at the first
failure (pytest -x) points at the lowest broken layer: Space.decode literals, Space.rows literals,
Space.to_dicts, the ask/tell loop against optimize(), then same-seed optimize(). No test depends on another
test; every test builds its own Space, matrix, objective and optimizer.

Input space: d unit-cube coordinates per row for the Space checks (float64 each), and for the optimizer
checks every pair of direction and budget times the unbounded ask/tell sequences of a seeded ABC run.
Subset: one hand-written 5 x 6 matrix with the lower corner, the upper corner and three interior rows chosen
in the middle of integer and category bins; directions 'min' and 'max'; budgets 3 (inside the initial batch)
and 200 (many ABC cycles, each ending with the scout step); one fixed seed.

Unspecified behaviour, pinned here as characterisation only
-----------------------------------------------------------
- The log-scaled Int mapping (rint of exp between log(low - 0.5) and log(high + 0.5), then clip) is stated
  only in the code; the README says only 'integers low..high, inclusive'. Its literals are characterisation.
- Interior values of Float (linear and geometric interpolation) are not stated beyond 'real numbers in
  [low, high]'; only endpoints and the midpoint are spec-backed.
- The README does not promise that an ask/tell loop and optimize() produce the same sequence.
- That a Categorical returns the identical choice object (identity, not only equality) is not stated.
- The README sentence 'The same seed reproduces the same search' does not say across which environments;
  only same-process reproducibility is checked.
- Whether different seeds give different searches is unspecified and is not checked.

Regression tests for the recorded defects of the original hyp_optimizer and for attacker mutants.

Order (fault history first, then fault-based, then attacker): the tests appear in the priority of the
recorded defects of the original implementation, so a run stopped at the first failure reports the most
likely regression first, and the first eight tests already touch operators, space (through move), abc,
facade and study.

1. ABC.py:98 wraps the Karaboga move in round(), so a continuous coordinate snaps to an integer.
2. ABC.py:77-78 store 1 - accuracy at init, ABC.py:117 and :144 store the raw test accuracy in the
   employed and onlooker phases, so a stored score means different things in different phases.
3. Config.py:22 computes FOOD_NUMBER before the -n option is parsed at Config.py:77, so the colony size
   given by the user is ignored.
4. ABC.py:87-88 set globalOpt from food 0 without its model, and the strict < at ABC.py:64 never records
   food 0 again, so the best is lost when the first initial source is the best one.
5. ABC.py:45 and ABC.py:86 both call increase_eval for one initial evaluation, so the budget is counted twice.
6. Fault-based (simulated earlier mistakes): partner k equal to i (ABC.py:92-94), scout at trial > limit
   instead of >= (ABC.py:152), acceptance on >= instead of > (ABC.py:113 and :140).
7. Attacker mutants that survived the earlier suite: facade budget off-by-one, Study min/max swap or
   keeping failed trials, n_calls counting cached trials, reversed categorical order in one decode path,
   and the partial initial tell.

Every test asserts the specified (correct) behaviour, so it passes on the current library and fails on the
defective or mutated version. Oracles: README, docstrings, the task specification of the library, the original
ABC.py and Karaboga & Basturk (2007). Expected values are hand-computed.

Unspecified, not asserted:
- best_params and best_score of an optimizer that has only seen non-finite scores;
- which trial Study.best_trial returns when two finite scores tie;
- the state of the candidates that optimize() asks for but truncates when max_evals ends inside the
  initial batch;
- the exact float edges of a categorical share (the float nearest to 1/3 is below the true 1/3), so the
  categorical cases use values clearly inside a share plus the closed ends 0.0 and 1.0;
- the ids of the candidates handed out after the initial batch (no spec source says they must be fresh).

Covered by sibling work items, not repeated here:
- Study.best_trial raising ValueError for an empty, all-NaN or all-infinite study (error-paths test_study.py);
- the scout being a uniform sample that shares no coordinate with the initial points, and the cycle
  counter after the scout (abc phase-schedule and boundaries items);
- study.best_score, opt.best_score and best_params agreeing for direction 'max' (operators-formulas
  test_facade_direction.py);
- telling an already-told candidate raising ValueError (error-paths).

SEEDS: three fixed seeds, so a property that should hold for every random stream is not checked on one
lucky stream only. ANY_SEED: fixed for determinism where the result does not depend on random draws.
"""

import math

import numpy as np
import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space, Study, Trial, optimize
from metahpo.operators import neighbour_value

CHOICES = ("relu", "tanh", "gelu")
"""Non-numeric categories, so that a returned index or a copied string is told apart from the choice."""


NAMES = ("x", "n", "lr", "act", "units", "drop")
"""Parameter order of the pinned Space; kinds are interleaved so the per-kind column scatter is exercised."""


SEED = 11
"""Any fixed seed; 11 has no special meaning, it only makes the seeded runs repeatable."""


OTHER_SEED = 12
"""Any seed other than SEED, for the unrelated search that runs between two compared runs."""


N_EVALS_INSIDE_INIT = 3
"""Below colony_size // 2 = 5 initial sources, so the initial batch is truncated and told partially."""


N_EVALS_MANY = 200
"""About 19 ABC cycles of 2 * 5 evaluations, each ending with the scout step; repeats are cached.

Whether a scout candidate is actually issued depends on the seed stream (here it is for "max" only), so it is
not asserted.
"""


PINNED_DECODE = {
    "x": [-2.0, 6.0, 0.0, 3.0, 2.0],
    "n": [3.0, 7.0, 4.0, 6.0, 3.0],
    "lr": [1e-3, 10.0, 1e-2, 1.0, 0.1],
    "act": [0.0, 2.0, 1.0, 2.0, 0.0],
    "units": [1.0, 1000.0, 5.0, 48.0, 15.0],
    "drop": [0.0, 0.5, 0.25, 0.05, 0.45],
}
"""Hand-derived decode of the fixed matrix, per column.

x = -2 + 8u. n = 3 + floor(5u), clipped to 7 at u = 1. lr = 1e-3 * 10**(4u). act = floor(3u), clipped to 2
at u = 1. units = rint(0.5 * 2001**u), clipped to 1..1000: 0.5 -> 1, 1000.5 -> 1000, 4.89 -> 5,
47.83 -> 48, 15.29 -> 15. drop = 0.5u.
"""


PINNED_PARAMS = (
    {"x": -2.0, "n": 3, "lr": 1e-3, "act": "relu", "units": 1, "drop": 0.0},
    {"x": 6.0, "n": 7, "lr": 10.0, "act": "gelu", "units": 1000, "drop": 0.5},
    {"x": 0.0, "n": 4, "lr": 1e-2, "act": "tanh", "units": 5, "drop": 0.25},
    {"x": 3.0, "n": 6, "lr": 1.0, "act": "gelu", "units": 48, "drop": 0.05},
    {"x": 2.0, "n": 3, "lr": 0.1, "act": "relu", "units": 15, "drop": 0.45},
)
"""PINNED_DECODE rows with Int values as int and category indices mapped through CHOICES."""


INT_NAMES = ("n", "units")


FLOAT_NAMES = ("x", "lr", "drop")


EXACT_NAMES = ("n", "act", "units")


APPROX_NAMES = ("x", "lr", "drop")


ROW_IDS = ("lower-corner", "upper-corner", "bin-mid-a", "bin-mid-b", "bin-mid-c")
"""Readable ids of the five rows of the fixed unit matrix."""


REL = 1e-12
"""Relative tolerance for columns that go through log or exp; far below any decode change of interest."""


ACT_PENALTY = {"relu": 0.0, "tanh": 0.3, "gelu": 0.7}


BEST_EXACT = ("units", "act")


BEST_APPROX = ("lr", "drop")


def _pinned_space():
    """Mixed Space with every decode kind: linear Float, linear Int, log Float, Categorical, log Int."""
    return Space(
        {
            "x": Float(-2.0, 6.0),
            "n": Int(3, 7),
            "lr": Float(1e-3, 10.0, log=True),
            "act": Categorical(CHOICES),
            "units": Int(1, 1000, log=True),
            "drop": Float(0.0, 0.5),
        }
    )


def _fixed_unit_matrix():
    """Rows: lower corner, upper corner, then three interior rows in the middle of Int and Categorical bins.

    For Int(3, 7), 5u = 1.5, 3.5 and 0.5 are bin middles, so floor, ceil and rint give different integers.
    For the Categorical, 3u = 1.5, 2.7 and 0.3. For the log Int the fractional parts 0.89, 0.83 and 0.29 are
    at least 0.2 away from .5, so a one-ulp difference of exp cannot flip the rounding.
    """
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.25, 0.3, 0.25, 0.5, 0.3, 0.5],
            [0.625, 0.7, 0.75, 0.9, 0.6, 0.1],
            [0.5, 0.1, 0.5, 0.1, 0.45, 0.9],
        ]
    )


def _search_space():
    """Four-parameter Space spreading the decode kinds for the optimizer characterisations."""
    return Space(
        {
            "lr": Float(1e-4, 1e-1, log=True),
            "units": Int(2, 64),
            "act": Categorical(["relu", "tanh", "gelu"]),
            "drop": Float(0.0, 0.5),
        }
    )


def _pure_objective(p):
    """Deterministic, side-effect free and NaN-free; reads every parameter of _search_space."""
    return (
        (math.log10(p["lr"]) + 2.5) ** 2
        + ((p["units"] - 20) / 10) ** 2
        + ACT_PENALTY[p["act"]]
        + (p["drop"] - 0.2) ** 2
    )


def _make_objective_with_nan_region():
    """Fresh pure objective returning NaN for gelu with drop > 0.4, so failed trials can appear in a study."""

    def objective(p):
        if p["act"] == "gelu" and p["drop"] > 0.4:
            return math.nan
        return _pure_objective(p)

    return objective


def _drive_ask_tell(space, direction, seed, n_evals, objective):
    """Ask/tell loop evaluating exactly n_evals candidates; every batch is sliced to the remaining budget."""
    opt = ABCOptimizer(space, direction=direction, seed=seed)
    scores, params = [], []
    while len(scores) < n_evals:
        cands = opt.ask()[: n_evals - len(scores)]
        batch = [objective(c.params) for c in cands]
        scores += batch
        params += [c.params for c in cands]
        opt.tell(cands, batch)
    return opt, scores, params


def _run_unrelated_search(direction):
    """Runs a different seeded search on its own Space, as an unrelated run between the two compared ones."""
    optimize(_pure_objective, _search_space(), direction=direction, max_evals=N_EVALS_MANY, seed=OTHER_SEED)


def _seeded_study(space, direction):
    """optimize() with a fresh NaN-region objective, N_EVALS_MANY evaluations and SEED on the given Space."""
    objective = _make_objective_with_nan_region()
    return optimize(objective, space, direction=direction, max_evals=N_EVALS_MANY, seed=SEED)


def _study_record(study):
    """Every observable field of a Study, in a form that compares with == except for NaN scores."""
    return {
        "params": [t.params for t in study.trials],
        "cached": [t.cached for t in study.trials],
        "x": [t.x.tolist() for t in study.trials],
        "best_params": study.best_params,
        "best_score": study.best_score,
        "n_calls": study.n_calls,
    }


@pytest.mark.parametrize("name", EXACT_NAMES, ids=[f"exact-{n}" for n in EXACT_NAMES])
def test_space_decode_should_match_pinned_whole_values_when_unit_matrix_is_fixed(name):
    """Given the pinned Space and the fixed matrix, when it is decoded, then the Int and Categorical column
    equals the hand-derived whole floats and indices exactly.

    Oracle: README Search space table, Int docstring (equal share, so floor of 5u for Int(3, 7)), Categorical
    docstring (equal share, so thirds), Space.decode docstring (integers as whole numbers, categories as
    indices); log Int is characterisation. Guards site: Int floor and clip in _decode_int, Categorical clamp
    to n - 1 in _decode_categorical, rint in the log-Int branch; observed failure is a shifted integer or an
    index 3. Postcondition: none, decode is pure.
    """
    space = _pinned_space()
    u = _fixed_unit_matrix()

    raw = space.decode(u)

    assert raw.shape == (5, 6)
    assert raw[:, NAMES.index(name)].tolist() == PINNED_DECODE[name]


@pytest.mark.parametrize("name", APPROX_NAMES, ids=[f"approx-{n}" for n in APPROX_NAMES])
def test_space_decode_should_match_pinned_float_values_when_unit_matrix_is_fixed(name):
    """Given the pinned Space and the fixed matrix, when it is decoded, then the Float column equals the
    hand-derived values within rel=1e-12.

    Oracle: README (real numbers in [low, high]) and the endpoints and midpoint of the spec test
    test_float_linear_and_log_bounds; interior values are characterisation. approx is used because log
    decoding does not hit its endpoints exactly (1e-3 decodes to 0.0010000000000000002 here). Guards site:
    linear and geometric interpolation in _decode_float and the column scatter in Space.decode.
    Postcondition: none, decode is pure.
    """
    space = _pinned_space()
    u = _fixed_unit_matrix()

    raw = space.decode(u)

    assert raw.shape == (5, 6)
    assert raw[:, NAMES.index(name)].tolist() == pytest.approx(PINNED_DECODE[name], rel=REL)


@pytest.mark.parametrize("row", range(5), ids=ROW_IDS)
def test_space_rows_should_return_pinned_keys_of_whole_floats_when_unit_matrix_is_fixed(row):
    """Given the pinned Space and the fixed matrix, when rows() is called, then the key of the row is a tuple
    of six floats equal to the pinned raw row.

    Oracle: Space.rows docstring (a key is the hashable raw row) and Space.decode docstring (integers as whole
    numbers, categories as indices). type(...) is float shows that the key holds the raw row, not converted
    Python values. Postcondition: none, rows is pure.
    """
    space = _pinned_space()
    u = _fixed_unit_matrix()
    expected = tuple(PINNED_DECODE[name][row] for name in NAMES)

    keys, params = space.rows(u)

    assert len(keys) == len(params) == 5
    assert [type(v) for v in keys[row]] == [float] * 6
    assert keys[row] == pytest.approx(expected, rel=REL)
    assert [keys[row][NAMES.index(n)] for n in EXACT_NAMES] == [expected[NAMES.index(n)] for n in EXACT_NAMES]


@pytest.mark.parametrize("row", range(5), ids=ROW_IDS)
def test_space_rows_should_return_pinned_typed_params_when_unit_matrix_is_fixed(row):
    """Given the pinned Space and the fixed matrix, when rows() is called, then the params dict of the row has
    the Space name order, int values for Int, float values for Float and the stored choice object itself.

    Oracle: Space.rows docstring (params are the dictionaries the user's function receives, with Python
    types), README Quick start, existing test_to_dicts_returns_python_types. type(...) is int is the deciding
    check: 3 == 3.0, so equality alone would let an Int left as float pass. Guards site: the int conversion
    and the choices lookup in Space.rows. Postcondition: none, rows is pure.
    """
    space = _pinned_space()
    u = _fixed_unit_matrix()
    expected = PINNED_PARAMS[row]

    keys, params = space.rows(u)
    p = params[row]

    assert list(p) == list(NAMES)
    assert [type(p[n]) for n in INT_NAMES] == [int, int]
    assert [type(p[n]) for n in FLOAT_NAMES] == [float, float, float]
    assert p["act"] is CHOICES[CHOICES.index(expected["act"])]
    assert {n: p[n] for n in EXACT_NAMES} == {n: expected[n] for n in EXACT_NAMES}
    assert [p[n] for n in APPROX_NAMES] == pytest.approx([expected[n] for n in APPROX_NAMES], rel=REL)


def test_space_to_dicts_should_equal_rows_params_when_unit_matrix_is_fixed():
    """Given the pinned Space and the fixed matrix, when to_dicts() is called, then it returns exactly the
    params half of rows() on the same matrix.

    Oracle: Space.to_dicts docstring (the parameter dictionaries the user's function receives), which rows()
    also returns as its second item. Postcondition: none, to_dicts is pure.
    """
    space = _pinned_space()
    u = _fixed_unit_matrix()
    keys, params = space.rows(u)

    dicts = space.to_dicts(u)

    assert dicts == params


@pytest.mark.parametrize("n_evals", [N_EVALS_INSIDE_INIT, N_EVALS_MANY], ids=["inside-init", "many-cycles"])
@pytest.mark.parametrize("direction", ["min", "max"])
def test_abc_ask_tell_loop_should_reproduce_optimize_trials_when_batches_are_sliced(direction, n_evals):
    """Given one Space, a pure objective and seed 11, when optimize() and an ask/tell loop that slices every
    batch to the remaining budget each evaluate n_evals candidates, then scores, trial params and the best
    result are the same.

    This is a same-run characterisation of current behaviour; the README does not promise it. Oracle: README
    'Ask and tell' and the optimize docstring (max_evals counts evaluations; with cache=True a repeat reuses
    the earlier score, so a pure objective sees the same score). Budget 3 cuts the initial batch of 5 and
    tells it partially; budget 200 runs about 19 cycles of employed, onlooker and scout steps with cached
    repeats. Float entries of best_params use approx, because optimize takes them from the candidates
    (Space.move) and the optimizer re-decodes best_x with to_dicts; the two decoders are not promised to
    agree to the last bit.
    """
    space = _search_space()
    opt, loop_scores, loop_params = _drive_ask_tell(space, direction, SEED, n_evals, _pure_objective)

    study = optimize(_pure_objective, space, direction=direction, max_evals=n_evals, seed=SEED)

    assert len(study.trials) == n_evals
    assert study.scores.tolist() == loop_scores
    assert [t.params for t in study.trials] == loop_params
    assert study.best_score == opt.best_score
    assert [study.best_params[k] for k in BEST_EXACT] == [opt.best_params[k] for k in BEST_EXACT]
    assert [study.best_params[k] for k in BEST_APPROX] == pytest.approx(
        [opt.best_params[k] for k in BEST_APPROX], rel=REL
    )


@pytest.mark.parametrize("direction", ["min", "max"])
def test_optimize_should_reproduce_the_whole_study_when_seed_is_repeated(direction):
    """Given two separately built but equal Spaces and fresh objectives with a NaN region, when optimize()
    runs twice with seed 11 and an unrelated search with another seed runs in between, then every field of
    the two studies is equal.

    Oracle: README Behaviour 'The same seed reproduces the same search'; the expected value is the first run
    in the same process. Distinct Space instances show that reproducibility does not rest on object identity,
    and the unrelated search shows that no state leaks from one run into the next. Scores use
    equal_nan=True because NaN trials may occur; whether they do depends on the seed stream and is not
    asserted. Different seeds are not compared with each other: unspecified.
    """
    first_space, second_space = _search_space(), _search_space()
    first = _seeded_study(first_space, direction)
    _run_unrelated_search(direction)

    second = _seeded_study(second_space, direction)

    assert first_space is not second_space
    assert np.array_equal(second.scores, first.scores, equal_nan=True)
    assert _study_record(second) == _study_record(first)


ANY_SEED = 0


SEEDS = (0, 1, 2)


NEIGHBOUR_PHASES = ("employed", "onlooker")


LETTER_CHOICES = ("a", "b", "c")


NO_X = np.zeros(1)


def _neighbour_x_values(seed, n_asks):
    """Runs n_asks single asks after the initial batch of a 1-D Float(-5, 5) search on x**2 and returns
    params['x'] of every employed or onlooker candidate (scouts are uniform samples and are left out)."""
    space = Space({"x": Float(-5.0, 5.0)})
    opt = ABCOptimizer(space, colony_size=10, limit=15, seed=seed)
    initial = opt.ask()
    opt.tell(initial, [c.params["x"] ** 2 for c in initial])
    values = []
    for _ in range(n_asks):
        (cand,) = opt.ask()
        if opt.phase in NEIGHBOUR_PHASES:
            values.append(cand.params["x"])
        opt.tell([cand], [cand.params["x"] ** 2])
    return values


def _count_strictly_inside(values, low, high):
    """Number of values v with low < v < high."""
    return sum(low < v < high for v in values)


def _count_not_whole(values):
    """Number of values that are not whole numbers in raw space."""
    return sum(not float(v).is_integer() for v in values)


def _initial_and_neighbour_coordinates(seed, n_asks, initial_score, neighbour_score):
    """Tells the initial batch of a colony of 4 on Float(0, 1) with initial_score, then n_asks neighbours
    with neighbour_score; returns the initial x[0] values and the x[0] of every neighbour candidate."""
    space = Space({"a": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=10**6, seed=seed)
    initial = opt.ask()
    opt.tell(initial, [initial_score] * len(initial))
    neighbours = []
    for _ in range(n_asks):
        (cand,) = opt.ask()
        neighbours.append(float(cand.x[0]))
        opt.tell([cand], [neighbour_score])
    return {float(c.x[0]) for c in initial}, neighbours


def _first_scout_ask_number(opt, constant_score, max_asks):
    """Tells the initial batch and every later candidate constant_score; returns the 1-based number of the
    first single ask after which opt.phase is 'scout', or None if none comes within max_asks."""
    initial = opt.ask()
    opt.tell(initial, [constant_score] * len(initial))
    for number in range(1, max_asks + 1):
        (cand,) = opt.ask()
        if opt.phase == "scout":
            return number
        opt.tell([cand], [constant_score])
    return None


def _tell_initial_then_every_employed(opt, initial_score, employed_score):
    """Tells the whole initial batch initial_score and each of the food_number employed neighbours
    employed_score, leaving opt at the start of the onlooker phase."""
    initial = opt.ask()
    opt.tell(initial, [initial_score] * len(initial))
    for _ in range(opt.food_number):
        (employed,) = opt.ask()
        opt.tell([employed], [employed_score])


def _trial(number, score, cached=False):
    """A Trial in which only number, score and cached matter."""
    return Trial(number, {"i": number}, score, cached, NO_X)


@pytest.mark.parametrize(
    ("x_ij", "x_kj", "phi", "expected"),
    [
        pytest.param(0.25, 0.125, 0.5, 0.3125, id="round-would-give-0"),
        pytest.param(0.75, 0.5, 0.5, 0.875, id="round-would-give-1"),
        pytest.param(0.625, 0.375, -0.5, 0.5, id="negative-phi-half-round-would-give-0"),
    ],
)
def test_neighbour_value_should_return_the_unrounded_move_when_it_stays_inside_the_cube(
    x_ij, x_kj, phi, expected
):
    """Given an interior move, when neighbour_value is called, then it returns x_ij + phi * (x_ij - x_kj)
    unrounded.

    Oracle: Karaboga & Basturk (2007) v_ij = x_ij + phi * (x_ij - x_kj) and the neighbour_value docstring
    (only values outside [0, 1] are clipped); hand calculation 0.25 + 0.5 * 0.125 = 0.3125,
    0.75 + 0.5 * 0.25 = 0.875, 0.625 - 0.5 * 0.25 = 0.5. Inputs are dyadic, so equality is exact, and each
    expected value is rounded by round() to an integer (0, 1, 0), which is why they are decisive.
    Guards the recorded defect site ABC.py:98 (round around the move); observed failure would be 0.0 or 1.0.
    The existing test_neighbour_changes_one_coordinate_and_stays_in_the_cube also kills that mutant; this test
    carries the defect citation. Postcondition: none, pure function.
    """
    result = neighbour_value(x_ij, x_kj, phi)

    assert result == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_abc_should_propose_interior_float_values_when_employed_and_onlooker_bees_move_a_float(seed):
    """Given a 1-D Float(-5, 5) search, when 300 neighbours are asked, then most of them lie strictly inside.

    Oracle: README (Float takes real numbers in [low, high]) with the continuous Karaboga move. With one
    dimension every neighbour moves the only coordinate, through Space.move and Float.decode_one. The bounds
    "at least half interior" and "more than 3 distinct values" are not spec numbers; they only separate a
    continuous move from one snapped to the cube corners, where every value is exactly -5 or 5 (0 interior,
    at most 2 distinct). Clipping to the bounds is legitimate, so 100 % interior is not required.
    The original rounds the raw value (Config.py:26-27 bounds), so the observed failure can also be values
    snapped to whole numbers in raw space; a real-valued Float must give at least half non-whole values.
    Guards the recorded defect site ABC.py:98 (round), seen end to end.
    """
    n_asks = 300

    values = _neighbour_x_values(seed, n_asks)

    assert _count_strictly_inside(values, -5.0, 5.0) >= len(values) // 2
    assert len(set(values)) > 3
    assert _count_not_whole(values) >= len(values) // 2


def test_optimize_should_store_the_returned_score_in_every_phase_when_direction_is_max():
    """Given direction 'max' and an objective returning x, when optimize runs 200 evaluations, then every
    trial's score is exactly the value the objective returned, in all four phases.

    Oracle: Optimizer docstring (scores are given in the user's direction; only the internal value is
    minimised) and the optimize docstring. 'max' is the class where the internal value is negated, so a
    leak of the internal value on any one path shows up as a sign flip. colony_size=4 and limit=3 make
    scouts fire within 200 evaluations; the recorded phases confirm that every path was exercised, so the
    test cannot pass because one phase never ran. cache=False so every trial comes from a call.
    Guards the recorded defect site ABC.py:77-78 versus :117/:144 (stored score changes meaning by phase).
    """
    space = Space({"x": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, direction="max", colony_size=4, limit=3, seed=ANY_SEED)
    phases = []
    returned = []

    def spy_objective(params):
        phases.append(opt.phase)
        returned.append(params["x"])
        return params["x"]

    study = optimize(spy_objective, space, direction="max", max_evals=200, optimizer=opt, cache=False)

    assert set(phases) == {"init", "employed", "onlooker", "scout"}
    assert [t.score for t in study.trials] == returned


@pytest.mark.parametrize("seed", SEEDS)
def test_abc_should_evaluate_the_same_points_when_maximising_f_or_minimising_minus_f(
    seed,
):
    """Given f and -f on the same 2-D space and seed, when one run maximises f and the other minimises -f,
    then both evaluate the same parameters, the scores are exact negatives and the optimizers' bests too.

    Oracle: Optimizer docstring (scores in the user's direction, minimised internally) and README (the same
    seed reproduces the same search): both runs minimise the identical internal value -f, so they draw the
    same uniforms and make the same decisions; IEEE negation is exact. Metamorphic relation, no golden values.
    colony_size=6 and limit=4 make scouts fire within 300 evaluations (asserted through the recorded phases),
    so init, employed, onlooker and scout tell paths are all compared.
    Guards the recorded defect site ABC.py:78 (1 - test accuracy stored at init) versus :117/:144 (raw test
    accuracy stored in the employed and onlooker phases): a tell path that stores the raw user score instead
    of the internal one makes the trajectories diverge.
    """
    space = Space({"a": Float(-5.0, 5.0), "b": Float(-5.0, 5.0)})
    opt_max = ABCOptimizer(space, direction="max", colony_size=6, limit=4, seed=seed)
    opt_min = ABCOptimizer(space, direction="min", colony_size=6, limit=4, seed=seed)
    phases = []

    def f(params):
        phases.append(opt_max.phase)
        return -((params["a"] - 1.0) ** 2) - (params["b"] + 2.0) ** 2

    def minus_f(params):
        return (params["a"] - 1.0) ** 2 + (params["b"] + 2.0) ** 2

    study_max = optimize(f, space, direction="max", max_evals=300, optimizer=opt_max, cache=False)
    study_min = optimize(minus_f, space, direction="min", max_evals=300, optimizer=opt_min, cache=False)

    assert "scout" in phases
    assert [t.params for t in study_max.trials] == [t.params for t in study_min.trials]
    np.testing.assert_array_equal(study_max.scores, -study_min.scores)
    assert opt_max.best_score == -opt_min.best_score


@pytest.mark.parametrize(
    ("colony_size", "expected"),
    [
        pytest.param(4, 2, id="smallest-valid-colony"),
        pytest.param(5, 2, id="odd-kills-ceil"),
        pytest.param(10, 5, id="default-colony"),
        pytest.param(11, 5, id="odd-kills-round-half-even"),
        pytest.param(12, 6, id="kills-constant-five"),
    ],
)
def test_abc_should_ask_half_the_colony_as_initial_batch_when_colony_size_is_given(colony_size, expected):
    """Given a colony_size, when the optimizer is asked first, then it hands out colony_size // 2 sources.

    Oracle: task specification (colony_size // 2 sources asked as one batch), Karaboga SN = colony / 2 and
    Config.py:22 int(NP / 2), a floor. Odd sizes discriminate floor from ceil (5 -> 3) and from round
    (11 -> 6; round(2.5) is 2 in Python, so 5 alone would not); 12 kills a constant 5 and 4 is the smallest
    valid size. The existing test_phase_order_and_evaluation_counts checks only 10 -> 5.
    Guards the recorded defect site Config.py:22 versus :77 (colony size given by the user ignored).
    """
    space = Space({"a": Float(-5.0, 5.0), "b": Float(-5.0, 5.0)})
    opt = ABCOptimizer(space, colony_size=colony_size, seed=ANY_SEED)

    initial = opt.ask()

    assert len(initial) == expected


def test_abc_should_record_the_first_initial_candidate_as_best_when_it_has_the_lowest_score():
    """Given a full initial batch whose first candidate scores lowest, when it is told, then that candidate
    is the optimizer's best.

    Oracle: README Ask and tell (opt.best_params and opt.best_score are the best seen) and the Optimizer
    docstring (direction 'min'); the expected best is the argmin of the told scores. Position 0 is the
    discriminating input; 0.0 is also the f >= 0 fitness boundary. Guards the recorded defect site
    ABC.py:87-88 with :64 (food 0 never recorded as best); observed failure would be another candidate's
    params or a best_score other than 0.0.
    """
    space = Space({"a": Float(-5.0, 5.0), "b": Float(-5.0, 5.0)})
    opt = ABCOptimizer(space, colony_size=10, seed=ANY_SEED)
    initial = opt.ask()

    opt.tell(initial, [0.0, 1.0, 2.0, 3.0, 4.0])

    assert opt.best_score == 0.0
    assert opt.best_params == initial[0].params


def test_abc_should_record_the_first_told_candidate_as_best_when_a_partial_initial_tell_maximises():
    """Given direction 'max' and only the first initial candidate told, when it is told 9.0, then it is the
    optimizer's best right away, in the user's direction.

    Oracle: README Ask and tell (best seen), Optimizer docstring (best_score in the user's direction) and the
    task specification (the initial batch may be told partially). One told candidate is the smallest case of
    "the first candidate is the best"; 'max' checks the negation on the same path. Guards the recorded defect
    site ABC.py:87-88 (best of food 0 not recorded).
    """
    space = Space({"a": Float(-5.0, 5.0), "b": Float(-5.0, 5.0)})
    opt = ABCOptimizer(space, direction="max", colony_size=10, seed=ANY_SEED)
    initial = opt.ask()

    opt.tell(initial[:1], [9.0])

    assert opt.best_score == 9.0
    assert opt.best_params == initial[0].params


def test_study_best_trial_should_be_the_first_trial_when_it_has_the_lowest_score():
    """Given a 'min' Study whose trial 0 has the lowest finite score, when best_trial is read, then it is
    trial 0.

    Oracle: Study semantics (best_trial is the minimum finite trial for 'min'); the expected trial is the
    argmin of [0.0, 1.0]. Position 0 is the discriminating input, the Study counterpart of the recorded
    defect site ABC.py:87-88 (best lost when it is the first source). Postcondition: the study is unchanged.
    """
    study = Study("min", [_trial(0, 0.0), _trial(1, 1.0)])

    best = study.best_trial

    assert best.number == 0


def test_optimize_should_call_the_objective_once_per_trial_when_the_budget_equals_the_initial_batch():
    """Given max_evals equal to the initial batch (5 for the default colony of 10), when optimize runs with
    the default cache, then the objective is called 5 times, 5 trials numbered 0..4 exist and n_calls is 5.

    Oracle: optimize docstring (max_evals counts the optimizer's evaluations), Study.n_calls docstring
    (calls of the user's function) and the task specification (exactly max_evals trials). The boundary
    max_evals == batch size is where counting each initial evaluation twice would stop after 2 or 3 calls.
    The objective is continuous, so the cache never hits. Guards the recorded defect site ABC.py:45 with :86
    (increase_eval twice per initial evaluation). The existing test_budget_is_exact_even_inside_the_initial_
    population checks only len(trials) for max_evals=3.
    """
    space = Space({"a": Float(-5.0, 5.0), "b": Float(-5.0, 5.0)})
    calls = []

    def spy_objective(params):
        calls.append(params)
        return params["a"] ** 2 + params["b"]

    study = optimize(spy_objective, space, max_evals=5, seed=ANY_SEED)

    assert len(calls) == 5
    assert [t.number for t in study.trials] == [0, 1, 2, 3, 4]
    assert study.n_calls == 5


@pytest.mark.parametrize("seed", (0, 1, 2, 3, 4))
def test_abc_should_move_towards_another_source_when_forming_a_neighbour(seed):
    """Given two sources that never move, when 200 neighbours are asked, then no neighbour equals a source.

    Oracle: Karaboga & Basturk (2007) (partner k != i) and ABC.py:92-94 (redraw while neighbour == index).
    colony_size=4 gives SN=2, the most sensitive case: a partner equal to i would be drawn with probability
    1/2 and gives v = x_ij exactly. With one dimension the moved coordinate is the whole point. Every
    neighbour is told 1.0, worse than the initial 0.0, so the sources stay; limit=10**6 rules out scouts.
    Equality with the other source would need phi == -1 exactly. Five seeds; fault-based test for the
    simulated mistake at ABC.py:92-94.
    """
    n_asks = 200

    initial, neighbours = _initial_and_neighbour_coordinates(seed, n_asks, 0.0, 1.0)

    assert initial.intersection(neighbours) == set()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        pytest.param(2, 5, id="limit-2-scout-after-one-cycle"),
        pytest.param(4, 9, id="limit-4-scout-after-two-cycles"),
        pytest.param(3, 9, id="odd-limit-3-no-scout-at-trial-2"),
    ],
)
def test_abc_should_send_a_scout_right_after_the_cycle_in_which_trial_reaches_limit(limit, expected, seed):
    """Given equal scores everywhere, when single asks follow the initial batch, then the first scout comes at
    ask 4 * ceil(limit / 2) + 1.

    Oracle: ABC.py:152 (scout when max trial >= limit), the task specification and a hand calculation:
    with SN=2 and equal fitness, onlooker p = 0.9 * 1 / 1 + 0.1 = 1, so the 2 onlookers visit both sources
    and every neighbour is a non-improvement; each source gains exactly 2 trials per cycle of 4 asks.
    trial reaches limit at the end of cycle ceil(limit / 2), so the scout is ask 4 * ceil(limit / 2) + 1.
    Even limits are decisive for the >= versus > boundary: there trial == limit exactly at a cycle end
    (with > the scout comes one cycle later, at ask 9 or 13). The odd limit 3 is decisive for the other
    side: trial is 2 == limit - 1 after cycle 1 and must not scout (a >= limit - 1 check would scout at ask
    5); trial 4 after cycle 2 scouts at ask 9. Fault-based test for the simulated mistake at ABC.py:152.
    """
    space = Space({"a": Float(-5.0, 5.0), "b": Float(-5.0, 5.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=limit, seed=seed)
    equal_score = 0.0
    max_asks = 4 * limit + 4

    number = _first_scout_ask_number(opt, constant_score=equal_score, max_asks=max_asks)

    assert number == expected


@pytest.mark.parametrize(
    ("onlooker_score", "expected_trial"),
    [
        pytest.param(0.5, [2, 1], id="worse-adds-a-trial"),
        pytest.param(0.0, [2, 1], id="tie-adds-a-trial"),
        pytest.param(-1.0, [0, 1], id="strictly-better-replaces-and-resets"),
    ],
)
def test_abc_should_count_a_trial_unless_the_neighbour_is_strictly_better(onlooker_score, expected_trial):
    """Given two sources scored 0.0 that each already hold one trial, when the first onlooker (source 0) is
    told a score, then opt.trial is [2, 1] for a worse or equal score and [0, 1] for a strictly better one.

    Oracle: task specification (greedy acceptance on fitness: strictly greater replaces, else trial += 1;
    a replaced source starts again with trial 0), ABC.py:140 and Karaboga greedy selection;
    fitness(0.5) = 2/3 < 1 = fitness(0.0) < 2 = fitness(-1.0). Setup by hand: both employed neighbours are
    told 0.5, so trial is [1, 1] and fit stays [1, 1]; then p = 0.9 * 1 / 1 + 0.1 = 1 for both sources, so
    the roulette sweep sends the first onlooker to source 0 whatever the uniform draw. Because trial[0] is 1
    before the onlooker is told, a strictly better score shows the reset (0 instead of 1 or 2). The tie is the
    discriminating boundary; worse and better are the neighbouring classes. Fault-based test for the
    simulated mistake >= at ABC.py:113 and :140, and for a replacement that keeps the old trial count.
    """
    space = Space({"a": Float(-5.0, 5.0), "b": Float(-5.0, 5.0)})
    opt = ABCOptimizer(space, colony_size=4, limit=10, seed=ANY_SEED)
    _tell_initial_then_every_employed(opt, initial_score=0.0, employed_score=0.5)
    assert (opt.phase, opt.trial) == ("onlooker", [1, 1])
    (onlooker,) = opt.ask()

    opt.tell([onlooker], [onlooker_score])

    assert opt.trial == expected_trial


@pytest.mark.parametrize(
    "max_evals",
    [
        pytest.param(1, id="smallest-budget"),
        pytest.param(2, id="inside-initial-batch-2"),
        pytest.param(3, id="inside-initial-batch-3"),
        pytest.param(4, id="one-below-initial-batch"),
        pytest.param(5, id="equals-initial-batch"),
        pytest.param(6, id="one-employed-ask"),
        pytest.param(7, id="two-employed-asks"),
    ],
)
def test_optimize_should_make_exactly_max_evals_calls_when_the_budget_is_small(max_evals):
    """Given a small budget and cache=False, when optimize runs, then trials, calls and n_calls all equal
    max_evals.

    Oracle: task specification (exact max_evals trials, even inside the initial batch) and the optimize
    docstring. The default colony of 10 asks 5 at once: 1..4 truncate inside that batch, 5 equals it, 6 and 7
    cross into single employed asks. Attacker mutants: slicing to max_evals - len(trials) + 1, looping while
    len(trials) <= max_evals, or no truncation.
    """
    space = Space({"a": Float(-5.0, 5.0), "b": Float(-5.0, 5.0)})
    calls = []

    def spy_objective(params):
        calls.append(params)
        return params["a"] ** 2 + params["b"]

    study = optimize(spy_objective, space, max_evals=max_evals, seed=ANY_SEED, cache=False)

    assert len(study.trials) == max_evals
    assert len(calls) == max_evals
    assert study.n_calls == max_evals


@pytest.mark.parametrize(
    ("direction", "expected_number", "expected_score"),
    [
        pytest.param("max", 2, 3.0, id="max-picks-largest-finite-even-if-cached"),
        pytest.param("min", 1, 1.0, id="min-picks-smallest-finite"),
    ],
)
def test_study_best_trial_should_pick_the_extreme_finite_score_for_the_direction(
    direction, expected_number, expected_score
):
    """Given trials scored [nan, 1.0, 3.0 (cached), -inf, 2.0, inf], when best_trial is read, then it is the
    largest finite trial for 'max' and the smallest finite one for 'min'.

    Oracle: Trial.failed docstring (NaN or infinite), README (NaN or infinity is the worst value) and the
    Study semantics (min for 'min', max for 'max'). NaN first makes min/max return nan if failed trials were
    kept; -inf and inf would win if only NaN were filtered; the cached trial holds the maximum, so cached
    trials must stay eligible. Attacker mutants: swapped min/max, keeping failed trials, filtering NaN only.
    """
    study = Study(
        direction,
        [
            _trial(0, math.nan),
            _trial(1, 1.0),
            _trial(2, 3.0, cached=True),
            _trial(3, -math.inf),
            _trial(4, 2.0),
            _trial(5, math.inf),
        ],
    )

    best = study.best_trial

    assert (best.number, best.score) == (expected_number, expected_score)


@pytest.mark.parametrize(
    ("trials", "expected"),
    [
        pytest.param(
            [_trial(0, math.nan), _trial(1, 1.0), _trial(2, 1.0, cached=True), _trial(3, 2.0)],
            3,
            id="one-cached-of-four-failed-call-counts",
        ),
        pytest.param([], 0, id="empty-study"),
    ],
)
def test_study_n_calls_should_count_the_trials_that_were_not_cached(trials, expected):
    """Given trials with cached flags, when n_calls is read, then it counts the uncached ones (3 of 4, 0 of
    0).

    Oracle: Study.n_calls docstring (calls of the user's function; cached repeats are not counted). A NaN
    score was still returned by a call, so the failed trial counts. Attacker mutants: summing cached flags
    (gives 1), counting all trials (gives 4), skipping failed trials (gives 2). The existing
    test_cache_skips_repeated_parameter_sets kills these end to end; this is the direct unit check.
    """
    study = Study("min", trials)

    n_calls = study.n_calls

    assert n_calls == expected


CATEGORICAL_CASES = [
    pytest.param(0.0, 0, id="lower-end-first-choice"),
    pytest.param(0.33, 0, id="inside-first-share-near-edge"),
    pytest.param(0.34, 1, id="inside-second-share-near-edge"),
    pytest.param(0.5, 1, id="second-share-centre"),
    pytest.param(0.66, 1, id="inside-second-share-upper"),
    pytest.param(0.67, 2, id="inside-third-share-near-edge"),
    pytest.param(1.0, 2, id="closed-upper-end-last-choice"),
]


@pytest.mark.parametrize(("u", "index"), CATEGORICAL_CASES)
def test_space_to_dicts_should_give_the_choice_of_the_share_in_declared_order(u, index):
    """Given Categorical(['a', 'b', 'c']), when a unit value is decoded with to_dicts, then the choice whose
    equal share contains u is returned, in declared order.

    Oracle: Categorical docstring (every choice gets an equal share of the unit interval), README table and
    the Space docstring (the unit cube [0, 1]); choice i owns [i/3, (i+1)/3) and the closed end 1.0 belongs
    to the last share. Values sit clearly inside a share on both sides of each edge. Attacker mutant:
    reversed order in the batch decode path (Space.rows).
    """
    space = Space({"k": Categorical(list(LETTER_CHOICES))})

    params = space.to_dicts(np.array([[u]]))[0]

    assert params["k"] is LETTER_CHOICES[index]


@pytest.mark.parametrize(("u", "index"), CATEGORICAL_CASES)
def test_space_move_should_give_the_choice_of_the_share_in_declared_order(u, index):
    """Given Categorical(['a', 'b', 'c']) and a decoded point, when its coordinate is moved to u, then the
    choice whose equal share contains u is returned, in declared order.

    Oracle: as for to_dicts, plus the Space.move docstring (the result is a decoded point differing only in
    coordinate j). The move path (Categorical.decode_one and to_python) is separate code from the batch
    decode, and the existing test_categorical_maps_to_choices covers to_dicts only. Attacker mutant: reversed
    order in the one-coordinate decode path alone.
    """
    space = Space({"k": Categorical(list(LETTER_CHOICES))})
    keys, params = space.rows(np.array([[0.5]]))

    _, new_params = space.move(keys[0], params[0], 0, u)

    assert new_params["k"] is LETTER_CHOICES[index]


def test_abc_ask_should_raise_runtime_error_when_part_of_the_initial_batch_is_still_pending():
    """Given 3 of 5 initial candidates told, when ask is called again, then RuntimeError is raised.

    Oracle: task specification (the initial batch may be told partially; asking again with a pending
    candidate raises RuntimeError). Partial telling is the attacker's input class: a mutant that ignores the
    pending sources re-issues the 2 untold ones instead of raising.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=10, seed=ANY_SEED)
    initial = opt.ask()
    opt.tell(initial[:3], [1.0, 2.0, 3.0])
    assert opt.phase == "init"

    with pytest.raises(RuntimeError):
        opt.ask()


def test_abc_should_leave_the_initial_phase_only_when_the_whole_batch_is_told_out_of_order():
    """Given a 5-candidate initial batch, when it is told as [0, 1, 2], then [4], then [3], then the phase is
    'init', 'init', 'employed', and the next ask hands out a single candidate.

    Oracle: task specification (init may be told partially; employed bees get one candidate per ask) and the
    ABC module docstring (only the initial population is a batch). Out-of-order completion is the attacker's
    input: a mutant that switches phase when any source is initialised reports 'employed' too early.
    """
    space = Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, colony_size=10, seed=ANY_SEED)
    initial = opt.ask()

    opt.tell(initial[:3], [1.0, 2.0, 3.0])
    after_first = opt.phase
    opt.tell([initial[4]], [5.0])
    after_second = opt.phase
    opt.tell([initial[3]], [4.0])
    after_last = opt.phase
    next_batch = opt.ask()

    assert [after_first, after_second, after_last] == ["init", "init", "employed"]
    assert len(next_batch) == 1
