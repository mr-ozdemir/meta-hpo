"""Unit tests of Trial and Study (study.py), including the error path of best_trial.

Unit tests of Trial and Study, built from hand-made Trial objects only (no optimizer, no Space, no optimize).

Grouped by method: Trial.failed, Study.best_trial, Study.best_params, Study.best_score, Study.n_calls,
Study.scores. Every expected value is a hand calculation on the literal scores of the test.

Unspecified, not asserted: tie-breaking between trials with equal best scores, and the behaviour of a Study
whose direction is neither 'min' nor 'max' (Study does not validate it).

Error path of Study.best_trial when no trial has a finite score.

Covered raise site: study.py, Study.best_trial, ValueError "finite". The full inventory is in the module
docstring of test_space_errors.py. best_trial has a single error path, so only the type is checked.

Input space: Every list of trials with any scores and cached flags, for both directions.
Subset: Hand-made trials with literal scores for both directions; pairs of trials with a two-element x
that differ in exactly one compared field or only in the timing fields, NaN scores, numpy arrays in
params and meta, and values of other classes; pairs of studies that differ in one field.
"""

import math

import numpy as np
import pytest

from metahpo import Study, Trial

IRRELEVANT_X = np.zeros(1)


def _trial(number, score, params=None, cached=False):
    """A Trial in which the score, and where given the params or cached flag, are the values that matter."""
    return Trial(number, {"n": number} if params is None else params, score, cached, IRRELEVANT_X)


def _study(direction, scores):
    """A Study whose trial i has scores[i] and params {'n': i}."""
    return Study(direction, [_trial(i, s) for i, s in enumerate(scores)])


def _study_with_cached_flags(flags):
    """A 'min' Study whose trial i has the cached flag flags[i]; the score 1.0 does not matter."""
    return Study("min", [_trial(i, 1.0, cached=c) for i, c in enumerate(flags)])


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        pytest.param(math.nan, True, id="nan-is-failed"),
        pytest.param(math.inf, True, id="plus-inf-is-failed"),
        pytest.param(-math.inf, True, id="minus-inf-is-failed"),
        pytest.param(0.0, False, id="zero-boundary-is-not-failed"),
        pytest.param(-5.0, False, id="negative-finite-is-not-failed"),
    ],
)
def test_trial_failed_should_be_true_only_when_score_is_not_finite(score, expected):
    """Given a Trial with one score, when failed is read, then it is True exactly for NaN and +-inf.

    Oracle: Trial.failed docstring 'True when the user's function returned NaN or an infinite value' and the
    README Behaviour bullet on NaN and infinity. Inputs: NaN, +inf and -inf are the three non-finite classes;
    the infinite cases are decisive against a NaN-only check (site: Trial.failed return). 0.0 is the zero
    boundary of the finite class and -5.0 shows that a negative score is not read as a failure.
    Postcondition: none, failed is a pure property.
    """
    trial = _trial(0, score)

    failed = trial.failed

    assert failed is expected


def test_study_best_trial_should_ignore_nan_trials_when_minimising():
    """Given a 'min' Study with scores [3.0, nan, 1.0, 2.0], when best_trial is read, then it is trial 2
    (1.0).

    Oracle: README 'NaN or infinity counts as the worst possible value for the chosen direction'; the minimum
    of the finite scores {3.0, 1.0, 2.0} is 1.0 at index 2. Inputs: the best trial is in the middle with NaN
    placed before it, so keeping NaN in the comparison or picking by position gives another trial
    (site: Study.best_trial filter and min branch).
    """
    study = _study("min", [3.0, math.nan, 1.0, 2.0])

    best = study.best_trial

    assert best.number == 2
    assert best.score == 1.0


def test_study_best_trial_should_ignore_infinite_and_nan_trials_when_maximising():
    """Given a 'max' Study with scores [1.0, inf, 4.0, nan, 2.0], when best_trial is read, then it is trial 2
    (4.0).

    Oracle: README Behaviour bullet, +inf is a failure and therefore the worst value for 'max'; the maximum of
    the finite scores {1.0, 4.0, 2.0} is 4.0 at index 2. Inputs: +inf is decisive, because a plain max over
    all trials, or a failed check that only looks for NaN, returns the inf trial; a 'min' pick returns the 1.0
    trial (sites: Trial.failed, Study.best_trial direction choice).
    """
    study = _study("max", [1.0, math.inf, 4.0, math.nan, 2.0])

    best = study.best_trial

    assert best.number == 2
    assert best.score == 4.0


@pytest.mark.parametrize(
    ("direction", "scores"),
    [
        pytest.param("min", [], id="empty-study"),
        pytest.param("min", [math.nan, math.nan], id="only-nan-min"),
        pytest.param("min", [math.nan, math.inf, -math.inf], id="nan-and-infinities-min"),
        pytest.param("max", [math.nan, math.inf, -math.inf], id="nan-and-infinities-max"),
        pytest.param("min", [math.inf, -math.inf], id="only-infinities-min"),
    ],
)
def test_study_best_trial_should_raise_value_error_when_no_trial_has_a_finite_score(direction, scores):
    """Given a Study without any finite score, when best_trial is read, then ValueError is raised.

    Oracle: task specification 'best_trial (ValueError if no finite trial)' and the Trial.failed docstring.
    Inputs: the empty Study is the zero-trial boundary; only-NaN is the plain failed class; the NaN/+inf/-inf
    mix is decisive against a NaN-only failed check (the infinities would survive the filter and be returned);
    both directions show the raise does not depend on the direction; +inf with -inf and no NaN shows that
    -inf is not taken as a best minimum. Site: Study.best_trial 'no finite trial -> raise ValueError'.
    """
    study = _study(direction, scores)

    with pytest.raises(ValueError):
        _ = study.best_trial


def test_study_best_params_should_return_params_of_best_trial_when_it_is_neither_first_nor_last():
    """Given a 'min' Study with scores [5.0, 1.0, 3.0], when best_params is read, then it is trial 1's params.

    Oracle: task specification, best_params belongs to the best trial; the minimum of [5.0, 1.0, 3.0] is at
    index 1. Inputs: the best trial is in the middle, so returning the first or the last trial's params is
    detected (site: Study.best_params body).
    """
    study = Study("min", [_trial(0, 5.0, {"x": 0}), _trial(1, 1.0, {"x": 1}), _trial(2, 3.0, {"x": 2})])

    params = study.best_params

    assert params == {"x": 1}


def test_study_best_score_should_return_score_of_best_trial_when_it_is_not_first():
    """Given a 'min' Study with scores [5.0, 1.0, 3.0], when best_score is read, then it is 1.0.

    Oracle: task specification, best_score is the best trial's score; min(5.0, 1.0, 3.0) = 1.0. Inputs: the
    first trial holds 5.0, so returning the first trial's score is detected (site: Study.best_score body).
    """
    study = _study("min", [5.0, 1.0, 3.0])

    score = study.best_score

    assert score == 1.0


def test_study_n_calls_should_count_only_non_cached_trials():
    """Given five trials with cached flags [False, True, False, True, True], when n_calls is read, then it is
    2.

    Oracle: Study.n_calls docstring 'Calls of the user's function; cached repeats are not counted'; two flags
    are False. Inputs: cached trials outnumber calls, so counting all trials (5) or counting the cached ones
    (3) is detected (site: Study.n_calls sum).
    """
    study = _study_with_cached_flags([False, True, False, True, True])

    n_calls = study.n_calls

    assert n_calls == 2


def test_study_scores_should_be_float64_array_in_trial_order_when_a_score_is_nan():
    """Given trial scores [0.1, nan, 2.5], when scores is read, then it is a float64 array [0.1, nan, 2.5].

    Oracle: task specification 'Study(direction) with scores array' and the public return type
    NDArray[np.float64] of Study.scores. Inputs: 0.1 is not exactly representable in float32, so a narrower
    dtype changes the value; the NaN in the middle checks that failed trials keep their place
    (site: Study.scores array construction).
    """
    study = _study("min", [0.1, math.nan, 2.5])

    scores = study.scores

    assert type(scores) is np.ndarray
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(scores, np.array([0.1, math.nan, 2.5]))


class _Incomparable:
    """A params value whose == raises, as a numpy array of several elements does in a boolean context."""

    def __eq__(self, other):
        raise RuntimeError("cannot compare")


def _pair_trial(**changes):
    """A trial with a two-element x, params holding a numpy array, a NaN-free score and meta; `changes`
    replaces fields."""
    fields = {
        "number": 3,
        "params": {"lr": 0.1, "w": np.array([1.0, 5.0])},
        "score": 0.5,
        "cached": False,
        "x": np.array([0.25, 0.75]),
        "meta": {"phase": "fly", "generation": 1, "individual": 2},
        "start_time": 100.0,
        "duration": 2.0,
    }
    return Trial(**{**fields, **changes})


def test_trial_should_equal_a_twin_with_a_two_element_x_and_an_array_in_its_params():
    """Given two trials built from equal fields with a two-element x and a numpy array in params, when they
    are compared with == and !=, then == gives True and != gives False, both as bools.

    Oracle: the Trial docstring (the timing fields are measurements, left out of ==, so the same seed gives
    equal trials) and Python's comparison protocol (== gives a bool). Decisive: the dataclass equality
    compared the x arrays inside a tuple, which raised 'The truth value of an array ... is ambiguous'.
    """
    a, b = _pair_trial(), _pair_trial()

    results = (a == b, a != b)

    assert results == (True, False)
    assert (type(results[0]), type(results[1])) == (bool, bool)


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"number": 4}, id="number"),
        pytest.param({"params": {"lr": 0.1, "w": np.array([1.0, 6.0])}}, id="array-in-params"),
        pytest.param({"params": {"lr": 0.2, "w": np.array([1.0, 5.0])}}, id="number-in-params"),
        pytest.param({"params": {"lr": 0.1, "w": np.array([1.0, 5.0, 1.0])}}, id="array-shape-in-params"),
        pytest.param({"score": 0.6}, id="score"),
        pytest.param({"score": math.nan}, id="nan-against-a-number"),
        pytest.param({"cached": True}, id="cached"),
        pytest.param({"x": np.array([0.25, 0.5])}, id="x"),
        pytest.param({"x": np.array([0.25, 0.75, 0.0])}, id="x-length"),
        pytest.param({"meta": {"phase": "init", "generation": 1, "individual": 2}}, id="meta"),
        pytest.param({"params": {"layers": [32, 65]}}, id="list-in-params"),
        pytest.param({"params": {"lr": 0.1, "w": (1.0, 5.0)}}, id="tuple-instead-of-array"),
        pytest.param({"params": {"lr": _Incomparable(), "w": np.array([1.0, 5.0])}}, id="comparison-raises"),
    ],
)
def test_trial_should_differ_from_a_trial_that_differs_in_one_compared_field(changes):
    """Given a trial and a copy that differs in exactly one of number, params, score, cached, x or meta,
    when they are compared, then == gives False and != gives True.

    Oracle: the Trial docstring (only the timing fields are left out of ==) and the dataclass field list.
    """
    a, b = _pair_trial(), _pair_trial(**changes)

    assert (a == b, a != b) == (False, True)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param({}, {"start_time": 200.0, "duration": 9.0}, id="other-timing"),
        pytest.param(
            {"score": float("nan")}, {"score": float("nan"), "start_time": 7.0}, id="both-failed-with-nan"
        ),
        pytest.param(
            {"params": {"layers": [32, 64], "pair": (1, 2)}},
            {"params": {"layers": [32, 64], "pair": (1, 2)}},
            id="equal-lists-and-tuples-in-params",
        ),
    ],
)
def test_trial_should_equal_a_trial_that_differs_only_in_timing_or_has_a_nan_score_as_well(left, right):
    """Given two trials that differ only in their start time and duration, that both failed with a NaN
    score, or whose params hold equal lists and tuples, when they are compared, then == gives True.

    Oracle: the Trial docstring (timing fields are left out of ==, so the same seed still gives equal
    trials; a failed trial is a NaN or infinite score), so two failed twins of the same seed are equal.
    Decisive: NaN == NaN is False, so a failed trial never equalled its own twin. The NaN scores are two
    different float objects, and lists and tuples in params are compared item by item.
    """
    assert _pair_trial(**left) == _pair_trial(**right)


@pytest.mark.parametrize(
    "other", [pytest.param(3, id="int"), pytest.param("trial", id="string"), pytest.param(None, id="none")]
)
def test_trial_and_study_should_leave_a_comparison_with_another_class_to_python(other):
    """Given a trial and a study compared with an int, a string and None, when == runs, then Trial.__eq__
    and Study.__eq__ give NotImplemented and the comparison falls back to False.

    Oracle: Python's data model (__eq__ returns NotImplemented for an unsupported operand) and the dataclass
    convention the classes kept (only an object of the same class compares).
    """
    trial, study = _pair_trial(), Study("min", [_pair_trial()])

    assert (trial.__eq__(other), study.__eq__(other)) == (NotImplemented, NotImplemented)
    assert (trial == other, study == other, trial != other) == (False, False, True)


def test_trial_and_study_should_stay_unhashable_as_before():
    """Given a trial, whose params dict is unhashable, and a study, when hash() is called, then both raise
    TypeError, as in 0.7.1.

    Oracle: the stage brief (hashability stays what it is) and the dataclass rules: a frozen Trial hashes its
    fields, which include a dict; a mutable Study with == has no hash.
    """
    with pytest.raises(TypeError):
        hash(_pair_trial())
    with pytest.raises(TypeError):
        hash(Study("min"))

    assert Study.__hash__ is None


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        pytest.param(Study("max", [_pair_trial(), _pair_trial(number=4)], "budget"), True, id="twin"),
        pytest.param(Study("min", [_pair_trial(), _pair_trial(number=4)], "budget"), False, id="direction"),
        pytest.param(
            Study("max", [_pair_trial(), _pair_trial(number=4)], "patience"), False, id="stop-reason"
        ),
        pytest.param(Study("max", [_pair_trial()], "budget"), False, id="fewer-trials"),
        pytest.param(Study("max", [_pair_trial(), _pair_trial(number=5)], "budget"), False, id="other-trial"),
    ],
)
def test_study_should_compare_direction_stop_reason_and_every_trial(other, expected):
    """Given a 'max' study with two trials and stop reason 'budget', when it is compared with a twin and with
    studies that differ in direction, stop reason, number of trials or one trial, then == gives True only
    for the twin and != the opposite, both as bools.

    Oracle: the Study fields (direction, trials, stop_reason) and Python's comparison protocol. Decisive:
    the dataclass equality compared the trials' x arrays and raised ValueError.
    """
    study = Study("max", [_pair_trial(), _pair_trial(number=4)], "budget")

    assert (study == other, study != other) == (expected, not expected)
