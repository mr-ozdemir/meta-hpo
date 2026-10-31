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
Subset: Hand-made trials with literal scores for both directions.
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
