"""Coverage-driven tests of Study: branches and the stdlib coverage measurement.

Study tests added after the stdlib coverage measurement: no finite trial, direction of the best, n_calls.

They close study line 42 and edge 41L (no finite trial), line 48 (best_params) and both sides of the
min/max pick at line 43, found red or one-sided by the baseline trace and branch probe.

Oracles: study.py docstrings ('True when the user's function returned NaN or an infinite value', 'cached
repeats are not counted'), README Behaviour ('A score of NaN or infinity counts as the worst possible value
for the chosen direction') and the task specification of Study (best_trial raises ValueError if no finite
trial exists; best_params and best_score follow best_trial).

Unspecified, therefore not asserted: which trial wins a tie of equal finite scores (all finite scores below
are distinct).

Branch-coverage tests of Study.best_trial, derived from the control flow graph below.

Control flow graph (course notation: nodes are the real line numbers of src/metahpo/study.py, straight-line
code is merged, every decision has one T and one F edge, s and e are entry and exit, a raise goes to e; the
conditional expression on line 43 is split into t and f nodes):

Study.best_trial, study.py 38-44
    s   -> 40 -> 41           finite = trials whose score is finite
    41  -T-> 42 -> e          no finite trial: raise ValueError
    41  -F-> 43
    43  -T-> 43t -> 44        direction == "min": pick = min
    43  -F-> 43f -> 44        pick = max
    44  -> e                  return the picked finite trial

Test requirements (branch coverage, one per T/F edge):
    STUDY-1  41 T       STUDY-2  41 F       STUDY-3  43 T       STUDY-4  43 F

Coverage level: 4 of 4 feasible branch requirements (100 percent), all by this module. best_params and
best_score delegate to best_trial and have no decision of their own; Trial.failed and n_calls are single
expressions (out of this module's scope, tested by other work items).

Unspecified, therefore not asserted: which trial is best when two finite scores are equal.

Box types (P-T4-08): the no-finite-trial test is designed from edge 41 T (white); the min and max tests come
from the task spec and the README (black).

Trials are built directly with the public Trial constructor; x is irrelevant to best_trial, so every trial
gets the same IRRELEVANT_X.

Input space: Every list of trials with any scores and cached flags, for both directions.
Subset: One scenario per branch of Study.best_trial and per unreached statement.
"""

import math

import numpy as np
import pytest

from metahpo import Study, Trial

ANY_X = np.zeros(1)


def _trial(number, score, cached=False):
    """A Trial whose params {'a': number} identify it; x is irrelevant to Study."""
    return Trial(number, {"a": number}, score, cached, ANY_X)


def _trials_with_scores(scores):
    """One uncached trial per score, numbered in order."""
    return [_trial(number, score) for number, score in enumerate(scores)]


def _failed_flags(study):
    """The failed flag of every trial, in trial order."""
    return [t.failed for t in study.trials]


def _mixed_trials():
    """Finite scores 3.0, -1.0, 2.0 (the last one cached) mixed with NaN, +inf and -inf."""
    return [
        _trial(0, 3.0),
        _trial(1, math.nan),
        _trial(2, -1.0),
        _trial(3, 2.0, cached=True),
        _trial(4, math.inf),
        _trial(5, -math.inf),
    ]


@pytest.mark.parametrize("attribute", ["best_trial", "best_params", "best_score"])
@pytest.mark.parametrize(
    ("direction", "scores"),
    [
        pytest.param("min", [], id="empty_study_zero_class"),
        pytest.param("max", [math.nan, math.inf, -math.inf], id="nan_plus_inf_minus_inf_only"),
    ],
)
def test_study_best_should_raise_value_error_when_no_trial_has_a_finite_score(direction, scores, attribute):
    """Given no trial or only NaN, +inf and -inf trials, when a best_* property is read, then ValueError.

    Oracle: task specification 'best_trial (ValueError if no finite trial)'; best_params and best_score are
    'of the best trial', so they raise alike. Inputs: the empty study is the zero class; NaN, +inf and -inf
    are the three non-finite classes, and -inf is decisive against a check for +inf only. Site: study line
    42, edge 41L.
    """
    study = Study(direction, _trials_with_scores(scores))
    assert _failed_flags(study) == [True] * len(scores)

    with pytest.raises(ValueError):
        getattr(study, attribute)


@pytest.mark.parametrize(
    ("direction", "expected_score", "expected_params"),
    [
        pytest.param("min", -1.0, {"a": 2}, id="min_picks_smallest_finite_not_minus_inf"),
        pytest.param("max", 3.0, {"a": 0}, id="max_picks_largest_finite_not_plus_inf"),
    ],
)
def test_study_best_should_be_the_extreme_finite_trial_for_the_direction(
    direction, expected_score, expected_params
):
    """Given finite and non-finite trials, when best_score and best_params are read, then the finite extreme
    wins.

    Oracle: README Behaviour (NaN or infinity is the worst value for the direction) and study.py 'failed'.
    Inputs: -inf is what a naive min would return and +inf what a naive max would return, so both are
    decisive; finite scores are distinct so no tie rule is involved. Sites: both sides of the min/max pick at
    study line 43 and best_params at line 48.
    """
    study = Study(direction, _mixed_trials())

    best_score = study.best_score
    best_params = study.best_params

    assert best_score == expected_score
    assert best_params == expected_params
    assert best_params is study.best_trial.params


def test_study_n_calls_should_not_count_the_cached_trial():
    """Given six trials of which one is cached, when n_calls is read, then it is 5.

    Oracle: study.py docstring 'Calls of the user's function; cached repeats are not counted'. Postcondition:
    none, the property is pure.
    """
    study = Study("min", _mixed_trials())

    n_calls = study.n_calls

    assert n_calls == 5


IRRELEVANT_X = np.zeros(1)


def _study(direction, scores):
    """A Study whose trials, numbered in order, have the given scores and params {'i': number}."""
    trials = [Trial(n, {"i": n}, s, False, IRRELEVANT_X) for n, s in enumerate(scores)]
    return Study(direction, trials)


@pytest.mark.parametrize("direction", ["min", "max"])
def test_study_best_trial_should_raise_value_error_in_either_direction_when_scores_are_nan_and_inf(direction):
    """Without a finite score there is no best trial, in either direction.

    Given a Study whose two trials scored NaN and +inf, crafted so that the finite list is empty (41 T).
    When best_trial is read.
    Then it raises ValueError.
    Covers: STUDY-1
    Box: white
    Source: task spec (best_trial raises ValueError if no finite trial); Trial.failed docstring (NaN or an
    infinite value means failed).
    """
    study = _study(direction, [math.nan, math.inf])

    with pytest.raises(ValueError):
        _ = study.best_trial


def test_study_best_trial_should_pick_the_smallest_finite_score_when_minimising():
    """Under 'min' the best trial is the finite trial with the smallest score; failed trials are skipped.

    Given scores [3.0, nan, -1.0, 2.0, -inf]: -inf is numerically smallest but failed, so it must be skipped
    (41 F, 43 T).
    When best_trial is read.
    Then it is trial 2 (score -1.0).
    Covers: STUDY-2, STUDY-3
    Box: black
    Source: task spec (failed = non-finite score; best_trial is the best finite trial); README Behaviour.
    """
    study = _study("min", [3.0, math.nan, -1.0, 2.0, -math.inf])

    best = study.best_trial

    assert best.number == 2


def test_study_best_trial_should_pick_the_largest_finite_score_when_maximising():
    """Under 'max' the best trial is the finite trial with the largest score; +inf is a failure, not a best.

    Given scores [3.0, inf, -1.0, nan]: +inf is numerically largest but failed (41 F, 43 F).
    When best_trial is read.
    Then it is trial 0 (score 3.0).
    Covers: STUDY-2, STUDY-4
    Box: black
    Source: task spec (failed = non-finite score); README Behaviour ('NaN or infinity counts as the worst
    possible value for the chosen direction').
    """
    study = _study("max", [3.0, math.inf, -1.0, math.nan])

    best = study.best_trial

    assert best.number == 0
