"""Unit tests of the stopping rules in metahpo.stop: Stopper, Patience, MaxTime and TargetScore.

Oracles: the docstrings of metahpo.stop (the module, Stopper, Patience, MaxTime and TargetScore) and of
Optimizer.settings(), which Stopper.settings() follows. The rules are fed hand-built run states the way
optimize() records a trial (the trial appended to state.study.trials, state.calls counting real calls and
state.best holding the best finite trial), so every expected answer is computed by hand from the score list;
how optimize() calls the rules is tested in test_facade_stop.py.

Unspecified, therefore not asserted: what a rule answers before any trial was recorded (optimize() never
asks then), and the error text of a rejected argument beyond naming the argument.

Input space: Every score sequence with cached flags in both directions, every evals and min_delta of
Patience, every split of a time into seconds, minutes and hours, every target score and any subclass of
Stopper, which are unbounded.
Subset: Hand-written score sequences of four to seven trials that separate a strict from a non-strict
comparison, a moving from a fixed reference, counted from uncounted cached repeats and finite from NaN,
inf and -inf scores, each in the direction where it matters; times one tick below and at the limit; the
boundary values 0, 1 and -1, fractions, booleans, strings, NaN and both infinities for every argument.
"""

import math

import numpy as np
import pytest

from metahpo import (
    Float,
    MaxTime,
    Optimizer,
    Patience,
    RunState,
    Space,
    Stopper,
    Study,
    TargetScore,
    Trial,
)

ANY_SEED = 0

NAN = math.nan

INF = math.inf


def _state(direction):
    """A run state of a one-parameter space before its first trial."""
    space = Space({"x": Float(0.0, 1.0)})
    optimizer = Optimizer(space, direction=direction, seed=ANY_SEED)
    return RunState(space, optimizer, direction, 100, ANY_SEED, Study(direction))


def _answers(rule, scores, direction="min", cached=()):
    """The answer of `rule` after each trial when the scores are recorded one after the other as optimize()
    records them; the trial numbers in `cached` are cached repeats, which do not count as calls."""
    state = _state(direction)
    answers = []
    for number, score in enumerate(scores):
        trial = Trial(number, {"x": 0.5}, score, number in cached, np.zeros(1))
        state.study.trials.append(trial)
        state.calls += number not in cached
        if state.better(score, math.nan if state.best is None else state.best.score):
            state.best = trial
        answers.append(rule.should_stop(state))
    return answers


START = 1000.0
"""The start of the hand-built runs of the MaxTime tests, in seconds since the epoch."""


def _ended_after(seconds, paused=0.0):
    """A run started at START with one trial that ends `seconds` of search time later: it starts after half
    of them plus the `paused` seconds the run was stopped before it (a resume's pause) and lasts the other
    half."""
    state = _state("min")
    state.started = START
    state.pauses = [(0, paused)]
    trial = Trial(0, {"x": 0.5}, 1.0, False, np.zeros(1), {}, START + paused + seconds / 2, seconds / 2)
    state.study.trials.append(trial)
    return state


class _OwnRule(Stopper):
    """A rule of one's own with a stored, a renamed and a defaulted constructor parameter."""

    reason = "own"

    def __init__(self, limit, label="x", extra=None):
        self.limit = limit
        self.label = label
        self.kept_as_other_name = extra


@pytest.mark.parametrize(
    ("direction", "scores"),
    [
        pytest.param("min", [5.0, 4.0, 4.5, 3.0, 4.0, 3.0, 4.0], id="min"),
        pytest.param("max", [-5.0, -4.0, -4.5, -3.0, -4.0, -3.0, -4.0], id="max"),
    ],
)
def test_patience_should_stop_after_exactly_evals_calls_without_improvement(direction, scores):
    """Given Patience(3) and seven calls that improve at calls 1, 2 and 4 and then tie the best, when the
    scores are fed one by one, then it says stop only after the seventh call.

    Oracle: Patience docstring (the first finite score is an improvement; an improvement is better than the
    reference in the run's direction; stop after `evals` calls in a row without one). Hand count: calls 5, 6
    and 7 bring none, and the tie at call 6 is no improvement, so a non-strict comparison would stop later
    and a counter that ignores the improvement at call 4 would stop at call 5.
    """
    rule = Patience(3)

    answers = _answers(rule, scores, direction)

    assert answers == [False, False, False, False, False, False, True]


def test_patience_should_count_only_gains_larger_than_min_delta_and_keep_its_reference():
    """Given Patience(3, min_delta=0.5) and scores 1.0, 0.75, 0.5, 0.25, 0.0, 0.5, 0.5 when minimising, when
    they are fed one by one, then it says stop only at the seventh call.

    Oracle: Patience docstring (an improvement is better than the reference by more than min_delta; only an
    improvement moves the reference). Hand count: against the reference 1.0 the gains are 0.25, 0.5 (not
    more than 0.5) and 0.75, so call 4 is an improvement and calls 5 to 7 end the patience. A rule that
    moved the reference to 0.75 or 0.5 would see a gain of at most 0.5 at call 4 and stop there; a rule
    that took the gain of 0.5 at call 3 as an improvement would stop at call 6.
    """
    rule = Patience(3, min_delta=0.5)

    answers = _answers(rule, [1.0, 0.75, 0.5, 0.25, 0.0, 0.5, 0.5])

    assert answers == [False, False, False, False, False, False, True]


def test_patience_should_not_take_a_gain_equal_to_min_delta_as_an_improvement():
    """Given Patience(2, min_delta=0.5) and the scores 1.0, 1.5, 1.25 when maximising, when they are fed,
    then it says stop at the third call.

    Oracle: Patience docstring (better by more than min_delta). The gain of call 2 is exactly 0.5 and that
    of call 3 is 0.25, so neither is an improvement; a rule that compared with >= would restart the count at
    call 2 and not stop yet.
    """
    rule = Patience(2, min_delta=0.5)

    answers = _answers(rule, [1.0, 1.5, 1.25], "max")

    assert answers == [False, False, True]


def test_patience_should_not_count_cached_repeats_as_calls():
    """Given Patience(3) and a constant score where trials 2 and 3 are cached repeats, when the six trials are
    fed, then it says stop at trial 5, the fourth real call, and not at trial 3.

    Oracle: Patience docstring (calls count as for max_evals: a cached repeat is no call). Hand count: call
    1 is the first finite score; trials 1, 4 and 5 are the three calls without improvement.
    """
    rule = Patience(3)

    answers = _answers(rule, [1.0] * 6, cached=(2, 3))

    assert answers == [False, False, False, False, False, True]


@pytest.mark.parametrize("direction", ["min", "max"])
def test_patience_should_count_nan_and_infinite_scores_as_calls_without_improvement(direction):
    """Given Patience(3) and the scores NaN, inf, -inf, 2.0, NaN, NaN, NaN, when they are fed, then it says
    stop at the third call and again only at the seventh.

    Oracle: Patience docstring (a NaN or infinite score is a call without improvement; the first finite
    score of a run is always an improvement). Neither infinity improves in either direction; 2.0 restarts
    the count, and the three NaN after it end the patience.
    """
    rule = Patience(3)

    answers = _answers(rule, [NAN, INF, -INF, 2.0, NAN, NAN, NAN], direction)

    assert answers == [False, False, True, False, False, False, True]


@pytest.mark.parametrize(
    ("evals", "min_delta"),
    [
        pytest.param(0, 0.0, id="evals-0"),
        pytest.param(-1, 0.0, id="evals-negative"),
        pytest.param(1.5, 0.0, id="evals-fraction"),
        pytest.param(2.0, 0.0, id="evals-float"),
        pytest.param(True, 0.0, id="evals-bool"),
        pytest.param("3", 0.0, id="evals-string"),
        pytest.param(3, -0.1, id="min_delta-negative"),
        pytest.param(3, NAN, id="min_delta-nan"),
        pytest.param(3, INF, id="min_delta-inf"),
    ],
)
def test_patience_should_reject_an_invalid_evals_or_min_delta(evals, min_delta):
    """Given an evals that is not an integer of at least 1 or a min_delta that is not a finite number >= 0,
    when Patience is built, then ValueError is raised.

    Oracle: Patience docstring (`evals` is an integer of at least 1 and `min_delta` a finite number >= 0);
    a bool is not taken for an integer, as for optimize(runs=...).
    """
    with pytest.raises(ValueError, match="evals|min_delta"):
        Patience(evals, min_delta)


def test_patience_should_accept_the_smallest_values_and_a_numpy_integer():
    """Given evals=1 as a numpy integer and min_delta=0, when Patience is built and fed a constant score,
    then it says stop at the second call and records evals as a plain int.

    Oracle: Patience docstring (an integer of at least 1, a finite number >= 0); the first call is the first
    finite score, the second one brings no improvement.
    """
    rule = Patience(np.int64(1), min_delta=0)

    answers = _answers(rule, [1.0, 1.0])

    assert answers == [False, True]
    assert type(rule.evals) is int


@pytest.mark.parametrize(
    ("rule", "limit"),
    [
        pytest.param(MaxTime(seconds=90), 90.0, id="seconds"),
        pytest.param(MaxTime(minutes=1.5), 90.0, id="minutes"),
        pytest.param(MaxTime(hours=0.5), 1800.0, id="hours"),
        pytest.param(MaxTime(seconds=30, minutes=1, hours=1), 3690.0, id="sum-of-the-parts"),
        pytest.param(MaxTime(0.25), 0.25, id="positional-seconds"),
    ],
)
def test_max_time_should_stop_once_the_search_time_reaches_the_sum_of_its_parts(rule, limit):
    """Given MaxTime with a time given in seconds, minutes, hours or all three, when the last trial ends one
    millisecond below the limit and exactly at it, then it says stop only at the limit.

    Oracle: MaxTime docstring (stops when the run has searched for at least seconds + 60 * minutes +
    3600 * hours seconds; the time is state.elapsed() at the trial's start_time + duration).
    """
    states = [_ended_after(limit - 0.001), _ended_after(limit)]

    answers = [rule.should_stop(state) for state in states]

    assert answers == [False, True]


@pytest.mark.parametrize(
    ("clock", "seconds", "expected"),
    [
        pytest.param(START + 3600.0, 89.0, False, id="clock-past-the-limit-trial-before-it"),
        pytest.param(START, 90.0, True, id="clock-at-the-start-trial-at-the-limit"),
    ],
)
def test_max_time_should_decide_from_the_recorded_end_of_the_trial_not_from_the_clock(
    monkeypatch, clock, seconds, expected
):
    """Given MaxTime(seconds=90), a wall clock (RunState.now) that reads `clock` and a last trial that ends
    `seconds` of search time after the start, when the rule is asked, then it answers from the end of the
    trial alone: go on for a trial that ended at 89 s while the clock reads an hour, stop for one that ended
    at 90 s while the clock still reads the start.

    Oracle: MaxTime docstring (the time is state.elapsed() at the trial's start_time + duration; the answer
    depends only on what the results folder records, so resume() replays it exactly). Decisive: a rule
    that reads the clock when it is asked, after the listeners of the trial ran, decides otherwise than its
    replay, which only knows the logged end of the trial.
    """
    monkeypatch.setattr(RunState, "now", lambda state: clock)

    answer = MaxTime(seconds=90).should_stop(_ended_after(seconds))

    assert answer is expected


def test_max_time_should_leave_out_the_time_the_run_was_stopped_before_the_trial():
    """Given MaxTime(seconds=90) and trials that end 89 s and 90 s of search time after the start, each 1000
    s later on the wall clock because the run was stopped for 1000 s before them, when the rule is asked,
    then it goes on after the first and stops after the second.

    Oracle: MaxTime docstring (the time leaves out the time a run was stopped) and RunState.elapsed (the
    time since `started` minus the time the run was stopped before that trial).
    """
    states = [_ended_after(89.0, paused=1000.0), _ended_after(90.0, paused=1000.0)]

    answers = [MaxTime(seconds=90).should_stop(state) for state in states]

    assert answers == [False, True]


@pytest.mark.parametrize(
    "parts",
    [
        pytest.param({}, id="no-time"),
        pytest.param({"seconds": 0, "minutes": 0, "hours": 0}, id="all-zero"),
        pytest.param({"seconds": -1}, id="negative-seconds"),
        pytest.param({"minutes": -0.5, "hours": 1}, id="negative-part-with-positive-sum"),
        pytest.param({"hours": NAN}, id="nan"),
        pytest.param({"seconds": INF}, id="inf"),
        pytest.param({"hours": 1e308}, id="sum-overflows"),
    ],
)
def test_max_time_should_reject_a_time_that_is_not_positive_and_finite(parts):
    """Given parts of which one is negative, NaN or infinite, or whose sum is 0 or overflows, when MaxTime is
    built, then ValueError is raised.

    Oracle: MaxTime docstring (every part is a finite number >= 0 and their sum is positive); 1e308 hours
    are finite but 3600 times that is not, so the sum is not finite.
    """
    with pytest.raises(ValueError):
        MaxTime(**parts)


@pytest.mark.parametrize(
    ("direction", "target", "scores", "expected"),
    [
        pytest.param("min", 0.5, [0.9, 0.7, 0.5, 0.8], [False, False, True, True], id="min-reached-exactly"),
        pytest.param("min", 0.5, [0.9, 0.6, 0.51], [False, False, False], id="min-not-reached"),
        pytest.param("max", 0.5, [0.1, 0.3, 0.5, 0.2], [False, False, True, True], id="max-reached-exactly"),
        pytest.param("max", 0.5, [0.1, 0.4, 0.49], [False, False, False], id="max-not-reached"),
        pytest.param("max", 0.5, [NAN, INF, -INF, 0.6], [False, False, False, True], id="max-non-finite"),
        pytest.param("min", 0.5, [NAN, -INF, INF, 0.4], [False, False, False, True], id="min-non-finite"),
    ],
)
def test_target_score_should_stop_once_the_best_score_is_at_least_as_good_as_the_target(
    direction, target, scores, expected
):
    """Given TargetScore(0.5) and scores that reach it exactly, never reach it or reach it only after NaN and
    infinite scores, when they are fed, then it says stop from the trial whose best reaches the target on.

    Oracle: TargetScore docstring (at most `score` when minimising, at least `score` when maximising, judged
    on the best score so far) and RunState (best is the best finite trial, so NaN and infinite scores never
    become the best and never reach a target).
    """
    rule = TargetScore(target)

    answers = _answers(rule, scores, direction)

    assert answers == expected


@pytest.mark.parametrize("score", [NAN, INF, -INF, "high"])
def test_target_score_should_reject_a_score_that_is_not_a_finite_number(score):
    """Given NaN, an infinity or a string, when TargetScore is built, then ValueError is raised.

    Oracle: TargetScore docstring (`score` is finite); float('high') raises ValueError itself.
    """
    with pytest.raises(ValueError):
        TargetScore(score)


@pytest.mark.parametrize(
    ("rule", "reason"),
    [
        pytest.param(Patience(5), "patience", id="patience"),
        pytest.param(MaxTime(minutes=1), "time", id="max-time"),
        pytest.param(TargetScore(0.9), "target", id="target-score"),
    ],
)
def test_built_in_rule_should_carry_its_documented_stop_reason(rule, reason):
    """Given a built-in rule, when its reason is read, then it is the documented stop reason.

    Oracle: the docstrings of Patience ('patience'), MaxTime ('time') and TargetScore ('target').
    """
    found = rule.reason

    assert found == reason


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        pytest.param(Patience(5, 0.25), {"evals": 5, "min_delta": 0.25}, id="patience"),
        pytest.param(Patience(7), {"evals": 7, "min_delta": 0.0}, id="patience-default"),
        pytest.param(MaxTime(minutes=2), {"seconds": 0.0, "minutes": 2.0, "hours": 0.0}, id="max-time"),
        pytest.param(TargetScore(1), {"score": 1.0}, id="target-score"),
        pytest.param(_OwnRule(3), {"limit": 3, "label": "x"}, id="own-rule"),
    ],
)
def test_stopper_settings_should_read_the_constructor_parameters_from_their_attributes(rule, expected):
    """Given a built-in rule or a rule of one's own, when settings() is read, then it holds every
    constructor parameter kept under its own name with its value, and leaves out one kept under another
    name.

    Oracle: Stopper.settings docstring (the constructor parameters, each read from the attribute of the same
    name; a parameter without such an attribute is left out) and Optimizer.settings, read the same way.
    """
    settings = rule.settings()

    assert settings == expected


def test_stopper_base_class_should_have_no_reason_and_leave_should_stop_to_subclasses():
    """Given the Stopper base class, when its reason is read and should_stop is called, then the reason is
    empty and NotImplementedError is raised.

    Oracle: Stopper docstring (subclass it, set `reason` and override should_stop()); an empty reason is
    one that optimize() rejects.
    """
    rule = Stopper()

    with pytest.raises(NotImplementedError):
        rule.should_stop(_state("min"))
    assert (rule.reason, rule.settings()) == ("", {})
