"""Unit tests of metahpo.prune: TrialPruned, Pruner, NopPruner, PercentilePruner, MedianPruner and
RunningTrial.

Oracles: the docstrings of metahpo.prune (the module, TrialPruned, Pruner, NopPruner, PercentilePruner,
MedianPruner and RunningTrial), which follow Optuna's pruners: a trial is pruned when its best reported value
so far is worse than the percentile of the values the completed trials reported at exactly its last step, with
n_startup_trials, n_warmup_steps, interval_steps and n_min_trials as in Optuna. The pruners are fed hand-built
studies (Trial objects with intermediate values) and RunningTrial handles, so every threshold is computed by
hand from four to five reference values with numpy's default linear percentile; how optimize() asks the pruner
is tested in test_facade_prune.py.

Unspecified, therefore not asserted: the error text of a rejected argument beyond naming the argument, and
what prune() answers for a study of another direction than the trial's run (optimize() never mixes them).

Input space: Every study of completed, pruned, failed and cached trials with any intermediate values, every
reported step sequence, every percentile and every setting of the pruners, which are unbounded. Subset:
Reference sets of one to five completed trials at steps 0 to 3; trials that report every step, skip steps or
repeat one; the comparison at, just above and just below the threshold in both directions; NaN in the trial
and in the reference; startup, warm-up, interval and minimum-count limits one below and at their value; the
boundary values 0, 1, 100 and -1, fractions, booleans, strings, NaN and both infinities for every setting.
"""

import math

import numpy as np
import pytest

from metahpo import (
    MedianPruner,
    NopPruner,
    PercentilePruner,
    Pruner,
    RunningTrial,
    Study,
    Trial,
    TrialPruned,
)

NAN = math.nan

INF = math.inf


def _trial(number, values, score=1.0, pruned=False, cached=False):
    """A recorded trial of a one-parameter space with the given intermediate values."""
    return Trial(number, {"x": 0.5}, score, cached, np.zeros(1), {}, 0.0, 0.0, pruned, dict(values))


def _study(direction, *reference):
    """A study of completed trials, one per dict of intermediate values in `reference`."""
    study = Study(direction)
    study.trials.extend(_trial(number, values) for number, values in enumerate(reference))
    return study


def _running(study, values, protected=False):
    """A running trial of `study` that reported `values` (step -> value) in order."""
    handle = RunningTrial(len(study.trials), {"x": 0.5}, study, MedianPruner(n_startup_trials=0), protected)
    [handle.report(value, step) for step, value in values.items()]
    return handle


FIVE = ({0: 1.0}, {0: 2.0}, {0: 3.0}, {0: 4.0}, {0: 5.0})
"""Five completed trials that reported 1.0 to 5.0 at step 0: median 3.0, 25th percentile 2.0, 75th 4.0."""


@pytest.mark.parametrize(
    ("direction", "value", "expected"),
    [
        pytest.param("min", 3.5, True, id="min-worse"),
        pytest.param("min", 3.0, False, id="min-equal"),
        pytest.param("min", 2.5, False, id="min-better"),
        pytest.param("max", 2.5, True, id="max-worse"),
        pytest.param("max", 3.0, False, id="max-equal"),
        pytest.param("max", 3.5, False, id="max-better"),
    ],
)
def test_median_pruner_should_prune_only_a_value_strictly_worse_than_the_median_at_the_step(
    direction, value, expected
):
    """Given five completed trials that reported 1.0 to 5.0 at step 0 and a trial that reported `value` at
    step 0, when MedianPruner(n_startup_trials=0) is asked, then it prunes only a value strictly worse than
    the median 3.0 in the run's direction.

    Oracle: MedianPruner docstring (worse than the median of the values the completed trials reported at the
    same step). Hand count: the median of 1..5 is 3.0; equality is not worse, so a non-strict comparison would
    prune the equal case and a comparison in the wrong direction would swap the other two.
    """
    study = _study(direction, *FIVE)
    trial = _running(study, {0: value})

    pruned = MedianPruner(n_startup_trials=0).prune(study, trial)

    assert pruned is expected


@pytest.mark.parametrize(
    ("direction", "value", "expected"),
    [
        pytest.param("min", 2.5, True, id="min-outside-best-quarter"),
        pytest.param("min", 2.0, False, id="min-at-25th"),
        pytest.param("max", 3.5, True, id="max-outside-best-quarter"),
        pytest.param("max", 4.0, False, id="max-at-75th"),
    ],
)
def test_percentile_pruner_should_keep_only_the_best_quarter_with_percentile_25(direction, value, expected):
    """Given five completed trials that reported 1.0 to 5.0 at step 0, when PercentilePruner(25.0,
    n_startup_trials=0) judges a trial that reported `value`, then it keeps the trial only while it is within
    the best quarter, counted from the bottom when minimising and from the top when maximising.

    Oracle: PercentilePruner docstring (percentile=25.0 keeps a trial only while it is among the best quarter;
    the threshold is counted from the best end). Hand count with the linear percentile of 1..5: the 25th
    percentile is 2.0 and the 75th is 4.0; a value at the threshold is kept.
    """
    study = _study(direction, *FIVE)
    trial = _running(study, {0: value})

    pruned = PercentilePruner(25.0, n_startup_trials=0).prune(study, trial)

    assert pruned is expected


def test_percentile_pruner_should_compare_the_best_value_so_far_not_the_last_one():
    """Given completed trials that reported 1.0 to 5.0 at step 1 and a trial that reported 1.0 at step 0 and
    4.0 at step 1 when minimising, when MedianPruner judges it at step 1, then it is not pruned.

    Oracle: PercentilePruner docstring (a trial whose best intermediate value so far is worse than the
    threshold). Hand count: the last value 4.0 is worse than the median 3.0, the best value 1.0 is not, so a
    pruner that compared the last value would prune.
    """
    study = _study("min", *[{1: v} for v in (1.0, 2.0, 3.0, 4.0, 5.0)])
    trial = _running(study, {0: 1.0, 1: 4.0})

    pruned = MedianPruner(n_startup_trials=0).prune(study, trial)

    assert pruned is False


def test_percentile_pruner_should_use_only_completed_trials_that_reported_exactly_the_last_step():
    """Given two completed trials that reported 9.0 at step 1, two that reported 1.0 only at step 0 and 2, and
    pruned, failed and cached trials that reported 0.0 at step 1, when MedianPruner(n_startup_trials=0,
    n_min_trials=2) judges a trial whose best value 5.0 was reported at step 1 when minimising, then it is not
    pruned.

    Oracle: PercentilePruner docstring (the reference is the completed trials, cached repeats not counted,
    that reported a value at exactly the step the trial reported last). Hand count: the reference at step 1 is
    [9.0, 9.0], median 9.0, and 5.0 is better; counting the trials of steps 0 and 2 (no interpolation) or the
    pruned, failed and cached trials would bring the median to 1.0 or 0.0 and prune.
    """
    study = Study("min")
    study.trials.extend(
        [
            _trial(0, {1: 9.0}),
            _trial(1, {1: 9.0}),
            _trial(2, {0: 1.0, 2: 1.0}),
            _trial(3, {0: 1.0, 2: 1.0}),
            _trial(4, {1: 0.0}, score=0.0, pruned=True),
            _trial(5, {1: 0.0}, score=NAN),
            _trial(6, {1: 0.0}, score=0.0, cached=True),
        ]
    )
    trial = _running(study, {1: 5.0})

    pruned = MedianPruner(n_startup_trials=0, n_min_trials=2).prune(study, trial)

    assert pruned is False


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        pytest.param(2, False, id="one-below-startup"),
        pytest.param(3, True, id="at-startup"),
    ],
)
def test_percentile_pruner_should_wait_for_n_startup_trials_completed_trials(completed, expected):
    """Given `completed` completed trials that reported 1.0 at step 0 next to a pruned and a cached trial,
    when MedianPruner(n_startup_trials=3) judges a trial that reported 2.0 at step 0 when minimising, then it
    prunes only once three trials have completed.

    Oracle: PercentilePruner docstring (not pruned while fewer than `n_startup_trials` trials have completed;
    cached repeats do not count). Hand count: the median of the reference is 1.0 and 2.0 is worse; the pruned
    and the cached trial would reach three with two completed ones if they counted.
    """
    study = _study("min", *[{0: 1.0}] * completed)
    study.trials.append(_trial(completed, {0: 1.0}, pruned=True))
    study.trials.append(_trial(completed + 1, {0: 1.0}, cached=True))
    trial = _running(study, {0: 2.0})

    pruned = MedianPruner(n_startup_trials=3).prune(study, trial)

    assert pruned is expected


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        pytest.param(1, False, id="one-below-warmup"),
        pytest.param(2, True, id="at-warmup"),
    ],
)
def test_percentile_pruner_should_not_prune_before_n_warmup_steps(step, expected):
    """Given completed trials that reported 1.0 at steps 0 to 3 and a trial that reported 2.0 at `step`, when
    MedianPruner(n_startup_trials=0, n_warmup_steps=2) judges it when minimising, then it prunes only from
    step 2 on.

    Oracle: PercentilePruner docstring (not pruned while its last step is below `n_warmup_steps`). Hand count:
    the median at every step is 1.0 and 2.0 is worse.
    """
    reference = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    study = _study("min", reference, reference, reference)
    trial = _running(study, {step: 2.0})

    pruned = MedianPruner(n_startup_trials=0, n_warmup_steps=2).prune(study, trial)

    assert pruned is expected


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        pytest.param([0, 1, 2, 3], True, id="3-first-of-interval-3-5"),
        pytest.param([0, 3, 4], False, id="4-after-3-in-the-same-interval"),
        pytest.param([0, 4], True, id="4-first-reported-in-its-interval"),
        pytest.param([0, 1], False, id="1-after-0-in-interval-0-2"),
    ],
)
def test_percentile_pruner_should_judge_only_the_first_step_reported_in_each_interval(steps, expected):
    """Given completed trials that reported 1.0 at steps 0 to 4 and a trial that reported 2.0 at each of
    `steps`, when MedianPruner(n_startup_trials=0, interval_steps=3) judges it when minimising, then it is
    pruned only when its last step is the first step it reported in its interval (0-2, 3-5).

    Oracle: PercentilePruner docstring (after the warm-up a trial is judged once per `interval_steps` steps,
    at the first step it reports in each interval counted from `n_warmup_steps`). Every value 2.0 is worse
    than the median 1.0, so a judged trial is pruned and the answer shows whether it was judged: step 4 is
    judged when step 3 was skipped, and not when step 3 was reported.
    """
    reference = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    study = _study("min", reference, reference, reference)
    trial = _running(study, dict.fromkeys(steps, 2.0))

    pruned = MedianPruner(n_startup_trials=0, interval_steps=3).prune(study, trial)

    assert pruned is expected


@pytest.mark.parametrize(
    ("n_min_trials", "expected"),
    [
        pytest.param(3, True, id="at-minimum"),
        pytest.param(4, False, id="one-above"),
    ],
)
def test_percentile_pruner_should_need_n_min_trials_values_at_the_step(n_min_trials, expected):
    """Given three completed trials that reported 1.0 at step 0, when MedianPruner(n_startup_trials=0,
    n_min_trials=n) judges a trial that reported 2.0 at step 0 when minimising, then it prunes only when at
    least n completed trials reported the step.

    Oracle: PercentilePruner docstring (not pruned when fewer than `n_min_trials` completed trials reported
    that step).
    """
    study = _study("min", {0: 1.0}, {0: 1.0}, {0: 1.0})
    trial = _running(study, {0: 2.0})

    pruned = MedianPruner(n_startup_trials=0, n_min_trials=n_min_trials).prune(study, trial)

    assert pruned is expected


def test_percentile_pruner_should_prune_a_trial_whose_every_value_is_nan():
    """Given five completed trials that reported 1.0 to 5.0 at step 0 and a trial that reported NaN at steps 0
    and 1, when MedianPruner(n_startup_trials=0) judges it, then it is pruned, although no completed trial
    reported step 1.

    Oracle: PercentilePruner docstring (a trial whose every reported value is NaN is pruned).
    """
    study = _study("min", *FIVE)
    trial = _running(study, {0: NAN, 1: NAN})

    pruned = MedianPruner(n_startup_trials=0).prune(study, trial)

    assert pruned is True


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        pytest.param(({0: 1.0}, {0: NAN}, {0: 1.0}), True, id="nan-left-out"),
        pytest.param(({0: NAN}, {0: NAN}), False, id="all-nan"),
    ],
)
def test_percentile_pruner_should_leave_nan_reference_values_out(reference, expected):
    """Given completed trials whose values at step 0 hold NaN, when MedianPruner(n_startup_trials=0) judges a
    trial that reported 2.0 at step 0 when minimising, then the NaN values are left out of the median, and a
    reference of only NaN values prunes nothing.

    Oracle: PercentilePruner docstring (numpy's nanpercentile gives the threshold) and _threshold (NaN when
    all are NaN, which prunes nothing). Hand count: without the NaN the median is 1.0 and 2.0 is worse.
    """
    study = _study("min", *reference)
    trial = _running(study, {0: 2.0})

    pruned = MedianPruner(n_startup_trials=0).prune(study, trial)

    assert pruned is expected


def test_percentile_pruner_should_not_prune_without_a_completed_trial_or_a_report():
    """Given a study without trials and a study of five completed trials, when
    MedianPruner(n_startup_trials=0) judges a trial with a report in the first and a trial without one in the
    second, then neither is pruned.

    Oracle: PercentilePruner (no completed trial: nothing to compare with) and RunningTrial.last_step (None
    before the first report, which prunes nothing).
    """
    empty = Study("min")
    filled = _study("min", *FIVE)
    reported = _running(empty, {0: 100.0})
    silent = RunningTrial(5, {"x": 0.5}, filled, MedianPruner(n_startup_trials=0))
    pruner = MedianPruner(n_startup_trials=0)

    answers = (pruner.prune(empty, reported), pruner.prune(filled, silent))

    assert answers == (False, False)


@pytest.mark.parametrize(
    "percentile",
    [
        pytest.param(-1.0, id="negative"),
        pytest.param(100.5, id="above-100"),
        pytest.param(NAN, id="nan"),
        pytest.param(INF, id="inf"),
        pytest.param(True, id="bool"),
        pytest.param("50", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_percentile_pruner_should_reject_a_percentile_outside_0_to_100(percentile):
    """Given a percentile that is not a number between 0 and 100, when PercentilePruner is built, then it
    raises ValueError naming percentile.

    Oracle: PercentilePruner (percentile between 0 and 100 inclusive, as in Optuna).
    """
    with pytest.raises(ValueError, match="percentile"):
        PercentilePruner(percentile)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("n_startup_trials", -1, id="startup-negative"),
        pytest.param("n_warmup_steps", -1, id="warmup-negative"),
        pytest.param("interval_steps", 0, id="interval-zero"),
        pytest.param("n_min_trials", 0, id="min-trials-zero"),
        pytest.param("n_startup_trials", 1.5, id="startup-fraction"),
        pytest.param("interval_steps", 2.0, id="interval-whole-float"),
        pytest.param("n_warmup_steps", True, id="warmup-bool"),
        pytest.param("n_min_trials", "2", id="min-trials-string"),
    ],
)
def test_median_pruner_should_reject_an_invalid_integer_setting(name, value):
    """Given an integer setting below its smallest value, a fraction, a whole float, a bool or a string, when
    MedianPruner is built with it, then it raises ValueError naming the setting.

    Oracle: PercentilePruner (n_startup_trials and n_warmup_steps >= 0, interval_steps and n_min_trials >= 1,
    as in Optuna) and the library's rule for integer arguments (_integer: Python or numpy integers only).
    """
    with pytest.raises(ValueError, match=name):
        MedianPruner(**{name: value})


def test_pruners_should_accept_the_boundary_settings_and_numpy_numbers():
    """Given the smallest valid settings, numpy integers and the percentiles 0 and 100, when the pruners are
    built, then their settings() hold them as Python numbers, MedianPruner without the fixed percentile.

    Oracle: Pruner.settings (the constructor parameters read from the attributes of the same name) and
    PercentilePruner (0 and 100 inclusive).
    """
    median = MedianPruner(np.int64(0), np.int32(0), 1, n_min_trials=np.int8(1))
    low, high = PercentilePruner(0), PercentilePruner(np.float64(100.0))

    found = (median.settings(), low.settings()["percentile"], high.settings()["percentile"])

    assert found == (
        {"n_startup_trials": 0, "n_warmup_steps": 0, "interval_steps": 1, "n_min_trials": 1},
        0.0,
        100.0,
    )
    assert type(median.settings()["n_startup_trials"]) is int


def test_percentile_pruner_settings_should_hold_every_constructor_parameter():
    """Given PercentilePruner(30.0, 2, 4, 3, n_min_trials=5), when settings() is read, then it holds the five
    constructor parameters with their values, as config.json records them.

    Oracle: Pruner.settings docstring.
    """
    pruner = PercentilePruner(30.0, 2, 4, 3, n_min_trials=5)

    found = pruner.settings()

    assert found == {
        "percentile": 30.0,
        "n_startup_trials": 2,
        "n_warmup_steps": 4,
        "interval_steps": 3,
        "n_min_trials": 5,
    }


def test_nop_pruner_should_never_prune_and_the_base_class_should_leave_prune_to_subclasses():
    """Given a trial whose every value is NaN next to five completed trials, when NopPruner and the Pruner
    base class are asked, then NopPruner answers False and the base class raises NotImplementedError.

    Oracle: NopPruner docstring (never prunes) and Pruner.prune (subclasses override it).
    """
    study = _study("min", *FIVE)
    trial = _running(study, {0: NAN})

    answer = NopPruner().prune(study, trial)

    assert answer is False
    with pytest.raises(NotImplementedError):
        Pruner().prune(study, trial)


def test_trial_pruned_should_be_an_exception():
    """Given TrialPruned, when its base classes are read, then it is an Exception, so an objective raises it
    and `except Exception` in user code would catch it, as Optuna's TrialPruned.

    Oracle: TrialPruned docstring (raised by the objective).
    """
    bases = TrialPruned.__mro__

    assert Exception in bases
    assert issubclass(TrialPruned, Exception)


def test_running_trial_should_keep_the_first_value_of_a_repeated_step_and_count_the_repeat():
    """Given a running trial, when it reports 0.5 at step 0, 0.25 at step 2 and 0.75 again at step 0, then the
    intermediate values are {0: 0.5, 2: 0.25}, the last step is 2 and the repeat of step 0 is noted.

    Oracle: RunningTrial.report docstring (a step reported before keeps its first value; the later ones are
    ignored and warned about) and RunningTrial.last_step (the largest step reported).
    """
    trial = RunningTrial(0, {"x": 0.5}, Study("min"), NopPruner())

    trial.report(0.5, 0)
    trial.report(0.25, 2)
    trial.report(0.75, 0)

    assert trial.intermediate_values == {0: 0.5, 2: 0.25}
    assert trial.last_step == 2
    assert trial._repeated == [0]


def test_running_trial_should_convert_values_with_float_and_accept_numpy_steps():
    """Given a running trial, when it reports a numpy float32, an int and a numpy integer step, then the
    values are stored as Python floats under Python int steps.

    Oracle: RunningTrial.report docstring (anything float() converts; a Python or numpy integer step >= 0).
    """
    trial = RunningTrial(0, {"x": 0.5}, Study("min"), NopPruner())

    trial.report(np.float32(0.5), np.int64(0))
    trial.report(3, 1)

    values = trial.intermediate_values
    assert values == {0: 0.5, 1: 3.0}
    assert [type(v) for v in values.values()] == [float, float]
    assert [type(s) for s in values] == [int, int]


@pytest.mark.parametrize(
    ("value", "step", "match"),
    [
        pytest.param("high", 0, "value", id="text-value"),
        pytest.param(None, 0, "value", id="none-value"),
        pytest.param(0.5, -1, "step", id="negative-step"),
        pytest.param(0.5, 1.0, "step", id="float-step"),
        pytest.param(0.5, True, "step", id="bool-step"),
        pytest.param(0.5, "1", "step", id="string-step"),
    ],
)
def test_running_trial_should_reject_a_value_or_step_it_cannot_record(value, step, match):
    """Given a running trial, when it reports a value float() does not convert or a step that is not an
    integer >= 0, then report() raises ValueError naming the value or the step and records nothing.

    Oracle: RunningTrial.report docstring.
    """
    trial = RunningTrial(0, {"x": 0.5}, Study("min"), NopPruner())

    with pytest.raises(ValueError, match=match):
        trial.report(value, step)
    assert trial.intermediate_values == {}


def test_running_trial_should_not_ask_the_pruner_before_a_report_or_for_a_protected_trial():
    """Given five completed trials and a pruner that always prunes, when should_prune() is asked by a trial
    without a report, by a protected trial that reported NaN and by an unprotected trial that reported NaN,
    then only the last one answers True.

    Oracle: RunningTrial.should_prune docstring (always False for a trial of the optimizer's initial
    population and before the first report; otherwise the run's pruner decides).
    """
    study = _study("min", *FIVE)
    always = PercentilePruner(100.0, n_startup_trials=0)
    silent = RunningTrial(5, {"x": 0.5}, study, always)
    protected = RunningTrial(5, {"x": 0.5}, study, always, protected=True)
    exposed = RunningTrial(5, {"x": 0.5}, study, always)
    protected.report(NAN, 0)
    exposed.report(NAN, 0)

    answers = (silent.should_prune(), protected.should_prune(), exposed.should_prune())

    assert answers == (False, False, True)


def test_running_trial_should_refuse_to_be_used_after_it_finished():
    """Given a running trial that reported once and was closed, when report() or should_prune() is called,
    then each raises RuntimeError naming the trial, and the recorded values stay.

    Oracle: RunningTrial docstring (the handle works only while the objective runs; afterwards report() and
    should_prune() raise RuntimeError).
    """
    trial = RunningTrial(7, {"x": 0.5}, Study("min"), NopPruner())
    trial.report(0.5, 0)
    trial._close()

    with pytest.raises(RuntimeError, match="trial 7"):
        trial.report(0.25, 1)
    with pytest.raises(RuntimeError, match="trial 7"):
        trial.should_prune()
    assert trial.intermediate_values == {0: 0.5}


def test_running_trial_should_hand_out_copies_of_its_params_and_values():
    """Given a running trial built from a params dict, when the dict passed in, the trial's params and the
    returned intermediate values are changed, then the trial's own copies stay as they were.

    Oracle: RunningTrial docstring (`params` a copy of the trial's parameters) and intermediate_values (a
    copy).
    """
    given = {"x": 0.5}
    trial = RunningTrial(0, given, Study("min"), NopPruner())
    trial.report(1.0, 0)

    given["x"] = 9.0
    trial.intermediate_values[0] = 9.0

    assert trial.params == {"x": 0.5}
    assert trial.intermediate_values == {0: 1.0}
