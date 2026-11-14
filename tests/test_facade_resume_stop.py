"""Integration tests of resume() with stopping rules: the rules recorded in config.json, or given as stop=.

Oracles: the resume docstring (the recorded rules apply again; built-in rules are built again from their
settings, a rule of another class needs stop=; stop= replaces the recorded rules for every run with the
checks of optimize(); the rules are asked after every trial, the restored ones included, and their answer
counts once no logged trial is left; a run a rule had ended comes back without a call and with the same
stop_reason; a rule that would have stopped earlier ends the run right after its logged trials; a run the
rules let go on continues as an uninterrupted run with those rules; MaxTime counts the search time over every
session; a folder written before stopping rules existed has none; nothing in the folder changes before the
arguments are checked), the optimize docstring (stop), the report module docstring (config.json records
`stop` and every resume with its rules) and RunState.elapsed (search time leaves out the time a run was
stopped).

Every folder is written below pytest's tmp_path and the plot functions are replaced by stubs (fixture
no_plots). The uninterrupted runs that give the expected trials use the same seed and rules. The clock
tests replace the time module that metahpo.report and metahpo.facade read with a fake clock that the
objective moves ten minutes per evaluation, and move it ten hours between sessions, so no test sleeps.

Unspecified, therefore not asserted: wall-clock values written to the files, the wording of the error
messages beyond the named rule or argument, and the console lines.

Input space: Every results folder (rules, runs, seeds, point of stop) and every stop= argument of resume(),
which is unbounded.
Subset: The default ABC optimizer on a two-parameter sphere with a budget of 200; Patience, TargetScore and
MaxTime, alone and together, and a rule of one's own; folders ended by a rule, interrupted by Ctrl+C inside
the first batch and later, and with three runs interrupted in the second; replacing rules by a stricter, a
more generous and no rule; a folder without the `stop` entry; stop= values of the wrong type and with a
reserved reason.
"""

import json
import math

import pytest

import metahpo
from metahpo import (
    Callback,
    Float,
    MaxTime,
    Patience,
    Space,
    Stopper,
    TargetScore,
    optimize,
    resume,
)

ANY_SEED = 0

BUDGET = 200

TEN_MINUTES = 600.0

TEN_HOURS = 36_000.0


@pytest.fixture
def no_plots(monkeypatch):
    """Replaces make_plots and make_run_plots with stubs that draw nothing."""
    monkeypatch.setattr("metahpo.plots.make_plots", _no_plots)
    monkeypatch.setattr("metahpo.plots.make_run_plots", _no_plots)


def _no_plots(*args, **kwargs):
    return []


class _Clock:
    """A fake time module: time(), perf_counter() and monotonic() all read `now`, which only moves when a
    test or an objective moves it."""

    def __init__(self):
        self.now = 1_800_000_000.0

    def time(self):
        return self.now

    def perf_counter(self):
        return self.now

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    """A fake clock in place of the time module of metahpo.report and metahpo.facade."""
    fake = _Clock()
    monkeypatch.setattr(metahpo.report, "time", fake)
    monkeypatch.setattr(metahpo.facade, "time", fake)
    return fake


def _space():
    """A fresh two-parameter Float space."""
    return Space({"x": Float(-5.0, 5.0), "y": Float(-5.0, 5.0)})


def _sphere(params):
    """The sphere function, smallest at x = y = 0."""
    return params["x"] ** 2 + params["y"] ** 2


def _counting(function=_sphere):
    """An objective that counts its calls in the returned list."""
    calls = [0]

    def objective(params):
        calls[0] += 1
        return function(params)

    return objective, calls


def _interrupting(after, function=_sphere):
    """An objective that raises KeyboardInterrupt on call `after` + 1."""
    calls = [0]

    def objective(params):
        calls[0] += 1
        if calls[0] > after:
            raise KeyboardInterrupt
        return function(params)

    return objective


def _scripted(scores, after):
    """An objective that returns scores[n] on call n + 1 and raises KeyboardInterrupt on call `after` + 1."""
    calls = [0]

    def objective(params):
        calls[0] += 1
        if calls[0] > after:
            raise KeyboardInterrupt
        return scores[calls[0] - 1]

    return objective


def _slow(clock, after=None):
    """The sphere as an objective that takes ten minutes on the fake clock; with `after` it raises
    KeyboardInterrupt on call `after` + 1, before the clock moves."""
    calls = [0]

    def objective(params):
        calls[0] += 1
        if after is not None and calls[0] > after:
            raise KeyboardInterrupt
        clock.now += TEN_MINUTES
        return _sphere(params)

    return objective


def _refusing(params):
    raise AssertionError("the objective must not be called")


def _folder(root):
    """The only results folder below root."""
    folders = [p for p in root.iterdir() if p.is_dir()]
    assert len(folders) == 1
    return folders[0]


def _start(root, function, stop=None, runs=1, seed=ANY_SEED):
    """optimize() minimising `function` with the given rules, written below root; returns the study (or
    studies) and the results folder."""
    study = optimize(
        function, _space(), max_evals=BUDGET, seed=seed, verbose=0, output=root, stop=stop, runs=runs
    )
    return study, _folder(root)


def _reference(stop=None, runs=1, seed=ANY_SEED, function=_sphere):
    """The uninterrupted run with the given rules, without a results folder."""
    return optimize(
        function, _space(), max_evals=BUDGET, seed=seed, verbose=0, output=None, stop=stop, runs=runs
    )


def _trajectory(study):
    """What identifies a search: every trial's number, params, exact score, cached flag, meta and x, and the
    stop reason."""
    trials = [(t.number, t.params, repr(t.score), t.cached, t.meta, t.x.tolist()) for t in study.trials]
    return trials, study.stop_reason


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _records(folder, run=1):
    """The lines of trials.jsonl without their wall-clock fields."""
    lines = (folder / f"run_{run:03d}" / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    clock_fields = ("start_epoch", "duration_s", "start_time", "end_time", "elapsed_s")
    return [{k: v for k, v in json.loads(line).items() if k not in clock_fields} for line in lines]


def _files(folder):
    """Every file below the folder with its bytes."""
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()
    }


class _Close(Stopper):
    """A rule of one's own: stops when the last trial lies within `radius` of the origin of the sphere."""

    reason = "close"

    def __init__(self, radius):
        self.radius = radius

    def should_stop(self, state):
        return state.study.trials[-1].score < self.radius**2


class _Reason(Stopper):
    """A rule whose reason is given; it never says stop."""

    def __init__(self, reason):
        self.reason = reason

    def should_stop(self, state):
        return False


class _Elapsed(Callback):
    """Records state.elapsed() after every trial that resume() restores."""

    def __init__(self):
        self.restored = []

    def on_trial(self, state, trial, new_best):
        if state.replaying:
            self.restored.append(state.elapsed())


@pytest.mark.parametrize(
    "rules",
    [
        pytest.param([Patience(15)], id="patience"),
        pytest.param([TargetScore(0.5)], id="target-score"),
        pytest.param([Patience(40, min_delta=0.25), TargetScore(0.05)], id="two-rules"),
    ],
)
def test_resume_should_return_a_folder_a_rule_ended_without_a_call_and_with_its_reason(
    tmp_path, no_plots, rules
):
    """Given a folder whose run a built-in rule ended, when resume() runs without stop=, then it returns the
    same trials and stop_reason without calling the objective, and the resume records the rules that
    config.json holds, built again from their settings.

    Oracle: resume docstring (the recorded rules apply again, built-in rules built again from their
    settings; a run a rule had ended comes back without a call and with the same stop_reason) and the report
    module docstring (config.json records `stop`; every resume records its rules).
    """
    original, folder = _start(tmp_path, _sphere, rules)
    recorded = _json(folder / "config.json")["stop"]

    study = resume(_refusing, folder, verbose=0)

    config = _json(folder / "config.json")
    assert _trajectory(study) == _trajectory(original)
    assert study.stop_reason in ("patience", "target")
    assert config["stop"] == recorded == [{"name": type(r).__name__, "settings": r.settings()} for r in rules]
    assert config["resumes"][-1]["stop"] == recorded


@pytest.mark.parametrize(
    ("rules", "after"),
    [
        pytest.param([Patience(15)], 3, id="patience-inside-the-first-batch"),
        pytest.param([Patience(15)], 12, id="patience-later"),
        pytest.param([TargetScore(0.5)], 20, id="target-score"),
        pytest.param([Patience(40, min_delta=0.25), TargetScore(0.05)], 30, id="two-rules"),
    ],
)
def test_resume_should_continue_a_run_interrupted_before_its_rule_fired_like_an_uninterrupted_run(
    tmp_path, no_plots, rules, after
):
    """Given a run with rules interrupted by Ctrl+C before any rule said stop, when resume() continues it,
    then the study, trials.jsonl and summary equal those of an uninterrupted run with the same rules, which a
    rule ended.

    Oracle: resume docstring (a run whose rules let it go on continues as an uninterrupted run with those
    rules would have; the rules see the restored trials, so their state is rebuilt).
    """
    _, folder = _start(tmp_path / "stopped", _interrupting(after), rules)
    _, whole = _start(tmp_path / "whole", _sphere, rules)
    reference = _reference(rules)

    study = resume(_sphere, folder, verbose=0)

    summary, expected = _json(folder / "summary.json"), _json(whole / "summary.json")
    assert _trajectory(study) == _trajectory(reference)
    assert _records(folder) == _records(whole)
    assert (summary["stop_reason"], summary["calls"]) == (expected["stop_reason"], expected["calls"])
    assert reference.stop_reason != "budget"


@pytest.mark.parametrize(
    "scores",
    [
        pytest.param([5.0, 6.0, 7.0, 8.0, 1.0], id="improves-at-the-last-logged-trial"),
        pytest.param([5.0, 6.0, 7.0, 8.0, 9.0], id="still-stopping-at-the-end"),
    ],
)
def test_resume_should_end_a_run_right_after_the_restore_when_a_given_rule_would_have_stopped_it_earlier(
    tmp_path, no_plots, scores
):
    """Given a run without rules interrupted after five calls with scores 5, 6, 7, 8 and then 1 or 9, when
    resume() continues it with stop=Patience(2), which would have stopped it at the third call, then all
    five logged trials come back, the objective is not called and the stop_reason is 'patience', also when
    the last logged trial is an improvement after which Patience alone would go on.

    Oracle: resume docstring (a rule's answer counts once no logged trial is left; a run with a rule that
    would have stopped it earlier ends right after its logged trials, which are never discarded).
    """
    _, folder = _start(tmp_path, _scripted(scores, 5))

    study = resume(_refusing, folder, verbose=0, stop=Patience(2))

    assert (len(study.trials), study.stop_reason) == (5, "patience")
    assert [t.score for t in study.trials] == scores


def test_resume_should_replace_the_recorded_rules_by_a_more_generous_rule_given_as_stop(tmp_path, no_plots):
    """Given a folder that Patience(10) ended, when resume() runs with stop=Patience(40), then the run goes
    on and ends like an uninterrupted run with Patience(40), and config.json and the resume record the new
    rule.

    Oracle: resume docstring (stop= replaces the recorded rules; config.json then records the new rules; a
    run whose rules let it go on continues as an uninterrupted run with those rules).
    """
    _, folder = _start(tmp_path / "stopped", _sphere, [Patience(10)])
    _, whole = _start(tmp_path / "whole", _sphere, [Patience(40)])
    function, calls = _counting()

    study = resume(function, folder, verbose=0, stop=Patience(40))

    config = _json(folder / "config.json")
    expected = [{"name": "Patience", "settings": {"evals": 40, "min_delta": 0.0}}]
    assert _trajectory(study) == _trajectory(_reference([Patience(40)]))
    assert _records(folder) == _records(whole)
    assert (config["stop"], config["resumes"][-1]["stop"]) == (expected, expected)
    assert calls[0] == study.n_calls - 25


def test_resume_should_run_to_the_budget_with_an_empty_stop(tmp_path, no_plots):
    """Given a folder that Patience(10) ended after 25 calls, when resume() runs with stop=[], then the run
    goes on to the budget like an uninterrupted run without rules, and config.json records no rules.

    Oracle: resume docstring (an empty sequence means no rules; config.json then records the new rules).
    """
    _, folder = _start(tmp_path, _sphere, [Patience(10)])

    study = resume(_sphere, folder, verbose=0, stop=[])

    assert _trajectory(study) == _trajectory(_reference())
    assert (study.n_calls, study.stop_reason) == (BUDGET, "budget")
    assert _json(folder / "config.json")["stop"] == []


def test_resume_should_ask_for_stop_when_the_folder_records_a_rule_of_ones_own(tmp_path, no_plots):
    """Given a folder ended by a rule of one's own, when resume() runs without stop=, then ValueError names
    the rule's class and stop=, and the folder stays byte for byte as it was; with the rule given as stop=
    the run comes back without a call.

    Oracle: resume docstring (a folder written with a rule of another class needs stop=, else ValueError;
    nothing in the folder changes before the arguments are checked).
    """
    original, folder = _start(tmp_path, _sphere, [_Close(0.5)])
    before = _files(folder)

    with pytest.raises(ValueError, match="_Close.*stop="):
        resume(_refusing, folder, verbose=0)
    after = _files(folder)
    study = resume(_refusing, folder, verbose=0, stop=_Close(0.5))

    assert after == before
    assert _trajectory(study) == _trajectory(original)
    assert study.stop_reason == "close"


@pytest.mark.parametrize(
    ("stop", "error"),
    [
        pytest.param(5, TypeError, id="number"),
        pytest.param("patience", TypeError, id="string"),
        pytest.param([Patience(3), None], TypeError, id="list-with-none"),
        pytest.param([_Reason("budget")], ValueError, id="reserved-reason"),
        pytest.param(_Reason(""), ValueError, id="empty-reason"),
    ],
)
def test_resume_should_reject_a_bad_stop_and_leave_the_folder_unchanged(tmp_path, no_plots, stop, error):
    """Given an interrupted folder, when resume() is called with a stop= that optimize() would reject, then
    it raises the same kind of error and every file of the folder stays byte for byte as it was.

    Oracle: resume docstring (stop= is checked as in optimize(); nothing in the folder changes before the
    arguments are checked) and the optimize docstring (TypeError for anything but a Stopper, ValueError for
    an empty or reserved reason).
    """
    _, folder = _start(tmp_path, _interrupting(10), [Patience(15)])
    before = _files(folder)

    with pytest.raises(error):
        resume(_refusing, folder, verbose=0, stop=stop)

    assert _files(folder) == before


def test_resume_should_restore_finished_runs_and_continue_the_interrupted_one_with_their_rules(
    tmp_path, no_plots
):
    """Given three runs with Patience(15) interrupted on the sixth call of the second run, when resume()
    continues the folder, then every run equals its uninterrupted twin, the first comes back without a call
    and the objective is called only for the rest of the second run and for the third.

    Oracle: resume docstring (finished runs are restored, the unfinished run continues, runs that never
    started run; the rules see every restored trial) and the optimize docstring (every run gets its own copy
    of the rules).
    """
    reference = _reference([Patience(15)], runs=3, seed=[1, 2, 3])
    first = reference[0].n_calls
    _, folder = _start(tmp_path, _interrupting(first + 5), [Patience(15)], runs=3, seed=[1, 2, 3])
    function, calls = _counting()

    studies = resume(function, folder, verbose=0)

    assert [_trajectory(s) for s in studies] == [_trajectory(s) for s in reference]
    assert [s.stop_reason for s in studies] == ["patience"] * 3
    assert calls[0] == reference[1].n_calls - 5 + reference[2].n_calls


def test_resume_should_treat_a_folder_without_a_stop_entry_as_one_without_rules(tmp_path, no_plots):
    """Given an interrupted folder whose config.json has no `stop` entry, as folders written before stopping
    rules existed, when resume() continues it, then the run goes on to the budget like an uninterrupted run
    without rules and config.json records an empty `stop`.

    Oracle: resume docstring (a folder written before stopping rules existed has none) and the report
    module docstring (a folder without `stop` counts as no rules).
    """
    _, folder = _start(tmp_path, _interrupting(30))
    config = _json(folder / "config.json")
    config.pop("stop")
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")

    study = resume(_sphere, folder, verbose=0)

    assert _trajectory(study) == _trajectory(_reference())
    assert _json(folder / "config.json")["stop"] == []


def test_resume_should_keep_a_run_that_max_time_ended_finished_and_give_it_more_time_with_a_longer_limit(
    tmp_path, no_plots, clock
):
    """Given a run that MaxTime(hours=1) ended after six ten-minute evaluations and a clock moved ten hours
    before each resume, when resume() runs without stop= and then with stop=MaxTime(hours=2), then the first
    returns the six trials without a call and the second goes on for exactly six more evaluations, like an
    uninterrupted run with MaxTime(hours=2).

    Oracle: resume docstring (MaxTime counts the search time over every session, the time the run was
    stopped left out; a longer MaxTime gives the run more time) and RunState.elapsed. Were the ten hours
    counted, the second resume would end right after the restore.
    """
    _, folder = _start(tmp_path, _slow(clock), [MaxTime(hours=1)])
    clock.now += TEN_HOURS

    finished = resume(_refusing, folder, verbose=0)
    clock.now += TEN_HOURS
    longer = resume(_slow(clock), folder, verbose=0, stop=MaxTime(hours=2))

    reference = _reference([MaxTime(hours=2)], function=_slow(clock))
    assert (len(finished.trials), finished.stop_reason) == (6, "time")
    assert (len(longer.trials), longer.stop_reason) == (12, "time")
    assert _trajectory(longer) == _trajectory(reference)


def test_resume_should_count_the_logged_search_time_at_the_restored_trials_not_the_wall_clock(
    tmp_path, no_plots, clock
):
    """Given a run with MaxTime(hours=1) interrupted on its fourth ten-minute evaluation and a clock moved ten
    hours, when resume() continues it, then state.elapsed() at the three restored trials reads 600, 1200 and
    1800 seconds and the run ends after six trials with stop_reason 'time'.

    Oracle: RunState.elapsed (search seconds: the time since `started` minus the time the run was stopped)
    and the RunState docstring (while restored trials are replayed, now() is the end of the last of them);
    resume docstring (MaxTime counts the search time over every session). The interrupted fourth call
    moved no time, so the three restored trials end at 600, 1200 and 1800 seconds.
    """
    _, folder = _start(tmp_path, _slow(clock, after=3), [MaxTime(hours=1)])
    clock.now += TEN_HOURS
    recorder = _Elapsed()

    study = resume(_slow(clock), folder, verbose=0, callbacks=[recorder])

    assert recorder.restored == [600.0, 1200.0, 1800.0]
    assert (len(study.trials), study.stop_reason) == (6, "time")
    assert math.isclose(study.trials[-1].start_time - study.trials[0].start_time, TEN_HOURS + 5 * TEN_MINUTES)
