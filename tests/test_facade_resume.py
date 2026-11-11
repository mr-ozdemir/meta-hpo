"""Integration tests of resume(): a search that stopped continues from its results folder.

Oracles: the resume docstring (everything comes from the folder; the search goes on as if it had never
stopped, also without a seed; the trial that was running is evaluated again; finished runs are restored
without a call, the unfinished run is continued and runs that never started are run with their seeds; a
folder whose runs all finished evaluates nothing; max_evals gives the result of a run started with that
budget and must cover the logged calls; optimizer= for another optimizer and space= for choices JSON cannot
hold, both checked against the folder; a candidate that differs from the log raises RuntimeError before the
folder changes; a line cut short by a kill is dropped; the other files are written again; 0.5.x folders),
the RunState docstring (restored, replaying, pauses), the Callback docstring (restored events are sent with
replaying True), the ConsoleReporter docstring (one 'resumed' line) and the report module docstring
(trials.jsonl is the record of a run, every other file is derived; time leaves out the stops).

Every folder is written below pytest's tmp_path. The plot functions are replaced by stubs that draw nothing
(fixture no_plots), so the runs stay fast; the plots are tested in test_plots.py. A kill is simulated by
cutting trials.jsonl after a number of lines and leaving the start of the next line, as a process killed
in the middle of a write does; a power cut that a machine was off for is simulated by moving the logged
times into the past.

Unspecified, therefore not asserted: wall-clock values (start and end times, durations), the exact wording
of the error messages beyond the named trial or argument, and the console lines of the live part beyond
their number.

Input space: Every results folder (any optimizer, space, budget, runs, seeds, scores and point of stop),
every objective and every argument of resume(), which is unbounded.
Subset: The three optimizers with small swarms on a four-parameter mixed space and 40 evaluations; stops by
KeyboardInterrupt and by a simulated kill at the first call, inside the initial batch, inside a PSO flight
batch and later; one run without a seed; a finished folder; a budget extension that cuts a PSO batch; three
runs stopped in the second; a custom optimizer; a Categorical of functions; folders edited to fail the
check (other settings, a removed line, a broken line, a trial after the end); a second stop; a 0.5.1 folder
with and without a seed; a state.json left empty or cut short, with and without a seed; a budget that
crosses COMPRESS_FROM; scores NaN, inf and -inf.
"""

import csv
import gzip
import json
import math

import pytest

import metahpo
from metahpo import (
    ABCOptimizer,
    BBPSOOptimizer,
    Callback,
    Categorical,
    Float,
    Int,
    Optimizer,
    PSOOptimizer,
    Space,
    Study,
    optimize,
    resume,
)

ANY_SEED = 0

BUDGET = 40

MAKERS = [
    pytest.param(lambda s: ABCOptimizer(s, direction="max", colony_size=8, seed=ANY_SEED), id="abc"),
    pytest.param(lambda s: PSOOptimizer(s, direction="max", swarm_size=7, seed=ANY_SEED), id="pso"),
    pytest.param(lambda s: BBPSOOptimizer(s, direction="max", swarm_size=6, seed=ANY_SEED), id="bbpso"),
]

TIME_COLUMNS = {"start_time", "end_time", "duration_s", "elapsed_s", "time"}


@pytest.fixture
def no_plots(monkeypatch):
    """Replaces make_plots and make_run_plots with stubs that draw nothing."""
    monkeypatch.setattr("metahpo.plots.make_plots", _no_plots)
    monkeypatch.setattr("metahpo.plots.make_run_plots", _no_plots)


def _no_plots(*args, **kwargs):
    return []


def _space():
    """A fresh mixed space with a log Float, an Int, a Categorical of strings and a linear Float."""
    return Space(
        {
            "lr": Float(1e-4, 1e-1, log=True),
            "units": Int(16, 256),
            "act": Categorical(["relu", "tanh", "gelu"]),
            "dropout": Float(0.0, 0.5),
        }
    )


def _score(params):
    """An accuracy-like score that is NaN, inf or -inf for some unit counts."""
    special = {0: math.nan, 1: math.inf, 2: -math.inf}
    act = {"relu": 0.01, "tanh": 0.02, "gelu": 0.0}[params["act"]]
    lr = 0.02 * (math.log10(params["lr"]) + 3.0) ** 2
    value = 0.95 - lr - ((params["units"] - 128) / 256) ** 2 - (params["dropout"] - 0.2) ** 2 - act
    return special.get(params["units"] % 9, value)


def _counting(function=_score):
    """An objective that counts its calls in the returned list."""
    calls = [0]

    def objective(params):
        calls[0] += 1
        return function(params)

    return objective, calls


def _interrupting(after, function=_score):
    """An objective that raises KeyboardInterrupt on call `after` + 1."""
    calls = [0]

    def objective(params):
        calls[0] += 1
        if calls[0] > after:
            raise KeyboardInterrupt
        return function(params)

    return objective


def _refusing(params):
    raise AssertionError("the objective must not be called")


def _folder(root):
    """The only results folder below root."""
    folders = [p for p in root.iterdir() if p.is_dir()]
    assert len(folders) == 1
    return folders[0]


def _start(root, make, function, max_evals=BUDGET):
    """optimize() maximising `function` with the optimizer `make` builds, written below root; returns the
    study and the results folder."""
    space = _space()
    study = optimize(function, space, "max", max_evals, optimizer=make(space), verbose=0, output=root)
    return study, _folder(root)


def _reference(make, max_evals=BUDGET, function=_score):
    """The uninterrupted run of the optimizer `make` builds, without a results folder."""
    space = _space()
    return optimize(function, space, "max", max_evals, optimizer=make(space), verbose=0, output=None)


def _trajectory(study):
    """What identifies a search: every trial's number, params, exact score, cached flag, meta and x."""
    return [(t.number, t.params, repr(t.score), t.cached, t.meta, t.x.tolist()) for t in study.trials]


def _log(folder, run=1):
    return folder / f"run_{run:03d}" / "trials.jsonl"


def _kill(folder, keep, run=1):
    """Cuts trials.jsonl after `keep` lines and leaves the first 25 characters of the next line."""
    lines = _log(folder, run).read_text(encoding="utf-8").splitlines(keepends=True)
    _log(folder, run).write_text("".join(lines[:keep]) + lines[keep][:25], encoding="utf-8", newline="")
    return "".join(lines[:keep])


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path):
    """The rows of a CSV file (or .csv.gz) as dictionaries."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _timeless(rows):
    """The rows without their wall-clock columns."""
    return [{k: v for k, v in row.items() if k not in TIME_COLUMNS} for row in rows]


def _derived(folder):
    """The derived files of run_001 without wall-clock values."""
    run_dir = folder / "run_001"
    best = _json(run_dir / "best.json")
    best["best"] = {k: v for k, v in best["best"].items() if k not in ("found_at", "found_after_s")}
    names = ("trials", "generations", "population", "parameters", "categories", "best_history")
    return {
        "best": best,
        **{name: _timeless(_rows(run_dir / f"{name}.csv")) for name in names},
    }


def _summary_counts(folder):
    summary = _json(folder / "summary.json")
    keys = ("stop_reason", "calls", "trials", "cached_trials", "failed_trials", "generations", "max_evals")
    return {k: summary[k] for k in keys}


def _shift_into_the_past(folder, seconds):
    """Moves the start of run_001 and every logged start time `seconds` into the past, as if the machine had
    been off for that long since the run stopped."""
    state_path = folder / "run_001" / "state.json"
    state = _json(state_path)
    state["started"] -= seconds
    state_path.write_text(json.dumps(state), encoding="utf-8")
    records = [json.loads(line) for line in _log(folder).read_text(encoding="utf-8").splitlines()]
    moved = [{**r, "start_epoch": r["start_epoch"] - seconds} for r in records]
    _log(folder).write_text("".join(json.dumps(r) + "\n" for r in moved), encoding="utf-8", newline="")


def _make_legacy(folder):
    """Turns a folder into what meta-hpo 0.5.1 wrote: no state.json, no start_epoch, null for a non-finite
    score, Categorical choices as repr() text, no cache flag, no resumes and no colony_size setting."""
    (folder / "run_001" / "state.json").unlink()
    records = [json.loads(line) for line in _log(folder).read_text(encoding="utf-8").splitlines()]
    old = [_legacy_record(r) for r in records]
    _log(folder).write_text("".join(json.dumps(r) + "\n" for r in old), encoding="utf-8", newline="")
    config = _json(folder / "config.json")
    config["space"] = [_legacy_parameter(p) for p in config["space"]]
    config["optimizer"]["settings"].pop("colony_size")
    config.pop("cache")
    config.pop("resumes")
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _legacy_record(record):
    record = {k: v for k, v in record.items() if k != "start_epoch"}
    record["score"] = record["score"] if isinstance(record["score"], float) else None
    return record


def _legacy_parameter(parameter):
    parameter = dict(parameter)
    if parameter.pop("repr", None) is False:
        parameter["choices"] = [repr(c) for c in parameter["choices"]]
    return parameter


class _Replays(Callback):
    """Records state.replaying at every trial event."""

    def __init__(self):
        self.flags = []

    def on_trial(self, state, trial, new_best):
        self.flags.append(state.replaying)


class _RandomSearch(Optimizer):
    """A custom optimizer: one random candidate per ask."""

    def ask(self):
        return self._fresh(1)

    def tell(self, candidates, scores):
        for cand, score in zip(candidates, scores, strict=True):
            f = self._internal(float(score))
            if f < self.best_f or self.best_x is None:
                self.best_f, self.best_x = f, cand.x.copy()


@pytest.mark.parametrize("make", MAKERS)
def test_resume_should_continue_an_interrupted_run_exactly_like_an_uninterrupted_run(
    tmp_path, no_plots, make
):
    """Given each optimizer stopped by KeyboardInterrupt on its 10th call (inside PSO's second batch), when
    resume() continues the folder, then the study equals the uninterrupted run trial by trial and the
    objective is called once for every trial that was not logged, the interrupted one included.

    Oracle: resume docstring (the search goes on as if it had never stopped; the trial that was running is
    evaluated again).
    """
    _, folder = _start(tmp_path, make, _interrupting(9))
    objective, calls = _counting()

    study = resume(objective, folder, verbose=0)

    reference = _reference(make)
    assert _trajectory(study) == _trajectory(reference)
    assert calls[0] == reference.n_calls - 9


@pytest.mark.parametrize(
    ("make", "keep"),
    [
        pytest.param(MAKERS[0].values[0], 0, id="abc-first-call"),
        pytest.param(MAKERS[0].values[0], 2, id="abc-inside-the-initial-batch"),
        pytest.param(MAKERS[1].values[0], 10, id="pso-inside-a-flight-batch"),
        pytest.param(MAKERS[2].values[0], 4, id="bbpso-inside-the-initial-batch"),
        pytest.param(MAKERS[2].values[0], 23, id="bbpso-later"),
    ],
)
def test_resume_should_continue_a_killed_run_whose_last_log_line_is_torn(tmp_path, no_plots, make, keep):
    """Given a folder whose trials.jsonl was cut after `keep` lines in the middle of the next line, as a kill
    leaves it, when resume() continues it, then the study equals the uninterrupted run, the torn line is
    gone and the kept lines are still the start of the log.

    Oracle: resume docstring (a killed process continues exactly where it stopped; the line a kill cut short
    is dropped) and report module docstring (resume() never rewrites a logged line).
    """
    reference, folder = _start(tmp_path, make, _score)
    kept = _kill(folder, keep)

    study = resume(_score, folder, verbose=0)

    log = _log(folder).read_text(encoding="utf-8")
    assert _trajectory(study) == _trajectory(reference)
    assert log.startswith(kept)
    assert len(log.splitlines()) == len(reference.trials)


def test_resume_should_continue_a_run_without_a_seed_exactly(tmp_path, no_plots):
    """Given a default run started with seed=None and interrupted on its 12th call, when resume() continues
    it, then the study equals an uninterrupted run whose generator starts in the state that state.json
    records.

    Oracle: resume docstring (the search goes on as if it had never stopped, also for a run without a seed)
    and report module docstring (state.json holds the generator state before the first candidate).
    """
    space = _space()
    optimize(_interrupting(11), space, "max", BUDGET, seed=None, verbose=0, output=tmp_path)
    folder = _folder(tmp_path)
    twin_space = _space()
    twin = ABCOptimizer(twin_space, direction="max", seed=ANY_SEED)
    twin.rng.bit_generator.state = _json(folder / "run_001" / "state.json")["generator"]

    study = resume(_score, folder, verbose=0)

    reference = optimize(_score, twin_space, "max", BUDGET, optimizer=twin, verbose=0, output=None)
    assert _trajectory(study) == _trajectory(reference)


def test_resume_should_return_the_study_of_a_finished_folder_without_calling_the_function(
    tmp_path, no_plots, capsys, monkeypatch
):
    """Given a run that finished by its budget, when resume() is called with an objective that fails when
    called, then it returns the same study without a call and prints the header, the 'resumed' line and the
    final line.

    Oracle: resume docstring (a folder whose runs all finished evaluates nothing) and ConsoleReporter
    docstring (one 'resumed' line).
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)
    original, folder = _start(tmp_path, MAKERS[1].values[0], _score)

    study = resume(_refusing, folder)

    lines = capsys.readouterr().out.splitlines()
    assert (_trajectory(study), study.stop_reason) == (_trajectory(original), "budget")
    assert len(lines) == 3
    assert lines[1].startswith(f"[meta-hpo] run 1 | resumed | {len(original.trials)} trials restored")


def test_resume_should_extend_the_budget_like_a_run_started_with_the_larger_budget(tmp_path, no_plots):
    """Given a PSO run (swarm 7) that finished at max_evals 30, which cut a flight batch, when resume() is
    called with max_evals 50, then the study equals a run started with 50 and config.json records 50.

    Oracle: resume docstring (max_evals gives the result of a run started with that budget).
    """
    make = MAKERS[1].values[0]
    _, folder = _start(tmp_path, make, _score, max_evals=30)

    study = resume(_score, folder, max_evals=50, verbose=0)

    assert _trajectory(study) == _trajectory(_reference(make, max_evals=50))
    assert _json(folder / "config.json")["max_evals"] == 50


@pytest.mark.parametrize(
    "max_evals",
    [
        pytest.param(19, id="below-the-logged-calls"),
        pytest.param(0, id="zero"),
        pytest.param(True, id="bool"),
        pytest.param(2.5, id="float"),
    ],
)
def test_resume_should_reject_a_budget_below_the_logged_calls_or_not_a_positive_integer(
    tmp_path, no_plots, max_evals
):
    """Given a folder that logged 20 calls, when resume() gets max_evals 19, 0, True or 2.5, then ValueError
    names max_evals and the log is unchanged.

    Oracle: resume docstring (max_evals must be at least the number of calls a run already logged).
    """
    _, folder = _start(tmp_path, MAKERS[0].values[0], _interrupting(20))
    before = _log(folder).read_bytes()

    with pytest.raises(ValueError, match="max_evals"):
        resume(_score, folder, max_evals=max_evals, verbose=0)

    assert _log(folder).read_bytes() == before


def test_resume_should_restore_finished_runs_continue_the_unfinished_one_and_start_the_rest(
    tmp_path, no_plots
):
    """Given three runs with seeds [1, 2, 3] interrupted on the 6th call of the second run, when resume()
    continues the folder, then it returns three studies equal to the uninterrupted runs and calls the
    objective only for the trials that were not logged.

    Oracle: resume docstring (finished runs are restored without a call, the unfinished run is continued and
    runs that never started are run with the seeds config.json records; a list of Study for several runs).
    """
    optimize(
        _interrupting(BUDGET + 5), _space(), "max", BUDGET, seed=[1, 2, 3], runs=3, verbose=0, output=tmp_path
    )
    objective, calls = _counting()

    studies = resume(objective, _folder(tmp_path), verbose=0)

    reference = optimize(_score, _space(), "max", BUDGET, seed=[1, 2, 3], runs=3, verbose=0, output=None)
    assert [_trajectory(s) for s in studies] == [_trajectory(s) for s in reference]
    assert calls[0] == 3 * BUDGET - (BUDGET + 5)


def test_resume_should_require_the_optimizer_of_a_folder_written_by_a_custom_optimizer(tmp_path, no_plots):
    """Given a folder written by a custom optimizer, when resume() is called without optimizer=, then
    ValueError asks for optimizer=.

    Oracle: resume docstring (a folder of another optimizer without optimizer= raises ValueError).
    """
    space = _space()
    optimize(
        _interrupting(7), space, "max", 20, optimizer=_RandomSearch(space, "max", ANY_SEED), output=tmp_path
    )

    with pytest.raises(ValueError, match="optimizer="):
        resume(_score, _folder(tmp_path), verbose=0)

    assert len(_log(_folder(tmp_path)).read_text(encoding="utf-8").splitlines()) == 7


def test_resume_should_continue_a_custom_optimizer_passed_as_a_function(tmp_path, no_plots):
    """Given a folder written by a custom optimizer, when resume() gets a function seed -> optimizer, then the
    study equals the uninterrupted run of that optimizer.

    Oracle: resume docstring (optimizer is an Optimizer object or a function seed -> optimizer, as in
    optimize()).
    """
    space = _space()
    optimize(
        _interrupting(7), space, "max", 20, optimizer=_RandomSearch(space, "max", ANY_SEED), output=tmp_path
    )
    again = _space()

    study = resume(
        _score, _folder(tmp_path), optimizer=lambda seed: _RandomSearch(again, "max", seed), verbose=0
    )

    twin = _space()
    reference = optimize(_score, twin, "max", 20, optimizer=_RandomSearch(twin, "max", ANY_SEED), output=None)
    assert _trajectory(study) == _trajectory(reference)


def _function_space(high=1.0):
    """A space whose Categorical holds functions, which JSON cannot hold."""
    return Space({"agg": Categorical([min, max]), "x": Float(0.0, high)})


def _aggregate(params):
    return params["agg"](params["x"], 0.5)


def _start_function_space(root):
    space = _function_space()
    optimize(_interrupting(6, _aggregate), space, "max", 20, seed=ANY_SEED, verbose=0, output=root)
    return _folder(root)


def test_resume_should_require_the_space_when_choices_are_not_json_values(tmp_path, no_plots):
    """Given a folder whose space has a Categorical of the functions min and max, when resume() is called
    without space=, then ValueError asks for space=.

    Oracle: resume docstring (space is needed when a Categorical has choices that JSON cannot hold) and
    report module docstring (such a Categorical is marked "repr": true).
    """
    folder = _start_function_space(tmp_path)

    with pytest.raises(ValueError, match="space="):
        resume(_aggregate, folder, verbose=0)

    assert _json(folder / "config.json")["space"][0]["repr"] is True


def test_resume_should_continue_with_the_passed_space_of_functions(tmp_path, no_plots):
    """Given the same folder, when resume() gets a space built the same way, then the study equals the
    uninterrupted run.

    Oracle: resume docstring (space must be the space that config.json describes).
    """
    folder = _start_function_space(tmp_path)

    study = resume(_aggregate, folder, space=_function_space(), verbose=0)

    reference = optimize(_aggregate, _function_space(), "max", 20, seed=ANY_SEED, verbose=0, output=None)
    assert _trajectory(study) == _trajectory(reference)


def test_resume_should_reject_a_space_that_is_not_the_space_of_the_folder(tmp_path, no_plots):
    """Given the same folder, when resume() gets a space whose Float ends at 2.0 instead of 1.0, then
    ValueError names the space.

    Oracle: resume docstring (an optimizer or space that does not match the folder raises ValueError).
    """
    folder = _start_function_space(tmp_path)

    with pytest.raises(ValueError, match="space"):
        resume(_aggregate, folder, space=_function_space(high=2.0), verbose=0)

    assert _json(folder / "config.json")["resumes"] == []


def test_resume_should_refuse_an_optimizer_with_other_settings_and_leave_the_folder_unchanged(
    tmp_path, no_plots
):
    """Given a folder written by ABCOptimizer with colony_size 8, when resume() gets one with colony_size 6,
    then RuntimeError names trial 3, the first that differs, and trials.jsonl and config.json are unchanged.

    Oracle: resume docstring (a candidate that differs from the logged trial raises RuntimeError naming the
    run and the trial, and the folder stays as it is).
    """
    _, folder = _start(tmp_path, MAKERS[0].values[0], _interrupting(20))
    before = (_log(folder).read_bytes(), (folder / "config.json").read_bytes())
    space = _space()

    with pytest.raises(RuntimeError, match="trial 3 "):
        resume(_score, folder, optimizer=ABCOptimizer(space, "max", 6, seed=ANY_SEED), verbose=0)

    assert (_log(folder).read_bytes(), (folder / "config.json").read_bytes()) == before


def _remove_line(folder, index):
    lines = _log(folder).read_text(encoding="utf-8").splitlines(keepends=True)
    _log(folder).write_text("".join(lines[:index] + lines[index + 1 :]), encoding="utf-8", newline="")


def _break_line(folder, index):
    lines = _log(folder).read_text(encoding="utf-8").splitlines(keepends=True)
    lines[index] = "not a trial\n"
    _log(folder).write_text("".join(lines), encoding="utf-8", newline="")


def _append_trial_after_the_end(folder):
    lines = _log(folder).read_text(encoding="utf-8").splitlines()
    extra = {**json.loads(lines[-1]), "trial": len(lines), "cached": True}
    _log(folder).write_text("\n".join([*lines, json.dumps(extra)]) + "\n", encoding="utf-8", newline="")


def test_resume_should_refuse_a_log_with_a_removed_line(tmp_path, no_plots):
    """Given a log whose 6th line was removed, when resume() is called, then RuntimeError names trial 5 and
    the log stays as it is.

    Oracle: resume docstring (an edited log raises RuntimeError naming the trial).
    """
    _, folder = _start(tmp_path, MAKERS[1].values[0], _interrupting(20))
    _remove_line(folder, 5)
    edited = _log(folder).read_bytes()

    with pytest.raises(RuntimeError, match="trial 5 "):
        resume(_score, folder, verbose=0)

    assert _log(folder).read_bytes() == edited


def test_resume_should_refuse_a_log_with_a_broken_line(tmp_path, no_plots):
    """Given a log whose 4th line is not JSON, when resume() is called, then RuntimeError names line 4.

    Oracle: resume docstring (the folder cannot be continued and stays as it is) and report module docstring
    (one JSON line per trial).
    """
    _, folder = _start(tmp_path, MAKERS[2].values[0], _interrupting(20))
    _break_line(folder, 3)

    with pytest.raises(RuntimeError, match="line 4 "):
        resume(_score, folder, verbose=0)

    assert _log(folder).read_text(encoding="utf-8").splitlines()[3] == "not a trial"


def test_resume_should_refuse_a_log_that_holds_a_trial_after_the_end_of_the_search(tmp_path, no_plots):
    """Given a space of three values searched to exhaustion and a fabricated cached trial appended to its
    log, when resume() is called, then RuntimeError says the folder cannot be continued.

    Oracle: resume docstring (a log that does not match the search raises RuntimeError and the folder stays
    as it is).
    """
    optimize(
        lambda p: p["k"], Space({"k": Int(0, 2)}), max_evals=50, seed=ANY_SEED, verbose=0, output=tmp_path
    )
    folder = _folder(tmp_path)
    _append_trial_after_the_end(folder)

    with pytest.raises(RuntimeError, match="cannot be continued"):
        resume(lambda p: p["k"], folder, verbose=0)

    assert _json(folder / "config.json")["resumes"] == []


def test_resume_should_continue_a_resumed_run_that_stopped_again(tmp_path, no_plots):
    """Given a run interrupted on its 8th call and a resume interrupted on its 10th call, when resume() is
    called once more, then the study equals the uninterrupted run and state.json lists two pauses.

    Oracle: resume docstring (a run that stopped for any reason continues) and RunState docstring (pauses
    lists every resume of the run).
    """
    make = MAKERS[2].values[0]
    _, folder = _start(tmp_path, make, _interrupting(7))
    resume(_interrupting(9), folder, verbose=0)

    study = resume(_score, folder, verbose=0)

    assert _trajectory(study) == _trajectory(_reference(make))
    assert len(_json(folder / "run_001" / "state.json")["pauses"]) == 2


@pytest.mark.parametrize("make", MAKERS)
def test_resume_should_write_the_derived_files_of_an_uninterrupted_run(tmp_path, no_plots, make):
    """Given each optimizer interrupted on its 13th call and resumed, when its derived files are compared
    with those of an uninterrupted run, then trials.csv, generations.csv, population.csv, parameters.csv,
    categories.csv, best_history.csv, best.json and the counts of summary.json are equal apart from
    wall-clock values.

    Oracle: report module docstring (every other file of a run folder is derived from the trials and is
    written again from the start when a run is resumed).
    """
    _start(tmp_path / "stopped", make, _interrupting(12))
    folder = _folder(tmp_path / "stopped")

    resume(_score, folder, verbose=0)

    _, whole = _start(tmp_path / "whole", make, _score)
    assert _derived(folder) == _derived(whole)
    assert _summary_counts(folder) == _summary_counts(whole)


def test_resume_should_leave_the_time_the_machine_was_off_out_of_every_duration(tmp_path, no_plots):
    """Given an interrupted run whose logged times are moved 10000 s into the past, as if the machine had
    been off since, when resume() continues it, then the run's duration, the time to the best trial and
    every elapsed_s and generation duration stay below 1000 s, and state.json records the pause.

    Oracle: report module docstring (time counts search time, the wall-clock time since the run started
    minus the time the run was stopped) and RunState docstring (pauses).
    """
    _, folder = _start(tmp_path, MAKERS[0].values[0], _interrupting(10))
    _shift_into_the_past(folder, 10_000.0)

    resume(_score, folder, verbose=0)

    summary = _json(folder / "summary.json")
    elapsed = [float(r["elapsed_s"]) for r in _rows(folder / "run_001" / "trials.csv")]
    durations = [float(r["duration_s"]) for r in _rows(folder / "run_001" / "generations.csv")]
    assert max([summary["duration_s"], summary["best"]["found_after_s"], *elapsed, *durations]) < 1000.0
    assert _json(folder / "run_001" / "state.json")["pauses"][0][1] > 9000.0


def test_resume_should_send_the_restored_trials_to_callbacks_marked_as_replayed(tmp_path, no_plots):
    """Given a run interrupted on its 11th call, when resume() runs with a recording callback, then the
    callback sees the logged trials with replaying True, then the new trials with replaying False.

    Oracle: Callback docstring (resume() sends the events of the restored trials, with state.replaying True)
    and RunState docstring (replaying turns False at the first trial that is evaluated).
    """
    _, folder = _start(tmp_path, MAKERS[1].values[0], _interrupting(10))
    restored = len(_log(folder).read_text(encoding="utf-8").splitlines())
    recorder = _Replays()

    study = resume(_score, folder, verbose=0, callbacks=[recorder])

    assert recorder.flags == [True] * restored + [False] * (len(study.trials) - restored)


def test_console_should_print_one_resumed_line_and_no_lines_for_the_restored_trials(
    tmp_path, no_plots, capsys, monkeypatch
):
    """Given a run interrupted on its 11th call, when resume() runs with verbose=2, then the second line is
    the 'resumed' line with 10 trials and 10 calls and one trial line is printed per new trial.

    Oracle: ConsoleReporter docstring (the restored trials print no lines; one 'resumed' line comes before
    the lines of the trials evaluated anew).
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)
    _, folder = _start(tmp_path, MAKERS[0].values[0], _interrupting(10))

    study = resume(_score, folder, verbose=2)

    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("[meta-hpo] run 1 | resumed | 10 trials restored (10 calls) | best ")
    assert len([line for line in lines if line.startswith("  trial")]) == len(study.trials) - 10


def test_resume_should_write_the_compressed_files_when_the_new_budget_reaches_compress_from(
    tmp_path, no_plots, monkeypatch
):
    """Given COMPRESS_FROM lowered to 40 and a run that finished at max_evals 30 with plain CSV files, when
    resume() extends it to 45, then trials.csv.gz holds every trial and the plain trials.csv and
    population.csv are gone.

    Oracle: report module docstring (max_evals >= COMPRESS_FROM writes trials.csv and population.csv
    gzip-compressed; the derived files are written again when a run is resumed).
    """
    monkeypatch.setattr(metahpo.report, "COMPRESS_FROM", 40)
    _, folder = _start(tmp_path, MAKERS[0].values[0], _score, max_evals=30)

    study = resume(_score, folder, max_evals=45, verbose=0)

    run_dir = folder / "run_001"
    assert len(_rows(run_dir / "trials.csv.gz")) == len(study.trials)
    assert ((run_dir / "trials.csv").exists(), (run_dir / "population.csv").exists()) == (False, False)


def test_resume_should_continue_a_folder_written_by_version_0_5_1(tmp_path, no_plots):
    """Given a seeded default run interrupted on its 9th call and turned into a 0.5.1 folder (no state.json,
    no start_epoch, null scores, choices as text), when resume() gets the space, then it evaluates the same
    parameters as the uninterrupted run.

    Oracle: resume docstring (0.5.x runs are repeated from the recorded seed, their Categorical choices need
    space, a non-finite score logged as null comes back as NaN).
    """
    optimize(_interrupting(8), _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=tmp_path)
    folder = _folder(tmp_path)
    _make_legacy(folder)

    study = resume(_score, folder, space=_space(), verbose=0)

    reference = optimize(_score, _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=None)
    assert [t.params for t in study.trials] == [t.params for t in reference.trials]


def _start_legacy_without_seed(root):
    """A 0.5.1 folder of an ABCOptimizer object seeded 5, so config.json records no seed."""
    space = _space()
    opt = ABCOptimizer(space, direction="max", seed=5)
    optimize(_interrupting(8), space, "max", BUDGET, optimizer=opt, verbose=0, output=root)
    folder = _folder(root)
    _make_legacy(folder)
    return folder


def test_resume_should_refuse_a_0_5_1_folder_without_a_seed(tmp_path, no_plots):
    """Given a 0.5.1 folder without a seed, when resume() is called without optimizer=, then ValueError says
    the search cannot be repeated without the seed.

    Oracle: resume docstring (a 0.5.x run without a seed raises ValueError unless optimizer is given).
    """
    folder = _start_legacy_without_seed(tmp_path)

    with pytest.raises(ValueError, match="seed"):
        resume(_score, folder, space=_space(), verbose=0)

    assert not (folder / "run_001" / "state.json").exists()


def test_resume_should_continue_a_0_5_1_folder_without_a_seed_with_the_passed_optimizer(tmp_path, no_plots):
    """Given a 0.5.1 folder without a seed, when resume() gets an ABCOptimizer seeded 5 as it was, then it
    evaluates the same parameters as the uninterrupted run.

    Oracle: resume docstring (unless optimizer is given).
    """
    folder = _start_legacy_without_seed(tmp_path)
    space = _space()

    study = resume(_score, folder, optimizer=ABCOptimizer(space, direction="max", seed=5), verbose=0)

    twin = _space()
    reference = optimize(
        _score, twin, "max", BUDGET, optimizer=ABCOptimizer(twin, "max", seed=5), output=None
    )
    assert [t.params for t in study.trials] == [t.params for t in reference.trials]


def test_resume_should_reject_a_folder_without_config_json(tmp_path):
    """Given an empty folder, when resume() is called, then ValueError says it is not a results folder.

    Oracle: resume docstring (folder is the results folder that holds config.json).
    """
    folder = tmp_path

    with pytest.raises(ValueError, match="config.json"):
        resume(_score, folder, verbose=0)

    assert list(folder.iterdir()) == []


def test_resume_should_return_a_study_for_a_single_run(tmp_path, no_plots):
    """Given a single run interrupted on its 5th call, when resume() returns, then the result is a Study.

    Oracle: resume docstring (the result is a Study, or a list of Study when the folder holds several runs).
    """
    _, folder = _start(tmp_path, MAKERS[0].values[0], _interrupting(4))

    study = resume(_score, folder, verbose=0)

    assert isinstance(study, Study)


def _spoil_state(folder, text):
    """Replaces run_001/state.json with `text`, as a power cut at the start of the run can leave it."""
    (folder / "run_001" / "state.json").write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("text", "keep"),
    [
        pytest.param("", 9, id="empty-with-logged-trials"),
        pytest.param('{"started": 17', 9, id="cut-json-with-logged-trials"),
        pytest.param("", 0, id="empty-without-logged-trials"),
    ],
)
def test_resume_should_repeat_a_seeded_run_from_its_seed_when_state_json_is_unreadable(
    tmp_path, no_plots, text, keep
):
    """Given a default run with seed 0 whose log is cut after `keep` lines and whose state.json is empty or
    cut short, when resume() continues it, then the study equals the uninterrupted run.

    Oracle: resume docstring (a state.json that is empty or not valid JSON counts as missing: the run is
    repeated from its seed) and report module docstring (resume() treats such a state.json as missing).
    """
    optimize(_score, _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=tmp_path)
    folder = _folder(tmp_path)
    _kill(folder, keep)
    _spoil_state(folder, text)

    study = resume(_score, folder, verbose=0)

    reference = optimize(_score, _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=None)
    assert _trajectory(study) == _trajectory(reference)


def test_resume_should_refuse_a_run_without_a_seed_whose_state_json_is_unreadable(tmp_path, no_plots):
    """Given a run of an ABCOptimizer object, so config.json records no seed, whose state.json is empty,
    when resume() is called without optimizer=, then ValueError says the generator state is missing or
    unreadable and does not name 0.5.x, and trials.jsonl is unchanged.

    Oracle: resume docstring (without a seed such a run needs optimizer) and report module docstring.
    """
    space = _space()
    opt = ABCOptimizer(space, direction="max", seed=5)
    optimize(_interrupting(8), space, "max", BUDGET, optimizer=opt, verbose=0, output=tmp_path)
    folder = _folder(tmp_path)
    _spoil_state(folder, "")
    before = _log(folder).read_bytes()

    with pytest.raises(ValueError, match="missing or unreadable") as raised:
        resume(_score, folder, verbose=0)

    assert "0.5" not in str(raised.value)
    assert _log(folder).read_bytes() == before
