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
crosses COMPRESS_FROM; scores NaN, inf and -inf. On a space where particles on a wall repeat parameter sets:
PSO budgets that cut a flight after a cached repeat, extended, and the same folders as 0.7.0 and older wrote
them (written with the loop those versions had, which resume() keeps to replay them: _Run.loop with
cut_first), finished, extended and interrupted after the swarm was asked again. A Categorical of closures,
lambdas, bound methods, partials and plain objects built again from new objects, its config.json with the
memory addresses 0.7.0 kept, and spaces whose choices really differ. A state.json without each of its keys,
and with content of the wrong kind; a config.json without each key resume() reads, with values of the wrong
kind, and with a space, settings or seed that do not build; a log line whose trial number overflows. A
Categorical of numpy arrays resumed with the space passed. log_every and verbose that are no integers,
False as verbose, and a numpy integer max_evals. The integer settings of ABC and PSO recorded as
whole floats, as 0.6.0 and 0.7.0 stored them. Custom optimizers whose class is called ABCOptimizer,
PSOOptimizer or BBPSOOptimizer, and library optimizers in folders without the optimizer's module.
"""

import csv
import functools
import gzip
import json
import math

import numpy as np
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
    score, Categorical choices as repr() text, no cache flag, no resumes, no stopping rules and no colony_size
    setting."""
    (folder / "run_001" / "state.json").unlink()
    records = [json.loads(line) for line in _log(folder).read_text(encoding="utf-8").splitlines()]
    old = [_legacy_record(r) for r in records]
    _log(folder).write_text("".join(json.dumps(r) + "\n" for r in old), encoding="utf-8", newline="")
    config = _json(folder / "config.json")
    config["space"] = [_legacy_parameter(p) for p in config["space"]]
    config["optimizer"]["settings"].pop("colony_size")
    config.pop("cache")
    config.pop("resumes")
    config.pop("stop")
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


def _cut_space():
    """Int(1, 4) x Categorical(['gini', 'entropy']) x Float(0, 1): particles that sit on a wall of the Float
    hand out repeats, so a budget that cuts a PSO flight often keeps a cached repeat before the cut."""
    return Space({"depth": Int(1, 4), "crit": Categorical(["gini", "entropy"]), "f": Float(0.0, 1.0)})


def _rounded(params):
    """A score rounded to two decimals, so many parameter sets tie."""
    return round(params["depth"] * 0.1 + (params["crit"] == "gini") * 0.05 + params["f"] * 0.01, 2)


def _cut_start(root, swarm, seed, max_evals, function=_rounded):
    """optimize() maximising `function` on the cut space with a PSOOptimizer, written below root; returns the
    study and the results folder."""
    space = _cut_space()
    opt = PSOOptimizer(space, "max", swarm_size=swarm, seed=seed)
    study = optimize(function, space, "max", max_evals, optimizer=opt, verbose=0, output=root)
    return study, _folder(root)


def _old_cut_start(monkeypatch, root, swarm, seed, max_evals, function=_rounded):
    """_cut_start with the loop of meta-hpo 0.7.0 and older, which resume() keeps to replay their folders
    (_Run.loop with cut_first): a batch is cut to the remaining calls first, and the swarm is asked again for
    a call that a cached repeat left over. The folder is the one such a version wrote."""
    with monkeypatch.context() as patch:
        patch.setattr(metahpo.facade, "_Start", functools.partial(metahpo.facade._Start, cut_first=True))
        return _cut_start(root, swarm, seed, max_evals, function)


@pytest.mark.parametrize("seed", [1, 3, 4])
def test_resume_should_extend_a_run_whose_budget_cut_a_flight_after_a_cached_repeat_like_a_longer_run(
    tmp_path, no_plots, seed
):
    """Given a PSO run (swarm 20) on the cut space that finished at max_evals 30 inside its second flight,
    after a cached repeat, when resume() is called with max_evals 50, then the study, its stop reason and
    generations.csv equal those of a run started with 50.

    Oracle: resume docstring (a new max_evals gives the result of a run started with that budget).
    Decisive: these seeds made 0.7.0 ask the swarm again for the call the cached repeat left over, so the
    log held a trial the longer run never tries and the resume raised RuntimeError.
    """
    _, folder = _cut_start(tmp_path / "stopped", 20, seed, 30)

    study = resume(_rounded, folder, max_evals=50, verbose=0)

    reference, whole = _cut_start(tmp_path / "whole", 20, seed, 50)
    assert (_trajectory(study), study.stop_reason) == (_trajectory(reference), reference.stop_reason)
    generations = [_timeless(_rows(f / "run_001" / "generations.csv")) for f in (folder, whole)]
    assert generations[0] == generations[1]


def test_resume_should_return_a_finished_folder_whose_last_flight_an_older_version_cut_without_a_call(
    tmp_path, no_plots, monkeypatch
):
    """Given a folder as 0.7.0 wrote it for a PSO run (swarm 20, seed 1) that finished at max_evals 30 with
    31 trials, the last one handed out when the swarm was asked again, when resume() is called with the same
    budget, then it returns those trials and stop_reason 'budget' without a call.

    Oracle: resume docstring (a folder whose runs all finished evaluates nothing). Decisive: the loop of
    this version would hand out the rest of the cut flight instead of asking again, so a resume that
    replayed the folder only with it would raise RuntimeError.
    """
    original, folder = _old_cut_start(monkeypatch, tmp_path, 20, 1, 30)

    study = resume(_refusing, folder, verbose=0)

    assert len(_log(folder).read_text(encoding="utf-8").splitlines()) == 31
    assert (_trajectory(study), study.stop_reason) == (_trajectory(original), "budget")


def test_resume_should_refuse_to_extend_a_folder_whose_last_flight_an_older_version_cut(
    tmp_path, no_plots, monkeypatch
):
    """Given the same folder as 0.7.0 wrote it, when resume() is called with max_evals 50, then ValueError
    names the recorded budget max_evals=30, and trials.jsonl and config.json are unchanged.

    Oracle: resume docstring (max_evals gives the result of a run started with that budget, which the trial
    0.7.0 handed out after asking the swarm again rules out; the folder stays as it is when it cannot be
    continued).
    """
    _, folder = _old_cut_start(monkeypatch, tmp_path, 20, 1, 30)
    before = (_log(folder).read_bytes(), (folder / "config.json").read_bytes())

    with pytest.raises(ValueError, match="max_evals=30"):
        resume(_rounded, folder, max_evals=50, verbose=0)

    assert (_log(folder).read_bytes(), (folder / "config.json").read_bytes()) == before


def test_resume_should_continue_a_folder_an_older_version_interrupted_after_it_asked_the_swarm_again(
    tmp_path, no_plots, monkeypatch
):
    """Given a folder as 0.7.0 wrote it for a PSO run (swarm 7, seed 16, max_evals 25) interrupted on its
    25th call, after the swarm was asked again for a call a cached repeat left over, when resume() continues
    it with the same budget, then the study equals the uninterrupted run of 0.7.0, which differs from the
    run of this version.

    Oracle: resume docstring (a run continues exactly where it stopped). Decisive: the log holds a trial
    handed out after asking again, so only the loop of 0.7.0 replays and continues it.
    """
    _, folder = _old_cut_start(monkeypatch, tmp_path / "stopped", 7, 16, 25, _interrupting(24, _rounded))

    study = resume(_rounded, folder, verbose=0)

    old, _ = _old_cut_start(monkeypatch, tmp_path / "old", 7, 16, 25)
    new, _ = _cut_start(tmp_path / "new", 7, 16, 25)
    assert _trajectory(study) == _trajectory(old)
    assert _trajectory(old) != _trajectory(new)


class _Doubler:
    """A plain object with the default repr, which holds a memory address: a choice itself and, through its
    bound method, a method choice."""

    def __call__(self, x):
        return 2.0 * x

    def halve(self, x):
        return 0.5 * x


def _closures():
    """Two new functions on every call, so their memory addresses differ from call to call as from process
    to process."""

    def relu(x):
        return max(x, 0.0)

    def shift(x, by):
        return x + by

    return relu, shift


CHOICE_KINDS = [
    pytest.param(lambda: [_closures()[0], abs], id="function"),
    pytest.param(lambda: [lambda x: x, lambda x: -x], id="lambda"),
    pytest.param(lambda: [_Doubler().halve, _Doubler().__call__], id="bound-method"),
    pytest.param(
        lambda: [functools.partial(_closures()[1], by=1.0), functools.partial(_closures()[1], by=-1.0)],
        id="partial",
    ),
    pytest.param(lambda: [_Doubler(), _Doubler()], id="plain-object"),
]


def _choice_space(choices):
    return Space({"act": Categorical(choices), "x": Float(-1.0, 1.0)})


def _apply(params):
    return params["act"](params["x"])


def _positions(study):
    """Every trial's number, exact score, cached flag and position, which identify a search whose params
    hold objects that compare by identity."""
    return [(t.number, repr(t.score), t.cached, t.x.tolist()) for t in study.trials]


@pytest.mark.parametrize("choices", CHOICE_KINDS)
def test_resume_should_continue_with_a_space_of_new_objects_of_the_same_kinds_as_a_new_process_builds(
    tmp_path, no_plots, choices
):
    """Given a folder whose Categorical holds functions, lambdas, bound methods, partials or plain objects,
    interrupted on its 7th call, when resume() gets a space built again from new objects, whose memory
    addresses differ as they do in a new process, then the study equals the uninterrupted run and
    config.json holds no memory address.

    Oracle: resume docstring (space must be the space that config.json describes) and report module
    docstring (the text of such a choice leaves out the memory address, so it is the same in every process).
    """
    optimize(_interrupting(6, _apply), _choice_space(choices()), "max", 20, seed=ANY_SEED, output=tmp_path)
    folder = _folder(tmp_path)

    study = resume(_apply, folder, space=_choice_space(choices()), verbose=0)

    reference = optimize(_apply, _choice_space(choices()), "max", 20, seed=ANY_SEED, output=None)
    assert _positions(study) == _positions(reference)
    assert " at 0x" not in (folder / "config.json").read_text(encoding="utf-8")


def _old_addresses(folder):
    """Writes every choice text of config.json the way 0.7.0 kept it, with a memory address before the
    closing bracket."""
    config = _json(folder / "config.json")
    config["space"][0]["choices"] = [
        c[:-1] + " at 0x000001D2C3B4A5F6>" for c in config["space"][0]["choices"]
    ]
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_resume_should_accept_a_folder_whose_config_json_kept_the_memory_addresses_of_older_versions(
    tmp_path, no_plots
):
    """Given a folder of a Categorical of a closure and abs interrupted on its 7th call, whose config.json
    records the choices with memory addresses as 0.7.0 did, when resume() gets the space built again, then
    the study equals the uninterrupted run.

    Oracle: resume docstring (space must be the space that config.json describes) and report module
    docstring (the text leaves out the memory address, which older versions kept).
    """
    choices = CHOICE_KINDS[0].values[0]
    optimize(_interrupting(6, _apply), _choice_space(choices()), "max", 20, seed=ANY_SEED, output=tmp_path)
    folder = _folder(tmp_path)
    _old_addresses(folder)

    study = resume(_apply, folder, space=_choice_space(choices()), verbose=0)

    reference = optimize(_apply, _choice_space(choices()), "max", 20, seed=ANY_SEED, output=None)
    assert _positions(study) == _positions(reference)


def _tanh_closure():
    def tanh(x):
        return math.tanh(x)

    return tanh


@pytest.mark.parametrize(
    "space",
    [
        pytest.param(lambda: _choice_space([_tanh_closure(), abs]), id="function-of-another-name"),
        pytest.param(lambda: _choice_space([_closures()[0], abs, min]), id="three-choices"),
        pytest.param(lambda: _choice_space([_closures()[0], "linear"]), id="other-json-value"),
        pytest.param(
            lambda: Space({"fn": Categorical([_closures()[0], abs]), "x": Float(-1.0, 1.0)}),
            id="other-parameter-name",
        ),
    ],
)
def test_resume_should_still_reject_a_space_whose_choices_really_differ(tmp_path, no_plots, space):
    """Given a folder whose Categorical holds a closure relu and abs (one with a memory address in its repr,
    one without), when resume() gets a space with a function of another name, three choices, a string
    instead of abs or another parameter name, then ValueError names the space and the folder records no
    resume.

    Oracle: resume docstring (a space that does not match the folder raises ValueError).
    """
    choices = CHOICE_KINDS[0].values[0]
    optimize(_interrupting(6, _apply), _choice_space(choices()), "max", 20, seed=ANY_SEED, output=tmp_path)
    folder = _folder(tmp_path)

    with pytest.raises(ValueError, match="space"):
        resume(_apply, folder, space=space(), verbose=0)

    assert _json(folder / "config.json")["resumes"] == []


def _edit_state(folder, change):
    """Replaces run_001/state.json with change(its content)."""
    path = folder / "run_001" / "state.json"
    path.write_text(json.dumps(change(_json(path))), encoding="utf-8")


def _without(*keys):
    return lambda record: {k: v for k, v in record.items() if k not in keys}


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_without("generator"), id="no-generator-repeated-from-the-seed"),
        pytest.param(_without("started"), id="no-start-from-the-first-trial"),
        pytest.param(_without("pauses"), id="no-pauses-of-a-run-never-resumed"),
        pytest.param(_without("generator", "started", "pauses"), id="empty-object"),
    ],
)
def test_resume_should_rebuild_what_state_json_lacks_from_the_other_records(tmp_path, no_plots, change):
    """Given a default run with seed 0 interrupted on its 10th call whose state.json lacks its generator
    state, its start time, its pauses or all three, when resume() continues it, then the study equals the
    uninterrupted run.

    Oracle: resume docstring (without the generator state the run is repeated from its seed) and the
    RunState docstring (pauses lists the resumes of the run, so a run that was never resumed has none); the
    start time only moves the time columns, which the study does not hold.
    """
    optimize(_interrupting(9), _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=tmp_path)
    folder = _folder(tmp_path)
    _edit_state(folder, change)

    study = resume(_score, folder, verbose=0)

    reference = optimize(_score, _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=None)
    assert _trajectory(study) == _trajectory(reference)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda state: [state], id="a-list"),
        pytest.param(
            lambda state: {**state, "generator": {"bit_generator": "MT19937"}}, id="foreign-generator"
        ),
        pytest.param(lambda state: {**state, "started": "soon"}, id="start-not-a-number"),
        pytest.param(lambda state: {**state, "started": 1e400}, id="start-infinite"),
        pytest.param(lambda state: {**state, "pauses": [[1]]}, id="pause-without-seconds"),
        pytest.param(lambda state: {**state, "pauses": [[-1, 0.0]]}, id="pause-at-a-negative-trial"),
        pytest.param(lambda state: {**state, "pauses": 3}, id="pauses-not-a-list"),
    ],
)
def test_resume_should_refuse_a_state_json_that_does_not_hold_the_state_of_a_run(tmp_path, no_plots, change):
    """Given a run interrupted on its 10th call whose state.json holds a list, a generator state of another
    kind, a start time that is not a finite number or pauses that are not (trial, seconds) pairs, when
    resume() is called, then ValueError names state.json and trials.jsonl is unchanged.

    Oracle: error conventions of the library (a broken results folder raises ValueError with a message that
    names the file) and resume docstring (the folder stays as it is when it cannot be continued).
    """
    optimize(_interrupting(9), _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=tmp_path)
    folder = _folder(tmp_path)
    _edit_state(folder, change)
    before = _log(folder).read_bytes()

    with pytest.raises(ValueError, match="state.json"):
        resume(_score, folder, verbose=0)

    assert _log(folder).read_bytes() == before


def test_resume_should_refuse_a_resumed_run_whose_state_json_lost_its_pauses(tmp_path, no_plots):
    """Given a run interrupted on its 8th call, resumed and interrupted again, whose state.json then lost its
    pauses, when resume() is called, then ValueError names state.json, since the time the run was stopped
    cannot be rebuilt.

    Oracle: RunState docstring (pauses lists every resume of the run) and the error conventions (a broken
    results folder raises ValueError naming the file).
    """
    optimize(_interrupting(7), _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=tmp_path)
    folder = _folder(tmp_path)
    resume(_interrupting(5), folder, verbose=0)
    _edit_state(folder, _without("pauses"))

    with pytest.raises(ValueError, match="state.json"):
        resume(_score, folder, verbose=0)

    assert len(_json(folder / "config.json")["resumes"]) == 1


def _set(key, value):
    return lambda config: {**config, key: value}


def _space_entry(**fields):
    return lambda config: {
        **config,
        "space": [config["space"][0], {**config["space"][1], **fields}, *config["space"][2:]],
    }


DAMAGED_CONFIGS = [
    pytest.param(lambda config: [config], id="a-list"),
    pytest.param(_without("runs"), id="no-runs"),
    pytest.param(_set("runs", "two"), id="runs-a-string"),
    pytest.param(_set("runs", True), id="runs-a-boolean"),
    pytest.param(_set("direction", "up"), id="unknown-direction"),
    pytest.param(_without("max_evals"), id="no-max-evals"),
    pytest.param(_set("max_evals", 0), id="max-evals-zero"),
    pytest.param(_without("seeds"), id="no-seeds"),
    pytest.param(_set("seeds", [0, 1]), id="more-seeds-than-runs"),
    pytest.param(_without("optimizer"), id="no-optimizer"),
    pytest.param(_set("optimizer", {"name": "ABCOptimizer"}), id="optimizer-without-settings"),
    pytest.param(_without("space"), id="no-space"),
    pytest.param(_set("space", [{}]), id="space-entry-empty"),
    pytest.param(_space_entry(type="Normal"), id="unknown-parameter-type"),
    pytest.param(_space_entry(low="small"), id="bound-a-string"),
    pytest.param(_space_entry(log="yes"), id="log-not-a-boolean"),
    pytest.param(
        _set("space", [{"name": "act", "type": "Categorical", "choices": "relu"}]), id="choices-a-string"
    ),
    pytest.param(
        _set("space", [{"name": "act", "type": "Categorical", "choices": [1], "repr": True}]),
        id="text-choice-a-number",
    ),
    pytest.param(_set("cache", "yes"), id="cache-not-a-boolean"),
    pytest.param(_set("stop", [{}]), id="stop-entry-empty"),
    pytest.param(
        _set("stop", [{"name": "Patience", "settings": {}, "module": 3}]), id="stop-module-a-number"
    ),
    pytest.param(_set("resumes", {}), id="resumes-not-a-list"),
    pytest.param(
        lambda config: {**config, "optimizer": {**config["optimizer"], "module": 3}},
        id="optimizer-module-a-number",
    ),
]


@pytest.mark.parametrize("change", DAMAGED_CONFIGS)
def test_resume_should_refuse_a_config_json_that_lacks_what_resume_reads_and_name_it(
    tmp_path, no_plots, change
):
    """Given a run interrupted on its 10th call whose config.json lacks a key resume() reads or holds a value
    of the wrong kind there, when resume() is called, then ValueError names config.json and trials.jsonl is
    unchanged.

    Oracle: error conventions of the library (a broken results folder raises ValueError naming the file)
    and the report module docstring (what config.json records for a resume).
    """
    optimize(_interrupting(9), _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=tmp_path)
    folder = _folder(tmp_path)
    path = folder / "config.json"
    path.write_text(json.dumps(change(_json(path))), encoding="utf-8")
    before = _log(folder).read_bytes()

    with pytest.raises(ValueError, match="config.json"):
        resume(_score, folder, verbose=0)

    assert _log(folder).read_bytes() == before


@pytest.mark.parametrize("text", [pytest.param("", id="empty"), pytest.param('{"runs": 1', id="cut-short")])
def test_resume_should_refuse_a_config_json_that_is_not_valid_json_and_name_it(tmp_path, no_plots, text):
    """Given a run interrupted on its 10th call whose config.json is empty or cut short, when resume() is
    called, then ValueError names config.json and says it is not valid JSON.

    Oracle: error conventions of the library (a broken results folder raises ValueError naming the file).
    """
    optimize(_interrupting(9), _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=tmp_path)
    folder = _folder(tmp_path)
    (folder / "config.json").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="config.json") as raised:
        resume(_score, folder, verbose=0)

    assert "valid JSON" in str(raised.value)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_space_entry(low=300), id="int-low-above-high"),
        pytest.param(
            lambda config: {**config, "optimizer": {**config["optimizer"], "settings": {"colonies": 8}}},
            id="unknown-optimizer-setting",
        ),
        pytest.param(_set("seeds", [-1]), id="negative-seed"),
        pytest.param(
            lambda config: {**config, "optimizer": {**config["optimizer"], "settings": {"limit": 2.5}}},
            id="limit-not-a-whole-number",
        ),
    ],
)
def test_resume_should_name_config_json_when_its_space_settings_or_seed_do_not_build(
    tmp_path, no_plots, change
):
    """Given a run interrupted on its 10th call whose config.json describes an Int with low above high,
    records a setting ABCOptimizer does not take or a negative seed, when resume() is called, then
    ValueError names config.json and trials.jsonl is unchanged.

    Oracle: error conventions of the library (a broken results folder raises ValueError naming the file).
    """
    optimize(_interrupting(9), _space(), "max", BUDGET, seed=ANY_SEED, verbose=0, output=tmp_path)
    folder = _folder(tmp_path)
    path = folder / "config.json"
    path.write_text(json.dumps(change(_json(path))), encoding="utf-8")
    before = _log(folder).read_bytes()

    with pytest.raises(ValueError, match="config.json"):
        resume(_score, folder, verbose=0)

    assert _log(folder).read_bytes() == before


def test_resume_should_refuse_a_log_line_whose_trial_number_does_not_fit_an_integer(tmp_path, no_plots):
    """Given a log whose 3rd line records the trial number 1e400, which JSON reads as infinity, when resume()
    is called, then RuntimeError names line 3, as for any other line that is not a trial record.

    Oracle: resume docstring (the folder cannot be continued and stays as it is) and report module docstring
    (one JSON line per trial).
    """
    _, folder = _start(tmp_path, MAKERS[0].values[0], _interrupting(20))
    lines = _log(folder).read_text(encoding="utf-8").splitlines(keepends=True)
    lines[2] = lines[2].replace('"trial": 2,', '"trial": 1e400,')
    _log(folder).write_text("".join(lines), encoding="utf-8", newline="")

    with pytest.raises(RuntimeError, match="line 3 "):
        resume(_score, folder, verbose=0)

    assert "1e400" in _log(folder).read_text(encoding="utf-8")


def _array_space():
    """A Categorical of three numpy arrays, such as class weights, beside a Float."""
    weights = [np.array([1.0, 1.0]), np.array([1.0, 5.0]), np.array([1.0, 10.0])]
    return Space({"w": Categorical(weights), "x": Float(0.0, 1.0)})


def _weighted(params):
    return float(params["w"].sum()) * params["x"]


def test_resume_should_continue_a_folder_whose_categorical_holds_numpy_arrays_with_the_passed_space(
    tmp_path, no_plots
):
    """Given a folder whose Categorical holds three numpy arrays, interrupted on its 7th call, when
    resume() gets the space built again, then the study equals the uninterrupted run.

    Oracle: resume docstring (space is needed when a Categorical has choices that JSON cannot hold) and the
    Categorical docstring (any object can be a choice). Decisive: writing categories.csv compared the
    arrays with ==, so the interrupted run already raised ValueError.
    """
    optimize(_interrupting(6, _weighted), _array_space(), "max", 20, seed=ANY_SEED, output=tmp_path)
    folder = _folder(tmp_path)

    study = resume(_weighted, folder, space=_array_space(), verbose=0)

    reference = optimize(_weighted, _array_space(), "max", 20, seed=ANY_SEED, output=None)
    assert _positions(study) == _positions(reference)


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        pytest.param({"log_every": 0}, "log_every", id="log-every-0-without-console"),
        pytest.param({"log_every": 2.0}, "log_every", id="log-every-whole-float"),
        pytest.param({"verbose": 1.5}, "verbose", id="verbose-fraction"),
        pytest.param({"verbose": -1}, "verbose", id="verbose-negative"),
        pytest.param({"max_evals": 50.0}, "max_evals", id="max-evals-whole-float"),
    ],
)
def test_resume_should_reject_an_integer_argument_that_is_no_integer_before_the_folder_changes(
    tmp_path, no_plots, kwargs, name
):
    """Given a run interrupted on its 10th call, when resume() gets log_every 0 or 2.0, verbose 1.5 or -1, or
    max_evals 50.0, then ValueError names the argument and every file of the folder is unchanged.

    Oracle: resume docstring (verbose, log_every and callbacks work as in optimize(); nothing in the
    folder changes before the arguments are checked) and the library's error convention. Decisive:
    log_every=0 and verbose=1.5 were taken, and the run was restored and written again.
    """
    _, folder = _start(tmp_path, MAKERS[0].values[0], _interrupting(9))
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    with pytest.raises(ValueError, match=name):
        resume(_refusing, folder, **kwargs)

    assert {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before


def test_resume_should_accept_a_numpy_integer_max_evals_and_record_a_python_int(tmp_path, no_plots):
    """Given a run that finished at max_evals 40, when resume() gets max_evals=np.int64(50), then the study
    equals a run started with 50 and config.json records 50 as a plain JSON integer.

    Oracle: resume docstring (max_evals gives the result of a run started with that budget).
    """
    make = MAKERS[0].values[0]
    _, folder = _start(tmp_path, make, _score)

    study = resume(_score, folder, max_evals=np.int64(50), verbose=0)

    recorded = _json(folder / "config.json")["max_evals"]
    assert _trajectory(study) == _trajectory(_reference(make, max_evals=50))
    assert (recorded, type(recorded)) == (50, int)


def test_resume_should_take_false_as_verbose_level_0(tmp_path, no_plots, capsys, monkeypatch):
    """Given a finished folder and no METAHPO_VERBOSE, when resume() gets verbose=False, then it returns the
    study without a call and prints nothing.

    Oracle: resume docstring (verbose works as in optimize()) and the optimize docstring (False is level 0).
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)
    original, folder = _start(tmp_path, MAKERS[0].values[0], _score)

    study = resume(_refusing, folder, verbose=False)

    assert (_trajectory(study), capsys.readouterr().out) == (_trajectory(original), "")


def _whole_floats(folder, names):
    """Writes the settings `names` of config.json as floats, as 0.6.0 and 0.7.0 recorded a setting given as
    10.0."""
    config = _json(folder / "config.json")
    settings = config["optimizer"]["settings"]
    config["optimizer"]["settings"] = {k: float(v) if k in names else v for k, v in settings.items()}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")


@pytest.mark.parametrize(
    ("make", "names"),
    [
        pytest.param(MAKERS[0].values[0], ("colony_size", "limit", "max_tries"), id="abc"),
        pytest.param(MAKERS[1].values[0], ("swarm_size", "max_tries"), id="pso"),
        pytest.param(MAKERS[2].values[0], ("swarm_size", "max_tries"), id="bbpso"),
    ],
)
def test_resume_should_take_an_integer_setting_recorded_as_a_whole_float_as_that_integer(
    tmp_path, no_plots, make, names
):
    """Given each optimizer interrupted on its 10th call whose config.json records its integer settings as
    whole floats (limit 15.0, max_tries 100.0, ...), as 0.6.0 and 0.7.0 recorded a setting given that way,
    when resume() rebuilds the optimizer from the folder, then the study equals the uninterrupted run.

    Oracle: resume docstring (folders of older versions continue; the optimizer is built again with its
    recorded settings) and the stage brief (a recorded whole-number float is taken as that integer).
    Decisive: the integer settings now reject floats, so such a folder raised ValueError naming config.json.
    """
    _, folder = _start(tmp_path, make, _interrupting(9))
    _whole_floats(folder, names)

    study = resume(_score, folder, verbose=0)

    assert _trajectory(study) == _trajectory(_reference(make))


NAMED_LIKE_LIBRARY_OPTIMIZERS = [
    pytest.param(type("ABCOptimizer", (_RandomSearch,), {}), id="abc"),
    pytest.param(type("PSOOptimizer", (_RandomSearch,), {}), id="pso"),
    pytest.param(type("BBPSOOptimizer", (_RandomSearch,), {}), id="bbpso"),
]


@pytest.mark.parametrize("cls", NAMED_LIKE_LIBRARY_OPTIMIZERS)
def test_resume_should_ask_for_the_optimizer_when_a_custom_class_carries_the_name_of_a_library_optimizer(
    tmp_path, no_plots, cls
):
    """Given a folder written by a custom random search whose class is called ABCOptimizer, PSOOptimizer or
    BBPSOOptimizer, interrupted on its 4th call, when resume() runs without optimizer=, then ValueError asks
    for optimizer= and the folder stays byte for byte as it was; with the custom optimizer passed the study
    equals the uninterrupted run.

    Oracle: resume docstring (optimizer is needed for a folder written by an optimizer other than the
    library's ABC, PSO and BBPSO; a folder of another optimizer without it raises ValueError) and the report
    module docstring (config.json records the module of the optimizer class). Decisive: the class name
    alone matched, so the library class was built in its place and continued the search silently, because
    its first draws equal those of the random search.
    """
    space = _space()
    optimize(_interrupting(3), space, "max", 12, optimizer=cls(space, "max", ANY_SEED), output=tmp_path)
    folder = _folder(tmp_path)
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    with pytest.raises(ValueError, match="optimizer="):
        resume(_score, folder, verbose=0)
    unchanged = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before
    again = _space()
    study = resume(_score, folder, optimizer=cls(again, "max", ANY_SEED), verbose=0)

    twin = _space()
    reference = optimize(_score, twin, "max", 12, optimizer=cls(twin, "max", ANY_SEED), output=None)
    assert unchanged
    assert _trajectory(study) == _trajectory(reference)


@pytest.mark.parametrize("make", MAKERS)
def test_resume_should_rebuild_a_library_optimizer_of_a_folder_that_records_no_module(
    tmp_path, no_plots, make
):
    """Given each library optimizer interrupted on its 10th call whose config.json records no module for the
    optimizer, as folders of 0.6.0, 0.7.0 and 0.7.1 do, when resume() runs without optimizer=, then the
    study equals the uninterrupted run.

    Oracle: resume docstring (folders of older versions continue; the library optimizers are built again
    from config.json).
    """
    _, folder = _start(tmp_path, make, _interrupting(9))
    config = _json(folder / "config.json")
    config["optimizer"].pop("module")
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")

    study = resume(_score, folder, verbose=0)

    assert _trajectory(study) == _trajectory(_reference(make))
