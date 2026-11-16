"""Unit tests of the reporting: ConsoleReporter, ResultWriter, the run events and the stop reasons.

Oracles: the module docstring of metahpo.report (events, console levels, folder layout, files written while
the search runs, environment overrides, compression from COMPRESS_FROM), the docstrings of Callback,
RunState, Trial (meta, start_time, duration) and Study (stop_reason), Optimizer.population and
Optimizer.generation, and the optimize docstring (verbose, output, callbacks, KeyboardInterrupt and errors).

Every run writes into pytest's tmp_path; the suite switches the default folder and console off through
METAHPO_OUTPUT and METAHPO_VERBOSE in conftest, and the console tests remove METAHPO_VERBOSE first. Plots are
switched off here (ResultWriter(plots=False) or through a stub) and tested in test_plots.py.

The clock that spaces the syncs of trials.jsonl and os.fsync are replaced through metahpo.report in the
sync tests, so they do not depend on real time.

Unspecified, therefore not asserted: exact float formatting of the console lines, timings, folder names
beyond the time stamp and optimizer suffix.

Input space: Every objective, space, optimizer, budget and output setting, which is unbounded.
Subset: The three optimizers on a four-parameter mixed space with small budgets; console levels 0, 1 and 2;
log_every 1 and 2; objectives that stop with each stop reason (budget, exhausted, interrupted, error); an
all-NaN objective; a budget above COMPRESS_FROM after lowering it; non-default settings of every optimizer
and a custom optimizer; choices of every JSON type, of Python types and NaN; scores NaN, inf and -inf;
trials 0.5 s and 2 s apart on a fake clock; no stopping rule, one rule, and the three built-in rules in a
given order, with one and two runs; bounds given as numpy scalars; Categoricals of numpy arrays and of
0, False, 1, True and 1.0; numpy integer and float settings of every optimizer and stopping rule; a
custom optimizer with numpy scalars and arrays in its candidate meta and its population; an objective
that changes the working directory with a relative output, METAHPO_OUTPUT and the default folder; a
replace of best.json, config.json, state.json and summary.json refused for a while or for good (a
simulated reader that holds the file); callbacks that raise RuntimeError or KeyboardInterrupt in
on_run_start, on_trial, on_generation and on_run_end; a folder name taken by another search after the
check for it.
"""

import csv
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
    ConsoleReporter,
    Float,
    Int,
    MaxTime,
    Patience,
    PSOOptimizer,
    ResultWriter,
    Space,
    TargetScore,
    optimize,
)

ANY_SEED = 0


BUDGET = 60


def _space():
    """A fresh mixed space with a log Float, an Int, a Categorical and a linear Float."""
    return Space(
        {
            "lr": Float(1e-4, 1e-1, log=True),
            "units": Int(16, 256),
            "act": Categorical(["relu", "tanh", "gelu"]),
            "dropout": Float(0.0, 0.5),
        }
    )


def _score(params):
    """An accuracy-like score with its peak 0.95 near lr = 1e-3, units = 128, gelu and dropout 0.2."""
    act = {"relu": 0.01, "tanh": 0.02, "gelu": 0.0}[params["act"]]
    lr = 0.02 * (math.log10(params["lr"]) + 3.0) ** 2
    return 0.95 - lr - ((params["units"] - 128) / 256) ** 2 - (params["dropout"] - 0.2) ** 2 - act


def _run(tmp_path, optimizer=None, max_evals=BUDGET, function=_score, **kwargs):
    """optimize() maximising `function` with results written below tmp_path and plots off; returns
    (study, results folder)."""
    space = optimizer.space if optimizer is not None else _space()
    study = optimize(
        function,
        space,
        direction="max",
        max_evals=max_evals,
        optimizer=optimizer,
        seed=ANY_SEED,
        verbose=0,
        output=None,
        callbacks=[ResultWriter(tmp_path, plots=False), *kwargs.pop("callbacks", [])],
        **kwargs,
    )
    return study, _folder(tmp_path)


def _folder(root):
    """The only results folder below root."""
    folders = [p for p in root.iterdir() if p.is_dir()]
    assert len(folders) == 1
    return folders[0]


def _rows(path):
    """The rows of a CSV file (or .csv.gz) as dictionaries."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _interrupting(after):
    """An objective that raises KeyboardInterrupt on call `after` + 1."""
    calls = [0]

    def objective(params):
        calls[0] += 1
        if calls[0] > after:
            raise KeyboardInterrupt
        return _score(params)

    return objective


def _failing(after):
    """An objective that raises RuntimeError on call `after` + 1."""
    calls = [0]

    def objective(params):
        calls[0] += 1
        if calls[0] > after:
            raise RuntimeError("out of memory")
        return _score(params)

    return objective


class _Recorder(Callback):
    """Records the order of events and the content of best.json after every new best."""

    def __init__(self, root):
        self.root = root
        self.events = []
        self.best_files = []

    def on_run_start(self, state):
        self.events.append("start")

    def on_trial(self, state, trial, new_best):
        self.events.append("trial")
        self.best_files.append(
            new_best and _json(_folder(self.root) / "run_001" / "best.json")["best"]["score"]
        )

    def on_generation(self, state, generation, trials, population):
        self.events.append("generation")

    def on_run_end(self, state):
        self.events.append("end")


def test_result_writer_should_write_the_documented_files(tmp_path):
    """Given optimize() with a ResultWriter, when the run ends, then the folder holds config.json,
    summary.json,
    report.md and run_001 with trials.csv, trials.jsonl, state.json, best.json, best_history.csv,
    generations.csv, population.csv, parameters.csv, categories.csv and a plots folder.

    Oracle: report module docstring (folder layout).
    """
    expected = {
        "config.json",
        "summary.json",
        "report.md",
        "run_001/trials.csv",
        "run_001/trials.jsonl",
        "run_001/state.json",
        "run_001/best.json",
        "run_001/best_history.csv",
        "run_001/generations.csv",
        "run_001/population.csv",
        "run_001/parameters.csv",
        "run_001/categories.csv",
        "run_001/plots",
    }

    _, folder = _run(tmp_path)

    found = {p.relative_to(folder).as_posix() for p in folder.rglob("*")} - {"run_001"}
    assert found == expected
    assert folder.name.endswith("_abc")


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda s: ABCOptimizer(s, direction="max", seed=ANY_SEED), id="abc"),
        pytest.param(lambda s: PSOOptimizer(s, direction="max", seed=ANY_SEED), id="pso"),
        pytest.param(lambda s: BBPSOOptimizer(s, direction="max", seed=ANY_SEED), id="bbpso"),
    ],
)
def test_trials_csv_should_hold_one_row_per_trial_with_its_meta_and_score(tmp_path, make):
    """Given each optimizer, when the run ends, then trials.csv has one row per trial whose trial number,
    generation, phase, individual, score and cached flag equal the Trial, with every parameter and x column.

    Oracle: report module docstring (trials.csv) and the Trial docstring (meta).
    """
    study, folder = _run(tmp_path, make(_space()))

    rows = _rows(folder / "run_001" / "trials.csv")

    expected = [
        (
            str(t.number),
            str(t.meta["generation"]),
            t.meta["phase"],
            str(t.meta["individual"]),
            str(int(t.cached)),
        )
        for t in study.trials
    ]
    assert [(r["trial"], r["generation"], r["phase"], r["individual"], r["cached"]) for r in rows] == expected
    assert [float(r["score"]) for r in rows] == pytest.approx([t.score for t in study.trials])
    assert {"lr", "units", "act", "dropout", "x_lr", "x_dropout", "start_time", "duration_s"} <= set(rows[0])


def test_trials_should_carry_a_start_time_and_a_duration(tmp_path):
    """Given a run, when its trials are read, then every real trial has a finite start time and a duration of
    at least 0, and every cached trial a duration of 0.

    Oracle: Trial docstring (start_time in seconds since the epoch, duration 0 for a cached repeat).
    """
    study, _ = _run(tmp_path, PSOOptimizer(_space(), direction="max", seed=ANY_SEED))

    real = [t for t in study.trials if not t.cached]
    cached = [t for t in study.trials if t.cached]

    assert all(math.isfinite(t.start_time) and t.duration >= 0.0 for t in real)
    assert all(t.duration == 0.0 for t in cached)


def test_best_json_should_hold_the_best_trial_and_the_stop_reason(tmp_path):
    """Given a finished run, when best.json is read, then its status is finished, its stop reason is 'budget'
    and its best score, params and trial number are the study's best.

    Oracle: report module docstring (best.json) and the Study docstring (stop_reason 'budget').
    """
    study, folder = _run(tmp_path)

    best = _json(folder / "run_001" / "best.json")

    assert (best["status"], best["stop_reason"]) == ("finished", "budget")
    assert (best["best"]["score"], best["best"]["params"]) == (study.best_score, study.best_params)
    assert best["best"]["trial"] == study.best_trial.number


def test_best_json_should_be_updated_on_disk_after_every_new_best(tmp_path):
    """Given a callback that reads best.json after each trial, when a new best arrives, then the file already
    holds that score.

    Oracle: report module docstring (the best result is on disk while the search runs) and the Callback
    order (ResultWriter is told before the user's callbacks).
    """
    recorder = _Recorder(tmp_path)

    study, _ = _run(tmp_path, callbacks=[recorder])

    news = [t.score for t, seen in zip(study.trials, recorder.best_files, strict=True) if seen is not False]
    assert [s for s in recorder.best_files if s is not False] == news


def test_callbacks_should_receive_the_events_in_order(tmp_path):
    """Given a recording callback, when a run ends, then it saw run start first, one trial event per trial,
    at least one generation event, and run end last.

    Oracle: Callback docstring and the report module docstring (events).
    """
    recorder = _Recorder(tmp_path)

    study, _ = _run(tmp_path, callbacks=[recorder])

    assert (recorder.events[0], recorder.events[-1]) == ("start", "end")
    assert recorder.events.count("trial") == len(study.trials)
    assert recorder.events.count("generation") >= 1


def test_generations_csv_should_count_calls_up_to_the_last_call_of_the_run(tmp_path):
    """Given a run, when generations.csv is read, then the calls column never decreases, its last value is the
    number of real calls, and the best-so-far column never gets worse (maximising).

    Oracle: report module docstring (generations.csv: calls and best so far at the end of each generation).
    """
    study, folder = _run(tmp_path, max_evals=120)

    rows = _rows(folder / "run_001" / "generations.csv")

    calls = [int(r["calls"]) for r in rows]
    best = [float(r["best_so_far"]) for r in rows]
    assert calls == sorted(calls) and calls[-1] == study.n_calls
    assert best == sorted(best)
    assert best[-1] == pytest.approx(study.best_score)


@pytest.mark.parametrize(
    ("make", "extra"),
    [
        pytest.param(lambda s: ABCOptimizer(s, direction="max", seed=ANY_SEED), "stagnation", id="abc"),
        pytest.param(lambda s: PSOOptimizer(s, direction="max", seed=ANY_SEED), "speed", id="pso"),
        pytest.param(lambda s: BBPSOOptimizer(s, direction="max", seed=ANY_SEED), "operation", id="bbpso"),
    ],
)
def test_population_csv_should_hold_the_population_of_every_generation(tmp_path, make, extra):
    """Given each optimizer, when population.csv is read, then every generation listed in generations.csv has
    as many rows as the population has members, with the optimizer-specific column.

    Oracle: report module docstring (population.csv) and Optimizer.population (x, score and optimizer
    columns of length n).
    """
    opt = make(_space())
    _, folder = _run(tmp_path, opt)

    rows = _rows(folder / "run_001" / "population.csv")

    size = len(opt.population()["score"])
    generations = {r["generation"] for r in _rows(folder / "run_001" / "generations.csv")}
    assert len(rows) == size * len(generations)
    assert extra in rows[0]


def test_categories_csv_should_count_every_successful_real_trial_once(tmp_path):
    """Given a run, when categories.csv is read, then the trial counts of the three choices of 'act' add up to
    the real trials that did not fail.

    Oracle: report module docstring (categories.csv).
    """
    study, folder = _run(tmp_path)

    rows = _rows(folder / "run_001" / "categories.csv")

    assert sum(int(r["trials"]) for r in rows) == sum(not t.cached and not t.failed for t in study.trials)
    assert [r["choice"] for r in rows] == ["relu", "tanh", "gelu"]


def test_parameters_csv_should_list_every_parameter_with_the_best_value(tmp_path):
    """Given a run, when parameters.csv is read, then it has one row per parameter in space order with the
    best
    value of that parameter.

    Oracle: report module docstring (parameters.csv).
    """
    study, folder = _run(tmp_path)

    rows = _rows(folder / "run_001" / "parameters.csv")

    assert [r["parameter"] for r in rows] == ["lr", "units", "act", "dropout"]
    assert [r["best_value"] for r in rows] == [str(v) for v in study.best_params.values()]


def test_best_history_should_list_strict_improvements_ending_at_the_best(tmp_path):
    """Given a run, when best_history.csv is read, then its scores strictly increase and end at the best.

    Oracle: report module docstring (best_history.csv: every improvement).
    """
    study, folder = _run(tmp_path)

    scores = [float(r["score"]) for r in _rows(folder / "run_001" / "best_history.csv")]

    assert all(a < b for a, b in zip(scores, scores[1:], strict=False))
    assert scores[-1] == pytest.approx(study.best_score)


def test_summary_and_config_should_describe_the_run(tmp_path):
    """Given a run, when summary.json and config.json are read, then they name the optimizer, the direction,
    the budget, the calls, the stop reason and the four parameters.

    Oracle: report module docstring (config.json and summary.json).
    """
    study, folder = _run(tmp_path)

    summary, config = _json(folder / "summary.json"), _json(folder / "config.json")

    assert (summary["optimizer"], summary["calls"], summary["stop_reason"]) == (
        "ABCOptimizer",
        study.n_calls,
        "budget",
    )
    assert (config["direction"], config["max_evals"]) == ("max", BUDGET)
    assert [p["name"] for p in config["space"]] == ["lr", "units", "act", "dropout"]


def test_report_md_should_name_the_best_score(tmp_path):
    """Given a run, when report.md is read, then it states the stop reason and the best score.

    Oracle: report module docstring (report.md, a human-readable summary).
    """
    study, folder = _run(tmp_path)

    text = (folder / "report.md").read_text(encoding="utf-8")

    assert "Stop reason: **budget**" in text
    assert f"Score: **{study.best_score}**" in text


def test_keyboard_interrupt_should_return_the_study_so_far_and_write_the_results(tmp_path):
    """Given an objective that raises KeyboardInterrupt on its 8th call, when optimize() runs, then it returns
    the 7 trials so far with stop_reason 'interrupted', writes best.json with that reason, and the optimizer
    can still be asked.

    Oracle: optimize docstring ('A KeyboardInterrupt stops the search and returns the study so far') and the
    report module docstring (files are written also after KeyboardInterrupt).
    """
    opt = PSOOptimizer(_space(), direction="max", seed=ANY_SEED)

    study, folder = _run(tmp_path, opt, function=_interrupting(7))

    assert (len(study.trials), study.stop_reason) == (7, "interrupted")
    assert _json(folder / "run_001" / "best.json")["stop_reason"] == "interrupted"
    assert len(opt.ask()) >= 1


def test_error_in_the_objective_should_be_raised_after_the_results_are_written(tmp_path):
    """Given an objective that raises RuntimeError on its 5th call, when optimize() runs, then the error is
    raised, and best.json holds stop reason 'error' with trials.csv listing the 4 finished trials.

    Oracle: optimize docstring ('an exception raised by function is re-raised after the results so far are
    written').
    """
    space = _space()

    with pytest.raises(RuntimeError, match="out of memory"):
        optimize(
            _failing(4), space, direction="max", max_evals=BUDGET, seed=ANY_SEED, verbose=0, output=tmp_path
        )

    run_dir = _folder(tmp_path) / "run_001"
    assert _json(run_dir / "best.json")["stop_reason"] == "error"
    assert len(_rows(run_dir / "trials.csv")) == 4


def test_stop_reason_should_be_exhausted_when_every_parameter_set_was_evaluated(tmp_path):
    """Given Space({'k': Int(0, 2)}) and max_evals 50, when optimize() runs, then the stop reason is
    'exhausted'.

    Oracle: Study docstring (stop_reason 'exhausted') and the optimize docstring (a space without Float stops
    once every set has been evaluated).
    """
    space = Space({"k": Int(0, 2)})

    study = optimize(lambda p: p["k"], space, max_evals=50, seed=ANY_SEED, verbose=0, output=None)

    assert (study.stop_reason, study.n_calls) == ("exhausted", 3)


def test_an_all_nan_run_should_report_that_no_trial_had_a_finite_score(tmp_path):
    """Given an objective that always returns NaN, when the run ends, then best.json has no best and report.md
    says that no trial returned a finite score.

    Oracle: report module docstring and the Study docstring (best_trial needs a finite score).
    """
    _, folder = _run(tmp_path, function=lambda p: math.nan)

    best = _json(folder / "run_001" / "best.json")

    assert best["best"] is None
    assert "No trial returned a finite score." in (folder / "report.md").read_text(encoding="utf-8")


def test_large_runs_should_write_compressed_trials_and_population(tmp_path, monkeypatch):
    """Given COMPRESS_FROM lowered to 10, when a run with max_evals 30 ends, then trials.csv.gz and
    population.csv.gz are written and read back with the same number of trials.

    Oracle: report module docstring (max_evals >= COMPRESS_FROM writes trials.csv and population.csv
    gzip-compressed).
    """
    monkeypatch.setattr(metahpo.report, "COMPRESS_FROM", 10)

    study, folder = _run(tmp_path, max_evals=30)

    run_dir = folder / "run_001"
    assert len(_rows(run_dir / "trials.csv.gz")) == len(study.trials)
    assert (run_dir / "population.csv.gz").is_file() and not (run_dir / "trials.csv").exists()


def test_two_runs_in_the_same_second_should_get_separate_folders(tmp_path):
    """Given two ResultWriters with the same root started in the same second, when both runs end, then two
    folders exist, the second with a numeric suffix.

    Oracle: report module docstring (time-stamped folders are not overwritten).
    """
    space = _space()
    writers = [ResultWriter(tmp_path, plots=False), ResultWriter(tmp_path, plots=False)]
    started = 1_800_000_000.0

    folders = [_start(w, space, started) for w in writers]

    assert folders[0] != folders[1]
    assert folders[1].name.startswith(folders[0].name) and folders[1].name.endswith("_2")


def _start(writer, space, started):
    """Starts `writer` for an ABC run whose state has the given start time; returns its folder."""
    state = metahpo.RunState(
        space, ABCOptimizer(space, seed=ANY_SEED), "min", 10, ANY_SEED, metahpo.Study("min")
    )
    state.started = started
    writer.on_run_start(state)
    writer.close()
    return writer.folder


def test_console_should_print_a_header_one_line_per_generation_and_a_final_line(
    tmp_path, capsys, monkeypatch
):
    """Given verbose=1 and log_every=1, when a run ends, then the output has the run header, one 'gen' line
    per
    generation and the final line with the stop reason.

    Oracle: report module docstring (ConsoleReporter prints one line per generation with verbose=1).
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)

    study = optimize(
        _score, _space(), direction="max", max_evals=BUDGET, seed=ANY_SEED, verbose=1, output=None
    )

    lines = capsys.readouterr().out.splitlines()
    gens = [line for line in lines if line.startswith("gen ")]
    assert lines[0].startswith("[meta-hpo] run 1 | ABCOptimizer")
    assert len(gens) == len({t.meta["generation"] for t in study.trials})
    assert "finished (budget)" in lines[-1]


def test_console_should_print_every_second_generation_with_log_every_two(capsys, monkeypatch):
    """Given verbose=1 and log_every=2, when a run ends, then fewer generation lines are printed than with
    log_every=1, but the last generation is always printed.

    Oracle: report module docstring ('every log_every generations').
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)

    optimize(
        _score,
        _space(),
        direction="max",
        max_evals=BUDGET,
        seed=ANY_SEED,
        verbose=1,
        log_every=2,
        output=None,
    )

    every_second = [line for line in capsys.readouterr().out.splitlines() if line.startswith("gen ")]
    optimize(_score, _space(), direction="max", max_evals=BUDGET, seed=ANY_SEED, verbose=1, output=None)
    every = [line for line in capsys.readouterr().out.splitlines() if line.startswith("gen ")]
    assert len(every_second) < len(every)
    assert every_second[-1].split("|")[0] == every[-1].split("|")[0]


def test_console_should_print_one_line_per_trial_with_verbose_two(capsys, monkeypatch):
    """Given verbose=2, when a run ends, then one 'trial' line per trial is printed.

    Oracle: report module docstring (verbose=2 prints one line per trial as well).
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)

    study = optimize(
        _score, _space(), direction="max", max_evals=BUDGET, seed=ANY_SEED, verbose=2, output=None
    )

    trial_lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("  trial")]
    assert len(trial_lines) == len(study.trials)


def test_console_should_stay_silent_with_verbose_zero(capsys, monkeypatch):
    """Given verbose=0, when a run ends, then nothing is printed.

    Oracle: optimize docstring ('verbose=0 nothing').
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)

    optimize(_score, _space(), direction="max", max_evals=BUDGET, seed=ANY_SEED, verbose=0, output=None)

    assert capsys.readouterr().out == ""


def test_environment_variables_should_replace_the_default_output_and_verbose(tmp_path, capsys, monkeypatch):
    """Given METAHPO_OUTPUT set to tmp_path and METAHPO_VERBOSE=0, when optimize() runs with its defaults,
    then
    the folder is written below tmp_path and nothing is printed.

    Oracle: report module docstring (the environment variables replace the defaults of output and verbose).
    """
    monkeypatch.setenv("METAHPO_OUTPUT", str(tmp_path))
    monkeypatch.setenv("METAHPO_VERBOSE", "0")

    optimize(_score, _space(), direction="max", max_evals=20, seed=ANY_SEED)

    assert (_folder(tmp_path) / "summary.json").is_file()
    assert capsys.readouterr().out == ""


def test_console_reporter_should_reject_log_every_below_one():
    """Given log_every=0, when a ConsoleReporter is built, then ValueError is raised.

    Oracle: ConsoleReporter ('log_every must be at least 1').
    """
    with pytest.raises(ValueError, match="log_every"):
        ConsoleReporter(1, 0)


@pytest.mark.parametrize(
    ("make", "phases"),
    [
        pytest.param(lambda s: ABCOptimizer(s, seed=ANY_SEED), {"init", "employed", "onlooker"}, id="abc"),
        pytest.param(lambda s: PSOOptimizer(s, seed=ANY_SEED), {"init", "fly"}, id="pso"),
        pytest.param(lambda s: BBPSOOptimizer(s, seed=ANY_SEED), {"init", "move"}, id="bbpso"),
    ],
)
def test_candidates_should_carry_their_phase_generation_and_individual(make, phases):
    """Given each optimizer, when 120 calls run, then every trial's meta has a phase, a generation and an
    individual, and the documented phases appear.

    Oracle: Candidate docstring (meta: phase, generation, individual).
    """
    space = _space()

    study = optimize(
        _score, space, max_evals=120, optimizer=make(space), seed=ANY_SEED, verbose=0, output=None
    )

    seen = {t.meta["phase"] for t in study.trials}
    assert all({"phase", "generation", "individual"} <= set(t.meta) for t in study.trials)
    assert phases <= seen


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda s: ABCOptimizer(s, seed=ANY_SEED), id="abc"),
        pytest.param(lambda s: PSOOptimizer(s, seed=ANY_SEED), id="pso"),
        pytest.param(lambda s: BBPSOOptimizer(s, seed=ANY_SEED), id="bbpso"),
    ],
)
def test_population_should_hold_positions_and_scores_of_equal_length(make):
    """Given each optimizer after 60 calls, when population() is read, then x has one row per member with d
    columns and score has the same length; the generation counter has advanced.

    Oracle: Optimizer.population and Optimizer.generation docstrings.
    """
    space = _space()
    opt = make(space)
    optimize(_score, space, max_evals=BUDGET, optimizer=opt, seed=ANY_SEED, verbose=0, output=None)

    population = opt.population()

    assert population["x"].shape == (len(population["score"]), space.dim)
    assert opt.generation >= 1


def test_base_optimizer_should_have_no_population_and_generation_zero():
    """Given the Optimizer base class, when population() and generation are read, then they are None and 0.

    Oracle: Optimizer.population ('None for an optimizer without one') and Optimizer.generation (0).
    """
    opt = metahpo.Optimizer(_space(), seed=ANY_SEED)

    assert (opt.population(), opt.generation) == (None, 0)


class _RandomSearch(metahpo.Optimizer):
    """A minimal optimizer without population, generations or candidate meta: one random candidate per ask."""

    def ask(self):
        return self._fresh(1)

    def tell(self, candidates, scores):
        for cand, score in zip(candidates, scores, strict=True):
            f = self._internal(float(score))
            if f < self.best_f or self.best_x is None:
                self.best_f, self.best_x = f, cand.x.copy()


class _FailingCancel(PSOOptimizer):
    """A PSOOptimizer whose cancel() fails, to show that an interrupt still returns the study."""

    def cancel(self, candidates):
        raise RuntimeError("cancel failed")


class _InterruptAt(Callback):
    """Raises KeyboardInterrupt from on_trial after `count` trials."""

    def __init__(self, count):
        self.count = count

    def on_trial(self, state, trial, new_best):
        if trial.number + 1 == self.count:
            raise KeyboardInterrupt


def test_result_writer_should_handle_an_optimizer_without_population_or_meta(tmp_path):
    """Given an optimizer without population, generation counter or candidate meta, when a run ends, then
    generations.csv has one row with empty population columns, no population.csv is written and the trials
    have empty phase columns.

    Oracle: Optimizer.population (None for an optimizer without one) and the report module docstring.
    """
    space = _space()

    _, folder = _run(tmp_path, _RandomSearch(space, direction="max", seed=ANY_SEED), max_evals=12)

    rows = _rows(folder / "run_001" / "generations.csv")
    assert (len(rows), rows[0]["pop_mean"], rows[0]["diversity_std"]) == (1, "nan", "nan")
    assert not (folder / "run_001" / "population.csv").exists()
    assert {r["phase"] for r in _rows(folder / "run_001" / "trials.csv")} == {""}


def test_result_writer_should_leave_the_time_columns_empty_for_a_trial_without_a_start_time(tmp_path):
    """Given a ResultWriter told a Trial built without a start time, when its row is read, then start_time and
    end_time are empty.

    Oracle: Trial docstring (start_time NaN when unknown) and the report module docstring (trials.csv).
    """
    space = _space()
    writer = ResultWriter(tmp_path, plots=False)
    state = metahpo.RunState(
        space, ABCOptimizer(space, seed=ANY_SEED), "min", 5, ANY_SEED, metahpo.Study("min")
    )
    writer.on_run_start(state)
    trial = metahpo.Trial(
        0, {"lr": 0.001, "units": 64, "act": "relu", "dropout": 0.1}, 0.5, False, np.zeros(4)
    )

    writer.on_trial(state, trial, False)
    writer.close()

    row = _rows(writer.run_dir / "trials.csv")[0]
    assert (row["start_time"], row["end_time"]) == ("", "")


def test_result_writer_should_tolerate_being_closed_twice(tmp_path):
    """Given a finished run, when the writer is closed again, then nothing is raised and best.json stays.

    Oracle: ResultWriter.close docstring (closes the files written while the search runs).
    """
    writer = ResultWriter(tmp_path, plots=False)
    optimize(_score, _space(), max_evals=10, seed=ANY_SEED, verbose=0, output=None, callbacks=[writer])

    writer.close()

    assert (writer.run_dir / "best.json").is_file()


def test_config_json_should_store_a_numpy_integer_seed_as_a_number(tmp_path):
    """Given seed=np.int64(3), when the run ends, then config.json holds the number 3.

    Oracle: report module docstring (config.json holds the seed); JSON has no numpy types.
    """
    optimize(_score, _space(), max_evals=10, seed=np.int64(3), verbose=0, output=tmp_path)

    config = _json(_folder(tmp_path) / "config.json")

    assert config["seed"] == 3


def test_best_json_should_store_a_choice_that_json_cannot_hold_as_its_repr(tmp_path):
    """Given a Categorical whose choices are Python types, when the run ends, then best.json stores the chosen
    type as its repr.

    Oracle: report module docstring (JSON files); a value JSON cannot represent is written as its repr.
    """
    space = Space({"kind": Categorical([int, float])})

    optimize(
        lambda p: {int: 1.0, float: 2.0}[p["kind"]],
        space,
        max_evals=5,
        seed=ANY_SEED,
        verbose=0,
        output=tmp_path,
    )

    best = _json(_folder(tmp_path) / "run_001" / "best.json")
    assert best["best"]["params"]["kind"] == repr(int)


def test_parameters_csv_should_report_nan_correlations_when_fewer_than_three_trials_succeed(tmp_path):
    """Given max_evals 2, when parameters.csv is read, then the correlations of the numeric parameters are
    nan.

    Oracle: report module docstring (parameters.csv); a correlation needs at least three points.
    """
    _, folder = _run(tmp_path, max_evals=2)

    rows = _rows(folder / "run_001" / "parameters.csv")

    numeric = [r for r in rows if r["type"] != "Categorical"]
    assert {r["pearson_with_score"] for r in numeric} == {"nan"}


def test_console_should_say_no_finite_score_when_every_score_is_nan(capsys, monkeypatch):
    """Given an objective that always returns NaN and verbose=1, when the run ends, then the final line says
    'no finite score'.

    Oracle: report module docstring (ConsoleReporter) and the Study docstring.
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)

    optimize(lambda p: math.nan, _space(), max_evals=10, seed=ANY_SEED, verbose=1, output=None)

    assert capsys.readouterr().out.splitlines()[-1].endswith("no finite score")


def test_interrupt_should_return_the_study_even_when_the_optimizer_cannot_cancel(tmp_path):
    """Given a PSOOptimizer whose cancel() raises and an objective interrupted on its 5th call (inside the
    initial batch of 20), when optimize() runs, then it still returns the 4 trials with stop_reason
    'interrupted'.

    Oracle: optimize docstring (a KeyboardInterrupt returns the study so far).
    """
    opt = _FailingCancel(_space(), direction="max", seed=ANY_SEED)

    study, _ = _run(tmp_path, opt, function=_interrupting(4))

    assert (len(study.trials), study.stop_reason) == (4, "interrupted")


def test_interrupt_after_the_last_candidate_of_a_batch_should_tell_the_whole_batch(tmp_path):
    """Given a callback that raises KeyboardInterrupt after the 5th trial, the last of ABC's initial batch,
    when optimize() runs, then the study holds 5 trials and the optimizer has left its initial phase.

    Oracle: optimize docstring (interrupt) and the facade's handling of an unfinished batch (the evaluated
    part is told).
    """
    opt = ABCOptimizer(_space(), direction="max", seed=ANY_SEED)

    study, _ = _run(tmp_path, opt, callbacks=[_InterruptAt(5)])

    assert (len(study.trials), study.stop_reason, opt.phase) == (5, "interrupted", "employed")


class _BrokenStream:
    """A console stream whose every write fails, like a pipe whose reader has gone."""

    def __init__(self):
        self.writes = 0

    def write(self, text):
        self.writes += 1
        raise OSError(22, "Invalid argument")

    def flush(self):
        raise OSError(22, "Invalid argument")


def test_console_should_stop_printing_and_let_the_run_finish_when_the_stream_breaks(tmp_path):
    """Given a ConsoleReporter whose stream raises OSError on every write, when optimize() runs, then the
    study is complete, the results folder holds summary.json and the reporter tried to write only once.

    Oracle: ConsoleReporter docstring (an unwritable console stops the printing; the search and the other
    listeners go on).
    """
    stream = _BrokenStream()

    study, folder = _run(tmp_path, callbacks=[ConsoleReporter(verbose=2, stream=stream)])

    assert (len(study.trials), study.stop_reason) == (BUDGET, "budget")
    assert (folder / "summary.json").is_file()
    assert stream.writes == 1


def test_result_writer_should_write_every_run_into_one_folder_with_a_summary_per_run(tmp_path):
    """Given runs=3, when the runs end, then there is one results folder with run_001 to run_003, each with
    its own summary.json, and config.json records three runs and their seeds.

    Oracle: report module docstring (folder layout with several runs) and the optimize docstring (all runs
    are written to the same results folder).
    """
    _, folder = _run(tmp_path, runs=3)

    names = sorted(p.name for p in folder.iterdir() if p.is_dir())
    assert names == ["run_001", "run_002", "run_003"]
    assert [(folder / n / "summary.json").is_file() for n in names] == [True, True, True]
    config = _json(folder / "config.json")
    assert (config["runs"], config["seeds"]) == (3, [ANY_SEED, ANY_SEED, ANY_SEED])


def test_summary_should_name_the_best_run_and_the_statistics_of_the_run_scores(tmp_path):
    """Given seed=[1, 2, 3] and runs=3, when the runs end, then summary.json names the run with the highest
    best score, lists the three runs with their seeds, and its run_scores mean is the mean of the best
    scores.

    Oracle: report module docstring (summary.json with several runs: runs, best_run, run_scores).
    """
    studies = optimize(
        _score,
        _space(),
        direction="max",
        max_evals=BUDGET,
        seed=[1, 2, 3],
        runs=3,
        verbose=0,
        output=None,
        callbacks=[ResultWriter(tmp_path, plots=False)],
    )

    summary = _json(_folder(tmp_path) / "summary.json")
    bests = [s.best_score for s in studies]
    assert summary["best_run"] == 1 + bests.index(max(bests))
    assert [r["seed"] for r in summary["runs"]] == [1, 2, 3]
    assert summary["run_scores"]["mean"] == pytest.approx(float(np.mean(bests)))


def test_summary_should_name_the_first_run_as_best_when_runs_tie(tmp_path):
    """Given one integer seed and runs=2, so both runs reach the same best score, when the runs end, then
    best_run is 1.

    Oracle: report module docstring (best_run) and the optimize docstring (one integer is used by every run).
    """
    _, folder = _run(tmp_path, runs=2)

    summary = _json(folder / "summary.json")

    assert (summary["best_run"], summary["run_scores"]["std"]) == (1, 0.0)


def test_report_md_should_start_with_a_table_of_the_runs(tmp_path):
    """Given runs=3, when report.md is read, then it has a Runs section that links every run folder.

    Oracle: report module docstring (report.md starts with a table of the runs).
    """
    _, folder = _run(tmp_path, runs=3)

    text = (folder / "report.md").read_text(encoding="utf-8")

    assert "## Runs" in text
    assert ["`run_001`" in text, "`run_002`" in text, "`run_003`" in text] == [True, True, True]


def test_single_run_should_keep_the_single_run_summary(tmp_path):
    """Given the default runs, when the run ends, then summary.json has no runs entry, the run folder has no
    summary.json, and config.json records one run.

    Oracle: report module docstring (the per-run summary and the runs entries exist only with several runs).
    """
    _, folder = _run(tmp_path)

    summary = _json(folder / "summary.json")

    assert ("runs" in summary, (folder / "run_001" / "summary.json").exists()) == (False, False)
    assert _json(folder / "config.json")["runs"] == 1


def test_summary_should_cover_the_runs_so_far_after_an_interrupt(tmp_path):
    """Given runs=3 and an objective interrupted in the second run, when optimize() returns, then
    summary.json lists two runs and the second one stopped as interrupted.

    Oracle: report module docstring (summary.json is rewritten after every run, so it covers the runs so far).
    """
    _, folder = _run(tmp_path, function=_interrupting(BUDGET + 5), runs=3)

    summary = _json(folder / "summary.json")

    assert [r["stop_reason"] for r in summary["runs"]] == ["budget", "interrupted"]


def test_console_should_number_the_runs_and_end_with_a_line_over_all_runs(capsys, monkeypatch):
    """Given runs=2 and verbose=1, when the runs end, then the first line names run 1/2 and the last line
    sums up both runs.

    Oracle: report module docstring (ConsoleReporter) and the RunState docstring (run, runs).
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)

    optimize(_score, _space(), direction="max", max_evals=BUDGET, seed=ANY_SEED, runs=2, output=None)

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("[meta-hpo] run 1/2 |")
    assert lines[-1].startswith("[meta-hpo] 2 runs | best ")


def test_console_should_say_no_finite_score_over_runs_that_never_scored(capsys, monkeypatch):
    """Given runs=2 and an objective that always returns NaN, when the runs end, then the last line says
    that the two runs have no finite score.

    Oracle: report module docstring (ConsoleReporter) and the Study docstring (NaN is no finite score).
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)

    optimize(lambda p: math.nan, _space(), max_evals=10, seed=ANY_SEED, runs=2, output=None)

    assert capsys.readouterr().out.splitlines()[-1] == "[meta-hpo] 2 runs | no finite score"


def test_state_json_should_hold_the_start_time_and_the_generator_state_before_the_first_candidate(tmp_path):
    """Given an ABCOptimizer whose generator state is copied before optimize(), when the run ends, then
    state.json holds that state, a start time no later than the first trial and no pauses.

    Oracle: report module docstring (state.json: the start time of the run, the generator state before the
    first candidate was asked, and pauses).
    """
    opt = ABCOptimizer(_space(), direction="max", seed=ANY_SEED)
    before = json.loads(json.dumps(opt.rng.bit_generator.state))

    study, folder = _run(tmp_path, opt)

    state = _json(folder / "run_001" / "state.json")
    assert (state["generator"], state["pauses"]) == (before, [])
    assert state["started"] <= study.trials[0].start_time


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        pytest.param(
            lambda s: ABCOptimizer(s, "max", colony_size=12, limit=7, seed=ANY_SEED, max_tries=50),
            {"colony_size": 12, "limit": 7, "max_tries": 50},
            id="abc",
        ),
        pytest.param(
            lambda s: PSOOptimizer(s, "max", 5, 0.5, 1.0, 2.0, 0.3, seed=ANY_SEED, max_tries=50),
            {
                "swarm_size": 5,
                "inertia": 0.5,
                "cognitive": 1.0,
                "social": 2.0,
                "max_velocity": 0.3,
                "max_tries": 50,
            },
            id="pso",
        ),
        pytest.param(
            lambda s: BBPSOOptimizer(s, "max", 8, 0.9, 0.25, 0.2, 0.8, seed=ANY_SEED, max_tries=50),
            {
                "swarm_size": 8,
                "target": 0.9,
                "crossover": 0.25,
                "q_alpha": 0.2,
                "q_gamma": 0.8,
                "max_tries": 50,
            },
            id="bbpso",
        ),
    ],
)
def test_config_json_should_record_every_constructor_setting_of_each_optimizer(tmp_path, make, expected):
    """Given each optimizer built with settings other than its defaults, when the run ends, then config.json
    records every constructor setting besides space, direction and seed, the cache flag and no resumes.

    Oracle: report module docstring (config.json records the cache flag and the optimizer with every
    constructor setting; resumes lists every resume) and Optimizer.settings (every further parameter of the
    constructor).
    """
    _, folder = _run(tmp_path, make(_space()), max_evals=12)

    config = _json(folder / "config.json")

    assert config["optimizer"]["settings"] == expected
    assert (config["cache"], config["resumes"]) == (True, [])


class _Stepper(metahpo.Optimizer):
    """A custom optimizer with a setting kept under its own name, one that may be None and one kept under
    another name."""

    def __init__(self, space, direction="min", step=0.1, label=None, seed=None, width=3):
        super().__init__(space, direction, seed)
        self.step = step
        self.label = label
        self._width = width


def test_optimizer_settings_should_read_the_constructor_parameters_kept_under_their_own_names():
    """Given a custom optimizer whose step and label are attributes of the same name and whose width is kept
    as _width, when settings() is read, then it holds step and label, None included, and leaves out width.

    Oracle: Optimizer.settings (every further parameter of the constructor is read from the attribute of
    the same name; a parameter without such an attribute is left out).
    """
    opt = _Stepper(_space(), step=0.5, seed=ANY_SEED)

    settings = opt.settings()

    assert settings == {"step": 0.5, "label": None}


def test_config_json_should_store_json_choices_as_values_and_mark_other_choices_as_repr(tmp_path):
    """Given a Categorical of 1, 1.0, True, None and 'a' and a Categorical of the types int and float, when
    the run ends, then config.json stores the first as values of the same types and the second as repr()
    text marked "repr": true.

    Oracle: report module docstring (choices that are strings, integers, finite floats, booleans or None are
    JSON values; any other Categorical stores the repr() of each choice and is marked "repr": true).
    """
    space = Space({"plain": Categorical([1, 1.0, True, None, "a"]), "kind": Categorical([int, float])})
    optimize(
        lambda p: 1.0,
        space,
        max_evals=4,
        seed=ANY_SEED,
        verbose=0,
        output=None,
        callbacks=[ResultWriter(tmp_path, plots=False)],
    )

    plain, kind = _json(_folder(tmp_path) / "config.json")["space"]

    expected = [(int, 1), (float, 1.0), (bool, True), (type(None), None), (str, "a")]
    assert [(type(c), c) for c in plain["choices"]] == expected
    assert (plain["repr"], kind["repr"], kind["choices"]) == (False, True, [repr(int), repr(float)])


def test_config_json_should_mark_a_categorical_with_a_nan_choice_as_repr(tmp_path):
    """Given a Categorical of 0.5 and NaN, when the run ends, then config.json stores its choices as repr()
    text, because JSON has no NaN.

    Oracle: report module docstring (only finite floats are JSON values).
    """
    space = Space({"value": Categorical([0.5, math.nan])})
    optimize(
        lambda p: 1.0,
        space,
        max_evals=2,
        seed=ANY_SEED,
        verbose=0,
        output=None,
        callbacks=[ResultWriter(tmp_path, plots=False)],
    )

    (value,) = _json(_folder(tmp_path) / "config.json")["space"]

    assert (value["repr"], value["choices"]) == (True, ["0.5", "nan"])


def _special(params):
    """NaN, inf or -inf for the choice of that name, 0.5 for 'ok'."""
    return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf, "ok": 0.5}[params["s"]]


def test_trials_jsonl_should_write_nan_and_infinite_scores_exactly_with_the_exact_start(tmp_path):
    """Given an objective that returns NaN, inf, -inf and 0.5 for four choices, when the run ends, then
    trials.jsonl holds the strings 'nan', 'inf' and '-inf' and the number 0.5, and every start_epoch is the
    trial's start_time.

    Oracle: report module docstring (a NaN or infinite score as the string 'nan', 'inf' or '-inf';
    start_epoch, the exact start in seconds since the epoch).
    """
    space = Space({"s": Categorical(["nan", "inf", "-inf", "ok"])})

    study = optimize(
        _special,
        space,
        max_evals=4,
        seed=ANY_SEED,
        verbose=0,
        output=None,
        callbacks=[ResultWriter(tmp_path, plots=False)],
    )

    lines = (_folder(tmp_path) / "run_001" / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    expected = {"nan": "nan", "inf": "inf", "-inf": "-inf", "ok": 0.5}
    assert {r["params"]["s"]: r["score"] for r in records} == expected
    assert [r["start_epoch"] for r in records] == [t.start_time for t in study.trials]


class _Clock:
    """A fake monotonic clock that moves `step` seconds forward at every reading; the first reads 0."""

    def __init__(self, step):
        self.step = step
        self.now = -step

    def __call__(self):
        self.now += self.step
        return self.now


@pytest.mark.parametrize(
    ("step", "syncs"),
    [
        pytest.param(0.5, 2 + 5 + 1, id="cheap-trials-half-a-second-apart"),
        pytest.param(2.0, 2 + 10 + 1, id="slow-trials"),
    ],
)
def test_trials_jsonl_should_be_synced_once_a_second_and_when_the_run_ends(
    tmp_path, monkeypatch, step, syncs
):
    """Given a fake clock that moves 0.5 s or 2 s per trial, when a run of 10 trials ends, then trials.jsonl
    was synced once per second of the clock (after every second trial, or after every trial) and once more
    when the run ended, besides the two syncs of config.json and state.json at the start.

    Oracle: report module docstring (synced to disk once FSYNC_EVERY seconds have passed since the last
    sync; a training that takes longer is synced after every trial; config.json and state.json are synced
    when a run starts) and FSYNC_EVERY (1 second); the sync at the end comes from ResultWriter.close.
    """
    synced = []
    monkeypatch.setattr(metahpo.report, "_monotonic", _Clock(step))
    monkeypatch.setattr(metahpo.report.os, "fsync", synced.append)
    space = Space({"x": Float(0.0, 1.0), "y": Float(0.0, 1.0)})

    study = optimize(
        lambda p: p["x"],
        space,
        max_evals=10,
        seed=ANY_SEED,
        verbose=0,
        output=None,
        callbacks=[ResultWriter(tmp_path, plots=False)],
    )

    assert len(study.trials) == 10
    assert len(synced) == syncs


class _DiskEvents:
    """Records os.fsync calls and the file names that os.replace puts in place, in order."""

    def __init__(self, replace):
        self.replace_file = replace
        self.events = []

    def fsync(self, fd):
        self.events.append("fsync")

    def replace(self, src, dst):
        self.events.append(dst.name)
        self.replace_file(src, dst)


def test_config_and_state_json_should_be_synced_to_disk_when_a_run_starts_and_best_json_should_not(
    tmp_path, monkeypatch
):
    """Given os.fsync and os.replace recorded and a clock that never moves, when a run of 5 trials ends,
    then config.json and state.json are each synced right before they replace the old file, best.json and
    the summary files are not, and the only other sync is the one of trials.jsonl at the end of the run.

    Oracle: report module docstring (config.json and state.json are synced to disk when a run starts;
    trials.jsonl is synced once FSYNC_EVERY seconds have passed) and the _write_json docstring (the files
    that are rewritten often are not synced).
    """
    disk = _DiskEvents(metahpo.report.os.replace)
    monkeypatch.setattr(metahpo.report, "_monotonic", _Clock(0.0))
    monkeypatch.setattr(metahpo.report.os, "fsync", disk.fsync)
    monkeypatch.setattr(metahpo.report.os, "replace", disk.replace)

    _run(tmp_path, max_evals=5)

    assert disk.events[:5] == ["fsync", "config.json", "fsync", "state.json", "best.json"]
    assert disk.events.count("fsync") == 3


RULES = [
    pytest.param([], [], id="no-rules"),
    pytest.param(
        [TargetScore(0.99), Patience(5, 0.25), MaxTime(minutes=2)],
        [
            {"name": "TargetScore", "module": "metahpo.stop", "settings": {"score": 0.99}},
            {"name": "Patience", "module": "metahpo.stop", "settings": {"evals": 5, "min_delta": 0.25}},
            {
                "name": "MaxTime",
                "module": "metahpo.stop",
                "settings": {"seconds": 0.0, "minutes": 2.0, "hours": 0.0},
            },
        ],
        id="three-rules-in-order",
    ),
]


@pytest.mark.parametrize(("rules", "expected"), RULES)
def test_config_json_should_record_the_stopping_rules_in_order_with_their_settings(tmp_path, rules, expected):
    """Given a run without rules or with three rules in a given order, when config.json is read, then `stop`
    lists every rule in that order with its class name, the module of its class and its settings, and is
    empty without rules.

    Oracle: report module docstring (`stop` lists the stopping rules of the runs in order, each with its class
    name and `settings` (Stopper.settings()), and is empty for a search without rules; each rule also records
    the `module` of its class).
    """
    _, folder = _run(tmp_path, stop=rules)

    config = _json(folder / "config.json")

    assert config["stop"] == expected


@pytest.mark.parametrize(
    ("rules", "ending"),
    [
        pytest.param([], " | seed=0 | 4 parameters", id="no-rules"),
        pytest.param(
            [Patience(50), TargetScore(0.99)],
            " | 4 parameters | stop=Patience(evals=50, min_delta=0.0), TargetScore(score=0.99)",
            id="two-rules",
        ),
    ],
)
def test_console_header_should_name_the_stopping_rules_only_when_the_run_has_some(
    capsys, monkeypatch, rules, ending
):
    """Given a run without rules or with Patience(50) and TargetScore(0.99), when the console header is read,
    then it ends with the rules and their settings, and without rules it ends with the parameter count as
    before.

    Oracle: ConsoleReporter (the header names the stopping rules only when a run has some) and the README
    Results section, whose header line ends with '| seed=1 | 4 parameters'.
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)

    optimize(_score, _space(), direction="max", max_evals=10, seed=ANY_SEED, output=None, stop=rules)

    header = capsys.readouterr().out.splitlines()[0]
    assert header.startswith("[meta-hpo] run 1 | ABCOptimizer | direction=max | max_evals=10 | seed=0")
    assert header.endswith(ending)


@pytest.mark.parametrize(
    ("rules", "lines"),
    [
        pytest.param([], [], id="no-rules"),
        pytest.param([Patience(50)], ["- Stopping rules: Patience(evals=50, min_delta=0.0)"], id="patience"),
    ],
)
def test_report_md_should_name_the_stopping_rules_only_when_the_run_has_some(tmp_path, rules, lines):
    """Given a run without rules or with Patience(50), when report.md is read, then it has a 'Stopping rules'
    line with the rule and its settings only when the run has rules.

    Oracle: the report module docstring (report.md is the readable summary of the configuration) and
    config.json `stop`, which the line repeats.
    """
    _, folder = _run(tmp_path, stop=rules)

    report = (folder / "report.md").read_text(encoding="utf-8").splitlines()

    assert [line for line in report if line.startswith("- Stopping rules")] == lines


class _Rules(Callback):
    """Records the stopping rules each run starts with."""

    def __init__(self):
        self.rules = []

    def on_run_start(self, state):
        self.rules.append(state.rules)


def test_run_state_should_hold_each_runs_own_copies_of_the_rules(tmp_path):
    """Given runs=2 with one Patience rule, when a callback reads state.rules at the start of each run, then
    each run holds a Patience with the given settings that is neither the given object nor the other run's.

    Oracle: RunState docstring (`rules` holds the run's own copies of the stopping rules) and the optimize
    docstring (every run gets its own copy (copy.deepcopy) of the rules).
    """
    given = Patience(50, 0.25)
    recorder = _Rules()

    _run(tmp_path, runs=2, stop=[given], callbacks=[recorder])

    first, second = recorder.rules
    assert [r.settings() for r in first + second] == [given.settings()] * 2
    assert (first[0] is given, second[0] is given, first[0] is second[0]) == (False, False, False)


def test_config_json_should_record_numpy_bounds_as_plain_numbers(tmp_path):
    """Given a space with Int(np.int64(1), np.uint8(9)) and Float(np.float16(0), np.float32(1)), when
    config.json is read, then the space records the bounds as the plain JSON numbers 1, 9, 0.0 and 1.0.

    Oracle: report module docstring (config.json records the space) and the Int and Float docstrings
    (bounds are stored as Python ints and floats).
    """
    space = Space({"a": Int(np.int64(1), np.uint8(9)), "f": Float(np.float16(0), np.float32(1))})
    optimize(lambda p: p["a"] + p["f"], space, max_evals=5, seed=ANY_SEED, verbose=0, output=tmp_path)

    text = (_folder(tmp_path) / "config.json").read_text(encoding="utf-8")

    assert [(p["low"], p["high"]) for p in json.loads(text)["space"]] == [(1, 9), (0.0, 1.0)]
    assert "np." not in text


def _odd_choices_run(tmp_path, choices):
    """optimize() minimising lr on a space of the Categorical `choices` beside a Float, written below
    tmp_path with plots off; returns the study and the results folder."""
    space = Space({"c": Categorical(choices), "lr": Float(0.1, 1.0)})
    study = optimize(
        lambda p: p["lr"],
        space,
        max_evals=40,
        seed=ANY_SEED,
        verbose=0,
        output=None,
        callbacks=[ResultWriter(tmp_path, plots=False)],
    )
    return study, space.parameters[0].choices, _folder(tmp_path)


def _trials_per_choice(study, choices):
    """For every choice, the number of real, finite trials whose params hold that very object."""
    real = [t for t in study.trials if not t.cached and not t.failed]
    return [str(sum(t.params["c"] is choice for t in real)) for choice in choices]


@pytest.mark.parametrize(
    "choices",
    [
        pytest.param([np.array([1.0, 1.0]), np.array([1.0, 5.0]), np.array([1.0, 10.0])], id="numpy-arrays"),
        pytest.param([0, False, 1, True, 1.0], id="equal-across-types"),
    ],
)
def test_categories_csv_should_count_each_choice_by_its_own_trials_for_arrays_and_values_equal_across_types(
    tmp_path, choices
):
    """Given a Categorical of three numpy arrays, or of 0, False, 1, True and 1.0, when the run writes
    categories.csv and parameters.csv, then every choice has one row whose trial count is the number of
    trials that received that very choice, and distinct_values counts the choices used.

    Oracle: the Categorical docstring (any object can be a choice; choices equal across types stay
    different choices) and the report module docstring (categories.csv has one row per category).
    Decisive: matching trials to choices with == raised 'The truth value of an array ... is ambiguous' for
    arrays and counted 1, True and 1.0 together (25 trials each in a run of 40).
    """
    study, kept, folder = _odd_choices_run(tmp_path, choices)

    rows = _rows(folder / "run_001" / "categories.csv")
    parameters = _rows(folder / "run_001" / "parameters.csv")

    assert [row["trials"] for row in rows] == _trials_per_choice(study, kept)
    assert parameters[0]["distinct_values"] == str(sum(n != "0" for n in _trials_per_choice(study, kept)))


NUMPY_SETTINGS = [
    pytest.param(
        lambda s: ABCOptimizer(s, "max", np.int64(8), np.int32(5), seed=ANY_SEED, max_tries=np.int16(50)),
        {"colony_size": 8, "limit": 5, "max_tries": 50},
        id="abc",
    ),
    pytest.param(
        lambda s: PSOOptimizer(
            s, "max", np.int64(7), np.float32(0.5), np.float64(1.5), 2, np.float16(0.5), ANY_SEED, np.int8(9)
        ),
        {
            "swarm_size": 7,
            "inertia": 0.5,
            "cognitive": 1.5,
            "social": 2.0,
            "max_velocity": 0.5,
            "max_tries": 9,
        },
        id="pso",
    ),
    pytest.param(
        lambda s: BBPSOOptimizer(
            s, "max", np.int32(6), np.float32(0.5), 1, np.float64(0.25), 0, ANY_SEED, np.uint8(7)
        ),
        {"swarm_size": 6, "target": 0.5, "crossover": 1.0, "q_alpha": 0.25, "q_gamma": 0.0, "max_tries": 7},
        id="bbpso",
    ),
]


@pytest.mark.parametrize(("make", "expected"), NUMPY_SETTINGS)
def test_config_json_should_record_numpy_and_int_settings_as_plain_python_numbers(tmp_path, make, expected):
    """Given each optimizer built with numpy integers for its integer settings and numpy floats or ints for
    its float settings, and Patience(np.int64(30), np.float32(0.5)), when config.json is read, then every
    setting is the plain JSON number of the same value, integers as ints and float settings as floats.

    Oracle: Optimizer.settings() and Stopper.settings() (config.json records the constructor settings),
    the stage brief (integer settings stored as int, float settings as plain Python numbers). Decisive: a
    numpy integer max_tries was kept as np.int64, so json.dumps(optimizer.settings()) raised TypeError.
    """
    optimizer = make(_space())
    _run(tmp_path, optimizer, max_evals=8, stop=[Patience(np.int64(30), np.float32(0.5))])

    config = _json(_folder(tmp_path) / "config.json")

    settings = config["optimizer"]["settings"]
    assert settings == expected
    assert [type(v) for v in settings.values()] == [type(v) for v in expected.values()]
    assert config["stop"][0]["settings"] == {"evals": 30, "min_delta": 0.5}
    assert json.loads(json.dumps(optimizer.settings())) == expected


def test_console_reporter_should_reject_a_log_every_that_is_no_integer():
    """Given log_every 2.0, when ConsoleReporter is built, then ValueError names log_every.

    Oracle: ConsoleReporter (one line every `log_every` generations) and the library's error convention;
    a whole float is no integer. Decisive: 2.0 was taken as it was.
    """
    with pytest.raises(ValueError, match="log_every") as excinfo:
        ConsoleReporter(verbose=1, log_every=2.0)

    assert excinfo.type is ValueError


class _NumpySettings(_RandomSearch):
    """A custom random search whose step and label settings are kept as the values given, numpy ones too."""

    def __init__(self, space, direction="min", step=0.1, label=None, seed=None):
        super().__init__(space, direction, seed)
        self.step = step
        self.label = label


def test_config_json_should_record_numpy_settings_of_a_custom_optimizer_as_plain_numbers(tmp_path):
    """Given a custom optimizer whose step setting is np.float32(0.5) and whose label is np.int64(3), when
    config.json is read, then it records the plain JSON numbers 0.5 and 3 and holds no numpy text.

    Oracle: report module docstring (config.json records the optimizer with every constructor setting) and
    the stage brief (nothing numpy-typed reaches config.json).
    """
    optimizer = _NumpySettings(_space(), "max", step=np.float32(0.5), label=np.int64(3), seed=ANY_SEED)
    _, folder = _run(tmp_path, optimizer, max_evals=3)

    text = (folder / "config.json").read_text(encoding="utf-8")

    assert json.loads(text)["optimizer"]["settings"] == {"step": 0.5, "label": 3}
    assert "np." not in text


class _NumpyMeta(_RandomSearch):
    """A custom random search whose candidates carry numpy scalars and an array in their meta and whose
    population holds numpy scores, a numpy integer column and an array column."""

    def ask(self):
        meta = {
            "generation": np.int64(0),
            "phase": "draw",
            "individual": np.int32(2),
            "weight": np.float32(0.5),
            "flag": np.bool_(True),
            "shape": np.array([1, 2]),
        }
        return [metahpo.Candidate(c.id, c.x, c.params, c.key, meta) for c in self._fresh(1)]

    def population(self):
        return {
            "x": np.zeros((2, self.space.dim)),
            "score": [np.float32(1.0), 2.0],
            "age": [np.int64(1), np.int64(2)],
            "tag": [np.array([1, 2]), np.array([3, 4])],
        }


def _numpy_meta_run(tmp_path):
    """A run of _NumpyMeta on a Float beside a Categorical of numpy arrays, written below tmp_path."""
    weights = [np.array([1.0, 5.0]), np.array([2.0, 3.0])]
    space = Space({"a": Float(0.0, 1.0), "w": Categorical(weights)})
    optimizer = _NumpyMeta(space, "max", seed=ANY_SEED)
    optimize(lambda p: p["a"], space, "max", 3, optimizer=optimizer, verbose=0, output=tmp_path)
    return _folder(tmp_path)


def test_results_files_should_write_numpy_values_of_the_meta_as_the_plain_numbers_and_lists_they_hold(
    tmp_path,
):
    """Given a custom optimizer whose candidate meta holds np.int64, np.int32, np.float32, np.bool_ and a
    numpy array, whose population holds numpy values and whose space has a Categorical of numpy arrays, when
    the run writes its folder, then trials.jsonl and trials.csv hold the meta as plain JSON numbers, a
    boolean and a list, population.csv the extra columns as numbers and list text, and no results file
    but config.json (which records the Categorical as repr text, see the next test) holds numpy text.

    Oracle: report module docstring (trials.jsonl holds each trial's meta and params; every file is
    derived from the trials) and the stage brief (numpy scalars as the plain numbers they hold, arrays as
    lists wherever a value goes to JSON or CSV). Decisive: json.dumps of the meta for trials.csv raised
    'Object of type float32 is not JSON serializable', and arrays were written as 'array([1., 5.])' text.
    """
    folder = _numpy_meta_run(tmp_path)

    record = json.loads((folder / "run_001" / "trials.jsonl").read_text(encoding="utf-8").splitlines()[0])
    row = _rows(folder / "run_001" / "trials.csv")[0]
    population = _rows(folder / "run_001" / "population.csv")[0]
    files = [p for p in folder.rglob("*") if p.suffix in (".json", ".jsonl", ".csv", ".md")]
    texts = [p.read_text(encoding="utf-8") for p in files if p.name != "config.json"]

    assert (record["generation"], record["individual"]) == (0, 2)
    assert record["detail"] == {"weight": 0.5, "flag": True, "shape": [1, 2]}
    assert json.loads(row["detail"]) == {"weight": 0.5, "flag": True, "shape": [1, 2]}
    assert (population["age"], population["tag"]) == ("1", "[1, 2]")
    assert [("np." in t or "array(" in t) for t in texts] == [False] * len(texts)


def test_config_json_should_still_record_a_categorical_of_numpy_arrays_as_marked_repr_text(tmp_path):
    """Given the same run, when config.json is read, then its Categorical of numpy arrays is still recorded
    as repr() text marked "repr": true, while trials.jsonl writes the chosen array as a list.

    Oracle: report module docstring (any Categorical whose choices are not all JSON values stores the
    repr() of each choice and is marked "repr": true) and the stage brief (writing arrays as lists must not
    change how a choice is recorded in config.json).
    """
    folder = _numpy_meta_run(tmp_path)

    config = _json(folder / "config.json")
    record = json.loads((folder / "run_001" / "trials.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert config["space"][1] == {
        "name": "w",
        "type": "Categorical",
        "choices": ["array([1., 5.])", "array([2., 3.])"],
        "repr": True,
    }
    assert record["params"]["w"] in ([1.0, 5.0], [2.0, 3.0])


@pytest.mark.parametrize(
    ("make", "module"),
    [
        pytest.param(lambda s: ABCOptimizer(s, "max", seed=ANY_SEED), "metahpo.optimizers.abc", id="abc"),
        pytest.param(lambda s: PSOOptimizer(s, "max", seed=ANY_SEED), "metahpo.optimizers.pso", id="pso"),
        pytest.param(
            lambda s: BBPSOOptimizer(s, "max", seed=ANY_SEED), "metahpo.optimizers.bbpso", id="bbpso"
        ),
        pytest.param(lambda s: _RandomSearch(s, "max", ANY_SEED), __name__, id="custom"),
    ],
)
def test_config_json_should_record_the_module_of_the_optimizer_class(tmp_path, make, module):
    """Given each library optimizer and a custom one, when config.json is read, then the optimizer record
    holds the module of its class next to its name.

    Oracle: report module docstring (config.json records the optimizer with every constructor setting and
    the module of its class, so resume() builds again only the library's own optimizers).
    """
    optimizer = make(_space())
    _, folder = _run(tmp_path, optimizer, max_evals=4)

    record = _json(folder / "config.json")["optimizer"]

    assert (record["name"], record["module"]) == (type(optimizer).__name__, module)


def _moving(where, function=_score, at=3):
    """`function` as an objective that changes the working directory to `where` on call `at`, as a trainer
    that moves into its data folder does."""
    calls = [0]

    def objective(params):
        calls[0] += 1
        if calls[0] == at:
            where[0].chdir(where[1])
        return function(params)

    return objective


@pytest.mark.parametrize(
    ("output", "env", "below"),
    [
        pytest.param("rel_out", None, "rel_out", id="relative-output"),
        pytest.param(metahpo.report.DEFAULT_OUTPUT, "env_out", "env_out", id="metahpo-output-path"),
        pytest.param(
            metahpo.report.DEFAULT_OUTPUT, metahpo.report.DEFAULT_OUTPUT, "hpo_results", id="default"
        ),
    ],
)
def test_results_folder_should_stay_where_the_search_started_when_the_objective_changes_directory(
    tmp_path, monkeypatch, output, env, below
):
    """Given the working directory tmp_path, a relative results folder (an output argument, METAHPO_OUTPUT
    or the default 'hpo_results') and an objective that moves into tmp_path/elsewhere on its 3rd call, when
    optimize() runs 20 evaluations, then the run finishes, every file is written below tmp_path/<folder>,
    nothing below elsewhere, and config.json records the folder as an absolute path.

    Oracle: report module docstring (a time-stamped folder below `output`, default 'hpo_results' in the
    working directory) and the ResultWriter._make_folder docstring (the folder becomes an absolute path when
    the search starts). Decisive: the writer kept the relative path, so the next file it opened after the
    chdir raised FileNotFoundError and four files stayed open.
    """
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("METAHPO_OUTPUT", env or "none")
    objective = _moving((monkeypatch, tmp_path / "elsewhere"))

    study = optimize(objective, _space(), "max", 20, seed=ANY_SEED, verbose=0, output=output)

    folder = _folder(tmp_path / below)
    config = _json(folder / "config.json")
    assert (study.stop_reason, (folder / "summary.json").is_file()) == ("budget", True)
    assert list((tmp_path / "elsewhere").iterdir()) == []
    assert config["folder"] == str(folder.absolute())


class _Refusing:
    """A stand-in for metahpo.report._os_replace that refuses, with PermissionError, the replace of the file
    named `name` while `refuse(text)` is true for the new content, or the first `times` attempts when given;
    other files are replaced as usual."""

    def __init__(self, name, refuse=None, times=None):
        self.name = name
        self.refuse = refuse
        self.times = times
        self.refused = 0

    def __call__(self, source, target):
        text = source.read_text(encoding="utf-8")
        hit = target.name == self.name and (
            self.refuse(text) if self.refuse is not None else self.refused < self.times
        )
        self.refused += hit
        hit and _raise_permission(target)
        source.replace(target)


def _raise_permission(target):
    raise PermissionError(13, "another process holds the file", str(target))


def _refused_run(tmp_path, monkeypatch, refusing):
    """A run with plots off whose JSON replaces go through `refusing`; returns the recorded waits."""
    waits = []
    monkeypatch.setattr(metahpo.report, "_os_replace", refusing)
    monkeypatch.setattr(metahpo.report, "_wait", waits.append)
    return waits


def _temporary_files(root):
    return [p for p in root.rglob("*.tmp")]


def test_best_json_should_be_skipped_while_another_process_holds_it_and_written_when_the_run_ends(
    tmp_path, monkeypatch
):
    """Given a reader that holds best.json for every 'running' update, so its replace is always refused,
    when a run of 30 evaluations goes on, then the search is not stopped, every refused update is tried
    PROGRESS_TRIES times PROGRESS_WAIT seconds apart and then left out, the final best.json is written with
    status 'finished', and no temporary file is left.

    Oracle: _write_json docstring (a file that is not required, best.json while the run goes on, is left as
    it was after PROGRESS_TRIES attempts and written again at the next update and when the run ends; the
    temporary file is removed). Decisive: os.replace raised PermissionError and ended the search.
    """
    refusing = _Refusing("best.json", refuse=lambda text: '"running"' in text)
    waits = _refused_run(tmp_path, monkeypatch, refusing)

    study, folder = _run(tmp_path, max_evals=30)

    best = _json(folder / "run_001" / "best.json")
    tries = metahpo.report.PROGRESS_TRIES
    assert (study.stop_reason, best["status"], best["best"]["trial"]) == (
        "budget",
        "finished",
        study.best_trial.number,
    )
    assert refusing.refused > 0 and refusing.refused % tries == 0
    assert waits == [metahpo.report.PROGRESS_WAIT] * (refusing.refused // tries * (tries - 1))
    assert _temporary_files(tmp_path) == []


@pytest.mark.parametrize("name", ["config.json", "state.json", "best.json", "summary.json"])
def test_a_json_file_should_be_written_when_a_reader_releases_it_within_the_retries(
    tmp_path, monkeypatch, name
):
    """Given a reader that holds config.json, state.json, best.json or summary.json for the first 3 attempts
    of its replace, when the run of 12 evaluations ends, then the file is written and the search waited
    between the refused attempts.

    Oracle: _write_json docstring (a refused replace is tried REPLACE_TRIES times, REPLACE_WAIT seconds apart,
    or PROGRESS_TRIES times for best.json while the run goes on).
    """
    refusing = _Refusing(name, times=3)
    waits = _refused_run(tmp_path, monkeypatch, refusing)

    study, folder = _run(tmp_path, max_evals=12)

    written = [p.name for p in folder.rglob(name)]
    assert (study.stop_reason, written, refusing.refused) == ("budget", [name], 3)
    assert len(waits) >= 2 and _temporary_files(tmp_path) == []


@pytest.mark.parametrize("name", ["config.json", "state.json"])
def test_a_file_resume_depends_on_should_stop_the_search_before_any_call_when_it_stays_held(
    tmp_path, monkeypatch, name
):
    """Given a reader that holds config.json or state.json for good, when optimize() starts, then
    PermissionError is raised after REPLACE_TRIES attempts, the objective is never called, no temporary file
    is left and no file stays open.

    Oracle: _write_json docstring (the error of a required file is raised after the attempts; the temporary
    file is removed) and the report module docstring (config.json and state.json are what a resume needs).
    """
    refusing = _Refusing(name, refuse=lambda text: True)
    waits = _refused_run(tmp_path, monkeypatch, refusing)
    calls = []

    with pytest.raises(PermissionError):
        _run(tmp_path, max_evals=12, function=lambda p: calls.append(p) or 0.0)

    assert (calls, refusing.refused, len(waits)) == (
        [],
        metahpo.report.REPLACE_TRIES,
        metahpo.report.REPLACE_TRIES - 1,
    )
    assert _temporary_files(tmp_path) == []


def test_a_held_summary_json_should_raise_after_every_other_result_of_the_run_was_written(
    tmp_path, monkeypatch
):
    """Given a reader that holds summary.json for good, when a run of 12 evaluations ends, then
    PermissionError is raised, while best.json (finished), parameters.csv, categories.csv and report.md are
    written and no temporary file is left.

    Oracle: ResultWriter.on_run_end docstring (a JSON file that cannot be replaced does not stop the others;
    its error is raised once every file was written).
    """
    refusing = _Refusing("summary.json", refuse=lambda text: True)
    _refused_run(tmp_path, monkeypatch, refusing)

    with pytest.raises(PermissionError):
        _run(tmp_path, max_evals=12)

    folder = _folder(tmp_path)
    written = [
        (folder / n).is_file() for n in ("report.md", "run_001/parameters.csv", "run_001/categories.csv")
    ]
    assert (written, _json(folder / "run_001" / "best.json")["status"]) == ([True, True, True], "finished")
    assert (folder / "summary.json").exists() is False and _temporary_files(tmp_path) == []


class _Raising(Callback):
    """A callback that raises `error` in `event` at trial `at` (on_trial), generation `at` (on_generation)
    or at once (on_run_start, on_run_end)."""

    def __init__(self, event, error, at=7):
        self.event = event
        self.error = error
        self.at = at

    def on_run_start(self, state):
        self._raise_for("on_run_start", 0)

    def on_trial(self, state, trial, new_best):
        self._raise_for("on_trial", trial.number)

    def on_generation(self, state, generation, trials, population):
        self._raise_for("on_generation", generation)

    def on_run_end(self, state):
        self._raise_for("on_run_end", self.at)

    def _raise_for(self, event, number):
        (event, number >= self.at) == (self.event, True) and self._throw()

    def _throw(self):
        raise self.error


class _Ends(Callback):
    """Records the runs whose end it received."""

    def __init__(self):
        self.ended = []

    def on_run_end(self, state):
        self.ended.append(state.study.stop_reason)


def _callback_run(tmp_path, callbacks, max_evals=40):
    """optimize() maximising _score with plots off and the given callbacks after the writer."""
    return optimize(
        _score,
        _space(),
        "max",
        max_evals,
        seed=ANY_SEED,
        verbose=0,
        output=None,
        callbacks=[ResultWriter(tmp_path, plots=False), *callbacks],
    )


@pytest.mark.parametrize(
    ("event", "at", "ended"),
    [
        pytest.param("on_run_start", 0, [], id="on-run-start"),
        pytest.param("on_trial", 7, ["error"], id="on-trial"),
        pytest.param("on_generation", 1, ["error"], id="on-generation"),
    ],
)
def test_a_raising_callback_should_end_the_run_like_an_error_of_the_objective_and_leave_a_resumable_folder(
    tmp_path, event, at, ended
):
    """Given a callback that raises a RuntimeError in on_run_start, on_trial at trial 7 or on_generation at
    generation 1, followed by a callback that records the end of the run, when optimize() runs, then the
    same exception object is raised, best.json and summary.json record the stop reason 'error', the
    recording callback received the end of the run (unless on_run_start failed before it was started), and
    resume() of the folder gives the uninterrupted run.

    Oracle: optimize docstring (an exception raised by `function` is re-raised after the results so far are
    written) applied to callbacks by the _Run.finish docstring (every listener receives the end of the run,
    the original exception counts), the _Run.start docstring (a listener after the one whose on_run_start
    raised is not started) and the resume docstring (a run that stopped for an error continues).
    Decisive: the library's writer never received on_run_end when the callback raised again for the last
    generation, so the summary files were missing.
    """
    error = RuntimeError("bug in my callback")
    ends = _Ends()

    with pytest.raises(RuntimeError) as raised:
        _callback_run(tmp_path, [_Raising(event, error, at), ends])

    folder = _folder(tmp_path)
    reasons = (
        _json(folder / "run_001" / "best.json")["stop_reason"],
        _json(folder / "summary.json")["stop_reason"],
    )
    resumed = metahpo.resume(_score, folder, verbose=0)
    reference = optimize(_score, _space(), "max", 40, seed=ANY_SEED, verbose=0, output=None)
    assert raised.value is error
    assert reasons == ("error", "error")
    assert ends.ended == ended
    assert [t.x.tolist() for t in resumed.trials] == [t.x.tolist() for t in reference.trials]


def test_a_keyboard_interrupt_in_a_callback_should_end_the_run_as_interrupted_with_its_results(tmp_path):
    """Given a callback that raises KeyboardInterrupt in every on_generation from generation 1 on, also in the
    one for the last unfinished generation, when optimize() runs, then it returns the study with stop_reason
    'interrupted' and best.json and summary.json record it.

    Oracle: optimize docstring (a KeyboardInterrupt stops the search and returns the study so far with
    stop_reason 'interrupted'; the results are written). Decisive: the interrupt raised again while the
    last generation was sent, so the writer never finished and the interrupt escaped optimize().
    """
    study = _callback_run(tmp_path, [_Raising("on_generation", KeyboardInterrupt(), at=1)])

    folder = _folder(tmp_path)
    assert (study.stop_reason, _json(folder / "summary.json")["stop_reason"]) == (
        "interrupted",
        "interrupted",
    )
    assert _json(folder / "run_001" / "best.json")["status"] == "finished"


def test_an_on_run_end_that_raises_should_not_hide_the_first_error_or_stop_the_writer(tmp_path):
    """Given a callback that raises 'first' in on_trial at trial 7 and 'second' in its own on_run_end, and a
    callback after it that records the end of the run, when optimize() runs, then 'first' is raised, the
    writer finished (summary.json records 'error') and the recording callback received the end.

    Oracle: _Run.finish docstring (every listener receives the end of the run also when some raise; after
    an error the original exception is the one that counts). Decisive: the error of on_run_end replaced the
    original one.
    """
    first, second = RuntimeError("first"), RuntimeError("second")

    class _Both(Callback):
        def on_trial(self, state, trial, new_best):
            trial.number == 7 and _Raising("x", first)._throw()

        def on_run_end(self, state):
            raise second

    ends = _Ends()

    with pytest.raises(RuntimeError) as raised:
        _callback_run(tmp_path, [_Both(), ends])

    assert (raised.value, ends.ended) == (first, ["error"])
    assert _json(_folder(tmp_path) / "summary.json")["stop_reason"] == "error"


def test_an_on_run_end_that_raises_in_a_finished_run_should_raise_after_the_writer_finished(tmp_path):
    """Given a callback whose on_run_end raises when a run of 20 evaluations ends by its budget, when
    optimize() runs, then that exception is raised and summary.json records the stop reason 'budget'.

    Oracle: _Run.finish docstring (the first exception of a listener is raised once every listener received
    the end of the run).
    """
    error = RuntimeError("bug at the end")

    with pytest.raises(RuntimeError) as raised:
        _callback_run(tmp_path, [_Raising("on_run_end", error, at=0)], max_evals=20)

    assert raised.value is error
    assert _json(_folder(tmp_path) / "summary.json")["stop_reason"] == "budget"


def test_a_writer_should_claim_the_next_free_name_when_another_search_takes_its_name_after_a_check(
    tmp_path, monkeypatch
):
    """Given a folder name of this second already created by another search, and an exists() check that
    still answers False, as it does for a search that checked a moment before the other one created the
    folder, when a writer claims its folder, then it takes the name with the suffix _2 and raises nothing.

    Oracle: ResultWriter._make_folder docstring (the name is claimed with one mkdir, which fails when it
    exists; a taken name gets the suffix _2, _3, ...). Decisive: checking for the name and creating it were
    two steps, so searches that started in the same second in threads or processes raised FileExistsError.
    """
    space = _space()
    first = _start(ResultWriter(tmp_path, plots=False), space, 1_800_000_000.0)

    with monkeypatch.context() as patch:
        patch.setattr(type(tmp_path), "exists", lambda self, **kwargs: False)
        second = _start(ResultWriter(tmp_path, plots=False), space, 1_800_000_000.0)

    assert second.name == first.name + "_2"


class _Sleeps:
    """A stand-in for the time module of metahpo.report that records the seconds asked to sleep."""

    def __init__(self):
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)


def test_wait_between_two_replace_attempts_should_sleep_the_given_seconds(monkeypatch):
    """Given the time module of metahpo.report replaced by a recorder, when _wait(REPLACE_WAIT) runs, then it
    sleeps REPLACE_WAIT seconds once.

    Oracle: the _wait docstring (sleeps between two attempts to replace a JSON file) and REPLACE_WAIT.
    """
    recorder = _Sleeps()
    monkeypatch.setattr(metahpo.report, "time", recorder)

    metahpo.report._wait(metahpo.report.REPLACE_WAIT)

    assert recorder.slept == [metahpo.report.REPLACE_WAIT]


def test_a_keyboard_interrupt_in_on_run_end_of_a_finished_run_should_return_the_runs_so_far(tmp_path):
    """Given runs=2 and a callback whose on_run_end raises KeyboardInterrupt when the first run ended by its
    budget, when optimize() runs, then it returns the first study with stop_reason 'budget' and does not
    start the second run.

    Oracle: optimize docstring (a KeyboardInterrupt stops the remaining runs as well and the studies so far
    are returned) and the _Run.finish docstring (every listener receives the end of the run first).
    """
    studies = optimize(
        _score,
        _space(),
        "max",
        12,
        seed=ANY_SEED,
        verbose=0,
        runs=2,
        output=None,
        callbacks=[ResultWriter(tmp_path, plots=False), _Raising("on_run_end", KeyboardInterrupt(), at=0)],
    )

    assert [s.stop_reason for s in studies] == ["budget"]
    assert (_folder(tmp_path) / "run_002").exists() is False
