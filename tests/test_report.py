"""Unit tests of the reporting: ConsoleReporter, ResultWriter, the run events and the stop reasons.

Oracles: the module docstring of metahpo.report (events, console levels, folder layout, files written while
the search runs, environment overrides, compression from COMPRESS_FROM), the docstrings of Callback,
RunState, Trial (meta, start_time, duration) and Study (stop_reason), Optimizer.population and
Optimizer.generation, and the optimize docstring (verbose, output, callbacks, KeyboardInterrupt and errors).

Every run writes into pytest's tmp_path; the suite switches the default folder and console off through
METAHPO_OUTPUT and METAHPO_VERBOSE in conftest, and the console tests remove METAHPO_VERBOSE first. Plots are
switched off here (ResultWriter(plots=False) or through a stub) and tested in test_plots.py.

Unspecified, therefore not asserted: exact float formatting of the console lines, timings, folder names
beyond the time stamp and optimizer suffix.

Input space: Every objective, space, optimizer, budget and output setting, which is unbounded.
Subset: The three optimizers on a four-parameter mixed space with small budgets; console levels 0, 1 and 2;
log_every 1 and 2; objectives that stop with each stop reason (budget, exhausted, interrupted, error); an
all-NaN objective; a budget above COMPRESS_FROM after lowering it.
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
    PSOOptimizer,
    ResultWriter,
    Space,
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
    report.md and run_001 with trials.csv, trials.jsonl, best.json, best_history.csv, generations.csv,
    population.csv, parameters.csv, categories.csv and a plots folder.

    Oracle: report module docstring (folder layout).
    """
    expected = {
        "config.json",
        "summary.json",
        "report.md",
        "run_001/trials.csv",
        "run_001/trials.jsonl",
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
