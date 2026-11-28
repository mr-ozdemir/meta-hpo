"""Unit tests of pruned trials in the reporting: the console, trials.csv, generations.csv, intermediate.csv,
summary.json, report.md, the parameter analysis and the learning-curve plot.

Oracles: the module docstring of metahpo.report (trials.csv marks pruned trials in `pruned`, generations.csv
counts them, summary.json and report.md count completed, pruned and failed trials and `reported_steps`,
intermediate.csv holds one row per reported value of every evaluated trial, the score statistics and the
parameter analysis leave pruned trials out; the console names the step a trial was pruned at and counts
pruned trials when there are any), the plots module docstring (12_learning_curves when trials reported
values; score plots leave pruned scores out) and the resume docstring (every other file of a run folder is
written again from the start).

The searches prune by script: the objective reports the falling loss of _curve at steps 0 to 3 and a pruner
of the test prunes every trial of an odd number after its first report, so the expected counts follow from
the trial numbers; the initial population is never pruned. Every run writes into tmp_path, with plots off
unless a test draws them.

Unspecified, therefore not asserted: the pixels of the learning-curve plot, float formatting beyond what
the console and report.md lines contain, and the column order of the files beyond the new columns.

Input space: Every pruning search with any optimizer, budget and reported values, which is unbounded.
Subset: ABC with two food sources and PSO with six particles on a two-parameter Float space, budgets of 12 to
24 calls, odd trials pruned at step 0, a pruned trial reporting a value far better than every completed one,
cached repeats in a four-value Int space, an interrupted and resumed run, and a run whose objective reports
nothing.
"""

import contextlib
import csv
import io
import json

import pytest

from metahpo import (
    ABCOptimizer,
    Float,
    NopPruner,
    Optimizer,
    Pruner,
    PSOOptimizer,
    Space,
    TrialPruned,
    optimize,
    resume,
)

ANY_SEED = 0

STEPS = 4


@pytest.fixture
def no_plots(monkeypatch):
    """Replaces make_plots and make_run_plots with stubs that draw nothing."""
    monkeypatch.setattr("metahpo.plots.make_plots", _no_plots)
    monkeypatch.setattr("metahpo.plots.make_run_plots", _no_plots)


def _no_plots(*args, **kwargs):
    return []


def _space():
    """A fresh two-parameter Float space."""
    return Space({"x": Float(-5.0, 5.0), "y": Float(-5.0, 5.0)})


def _sphere(params):
    return params["x"] ** 2 + params["y"] ** 2


def _curve(params, step):
    """A loss that falls by one per step towards the sphere value."""
    return _sphere(params) + (STEPS - 1 - step)


def _learning(params, trial):
    """Reports the falling loss at steps 0 to 3, asking the pruner after each report."""
    for step in range(STEPS):
        trial.report(_curve(params, step), step)
        if trial.should_prune():
            raise TrialPruned()
    return _sphere(params)


def _plain(params):
    return _sphere(params)


class _Odd(Pruner):
    """Prunes every trial of an odd number."""

    def prune(self, study, trial):
        return trial.number % 2 == 1


def _abc(space):
    return ABCOptimizer(space, colony_size=4, seed=ANY_SEED)


def _run(tmp_path, function=_learning, budget=12, verbose=0, optimizer=_abc, pruner=None):
    """A minimising search written below tmp_path; returns the study and its results folder."""
    space = _space()
    study = optimize(
        function,
        space,
        max_evals=budget,
        optimizer=optimizer(space),
        output=tmp_path,
        verbose=verbose,
        pruner=_Odd() if pruner is None else pruner,
    )
    return study, next(p for p in tmp_path.iterdir() if p.is_dir())


def _table(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_trials_csv_should_mark_exactly_the_pruned_trials_in_a_column_after_failed(tmp_path, no_plots):
    """Given a search whose odd trials after the initial population are pruned, when trials.csv is read, then
    its header has `pruned` between `failed` and `cached`, and the column holds 1 exactly for the pruned
    trials.

    Oracle: report module docstring (trials.csv marks pruned trials in the column `pruned`). Hand count: ABC
    with two food sources protects trials 0 and 1, so trials 3, 5, 7, 9 and 11 are pruned.
    """
    study, folder = _run(tmp_path)

    rows = _table(folder / "run_001" / "trials.csv")
    header = list(rows[0])
    assert header[header.index("failed") + 1 : header.index("failed") + 3] == ["pruned", "cached"]
    assert [int(r["trial"]) for r in rows if r["pruned"] == "1"] == [3, 5, 7, 9, 11]
    assert [t.number for t in study.trials if t.pruned] == [3, 5, 7, 9, 11]


def test_generations_csv_should_count_pruned_trials_and_leave_their_scores_out_of_the_statistics(
    tmp_path, no_plots
):
    """Given a PSO search of six particles whose odd trials after the initial population are pruned, when
    generations.csv is read, then each row counts the pruned trials of its generation and its gen_best is the
    best completed score of the generation, although every pruned score is the higher loss of step 0.

    Oracle: report module docstring (generations.csv counts pruned trials per generation; the score statistics
    leave pruned trials out). The expected rows are counted from the study by the generation in each trial's
    meta.
    """
    study, folder = _run(
        tmp_path, budget=24, optimizer=lambda space: PSOOptimizer(space, swarm_size=6, seed=ANY_SEED)
    )

    rows = _table(folder / "run_001" / "generations.csv")
    by_generation = [[t for t in study.trials if t.meta["generation"] == int(r["generation"])] for r in rows]
    assert [int(r["pruned"]) for r in rows] == [sum(t.pruned for t in g) for g in by_generation]
    assert [float(r["gen_best"]) for r in rows] == [
        min(t.score for t in g if t.state == "complete") for g in by_generation
    ]
    assert sum(int(r["pruned"]) for r in rows) > 0


def test_intermediate_csv_should_hold_one_row_per_reported_value_of_every_evaluated_trial(tmp_path, no_plots):
    """Given a search whose completed trials report four values and whose pruned trials one, when
    intermediate.csv is read, then it holds one row (run, trial, step, value, pruned) per reported value, in
    trial order, with the trial's pruned flag.

    Oracle: report module docstring (intermediate.csv holds one row per reported value of every evaluated
    trial). Hand count: seven completed trials of four values and five pruned trials of one value are 33 rows.
    """
    study, folder = _run(tmp_path)

    rows = _table(folder / "run_001" / "intermediate.csv")
    expected = [
        ["1", str(t.number), str(step), repr(value), str(int(t.pruned))]
        for t in study.trials
        for step, value in t.intermediate_values.items()
    ]
    assert list(rows[0]) == ["run", "trial", "step", "value", "pruned"]
    assert [list(r.values()) for r in rows] == expected
    assert len(rows) == 7 * STEPS + 5


class _Twice(Optimizer):
    """A scripted optimizer that hands out every random point twice in a row, so every second trial is a
    cached repeat of the one before."""

    def __init__(self, space, seed):
        super().__init__(space, seed=seed)
        self.last = None

    def ask(self):
        point = self.space.sample(self.rng, 1) if self.last is None else self.last
        self.last = point if self.last is None else None
        return self._candidates(point)

    def tell(self, candidates, scores, pruned=None):
        pass

    def cancel(self, candidates):
        pass


def _pruning_positive(params, trial):
    """Reports the sphere value and raises TrialPruned for x > 0."""
    trial.report(_sphere(params), 0)
    if params["x"] > 0.0:
        raise TrialPruned()
    return _sphere(params)


def test_intermediate_csv_and_reported_steps_should_leave_cached_repeats_out(tmp_path, no_plots):
    """Given a scripted optimizer that hands out every point twice and trials that report one value each, when
    the results folder is read, then intermediate.csv and summary.json's reported_steps count the evaluated
    trials only, while pruned_trials counts the cached repeats of pruned trials too.

    Oracle: report module docstring (cached repeats left out of intermediate.csv and not counted in
    `reported_steps`).
    """
    space = _space()
    study = optimize(
        _pruning_positive, space, max_evals=20, optimizer=_Twice(space, ANY_SEED), output=tmp_path
    )
    folder = next(p for p in tmp_path.iterdir() if p.is_dir())

    rows = _table(folder / "run_001" / "intermediate.csv")
    summary = _json(folder / "summary.json")
    assert [int(r["trial"]) for r in rows] == [t.number for t in study.trials if not t.cached]
    assert summary["reported_steps"] == study.n_calls
    assert summary["pruned_trials"] == sum(t.pruned for t in study.trials)
    assert any(t.pruned and t.cached for t in study.trials)


def test_summary_json_should_count_completed_pruned_and_failed_trials_and_the_reported_steps(
    tmp_path, no_plots
):
    """Given the scripted search of 12 calls, when summary.json is read, then it counts seven completed, five
    pruned and no failed trials and 33 reported steps.

    Oracle: report module docstring (summary.json counts the completed, pruned and failed trials and the
    intermediate values reported as `reported_steps`).
    """
    _, folder = _run(tmp_path)

    summary = _json(folder / "summary.json")

    found = {k: summary[k] for k in ("trials", "complete_trials", "pruned_trials", "failed_trials")}
    assert found == {"trials": 12, "complete_trials": 7, "pruned_trials": 5, "failed_trials": 0}
    assert summary["reported_steps"] == 7 * STEPS + 5


def test_report_md_should_name_pruned_trials_pruner_and_intermediate_csv_only_when_values_were_reported(
    tmp_path, no_plots
):
    """Given the scripted pruning search and a search whose objective reports nothing, when their report.md
    files are read, then only the first names the pruned trials with the reported values, the pruner and
    intermediate.csv.

    Oracle: report module docstring (report.md counts the pruned trials and the intermediate values reported)
    and the folder layout (intermediate.csv only when trials reported values).
    """
    _, pruning = _run(tmp_path / "pruning")
    _, plain = _run(tmp_path / "plain", function=_plain, pruner=NopPruner())

    with_values = (pruning / "report.md").read_text(encoding="utf-8")
    without = (plain / "report.md").read_text(encoding="utf-8")
    assert "failed 0, pruned 5; 33 intermediate values reported)" in with_values
    assert "- Pruner: _Odd()" in with_values
    assert "run_001/intermediate.csv" in with_values
    assert ("pruned" in without, "Pruner" in without, (plain / "run_001" / "intermediate.csv").exists()) == (
        False,
        False,
        False,
    )


def test_console_should_name_the_pruning_step_and_count_pruned_trials_only_when_there_are_any(
    tmp_path, no_plots
):
    """Given the scripted pruning search at verbose 2, when its console lines are read, then every pruned
    trial's line says 'pruned at step 0', and a generation line holds 'pruned <n>' exactly when its generation
    had pruned trials.

    Oracle: report module docstring (the console counts pruned trials when there are any, and a pruned trial's
    line names the step it was pruned at). Hand count: the ABC generation lines come from the trials of each
    cycle.
    """
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        study, _ = _run(tmp_path, verbose=2)

    lines = out.getvalue().splitlines()
    trial_lines = [line for line in lines if line.startswith("  trial")]
    shown = {int(line.split()[1]): " pruned " in line for line in lines if line.startswith("gen ")}
    expected = {g: any(t.pruned for t in study.trials if t.meta["generation"] == g) for g in shown}
    assert [("(pruned at step 0)" in line) for line in trial_lines] == [t.pruned for t in study.trials]
    assert shown == expected
    assert any(shown.values())


def test_parameter_analysis_should_leave_pruned_trials_out(tmp_path, no_plots):
    """Given a search whose pruned trials report a value of -1000, far below every completed score, when
    parameters.csv is read, then the score correlations equal those of a twin search whose pruned trials
    report a value near the completed ones.

    Oracle: report module docstring (the parameter analysis leaves pruned trials out, as it leaves out failed
    ones). The two searches differ only in the last values of their pruned trials, which the optimizers rank
    by step alone here (all pruned at step 0 with no pruned rival in a food source), so their completed trials
    are the same.
    """
    _, far = _run(tmp_path / "far", function=_reporting(-1000.0))
    _, near = _run(tmp_path / "near", function=_reporting(1.0))

    columns = ("parameter", "pearson_with_score", "spearman_with_score")
    found = [{c: r[c] for c in columns} for r in _table(far / "run_001" / "parameters.csv")]
    expected = [{c: r[c] for c in columns} for r in _table(near / "run_001" / "parameters.csv")]
    assert found == expected


def _reporting(value):
    """An objective that reports `value` at step 0 for the trials the pruner prunes and the falling loss for
    the others."""

    def objective(params, trial):
        trial.report(value if trial.number % 2 and trial.number > 1 else _curve(params, 0), 0)
        if trial.should_prune():
            raise TrialPruned()
        return _sphere(params)

    return objective


def test_learning_curves_should_be_drawn_only_for_a_run_that_reported_values(tmp_path):
    """Given the scripted pruning search and a search whose objective reports nothing, both with plots on,
    when their plot folders are listed, then only the first holds 12_learning_curves.png and both hold the
    other eleven plots.

    Oracle: plots module docstring (12_learning_curves when trials reported intermediate values) and report
    module docstring (plots/*.png).
    """
    _, pruning = _run(tmp_path / "pruning")
    _, plain = _run(tmp_path / "plain", function=_plain, pruner=NopPruner())

    with_values = sorted(p.name for p in (pruning / "run_001" / "plots").iterdir())
    without = sorted(p.name for p in (plain / "run_001" / "plots").iterdir())
    assert with_values == [*without, "12_learning_curves.png"]
    assert len(without) == 11
    assert _json(pruning / "summary.json")["notes"] == []


def test_learning_curves_should_be_drawn_for_a_run_that_reported_values_but_pruned_none(tmp_path):
    """Given a search whose objective reports values but whose pruner is NopPruner, with plots on, when its
    plot folder and notes are read, then 12_learning_curves.png is written without a pruned trial and no plot
    failed.

    Oracle: plots module docstring (12_learning_curves when trials reported intermediate values, pruned trials
    apart, so a run without pruned trials draws the completed ones alone).
    """
    study, folder = _run(tmp_path, pruner=NopPruner())

    names = sorted(p.name for p in (folder / "run_001" / "plots").iterdir())
    assert "12_learning_curves.png" in names
    assert _json(folder / "summary.json")["notes"] == []
    assert [t.pruned for t in study.trials] == [False] * 12


def _interrupting(after):
    """_learning raising KeyboardInterrupt on call `after` + 1."""
    calls = [0]

    def objective(params, trial):
        calls[0] += 1
        if calls[0] > after:
            raise KeyboardInterrupt
        return _learning(params, trial)

    return objective


def test_resume_should_write_the_pruning_files_as_the_uninterrupted_run_wrote_them(tmp_path, no_plots):
    """Given a scripted pruning search interrupted after 7 calls and resumed with the same pruner, when its
    trials.csv pruned column, intermediate.csv and generations.csv pruned column are read, then they equal
    those of the uninterrupted search.

    Oracle: report module docstring (every other file of a run folder is derived from the trials and is
    written again from the start when a run is resumed) and the resume docstring (restored trials keep their
    state and values).
    """
    space = _space()

    def make(seed):
        return ABCOptimizer(space, colony_size=4, seed=seed)

    optimize(
        _learning, space, max_evals=12, optimizer=make, seed=ANY_SEED, output=tmp_path / "full", pruner=_Odd()
    )
    optimize(
        _interrupting(7),
        space,
        max_evals=12,
        optimizer=make,
        seed=ANY_SEED,
        output=tmp_path / "cut",
        pruner=_Odd(),
    )
    full = next((tmp_path / "full").iterdir()) / "run_001"
    cut = next((tmp_path / "cut").iterdir())

    resume(_learning, cut, optimizer=make, pruner=_Odd())

    def view(run_dir):
        trials = [r["pruned"] for r in _table(run_dir / "trials.csv")]
        steps = [(r["trial"], r["step"], r["value"]) for r in _table(run_dir / "intermediate.csv")]
        generations = [r["pruned"] for r in _table(run_dir / "generations.csv")]
        return trials, steps, generations

    assert view(cut / "run_001") == view(full)
