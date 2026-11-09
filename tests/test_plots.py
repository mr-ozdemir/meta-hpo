"""Unit tests of metahpo.plots through the results folder of real runs.

Oracles: the module docstring of metahpo.plots (matplotlib is a dependency; make_plots draws every
plot of a run into run_dir/plots) and of metahpo.report (report.md notes failed plots).

The note tests replace make_plots with a stub that raises.

Unspecified, therefore not asserted: the look of the plots and their pixel sizes.

Input space: Every run that can be plotted, which is unbounded.
Subset: One small run per optimizer on a mixed space, an all-NaN run, and stubs that raise ImportError and
RuntimeError.
"""

import math

import pytest

import metahpo
from metahpo import (
    ABCOptimizer,
    BBPSOOptimizer,
    Categorical,
    Float,
    Int,
    PSOOptimizer,
    ResultWriter,
    Space,
    optimize,
)

ANY_SEED = 0


EXPECTED = {
    "01_convergence.png",
    "02_generations.png",
    "03_diversity.png",
    "04_score_distribution.png",
    "05_parameter_vs_score.png",
    "06_parameter_evolution.png",
    "07_parallel_coordinates.png",
    "08_correlation.png",
    "09_durations.png",
    "10_phases.png",
    "11_improvements.png",
}


def _space():
    """A fresh mixed space with a log Float, an Int, a Categorical and a linear Float."""
    return Space(
        {
            "lr": Float(1e-4, 1e-1, log=True),
            "units": Int(16, 256),
            "act": Categorical(["relu", "gelu"]),
            "dropout": Float(0.0, 0.5),
        }
    )


def _score(params):
    """A smooth score with its maximum near lr = 1e-3, units = 128 and gelu."""
    act = 0.0 if params["act"] == "gelu" else 0.01
    return 0.95 - 0.02 * (math.log10(params["lr"]) + 3.0) ** 2 - ((params["units"] - 128) / 256) ** 2 - act


def _plots(tmp_path, optimizer, function=_score):
    """Runs optimize() with a plotting ResultWriter below tmp_path; returns the names of the plot files."""
    optimize(
        function,
        optimizer.space,
        direction="max",
        max_evals=80,
        optimizer=optimizer,
        seed=ANY_SEED,
        verbose=0,
        output=None,
        callbacks=[ResultWriter(tmp_path)],
    )
    folder = next(p for p in tmp_path.iterdir() if p.is_dir())
    return {p.name for p in (folder / "run_001" / "plots").iterdir()}, folder


def _raise(error):
    """A make_plots stub that raises `error`."""

    def stub(*args, **kwargs):
        raise error

    return stub


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda s: ABCOptimizer(s, direction="max", seed=ANY_SEED), id="abc"),
        pytest.param(lambda s: PSOOptimizer(s, direction="max", seed=ANY_SEED), id="pso"),
        pytest.param(lambda s: BBPSOOptimizer(s, direction="max", seed=ANY_SEED), id="bbpso"),
    ],
)
def test_make_plots_should_write_every_plot_for_each_optimizer(tmp_path, make):
    """Given a run of each optimizer, when the run ends, then the eleven plot files
    exist and report.md links them.

    Oracle: plots module docstring (make_plots draws every plot of a run) and the report module docstring
    (report.md links the plots).
    """
    names, folder = _plots(tmp_path, make(_space()))

    assert names == EXPECTED
    assert "plots/01_convergence.png" in (folder / "report.md").read_text(encoding="utf-8")


def test_make_plots_should_cope_with_a_run_without_finite_scores(tmp_path):
    """Given an objective that always returns NaN, when the run ends, then the plots that need no finite score
    are written and no plot raises.

    Oracle: plots module docstring; report.md notes a failed plot, so no note means none failed.
    """
    names, folder = _plots(
        tmp_path, ABCOptimizer(_space(), direction="max", seed=ANY_SEED), lambda p: math.nan
    )

    assert {"01_convergence.png", "09_durations.png"} <= names
    assert "Plots failed" not in (folder / "report.md").read_text(encoding="utf-8")


def test_report_should_note_a_missing_plot_library_as_a_plot_failure(tmp_path, monkeypatch):
    """Given make_plots raising ImportError, as a broken matplotlib install would, when the run ends, then
    report.md notes the failure and no plot is written.

    Oracle: plots module docstring (a failed plot is noted in report.md and the other files are written).
    """
    monkeypatch.setattr(metahpo.plots, "make_plots", _raise(ImportError("no matplotlib")))

    names, folder = _plots(tmp_path, ABCOptimizer(_space(), direction="max", seed=ANY_SEED))

    assert names == set()
    assert "Plots failed: ImportError: no matplotlib" in (folder / "report.md").read_text(encoding="utf-8")


def test_report_should_note_a_plot_failure_and_still_write_the_summary(tmp_path, monkeypatch):
    """Given make_plots raising RuntimeError, when the run ends, then report.md notes the failure and
    summary.json is written.

    Oracle: report module docstring (summaries are written when the run ends).
    """
    monkeypatch.setattr(metahpo.plots, "make_plots", _raise(RuntimeError("disk full")))

    _, folder = _plots(tmp_path, ABCOptimizer(_space(), direction="max", seed=ANY_SEED))

    assert "Plots failed: RuntimeError: disk full" in (folder / "report.md").read_text(encoding="utf-8")
    assert (folder / "summary.json").is_file()


class _RandomSearch(metahpo.Optimizer):
    """A minimal optimizer without population, generations or candidate meta: one random candidate per ask."""

    def ask(self):
        return self._fresh(1)

    def tell(self, candidates, scores):
        for cand, score in zip(candidates, scores, strict=True):
            f = self._internal(float(score))
            if f < self.best_f or self.best_x is None:
                self.best_f, self.best_x = f, cand.x.copy()


def test_make_plots_should_skip_the_diversity_and_phase_plots_for_an_optimizer_without_them(tmp_path):
    """Given an optimizer without population and candidate meta, when the run ends, then every plot except
    the diversity and phase plots is written.

    Oracle: plots module docstring; diversity needs a population and phases need candidate meta.
    """
    names, _ = _plots(tmp_path, _RandomSearch(_space(), direction="max", seed=ANY_SEED))

    assert names == EXPECTED - {"03_diversity.png", "10_phases.png"}


def test_make_plots_should_draw_only_the_convergence_plot_for_a_run_without_trials(tmp_path):
    """Given a run state with no trials, no generations and no improvements, when make_plots is called, then
    only the convergence plot is written.

    Oracle: plots module docstring (make_plots draws the plots of a run); a plot needs data to show.
    """
    space = _space()
    state = metahpo.RunState(
        space, ABCOptimizer(space, seed=ANY_SEED), "min", 5, ANY_SEED, metahpo.Study("min")
    )

    written = metahpo.plots.make_plots(state, tmp_path, [], [])

    assert len(written) == 1
    assert {p.name for p in (tmp_path / "plots").iterdir()} == {"01_convergence.png"}


def test_make_plots_should_draw_the_score_histogram_without_a_density_when_every_score_is_equal(tmp_path):
    """Given an objective that always returns 0.5, when the run ends, then the score distribution is still
    written and no plot fails, although a density of a single value cannot be estimated.

    Oracle: plots module docstring; report.md notes a failed plot, so no note means none failed.
    """
    names, folder = _plots(tmp_path, ABCOptimizer(_space(), direction="max", seed=ANY_SEED), lambda p: 0.5)

    assert "04_score_distribution.png" in names
    assert "Plots failed" not in (folder / "report.md").read_text(encoding="utf-8")
