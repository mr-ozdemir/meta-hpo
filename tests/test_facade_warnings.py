"""Integration tests of the warnings of optimize() and resume(): the MetaHpoWarning category, the ignored
seed warned about before the search starts, a run without a finite score and failed plots warned about at
the end of a call, one warning per condition in a fixed order, and the standard warning filters.

Oracles: the MetaHpoWarning docstring (the category of every warning meta-hpo emits; each condition once per
call, pointing at the line that called optimize() or resume(); a condition the arguments show after every
argument check has passed and before the search starts, when nothing is written and the objective was not
called; a condition of the outcome when the call has finished and every result file was written; the
standard filters apply), the optimize docstring (a seed given together with an optimizer object has no
effect; an argument that raises comes without a warning; a run without a finite score, then failed plots,
also after a KeyboardInterrupt; the first warning is the one raised under the 'error' filter), the resume
docstring (the same rule on every trial of the runs when the call returns, also when nothing was evaluated)
and the plots and report module docstrings (a failed plot is noted in summary.json). Python's warnings module
defines the filters and the recorded filename that the stacklevel points to.

Unspecified, therefore not asserted: the exact wording of a warning beyond the words matched, and warnings
that third-party libraries emit while the plots are drawn.

Input space: Every argument of optimize() and resume(), every objective, every warning filter and every
results folder, which is unbounded.
Subset: An optimizer object of each library class with a seed as an int, a list and a numpy integer; a
factory and the default optimizer with a seed and an object without one; five argument errors together with
the ignored seed; objectives returning only NaN, +inf or -inf, NaN in one run of three, a finite score among
NaN, a KeyboardInterrupt before and after the first finite score, a raising objective, a rule that stops a
NaN run; resume() of an interrupted and of a finished NaN folder; make_plots and make_run_plots stubs that
raise or note a failure after a third-party warning, a figure method that breaks two plots, no results
folder; the filters 'always', 'error' and 'ignore' for the category.
"""

import json
import math
import warnings

import numpy as np
import pytest

import metahpo
from metahpo import (
    ABCOptimizer,
    BBPSOOptimizer,
    Callback,
    Float,
    MetaHpoWarning,
    Patience,
    PSOOptimizer,
    ResultWriter,
    Space,
    optimize,
    resume,
)

BUDGET = 20

NO_FINITE = "ended without a finite score"

PLOTS_FAILED = "noted, the first: "

OBJECTS = [
    pytest.param(lambda s: ABCOptimizer(s, seed=7), id="abc"),
    pytest.param(lambda s: PSOOptimizer(s, seed=7), id="pso"),
    pytest.param(lambda s: BBPSOOptimizer(s, seed=7), id="bbpso"),
]


def _space():
    return Space({"x": Float(0.0, 1.0), "y": Float(0.0, 1.0)})


def _score(params):
    return params["x"] + params["y"]


def _positions(study):
    return [t.x.tolist() for t in study.trials]


def _meta_hpo_warnings(record):
    return [w for w in record if issubclass(w.category, MetaHpoWarning)]


def _folder(root):
    folders = [p for p in root.iterdir() if p.is_dir()]
    assert len(folders) == 1
    return folders[0]


class _Counting:
    """An objective that counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, params):
        self.calls += 1
        return _score(params)


class _Starts(Callback):
    """A callback that counts the runs that started."""

    def __init__(self):
        self.started = 0

    def on_run_start(self, state):
        self.started += 1


def _nan(params):
    return math.nan


def _no_plots(*args, **kwargs):
    return []


def _without_plots(monkeypatch):
    monkeypatch.setattr("metahpo.plots.make_plots", _no_plots)
    monkeypatch.setattr("metahpo.plots.make_run_plots", _no_plots)


def _raise(error):
    def stub(*args, **kwargs):
        raise error

    return stub


def _recorded(call):
    """Runs `call` with every warning recorded; returns its result and the MetaHpoWarnings."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = call()
    return result, _meta_hpo_warnings(record)


class _NanInRun(Callback):
    """An objective that returns NaN in the runs named in `runs` and a finite score in the others; as a
    callback it learns which run is going on."""

    def __init__(self, runs):
        self.runs = runs
        self.run = 0

    def on_run_start(self, state):
        self.run = state.run

    def __call__(self, params):
        return math.nan if self.run in self.runs else _score(params)


class _Interrupting:
    """An objective that returns `scores` in turn and raises KeyboardInterrupt once they are used up."""

    def __init__(self, scores):
        self.scores = list(scores)

    def __call__(self, params):
        if not self.scores:
            raise KeyboardInterrupt
        return self.scores.pop(0)


def _third_party_then_note(state, run_dir, generations, best_history, notes=None):
    """A make_plots stub that emits a warning as matplotlib may, then notes one failed plot."""
    warnings.warn("a third-party warning while drawing", UserWarning, stacklevel=1)
    notes.append("Plot 01_convergence failed: RuntimeError: stub")
    return []


def _broken_colorbar(*args, **kwargs):
    raise RuntimeError("colorbar broken")


def test_meta_hpo_warning_should_be_a_user_warning_exported_by_the_package():
    """Given the package, when MetaHpoWarning is looked up, then it is a subclass of UserWarning and the class
    that metahpo.report defines.

    Oracle: MetaHpoWarning docstring (the category of every warning meta-hpo emits) and the convention of
    scikit-learn's ConvergenceWarning(UserWarning), which the category follows.
    """
    category = metahpo.MetaHpoWarning

    assert issubclass(category, UserWarning)
    assert category is metahpo.report.MetaHpoWarning


@pytest.mark.parametrize("make", OBJECTS)
@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(123, id="int"),
        pytest.param([123], id="list"),
        pytest.param(np.int64(123), id="numpy-int"),
    ],
)
def test_optimize_should_warn_once_that_the_seed_has_no_effect_on_an_optimizer_object(make, seed):
    """Given an optimizer object of each library class seeded 7 and seed 123 given to optimize() as an int, a
    list or a numpy integer, when optimize() runs, then exactly one MetaHpoWarning says that seed has no
    effect and how to seed the search, it points at the line of this test that called optimize(), and the
    study equals the run of the same object without seed=.

    Oracle: optimize docstring (a seed given together with an optimizer object has no effect and is warned
    about) and the MetaHpoWarning docstring (each condition once per call). Decisive: the seed was ignored
    in silence while config.json, summary.json and the console recorded it as if it had seeded the search.
    """
    space, twin = _space(), _space()

    with pytest.warns(MetaHpoWarning) as record:
        study = optimize(_score, space, max_evals=BUDGET, optimizer=make(space), seed=seed, output=None)

    reference = optimize(_score, twin, max_evals=BUDGET, optimizer=make(twin), output=None)
    caught = _meta_hpo_warnings(record)
    assert len(caught) == 1
    assert "has no effect because optimizer= is an optimizer object" in str(caught[0].message)
    assert "pass the seed to the optimizer" in str(caught[0].message)
    assert caught[0].filename == __file__
    assert _positions(study) == _positions(reference)


def _factory_with_seed():
    space = _space()
    return optimize(
        _score, space, max_evals=BUDGET, optimizer=lambda seed: ABCOptimizer(space, seed=seed), seed=3
    )


def _default_optimizer_with_seed():
    return optimize(_score, _space(), max_evals=BUDGET, seed=3)


def _object_without_seed():
    space = _space()
    return optimize(_score, space, max_evals=BUDGET, optimizer=ABCOptimizer(space, seed=3))


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(_factory_with_seed, id="factory-with-seed"),
        pytest.param(_default_optimizer_with_seed, id="default-optimizer-with-seed"),
        pytest.param(_object_without_seed, id="object-without-seed"),
    ],
)
def test_optimize_should_not_warn_when_the_seed_takes_effect_or_none_is_given(call):
    """Given a factory seed -> optimizer with a seed, the default optimizer with a seed, and an optimizer
    object without a seed, when optimize() runs with every warning recorded, then no MetaHpoWarning is
    among them.

    Oracle: optimize docstring (only a seed given together with an optimizer object is warned about).
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()

    assert _meta_hpo_warnings(record) == []


def test_optimize_should_raise_the_seed_warning_turned_into_an_error_before_anything_is_written_or_called(
    tmp_path, capsys
):
    """Given every warning turned into an error and an optimizer object with seed=123, when optimize() runs
    with a results folder, the console at level 1 and a callback, then MetaHpoWarning is raised while the
    output folder is still empty, the console printed nothing, no run started and the objective was never
    called.

    Oracle: MetaHpoWarning docstring (a condition the arguments show is warned about before the search
    starts, when nothing is written and the objective was not called, so the warning turned into an
    exception costs no result).
    """
    space = _space()
    objective, starts = _Counting(), _Starts()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(MetaHpoWarning, match="has no effect"):
            optimize(
                objective,
                space,
                max_evals=BUDGET,
                optimizer=ABCOptimizer(space, seed=7),
                seed=123,
                output=tmp_path,
                verbose=1,
                callbacks=[starts],
            )

    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().out == ""
    assert (objective.calls, starts.started) == (0, 0)


def _other_direction(space):
    return ABCOptimizer(space, direction="max", seed=7), {}


def _other_space(space):
    return ABCOptimizer(_space(), seed=7), {}


def _two_runs(space):
    return ABCOptimizer(space, seed=7), {"runs": 2}


def _zero_budget(space):
    return ABCOptimizer(space, seed=7), {"max_evals": 0}


def _foreign_rule(space):
    return ABCOptimizer(space, seed=7), {"stop": [3]}


@pytest.mark.parametrize(
    "arguments, error",
    [
        pytest.param(_other_direction, ValueError, id="other-direction"),
        pytest.param(_other_space, ValueError, id="other-space"),
        pytest.param(_two_runs, ValueError, id="object-for-two-runs"),
        pytest.param(_zero_budget, ValueError, id="zero-budget"),
        pytest.param(_foreign_rule, TypeError, id="rule-not-a-stopper"),
    ],
)
def test_optimize_should_raise_an_argument_error_without_a_warning_before_it(arguments, error):
    """Given an optimizer object with seed=123 and one wrong argument (an optimizer of the other direction or
    of another space, two runs for one object, a budget of zero, a rule that is not a Stopper), when
    optimize() runs with every warning recorded, then the argument error is raised and no MetaHpoWarning was
    emitted.

    Oracle: optimize docstring (the ignored seed is warned about after every argument check has passed; an
    argument that raises comes without a warning).
    """
    space = _space()
    optimizer, wrong = arguments(space)
    kwargs = {"max_evals": BUDGET, "output": None, **wrong}

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(error):
            optimize(_score, space, optimizer=optimizer, seed=123, **kwargs)

    assert _meta_hpo_warnings(record) == []


@pytest.mark.parametrize(
    "objective, stop",
    [
        pytest.param(_nan, None, id="nan"),
        pytest.param(lambda p: math.inf, None, id="plus-infinity"),
        pytest.param(lambda p: -math.inf, None, id="minus-infinity"),
        pytest.param(_nan, Patience(5), id="nan-stopped-by-patience"),
    ],
)
def test_optimize_should_warn_once_when_the_run_ended_without_a_finite_score(objective, stop):
    """Given an objective that returns only NaN, +inf or -inf, also when Patience(5) stops the run, when
    optimize() runs, then exactly one MetaHpoWarning says the run ended without a finite score and what to
    check, points at the line of this test, and the study is the one an unwarned run returns.

    Oracle: optimize docstring (a run that ended without a finite score is warned about when the call has
    finished; each condition once per call). Decisive: the study had no best trial, best.json held none and
    only the console's last line said so.
    """
    with pytest.warns(MetaHpoWarning) as record:
        study = optimize(objective, _space(), max_evals=BUDGET, seed=3, stop=stop, output=None)

    caught = _meta_hpo_warnings(record)
    assert [(str(w.message).startswith("the run " + NO_FINITE), w.filename) for w in caught] == [
        (True, __file__)
    ]
    assert "check that the objective returns a finite number" in str(caught[0].message)
    assert all(t.failed for t in study.trials)


@pytest.mark.parametrize(
    "nan_runs, start",
    [
        pytest.param({2}, "1 of the 3 runs " + NO_FINITE, id="one-of-three"),
        pytest.param({1, 2, 3}, "all 3 runs " + NO_FINITE, id="all-three"),
    ],
)
def test_optimize_should_count_the_runs_without_a_finite_score_in_one_warning(nan_runs, start):
    """Given runs=3 and an objective that returns NaN in run 2 only, or in every run, when optimize() runs,
    then exactly one MetaHpoWarning names how many of the runs ended without a finite score.

    Oracle: optimize docstring (a run without a finite score is warned about once per call).
    """
    objective = _NanInRun(nan_runs)

    _, caught = _recorded(
        lambda: optimize(objective, _space(), max_evals=BUDGET, seed=3, runs=3, callbacks=[objective])
    )

    assert [str(w.message).startswith(start) for w in caught] == [True]


def _finite_among_nan():
    return optimize(
        lambda p: p["x"] if p["x"] < 0.3 else math.nan, _space(), max_evals=BUDGET, seed=3, runs=2
    )


def _interrupted_after_a_finite_score():
    return optimize(_Interrupting([0.5, math.nan]), _space(), max_evals=BUDGET, seed=3)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(_finite_among_nan, id="a-finite-score-in-every-run"),
        pytest.param(_interrupted_after_a_finite_score, id="interrupted-after-a-finite-score"),
    ],
)
def test_optimize_should_not_warn_when_every_run_has_a_finite_score(call):
    """Given two runs that each find a finite score among NaN scores, and a run interrupted after its first
    finite score, when optimize() runs with every warning recorded, then no MetaHpoWarning is among them.

    Oracle: optimize docstring (only a run without a finite score is warned about).
    """
    _, caught = _recorded(call)

    assert caught == []


def test_optimize_should_warn_about_a_run_interrupted_before_its_first_trial():
    """Given an objective that raises KeyboardInterrupt at its first call, when optimize() returns the
    interrupted study, then one MetaHpoWarning says the run ended without a finite score, or none before an
    interrupt.

    Oracle: optimize docstring (warned also when a KeyboardInterrupt ended the call; none before the run was
    interrupted). The returned study has no best trial, which is what the warning is about.
    """
    study, caught = _recorded(lambda: optimize(_Interrupting([]), _space(), max_evals=BUDGET, seed=3))

    assert (study.stop_reason, len(study.trials)) == ("interrupted", 0)
    assert [("or none before an interrupt" in str(w.message)) for w in caught] == [True]


def test_optimize_should_raise_the_objective_error_without_a_warning():
    """Given an objective that raises RuntimeError, when optimize() runs with every warning recorded, then
    the error propagates and no MetaHpoWarning was emitted, although the run has no finite score.

    Oracle: optimize docstring (a call that raises during the search emits none of these warnings).
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="broke"):
            optimize(_raise(RuntimeError("broke")), _space(), max_evals=BUDGET, seed=3)

    assert _meta_hpo_warnings(record) == []


def test_resume_should_warn_by_the_same_rule_when_it_evaluates_and_when_it_restores_everything(
    tmp_path, monkeypatch
):
    """Given a NaN run interrupted after 8 calls, when resume() finishes it and then resumes the finished
    folder again, restoring every trial without a call, then each resume() emits one MetaHpoWarning about the
    run without a finite score, pointing at the line of this test that called resume().

    Oracle: resume docstring (a run without a finite score, judged on every trial of the run when the call
    returns, also when every run was restored and nothing was evaluated).
    """
    _without_plots(monkeypatch)
    _recorded(
        lambda: optimize(_Interrupting([math.nan] * 8), _space(), max_evals=BUDGET, seed=3, output=tmp_path)
    )
    folder = _folder(tmp_path)

    with pytest.warns(MetaHpoWarning) as first:
        finished = resume(_nan, folder, verbose=0)
    with pytest.warns(MetaHpoWarning) as second:
        again = resume(lambda p: pytest.fail("the objective must not be called"), folder, verbose=0)

    assert (finished.stop_reason, len(finished.trials), len(again.trials)) == ("budget", BUDGET, BUDGET)
    assert [(NO_FINITE in m.message.args[0], m.filename) for m in _meta_hpo_warnings(first)] == [
        (True, __file__)
    ]
    assert [(NO_FINITE in m.message.args[0], m.filename) for m in _meta_hpo_warnings(second)] == [
        (True, __file__)
    ]


@pytest.mark.parametrize(
    "runs, stubs, count, first",
    [
        pytest.param(
            1,
            {"make_plots": _raise(ImportError("no matplotlib"))},
            "1 plot failure was ",
            "Plots failed: ImportError: no matplotlib",
            id="plot-library-missing",
        ),
        pytest.param(
            3,
            {
                "make_plots": _raise(ImportError("no matplotlib")),
                "make_run_plots": _raise(ImportError("no matplotlib")),
            },
            "4 plot failures were ",
            "Plots failed: ImportError: no matplotlib",
            id="plot-library-missing-in-three-runs",
        ),
        pytest.param(
            2,
            {"make_run_plots": _raise(RuntimeError("disk full"))},
            "1 plot failure was ",
            "Run plots failed: RuntimeError: disk full",
            id="run-comparison-failed",
        ),
    ],
)
def test_optimize_should_warn_once_with_the_number_of_failed_plots_and_the_first_failure(
    tmp_path, monkeypatch, runs, stubs, count, first
):
    """Given make_plots raising ImportError in one run and in three runs (whose run comparison raises too),
    and make_run_plots raising RuntimeError for two runs, when optimize() writes the results folder, then
    exactly one MetaHpoWarning names the number of noted plot failures over every run and the run comparison,
    the first failure and summary.json, points at the line of this test, and summary.json keeps its notes.

    Oracle: optimize docstring (plots a ResultWriter of the call could not draw, which summary.json notes, are
    warned about once when the call has finished) and the report module docstring (failed plots are noted).
    """
    monkeypatch.setattr("metahpo.plots.make_plots", stubs.get("make_plots", _no_plots))
    monkeypatch.setattr("metahpo.plots.make_run_plots", stubs.get("make_run_plots", _no_plots))

    with pytest.warns(MetaHpoWarning) as record:
        optimize(_score, _space(), max_evals=BUDGET, seed=3, runs=runs, output=tmp_path, verbose=0)

    notes = json.loads((_folder(tmp_path) / "summary.json").read_text(encoding="utf-8"))["notes"]
    caught = _meta_hpo_warnings(record)
    assert [(str(w.message).startswith(count + PLOTS_FAILED + first), w.filename) for w in caught] == [
        (True, __file__)
    ]
    assert "summary.json" in str(caught[0].message)
    assert first in notes


def test_optimize_should_count_each_plot_that_failed_by_name(tmp_path, monkeypatch):
    """Given Figure.colorbar raising, which breaks the parameter evolution and correlation plots, when a run
    with real plots ends, then one MetaHpoWarning counts the two failures and names the first.

    Oracle: optimize docstring (failed plots are warned about with their number) and the plots module
    docstring (each failed plot is noted by name with its error).
    """
    monkeypatch.setattr("matplotlib.figure.Figure.colorbar", _broken_colorbar)

    _, caught = _recorded(
        lambda: optimize(_score, _space(), max_evals=BUDGET, seed=3, output=tmp_path, verbose=0)
    )

    assert _messages_of(caught) == [
        "2 plot failures were "
        + PLOTS_FAILED
        + "Plot 06_parameter_evolution failed: RuntimeError: colorbar broken"
    ]


def _messages_of(caught):
    return [str(w.message).split(";")[0] for w in caught]


def _no_folder():
    return optimize(_score, _space(), max_evals=BUDGET, seed=3, output=None)


def _writer_without_plots(root):
    return optimize(
        _score, _space(), max_evals=BUDGET, seed=3, output=None, callbacks=[ResultWriter(root, plots=False)]
    )


def test_optimize_should_not_warn_about_plots_without_a_results_folder_or_with_plots_off(
    tmp_path, monkeypatch
):
    """Given make_plots raising, when optimize() runs without a results folder, and with a ResultWriter whose
    plots are off, then no MetaHpoWarning is recorded, because no plot was drawn.

    Oracle: optimize docstring (only plots that a ResultWriter of the call could not draw are warned about).
    """
    monkeypatch.setattr("metahpo.plots.make_plots", _raise(RuntimeError("never drawn")))

    _, without_folder = _recorded(_no_folder)
    _, plots_off = _recorded(lambda: _writer_without_plots(tmp_path))

    assert (without_folder, plots_off) == ([], [])


def test_resume_should_warn_about_the_plots_it_could_not_draw(tmp_path, monkeypatch):
    """Given a finished folder and make_plots raising when resume() draws the plots again, when resume()
    restores the run, then one MetaHpoWarning about the failed plot points at the line that called resume().

    Oracle: resume docstring (plots that could not be drawn are warned about when the call has finished).
    """
    _without_plots(monkeypatch)
    optimize(_score, _space(), max_evals=BUDGET, seed=3, output=tmp_path, verbose=0)
    monkeypatch.setattr("metahpo.plots.make_plots", _raise(RuntimeError("disk full")))

    with pytest.warns(MetaHpoWarning) as record:
        resume(_score, _folder(tmp_path), verbose=0)

    assert [
        (m.message.args[0].startswith("1 plot failure was "), m.filename) for m in _meta_hpo_warnings(record)
    ] == [(True, __file__)]


def test_the_plot_block_should_silence_third_party_warnings_and_our_warning_should_come_after_every_file(
    tmp_path, monkeypatch
):
    """Given every warning turned into an error, two NaN runs and a make_plots stub that emits a third-party
    UserWarning and then notes a failed plot, when optimize() runs, then the third-party warning stays
    silenced inside the plot block (the note is the stub's, not 'Plots failed: UserWarning'), MetaHpoWarning
    is raised after both runs finished and wrote best.json, summary.json and report.md, and resume() of the
    folder returns both runs without a call.

    Oracle: MetaHpoWarning docstring (a condition of the outcome is warned about when the call has finished
    and every result file was written, so a warning turned into an exception costs no result), the report
    module docstring (the plot warnings of third-party libraries are silenced) and the resume docstring.
    """
    _without_plots(monkeypatch)
    monkeypatch.setattr("metahpo.plots.make_plots", _third_party_then_note)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(MetaHpoWarning, match=NO_FINITE):
            optimize(_nan, _space(), max_evals=BUDGET, seed=3, runs=2, output=tmp_path, verbose=0)

    folder = _folder(tmp_path)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    best = [json.loads((folder / f"run_00{n}" / "best.json").read_text(encoding="utf-8")) for n in (1, 2)]
    resumed, _ = _recorded(lambda: resume(lambda p: pytest.fail("no call"), folder, verbose=0))
    assert [r["stop_reason"] for r in summary["runs"]] == ["budget", "budget"]
    assert [b["status"] for b in best] == ["finished", "finished"]
    assert summary["notes"] == ["Plot 01_convergence failed: RuntimeError: stub"]
    assert (folder / "report.md").is_file()
    assert [len(study.trials) for study in resumed] == [BUDGET, BUDGET]


def test_optimize_should_emit_both_outcome_warnings_in_a_fixed_order(tmp_path, monkeypatch):
    """Given a NaN objective and make_plots raising, when optimize() runs, then two MetaHpoWarnings are
    emitted, the run without a finite score first and the failed plots second.

    Oracle: optimize docstring (the outcome warnings, in this order: a run without a finite score, then failed
    plots).
    """
    _without_plots(monkeypatch)
    monkeypatch.setattr("metahpo.plots.make_plots", _raise(RuntimeError("disk full")))

    _, caught = _recorded(
        lambda: optimize(_nan, _space(), max_evals=BUDGET, seed=3, output=tmp_path, verbose=0)
    )

    assert [(NO_FINITE in str(w.message), PLOTS_FAILED in str(w.message)) for w in caught] == [
        (True, False),
        (False, True),
    ]


def test_the_ignore_filter_should_silence_every_meta_hpo_warning(tmp_path, monkeypatch):
    """Given warnings.filterwarnings('ignore', category=MetaHpoWarning), an optimizer object with a seed, a
    NaN objective and make_plots raising, when optimize() runs, then no MetaHpoWarning is recorded.

    Oracle: MetaHpoWarning docstring (the standard filters apply; filterwarnings('ignore',
    category=MetaHpoWarning) silences them).
    """
    _without_plots(monkeypatch)
    monkeypatch.setattr("metahpo.plots.make_plots", _raise(RuntimeError("disk full")))
    space = _space()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=MetaHpoWarning)
        optimize(
            _nan,
            space,
            max_evals=BUDGET,
            optimizer=ABCOptimizer(space, seed=7),
            seed=123,
            output=tmp_path,
            verbose=0,
        )

    assert _meta_hpo_warnings(record) == []
