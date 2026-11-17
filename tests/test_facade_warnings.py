"""Integration tests of the warnings of optimize() and resume(): the MetaHpoWarning category, a condition of
the arguments warned about before the search starts, a condition of the outcome warned about at the end of a
call, one warning per condition, and the standard warning filters.

Oracles: the MetaHpoWarning docstring (the category of every warning meta-hpo emits; each condition once per
call, pointing at the line that called optimize() or resume(); a condition the arguments show after every
argument check has passed and before the search starts, when nothing is written and the objective was not
called; a condition of the outcome when the call has finished and every result file was written; the
standard filters apply), the optimize docstring (a seed given together with an optimizer object has no
effect and is warned about; an argument that raises comes without a warning) and the resume docstring (a
folder whose runs all finished evaluates nothing). Python's warnings module defines the filters and the
recorded filename that the stacklevel points to.

Unspecified, therefore not asserted: the exact wording of a warning beyond the words matched, and warnings
that third-party libraries emit while the plots are drawn.

Input space: Every argument of optimize() and resume(), every warning filter and every results folder, which
is unbounded.
Subset: An optimizer object of each library class with a seed as an int, a list and a numpy integer; a
factory and the default optimizer with a seed and an object without one; five argument errors together with
the ignored seed; the filters 'always', 'error' and 'ignore' for the category; an outcome notice put in place
of the empty list of the finished call, for optimize() and resume(), with and without the 'error' filter.
"""

import json
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
    PSOOptimizer,
    Space,
    optimize,
    resume,
)

BUDGET = 20

OUTCOME = "an outcome notice"

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


def _with_outcome_notice(monkeypatch):
    monkeypatch.setattr(metahpo.facade, "_finished_notices", lambda studies: [OUTCOME, OUTCOME])


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


def test_the_warning_of_a_finished_call_should_point_at_the_caller_of_optimize_and_of_resume(
    tmp_path, monkeypatch
):
    """Given an outcome notice that the finished call reports twice, when optimize() writes a folder and
    resume() continues it, then each call emits the notice once, pointing at the line of this test that made
    the call.

    Oracle: MetaHpoWarning docstring (each condition once per call, pointing at the line that called
    optimize() or resume(); a condition of the outcome when the call has finished).
    """
    _with_outcome_notice(monkeypatch)

    with pytest.warns(MetaHpoWarning) as first:
        optimize(_score, _space(), max_evals=BUDGET, seed=3, output=tmp_path, verbose=0)
    with pytest.warns(MetaHpoWarning) as second:
        resume(_score, _folder(tmp_path), verbose=0)

    expected = [(OUTCOME, __file__)]
    assert [(str(w.message), w.filename) for w in _meta_hpo_warnings(first)] == expected
    assert [(str(w.message), w.filename) for w in _meta_hpo_warnings(second)] == expected


def test_a_finished_call_should_raise_its_warning_turned_into_an_error_after_writing_a_complete_folder(
    tmp_path, monkeypatch
):
    """Given an outcome notice of the finished call and every warning turned into an error, when optimize()
    runs with a results folder, then MetaHpoWarning is raised, the folder holds the finished best.json,
    summary.json with stop reason 'budget', report.md and a log line per trial, and resume() of the folder
    returns the uninterrupted run without a call.

    Oracle: MetaHpoWarning docstring (a condition of the outcome is warned about when the call has finished
    and every result file was written, so a warning turned into an exception costs no result) and the
    resume docstring (a folder whose runs all finished evaluates nothing).
    """
    _with_outcome_notice(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(MetaHpoWarning, match=OUTCOME):
            optimize(_score, _space(), max_evals=BUDGET, seed=3, output=tmp_path, verbose=0)

    folder = _folder(tmp_path)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    best = json.loads((folder / "run_001" / "best.json").read_text(encoding="utf-8"))
    log = (folder / "run_001" / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    monkeypatch.setattr(metahpo.facade, "_finished_notices", lambda studies: [])
    resumed = resume(lambda p: pytest.fail("the objective must not be called"), folder, verbose=0)
    reference = optimize(_score, _space(), max_evals=BUDGET, seed=3, output=None)
    assert (summary["stop_reason"], best["status"], (folder / "report.md").is_file()) == (
        "budget",
        "finished",
        True,
    )
    assert len(log) == summary["trials"]
    assert _positions(resumed) == _positions(reference)


def test_the_ignore_filter_should_silence_every_meta_hpo_warning():
    """Given warnings.filterwarnings('ignore', category=MetaHpoWarning) and an optimizer object with a seed,
    when optimize() runs, then no MetaHpoWarning is recorded.

    Oracle: MetaHpoWarning docstring (the standard filters apply; filterwarnings('ignore',
    category=MetaHpoWarning) silences them).
    """
    space = _space()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=MetaHpoWarning)
        optimize(
            _score, space, max_evals=BUDGET, optimizer=ABCOptimizer(space, seed=7), seed=123, output=None
        )

    assert _meta_hpo_warnings(record) == []
