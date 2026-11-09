"""Unit tests of repeated runs in optimize(): `runs`, one seed per run and optimizer functions.

Oracles: the optimize docstring (runs repeats the whole search, each run with a fresh optimizer, its own
cache and the full max_evals; runs=1 returns a Study and more runs a list of Study; seed is None, one integer
used by every run, or one seed per run; an optimizer object only with runs=1, otherwise a function
seed -> optimizer; a KeyboardInterrupt stops the remaining runs) and the RunState docstring (run, runs).

The default results folder and console are switched off through METAHPO_OUTPUT and METAHPO_VERBOSE in
conftest; the folder written for several runs is tested in test_report.py.

Unspecified, therefore not asserted: the scores reached by a run, and the order in which an optimizer
function is called relative to the events of the previous run.

Input space: Every runs value, seed form (None, an integer, a sequence), optimizer form (None, an object,
a function) and objective, which is unbounded.
Subset: runs 1, 2 and 3; invalid runs 0, -1, 1.5 and True; an integer seed, a numpy integer seed and seed
lists of matching and wrong length; the default optimizer, an optimizer object, a PSO function and a
function that returns no optimizer; a four-value space for the cache; objectives that interrupt or fail in
the second run.
"""

import math

import numpy as np
import pytest

import metahpo
from metahpo import ABCOptimizer, Callback, Categorical, Float, Int, PSOOptimizer, Space, Study, optimize

ANY_SEED = 0

BUDGET = 30


def _space():
    """A fresh mixed space with a log Float, an Int and a Categorical."""
    return Space(
        {
            "lr": Float(1e-4, 1e-1, log=True),
            "units": Int(16, 256),
            "act": Categorical(["relu", "tanh", "gelu"]),
        }
    )


def _score(params):
    """An accuracy-like score with its peak near lr = 1e-3, units = 128 and gelu."""
    act = {"relu": 0.01, "tanh": 0.02, "gelu": 0.0}[params["act"]]
    return 0.95 - 0.02 * (math.log10(params["lr"]) + 3.0) ** 2 - ((params["units"] - 128) / 256) ** 2 - act


def _params(study):
    """The params of every trial of a study, in order."""
    return [t.params for t in study.trials]


class _Recorder(Callback):
    """Records (run, runs, seed) at the start of every run."""

    def __init__(self):
        self.starts = []

    def on_run_start(self, state):
        self.starts.append((state.run, state.runs, state.seed))


def _counting(calls, fail_after=None, error=None, score=_score):
    """An objective that counts its calls in calls[0], returns `score`, and raises `error` on call
    `fail_after` + 1."""

    def objective(params):
        calls[0] += 1
        if fail_after is not None and calls[0] > fail_after:
            raise error
        return score(params)

    return objective


def test_optimize_should_return_a_study_when_runs_is_one():
    """Given the default runs, when optimize() returns, then the result is a Study, not a list.

    Oracle: optimize docstring (with runs=1 optimize() returns a Study).
    """
    result = optimize(_score, _space(), direction="max", max_evals=BUDGET, seed=ANY_SEED)

    assert isinstance(result, Study)


def test_optimize_should_return_one_study_per_run_each_with_the_full_budget():
    """Given runs=3, when optimize() returns, then it is a list of three Study objects and each made
    max_evals real calls.

    Oracle: optimize docstring (more runs return a list of Study; every run gets the full max_evals).
    """
    studies = optimize(_score, _space(), direction="max", max_evals=BUDGET, seed=ANY_SEED, runs=3)

    assert [type(s) for s in studies] == [Study, Study, Study]
    assert [s.n_calls for s in studies] == [BUDGET, BUDGET, BUDGET]


@pytest.mark.parametrize("seed", [pytest.param(7, id="int"), pytest.param(np.int64(7), id="numpy-int")])
def test_optimize_should_give_every_run_the_same_integer_seed(seed):
    """Given one integer seed and runs=2, when optimize() returns, then both runs evaluated the same
    parameters in the same order, and both report that seed.

    Oracle: optimize docstring (one integer is used by every run) and the RunState docstring.
    """
    recorder = _Recorder()

    first, second = optimize(
        _score, _space(), direction="max", max_evals=BUDGET, seed=seed, runs=2, callbacks=[recorder]
    )

    assert _params(first) == _params(second)
    assert recorder.starts == [(1, 2, 7), (2, 2, 7)]


def test_optimize_should_run_each_seed_of_a_sequence_like_a_single_run_with_that_seed():
    """Given seed=[1, 2, 3] and runs=3, when optimize() returns, then run k evaluated exactly the
    parameters of a single run with the k-th seed.

    Oracle: optimize docstring (a sequence gives one seed per run; every run starts a fresh search).
    """
    space = _space()

    studies = optimize(_score, space, direction="max", max_evals=BUDGET, seed=[1, 2, 3], runs=3)

    singles = [optimize(_score, space, direction="max", max_evals=BUDGET, seed=s) for s in (1, 2, 3)]
    assert [_params(s) for s in studies] == [_params(s) for s in singles]


@pytest.mark.parametrize(
    "seed", [pytest.param([1, 2], id="too-few"), pytest.param([1, 2, 3, 4], id="too-many")]
)
def test_optimize_should_reject_a_seed_sequence_whose_length_differs_from_runs(seed):
    """Given runs=3 and a seed list of another length, when optimize() is called, then ValueError names
    both numbers and no run starts.

    Oracle: optimize docstring (a sequence holds one seed per run).
    """
    recorder = _Recorder()

    with pytest.raises(ValueError, match="runs=3"):
        optimize(_score, _space(), max_evals=BUDGET, seed=seed, runs=3, callbacks=[recorder])

    assert recorder.starts == []


@pytest.mark.parametrize(
    "runs",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
    ],
)
def test_optimize_should_reject_runs_that_are_not_a_positive_integer(runs):
    """Given runs 0, -1, 1.5 or True, when optimize() is called, then ValueError is raised.

    Oracle: optimize docstring (runs is the number of times the search is repeated).
    """
    with pytest.raises(ValueError, match="runs"):
        optimize(_score, _space(), max_evals=BUDGET, seed=ANY_SEED, runs=runs)


def test_optimize_should_reject_an_optimizer_object_for_several_runs():
    """Given an ABCOptimizer object and runs=2, when optimize() is called, then ValueError tells to pass a
    function seed -> optimizer and the optimizer is untouched.

    Oracle: optimize docstring (an optimizer object holds the state of one search; accepted only with runs=1).
    """
    space = _space()
    opt = ABCOptimizer(space, direction="max", seed=ANY_SEED)

    with pytest.raises(ValueError, match="function"):
        optimize(_score, space, direction="max", max_evals=BUDGET, optimizer=opt, seed=ANY_SEED, runs=2)

    assert opt.best_params is None


def test_optimize_should_build_a_fresh_optimizer_per_run_from_an_optimizer_function():
    """Given a function seed -> PSOOptimizer and seed=[4, 5], when optimize() runs twice, then the function
    was called with 4 and then 5 and each run used its own optimizer object.

    Oracle: optimize docstring (for several runs pass a function that builds an optimizer from a seed).
    """
    space = _space()
    built = []

    def make(seed):
        opt = PSOOptimizer(space, direction="max", swarm_size=5, seed=seed)
        built.append((seed, opt))
        return opt

    optimize(_score, space, direction="max", max_evals=BUDGET, optimizer=make, seed=[4, 5], runs=2)

    assert [seed for seed, _ in built] == [4, 5]
    assert built[0][1] is not built[1][1]


def test_optimize_should_accept_an_optimizer_function_with_a_single_run():
    """Given a function seed -> PSOOptimizer and the default runs, when optimize() returns, then it is one
    Study whose trials were proposed by the built optimizer.

    Oracle: optimize docstring (an optimizer function works for any number of runs).
    """
    space = _space()

    study = optimize(
        _score,
        space,
        direction="max",
        max_evals=BUDGET,
        optimizer=lambda seed: PSOOptimizer(space, direction="max", swarm_size=5, seed=seed),
        seed=ANY_SEED,
    )

    assert (type(study), study.trials[0].meta["phase"]) == (Study, "init")


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda seed: "not an optimizer", id="returns-a-string"),
        pytest.param(42, id="not-callable"),
    ],
)
def test_optimize_should_reject_an_optimizer_argument_that_yields_no_optimizer(make):
    """Given an optimizer function that returns a string, or an integer as optimizer, when optimize() is
    called, then TypeError is raised.

    Oracle: optimize docstring (optimizer is an Optimizer object or a function seed -> optimizer).
    """
    with pytest.raises(TypeError, match="Optimizer"):
        optimize(_score, _space(), max_evals=BUDGET, optimizer=make, seed=ANY_SEED, runs=2)


def test_optimize_should_give_every_run_its_own_cache():
    """Given a space of four parameter sets, max_evals 4 and runs=2, when optimize() returns, then the
    objective was called eight times: the second run trained every set again.

    Oracle: optimize docstring (every run has its own cache).
    """
    calls = [0]
    space = Space({"act": Categorical(["relu", "tanh", "gelu", "silu"])})

    studies = optimize(
        _counting(calls, score=lambda p: 0.5), space, direction="max", max_evals=4, seed=ANY_SEED, runs=2
    )

    assert (calls[0], [s.stop_reason for s in studies]) == (8, ["budget", "budget"])


def test_keyboard_interrupt_should_stop_the_remaining_runs_and_return_the_studies_so_far():
    """Given runs=3 and an objective that raises KeyboardInterrupt in the second run, when optimize()
    returns, then the list holds two studies, the first finished by budget and the second interrupted.

    Oracle: optimize docstring (a KeyboardInterrupt returns the studies so far and stops the remaining runs).
    """
    calls = [0]
    recorder = _Recorder()

    studies = optimize(
        _counting(calls, BUDGET + 5, KeyboardInterrupt),
        _space(),
        direction="max",
        max_evals=BUDGET,
        seed=ANY_SEED,
        runs=3,
        callbacks=[recorder],
    )

    assert [(s.n_calls, s.stop_reason) for s in studies] == [(BUDGET, "budget"), (5, "interrupted")]
    assert [run for run, _, _ in recorder.starts] == [1, 2]


def test_error_in_the_objective_should_stop_the_remaining_runs_and_propagate():
    """Given runs=3 and an objective that raises RuntimeError in the second run, when optimize() is called,
    then the error propagates and the third run never starts.

    Oracle: optimize docstring (an exception raised by function is re-raised).
    """
    calls = [0]
    recorder = _Recorder()

    with pytest.raises(RuntimeError, match="boom"):
        optimize(
            _counting(calls, BUDGET + 5, RuntimeError("boom")),
            _space(),
            max_evals=BUDGET,
            seed=ANY_SEED,
            runs=3,
            callbacks=[recorder],
        )

    assert [run for run, _, _ in recorder.starts] == [1, 2]


def test_run_state_should_default_to_a_single_run():
    """Given a RunState built without runs, when its run fields are read, then run and runs are both 1.

    Oracle: RunState docstring (run is the number of this run, runs the number of runs).
    """
    space = _space()
    state = metahpo.RunState(space, ABCOptimizer(space, seed=ANY_SEED), "min", 5, ANY_SEED, Study("min"))

    fields = (state.run, state.runs)

    assert fields == (1, 1)
