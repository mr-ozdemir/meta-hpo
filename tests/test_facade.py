"""Unit tests of optimize() (facade.py): small budgets, argument errors and user exceptions.

Unit tests of optimize() (facade.py) on tiny spaces with small budgets.

Grouped by statement of optimize: the max_evals bound, the optimizer space/direction check, the cache flag,
and pass-through of the objective's exceptions.

Equivalent mutants, documented instead of tested: 'cached = cache and key in seen' -> 'cached = key in seen'
(seen is written only when cache is True, so with cache=False it stays empty and both forms give False), and
'seen.setdefault(key, score)' -> 'seen[key] = score' (a cached key is never stored with a different score).

Unspecified, not asserted: whether a Space with equal content but another identity is accepted (the space
test below uses a Space that differs in content as well), and whether a partial Study is reachable after
the objective raised.

Error paths of optimize(): argument validation and pass-through of the objective's own exceptions.

Covered raise sites (facade.py): optimize ValueError "max_evals" and "same space". The objective's
exceptions propagate because optimize calls it without try/except. The full inventory is in the module
docstring of test_space_errors.py.

dummy_objective is a dummy: it is safe to call and returns 0.0, and the validation errors fire before or
independently of its value, so no assertion depends on it. The propagation test uses a raising stub instead,
because there the objective's behaviour is what the test exercises.

Unspecified, not asserted: which check fires first when max_evals < 1 is combined with a mismatched optimizer.

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: Tiny spaces, small budgets and one invalid argument or raising objective per error path; floats
(also whole ones), booleans, strings and negative values for every integer argument, and numpy integers
for every integer argument; False and True for verbose; seeds as a numpy array, a range, a tuple, a
generator and a 0-d numpy array.
"""

import json

import numpy as np
import pytest

from metahpo import ABCOptimizer, Categorical, Float, Space, optimize

ANY_SEED = 0


IRRELEVANT_SCORE = 0.0


class ObjectiveFailed(Exception):
    """A user exception that the library cannot confuse with its own ValueError or RuntimeError."""


def dummy_objective(params):
    """Dummy objective: its value never matters for the tests that use it."""
    return IRRELEVANT_SCORE


def stub_objective_raising(params):
    """Stub objective that always raises the user's own exception."""
    raise ObjectiveFailed


def _spy_objective(calls):
    """Spy objective that records every params dict it receives in `calls` and returns 0.0 or 1.0 by
    choice."""

    def spy_objective(params):
        calls.append(params)
        return {"a": 0.0, "b": 1.0}[params["c"]]

    return spy_objective


def _cached_flags(study):
    """The cached flag of every trial of `study`, in trial order."""
    return [t.cached for t in study.trials]


def test_optimize_should_raise_value_error_when_max_evals_is_zero():
    """Given max_evals=0, when optimize() is called, then ValueError is raised.

    Oracle: task specification 'ValueError for max_evals < 1'. Input: 0 is the invalid value next to the
    smallest valid budget, decisive against a '< 0' bound (site: optimize max_evals check -> raise).
    The keyword separates it from the 'same space' error of the same call.
    """
    space = Space({"x": Float(0.0, 1.0)})

    with pytest.raises(ValueError, match="max_evals"):
        optimize(dummy_objective, space, max_evals=0, seed=ANY_SEED)


def test_optimize_should_return_exactly_one_trial_when_max_evals_is_one():
    """Given max_evals=1, when optimize() runs, then the Study holds exactly one trial.

    Oracle: task specification 'exact max_evals trials (even inside the initial batch)'. Input: 1 is the
    smallest valid budget and lies inside the default initial batch of 5, so a budget check that is off by one
    or does not cut the batch is detected (sites: optimize max_evals check, batch slice).
    """
    space = Space({"x": Float(0.0, 1.0)})

    study = optimize(dummy_objective, space, max_evals=1, seed=ANY_SEED)

    assert len(study.trials) == 1


def test_optimize_should_raise_value_error_when_optimizer_uses_another_space():
    """Given an ABCOptimizer built on a different Space, when optimize() is called with it, then ValueError.

    Oracle: task specification 'ValueError for mismatched optimizer space'. Input: the other Space differs in
    name and bounds, so the case holds whether spaces are compared by identity or by content; the directions
    match, so only the space part of the check can fire (site: optimize 'opt.space is not space').
    """
    space = Space({"x": Float(0.0, 1.0)})
    opt = ABCOptimizer(Space({"y": Float(0.0, 2.0)}), seed=ANY_SEED)

    with pytest.raises(ValueError, match="same space"):
        optimize(dummy_objective, space, max_evals=5, optimizer=opt)


def test_optimize_should_raise_value_error_when_optimizer_uses_another_direction():
    """Given an ABCOptimizer with direction 'max', when optimize(direction='min') uses it, then ValueError.

    Oracle: task specification 'ValueError for mismatched optimizer direction'. Input: the optimizer is built
    on the same Space object, so only the direction part of the check can fire
    (site: optimize 'opt.direction != direction').
    """
    space = Space({"x": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, direction="max", seed=ANY_SEED)

    with pytest.raises(ValueError, match="same space"):
        optimize(dummy_objective, space, direction="min", max_evals=5, optimizer=opt)


def test_optimize_should_run_given_optimizer_when_space_and_direction_match():
    """Given an ABCOptimizer on the same Space and direction, when optimize() runs 20 evaluations with it,
    then the run completes with 20 trials and the optimizer reports the Study's best score.

    Oracle: task specification (only a mismatched optimizer is rejected; exact max_evals trials) and the
    base contract that best_score is the best told score in the user's direction; every score here is
    finite, so it equals Study.best_score. Input: 20 stays a small budget past the initial batch.
    """
    space = Space({"x": Float(0.0, 1.0)})
    opt = ABCOptimizer(space, seed=ANY_SEED)

    study = optimize(lambda p: (p["x"] - 0.3) ** 2, space, max_evals=20, optimizer=opt)

    assert len(study.trials) == 20
    assert opt.best_score == study.best_score


def test_optimize_should_call_function_on_every_evaluation_when_cache_is_false():
    """Given a two-choice space and cache=False, when optimize() runs 30 evaluations, then the function is
    called 30 times and no trial is marked cached.

    Oracle: facade docstring and README Behaviour bullet 2 make cache=True the only reuse mode; n_calls
    counts calls of the function. Input: two possible parameter sets over 30 evaluations make repeats certain,
    so any reuse with cache=False is visible (site: optimize cached flag). Positive counterpart: the
    repository test of the cache with cache=True.
    """
    space = Space({"c": Categorical(["a", "b"])})
    calls = []

    study = optimize(_spy_objective(calls), space, max_evals=30, seed=ANY_SEED, cache=False)

    assert len(calls) == 30
    assert study.n_calls == 30
    assert _cached_flags(study) == [False] * 30


def test_optimize_should_propagate_user_exception_when_function_raises():
    """Given an objective that raises ObjectiveFailed, when optimize() calls it, then ObjectiveFailed escapes.

    Oracle: task specification 'user exceptions propagate'. Input: a custom exception class, so swallowing it,
    turning it into a NaN score or wrapping it in a library error is detected.
    """
    space = Space({"x": Float(0.0, 1.0)})

    with pytest.raises(ObjectiveFailed):
        optimize(stub_objective_raising, space, max_evals=5, seed=ANY_SEED)


def _stub_objective_raising_on_call(failing_call, failure):
    """Builds a stub objective that returns IRRELEVANT_SCORE and raises `failure` on call number
    `failing_call`."""
    calls = []

    def stub_objective(params):
        calls.append(params)
        if len(calls) == failing_call:
            raise failure
        return IRRELEVANT_SCORE

    return stub_objective


@pytest.mark.parametrize(
    "max_evals",
    [
        pytest.param(0, id="zero-boundary"),
        pytest.param(-1, id="negative"),
    ],
)
def test_optimize_should_raise_value_error_naming_max_evals_when_the_budget_is_below_one(max_evals):
    """Given max_evals below 1, when optimize() is called, then ValueError is raised.

    Oracle: task specification 'ValueError for max_evals < 1' and the message 'max_evals must be at least 1'
    in optimize. Inputs: 0 is one below the smallest valid budget, -1 is the negative class. The keyword
    rules out the 'same space' site of the same call.
    """
    space = Space({"x": Float(0.0, 1.0)})

    with pytest.raises(ValueError, match="max_evals") as excinfo:
        optimize(dummy_objective, space, max_evals=max_evals, seed=ANY_SEED)

    assert excinfo.type is ValueError


@pytest.mark.parametrize(
    "build_optimizer",
    [
        pytest.param(
            lambda space: ABCOptimizer(Space({"x": Float(0.0, 1.0)}), seed=ANY_SEED),
            id="equal-content-but-other-space-object",
        ),
        pytest.param(
            lambda space: ABCOptimizer(space, direction="max", seed=ANY_SEED), id="direction-max-vs-min"
        ),
    ],
)
def test_optimize_should_raise_value_error_when_optimizer_space_or_direction_differs(build_optimizer):
    """Given an optimizer on another space or direction, when optimize() is called with it, then ValueError is
    raised.

    Oracle: task specification 'ValueError for mismatched optimizer space/direction' and the optimize code,
    which compares the space by identity (site: opt.space is not space). Inputs: a separate Space with the
    same content shows equal content is not enough; the direction case reuses the same space object so only
    the direction comparison can fire. max_evals=5 is valid, so the max_evals site cannot fire first.
    """
    space = Space({"x": Float(0.0, 1.0)})
    opt = build_optimizer(space)

    with pytest.raises(ValueError, match="same space") as excinfo:
        optimize(dummy_objective, space, direction="min", max_evals=5, optimizer=opt)

    assert excinfo.type is ValueError


@pytest.mark.parametrize(
    "failing_call",
    [
        pytest.param(1, id="first-call"),
        pytest.param(3, id="mid-initial-batch"),
        pytest.param(6, id="first-employed-candidate"),
    ],
)
def test_optimize_should_propagate_objective_exception_unchanged_when_objective_raises(failing_call):
    """Given an objective that raises its own exception, when optimize() runs, then that same instance
    propagates.

    Oracle: task specification 'user exceptions propagate'. Identity of the instance rules out wrapping,
    replacing or re-creating it. Inputs: with the default colony_size=10 the initial batch has 10 // 2 = 5
    sources, so call 1 and call 3 fall inside the batch path and call 6 is the first one-candidate
    (employed) ask. cache=False makes call n the n-th evaluation. The stub's returns before the raise are
    not asserted.
    """
    space = Space({"x": Float(0.0, 1.0)})
    failure = ObjectiveFailed("objective-failed-7f3a")
    stub_objective = _stub_objective_raising_on_call(failing_call, failure)

    with pytest.raises(ObjectiveFailed) as excinfo:
        optimize(stub_objective, space, max_evals=10, seed=ANY_SEED, cache=False)

    assert excinfo.type is ObjectiveFailed
    assert excinfo.value is failure


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        pytest.param({"max_evals": 20.0}, "max_evals", id="max-evals-whole-float"),
        pytest.param({"max_evals": True}, "max_evals", id="max-evals-bool"),
        pytest.param({"max_evals": "5"}, "max_evals", id="max-evals-string"),
        pytest.param({"runs": 2.0}, "runs", id="runs-whole-float"),
        pytest.param({"runs": True}, "runs", id="runs-bool"),
        pytest.param({"log_every": 0}, "log_every", id="log-every-0-without-console"),
        pytest.param({"log_every": 2.0}, "log_every", id="log-every-whole-float"),
        pytest.param({"verbose": 1.5}, "verbose", id="verbose-fraction"),
        pytest.param({"verbose": "2"}, "verbose", id="verbose-string"),
        pytest.param({"verbose": -1}, "verbose", id="verbose-negative"),
        pytest.param({"seed": 2.0}, "seed", id="seed-whole-float"),
        pytest.param({"seed": "12"}, "seed", id="seed-string"),
        pytest.param({"seed": True}, "seed", id="seed-bool"),
        pytest.param({"seed": -1}, "seed", id="seed-negative"),
        pytest.param({"seed": [1, -1], "runs": 2}, "seed", id="seed-of-the-second-run-negative"),
        pytest.param({"seed": [1, 2.0], "runs": 2}, "seed", id="seed-of-the-second-run-float"),
        pytest.param({"seed": np.array([[1, 2]]), "runs": 1}, "seed", id="seed-2-d-array"),
        pytest.param({"seed": 1.5 + 0j}, "seed", id="seed-not-iterable"),
    ],
)
def test_optimize_should_reject_an_integer_argument_that_is_no_integer_before_any_call_or_folder(
    tmp_path, kwargs, name
):
    """Given one integer argument of optimize() that is a float (also a whole one), a bool, a string or
    below its smallest value, when optimize() is called with an output folder, then ValueError names the
    argument, the objective is never called and no results folder is written.

    Oracle: the library's error convention (bad input raises ValueError before any training and before
    any folder is written) and the existing messages ('runs must be an integer of at least 1'). Decisive:
    max_evals=20.0 failed only inside the loop after the folder was written, log_every=0 passed with
    verbose=0, and a negative seed of the second run failed after the first run.
    """
    calls = []
    space = Space({"x": Float(0.0, 1.0)})
    arguments = {"max_evals": 5, "seed": ANY_SEED, "verbose": 0, **kwargs}
    seed = arguments.pop("seed")

    with pytest.raises(ValueError, match=name):
        optimize(_spy_objective(calls), space, seed=seed, output=tmp_path, **arguments)

    assert (calls, list(tmp_path.iterdir())) == ([], [])


def test_optimize_should_accept_numpy_integers_for_its_integer_arguments_and_record_python_ints(tmp_path):
    """Given max_evals, runs, log_every, verbose and seed as numpy integers, when optimize() runs with an
    output folder, then it makes two runs of 6 calls each and config.json records max_evals, runs and the
    seeds as plain JSON integers.

    Oracle: the stage brief and the optimize docstring (max_evals counts calls; runs repeats the search;
    seed is an integer per run); numpy integers are integers.
    """
    space = Space({"x": Float(0.0, 1.0)})

    studies = optimize(
        dummy_objective,
        space,
        max_evals=np.int64(6),
        runs=np.int32(2),
        log_every=np.int16(1),
        verbose=np.int8(0),
        seed=np.uint32(3),
        output=tmp_path,
    )

    config = json.loads(next(tmp_path.iterdir()).joinpath("config.json").read_text(encoding="utf-8"))
    assert [s.n_calls for s in studies] == [6, 6]
    assert (config["max_evals"], config["runs"], config["seeds"]) == (6, 2, [3, 3])
    assert [type(v) for v in (config["max_evals"], config["runs"], *config["seeds"])] == [int] * 4


@pytest.mark.parametrize(
    ("verbose", "printed"), [pytest.param(False, 0, id="false"), pytest.param(True, 3, id="true")]
)
def test_optimize_should_take_false_and_true_for_verbose_as_levels_0_and_1(
    monkeypatch, capsys, verbose, printed
):
    """Given verbose=False and verbose=True without METAHPO_VERBOSE, when optimize() runs one generation,
    then False prints nothing and True prints the header, the generation line and the end line of level 1.

    Oracle: optimize docstring (verbose takes False and True as 0 and 1; verbose=1 prints one line per
    generation) and 0.7.x, which took them that way. Decisive: rejecting a bool verbose broke a common habit.
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)
    space = Space({"x": Float(0.0, 1.0)})

    optimize(dummy_objective, space, max_evals=5, seed=ANY_SEED, verbose=verbose, output=None)

    assert len(capsys.readouterr().out.splitlines()) == printed


def _seed_generator():
    return (seed for seed in (1, 2))


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        pytest.param(lambda: np.array([1, 2]), [1, 2], id="numpy-array"),
        pytest.param(lambda: range(1, 3), [1, 2], id="range"),
        pytest.param(lambda: (1, np.int64(2)), [1, 2], id="tuple"),
        pytest.param(_seed_generator, [1, 2], id="generator"),
        pytest.param(lambda: np.array(7), [7, 7], id="zero-d-array"),
    ],
)
def test_optimize_should_accept_any_iterable_of_seeds_but_a_string_and_record_python_ints(
    tmp_path, seed, expected
):
    """Given seed as a 1-D numpy array, a range, a tuple, a generator or a 0-d numpy array, with runs=2,
    when optimize() runs with an output folder, then both runs are made and config.json records the seeds
    as plain JSON integers.

    Oracle: optimize docstring (seed is None, one integer or one seed per run; any iterable of seeds but a
    string) and 0.7.x, which took any iterable through list(seed). Decisive: a numpy array of seeds was
    refused once the seeds were checked.
    """
    space = Space({"x": Float(0.0, 1.0)})

    studies = optimize(dummy_objective, space, max_evals=4, runs=2, seed=seed(), verbose=0, output=tmp_path)

    config = json.loads(next(tmp_path.iterdir()).joinpath("config.json").read_text(encoding="utf-8"))
    assert [s.n_calls for s in studies] == [4, 4]
    assert (config["seeds"], [type(s) for s in config["seeds"]]) == (expected, [int, int])
