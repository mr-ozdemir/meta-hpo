"""Specification gaps of optimize() beyond tests/test_abc.py.

Oracles: the optimize docstring, README (Quick start, Behaviour bullet 2), CHANGELOG ('a configuration is
never trained twice') and the computed spec of the facade. The user objective is replaced by plain
recording functions; seeded runs are only compared with each other or checked for counts.

Unspecified and therefore not asserted: the numbering of Trial.number, whether optimize() uses seed when an
optimizer is passed, and whether an equal-content but distinct Space counts as the same space.

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: One test per specification gap of optimize().
"""

from unittest.mock import Mock

import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space, optimize

ANY_SEED = 0
SHARED_SEED = 9


class ObjectiveError(Exception):
    """Raised only by the objective double of the propagation test."""


def _mixed_space():
    """Three parameters of three kinds, 3 * 4 = 12 discrete combinations per float value."""
    return Space({"a": Float(-1.0, 1.0), "k": Categorical([1, 2, 3]), "n": Int(1, 4)})


def _small_discrete_space():
    """Six parameter sets only, so a run of tens of evaluations must repeat some of them."""
    return Space({"k": Categorical([1, 2, 3]), "n": Int(1, 2)})


def _cheap_objective(params):
    return params["a"] ** 2 + params["k"] + params["n"]


def _manual_scores(opt, function, max_evals):
    """The ask/tell loop of the README, truncated at max_evals evaluations; returns the scores in order."""
    scores = []
    while len(scores) < max_evals:
        cands = opt.ask()[: max_evals - len(scores)]
        batch = [float(function(c.params)) for c in cands]
        opt.tell(cands, batch)
        scores += batch
    return scores


def _params_key(params):
    return tuple(sorted(params.items()))


def _first_score_of_each_trial(study):
    """For every trial, the score of the first uncached trial with the same parameter set."""
    first = {}
    for t in study.trials:
        if not t.cached:
            first.setdefault(_params_key(t.params), t.score)
    return [first[_params_key(t.params)] for t in study.trials]


def _trial_scores(study):
    return [t.score for t in study.trials]


def _cached_flags(study):
    return [t.cached for t in study.trials]


def _types(values):
    return [type(v) for v in values]


def _key_orders(dicts):
    """The keys of each dict as a tuple, in insertion order."""
    return [tuple(d) for d in dicts]


@pytest.mark.parametrize(
    "max_evals",
    [
        pytest.param(1, id="minimum"),
        pytest.param(4, id="inside_initial_batch"),
        pytest.param(5, id="exactly_initial_batch"),
        pytest.param(6, id="first_employed_bee"),
        pytest.param(37, id="mid_cycle"),
    ],
)
def test_optimize_should_run_exactly_max_evals_trials_around_the_initial_batch(max_evals):
    """Given the default colony of 10 (an initial batch of 5), when optimize runs with max_evals, then the
    study holds exactly max_evals trials.

    Oracle: computed spec 'exact max_evals trials (even inside the initial batch)' and the optimize docstring
    '`max_evals` counts the optimizer's evaluations'. Boundaries: 1 the minimum, 4 inside the batch, 5 exactly
    the batch, 6 the first employed bee, 37 inside a later cycle; test_abc.py covers only 3.
    """
    space = _mixed_space()

    study = optimize(_cheap_objective, space, max_evals=max_evals, seed=ANY_SEED)

    assert len(study.trials) == max_evals


@pytest.mark.parametrize("max_evals", [pytest.param(0, id="zero"), pytest.param(-1, id="negative")])
def test_optimize_should_raise_value_error_when_max_evals_is_below_one(max_evals):
    """Given max_evals below 1, when optimize is called, then ValueError about max_evals is raised and the
    function was never called.

    Oracle: computed spec 'ValueError for max_evals<1'. Boundary: 0 is next to the valid minimum 1, which the
    budget test uses.
    """
    calls = []

    with pytest.raises(ValueError, match="max_evals"):
        optimize(calls.append, _small_discrete_space(), max_evals=max_evals, seed=ANY_SEED)

    assert calls == []


@pytest.mark.parametrize(
    ("make_optimizer", "direction"),
    [
        pytest.param(
            lambda space: ABCOptimizer(Space({"y": Int(1, 3)}), seed=ANY_SEED), "min", id="other_space"
        ),
        pytest.param(
            lambda space: ABCOptimizer(space, direction="max", seed=ANY_SEED), "min", id="other_direction"
        ),
    ],
)
def test_optimize_should_raise_value_error_when_optimizer_setup_differs(make_optimizer, direction):
    """Given an optimizer built on another space or for the other direction, when it is passed to optimize,
    then ValueError about the same space and direction is raised and the function was never called.

    Oracle: computed spec 'ValueError for ... mismatched optimizer space/direction'. Classes: a space with
    other parameters and a direction mismatch; no existing test passes an optimizer.
    """
    space = _small_discrete_space()
    opt = make_optimizer(space)
    calls = []

    with pytest.raises(ValueError, match="same space"):
        optimize(calls.append, space, direction=direction, optimizer=opt)

    assert calls == []


def test_optimize_should_call_function_for_every_trial_when_cache_is_false():
    """Given six possible parameter sets and 60 evaluations with cache=False, when optimize runs, then the
    function is called 60 times, n_calls is 60 and no trial is cached.

    Oracle: README 'With `cache=True` (the default), a repeated parameter set reuses its earlier score'; with
    the cache switched off nothing is reused, and the Study.n_calls docstring counts calls. Class: the
    cache=False branch, the negative partner of test_abc.py's cache test. 60 is ten times the six sets, so
    repeats are certain.
    """
    calls = []

    def spy_objective(params):
        calls.append(params)
        return params["k"] * params["n"]

    study = optimize(spy_objective, _small_discrete_space(), max_evals=60, seed=ANY_SEED, cache=False)

    assert len(calls) == 60
    assert study.n_calls == 60
    assert _cached_flags(study) == [False] * 60


def test_optimize_should_reuse_the_first_score_for_cached_trials():
    """Given a function that returns a different score on every call, when optimize runs 60 evaluations
    over six parameter sets with the default cache, then every trial of a parameter set has the score of its
    first call and the function never sees the same parameters twice.

    Oracle: README 'a repeated parameter set reuses its earlier score instead of training again' and
    CHANGELOG 'a configuration is never trained twice'. Class: reuse of the stored score, which the
    deterministic function of test_abc.py cannot tell apart from calling again.
    """
    calls = []

    def stub_objective(params):
        calls.append(tuple(sorted(params.items())))
        return float(len(calls))

    study = optimize(stub_objective, _small_discrete_space(), max_evals=60, seed=ANY_SEED)
    first_scores = _first_score_of_each_trial(study)

    assert _trial_scores(study) == first_scores
    assert len(set(calls)) == len(calls)
    assert study.n_calls == len(calls)


def test_optimize_should_use_an_abc_optimizer_by_default():
    """Given SHARED_SEED and 23 evaluations without an optimizer, when optimize runs, then its scores equal
    the README ask/tell loop of ABCOptimizer(space, seed=SHARED_SEED) truncated at 23. Any seed works; the
    two runs only need the same one.

    Oracle: optimize docstring 'The default optimizer is ABCOptimizer.' 23 is past the initial batch, so the
    comparison covers neighbour candidates as well. cache=False keeps the manual loop free of a cache.
    """
    space = _mixed_space()

    study = optimize(_cheap_objective, space, max_evals=23, seed=SHARED_SEED, cache=False)

    assert study.scores.tolist() == _manual_scores(
        ABCOptimizer(space, seed=SHARED_SEED), _cheap_objective, 23
    )


def test_optimize_should_ask_and_tell_the_optimizer_it_is_given():
    """Given an ABCOptimizer with colony_size 6 and SHARED_SEED, when it is passed to optimize for 23
    evaluations, then the scores equal the manual loop of an equal optimizer and the passed optimizer was
    told them.

    Oracle: computed spec 'optimize(function, space, ..., optimizer=None, ...)' with the optimize docstring;
    a passed optimizer replaces the default. Class: a custom optimizer, which no existing test passes. The
    colony differs from the default, so the default could not produce the same scores past the batch.
    """
    space = _mixed_space()
    given = ABCOptimizer(space, colony_size=6, seed=SHARED_SEED)

    study = optimize(_cheap_objective, space, max_evals=23, optimizer=given, cache=False)

    expected = _manual_scores(ABCOptimizer(space, colony_size=6, seed=SHARED_SEED), _cheap_objective, 23)
    assert study.scores.tolist() == expected
    assert given.best_score == min(expected)


def test_optimize_should_pass_a_dict_keyed_by_the_space_names():
    """Given a three-parameter space, when optimize runs 12 evaluations, then every call receives a dict
    whose keys are the space names in their order.

    Oracle: optimize docstring '`function` receives a dict of parameter values' and the Space docstring 'An
    ordered set of named parameters'. 12 evaluations cover the initial batch and neighbour candidates, whose
    dicts are built by Space.move; cache=False makes every evaluation a call.
    """
    space = _mixed_space()
    calls = []

    def spy_objective(params):
        calls.append(params)
        return _cheap_objective(params)

    optimize(spy_objective, space, max_evals=12, seed=ANY_SEED, cache=False)

    assert _types(calls) == [dict] * 12
    assert _key_orders(calls) == [space.names] * 12


def test_optimize_should_propagate_a_user_exception_unwrapped():
    """Given a function that raises its own exception on the third call, when optimize runs, then that
    exception class reaches the caller and no further call is made.

    Oracle: computed spec 'user exceptions propagate'; README is silent. Class: an error raised inside the
    initial batch of five; no existing test raises from the objective.
    """
    mock_objective = Mock(side_effect=[0.0, 0.0, ObjectiveError("third call")])

    with pytest.raises(ObjectiveError):
        optimize(mock_objective, _small_discrete_space(), max_evals=50, seed=ANY_SEED)

    assert mock_objective.call_count == 3
