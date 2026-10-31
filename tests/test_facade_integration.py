"""optimize() through the real ABC chain: Space, ABCOptimizer, cache and Study working together.

optimize() through the real ABC chain: behaviour that only shows when Space, ABCOptimizer, cache and Study
run together, edge and corner configurations, a volume run and multi-seed convergence.

Integration counterparts of the doubled facade behaviours (the facade unit tests use a scripted optimizer
or objective doubles for these; here the real ABCOptimizer on the same Space produces the candidates):
- exact max_evals, also inside the initial batch:
  test_optimize_should_stop_inside_the_initial_batch_when_max_evals_is_smaller_than_the_batch
- cache reuse of an earlier score:
  test_optimize_should_call_the_function_once_per_distinct_params_when_the_abc_chain_repeats_them
- NaN as the worst score for the direction:
  test_optimize_should_never_report_a_point_of_the_nan_region_as_best_when_minimising and
  test_optimize_should_never_report_a_point_of_the_nan_region_as_best_when_maximising
No module here uses unittest.mock or an Optimizer subclass; only the user's objective is a plain function.

Oracles: README (Search space table: Float values in [low, high], Int low..high inclusive, Categorical one of
choices; Behaviour: NaN or infinity is the worst value, cache reuses the earlier score, same seed same
search), CHANGELOG (a configuration is never trained twice), optimize, Study.n_calls, Candidate and
Space.move docstrings, the original hyp_optimizer ABC.py and Karaboga & Basturk (2007) for the phase
schedule (SN employed, SN onlookers, at most one scout per cycle) and for greedy, guided search.

Seeds: ANY_SEED is arbitrary and fixed. CONVERGENCE_SEEDS are five consecutive seeds, so a result is not
one lucky sample. No assertion compares with a value captured from a seeded run; seeded runs are checked
only for seed-independent properties (counts, bounds, agreement, improvement over the run's own start).

Test-design choices the spec does not promise (flagged in the test docstrings): strict improvement over
the initial batch and hitting an exact integer target within a fixed budget.

Unspecified, not asserted:
- whether an optimizer passed to optimize() can be asked again after the budget ended inside its initial
  batch (the untold initial candidates stay pending, so ask() raises RuntimeError like any ask with a
  pending candidate);
- whether a NaN score is cached like a finite one;
- optimizer best_params while only non-finite scores were told;
- any convergence rate.

optimize() with an explicit ABCOptimizer: Space, optimizer, cache and Study working together.

The unit tests check each component alone; these tests check what only shows when they run as one chain:
the facade hands the optimizer's candidates to the objective, stores Trials built from the candidate's
params and x, reuses cached scores, and tells the optimizer, so Study and optimizer must end up agreeing.
Non-finite scores are injected into the chain as faults (a failed or diverged training run).

Oracles: optimize docstring (max_evals counts evaluations, cache reuses the earlier score of a repeated
parameter set), README Behaviour (NaN or infinity is the worst value for the direction, cache), Study and
Trial docstrings (failed means non-finite, best_trial uses finite trials only), Optimizer docstring
(scores in the user's direction), Candidate and Space.to_dicts docstrings.

Doubles: stub_objective returns a cheap score with one peak; stub_objective_injecting_non_finite returns
NaN, +inf or -inf in turn on every INJECT_EVERY-th real call and records what it returned; _nan_objective
always fails. They replace only the user's objective; Space and ABCOptimizer are real.

Seeds: ANY_SEED is arbitrary and fixed; INTEGRATION_SEED is arbitrary too, and the cache test asserts that
its run contains cached trials. No assertion compares with a value captured from a seeded run.

Unspecified, not asserted:
- best_params of the optimizer while only non-finite scores were told;
- whether an optimizer can be reused after optimize() stopped inside its initial batch;
- which of several trials with the same best score Study.best_trial returns (ties here come only from
  repeated parameter sets, which have equal params).

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: Edge and corner configurations of the real ABC chain, a volume run and a few fixed seeds.
"""

import math
import statistics

import numpy as np
import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space, optimize

ANY_SEED = 0


DEFAULT_COLONY_SIZE = 10


LONG_RUN_EVALS = 600


EDGE_RUN_EVALS = 300


NAN_RUN_EVALS = 300


NAN_EDGE_LOW = 0.2


NAN_EDGE_HIGH = 0.8


CORNER_RUN_EVALS = 50


VOLUME_DIM = 50


VOLUME_EVALS = 5000


TARGET = 500


TARGET_BUDGET = 200


CONVERGENCE_SEEDS = (0, 1, 2, 3, 4)


CONVERGENCE_EVALS = 200


def _readme_space():
    """A fresh copy of the README quick-start space: log Float, Categorical, Int and linear Float."""
    return Space(
        {
            "lr": Float(1e-4, 1e-2, log=True),
            "kernel": Categorical([3, 5, 7]),
            "units": Int(16, 256),
            "dropout": Float(0.0, 0.5),
        }
    )


def readme_objective(params):
    """Stub objective on the README space, smallest at lr = 1e-3, kernel = 5, units = 100, dropout = 0."""
    return (
        (math.log10(params["lr"]) + 3.0) ** 2
        + (params["kernel"] - 5) ** 2
        + abs(params["units"] - 100) / 100
        + params["dropout"]
    )


def _spy(function, calls):
    """Wraps `function` so that every real call appends its params to `calls`."""

    def spy_objective(params):
        calls.append(params)
        return function(params)

    return spy_objective


def _sphere_space_3d():
    return Space({f"x{i}": Float(-5.0, 5.0) for i in range(3)})


def sphere(params):
    """Stub objective: sum of squares, smallest (0) at the origin."""
    return sum(v * v for v in params.values())


def _linear_mixed_space():
    """Mixed space without log scaling, so its bounds are exact at the unit-cube edges."""
    return Space({"a": Float(-5.0, 5.0), "n": Int(0, 10), "c": Categorical(["p", "q", "r"])})


def linear_mixed_objective(params):
    """Stub objective, smallest at a = 0, n = 3, c = 'q'."""
    return params["a"] ** 2 + (params["n"] - 3) ** 2 + (params["c"] != "q")


def _typed(params):
    """params as {name: (type, value)}, so that 1 == 1.0 or True == 1 cannot hide a type difference."""
    return {name: (type(value), value) for name, value in params.items()}


def _cache_violations(study):
    """(numbers of trials that break the cache contract, number of distinct params, number of cached trials).

    A first occurrence of a params set must be a real call (cached False); a later occurrence must be cached
    and carry the first occurrence's score.
    """
    first_score, violations, cached = {}, [], 0
    for trial in study.trials:
        params_key = tuple(sorted(trial.params.items()))
        cached += trial.cached
        if params_key not in first_score:
            first_score[params_key] = trial.score
            if trial.cached:
                violations.append(trial.number)
        elif not trial.cached or trial.score != first_score[params_key]:
            violations.append(trial.number)
    return violations, len(first_score), cached


def _trials_whose_params_differ_from_a_fresh_decode(space, study):
    """Numbers of the trials whose params (value or type) differ from space.rows of their own x."""
    mismatches = []
    for trial in study.trials:
        fresh = space.rows(np.atleast_2d(trial.x))[1][0]
        if _typed(trial.params) != _typed(fresh):
            mismatches.append(trial.number)
    return mismatches


def _trials_outside_linear_mixed_space(study):
    """Numbers of the trials with a parameter outside the README range of _linear_mixed_space."""
    bad = []
    for trial in study.trials:
        p = trial.params
        a_ok = type(p["a"]) is float and -5.0 <= p["a"] <= 5.0
        n_ok = type(p["n"]) is int and 0 <= p["n"] <= 10
        c_ok = p["c"] in ("p", "q", "r")
        if not (a_ok and n_ok and c_ok):
            bad.append(trial.number)
    return bad


def _trials_outside_unit_interval(study, dim):
    """Numbers of the trials that do not have exactly `dim` float params, each in [0.0, 1.0]."""
    bad = []
    for trial in study.trials:
        values = list(trial.params.values())
        if len(values) != dim or not all(type(v) is float and 0.0 <= v <= 1.0 for v in values):
            bad.append(trial.number)
    return bad


def _scout_trial_numbers(study, sn):
    """Numbers of the trials after the first sn whose x differs from every earlier trial's x in more than
    one coordinate.

    A neighbour copies its food source's x and changes one coordinate, and every food source is the x of an
    earlier trial (initial point, accepted neighbour or scout), so a neighbour differs from that trial in at
    most one coordinate. A scout is a fresh uniform point, which almost surely shares no coordinate with an
    earlier trial. The listed trials are therefore exactly the scouts.
    """
    xs = [np.asarray(trial.x) for trial in study.trials]
    return [n for n in range(sn, len(xs)) if all(np.count_nonzero(xs[n] != xs[m]) > 1 for m in range(n))]


def _sphere_runs(seeds, max_evals, sn):
    """For each seed: (seed, best of its first sn trials, i.e. the random initial population, final best)."""
    runs = []
    for seed in seeds:
        study = optimize(sphere, _sphere_space_3d(), max_evals=max_evals, seed=seed)
        initial_best = min(t.score for t in study.trials[:sn])
        runs.append((seed, initial_best, study.best_score))
    return runs


def _seeds_without_strict_improvement(runs):
    return [seed for seed, initial_best, final_best in runs if not final_best < initial_best]


def _spread_report(runs):
    finals = [final_best for _, _, final_best in runs]
    mean, low, high = statistics.mean(finals), min(finals), max(finals)
    return f"final best over seeds: mean={mean:.6g} min={low:.6g} max={high:.6g}"


def test_optimize_should_stop_inside_the_initial_batch_when_max_evals_is_smaller_than_the_batch():
    """Given a real ABCOptimizer with colony_size=10 (an initial batch of 10 // 2 = 5 candidates), when
    optimize() runs with max_evals=3, then exactly 3 trials numbered 0..2 and 3 real calls result.

    Integration counterpart of the scripted-optimizer budget test (P-T3-15). Oracle: task specification
    'exact max_evals trials (even inside the initial batch)', optimize docstring (max_evals counts
    evaluations). Input: 3 is strictly inside the batch, so the facade must cut the batch the real
    optimizer hands out. The continuous sphere space gives distinct params, so the cache cannot lower the
    call count. Whether the optimizer can be asked again afterwards is unspecified and not asserted.
    """
    space = _sphere_space_3d()
    optimizer = ABCOptimizer(space, colony_size=DEFAULT_COLONY_SIZE, seed=ANY_SEED)
    calls = []
    max_evals = 3

    study = optimize(_spy(sphere, calls), space, max_evals=max_evals, optimizer=optimizer)

    assert [t.number for t in study.trials] == [0, 1, 2]
    assert len(calls) == max_evals


def test_optimize_should_call_the_function_once_per_distinct_params_when_the_abc_chain_repeats_them():
    """Given the README mixed space and the default ABC chain, when 600 evaluations run, then every first
    occurrence of a params set is a real call, every repeat is cached with the first score, and the real
    calls equal the number of distinct params sets.

    Integration counterpart of the objective-double cache test (P-T3-15). Oracle: README Behaviour ('a
    repeated parameter set reuses its earlier score'), CHANGELOG ('a configuration is never trained twice'),
    Study.n_calls docstring ('cached repeats are not counted'). Input: 600 evaluations run deep into the
    employed and onlooker phases, where Space.move repeats Int and Categorical values; the test asserts
    that cached trials occur, so the check is not empty.
    """
    space = _readme_space()
    calls = []

    study = optimize(_spy(readme_objective, calls), space, max_evals=LONG_RUN_EVALS, seed=ANY_SEED)

    violations, distinct, cached = _cache_violations(study)
    assert violations == []
    assert (study.n_calls, len(calls)) == (distinct, distinct)
    assert cached > 0


def _nan_below_edge_objective(params):
    """Stub objective: NaN for x < NAN_EDGE_LOW, where the raw value x would be smallest, else x."""
    return math.nan if params["x"] < NAN_EDGE_LOW else params["x"]


def _nan_above_edge_objective(params):
    """Stub objective: NaN for x > NAN_EDGE_HIGH, where the raw value x would be largest, else x."""
    return math.nan if params["x"] > NAN_EDGE_HIGH else params["x"]


def test_optimize_should_never_report_a_point_of_the_nan_region_as_best_when_minimising():
    """Given a real ABCOptimizer minimising x on [0, 1] and an objective that returns NaN for x < 0.2,
    where the raw value would be smallest, when 300 evaluations run, then the Study best score is at least
    0.2 and the optimizer reports the same best score as the Study.

    Integration counterpart of the NaN-as-worst double test (P-T3-15). Oracle: README Behaviour and
    Optimizer docstring (NaN counts as the worst value for the direction; +inf when minimising; scores in
    the user's direction), README (study.best_score and opt.best_score both report the search result).
    Input: the NaN region is placed where a NaN treated as an ordinary or a good value would win, so a wrong
    mapping puts the best inside it. The test asserts that failed trials occur, so the region was really
    visited.
    """
    space = Space({"x": Float(0.0, 1.0)})
    optimizer = ABCOptimizer(space, direction="min", seed=ANY_SEED)

    study = optimize(
        _nan_below_edge_objective, space, direction="min", max_evals=NAN_RUN_EVALS, optimizer=optimizer
    )

    assert any(t.failed for t in study.trials)
    assert study.best_score >= NAN_EDGE_LOW
    assert optimizer.best_score == study.best_score


def test_optimize_should_never_report_a_point_of_the_nan_region_as_best_when_maximising():
    """Given a real ABCOptimizer maximising x on [0, 1] and an objective that returns NaN for x > 0.8,
    where the raw value would be largest, when 300 evaluations run, then the Study best score is at most 0.8
    and the optimizer reports the same best score as the Study.

    Mirror of the minimising test for the other direction (P-T3-15). Oracle: README Behaviour and Optimizer
    docstring (NaN counts as the worst value for the direction; -inf when maximising; best_score in the
    user's direction), README (study.best_score and opt.best_score both report the search result). Input: a
    mapping that ignored the direction (NaN as +inf user score) would put the best inside the NaN region,
    and an optimizer best_score that forgot to flip the internal sign back for 'max' would report about
    -0.8 instead of the Study's positive best. The test asserts that failed trials occur.
    """
    space = Space({"x": Float(0.0, 1.0)})
    optimizer = ABCOptimizer(space, direction="max", seed=ANY_SEED)

    study = optimize(
        _nan_above_edge_objective, space, direction="max", max_evals=NAN_RUN_EVALS, optimizer=optimizer
    )

    assert any(t.failed for t in study.trials)
    assert study.best_score <= NAN_EDGE_HIGH
    assert optimizer.best_score == study.best_score


def test_optimize_should_give_every_trial_the_params_of_a_fresh_decode_of_its_point():
    """Given the README mixed space and the default ABC chain, when 600 evaluations run, then the params of
    every trial equal, in value and type, space.rows of the trial's own x.

    Emergent behaviour (P-T3-30): initial params come from the vectorised numpy decode (Space.rows),
    neighbour params from Space.move, which decodes only the changed coordinate in pure Python and copies
    the rest. Only the chain runs both paths on the same points. Oracle: Space.move docstring ('(key,
    params) of a point differing from a decoded point only in coordinate j'), Candidate docstring (params
    as the user's function receives them for x). Input: the log Float column is where math.exp and numpy
    exp could disagree; 600 evaluations are far past the 5 initial ones.
    """
    space = _readme_space()

    study = optimize(readme_objective, space, max_evals=LONG_RUN_EVALS, seed=ANY_SEED)

    assert _trials_whose_params_differ_from_a_fresh_decode(space, study) == []


def test_optimize_should_leave_study_and_abc_optimizer_with_the_same_best_on_the_readme_space():
    """Given an explicit ABCOptimizer on the README mixed space, when optimize() runs 600 evaluations with
    it, then Study and optimizer report the same best score and the same best params, in value and type.

    Emergent behaviour (P-T3-30): Study takes its best from trial params (Space.move path for neighbours),
    the optimizer decodes its stored best x with Space.to_dicts (numpy path), including the log Float lr.
    Oracle: README shows both study.best_params / best_score and opt.best_params / best_score as the result
    of the search. Exact equality is used because both decode the same point; a tie between different
    params sets is not expected with a continuous objective.
    """
    space = _readme_space()
    optimizer = ABCOptimizer(space, seed=ANY_SEED)

    study = optimize(readme_objective, space, max_evals=LONG_RUN_EVALS, optimizer=optimizer)

    assert study.best_score == optimizer.best_score
    assert _typed(study.best_params) == _typed(optimizer.best_params)


@pytest.mark.parametrize(
    "edge_setting",
    [
        pytest.param({"colony_size": 4}, id="minimum-colony-size-4-two-food-sources"),
        pytest.param({"limit": 1}, id="minimum-limit-1"),
    ],
)
def test_optimize_should_run_many_cycles_with_every_trial_in_range_when_abc_uses_a_minimum_setting(
    edge_setting,
):
    """Given a real ABCOptimizer at a minimum valid setting on a linear mixed space, when 300 evaluations run,
    then exactly 300 trials result, every parameter stays in its README range, and many cycles pass.

    Edge cases (P-T1-43): colony_size=4 is the smallest valid colony (2 food sources, one possible neighbour
    partner); limit=1 is the smallest valid abandonment limit (that scouts really occur at this setting is a
    separate behaviour, checked by the limit-one scout test). Oracle: ABCOptimizer
    guards ('colony_size must be at least 4', 'limit must be at least 1'), README Search space table, task
    specification (exact max_evals). The cycle bound comes from the phase schedule of ABC.py and
    Karaboga & Basturk 2007: after the sn initial evaluations a cycle costs at most sn employed + sn onlooker
    + 1 scout evaluations, and the last cycle may be incomplete, so at least
    (max_evals - sn) // (2 * sn + 1) - 1 cycles complete. No log Float is used, so the bounds check does not
    depend on the separate log-endpoint behaviour.
    """
    space = _linear_mixed_space()
    optimizer = ABCOptimizer(space, seed=ANY_SEED, **edge_setting)
    sn = optimizer.food_number
    min_complete_cycles = (EDGE_RUN_EVALS - sn) // (2 * sn + 1) - 1

    study = optimize(linear_mixed_objective, space, max_evals=EDGE_RUN_EVALS, optimizer=optimizer)

    assert len(study.trials) == EDGE_RUN_EVALS
    assert _trials_outside_linear_mixed_space(study) == []
    assert optimizer.cycle >= min_complete_cycles


def test_optimize_should_call_the_function_once_when_a_single_value_space_meets_minimum_colony_and_limit():
    """Given Space({'x': Int(5, 5)}), a real ABCOptimizer with colony_size=4 and limit=1 and a constant
    objective, when 50 evaluations run, then the function is called once and every later trial is cached
    with params {'x': 5}.

    Corner case (P-T1-43, several edges at once): a one-value Int (low == high is allowed), the minimum
    colony, the minimum limit, and a constant score, so every greedy comparison ties and fails, which keeps
    the scout rule active; whatever phase produces a candidate, it decodes to the same configuration. Scouts
    are not observed here (one coordinate cannot tell a scout from a neighbour); the limit-one scout test
    observes them. Oracle: README
    (Int low..high inclusive; cache), CHANGELOG ('never trained twice'), Study.n_calls docstring, task
    specification (exact max_evals).
    """
    space = Space({"x": Int(5, 5)})
    optimizer = ABCOptimizer(space, colony_size=4, limit=1, seed=ANY_SEED)
    calls = []

    def constant_objective(params):
        return 1.0

    study = optimize(_spy(constant_objective, calls), space, max_evals=CORNER_RUN_EVALS, optimizer=optimizer)

    assert [_typed(t.params) for t in study.trials] == [{"x": (int, 5)}] * CORNER_RUN_EVALS
    assert [t.cached for t in study.trials] == [False] + [True] * (CORNER_RUN_EVALS - 1)
    assert (study.n_calls, len(calls)) == (1, 1)


def test_optimize_should_replace_sources_by_random_points_at_most_once_per_cycle_when_limit_is_one():
    """Given a real ABCOptimizer with limit=1 on the 3-d linear mixed space, when 300 evaluations run, then
    scouts (trials that are not one-coordinate moves of an earlier trial) occur, and there are no more of
    them than completed cycles.

    Edge case (P-T1-43): limit=1 makes a source scout-eligible after a single failed improvement, the
    behaviour that makes this setting an edge. Oracle: ABC.py and Karaboga & Basturk 2007 (a source whose
    trial counter reaches limit is replaced by a random point; at most one scout per cycle), task
    specification (the cycle counter increments when the scout step finishes, so every scout is counted in
    optimizer.cycle). Input: with limit=1 and 300 evaluations many greedy comparisons fail, so scouts are
    expected in every run; the count is checked against the run's own cycle counter, not against a captured
    value. The scout detector is explained in _scout_trial_numbers; a 3-d space makes 'more than one
    coordinate changed' impossible for a neighbour.
    """
    space = _linear_mixed_space()
    optimizer = ABCOptimizer(space, limit=1, seed=ANY_SEED)
    sn = optimizer.food_number

    study = optimize(linear_mixed_objective, space, max_evals=EDGE_RUN_EVALS, optimizer=optimizer)

    scouts = _scout_trial_numbers(study, sn)
    assert scouts != []
    assert len(scouts) <= optimizer.cycle


def test_optimize_should_never_replace_a_source_by_a_random_point_when_limit_exceeds_the_budget():
    """Given a real ABCOptimizer whose limit equals the whole budget of 300 evaluations, when those 300
    evaluations run, then no trial after the initial population is a scout.

    Negative pair of the limit-one scout test (P-T1-43). Oracle: ABC.py and Karaboga & Basturk 2007 (a scout
    replaces a source only when its trial counter reaches limit). Input: a trial counter grows by at most one
    per evaluation and the initial population uses sn of the 300 evaluations, so no counter can reach 300;
    any scout would mean the limit check is ignored.
    """
    space = _linear_mixed_space()
    unreachable_limit = EDGE_RUN_EVALS
    optimizer = ABCOptimizer(space, limit=unreachable_limit, seed=ANY_SEED)
    sn = optimizer.food_number

    study = optimize(linear_mixed_objective, space, max_evals=EDGE_RUN_EVALS, optimizer=optimizer)

    assert _scout_trial_numbers(study, sn) == []


def test_optimize_should_complete_a_high_dimensional_volume_run_with_every_trial_in_range():
    """Given a 50-dimensional Float(0, 1) space and a trivial objective, when 5000 evaluations run through
    the default chain, then exactly 5000 trials result and every trial has 50 float params in [0, 1].

    Load proxy (P-T3-28): a functional-at-volume check of the one-coordinate moves and of the uniform draws
    at a budget ten times the default. The rule's goal of no performance degradation between releases is
    not tested, because wall-clock checks are excluded from this suite. Oracle: README Search space table
    (Float values in [low, high]), task specification (exact max_evals). Linear Float bounds are exact at
    the unit-cube edges, so the bound check is sound.
    """
    space = Space({f"x{i}": Float(0.0, 1.0) for i in range(VOLUME_DIM)})

    def trivial_objective(params):
        return params["x0"]

    study = optimize(trivial_objective, space, max_evals=VOLUME_EVALS, seed=ANY_SEED)

    assert len(study.trials) == VOLUME_EVALS
    assert _trials_outside_unit_interval(study, VOLUME_DIM) == []


@pytest.mark.parametrize("seed", [pytest.param(s, id=f"seed-{s}") for s in CONVERGENCE_SEEDS])
def test_optimize_should_reach_the_exact_integer_target_when_the_score_is_the_distance_to_it(seed):
    """Given Space({'x': Int(0, 1000)}) and the score |x - 500| (a distance: smaller is better), when 200
    evaluations run through the default chain, then the best params are {'x': 500} with score 0.

    Guided search (P-T7-30): the score is a fitness that turns a miss into a gradient, and ABC exploits good
    sources (Karaboga & Basturk 2007). Uniform random search hits 500 within 200 draws with probability
    1 - (1000 / 1001) ** 200, about 0.18 per seed, so five seeds all hitting by chance is about 2e-4; that is
    why several seeds are used. Applying the search-based-testing rule to the library's own search is a
    test-design choice; the spec promises no convergence rate.
    """
    space = Space({"x": Int(0, 1000)})

    def distance_objective(params):
        return abs(params["x"] - TARGET)

    study = optimize(distance_objective, space, max_evals=TARGET_BUDGET, seed=seed)

    assert _typed(study.best_params) == {"x": (int, TARGET)}
    assert study.best_score == 0.0


def test_optimize_should_improve_on_the_initial_population_for_every_seed_on_the_sphere():
    """Given the 3-d sphere on Float(-5, 5) and the default chain, when 200 evaluations run for each of
    five seeds, then every seed's final best is strictly below the best of its own random initial
    population; the mean, min and max of the final bests are reported.

    Base case (P-T1-43: a typical problem with the default configuration) and multi-run reporting
    (P-T8-28, adapted from timing runs to convergence runs: several runs, mean and min/max spread, no
    single sample trusted). The initial population is the first colony_size // 2 = 5 trials. Oracle:
    greedy selection never loses the best source (Karaboga & Basturk 2007), so 'no worse' always holds;
    strict improvement is its non-trivial form and a test-design choice the spec does not promise. No
    absolute threshold is asserted.
    """
    sn = DEFAULT_COLONY_SIZE // 2

    runs = _sphere_runs(CONVERGENCE_SEEDS, CONVERGENCE_EVALS, sn)

    assert _seeds_without_strict_improvement(runs) == [], _spread_report(runs)


INTEGRATION_SEED = 1


INTEGRATION_EVALS = 61


INJECT_EVERY = 4


INJECTED_VALUES = (math.nan, math.inf, -math.inf)


FAILURE_RUN_EVALS = 200


ALL_NAN_EVALS = 30


WORST_FOR = {"min": math.inf, "max": -math.inf}


def _small_discrete_space():
    """A fresh mixed Space with small Int and Categorical ranges, so neighbour moves repeat parameter sets."""
    return Space({"a": Float(-1.0, 1.0), "b": Int(1, 4), "c": Categorical(["x", "y", "z"])})


def stub_objective(params):
    """Stub objective with its peak at a = 0.2, b = 4, c = 'y'."""
    return -abs(params["a"] - 0.2) + 0.1 * params["b"] + (0.3 if params["c"] == "y" else 0.0)


def _stub_objective_injecting_non_finite():
    """A stub objective returning NaN, +inf, -inf in turn on every INJECT_EVERY-th real call and
    a + 0.1 * b otherwise; returns (function, finite scores returned, injected values returned)."""
    finite, injected = [], []

    def stub_objective_injecting_non_finite(params):
        call_number = len(finite) + len(injected) + 1
        if call_number % INJECT_EVERY == 0:
            injected.append(INJECTED_VALUES[len(injected) % len(INJECTED_VALUES)])
            return injected[-1]
        finite.append(params["a"] + 0.1 * params["b"])
        return finite[-1]

    return stub_objective_injecting_non_finite, finite, injected


def _nan_objective(params):
    """Stub objective whose every run fails."""
    return math.nan


def _run_integration_study(space):
    """optimize() maximising stub_objective with an explicit seeded ABCOptimizer; returns (study, opt)."""
    opt = ABCOptimizer(space, direction="max", seed=INTEGRATION_SEED)
    study = optimize(stub_objective, space, direction="max", max_evals=INTEGRATION_EVALS, optimizer=opt)
    return study, opt


def _trial_numbers(study):
    return [t.number for t in study.trials]


def _decode_mismatches(space, study):
    """Numbers of the trials whose params (values or types) differ from space.to_dicts(trial.x)."""
    mismatches = []
    for trial in study.trials:
        expected = space.to_dicts(trial.x)[0]
        typed = {n: (type(v), v) for n, v in trial.params.items()}
        if typed != {n: (type(v), v) for n, v in expected.items()}:
            mismatches.append(trial.number)
    return mismatches


def _cache_report(study):
    """(number of cached trials, numbers of cached trials whose score differs from the first earlier
    real trial with equal params)."""
    first_score, cached, wrong = {}, 0, []
    for trial in study.trials:
        params_key = tuple(sorted(trial.params.items()))
        if trial.cached:
            cached += 1
            if params_key not in first_score or first_score[params_key] != trial.score:
                wrong.append(trial.number)
        else:
            first_score.setdefault(params_key, trial.score)
    return cached, wrong


def _count_failed_real_calls(study):
    return sum(t.failed and not t.cached for t in study.trials)


def _count_failed(study):
    return sum(t.failed for t in study.trials)


def test_optimize_should_return_a_study_whose_best_agrees_with_the_passed_abc_optimizer():
    """Given an explicit ABCOptimizer maximising over a mixed space, when optimize() runs 61 evaluations
    with it, then the Study best equals the optimizer best.

    Oracle: optimize docstring (the optimizer's evaluations are the trials), Optimizer docstring (scores
    in the user's direction), Study.best_trial. Study takes params from the candidates (Space.move for
    neighbours) and its best over the finite trials; the optimizer decodes its stored best x with to_dicts.
    The two agree only if the facade passed every score to the right candidate in the same direction.
    Inputs: direction 'max' checks the sign convention through all three components; 61 is not a multiple
    of the initial batch size 5 and covers more than one ABC cycle.
    """
    space = _small_discrete_space()

    study, opt = _run_integration_study(space)

    assert study.best_score == opt.best_score
    assert study.best_params == opt.best_params


def test_optimize_should_record_trials_in_order_whose_params_decode_their_x_with_an_abc_optimizer():
    """Given the same explicit ABCOptimizer run, when the trials are inspected, then they are numbered
    0..60 and every trial's params equal to_dicts of its x.

    Oracle: Trial fields, Candidate docstring (params are what the user's function receives for x),
    Space.to_dicts docstring, task specification (exactly max_evals trials). The params of neighbour
    trials come from Space.move and their x from the optimizer, so a disagreement between the optimizer
    and Space shows here as a mismatching trial number.
    """
    space = _small_discrete_space()

    study, opt = _run_integration_study(space)

    assert _trial_numbers(study) == list(range(INTEGRATION_EVALS))
    assert _decode_mismatches(space, study) == []


def test_optimize_should_give_a_cached_trial_the_score_of_the_earlier_trial_with_equal_params():
    """Given the same explicit ABCOptimizer run, when cached trials are compared with earlier real trials,
    then each cached trial has the earlier score.

    Oracle: optimize docstring and README Behaviour ('a repeated parameter set reuses its earlier score').
    The small Int and Categorical ranges make neighbour moves repeat parameter sets, so the run contains
    cached trials (asserted, so the check is not empty). A cache that stored the score under the wrong key
    or for the wrong candidate gives a cached trial a different score.
    """
    space = _small_discrete_space()

    study, opt = _run_integration_study(space)

    cached, wrong = _cache_report(study)
    assert wrong == []
    assert cached > 0


@pytest.mark.parametrize(
    ("direction", "pick"),
    [
        pytest.param("min", min, id="min-minus-inf-must-not-win"),
        pytest.param("max", max, id="max-plus-inf-must-not-win"),
    ],
)
def test_optimize_should_keep_the_best_finite_score_when_non_finite_scores_are_injected(direction, pick):
    """Given an objective returning NaN, +inf or -inf on every 4th real call, when optimize() runs 200
    evaluations, then optimizer and Study best are the best finite score.

    Fault injection: the non-finite values simulate failed or diverged training runs inside a normal run.
    Oracle: README Behaviour and Optimizer docstring (NaN or infinity is the worst value for the
    direction), Study.best_trial (finite trials only), Trial.failed. Inputs: all three non-finite values
    occur (asserted) in both directions, and each direction has one that would win if treated as an
    ordinary number (-inf for min, +inf for max), in the optimizer as well as in Study. Postconditions:
    both bests equal the extreme of the finite scores the stub returned, and exactly the injected real
    calls are failed non-cached trials.
    """
    space = _small_discrete_space()
    stub_objective_injecting_non_finite, finite, injected = _stub_objective_injecting_non_finite()
    opt = ABCOptimizer(space, direction=direction, colony_size=6, limit=3, seed=ANY_SEED)

    study = optimize(
        stub_objective_injecting_non_finite,
        space,
        direction=direction,
        max_evals=FAILURE_RUN_EVALS,
        optimizer=opt,
    )

    assert len(injected) >= len(INJECTED_VALUES)
    assert opt.best_score == pick(finite)
    assert study.best_score == pick(finite)
    assert _count_failed_real_calls(study) == len(injected)


@pytest.mark.parametrize("direction", ["min", "max"])
def test_optimize_should_leave_the_optimizer_best_at_the_worst_value_when_every_score_is_nan(direction):
    """Given an objective that always returns NaN, when optimize() runs ALL_NAN_EVALS evaluations, then
    every trial failed and the optimizer best is the worst value for the direction.

    Fault injection at its extreme: every training run fails. Oracle: Optimizer docstring (a NaN score
    becomes the worst value: +inf when minimising, -inf when maximising), Trial.failed. Study.best_trial
    raising ValueError in this state is covered by the Study error-path tests.
    """
    space = _small_discrete_space()
    opt = ABCOptimizer(space, direction=direction, seed=ANY_SEED)

    study = optimize(_nan_objective, space, direction=direction, max_evals=ALL_NAN_EVALS, optimizer=opt)

    assert opt.best_score == WORST_FOR[direction]
    assert _count_failed(study) == ALL_NAN_EVALS
