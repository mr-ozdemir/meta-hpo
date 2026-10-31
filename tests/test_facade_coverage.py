"""Coverage-driven tests of optimize(): branches and the stdlib coverage measurement.

Facade tests added after the stdlib coverage measurement: rejected inputs, supplied optimizer, cache, budget.

Each test closes a red line or a one-sided (yellow) branch of facade.py found by the baseline trace and
sys.monitoring branch probe: line 33 and edge 32L (max_evals below one), line 36 and both 35->36 edges
(other direction, other space), edge 34R (supplied optimizer), edges 43R and 45R (cache=False), and the
seen-cache definition-use pair (score stored, then reused).

Oracles: the optimize() docstring ('max_evals counts the calls of function', 'with cache=True a
repeated parameter set reuses the earlier score', 'The default optimizer is ABCOptimizer'), README Behaviour
(cache, same seed reproduces the same search), CHANGELOG ('a configuration is never trained twice') and the
task specification of optimize (exact max_evals trials even inside the initial batch, ValueError for
max_evals < 1 and for a mismatched optimizer space or direction, user exceptions propagate).

Unspecified, therefore not asserted: whether an optimizer built on an equal-content but distinct Space counts
as mismatched (the library compares identity; the test below uses a Space that also differs in content), and
which trial wins a tie of equal best scores (the scores below make ties irrelevant to the assertions).

Branch-coverage tests of metahpo.optimize, derived from the control flow graph below.

Control flow graph (course notation: nodes are the real line numbers of src/metahpo/facade.py, straight-line
code is merged, every decision has one T and one F edge, s and e are entry and exit, a raise goes to e; the
short-circuit operands of line 43 are the nodes 43a and 43b; loops use the while and for templates):

optimize, facade.py 16-50
    s   -> 32
    32  -T-> 33 -> e          max_evals < 1: raise ValueError
    32  -F-> 34
    34  -T-> 35               an optimizer was passed: use it
    34  -F-> 34d -> 35        default: ABCOptimizer(space, direction, seed)
    35  -T-> 36 -> e          direction differs (35a) or the space is another object (35b): raise ValueError
    35  -F-> 37-38 -> 39      study, seen = {}
    39  -T-> 40-41 -> 42      while fewer trials than max_evals: ask, cut to the remaining budget
    39  -F-> 50 -> e          return study
    42  -T-> 43a              for: next candidate
    42  -F-> 49 -> 39         for: candidates exhausted, tell them
    43a -T-> 43b              cache is true
    43a -F-> 44               cache is false: cached = False without looking up the key
    43b -T-> 44               key already seen
    43b -F-> 44               key not seen
    44  -T-> 44t -> 45        cached: score = seen[key]
    44  -F-> 44f -> 45        score = function(params)
    45  -T-> 46 -> 47-48      cache: seen.setdefault(key, score)
    45  -F-> 47-48
    47-48 -> 42               append the score and the Trial

Test requirements (branch coverage, one per T/F edge; 43a T is taken by every cache=True call and 43b T / F
coincide with 44 T / F when cache is true, so they are folded into FAC-11 and FAC-12):
    FAC-1   32 T            FAC-2   32 F            FAC-3   34 T            FAC-4   34 F
    FAC-5   35 T, as FAC-5a (direction differs) and FAC-5b (space is another object)
    FAC-6   35 F            FAC-7   39 T            FAC-8   39 F            FAC-9   42 T
    FAC-10  42 F            FAC-11  43b T and 44 T  FAC-12  44 F            FAC-13  43a F
    FAC-14  45 T            FAC-15  45 F
Boundary requirements of line 40 (not CFG edges, from the spec 'exact max_evals trials, even inside the
initial batch'):
    FAC-16  the slice cuts the batch            FAC-17  the slice keeps the whole batch

Coverage level: 17 of 17 feasible requirements (100 percent; 19 of 19 with FAC-5a and FAC-5b), all by this
module. Every requirement is feasible.

Note on FAC-13 and FAC-15: with cache=False the guard of line 45 never fills `seen`, so the short circuit of
43a is behaviourally redundant, and the reverse also holds; each edge is executed, but removing either guard
alone changes no output (only removing both does, which the cache=False test detects).

Unspecified, therefore not asserted: the optimizer state after a truncated initial batch (it stays in 'init'
with pending candidates); whether a structurally equal but distinct Space counts as 'the same space' (the code
compares identity; the spec says only 'mismatched optimizer space', so FAC-5b uses a genuinely different
space).

Box types (P-T4-08): white-box tests (max_evals < 1, cache=False) are designed from the edges above with
crafted inputs; black-box tests come from the README, the optimize docstring and the task spec.

Function calls are counted with plain functions that append to a local list. The default colony is
ABCOptimizer's colony_size=10, so the initial batch has DEFAULT_SN = 10 // 2 = 5 candidates.

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: One scenario per facade branch of the control flow graph and of the coverage measurement.
"""

import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space, optimize

ANY_SEED = 0


IRRELEVANT_SCORE = 0.0


DEFAULT_COLONY_SIZE = 10


DEFAULT_INITIAL_BATCH = DEFAULT_COLONY_SIZE // 2


class UserObjectiveError(Exception):
    """A user exception type that the library cannot confuse with its own ValueError or RuntimeError."""


def _two_parameter_space():
    """A linear Float and an Int: both decoders are exact, so params compare with ==."""
    return Space({"a": Float(-5.0, 5.0), "b": Int(0, 9)})


def _quadratic(params):
    """Plain objective with a single minimum at a = 0, b = 0."""
    return params["a"] ** 2 + params["b"]


def _sphere(params):
    """Sphere objective over every parameter."""
    return sum(v * v for v in params.values())


def _spy_objective(calls, score_of):
    """Spy objective: records every params dict in `calls` and returns score_of(params)."""

    def spy_objective(params):
        calls.append(params)
        return score_of(params)

    return spy_objective


def raising_objective(params):
    """Stub objective that raises the user's own exception on its first call."""
    raise UserObjectiveError("boom")


def _trial_rows(study):
    """(number, params, score, cached) of every trial, in trial order."""
    return [(t.number, t.params, t.score, t.cached) for t in study.trials]


def _trial_numbers(study):
    """The trial numbers of `study`, in trial order."""
    return [t.number for t in study.trials]


def _cached_flags(study):
    """The cached flag of every trial of `study`, in trial order."""
    return [t.cached for t in study.trials]


def _params_key(params):
    """A hashable form of a params dict."""
    return tuple(sorted(params.items()))


def _distinct_param_count(study):
    """Number of distinct params dicts among the trials."""
    return len({_params_key(t.params) for t in study.trials})


def _cache_reuse_violations(study):
    """Trial numbers that break the cache rule.

    The first trial of each parameter set must be uncached; every later trial of that set must be cached and
    carry the first trial's score.
    """
    first = {}
    violations = []
    for t in study.trials:
        key = _params_key(t.params)
        if key not in first:
            first[key] = t
            if t.cached:
                violations.append(t.number)
        elif not t.cached or t.score != first[key].score:
            violations.append(t.number)
    return violations


@pytest.mark.parametrize(
    "max_evals",
    [
        pytest.param(0, id="zero_first_invalid_value_below_the_minimum_one"),
        pytest.param(-1, id="negative_class"),
    ],
)
def test_optimize_should_raise_value_error_before_calling_the_function_when_max_evals_is_below_one(max_evals):
    """Given a budget below one, when optimize() is called, then ValueError is raised and f is never called.

    Oracle: task specification 'ValueError for max_evals < 1'. Inputs: 0 is the invalid value next to the
    smallest valid budget (decisive against a '< 0' bound), -1 stands for the negative class. Site: the
    max_evals check at the top of optimize (facade line 33, edge 32L). The keyword tells this ValueError
    apart from the 'same space' error of the same call.
    """
    calls = []
    spy_objective = _spy_objective(calls, lambda params: IRRELEVANT_SCORE)
    space = _two_parameter_space()

    with pytest.raises(ValueError, match="max_evals"):
        optimize(spy_objective, space, max_evals=max_evals, seed=ANY_SEED)

    assert calls == []


def test_optimize_should_raise_value_error_before_calling_the_function_when_the_optimizer_maximises_instead():
    """Given an optimizer built with direction 'max', when optimize(direction='min') gets it, then ValueError.

    Oracle: task specification 'ValueError for mismatched optimizer space/direction'. Input: the same Space
    object is passed, so only the direction operand of the check is true (site: facade line 35 left
    operand -> line 36). The function must not be called.
    """
    calls = []
    spy_objective = _spy_objective(calls, lambda params: IRRELEVANT_SCORE)
    space = _two_parameter_space()
    maximising = ABCOptimizer(space, direction="max", seed=ANY_SEED)

    with pytest.raises(ValueError, match="same space"):
        optimize(spy_objective, space, direction="min", optimizer=maximising)

    assert calls == []


def test_optimize_should_raise_value_error_before_calling_the_function_when_the_optimizer_has_another_space():
    """Given an optimizer built on a Space with other parameters, when optimize() gets it, then ValueError.

    Oracle: task specification 'ValueError for mismatched optimizer space/direction'. Input: both use
    direction 'min', so only the space operand decides (site: facade line 35 right operand -> line 36). The
    other Space differs in content, so the expectation holds whether the check uses identity or equality.
    """
    calls = []
    spy_objective = _spy_objective(calls, lambda params: IRRELEVANT_SCORE)
    space = _two_parameter_space()
    other_space = Space({"z": Float(0.0, 1.0)})
    elsewhere = ABCOptimizer(other_space, direction="min", seed=ANY_SEED)

    with pytest.raises(ValueError, match="same space"):
        optimize(spy_objective, space, direction="min", optimizer=elsewhere)

    assert calls == []


def test_optimize_should_drive_the_supplied_optimizer_when_its_space_and_direction_match():
    """Given a user-built ABCOptimizer on the same Space, when optimize() runs 40 evaluations, then it used
    it.

    Oracle: optimize() docstring and README ask/tell section: the optimizer is driven by ask/tell and reports
    best_params/best_score in the user's direction, which must agree with the Study built from the same
    tells. The ABC schedule (task specification) gives 5 initial + 5 employed + 5 onlooker evaluations, and
    the 16th evaluation finishes the scout step, so 40 > 16 evaluations leave cycle >= 1 for any seed.
    Site: facade line 34 right operand (supplied optimizer kept).
    """
    space = _two_parameter_space()
    supplied = ABCOptimizer(space, seed=4)
    budget = 40

    study = optimize(_quadratic, space, max_evals=budget, optimizer=supplied)

    assert len(study.trials) == budget
    assert supplied.best_score == study.best_score
    assert supplied.best_params == study.best_params
    assert supplied.cycle >= 1


def test_optimize_should_match_the_default_optimizer_when_the_supplied_one_has_default_settings():
    """Given ABCOptimizer(space, seed=4) supplied, when compared with optimize(seed=4), then the trials agree.

    Oracle: optimize() docstring 'The default optimizer is ABCOptimizer' with the task specification
    defaults (colony_size=10, limit=15), and README 'The same seed reproduces the same search'. Two seeded
    runs are compared with each other only; no trajectory is pinned.
    """
    space = _two_parameter_space()
    supplied = ABCOptimizer(space, seed=4)

    with_supplied = optimize(_quadratic, space, max_evals=40, optimizer=supplied)
    with_default = optimize(_quadratic, space, max_evals=40, seed=4)

    assert _trial_rows(with_supplied) == _trial_rows(with_default)


def test_optimize_should_call_the_function_for_every_trial_when_cache_is_false():
    """Given a space of only 4 parameter sets and cache=False, when 50 evaluations run, then f runs 50 times.

    Oracle: optimize() docstring makes reuse conditional on 'cache=True', and README Behaviour ties the cache
    to cache=True, so with cache=False every evaluation calls the function. Input: 4 distinct sets and 50
    evaluations force repeats, so a cache that ignores the flag would be visible. Sites: facade edges 43R
    (cache short circuit) and 45R (no store).
    """
    calls = []
    spy_objective = _spy_objective(calls, lambda params: params["k"] * params["n"])
    space = Space({"k": Categorical([1, 2]), "n": Int(1, 2)})
    budget = 50

    study = optimize(spy_objective, space, max_evals=budget, cache=False, seed=ANY_SEED)

    assert _distinct_param_count(study) <= 4
    assert len(calls) == budget
    assert study.n_calls == budget
    assert _cached_flags(study) == [False] * budget


def test_optimize_should_reuse_the_first_score_without_calling_the_function_when_a_parameter_set_repeats():
    """Given 4 possible parameter sets and the default cache, when 100 evaluations are allowed, then repeats
    reuse.

    Oracle: README Behaviour 'a repeated parameter set reuses its earlier score instead of training again',
    CHANGELOG 'a configuration is never trained twice', study.py 'cached repeats are not counted'. Input:
    score 10 * k + n + 0.5 differs per set, so a cached trial carrying another set's score is detected; 4
    sets are fewer than the 5 candidates of the default initial batch, so a repeat occurs for any seed and
    the run holds more trials than calls. Site: the seen-cache definition at facade line 42 used by the
    lookup at line 48.
    """
    calls = []
    spy_objective = _spy_objective(calls, lambda params: 10 * params["k"] + params["n"] + 0.5)
    space = Space({"k": Categorical([1, 2]), "n": Int(1, 2)})
    budget = 100

    study = optimize(spy_objective, space, max_evals=budget, seed=ANY_SEED)

    assert _cache_reuse_violations(study) == []
    assert len(calls) == _distinct_param_count(study) == space.size
    assert study.n_calls == len(calls)
    assert len(study.trials) > study.n_calls


@pytest.mark.parametrize(
    "max_evals",
    [
        pytest.param(1, id="minimum_valid_budget"),
        pytest.param(DEFAULT_INITIAL_BATCH, id="equals_the_initial_batch"),
        pytest.param(DEFAULT_INITIAL_BATCH + 1, id="one_past_the_initial_batch"),
    ],
)
def test_optimize_should_return_exactly_max_evals_numbered_trials_when_the_budget_is_near_the_initial_batch(
    max_evals,
):
    """Given budgets 1, 5 and 6, when optimize() runs, then the Study holds trials numbered 0..max_evals-1.

    Oracle: task specification 'exact max_evals trials (even inside the initial batch)' and the optimize()
    docstring 'max_evals counts the calls of function'. Inputs: the boundaries on both sides of the
    default initial batch of colony_size // 2 = 5 (the existing suite already checks 3). Site: the batch slice
    ask()[: max_evals - len(trials)].
    """
    space = Space({"x0": Float(-5.0, 5.0), "x1": Float(-5.0, 5.0)})

    study = optimize(_sphere, space, max_evals=max_evals, seed=ANY_SEED)

    assert _trial_numbers(study) == list(range(max_evals))


def test_optimize_should_propagate_the_user_exception_unchanged_when_the_function_raises():
    """Given a function that raises its own exception type, when optimize() calls it, then that type escapes.

    Oracle: task specification 'user exceptions propagate'. Input: a custom exception class shows that the
    error is neither wrapped nor replaced by a library error; its argument must be the original one.
    """
    space = _two_parameter_space()

    with pytest.raises(UserObjectiveError) as excinfo:
        optimize(raising_objective, space, seed=ANY_SEED)

    assert excinfo.value.args == ("boom",)


REPRO_SEED = 5


DEFAULT_SN = 10 // 2


IRRELEVANT_RETURN_SCORE = 1.0


CACHE_BUDGET = 20


def _unit_space_2d():
    """Two Float(0, 1) parameters: every decoded point is distinct, so the cache never hits."""
    return Space({"a": Float(0.0, 1.0), "b": Float(0.0, 1.0)})


def _constant_spy_objective(calls, score=IRRELEVANT_RETURN_SCORE):
    """Records every params dict it receives in `calls` and returns `score`."""

    def objective(params):
        calls.append(params)
        return score

    return objective


def _spy_objective_changing_per_call(calls):
    """Records every call and returns 10 * k + calls so far: a repeated call would give a new score."""

    def objective(params):
        calls.append(params)
        return 10 * params["k"] + len(calls)

    return objective


def _sum_of_squares(params):
    """Plain objective with a minimum at the origin."""
    return sum(v * v for v in params.values())


def _numbers(study):
    """Trial numbers in trial order."""
    return [t.number for t in study.trials]


def _cached_flags_in_order(study):
    """Trial.cached in trial order."""
    return [t.cached for t in study.trials]


def _distinct_params_count(study):
    """Number of distinct params dicts among the trials."""
    return len({tuple(sorted(t.params.items())) for t in study.trials})


def _expected_cached_flags(study):
    """False for the first trial of each params dict, True for every later trial of the same params."""
    seen, flags = set(), []
    for t in study.trials:
        key = tuple(sorted(t.params.items()))
        flags.append(key in seen)
        seen.add(key)
    return flags


def _scores_differing_from_first_occurrence(study):
    """Trial numbers whose score differs from the score of the first trial with the same params."""
    first, differing = {}, []
    for t in study.trials:
        key = tuple(sorted(t.params.items()))
        first.setdefault(key, t.score)
        if t.score != first[key]:
            differing.append(t.number)
    return differing


@pytest.mark.parametrize(
    "max_evals", [pytest.param(0, id="just-below-minimum"), pytest.param(-1, id="negative")]
)
def test_optimize_should_reject_a_budget_below_one_before_calling_the_function(
    max_evals,
):
    """A budget below one evaluation is rejected before any evaluation.

    Given a spy objective and max_evals = 0 (the boundary just below the valid minimum 1) or -1, crafted to
    force 32 T.
    When optimize is called.
    Then it raises ValueError (keyword 'max_evals') and the objective was never called.
    Covers: FAC-1
    Box: white
    Source: task spec (ValueError for max_evals < 1).
    """
    calls = []

    with pytest.raises(ValueError, match="max_evals"):
        optimize(_constant_spy_objective(calls), _unit_space_2d(), max_evals=max_evals, seed=ANY_SEED)

    assert calls == []


def test_optimize_should_run_like_a_supplied_abc_optimizer_with_the_same_seed_when_no_optimizer_is_given():
    """The default optimizer is ABCOptimizer: with the same seed it repeats a supplied ABCOptimizer's search.

    Given one space, the sum of squares and REPRO_SEED; run a uses the default optimizer (34 F), run b passes
    ABCOptimizer(space, seed=REPRO_SEED) (34 T, 35 F); 60 evaluations exceed the initial batch, so the main
    loop runs many times without cutting.
    When both runs are made.
    Then their score lists are equal, and the supplied optimizer's best score equals run b's best score (the
    supplied object did the search).
    Covers: FAC-2, FAC-3, FAC-4, FAC-6, FAC-7, FAC-8, FAC-9, FAC-10, FAC-12, FAC-14, FAC-17
    Box: black
    Source: optimize docstring ('The default optimizer is ABCOptimizer'); README ('The same seed reproduces
    the same search').
    """
    space = _unit_space_2d()
    supplied = ABCOptimizer(space, seed=REPRO_SEED)

    a = optimize(_sum_of_squares, space, max_evals=60, seed=REPRO_SEED)
    b = optimize(_sum_of_squares, space, max_evals=60, optimizer=supplied)

    assert a.scores.tolist() == b.scores.tolist()
    assert supplied.best_score == b.best_score


def test_optimize_should_raise_value_error_without_calling_the_function_when_optimizer_direction_differs():
    """An optimizer built for 'max' cannot serve a 'min' optimisation.

    Given the same space for both and ABCOptimizer(space, direction='max'); optimize asks for 'min' (35 T
    through the direction operand).
    When optimize is called.
    Then it raises ValueError (keyword 'same space') and the objective was never called.
    Covers: FAC-2, FAC-3, FAC-5
    Box: black
    Source: task spec (ValueError for a mismatched optimizer direction).
    """
    space = _unit_space_2d()
    opt = ABCOptimizer(space, direction="max", seed=ANY_SEED)
    calls = []

    with pytest.raises(ValueError, match="same space"):
        optimize(_constant_spy_objective(calls), space, direction="min", optimizer=opt)

    assert calls == []


def test_optimize_should_raise_value_error_without_calling_the_function_when_optimizer_space_differs():
    """An optimizer built on a different space is rejected even when the directions agree.

    Given ABCOptimizer on Space({'a': Float(0, 1), 'other': Categorical([1, 2])}) and optimize on the
    two-parameter space {'a', 'b'}, both 'min': the dimensions agree (2 and 2), so only a real space
    comparison detects the mismatch, and the direction operand is false, so the space operand decides (35 T
    through the second operand).
    When optimize is called.
    Then it raises ValueError (keyword 'same space') and the objective was never called.
    Covers: FAC-2, FAC-3, FAC-5
    Box: black
    Source: task spec (ValueError for a mismatched optimizer space).
    """
    opt = ABCOptimizer(Space({"a": Float(0.0, 1.0), "other": Categorical([1, 2])}), seed=ANY_SEED)
    calls = []

    with pytest.raises(ValueError, match="same space"):
        optimize(_constant_spy_objective(calls), _unit_space_2d(), optimizer=opt)

    assert calls == []


def test_optimize_should_reuse_the_first_score_of_a_repeated_parameter_set_when_cache_is_true():
    """With the cache on, a repeated parameter set gets its first score back and the function is not called.

    Given Space({'k': Categorical([1, 2])}) (only two parameter sets, so repeats occur even inside the initial
    batch) and an objective whose score changes on every call, so any extra call would show up as a new score.
    When optimize runs CACHE_BUDGET evaluations with cache=True (43a T, 43b and 44 both ways, 45 T).
    Then the function was called once per distinct parameter set, only repeats are marked cached, every repeat
    has the score of the first occurrence, and n_calls equals the number of calls.
    Covers: FAC-7, FAC-8, FAC-9, FAC-10, FAC-11, FAC-12, FAC-14
    Box: black
    Source: README Behaviour ('a repeated parameter set reuses its earlier score instead of training again');
    optimize docstring; Study.n_calls docstring ('cached repeats are not counted').
    """
    calls = []
    space = Space({"k": Categorical([1, 2])})

    study = optimize(_spy_objective_changing_per_call(calls), space, max_evals=CACHE_BUDGET, seed=ANY_SEED)

    assert len(calls) == _distinct_params_count(study)
    assert _cached_flags_in_order(study) == _expected_cached_flags(study)
    assert _scores_differing_from_first_occurrence(study) == []
    assert study.n_calls == len(calls)


def test_optimize_should_call_the_function_for_every_trial_when_cache_is_false_and_scores_change():
    """With the cache off, every trial calls the function, repeats included.

    Given the same two-choice space and per-call objective as the cache test.
    When optimize runs CACHE_BUDGET evaluations with cache=False (43a F short-circuits, 44 F, 45 F).
    Then the function was called CACHE_BUDGET times, no trial is cached, and n_calls is CACHE_BUDGET.
    Covers: FAC-12, FAC-13, FAC-15
    Box: white
    Source: optimize docstring (the score is reused only 'with cache=True'); Study.n_calls docstring.
    """
    calls = []
    space = Space({"k": Categorical([1, 2])})

    study = optimize(
        _spy_objective_changing_per_call(calls), space, max_evals=CACHE_BUDGET, seed=ANY_SEED, cache=False
    )

    assert len(calls) == CACHE_BUDGET
    assert _cached_flags_in_order(study) == [False] * CACHE_BUDGET
    assert study.n_calls == CACHE_BUDGET


@pytest.mark.parametrize(
    "max_evals",
    [pytest.param(1, id="minimum-valid-budget"), pytest.param(DEFAULT_SN - 1, id="one-below-the-batch")],
)
def test_optimize_should_cut_the_initial_batch_when_max_evals_is_smaller_than_it(max_evals):
    """A budget smaller than the initial batch gives exactly max_evals trials and calls.

    Given the default colony (DEFAULT_SN = 5 initial candidates) and max_evals = 1 (the valid minimum, 32 F)
    or 4 (one below the batch), so the slice of line 40 cuts the batch and the loop then ends (39 F).
    When optimize runs.
    Then there are exactly max_evals trials, numbered 0 .. max_evals - 1, and as many function calls.
    Covers: FAC-2, FAC-7, FAC-8, FAC-16
    Box: black
    Source: task spec ('exact max_evals trials, even inside the initial batch').
    """
    calls = []

    study = optimize(_constant_spy_objective(calls), _unit_space_2d(), max_evals=max_evals, seed=ANY_SEED)

    assert _numbers(study) == list(range(max_evals))
    assert len(calls) == max_evals


@pytest.mark.parametrize(
    "max_evals",
    [pytest.param(DEFAULT_SN, id="exactly-the-batch"), pytest.param(DEFAULT_SN + 4, id="batch-plus-singles")],
)
def test_optimize_should_keep_whole_batches_when_max_evals_is_not_smaller_than_the_initial_batch(max_evals):
    """A budget of at least the initial batch is spent exactly without cutting a batch.

    Given the default colony (5 initial candidates) and max_evals = 5 (the batch fits exactly) or 9 (four
    single asks follow the batch), so the slice of line 40 never cuts.
    When optimize runs.
    Then there are exactly max_evals trials, numbered 0 .. max_evals - 1, and as many function calls.
    Covers: FAC-7, FAC-8, FAC-17
    Box: black
    Source: task spec ('exact max_evals trials').
    """
    calls = []

    study = optimize(_constant_spy_objective(calls), _unit_space_2d(), max_evals=max_evals, seed=ANY_SEED)

    assert _numbers(study) == list(range(max_evals))
    assert len(calls) == max_evals
