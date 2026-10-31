"""README and CHANGELOG examples run end to end as system and acceptance tests.

Ownership: this module owns the acceptance tests of README.md (Quick start, Ask and tell, Search space,
Behaviour) and of the CHANGELOG [Unreleased] Added list. When either document changes, this module must be
updated with it. Each test runs a README user journey through the public package only (import metahpo as
mh, as in the README): a real Space, the real ABCOptimizer and the real optimize(); only the user's train
function is replaced by a cheap deterministic stub. Nothing inside metahpo is patched or read privately.

Realistic data: the Space is the README Quick start Space verbatim, and stub_objective is an
accuracy-like surface over it, 0.9 - 0.05 * (log10(lr) + 3) ** 2 - ((units - 128) / 256) ** 2
- (dropout - 0.2) ** 2 plus 0.02 for kernel 5 and 0.01 for kernel 7, with its peak 0.92 at lr = 1e-3,
kernel = 5, units = 128, dropout = 0.2. The peak lies inside every range, but ABC neighbour moves still
clip at u = 0 and u = 1, so a run can reach the bounds wherever the peak is.
Failures are modelled as a user sees them: a diverged run returns NaN, an overflowing metric returns an
infinity.

Seeds and budgets: README_SEED = 1, README_MAX_EVALS = 300 and the ask/tell configuration are the README
values. BEST_NOT_LAST_MAX_EVALS = 150 is the one non-README budget of the Quick start: with 300 evaluations
the last trial can lie next to the best params, which would hide a best_params that returns the last
trial; the test using 150 asserts as a precondition that the last trial does not hold the extreme score.
No assertion compares with a value captured from a seeded run; every check is a property that a correct
library satisfies for any seed (counts, types, ranges, best equals the extreme of the finite scores,
equality of two runs). A few tests assert a precondition first (for example that the seeded run
did reach the failing region), so a different trajectory makes them fail loudly instead of passing
vacuously.

Cost and reliability (system tests are slower and more fragile than unit tests): at most 300 evaluations
per run, a 300-evaluation run takes a few tens of milliseconds, the module finishes in about a second,
there is no wall clock, no sleep, no shared mutable module state, and every test builds its own Space.

Known defect behind the two range tests: with README_SEED = 1 the search never reaches the lr upper bound
(its largest lr in both the optimize run and the ask/tell loop stays near 2.3e-3), so the two range tests
pass. About half of the seeds (30 of seeds 0..59, the first is seed 2) do reach lr at u = 1.0 and then
receive lr = 0.010000000000000004 > 1e-2, the reported log-Float upper-bound defect (site: log branch of
_decode_float / Float.decode_one in space.py; listed in docs/test-izlenebilirlik.md). A legitimate
change of the trajectory (another seed, uniforms drawn one by one) can therefore turn a range test red;
that red is this defect, not flakiness. The seed is kept and the range assertions are not loosened.

CHANGELOG Added traceability:
1. Space with Float, Int and Categorical:
   test_optimize_should_give_every_trial_typed_in_range_params_when_running_the_quick_start.
   test_optimize_should_sample_lr_on_both_sides_of_its_geometric_midpoint_when_running_the_quick_start
   adds the realistic two-decade lr range. The wording "decoded ... in vectorised numpy" is an
   implementation detail that cannot be observed from the user side, so it is not covered here.
2. ABCOptimizer behind ask/tell: the four tests whose names contain "ask_tell". The bullet's other part,
   "following the phase order of the original implementation", is an internal schedule and is covered by
   the ABC unit tests (tests/test_abc.py), not by this module.
3. optimize() returning a Study with every trial and the best one:
   test_optimize_should_return_a_study_with_exactly_max_evals_numbered_trials_when_running_the_quick_start and
   test_optimize_should_report_the_extreme_finite_score_and_its_params_as_best.
4. Caching, a configuration is never trained twice: the four tests whose names contain "cache", one of
   them on a small discrete space where repeats are forced by the pigeonhole principle.
5. NaN and infinite scores are the worst value: the tests whose names contain "nan", "infinity" or
   "failed".

Unspecified, not asserted:
- which of several different parameter sets with the same best score is reported as best, and, on the
  ask/tell side, which of several candidates told the same best score is reported (tie order);
- the distribution of log-scaled values inside the closed range (only that lr spans its two decades);
- best_params and best_score of an ABCOptimizer before any finite score was told;
- whether different seeds give different searches, and what seed=None does;
- whether non-finite scores are cached (cache assertions therefore use objectives without NaN).

Input space: Every configuration of the README and CHANGELOG examples.
Subset: The README Quick start and Ask and tell examples, run with stub objectives and fixed seeds.
"""

import math

import pytest

import metahpo as mh

README_SEED = 1
README_MAX_EVALS = 300
README_COLONY_SIZE = 10
README_LIMIT = 15
QUICK_START_KEYS = {"lr", "kernel", "units", "dropout"}
KERNEL_BONUS = {3: 0.0, 5: 0.02, 7: 0.01}
DIVERGES_ABOVE_DROPOUT = 0.4
OVERFLOWS_BEYOND = 0.9
SMALL_COLONY = 4
ANY_SEED = 0
FINITE_SCORE = 0.3
ALL_FAILED_EVALS = 5
LOG_DECODE_REL = 1e-12
BEST_NOT_LAST_MAX_EVALS = 150
LR_GEOMETRIC_MIDPOINT = 1e-3
PIGEONHOLE_INT_LOW = 0
PIGEONHOLE_INT_HIGH = 3
PIGEONHOLE_CHOICES = ["x", "y"]
PIGEONHOLE_MAX_EVALS = 50
CHOICE_BONUS = {"x": 0.0, "y": 0.5}


def _quick_start_space():
    """A fresh copy of the README Quick start Space, written exactly as in the README."""
    return mh.Space(
        {
            "lr": mh.Float(1e-4, 1e-2, log=True),
            "kernel": mh.Categorical([3, 5, 7]),
            "units": mh.Int(16, 256),
            "dropout": mh.Float(0.0, 0.5),
        }
    )


def stub_objective(params):
    """Stub for the README train(): a smooth accuracy-like score with its peak 0.92 inside the space."""
    lr_term = 0.05 * (math.log10(params["lr"]) + 3.0) ** 2
    units_term = ((params["units"] - 128) / 256) ** 2
    dropout_term = (params["dropout"] - 0.2) ** 2
    return 0.9 - lr_term - units_term - dropout_term + KERNEL_BONUS[params["kernel"]]


def stub_kernel_objective(params):
    """Stub train() on the kernel space: 0.9 plus the kernel bonus, a distinct score for each kernel."""
    return 0.9 + KERNEL_BONUS[params["kernel"]]


def stub_discrete_objective(params):
    """Stub train() on the pigeonhole space: a distinct finite score for each of the 8 combinations."""
    return params["units"] + CHOICE_BONUS[params["activation"]]


def stub_loss(params):
    """Stub for a train() that returns a loss to minimise: the negated accuracy of stub_objective."""
    return -stub_objective(params)


def stub_objective_diverging(params):
    """Stub for a train() whose run diverges (NaN) when dropout exceeds DIVERGES_ABOVE_DROPOUT."""
    if params["dropout"] > DIVERGES_ABOVE_DROPOUT:
        return math.nan
    return stub_objective(params)


def stub_objective_overflowing_when_maximising(params):
    """Stub accuracy that overflows to +inf exactly where it would exceed OVERFLOWS_BEYOND, near the peak."""
    score = stub_objective(params)
    return math.inf if score > OVERFLOWS_BEYOND else score


def stub_loss_overflowing_when_minimising(params):
    """Stub loss that overflows to -inf exactly where it would fall below -OVERFLOWS_BEYOND, near the peak."""
    loss = stub_loss(params)
    return -math.inf if loss < -OVERFLOWS_BEYOND else loss


def dummy_failing_objective(params):
    """Stub for a train() whose every run diverges."""
    return math.nan


def _spy(objective):
    """Wraps objective in a spy that records a copy of every params dict it is called with."""
    calls = []

    def spy_objective(params):
        calls.append(dict(params))
        return objective(params)

    return spy_objective, calls


def _run_quick_start(objective, direction="max", cache=True):
    """The README Quick start call with the given stub in place of train."""
    space = _quick_start_space()
    return mh.optimize(
        objective, space, direction=direction, max_evals=README_MAX_EVALS, seed=README_SEED, cache=cache
    )


def _kernel_space():
    """The Quick start kernel parameter alone: 3 parameter sets, fewer than the 5 of the default initial
    batch (colony_size // 2)."""
    return mh.Space({"kernel": mh.Categorical([3, 5, 7])})


def _run_kernel_only(objective, cache=True):
    """The README Quick start call on the kernel space, with the given stub in place of train."""
    return mh.optimize(
        objective, _kernel_space(), direction="max", max_evals=README_MAX_EVALS, seed=README_SEED, cache=cache
    )


def _pigeonhole_space():
    """A discrete Space with (PIGEONHOLE_INT_HIGH - PIGEONHOLE_INT_LOW + 1) * 2 = 8 combinations."""
    return mh.Space(
        {
            "units": mh.Int(PIGEONHOLE_INT_LOW, PIGEONHOLE_INT_HIGH),
            "activation": mh.Categorical(PIGEONHOLE_CHOICES),
        }
    )


def _run_readme_ask_tell_loop(objective):
    """The README ask/tell loop with the given stub in place of train; returns (opt, told) where told lists
    every (candidate, score) pair in the order it was told."""
    space = _quick_start_space()
    opt = mh.ABCOptimizer(
        space, direction="max", colony_size=README_COLONY_SIZE, limit=README_LIMIT, seed=README_SEED
    )
    told = []
    evals = 0
    while evals < README_MAX_EVALS:
        candidates = opt.ask()
        scores = [objective(c.params) for c in candidates]
        opt.tell(candidates, scores)
        told.extend(zip(candidates, scores, strict=True))
        evals += len(candidates)
    return opt, told


def _range_violations(params):
    """Names of the README Search space rules that params breaks (keys, Python type, closed range)."""
    lr, kernel = params.get("lr"), params.get("kernel")
    units, dropout = params.get("units"), params.get("dropout")
    rules = {
        "keys": set(params) == QUICK_START_KEYS,
        "lr": type(lr) is float and 1e-4 <= lr <= 1e-2,
        "kernel": type(kernel) is int and kernel in (3, 5, 7),
        "units": type(units) is int and 16 <= units <= 256,
        "dropout": type(dropout) is float and 0.0 <= dropout <= 0.5,
    }
    return sorted(name for name, ok in rules.items() if not ok)


def _trials_breaking_the_space(study):
    """(trial number, broken rules) for every trial whose params break the README Search space rules."""
    return [(t.number, _range_violations(t.params)) for t in study.trials if _range_violations(t.params)]


def _candidates_breaking_the_space(told):
    """(candidate id, broken rules) for every told candidate whose params break the Search space rules."""
    return [(c.id, _range_violations(c.params)) for c, _ in told if _range_violations(c.params)]


def _largest_finite_told_score(told):
    """The largest finite score among the told (candidate, score) pairs."""
    return max(s for _, s in told if math.isfinite(s))


def _params_of_first_candidate_scoring(told, score):
    """params of the earliest told candidate whose score equals score."""
    return next(c.params for c, s in told if s == score)


def _smallest_lr(study):
    """The smallest lr any trial of study received."""
    return min(t.params["lr"] for t in study.trials)


def _largest_lr(study):
    """The largest lr any trial of study received."""
    return max(t.params["lr"] for t in study.trials)


def _params_of_first_trial_scoring(study, score):
    """params of the earliest trial whose score equals score."""
    return next(t.params for t in study.trials if t.score == score)


def _extreme_finite_score(study, pick):
    """pick (max or min) of the finite scores of study."""
    return pick(t.score for t in study.trials if math.isfinite(t.score))


def _cached_trials_with_a_different_score(study):
    """Numbers of the cached trials whose score differs from the earliest real trial with equal params."""
    first_score = {}
    wrong = []
    for trial in study.trials:
        params_key = tuple(sorted(trial.params.items()))
        if trial.cached and first_score.get(params_key) != trial.score:
            wrong.append(trial.number)
        if not trial.cached:
            first_score.setdefault(params_key, trial.score)
    return wrong


def _distinct(params_list):
    """Number of pairwise different parameter dicts in params_list."""
    return len({tuple(sorted(p.items())) for p in params_list})


def _trial_records(study):
    """(params, score, cached) of every trial of study, in trial order."""
    return [(t.params, t.score, t.cached) for t in study.trials]


def _failed_flag_mismatches(study):
    """Numbers of the trials whose failed flag disagrees with 'the score is NaN or infinite'."""
    return [t.number for t in study.trials if t.failed != (not math.isfinite(t.score))]


def test_optimize_should_return_a_study_with_exactly_max_evals_numbered_trials_when_running_the_quick_start():
    """Given the README Quick start (max_evals=300, seed=1) with a stub train, when optimize() runs, then it
    returns a Study holding 300 Trials numbered 0..299 in order.

    Oracle: README Quick start, the optimize docstring (max_evals counts the calls of function) and
    the CHANGELOG ('returning a Study with every trial'). 300 is decisive because the ABC initial batch has
    colony_size // 2 = 5 candidates, so a facade counting ask() rounds instead of trials would overshoot.
    """
    expected_numbers = list(range(README_MAX_EVALS))

    study = _run_quick_start(stub_objective)

    assert type(study) is mh.Study
    assert [t.number for t in study.trials] == expected_numbers
    assert {type(t) for t in study.trials} == {mh.Trial}


@pytest.mark.parametrize(
    ("direction", "objective", "pick"),
    [
        pytest.param("max", stub_objective, max, id="max-accuracy-best-is-largest"),
        pytest.param("min", stub_loss, min, id="min-loss-best-is-smallest"),
    ],
)
def test_optimize_should_report_the_extreme_finite_score_and_its_params_as_best(direction, objective, pick):
    """Given the Quick start search in either direction with BEST_NOT_LAST_MAX_EVALS = 150, when optimize()
    finishes, then best_score is the largest (max) or smallest (min) score of the study and best_params are
    the params that scored it.

    Oracle: README ('searches for the parameters that maximise or minimise that score') and the CHANGELOG
    ('a Study with every trial and the best one'). Both directions are cases because the stub scores vary,
    so a picker ignoring the direction reports the opposite extreme; ties between different parameter sets
    are practically impossible on the continuous stub, so 'the earliest trial with that score' is unique.
    Budget 150 instead of the README 300: at 300 the last trial is a cached repeat of the best params, so a
    best_params that returns the last trial's params would pass; the precondition asserts, without reading
    best_*, that the last trial does not hold the extreme score, which makes the best_params check decisive
    against that defect (site: Study.best_params).
    """
    space = _quick_start_space()

    study = mh.optimize(
        objective, space, direction=direction, max_evals=BEST_NOT_LAST_MAX_EVALS, seed=README_SEED
    )
    assert study.trials[-1].score != _extreme_finite_score(study, pick)

    best_score, best_params = study.best_score, study.best_params

    assert best_score == _extreme_finite_score(study, pick)
    assert best_params == _params_of_first_trial_scoring(study, best_score)


def test_optimize_should_give_every_trial_typed_in_range_params_when_running_the_quick_start():
    """Given the Quick start Space mixing a log Float, a Categorical, an Int and a linear Float, when the
    README search runs, then every trial's params and best_params have exactly the Space keys, plain Python
    types and values inside the declared closed ranges.

    Oracle: README Search space table (Float in [low, high], Int in low..high inclusive, Categorical one of
    choices) and the Space.rows docstring ('params ... with Python types'). 'type(v) is' is decisive
    against numpy scalars leaking out of the numpy decode; reaching the endpoints is not asserted.
    Reliability: seed 1 never reaches the lr upper bound, while about half of the seeds do and then fail
    on the reported log-Float upper-bound defect (lr = 0.010000000000000004; site: log branch of
    _decode_float / Float.decode_one in space.py; listed in docs/test-izlenebilirlik.md). A red run
    of this test is therefore that defect, not flakiness.
    """
    study = _run_quick_start(stub_objective)

    broken = _trials_breaking_the_space(study)
    best_broken = _range_violations(study.best_params)

    assert broken == []
    assert best_broken == []


def test_optimize_should_sample_lr_on_both_sides_of_its_geometric_midpoint_when_running_the_quick_start():
    """Given the Quick start Space with lr = Float(1e-4, 1e-2, log=True), a two-decade range, when the README
    search runs, then some trials receive an lr below its geometric midpoint 1e-3 and some an lr above it.

    Oracle: README Quick start (lr declared log-scaled over two decades) and Search space table, with
    realistic data as a user would see it. The stub peak sits at lr = 1e-3, so the search explores both
    decades around it; this is a fidelity check of realistic data, not a decisive check of the log decode,
    which the Space unit tests own. Wholeness of units is covered by 'type(units) is int' in the range test.
    """
    study = _run_quick_start(stub_objective)

    smallest, largest = _smallest_lr(study), _largest_lr(study)

    assert smallest < LR_GEOMETRIC_MIDPOINT
    assert largest > LR_GEOMETRIC_MIDPOINT


def test_optimize_should_mark_diverged_trials_failed_and_never_pick_one_as_best():
    """Given the Quick start search with a train() that returns NaN (a diverged run) whenever dropout is
    above 0.4, when optimize() finishes, then exactly the trials with a non-finite score are failed and the
    best trial is not one of them.

    Oracle: README Behaviour ('a score of NaN ... counts as the worst possible value'), Trial.failed
    docstring ('True when the user's function returned NaN or an infinite value'). The precondition shows
    the seeded run did reach the diverging region and also produced finite scores.
    """
    study = _run_quick_start(stub_objective_diverging)
    assert any(t.failed for t in study.trials) and any(not t.failed for t in study.trials)

    best = study.best_trial

    assert _failed_flag_mismatches(study) == []
    assert best.failed is False
    assert math.isfinite(best.score)


@pytest.mark.parametrize(
    ("direction", "objective", "sign"),
    [
        pytest.param(
            "max", stub_objective_overflowing_when_maximising, 1.0, id="max-plus-infinity-at-the-peak"
        ),
        pytest.param("min", stub_loss_overflowing_when_minimising, -1.0, id="min-minus-infinity-at-the-peak"),
    ],
)
def test_optimize_should_treat_an_infinity_in_the_good_direction_as_the_worst_score(
    direction, objective, sign
):
    """Given a train() that overflows to +inf when maximising (or -inf when minimising) exactly where its
    score would pass OVERFLOWS_BEYOND, when the Quick start search runs, then the reported best score is a
    finite value that does not pass OVERFLOWS_BEYOND.

    Oracle: README Behaviour and CHANGELOG ('NaN and infinite scores are treated as the worst value for the
    chosen direction'). Decisive input: an infinity on the good side would be the best value if it were
    compared naively. sign turns both directions into 'sign * best_score <= OVERFLOWS_BEYOND'. The
    precondition shows the seeded run did produce overflowing trials.
    """
    study = _run_quick_start(objective, direction=direction)
    assert any(t.failed for t in study.trials)

    best_score = study.best_score

    assert math.isfinite(best_score)
    assert sign * best_score <= OVERFLOWS_BEYOND


@pytest.mark.parametrize(
    ("direction", "bad_score"),
    [
        pytest.param("max", math.inf, id="max-plus-infinity"),
        pytest.param("max", -math.inf, id="max-minus-infinity"),
        pytest.param("max", math.nan, id="max-nan"),
        pytest.param("min", -math.inf, id="min-minus-infinity"),
        pytest.param("min", math.inf, id="min-plus-infinity"),
        pytest.param("min", math.nan, id="min-nan"),
    ],
)
def test_ask_tell_should_report_the_finite_score_as_best_when_a_nan_or_infinity_was_told_first(
    direction, bad_score
):
    """Given an ABCOptimizer on the Quick start Space with colony_size=4 (an initial batch of 2), when the
    batch is told [non-finite score, 0.3] in that order, then best_score is 0.3 and best_params are the
    params of the second candidate, for every non-finite value in both directions.

    Oracle: README Behaviour (NaN or infinity is the worst value for the direction) and the Optimizer
    docstring (scores in the user's direction). The bad score is told first so that it is the only score
    seen before the finite one; the cases on the good side (+inf for max, -inf for min) are the decisive
    ones, the others are consistency cases.
    """
    opt = mh.ABCOptimizer(_quick_start_space(), direction=direction, colony_size=SMALL_COLONY, seed=ANY_SEED)
    first, second = opt.ask()

    opt.tell([first, second], [bad_score, FINITE_SCORE])

    assert opt.best_score == FINITE_SCORE
    assert opt.best_params == second.params


@pytest.mark.parametrize("attribute", ["best_trial", "best_params", "best_score"])
def test_study_should_raise_value_error_for_the_best_when_every_trial_failed(attribute):
    """Given a Quick start search in which every training run diverges (NaN), when the user asks the Study
    for its best, then ValueError is raised instead of a failed trial being reported as best.

    Oracle: README Behaviour (NaN is the worst value) and Study.best_trial ('no trial returned a finite
    score'); best_params and best_score are read through best_trial, so the same rule applies to them.
    """
    study = mh.optimize(
        dummy_failing_objective,
        _quick_start_space(),
        direction="max",
        max_evals=ALL_FAILED_EVALS,
        seed=ANY_SEED,
    )
    assert all(t.failed for t in study.trials)

    with pytest.raises(ValueError):
        getattr(study, attribute)


def test_ask_tell_should_report_the_best_told_score_and_its_candidate_params_when_running_the_readme_loop():
    """Given the README ask/tell loop verbatim (direction='max', colony_size=10, limit=15, seed=1, 300
    evaluations) with a stub train, when the loop ends, then best_score is the largest score told and
    best_params are the params of the first candidate told that score.

    Oracle: README Ask and tell section and Optimizer.best_params / best_score (best in the user's
    direction). Comparing with the params of the first candidate told that score is decisive against
    reporting a candidate with a different score; several candidates share the largest score with equal
    params, so the tie order (first or last told) cannot be observed here and is not asserted. lr is
    log-scaled, and Candidate.params and best_params may decode it along different float
    paths, so values are compared with pytest.approx at a relative tolerance of 1e-12; ints compare exactly.
    The loop itself stops on the evaluation count, so that count is not asserted.
    """
    opt, told = _run_readme_ask_tell_loop(stub_objective)
    largest = _largest_finite_told_score(told)
    first_params = _params_of_first_candidate_scoring(told, largest)

    best_score, best_params = opt.best_score, opt.best_params

    assert best_score == largest
    assert best_params == pytest.approx(first_params, rel=LOG_DECODE_REL)
    assert set(best_params) == QUICK_START_KEYS


def test_ask_tell_should_hand_out_candidates_with_the_space_keys_python_types_and_ranges_in_the_readme_loop():
    """Given the README ask/tell loop with a stub train, when it runs, then every candidate is a Candidate
    whose params have the Space keys, Python types and values inside the declared ranges.

    Oracle: README Search space table, Candidate docstring ('params as the user's function receives them')
    and Space.rows docstring (Python types). Reliability: seed 1 never reaches the lr upper bound, while
    about half of the seeds do and then fail on the reported log-Float upper-bound defect
    (lr = 0.010000000000000004; site: log branch of _decode_float / Float.decode_one in space.py; listed in
    docs/test-izlenebilirlik.md). A red run of this test is therefore that defect, not flakiness.
    """
    opt, told = _run_readme_ask_tell_loop(stub_objective)

    broken = _candidates_breaking_the_space(told)
    candidate_types = {type(c) for c, _ in told}

    assert broken == []
    assert candidate_types == {mh.Candidate}


def test_optimize_should_reuse_the_earlier_score_for_a_repeated_parameter_set_when_cache_is_default():
    """Given the Quick start kernel choices alone with the default cache, when a parameter set repeats,
    then the repeated trial is cached and carries the score of the earliest real trial with equal params.

    Oracle: README Behaviour ('a repeated parameter set reuses its earlier score instead of training
    again') and the optimize docstring. Decisive input: the default initial batch hands out 5 candidates
    over 3 parameter sets, so by the pigeonhole principle a repeat occurs for any seed; the precondition
    asserts it. A Float space would not do: the optimizer hands out a repeat only when it finds no new
    parameter set (Optimizer docstring). The stub never returns NaN, so score equality is meaningful.
    """
    study = _run_kernel_only(stub_kernel_objective)
    assert any(t.cached for t in study.trials)

    wrong = _cached_trials_with_a_different_score(study)

    assert wrong == []


def test_optimize_should_never_call_train_twice_with_equal_params_when_cache_is_default():
    """Given the Quick start search with the default cache and a spy train, when it runs, then train is
    called exactly max_evals = 300 times, each time with a new parameter set, and Study.n_calls equals the
    real number of calls.

    Oracle: CHANGELOG ('a configuration is never trained twice'), README Behaviour (a repeat does not count
    against max_evals), the optimize docstring ('max_evals counts the calls of function') and the
    Study.n_calls docstring ('cached repeats are not counted'). The Quick start space has two Floats, so it
    is never exhausted and the whole budget is spent on calls.
    """
    spy_objective, calls = _spy(stub_objective)

    study = _run_quick_start(spy_objective)

    assert _distinct(calls) == len(calls)
    assert study.n_calls == len(calls)
    assert len(calls) == README_MAX_EVALS


def test_optimize_should_train_each_combination_at_most_once_when_cache_is_default_on_a_small_space():
    """Given a discrete Space Int(0, 3) x Categorical(['x', 'y']) with 8 combinations, a spy train and
    max_evals = 50, when optimize() runs with the default cache, then train is called with pairwise
    different params, once for each of the 8 combinations, the run stops with fewer than 50 trials,
    Study.n_calls equals the real calls, and every cached trial carries the score of the earliest real
    trial with equal params.

    Oracle: CHANGELOG ('a configuration is never trained twice'), README Behaviour (a repeated parameter
    set reuses its earlier score), the optimize docstring (the search stops early when every parameter set
    of a space without Float has been evaluated) and the Study.n_calls docstring. Decisive input: a budget
    of 50 calls could train every combination several times, so only the cache and the early stop keep the
    calls at 8; the 8 < 50 relation is asserted first. The stub never returns NaN, so score equality is
    meaningful.
    """
    combinations = (PIGEONHOLE_INT_HIGH - PIGEONHOLE_INT_LOW + 1) * len(PIGEONHOLE_CHOICES)
    assert combinations < PIGEONHOLE_MAX_EVALS
    spy_objective, calls = _spy(stub_discrete_objective)

    study = mh.optimize(spy_objective, _pigeonhole_space(), max_evals=PIGEONHOLE_MAX_EVALS, seed=README_SEED)

    assert len(study.trials) < PIGEONHOLE_MAX_EVALS
    assert study.n_calls == len(calls)
    assert len(calls) == combinations
    assert _distinct(calls) == len(calls)
    assert _cached_trials_with_a_different_score(study) == []


def test_optimize_should_call_train_for_every_trial_when_cache_is_false():
    """Given the Quick start kernel choices alone with cache=False and a spy train, when the search revisits
    parameter sets, then train is called for every one of the 300 trials and no trial is marked cached.

    Oracle: README Behaviour (the reuse happens 'with cache=True'), the optimize docstring and the
    Study.n_calls docstring. Pair of the default-cache kernel test above: without the cache every repeat is
    trained again, so 3 parameter sets do not end the run early. The precondition shows parameter sets did
    repeat, so a cache that ignored the flag would be caught.
    """
    spy_objective, calls = _spy(stub_kernel_objective)

    study = _run_kernel_only(spy_objective, cache=False)
    assert _distinct([t.params for t in study.trials]) < len(study.trials)

    assert len(calls) == README_MAX_EVALS
    assert study.n_calls == README_MAX_EVALS
    assert [t.cached for t in study.trials] == [False] * README_MAX_EVALS


def test_optimize_should_reproduce_the_same_search_when_the_seed_is_the_same():
    """Given two Quick start runs with seed=1, each on its own freshly built Space, when both finish, then
    they hold the same trials (params, score, cached flag) in the same order and the same best.

    Oracle: README Behaviour ('The same seed reproduces the same search'). A fresh Space per run checks that
    reproducibility does not rest on shared state. Whether different seeds differ is unspecified.
    """
    first = _run_quick_start(stub_objective)

    second = _run_quick_start(stub_objective)

    assert _trial_records(second) == _trial_records(first)
    assert second.best_params == first.best_params
    assert second.best_score == first.best_score


def test_ask_tell_should_reproduce_the_same_candidates_when_the_seed_is_the_same():
    """Given two README ask/tell loops with seed=1, each on its own freshly built Space, when both finish,
    then they handed out the same candidate params in the same order and report the same best score.

    Oracle: README Behaviour ('The same seed reproduces the same search') applied to the README Ask and
    tell section.
    """
    first_opt, first_told = _run_readme_ask_tell_loop(stub_objective)

    second_opt, second_told = _run_readme_ask_tell_loop(stub_objective)

    assert [c.params for c, _ in second_told] == [c.params for c, _ in first_told]
    assert second_opt.best_score == first_opt.best_score
