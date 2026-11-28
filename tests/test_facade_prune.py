"""Integration tests of pruning through optimize() and resume(): optimize(pruner=...) and resume(pruner=...).

Oracles: the optimize docstring (an objective with a parameter named `trial` gets a RunningTrial; TrialPruned
ends a trial as pruned; MedianPruner() for None; TypeError for anything but a Pruner; every run its own copy;
the initial population never pruned; a pruned trial counts as a call, scores its last reported value, is never
the best trial and is pruned again as a cached repeat; the warnings before and after the search), the resume
docstring (restored trials keep their state and values and the pruner is not asked again; the recorded pruner
goes on, a pruner of another class needs `pruner=`, a folder without one uses MedianPruner(); `pruner=`
replaces it), the Optimizer docstring (`pruned` passed only to a tell() with that parameter, NaN otherwise),
the Patience and TargetScore docstrings, and the report module's trials.jsonl and config.json records.

Expected trials come from objectives whose pruning is scripted by call number or parameter value, from a
pruner that always prunes, or from an uninterrupted run with the same seed that the resumed run must equal.
The default results folder and console are switched off through METAHPO_OUTPUT and METAHPO_VERBOSE in
conftest; the tests that read the results folder write into tmp_path with plots off.

Unspecified, therefore not asserted: which trials MedianPruner prunes in a real search (test_prune.py checks
the rule), the wall-clock fields of the results files, and the order of trials after the first pruned one
beyond what the uninterrupted twin shows.

Input space: Every objective signature, reported step sequence, pruner, optimizer, budget, cache flag, runs
value and results folder, which is unbounded. Subset: The three signatures that take a trial (keyword,
keyword-only, positional-only) and two that do not; pruning by the pruner, by the objective without a report
and in a cached repeat; the three optimizers with their initial populations; a scripted optimizer with and
without `pruned`; Patience and TargetScore against pruned values better than every finite one; runs 1 and 2;
interrupts after 7 and 23 calls; folders that record MedianPruner, NopPruner, a pruner of one's own, rejected
settings and no pruner.
"""

import json
import math
import warnings

import pytest

import metahpo
from metahpo import (
    ABCOptimizer,
    BBPSOOptimizer,
    Callback,
    Float,
    Int,
    MedianPruner,
    MetaHpoWarning,
    NopPruner,
    Optimizer,
    Patience,
    PercentilePruner,
    Pruner,
    PSOOptimizer,
    RunningTrial,
    Space,
    TargetScore,
    TrialPruned,
    optimize,
    resume,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*ended without a finite score:metahpo.MetaHpoWarning")

ANY_SEED = 0

BUDGET = 40

STEPS = 4

NAN = math.nan


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
    """The sphere function, smallest at x = y = 0."""
    return params["x"] ** 2 + params["y"] ** 2


def _curve(params, step):
    """A loss that falls by one per step towards the sphere value: its order between trials is the same at
    every step."""
    return _sphere(params) + (STEPS - 1 - step)


def _learning(params, trial):
    """An objective that reports the falling loss at steps 0 to STEPS - 1, asks the pruner after each report
    and returns the sphere value."""
    for step in range(STEPS):
        trial.report(_curve(params, step), step)
        if trial.should_prune():
            raise TrialPruned()
    return _sphere(params)


def _interrupting(after, function=_learning):
    """`function` raising KeyboardInterrupt on call `after` + 1."""
    calls = [0]

    def objective(params, trial):
        calls[0] += 1
        if calls[0] > after:
            raise KeyboardInterrupt
        return function(params, trial)

    return objective


def _refusing(params, trial):
    raise AssertionError("the objective must not be called")


class _Always(Pruner):
    """A pruner that prunes every trial it is asked about."""

    def prune(self, study, trial):
        return True


class _Refusing(Pruner):
    """A pruner that must not be asked."""

    def prune(self, study, trial):
        raise AssertionError("the pruner must not be asked")


class _Logging(MedianPruner):
    """MedianPruner that counts its own answers and appends each trial number to a shared log through a
    builtin list.append, which copy.deepcopy keeps as it is."""

    def __init__(self, log):
        super().__init__()
        self.append = log
        self.asked = 0

    def prune(self, study, trial):
        self.asked += 1
        self.append(trial.number)
        return super().prune(study, trial)


def _keyword(params, trial):
    _seen.append((type(trial), trial.number, trial.params == params))
    return _sphere(params)


def _keyword_only(params, *, trial):
    _seen.append((type(trial), trial.number, trial.params == params))
    return _sphere(params)


def _positional_only(params, trial, /):
    _seen.append((type(trial), trial.number, trial.params == params))
    return _sphere(params)


_seen = []
"""What the objectives above saw, emptied by the test that uses them."""


@pytest.mark.parametrize(
    "objective",
    [
        pytest.param(_keyword, id="keyword"),
        pytest.param(_keyword_only, id="keyword-only"),
        pytest.param(_positional_only, id="positional-only"),
    ],
)
def test_optimize_should_pass_a_running_trial_to_an_objective_with_a_trial_parameter(objective):
    """Given an objective with a parameter named trial (keyword, keyword-only or positional-only second), when
    optimize() runs six trials, then every call receives a RunningTrial whose number is the trial number and
    whose params equal the params passed.

    Oracle: optimize docstring (when `function` has a parameter named `trial`, it is called with a
    metahpo.RunningTrial) and RunningTrial (`number` the trial's number, `params` a copy of its parameters).
    """
    _seen.clear()

    optimize(objective, _space(), max_evals=6, seed=ANY_SEED)

    assert _seen == [(RunningTrial, number, True) for number in range(6)]


def _plain(params):
    return _sphere(params)


def _keywords(params, **options):
    return _sphere(params) + len(options)


@pytest.mark.parametrize(
    "objective",
    [pytest.param(_plain, id="one-argument"), pytest.param(_keywords, id="keywords-without-trial")],
)
def test_optimize_should_call_an_objective_without_a_trial_parameter_with_the_params_alone(objective):
    """Given an objective without a parameter named trial, also one that takes **options, when optimize()
    runs, then it is called with the params alone, every trial completes and none has intermediate values.

    Oracle: optimize docstring (an objective without a `trial` parameter is called as before). Hand check:
    _keywords would add 1 to the sphere value if it received a keyword.
    """
    study = optimize(objective, _space(), max_evals=10, seed=ANY_SEED)

    assert [t.state for t in study.trials] == ["complete"] * 10
    assert [t.intermediate_values for t in study.trials] == [{}] * 10
    assert [t.score for t in study.trials] == [_sphere(t.params) for t in study.trials]


@pytest.mark.parametrize(
    ("make", "initial"),
    [
        pytest.param(lambda space: ABCOptimizer(space, colony_size=4, seed=ANY_SEED), 2, id="abc"),
        pytest.param(lambda space: PSOOptimizer(space, swarm_size=6, seed=ANY_SEED), 6, id="pso"),
        pytest.param(lambda space: BBPSOOptimizer(space, swarm_size=6, seed=ANY_SEED), 6, id="bbpso"),
    ],
)
def test_optimize_should_never_prune_the_initial_population_and_prune_every_later_trial_at_its_first_report(
    make, initial
):
    """Given a pruner that always prunes and an objective that asks after every report, when optimize() runs
    20 calls with each optimizer, then the trials of the initial population complete, every later trial is
    pruned at step 0 with the value it reported there as its score, and every trial counts as a call.

    Oracle: optimize docstring (a trial of the optimizer's initial population is never pruned; a pruned trial
    counts as a call toward max_evals and its score is its last reported value) and TrialPruned.
    """
    space = _space()

    study = optimize(_learning, space, max_evals=20, optimizer=make(space), pruner=_Always())

    called = [t for t in study.trials if not t.cached]
    pruned = called[initial:]
    assert [t.state for t in called] == ["complete"] * initial + ["pruned"] * (20 - initial)
    assert [t.intermediate_values for t in pruned] == [{0: _curve(t.params, 0)} for t in pruned]
    assert [t.score for t in pruned] == [_curve(t.params, 0) for t in pruned]
    assert study.n_calls == 20


class _NewBests(Callback):
    """Records the trials that on_trial reports as a new best."""

    def __init__(self):
        self.numbers = []

    def on_trial(self, state, trial, new_best):
        self.numbers += [trial.number] * new_best


def _reporting_minus_100(params, trial):
    """Reports -100 at step 0, below every sphere value, and asks the pruner."""
    trial.report(-100.0, 0)
    if trial.should_prune():
        raise TrialPruned()
    return _sphere(params)


def test_optimize_should_never_make_a_pruned_trial_the_best_even_with_a_better_value():
    """Given an objective whose pruned trials report -100, below every completed score, when optimize() runs
    with a pruner that always prunes and ABC with two food sources, then best_trial is a completed trial of
    the initial population and on_trial reports no pruned trial as a new best.

    Oracle: optimize docstring (a pruned trial never becomes the best trial) and Study.best_trial (completed
    trials only).
    """
    space = _space()
    spy = _NewBests()

    study = optimize(
        _reporting_minus_100,
        space,
        max_evals=12,
        optimizer=ABCOptimizer(space, colony_size=4, seed=ANY_SEED),
        pruner=_Always(),
        callbacks=[spy],
    )

    assert study.best_trial.state == "complete"
    assert study.best_trial.number in (0, 1)
    assert set(spy.numbers) <= {0, 1}


def _pruning_positive_x(params):
    if params["x"] > 0.0:
        raise TrialPruned()
    return _sphere(params)


def test_optimize_should_record_a_trial_pruned_without_a_report_with_a_nan_score():
    """Given a one-argument objective that raises TrialPruned for every x > 0 without reporting, when
    optimize() runs, then those trials are pruned with a NaN score and no intermediate values, and the others
    complete.

    Oracle: TrialPruned docstring (the last reported value as its score, NaN when it reported none; an
    objective without a `trial` parameter may raise it as well).
    """
    study = optimize(_pruning_positive_x, _space(), max_evals=30, seed=ANY_SEED)

    positive = [t for t in study.trials if t.params["x"] > 0.0]
    assert positive
    assert {(t.state, repr(t.score), len(t.intermediate_values)) for t in positive} == {("pruned", "nan", 0)}
    assert {t.state for t in study.trials if t.params["x"] <= 0.0} == {"complete"}


def _pruning_two(calls):
    """An objective that logs i in `calls`, reports 0.5 and raises TrialPruned for i == 2."""

    def objective(params, trial):
        calls.append(params["i"])
        trial.report(0.5, 0)
        if params["i"] == 2:
            raise TrialPruned()
        return float(params["i"])

    return objective


def test_optimize_should_prune_a_cached_repeat_of_a_pruned_trial_again_without_a_call():
    """Given a four-value Int space and an objective that reports 0.5 and raises TrialPruned for i == 2, when
    optimize() runs 30 trials with the cache on, then every trial with i == 2 is pruned with the values {0:
    0.5}, the repeats among them are cached, and the objective was called once per uncached trial.

    Oracle: optimize docstring (a cached repeat of a pruned trial is pruned again without a call) and Trial (a
    cached repeat carries the outcome of the trial it repeats).
    """
    calls = []

    study = optimize(_pruning_two(calls), Space({"i": Int(0, 3)}), max_evals=30, seed=ANY_SEED)

    twos = [t for t in study.trials if t.params["i"] == 2]
    assert {(t.state, repr(t.score)) for t in twos} == {("pruned", "0.5")}
    assert [t.intermediate_values for t in twos] == [{0: 0.5}] * len(twos)
    assert [t.cached for t in twos][1:] == [True] * (len(twos) - 1)
    assert len(calls) == study.n_calls


def _breaking_third(calls):
    """An objective with a trial parameter that counts its calls and raises ValueError on the third."""

    def objective(params, trial):
        calls[0] += 1
        trial.report(1.0, 0)
        if calls[0] == 3:
            raise ValueError("broken")
        return _sphere(params)

    return objective


def test_optimize_should_reraise_an_error_of_the_objective_that_takes_a_trial():
    """Given an objective with a trial parameter that raises ValueError on its third call, when optimize()
    runs, then the ValueError propagates after three calls, as for any objective.

    Oracle: optimize docstring (TrialPruned ends a trial as pruned; an exception raised by `function` is
    re-raised after the results so far are written).
    """
    calls = [0]

    with pytest.raises(ValueError, match="broken"):
        optimize(_breaking_third(calls), _space(), max_evals=10, seed=ANY_SEED)
    assert calls[0] == 3


def _better_then_pruned(calls):
    """An objective that completes its first two calls with falling scores and then reports -100 and raises
    TrialPruned on every later call."""

    def objective(params, trial):
        calls[0] += 1
        if calls[0] > 2:
            trial.report(-100.0, 0)
            raise TrialPruned()
        return 10.0 - calls[0]

    return objective


def test_patience_should_count_pruned_trials_as_calls_without_improvement():
    """Given two completed calls with falling scores followed by pruned calls whose last value -100 beats
    them, when optimize() runs with Patience(3), then the run stops after the fifth trial with reason
    patience.

    Oracle: Patience docstring (a pruned trial is a call without improvement). Hand count: the last
    improvement is call 2, calls 3, 4 and 5 bring none.
    """
    calls = [0]

    study = optimize(_better_then_pruned(calls), _space(), max_evals=BUDGET, seed=ANY_SEED, stop=Patience(3))

    assert (len(study.trials), study.stop_reason) == (5, "patience")


def test_target_score_should_not_be_reached_by_a_pruned_trial():
    """Given the same objective and TargetScore(-50.0), when optimize() runs 12 calls, then the run ends with
    reason budget, as no completed trial reaches the target although the pruned values -100 would.

    Oracle: TargetScore docstring (the best score so far) and optimize docstring (a pruned trial never becomes
    the best trial).
    """
    calls = [0]

    study = optimize(
        _better_then_pruned(calls), _space(), max_evals=12, seed=ANY_SEED, stop=TargetScore(-50.0)
    )

    assert (len(study.trials), study.stop_reason) == (12, "budget")


class _Plain(Optimizer):
    """A scripted optimizer of one fresh candidate per ask whose tell() has no `pruned` parameter."""

    def __init__(self, space, seed):
        super().__init__(space, seed=seed)
        self.told = []

    def ask(self):
        return self._fresh(1)

    def tell(self, candidates, scores):
        self.told.append(list(scores))

    def cancel(self, candidates):
        pass


class _Steps(_Plain):
    """The same optimizer with a tell() that takes `pruned`."""

    def tell(self, candidates, scores, pruned=None):
        self.told.append((list(scores), pruned))


def _odd_pruned(params, trial):
    """Completes even trial numbers with 1.0 and prunes odd ones after reporting 0.25 and 0.5."""
    trial.report(0.25, 0)
    trial.report(0.5, 1)
    if trial.number % 2:
        raise TrialPruned()
    return 1.0


def test_optimize_should_pass_pruned_steps_only_to_a_tell_that_takes_them_and_nan_otherwise():
    """Given objectives that prune every odd trial after reporting steps 0 and 1, when optimize() runs four
    trials with a scripted optimizer without `pruned` and one with it, then the first receives NaN for the
    pruned scores and the second the last value 0.5 with the last step 1, and None for a completed trial.

    Oracle: Optimizer docstring (optimize() passes `pruned` only to an optimizer whose tell() has a parameter
    of that name; any other optimizer receives NaN as the score of a pruned candidate) and the meaning of the
    entry (the last step reported).
    """
    space = _space()
    plain, steps = _Plain(space, ANY_SEED), _Steps(space, ANY_SEED)

    optimize(_odd_pruned, space, max_evals=4, optimizer=plain)
    optimize(_odd_pruned, space, max_evals=4, optimizer=steps)

    assert [repr(s) for (s,) in plain.told] == ["1.0", "nan", "1.0", "nan"]
    assert steps.told == [([1.0], None), ([0.5], [1]), ([1.0], None), ([0.5], [1])]


@pytest.mark.parametrize(
    "pruner",
    [
        pytest.param("median", id="string"),
        pytest.param(MedianPruner, id="class"),
        pytest.param(Patience(3), id="stopper"),
    ],
)
def test_optimize_should_reject_a_pruner_that_is_not_a_pruner_before_any_call(pruner):
    """Given a pruner argument that is not a Pruner object (a string, the class itself, a stopping rule), when
    optimize() is called, then it raises TypeError naming pruner without calling the objective.

    Oracle: optimize docstring (anything else raises TypeError).
    """
    with pytest.raises(TypeError, match="pruner"):
        optimize(_refusing, _space(), max_evals=5, seed=ANY_SEED, pruner=pruner)


def test_optimize_should_give_every_run_its_own_copy_of_the_pruner_and_leave_the_given_one_unchanged():
    """Given a logging MedianPruner, when optimize() runs two runs whose trials ask it, then the shared log
    holds the trials of both runs while the object passed in was never asked.

    Oracle: optimize docstring (every run gets its own copy (copy.deepcopy) of the pruner).
    """
    log = []
    given = _Logging(log.append)

    optimize(_learning, _space(), max_evals=15, seed=[ANY_SEED, ANY_SEED + 1], runs=2, pruner=given)

    assert given.asked == 0
    assert log.count(14) == 2


def _plain_refusing(params):
    raise AssertionError("the objective must not be called")


def test_optimize_should_warn_before_the_search_when_a_given_pruner_cannot_act(tmp_path):
    """Given a one-argument objective and pruner=MedianPruner(), when optimize() runs under an error filter
    for MetaHpoWarning, then it raises the warning naming the missing trial parameter before any call and
    before a results folder is written.

    Oracle: optimize docstring (a condition the arguments show is warned about before the search starts:
    nothing is written and the objective is not called; optimize() warns when a pruner other than NopPruner is
    given while `function` has no parameter named `trial`).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", MetaHpoWarning)
        with pytest.raises(MetaHpoWarning, match="no parameter named trial"):
            optimize(
                _plain_refusing, _space(), max_evals=5, seed=ANY_SEED, output=tmp_path, pruner=MedianPruner()
            )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("objective", "pruner"),
    [
        pytest.param(_plain, None, id="default-pruner-one-argument"),
        pytest.param(_plain, NopPruner(), id="nop-pruner-one-argument"),
        pytest.param(_learning, MedianPruner(), id="reporting-objective"),
    ],
)
def test_optimize_should_not_warn_about_a_pruner_that_was_not_given_is_nop_or_could_act(objective, pruner):
    """Given the default pruner or NopPruner with a one-argument objective, or MedianPruner with an objective
    that reports, when optimize() runs under an error filter for MetaHpoWarning, then it finishes without a
    warning.

    Oracle: optimize docstring (the warnings name a pruner other than NopPruner that is given and cannot act
    or had no effect).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", MetaHpoWarning)
        study = optimize(objective, _space(), max_evals=12, seed=ANY_SEED, pruner=pruner)

    assert len(study.trials) == 12


def _silent(params, trial):
    return _sphere(params)


def _twice(params, trial):
    trial.report(1.0, 0)
    trial.report(2.0, 0)
    trial.report(3.0, 1)
    trial.report(4.0, 1)
    return _sphere(params)


@pytest.mark.parametrize(
    ("objective", "pruner", "expected"),
    [
        pytest.param(
            _silent,
            PercentilePruner(25.0),
            "pruner=PercentilePruner() had no effect because no trial reported an intermediate value",
            id="nothing-reported",
        ),
        pytest.param(
            _twice,
            None,
            "8 reports repeated a step that was already reported, the first: trial 0 reported step 0 again",
            id="repeated-steps",
        ),
    ],
)
def test_optimize_should_warn_once_after_the_search_about_a_pruner_without_effect_or_repeated_steps(
    objective, pruner, expected
):
    """Given an objective with a trial parameter that reports nothing (with a given PercentilePruner), or one
    that reports steps 0 and 1 twice in each of four trials, when optimize() runs four trials, then exactly
    one MetaHpoWarning comes, with the documented text.

    Oracle: optimize docstring (warned about when the call has finished: a given pruner other than NopPruner
    that had no effect because no trial reported an intermediate value; steps that a trial reported more than
    once, of which only the first value is kept). Hand count: two repeats in each of four trials are eight.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimize(objective, _space(), max_evals=4, seed=ANY_SEED, pruner=pruner)

    messages = [str(w.message) for w in caught if w.category is MetaHpoWarning]
    assert len(messages) == 1
    assert messages[0].startswith(expected)


def _nan_then_pruned(params, trial):
    """Returns NaN unless the pruner prunes the trial after its report at step 0."""
    trial.report(1.0, 0)
    if trial.should_prune():
        raise TrialPruned()
    return NAN


def test_optimize_should_name_pruning_when_a_run_ended_without_a_finite_score_after_pruned_trials():
    """Given an objective that returns NaN for the initial population of ABC and is pruned afterwards, when
    optimize() runs eight trials, then the warning about the run without a finite score names pruning as a
    cause.

    Oracle: optimize docstring (a run that ended without a finite score: the objective returned only NaN or
    infinite scores or its trials were pruned) and _no_finite_score.
    """
    space = _space()
    abc = ABCOptimizer(space, colony_size=4, seed=ANY_SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study = optimize(_nan_then_pruned, space, max_evals=8, optimizer=abc, pruner=_Always())

    messages = [str(w.message) for w in caught if w.category is MetaHpoWarning]
    assert [t.state for t in study.trials] == ["failed"] * 2 + ["pruned"] * 6
    assert messages == [
        "the run ended without a finite score: every trial was pruned or returned a NaN or infinite score, "
        "so its Study has no best trial (best_trial raises ValueError) and its best.json records none; check "
        "that the objective returns a finite number, for example that a training that fails or diverges does "
        "not come back as NaN or inf"
    ]


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _folder(root):
    """The only results folder below root."""
    folders = [p for p in root.iterdir() if p.is_dir()]
    assert len(folders) == 1
    return folders[0]


def _lines(folder, run=1):
    """The records of trials.jsonl of one run."""
    text = (folder / f"run_{run:03d}" / "trials.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_results_folder_should_record_the_pruner_and_the_state_and_values_of_each_trial(tmp_path, no_plots):
    """Given a run with PercentilePruner(40.0, n_startup_trials=2) and the reporting objective, when its
    results folder is read, then config.json records the pruner with its class, module and settings, every
    line of trials.jsonl holds the trial's intermediate values with the steps as text, and exactly the pruned
    trials carry state 'pruned'.

    Oracle: optimize docstring (config.json records the pruner) and report._pruning_record ('state' for a
    pruned trial and 'intermediate_values' for a trial that reported any).
    """
    pruner = PercentilePruner(40.0, n_startup_trials=2)

    study = optimize(_learning, _space(), max_evals=BUDGET, seed=ANY_SEED, output=tmp_path, pruner=pruner)

    folder = _folder(tmp_path)
    lines = _lines(folder)
    assert _json(folder / "config.json")["pruner"] == {
        "name": "PercentilePruner",
        "module": "metahpo.prune",
        "settings": pruner.settings(),
    }
    assert [line.get("state", "complete") for line in lines] == [t.state for t in study.trials]
    assert [line["intermediate_values"] for line in lines] == [
        {str(step): value for step, value in t.intermediate_values.items()} for t in study.trials
    ]
    assert "pruned" in [t.state for t in study.trials]


def test_results_folder_of_a_search_without_reports_should_keep_the_trial_keys_of_0_8_0(tmp_path, no_plots):
    """Given a one-argument objective, when optimize() writes its results folder, then no line of trials.jsonl
    holds 'state' or 'intermediate_values', and config.json records the default MedianPruner.

    Oracle: report._pruning_record (empty for a trial without either, so the line of a search without pruning
    keeps the keys of 0.8.0) and optimize docstring (MedianPruner() for None).
    """
    optimize(_plain, _space(), max_evals=10, seed=ANY_SEED, output=tmp_path)

    folder = _folder(tmp_path)
    keys = {key for line in _lines(folder) for key in line}
    assert keys.isdisjoint({"state", "intermediate_values"})
    assert _json(folder / "config.json")["pruner"]["name"] == "MedianPruner"


SHARED = _space()
"""The space the optimizer functions below build on, so optimize() and resume() share it."""


def _abc(seed):
    return ABCOptimizer(SHARED, colony_size=6, seed=seed)


def _pso(seed):
    return PSOOptimizer(SHARED, swarm_size=6, seed=seed)


def _bbpso(seed):
    return BBPSOOptimizer(SHARED, swarm_size=6, seed=seed)


def _small_abc(seed):
    return ABCOptimizer(SHARED, colony_size=4, seed=seed)


@pytest.mark.parametrize(
    "make", [pytest.param(_abc, id="abc"), pytest.param(_pso, id="pso"), pytest.param(_bbpso, id="bbpso")]
)
@pytest.mark.parametrize("after", [7, 23])
def test_resume_should_continue_an_interrupted_pruning_run_like_an_uninterrupted_one(
    tmp_path, no_plots, make, after
):
    """Given a pruning run interrupted after `after` calls, when resume() continues it with the recorded
    MedianPruner, then the study equals the uninterrupted run with the same seed, pruned trials and their
    intermediate values included.

    Oracle: resume docstring (the restored trials keep the state and the intermediate values trials.jsonl
    records; the recorded pruner goes on, so the search goes on as if it had never stopped).
    """
    reference = optimize(_learning, SHARED, max_evals=BUDGET, optimizer=make, seed=ANY_SEED)
    optimize(_interrupting(after), SHARED, max_evals=BUDGET, optimizer=make, seed=ANY_SEED, output=tmp_path)

    resumed = resume(_learning, _folder(tmp_path), optimizer=make)

    assert resumed == reference
    assert "pruned" in [t.state for t in reference.trials]


def test_resume_should_restore_a_finished_pruning_run_without_calling_the_objective_or_asking_the_pruner(
    tmp_path, no_plots
):
    """Given a finished pruning run, when resume() is called with an objective and a pruner that must not be
    used, then it returns the same study, pruned trials included.

    Oracle: resume docstring (the pruner is not asked again for the trials a resume restores; a folder whose
    runs all finished evaluates nothing).
    """
    finished = optimize(_learning, _space(), max_evals=BUDGET, seed=ANY_SEED, output=tmp_path)

    restored = resume(_refusing, _folder(tmp_path), pruner=_Refusing())

    assert restored == finished
    assert "pruned" in [t.state for t in finished.trials]


def test_resume_should_go_on_with_the_pruner_given_as_pruner_and_record_it(tmp_path, no_plots):
    """Given a run with a pruner that always prunes, interrupted after 9 calls, when resume() continues it
    with pruner=NopPruner(), then the restored trials keep their state, no later trial is pruned, and
    config.json records NopPruner.

    Oracle: resume docstring (`pruner=` replaces it for the trials still to come, and config.json then records
    it).
    """
    optimize(
        _interrupting(9),
        SHARED,
        max_evals=20,
        optimizer=_small_abc,
        seed=ANY_SEED,
        output=tmp_path,
        pruner=_Always(),
    )
    folder = _folder(tmp_path)

    resumed = resume(_learning, folder, optimizer=_small_abc, pruner=NopPruner())

    assert [t.state for t in resumed.trials] == ["complete"] * 2 + ["pruned"] * 7 + ["complete"] * 11
    assert _json(folder / "config.json")["pruner"]["name"] == "NopPruner"


def test_resume_should_ask_for_pruner_when_the_folder_records_a_pruner_of_ones_own(tmp_path, no_plots):
    """Given a folder written with a pruner of one's own, when resume() is called without pruner=, then it
    raises ValueError asking for pruner= and leaves trials.jsonl as it was; with pruner= it continues to the
    budget.

    Oracle: resume docstring (a folder written with a pruner of another class needs `pruner=`, else
    ValueError; nothing in the folder changes before the checks have passed).
    """
    optimize(_interrupting(5), SHARED, max_evals=12, seed=ANY_SEED, output=tmp_path, pruner=_Always())
    folder = _folder(tmp_path)
    before = (folder / "run_001" / "trials.jsonl").read_bytes()

    with pytest.raises(ValueError, match="pass pruner="):
        resume(_learning, folder, space=SHARED)
    assert (folder / "run_001" / "trials.jsonl").read_bytes() == before
    assert len(resume(_learning, folder, space=SHARED, pruner=_Always()).trials) == 12


def _rewrite_config(folder, edit):
    """Applies `edit` to the parsed config.json of `folder` and writes it back."""
    config = _json(folder / "config.json")
    edit(config)
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_resume_should_use_median_pruner_for_a_folder_that_records_no_pruner(tmp_path, no_plots):
    """Given an interrupted run with the default pruner whose config.json loses its pruner entry, as a folder
    of 0.8.0 has none, when resume() continues it, then it gives the uninterrupted run with MedianPruner().

    Oracle: resume docstring (a folder written before pruning existed uses MedianPruner()).
    """
    reference = optimize(_learning, SHARED, max_evals=BUDGET, seed=ANY_SEED)
    optimize(_interrupting(11), SHARED, max_evals=BUDGET, seed=ANY_SEED, output=tmp_path)
    folder = _folder(tmp_path)
    _rewrite_config(folder, lambda config: config.pop("pruner"))

    resumed = resume(_learning, folder, space=SHARED)

    assert resumed == reference


def test_resume_should_reject_pruner_settings_that_the_class_rejects(tmp_path, no_plots):
    """Given an interrupted run whose config.json records MedianPruner with interval_steps 0, when resume() is
    called, then it raises ValueError naming config.json and the pruner.

    Oracle: resume docstring (the pruners of the library are built again from their settings) and
    _resume_pruner (ValueError for settings the class rejects).
    """
    optimize(_interrupting(11), SHARED, max_evals=BUDGET, seed=ANY_SEED, output=tmp_path)
    folder = _folder(tmp_path)
    _rewrite_config(folder, lambda config: config["pruner"]["settings"].update(interval_steps=0))

    with pytest.raises(ValueError, match="config.json records settings of the pruner MedianPruner"):
        resume(_learning, folder, space=SHARED)


def _star_trial(params, *trial):
    return _sphere(params)


def _third_positional_trial(params, extra, trial, /):
    return _sphere(params)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        pytest.param(_keyword, "keyword", id="keyword"),
        pytest.param(_keyword_only, "keyword", id="keyword-only"),
        pytest.param(_positional_only, "positional", id="positional-only-second"),
        pytest.param(_third_positional_trial, "", id="positional-only-third"),
        pytest.param(_star_trial, "", id="star-args-named-trial"),
        pytest.param(_plain, "", id="no-trial"),
        pytest.param(dict, "", id="no-signature"),
    ],
)
def test_trial_style_should_hand_the_trial_only_to_a_parameter_that_can_take_it(function, expected):
    """Given objectives whose parameter named trial is a keyword, keyword-only, the second or the third
    positional-only parameter or *trial, one without such a parameter and a builtin whose signature cannot be
    read, when the facade decides how to pass the RunningTrial, then only the first three receive it.

    Oracle: optimize docstring (called as function(params, trial=...) when `function` has a parameter named
    `trial`) and _trial_style (a positional-only `trial` only as the second parameter; no trial when the
    signature cannot be read). Passing the handle to the third positional-only parameter or into *trial would
    hand it to a parameter that does not expect it.
    """
    style = metahpo.facade._trial_style(function)

    assert style == expected


@pytest.mark.parametrize(
    ("edit", "found"),
    [
        pytest.param({"state": "stopped"}, "stopped", id="unknown-state"),
        pytest.param({"intermediate_values": {"-1": 0.5}}, "-1", id="negative-step"),
        pytest.param({"intermediate_values": {"01": 0.5}}, "01", id="padded-step"),
        pytest.param({"intermediate_values": {"x": 0.5}}, "x", id="text-step"),
    ],
)
def test_resume_should_refuse_a_trials_line_with_an_unknown_state_or_step(tmp_path, no_plots, edit, found):
    """Given an interrupted pruning run whose first trials.jsonl line gets an unknown state or a step that is
    not written as an integer >= 0, when resume() reads the folder, then it raises RuntimeError naming the
    line as not a trial record, and the folder stays as it was.

    Oracle: resume docstring (a damaged record raises before anything in the folder changes) and
    report._pruned_state and report._step ('complete' or 'pruned'; a step written as an integer >= 0).
    The cause names the value that was refused.
    """
    optimize(_interrupting(9), SHARED, max_evals=20, seed=ANY_SEED, output=tmp_path)
    folder = _folder(tmp_path)
    path = folder / "run_001" / "trials.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    first = {**json.loads(lines[0]), **edit}
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(RuntimeError, match="line 1 of .* is not a trial record") as caught:
        resume(_learning, folder, space=SHARED)
    assert found in str(caught.value.__cause__)
    assert path.read_bytes() == before
