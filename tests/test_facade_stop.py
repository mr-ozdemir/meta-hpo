"""Integration tests of optimize(stop=...): stopping rules ending a run through the facade.

Oracles: the optimize docstring (stop is None, one Stopper or a sequence of them; after every trial, cached
repeats included, every rule is asked once, in order; the evaluated part of the batch is told, the rest is
withdrawn, and the reason of the first rule that said stop becomes the stop_reason, also when the same trial
used up the budget; every run gets its own copy of the rules; TypeError for anything but a Stopper and
ValueError for an empty or reserved reason), the Stopper docstring (the moment of the call, every rule
asked for every trial, no further call of the objective), the docstrings of Patience, MaxTime and
TargetScore, Study.stop_reason, RunState.elapsed and the report module docstring (summary.json, best.json,
the console end line).

Expected trials come from hand-written score lists whose stop trial is counted by hand, or from an
uninterrupted run with the same seed: its score list gives the trial where the rule stops, and the stopped
run must equal its prefix. The default results folder and console are switched off through METAHPO_OUTPUT
and METAHPO_VERBOSE in conftest; the results folder test writes into tmp_path with plots off. MaxTime reads
the search time from RunState.elapsed(), whose clock RunState.now() is replaced with one that moves a minute
per trial, so no test sleeps.

The rules a run uses are deep copies, so the recording rules here keep a builtin list.append or a module
function, which copy.deepcopy keeps as they are, to report back to the test.

Unspecified, therefore not asserted: what a rule that raises does to the run (it ends like an error in the
objective), and the number of lines the console prints before the end line.

Input space: Every objective, space, optimizer, budget, runs value and list of rules, which is unbounded.
Subset: The three ways a run can stop a batch (inside the initial batch of PSO, inside a later batch, after
a single ABC candidate); scripted scores in both directions with ties, NaN and infinities; a scripted
optimizer that hands out cached repeats; one, two and three rules in both orders; runs 1 and 3; budgets that
end on the stopping trial; every reserved reason, an empty, a blank and a non-string reason; non-Stopper
values of every container kind; MaxTime limits given in each unit.
"""

import csv
import json
import math

import numpy as np
import pytest

import metahpo
from metahpo import (
    Callback,
    Float,
    MaxTime,
    Optimizer,
    Patience,
    PSOOptimizer,
    ResultWriter,
    Space,
    Stopper,
    TargetScore,
    optimize,
)

ANY_SEED = 0

PSO_SEED = 3

BUDGET = 100

NAN = math.nan

INF = math.inf


def _space():
    """A fresh two-parameter Float space."""
    return Space({"x": Float(-5.0, 5.0), "y": Float(-5.0, 5.0)})


def _sphere(params):
    """The sphere function, smallest at x = y = 0."""
    return params["x"] ** 2 + params["y"] ** 2


def _pso():
    """A seeded PSOOptimizer whose batches hold five candidates."""
    return PSOOptimizer(_space(), swarm_size=5, seed=PSO_SEED)


def _pso_run(function=_sphere, optimizer=None, stop=None):
    """optimize() minimising `function` with `optimizer` (default a fresh _pso()) on its own space."""
    opt = _pso() if optimizer is None else optimizer
    return optimize(function, opt.space, max_evals=BUDGET, optimizer=opt, seed=PSO_SEED, stop=stop)


def _params(study):
    """The params of every trial of a study, in order."""
    return [t.params for t in study.trials]


def _rows(path):
    """The rows of a CSV file as dictionaries."""
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _scripted(scores, calls):
    """An objective that returns scores[n] on its call n + 1 whatever the params, and counts its calls in
    calls[0]; a call after the last score raises IndexError."""

    def objective(params):
        calls[0] += 1
        return scores[calls[0] - 1]

    return objective


def _counting(calls, function=_sphere):
    """An objective that counts its calls in calls[0] and returns function(params)."""

    def objective(params):
        calls[0] += 1
        return function(params)

    return objective


def _patience_stop(trials, evals):
    """The index of the trial at which Patience(evals) stops a minimising run, counted from the trials of an
    uninterrupted run: calls without a new best since the last new best, cached repeats left out."""
    best, last = INF, 0
    calls = 0
    for index, trial in enumerate(trials):
        calls += not trial.cached
        if math.isfinite(trial.score) and trial.score < best:
            best, last = trial.score, calls
        if calls - last >= evals:
            return index
    return None


class _After(Stopper):
    """A rule of one's own that says stop once it has seen `count` trials; `seen` counts them."""

    def __init__(self, count, reason):
        self.count = count
        self.reason = reason
        self.seen = 0

    def should_stop(self, state):
        self.seen += 1
        return self.seen >= self.count


class _Recording(Stopper):
    """A rule that hands the state to `record` on every call and says stop from trial `stop_at` on."""

    reason = "recorded"

    def __init__(self, record, stop_at=None):
        self.record = record
        self.stop_at = stop_at

    def should_stop(self, state):
        self.record(state)
        return len(state.study.trials) == self.stop_at


class _NamedRecording(Stopper):
    """A rule that appends (name, trials so far) through `record` and never says stop."""

    reason = "never"

    def __init__(self, name, record):
        self.name = name
        self.record = record

    def should_stop(self, state):
        self.record((self.name, len(state.study.trials)))
        return False


class _Events(Callback):
    """Counts the on_trial events it receives."""

    def __init__(self):
        self.trials = 0

    def on_trial(self, state, trial, new_best):
        self.trials += 1


def _snapshot(log, events):
    """A record function for _Recording that appends what the rule sees: the trials so far, the calls, the
    number of the best trial, the cached flag of the last trial and the on_trial events counted so far."""

    def record(state):
        best = state.best
        log.append(
            (
                len(state.study.trials),
                state.calls,
                best and best.number,
                state.study.trials[-1].cached,
                events.trials,
            )
        )

    return record


class _Scripted(Optimizer):
    """An optimizer that hands out one candidate per ask at the next unit-cube point of `points`; equal
    points give equal parameter sets, which the cache answers as repeats."""

    def __init__(self, space, points, seed=None):
        super().__init__(space, seed=seed)
        self.points = list(points)

    def ask(self):
        return self._candidates(np.array([[self.points.pop(0)]]))

    def tell(self, candidates, scores):
        pass

    def cancel(self, candidates):
        pass


class _Reason(Stopper):
    """A rule whose reason is given; it never says stop."""

    def __init__(self, reason):
        self.reason = reason

    def should_stop(self, state):
        return False


def _minute_per_trial(state):
    """A clock for RunState.now that reads the run's start plus one minute per recorded trial."""
    return state.started + 60.0 * len(state.study.trials)


@pytest.mark.parametrize(
    ("direction", "sign"),
    [pytest.param("min", 1.0, id="min"), pytest.param("max", -1.0, id="max")],
)
def test_optimize_should_end_the_run_when_patience_sees_evals_calls_without_improvement(direction, sign):
    """Given stop=Patience(3) and scores 5, 4, 4.5, 3, 4, 3, 4, 0, ... (negated when maximising), when
    optimize() runs with a budget of 20, then the run ends after trial 7 with stop_reason 'patience' and the
    objective was called seven times.

    Oracle: Patience docstring (the first finite score improves; ties do not; stop after `evals` calls in a
    row without improvement) and the Stopper docstring (no further call of the objective). Hand count:
    improvements at calls 1, 2 and 4; calls 5, 6 and 7 bring none, so the 0 of call 8 is never reached.
    """
    calls = [0]
    scores = [sign * s for s in [5.0, 4.0, 4.5, 3.0, 4.0, 3.0, 4.0] + [0.0] * 13]

    study = optimize(
        _scripted(scores, calls), _space(), direction=direction, max_evals=20, seed=ANY_SEED, stop=Patience(3)
    )

    assert (len(study.trials), study.n_calls, calls[0]) == (7, 7, 7)
    assert study.stop_reason == "patience"


def test_optimize_should_stop_at_the_trial_where_an_uninterrupted_run_reaches_the_target():
    """Given the score list of an uninterrupted seeded PSO run and a target equal to its third new best,
    when the same run is repeated with stop=[TargetScore(target)], then it equals the prefix of the
    uninterrupted run up to that trial, ends with stop_reason 'target', and the optimizer was told that
    trial.

    Oracle: TargetScore docstring (at most the target when minimising) and the optimize docstring (the
    evaluated part of the batch is told). The first trial whose score is at most the target is the one that
    set this new best, because every earlier score is larger.
    """
    full = _pso_run()
    scores = full.scores
    bests = [i for i in range(1, len(scores)) if scores[i] < scores[:i].min()]
    stop_index = bests[2]
    opt = _pso()

    study = _pso_run(optimizer=opt, stop=[TargetScore(scores[stop_index])])

    assert _params(study) == _params(full)[: stop_index + 1]
    assert study.stop_reason == "target"
    assert opt.best_score == scores[stop_index]


@pytest.mark.parametrize("evals", [pytest.param(6, id="evals-6"), pytest.param(11, id="evals-11")])
def test_optimize_should_stop_at_the_trial_where_an_uninterrupted_run_runs_out_of_patience(evals):
    """Given the trials of an uninterrupted seeded PSO run on the sphere, when the same run is repeated with
    stop=[Patience(evals)], then it equals the prefix of the uninterrupted run up to the trial where
    `evals` calls in a row brought no new best, and ends with stop_reason 'patience'.

    Oracle: Patience docstring with min_delta 0 (an improvement is a new best; cached repeats are no calls),
    counted on the uninterrupted run by _patience_stop; the run also holds a cached repeat in its first
    batches, so a count by trials would stop one trial early.
    """
    full = _pso_run()
    stop_index = _patience_stop(full.trials, evals)

    study = _pso_run(stop=[Patience(evals)])

    assert _params(study) == _params(full)[: stop_index + 1]
    assert study.stop_reason == "patience"
    assert sum(t.cached for t in full.trials[: stop_index + 1]) >= 1


def test_optimize_should_tell_the_evaluated_part_of_the_batch_and_withdraw_the_rest():
    """Given a rule that stops at trial 3 of the initial PSO batch of five, when optimize() returns, then the
    optimizer hands out the same next batch as a twin that was told the three evaluated candidates and had
    the other two cancelled, and the objective was called three times.

    Oracle: optimize docstring (the evaluated part of the current batch is told to the optimizer, the rest is
    withdrawn) and the Stopper docstring (no further call of the objective); the twin follows those steps by
    hand through the public ask/tell/cancel interface.
    """
    calls = [0]
    opt, twin = _pso(), _pso()
    batch = twin.ask()
    twin.tell(batch[:3], [_sphere(c.params) for c in batch[:3]])
    twin.cancel(batch[3:])

    study = _pso_run(_counting(calls), opt, stop=[_After(3, "three")])

    assert [c.params for c in opt.ask()] == [c.params for c in twin.ask()]
    assert (len(study.trials), calls[0], study.stop_reason) == (3, 3, "three")


def test_optimize_should_not_call_the_objective_after_a_rule_said_stop():
    """Given a rule that says stop after the first trial of the default ABC run, whose first batch holds five
    candidates, when optimize() returns, then the objective was called once and the study holds one trial.

    Oracle: Stopper docstring (the run ends without a further call of the objective; the rest of the batch
    is withdrawn).
    """
    calls = [0]

    study = optimize(_counting(calls), _space(), max_evals=BUDGET, seed=ANY_SEED, stop=[_After(1, "first")])

    assert (calls[0], len(study.trials), study.stop_reason) == (1, 1, "first")


@pytest.mark.parametrize(
    ("counts", "reason"),
    [
        pytest.param([(5, "late"), (3, "second"), (3, "third")], "second", id="second-before-third"),
        pytest.param([(5, "late"), (3, "third"), (3, "second")], "third", id="third-before-second"),
        pytest.param([(2, "early"), (3, "second")], "early", id="earlier-trial-wins"),
    ],
)
def test_optimize_should_take_the_reason_of_the_first_rule_in_order_that_says_stop(counts, reason):
    """Given rules of one's own that say stop after 5, 3 and 3 trials (in both orders of the last two) or
    after 2 and 3 trials, when optimize() runs, then the stop_reason is that of the first rule in the given
    order that said stop at the earliest trial.

    Oracle: optimize and Stopper docstrings (every rule is asked in order; the first rule in the order that
    says stop gives the reason).
    """
    rules = [_After(count, name) for count, name in counts]

    study = optimize(_sphere, _space(), max_evals=BUDGET, seed=ANY_SEED, stop=rules)

    assert study.stop_reason == reason
    assert len(study.trials) == min(count for count, _ in counts)


def test_optimize_should_ask_every_rule_once_after_every_trial_even_after_one_said_stop():
    """Given two recording rules and a third that says stop at trial 3, in the middle, when optimize() runs,
    then each recording rule was asked exactly once per trial, in order, including trial 3.

    Oracle: Stopper docstring (every rule is called exactly once after every trial, in order, also when an
    earlier rule already said stop).
    """
    log = []
    rules = [_NamedRecording("a", log.append), _After(3, "stopped"), _NamedRecording("b", log.append)]

    study = optimize(_sphere, _space(), max_evals=BUDGET, seed=ANY_SEED, stop=rules)

    assert log == [("a", 1), ("b", 1), ("a", 2), ("b", 2), ("a", 3), ("b", 3)]
    assert study.stop_reason == "stopped"


def test_optimize_should_ask_the_rules_after_the_trial_is_recorded_and_the_callbacks_were_told():
    """Given a scripted optimizer whose third candidate repeats the second, a recording rule and a callback
    that counts on_trial, when optimize() runs four trials with scores 3, 1, 1 (cached), 2, then the rule saw
    each trial already in the study, the calls and the best trial already updated, and as many on_trial
    events as trials, also for the cached repeat.

    Oracle: Stopper docstring (called after every trial, cached repeats included, after the trial is
    recorded in state.study.trials, state.best and state.calls are updated and the callbacks got on_trial).
    """
    log, events, calls = [], _Events(), [0]
    space = Space({"x": Float(0.0, 1.0)})
    optimizer = _Scripted(space, [0.1, 0.2, 0.2, 0.3], seed=ANY_SEED)

    optimize(
        _scripted([3.0, 1.0, 2.0], calls),
        space,
        max_evals=3,
        optimizer=optimizer,
        seed=ANY_SEED,
        callbacks=[events],
        stop=[_Recording(_snapshot(log, events))],
    )

    assert log == [(1, 1, 0, False, 1), (2, 2, 1, False, 2), (3, 2, 1, True, 3), (4, 3, 1, False, 4)]


def test_optimize_should_give_every_run_a_fresh_copy_of_the_rules_and_leave_the_given_ones_unchanged():
    """Given runs=3, a rule of one's own that stops after 4 trials and a Patience rule, when optimize()
    returns, then each run ended after its own 4 trials with that rule's reason, and the given rule objects
    hold exactly the state they were built with.

    Oracle: optimize docstring (every run gets its own copy (copy.deepcopy) of the rules, so a rule ends its
    own run only and the objects passed in are never changed).
    """
    after, patience = _After(4, "four"), Patience(50)

    studies = optimize(_sphere, _space(), max_evals=BUDGET, seed=[1, 2, 3], runs=3, stop=[after, patience])

    assert [(len(s.trials), s.stop_reason) for s in studies] == [(4, "four")] * 3
    assert after.seen == 0
    assert vars(patience) == vars(Patience(50))


def test_optimize_should_not_count_cached_repeats_toward_patience():
    """Given a scripted optimizer whose third and fourth candidates repeat the second, a constant score and
    stop=[Patience(3)], when optimize() runs, then the run ends after six trials, of which four were calls.

    Oracle: Patience docstring (calls count as for max_evals: a cached repeat is no call) and the optimize
    docstring (with cache=True a repeated parameter set reuses the earlier score). Hand count: call 1 is the
    first finite score, calls 2, 3 and 4 are trials 1, 4 and 5.
    """
    space = Space({"x": Float(0.0, 1.0)})
    optimizer = _Scripted(space, [0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6], seed=ANY_SEED)

    study = optimize(
        lambda p: 1.0, space, max_evals=BUDGET, optimizer=optimizer, seed=ANY_SEED, stop=[Patience(3)]
    )

    assert [t.cached for t in study.trials] == [False, False, True, True, False, False]
    assert (study.n_calls, study.stop_reason) == (4, "patience")


@pytest.mark.parametrize(
    ("direction", "rule", "scores", "trials", "reason"),
    [
        pytest.param("max", Patience(3), [NAN, INF, -INF] + [1.0] * 7, 3, "patience", id="patience-max"),
        pytest.param("min", Patience(3), [NAN, -INF, INF] + [1.0] * 7, 3, "patience", id="patience-min"),
        pytest.param("max", TargetScore(0.0), [NAN, INF] * 5, 10, "budget", id="target-max-never"),
        pytest.param("min", TargetScore(0.0), [NAN, -INF] * 5, 10, "budget", id="target-min-never"),
    ],
)
def test_optimize_should_treat_nan_and_infinite_scores_as_no_improvement_and_never_as_the_target(
    direction, rule, scores, trials, reason
):
    """Given scores NaN, inf and -inf, when optimize() runs with Patience(3) or TargetScore(0.0) and a budget
    of 10, then Patience stops after the three non-finite calls and TargetScore never stops.

    Oracle: Patience docstring (a NaN or infinite score is a call without improvement), TargetScore
    docstring (judged on the best score so far) and the RunState docstring (best is the best finite trial).
    """
    calls = [0]

    study = optimize(
        _scripted(scores, calls), _space(), direction=direction, max_evals=10, seed=ANY_SEED, stop=[rule]
    )

    assert (len(study.trials), study.stop_reason) == (trials, reason)


@pytest.mark.parametrize(
    ("direction", "scores", "target"),
    [
        pytest.param("min", [5.0, 4.0, 3.0, 2.0, 1.0], 1.0, id="min"),
        pytest.param("max", [1.0, 2.0, 3.0, 4.0, 5.0], 5.0, id="max"),
    ],
)
def test_optimize_should_record_the_rule_reason_when_the_same_trial_used_up_the_budget(
    direction, scores, target
):
    """Given max_evals=5 and a target first reached by the fifth call, when optimize() returns, then the
    stop_reason is 'target', not 'budget'.

    Oracle: optimize and Stopper docstrings (a rule's reason wins over 'budget' when the same trial used up
    the budget).
    """
    calls = [0]

    study = optimize(
        _scripted(scores, calls),
        _space(),
        direction=direction,
        max_evals=5,
        seed=ANY_SEED,
        stop=[TargetScore(target)],
    )

    assert (study.n_calls, study.stop_reason) == (5, "target")


def test_optimize_should_end_the_run_with_the_reason_of_a_stopper_of_your_own():
    """Given a rule of one's own with reason 'close' that stops when the last trial lies within 1 of the
    origin, when optimize() runs, then the run ends at the first such trial of the uninterrupted run with
    stop_reason 'close'.

    Oracle: Stopper docstring (subclass it, set `reason` and override should_stop(); the first rule that
    says stop gives the reason) and Study.stop_reason (the `reason` of a Stopper of your own).
    """
    full = optimize(_sphere, _space(), max_evals=BUDGET, seed=ANY_SEED)
    first = [t.number for t in full.trials if t.score < 1.0][0]

    study = optimize(_sphere, _space(), max_evals=BUDGET, seed=ANY_SEED, stop=_Close())

    assert _params(study) == _params(full)[: first + 1]
    assert study.stop_reason == "close"


class _Close(Stopper):
    """Stops when the last trial lies within distance 1 of the origin of the sphere."""

    reason = "close"

    def should_stop(self, state):
        return state.study.trials[-1].score < 1.0


@pytest.mark.parametrize(
    "stop",
    [
        pytest.param(5, id="number"),
        pytest.param("patience", id="string"),
        pytest.param([Patience(3), 5], id="list-with-a-number"),
        pytest.param((Patience(3), None), id="tuple-with-none"),
        pytest.param({Patience(3)}, id="set"),
        pytest.param(Patience, id="class-not-instance"),
        pytest.param(lambda state: True, id="function"),
    ],
)
def test_optimize_should_reject_a_stop_argument_that_is_not_a_stopper(stop):
    """Given a stop argument that is neither None, a Stopper nor a sequence of Stoppers, when optimize() is
    called, then TypeError is raised before the objective is called.

    Oracle: optimize docstring (anything but a Stopper raises TypeError), following the TypeError of an
    optimizer function that returns no Optimizer.
    """
    calls = [0]

    with pytest.raises(TypeError, match="stop"):
        optimize(_counting(calls), _space(), max_evals=5, seed=ANY_SEED, stop=stop)
    assert calls[0] == 0


@pytest.mark.parametrize(
    "reason",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param(None, id="none"),
        pytest.param(5, id="number"),
        pytest.param("budget", id="budget"),
        pytest.param("exhausted", id="exhausted"),
        pytest.param("repeats", id="repeats"),
        pytest.param("interrupted", id="interrupted"),
        pytest.param("error", id="error"),
    ],
)
def test_optimize_should_reject_a_rule_whose_reason_is_empty_or_reserved(tmp_path, reason):
    """Given a rule whose reason is empty, blank, not a string or a reason the search gives itself, when
    optimize() is called with an output folder, then ValueError is raised before the objective is called and
    before any results folder is written.

    Oracle: Stopper docstring (a non-empty string other than the reasons in RESERVED_REASONS; optimize()
    rejects any other value with ValueError) and Study.stop_reason (budget, exhausted, repeats, interrupted
    and error are the search's own reasons).
    """
    calls = [0]

    with pytest.raises(ValueError, match="reason"):
        optimize(
            _counting(calls),
            _space(),
            max_evals=5,
            seed=ANY_SEED,
            output=tmp_path,
            stop=[Patience(3), _Reason(reason)],
        )
    assert (calls[0], list(tmp_path.iterdir())) == (0, [])


def test_optimize_should_reject_the_stopper_base_class_without_a_reason():
    """Given the Stopper base class itself, whose reason is empty, when optimize() is called with it, then
    ValueError is raised.

    Oracle: Stopper docstring (set `reason`; optimize() rejects an empty one with ValueError).
    """
    rule = Stopper()

    with pytest.raises(ValueError, match="Stopper"):
        optimize(_sphere, _space(), max_evals=5, seed=ANY_SEED, stop=rule)
    assert rule.reason == ""


@pytest.mark.parametrize(
    "stop",
    [pytest.param(None, id="none"), pytest.param([], id="empty-list"), pytest.param((), id="empty-tuple")],
)
def test_optimize_should_run_to_the_budget_without_rules(stop):
    """Given stop None or an empty sequence, when optimize() runs, then it ends with stop_reason 'budget'
    after max_evals calls.

    Oracle: optimize docstring (stop is None, one Stopper or a sequence of them; without stopping rules the
    search stops early only when the space is exhausted).
    """
    study = optimize(_sphere, _space(), max_evals=12, seed=ANY_SEED, stop=stop)

    assert (study.n_calls, study.stop_reason) == (12, "budget")


def test_results_folder_should_record_the_rule_reason_and_end_the_run_as_usual(tmp_path, capsys, monkeypatch):
    """Given a run that Patience(3) ends after seven trials, written to a results folder with the console on,
    when the run ends, then summary.json and best.json record stop_reason 'patience' and status finished,
    trials.csv holds the seven trials, generations.csv ends with the last generation at 7 calls and the
    console ends with the line 'finished (patience) after 7 calls' right after the line of that generation.

    Oracle: optimize docstring (the run ends with the rule's reason as its stop_reason), the Stopper
    docstring (the run ends as usual: last generation, on_run_end, results folder, console line) and the
    report module docstring (summary.json, best.json, trials.csv, generations.csv, console end line).
    """
    monkeypatch.delenv("METAHPO_VERBOSE", raising=False)
    calls = [0]
    writer = ResultWriter(tmp_path, plots=False)
    scores = [5.0, 4.0, 4.5, 3.0, 4.0, 3.0, 4.0] + [0.0] * 13

    optimize(
        _scripted(scores, calls),
        _space(),
        max_evals=20,
        seed=ANY_SEED,
        output=None,
        callbacks=[writer],
        stop=[Patience(3)],
    )

    folder = writer.run_dir.parent
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    best = json.loads((writer.run_dir / "best.json").read_text(encoding="utf-8"))
    trials = _rows(writer.run_dir / "trials.csv")
    generations = _rows(writer.run_dir / "generations.csv")
    lines = capsys.readouterr().out.splitlines()
    assert (summary["status"], summary["stop_reason"]) == ("finished", "patience")
    assert (best["status"], best["stop_reason"]) == ("finished", "patience")
    assert (len(trials), generations[-1]["calls"]) == (7, "7")
    assert lines[-2].startswith("gen ")
    assert "finished (patience) after 7 calls" in lines[-1]


@pytest.mark.parametrize(
    ("rule", "trials"),
    [
        pytest.param(MaxTime(minutes=5), 5, id="minutes-on-a-trial"),
        pytest.param(MaxTime(seconds=30, minutes=4), 5, id="between-two-trials"),
        pytest.param(MaxTime(hours=0.1), 6, id="hours"),
        pytest.param(MaxTime(seconds=1), 1, id="first-trial"),
    ],
)
def test_max_time_should_end_the_run_after_the_first_trial_that_reaches_the_limit(monkeypatch, rule, trials):
    """Given a clock that moves one minute per recorded trial and a MaxTime limit of 5 minutes, 4.5 minutes,
    0.1 hours or one second, when optimize() runs, then the run ends with stop_reason 'time' after the first
    trial whose search time is at least the limit, and every evaluation it started was finished.

    Oracle: MaxTime docstring (state.elapsed() at least seconds + 60 * minutes + 3600 * hours; checked after
    every trial, never interrupting a running evaluation) and RunState.elapsed (the time since `started`).
    Hand count: 300 s at trial 5, 270 s first reached at trial 5 (240 s at trial 4), 360 s at trial 6, 60 s
    at trial 1.
    """
    monkeypatch.setattr(metahpo.RunState, "now", _minute_per_trial)
    calls = [0]

    study = optimize(_counting(calls), _space(), max_evals=BUDGET, seed=ANY_SEED, stop=[rule])

    assert (len(study.trials), calls[0], study.stop_reason) == (trials, trials, "time")
