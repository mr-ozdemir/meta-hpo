"""Integration tests of the BBPSOOptimizer ask/tell protocol with the real Space and optimize().

Oracles: the Optimizer docstring (tell() accepts only candidates this optimizer handed out and has not been
told yet, each at most once per call; every score must convert with float(); a failing batch raises and
leaves the optimizer unchanged; changing cand.params does not change the search), the Candidate docstring
(x in the unit cube, params as the user's function receives them, key the decoded identity), the optimize
docstring (max_evals counts calls; cut candidates are withdrawn with cancel()) and the BBPSOOptimizer.tell
and BBPSOOptimizer.cancel docstrings.

Unspecified, therefore not asserted: the message texts beyond the words matched; candidate values.

Input space: Every space, seed and ask/tell or optimize call sequence, which is unbounded.
Subset: A mixed space with every parameter kind; one seeded run per protocol rule,
the rejected inputs (foreign x, repeated candidate, told candidate, unknown candidate, bad score) and a
manual ask/tell loop that slices batches like optimize() does.
"""

import math

import numpy as np
import pytest

from metahpo import BBPSOOptimizer, Candidate, Categorical, Float, Int, Space, optimize

ANY_SEED = 0


ROUNDS = 12


def _mixed():
    """A fresh mixed space with a log Float, an Int, a Categorical and a linear Float."""
    return Space(
        {
            "lr": Float(1e-4, 1e-1, log=True),
            "units": Int(16, 256),
            "act": Categorical(["relu", "tanh", "gelu"]),
            "dropout": Float(0.0, 0.5),
        }
    )


def _score(params):
    """A smooth score on the mixed space."""
    act = {"relu": 0.1, "tanh": 0.2, "gelu": 0.0}[params["act"]]
    return (math.log10(params["lr"]) + 3.0) ** 2 + ((params["units"] - 128) / 256) ** 2 + act


def _keys_of_rounds(opt, rounds):
    """Asks `rounds` times, tells each batch its scores and returns every handed-out key."""
    keys = []
    for _ in range(rounds):
        cands = opt.ask()
        keys.extend(c.key for c in cands)
        opt.tell(cands, [_score(c.params) for c in cands])
    return keys


def _moves(opt, count):
    """Asks and tells `count` single moves; returns their candidates."""
    out = []
    for _ in range(count):
        cands = opt.ask()
        opt.tell(cands, [_score(c.params) for c in cands])
        out.extend(cands)
    return out


def _manual_loop(opt, max_evals):
    """The ask/tell loop of optimize(): batches cut to the remaining budget, the cut part cancelled, repeats
    answered from a cache; returns (key, score, cached) per trial."""
    seen, trials, calls = {}, [], 0
    while calls < max_evals:
        batch = opt.ask()
        cands, cut = batch[: max_evals - calls], batch[max_evals - calls :]
        opt.cancel(cut)
        scores = []
        for cand in cands:
            cached = cand.key in seen
            score = seen[cand.key] if cached else _score(cand.params)
            seen.setdefault(cand.key, score)
            calls += not cached
            scores.append(score)
            trials.append((cand.key, score, cached))
        opt.tell(cands, scores)
    return trials


def test_bbpso_candidates_should_carry_the_params_and_key_that_the_space_decodes_from_x():
    """Given a BBPSOOptimizer on the mixed space, when the initial batch and five moves are asked, then
    every candidate's key and params equal Space.rows of its x.

    Oracle: Candidate docstring (x, params as the function receives them, key the decoded identity).
    """
    space = _mixed()
    opt = BBPSOOptimizer(space, seed=ANY_SEED)
    init = opt.ask()
    opt.tell(init, [_score(c.params) for c in init])

    moves = _moves(opt, 5)

    cands = init + moves
    keys, params = space.rows(np.array([c.x for c in cands]))
    assert [c.key for c in cands] == keys
    assert [c.params for c in cands] == params


def test_bbpso_optimize_should_reproduce_a_manual_ask_tell_loop_that_slices_batches_like_optimize():
    """Given two BBPSOOptimizers with the same seed, when one runs inside optimize(max_evals=55) and the other
    in a manual loop that cuts and cancels batches the same way, then both produce the same trials.

    Oracle: optimize docstring (max_evals counts calls; candidates cut by the budget are withdrawn with
    cancel(); repeats are answered by the cache). 55 calls cover the initial batch, the moves and part of the
    first refinement.
    """
    space = _mixed()
    study = optimize(
        _score, space, max_evals=55, optimizer=BBPSOOptimizer(space, seed=ANY_SEED), seed=ANY_SEED
    )

    manual = _manual_loop(BBPSOOptimizer(space, seed=ANY_SEED), 55)

    keys = space.rows(np.array([t.x for t in study.trials]))[0]
    assert list(zip(keys, study.scores.tolist(), [t.cached for t in study.trials], strict=True)) == manual
    assert study.n_calls == 55


def test_bbpso_tell_should_reject_a_candidate_whose_x_was_changed_and_keep_the_optimizer_unchanged():
    """Given a BBPSOOptimizer and a twin with the same seed, when the first is told a batch whose last
    candidate carries a changed x, then ValueError is raised, and after both tell their batches correctly they
    hand out the same keys for twelve rounds.

    Oracle: Optimizer docstring ('tell() accepts only candidates this optimizer handed out'; 'a failing batch
    raises and leaves the optimizer unchanged').
    """
    opt, twin = BBPSOOptimizer(_mixed(), seed=ANY_SEED), BBPSOOptimizer(_mixed(), seed=ANY_SEED)
    batch, twin_batch = opt.ask(), twin.ask()
    last = batch[-1]
    forged = Candidate(last.id, np.clip(last.x + 0.01, 0.0, 1.0), last.params, last.key)

    with pytest.raises(ValueError, match="not handed out"):
        opt.tell([*batch[:-1], forged], [_score(c.params) for c in batch])
    opt.tell(batch, [_score(c.params) for c in batch])
    twin.tell(twin_batch, [_score(c.params) for c in twin_batch])

    assert _keys_of_rounds(opt, ROUNDS) == _keys_of_rounds(twin, ROUNDS)


def test_bbpso_tell_should_reject_a_candidate_named_twice_in_one_batch():
    """Given the initial batch, when tell() names its first candidate twice, then ValueError is raised and
    the batch can still be told once.

    Oracle: Optimizer docstring ('each at most once per call').
    """
    opt = BBPSOOptimizer(_mixed(), seed=ANY_SEED)
    batch = opt.ask()

    with pytest.raises(ValueError, match="already told"):
        opt.tell([*batch, batch[0]], [1.0] * (len(batch) + 1))
    opt.tell(batch, [1.0] * len(batch))

    assert opt.phase == "move"


def test_bbpso_tell_should_reject_a_candidate_that_was_already_told():
    """Given the initial batch told, when one of its candidates is told again, then ValueError is raised.

    Oracle: Optimizer docstring ('has not been told yet').
    """
    opt = BBPSOOptimizer(_mixed(), seed=ANY_SEED)
    batch = opt.ask()
    opt.tell(batch, [1.0] * len(batch))

    with pytest.raises(ValueError, match="already told"):
        opt.tell(batch[:1], [0.5])


@pytest.mark.parametrize("bad", [pytest.param("bad", id="string"), pytest.param(None, id="none")])
def test_bbpso_tell_should_keep_the_batch_pending_when_a_score_does_not_convert(bad):
    """Given the initial batch, when its last score cannot be converted by float(), then the error is raised
    and the whole batch can be told afterwards.

    Oracle: Optimizer docstring ('every score must convert with float()'; 'a failing batch raises and leaves
    the optimizer unchanged').
    """
    opt = BBPSOOptimizer(_mixed(), seed=ANY_SEED)
    batch = opt.ask()

    with pytest.raises((ValueError, TypeError)):
        opt.tell(batch, [1.0] * (len(batch) - 1) + [bad])
    opt.tell(batch, [1.0] * len(batch))

    assert opt.phase == "move"


def test_bbpso_should_not_change_the_search_when_the_caller_clears_params_before_telling():
    """Given a BBPSOOptimizer and a twin with the same seed, when the caller clears every candidate's params
    after scoring and before telling, then both hand out the same keys for twelve rounds.

    Oracle: Optimizer docstring ('changing cand.params does not change the search').
    """
    opt, twin = BBPSOOptimizer(_mixed(), seed=ANY_SEED), BBPSOOptimizer(_mixed(), seed=ANY_SEED)

    spoiled = _spoiled_keys(opt, ROUNDS)

    assert spoiled == _keys_of_rounds(twin, ROUNDS)


def _spoiled_keys(opt, rounds):
    """Like _keys_of_rounds, but every params dict is cleared after scoring and before the tell."""
    keys = []
    for _ in range(rounds):
        cands = opt.ask()
        keys.extend(c.key for c in cands)
        scores = [_score(c.params) for c in cands]
        for cand in cands:
            cand.params.clear()
        opt.tell(cands, scores)
    return keys


def test_bbpso_optimize_should_raise_value_error_when_the_optimizer_direction_differs():
    """Given a BBPSOOptimizer built for 'max', when optimize() runs with direction 'min', then ValueError is
    raised.

    Oracle: optimize ('the optimizer must use the same space and direction as optimize()').
    """
    space = _mixed()
    opt = BBPSOOptimizer(space, direction="max", seed=ANY_SEED)

    with pytest.raises(ValueError, match="same space and direction"):
        optimize(_score, space, max_evals=5, optimizer=opt, seed=ANY_SEED)


def test_bbpso_should_stay_usable_for_ask_and_tell_after_optimize_with_one_evaluation():
    """Given a BBPSOOptimizer passed to optimize(max_evals=1), when the caller goes on with twelve rounds of
    ask and tell, then nothing raises and the 19 cut initial particles are handed out first.

    Oracle: optimize docstring ('a passed optimizer can still be used with ask/tell afterwards').
    """
    space = _mixed()
    opt = BBPSOOptimizer(space, seed=ANY_SEED)
    optimize(_score, space, max_evals=1, optimizer=opt, seed=ANY_SEED)

    first = opt.ask()
    opt.tell(first, [_score(c.params) for c in first])

    assert (len(first), opt.phase) == (19, "move")
    assert len(_keys_of_rounds(opt, ROUNDS)) == ROUNDS
