"""direction='max' through optimize() with an explicit ABCOptimizer, checked at study, optimizer and
trajectory level.

Oracle: README ('searches for the parameters that maximise or minimise that score', direction='max'
examples), the Optimizer docstring ('Scores are given in the user's direction; internally everything is
minimised') and Study.best_trial (the best finite score for the direction).

The objective returns x on Int(0, 4), so the score equals the decoded parameter and small and large scores are
both reachable. A study-only check is not decisive: an optimizer that does not negate scores under 'max' still
leaves the study best correct. The optimizer's own best and the comparison with a minimisation of the negated
score tell them apart.

All assertions are seed-independent invariants (best equals the extreme of the evaluated scores; 'max' of f
equals 'min' of -f step by step). ANY_SEED = 0 only makes the run reproducible.

Unspecified, therefore not asserted: whether a run of MAX_EVALS evaluations reaches x = 4 (for some seeds it
stops at 3 or 2), and how many trials land on each value for a given seed.

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: One small discrete space under 'max', compared with 'min' of the negated score.
"""

import numpy as np
import pytest

from metahpo import ABCOptimizer, Int, Space, optimize

ANY_SEED = 0
MAX_EVALS = 60
LOWEST_X = 0
HIGHEST_X = 4


def _objective_returning_x(params):
    return params["x"]


def _objective_returning_minus_x(params):
    return -params["x"]


def _run(objective, direction):
    space = Space({"x": Int(LOWEST_X, HIGHEST_X)})
    opt = ABCOptimizer(space, direction=direction, seed=ANY_SEED)
    study = optimize(objective, space, direction=direction, max_evals=MAX_EVALS, optimizer=opt)
    return opt, study


def _evaluated_params(study):
    return [trial.params for trial in study.trials]


@pytest.fixture
def max_run():
    """A fresh seeded ABCOptimizer under 'max' with score x, driven by optimize() for MAX_EVALS
    evaluations."""
    return _run(_objective_returning_x, "max")


def test_optimize_should_report_the_largest_evaluated_score_as_study_best_when_direction_is_max(max_run):
    """Given a 'max' run over x in 0..4 with score x, when the study best is read, then it is the largest
    score.

    Oracle: README (maximise the score) and Study.best_trial picking the best finite score for the direction;
    because the score is x, the best parameters are {'x': best score}. The precondition that the run saw
    different scores makes the input decisive: Study.best_trial always using min would then give a smaller
    value. Site: Study.best_trial direction switch. Postcondition: none; reading the best is a query (the
    MAX_EVALS trial count is asserted as a precondition before it).
    """
    _, study = max_run
    assert len(study.trials) == MAX_EVALS
    assert np.min(study.scores) < np.max(study.scores)

    best = (study.best_score, study.best_params)

    assert best == (np.max(study.scores), {"x": np.max(study.scores)})


def test_abc_optimizer_should_report_the_largest_evaluated_score_as_its_best_when_direction_is_max(max_run):
    """Given the same 'max' run, when the optimizer's own best is read, then it is the largest evaluated
    score.

    Oracle: Optimizer docstring (scores in the user's direction, minimised internally) and the README
    ask/tell example printing opt.best_params and opt.best_score. Decisive because a missing negation under
    'max' makes the optimizer keep the smallest score (best_score -0.0, best_params {'x': 0} at this seed)
    and a missing re-negation of best_score reports the largest score with a minus sign.
    Site: Optimizer._internal and Optimizer.best_score sign handling. Postcondition: best_params and
    best_score describe the same evaluated point ({'x': best score}), so the reported best is a point the run
    evaluated.
    """
    opt, study = max_run
    assert np.min(study.scores) < np.max(study.scores)

    best = (opt.best_score, opt.best_params)

    assert best == (np.max(study.scores), {"x": np.max(study.scores)})


def test_abc_optimizer_should_evaluate_the_same_points_when_maximising_x_as_when_minimising_minus_x():
    """Given the same seed, when 'max' of x and 'min' of -x are run, then both evaluate the same parameter
    sequence.

    Oracle: Optimizer docstring 'Scores are given in the user's direction; internally everything is
    minimised', so maximising x and minimising -x are the same internal problem and the search must follow
    the same trajectory, with best scores of opposite sign. Decisive because an optimizer that minimises the
    raw score under 'max' would chase small x in one run and large x in the other.
    Site: Optimizer._internal negation under 'max'.
    """
    max_opt, max_study = _run(_objective_returning_x, "max")
    min_opt, min_study = _run(_objective_returning_minus_x, "min")

    max_points = _evaluated_params(max_study)
    min_points = _evaluated_params(min_study)

    assert max_points == min_points
    assert max_opt.best_score == -min_opt.best_score
