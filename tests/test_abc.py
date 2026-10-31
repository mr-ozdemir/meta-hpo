import math

import numpy as np
import pytest

from metahpo import ABCOptimizer, Categorical, Float, Int, Space, optimize
from metahpo.operators import fitness, fitness_scalar, neighbour_value, onlooker_probabilities


def sphere_space(d=4):
    return Space({f"x{i}": Float(-5.0, 5.0) for i in range(d)})


def sphere(p):
    return sum(v * v for v in p.values())


def test_neighbour_changes_one_coordinate_and_stays_in_the_cube():
    assert neighbour_value(0.5, 0.2, 1.0) == 0.8
    assert neighbour_value(0.9, 0.1, 1.0) == 1.0 and neighbour_value(0.1, 0.9, 1.0) == 0.0
    space = Space({"a": Float(0, 1), "b": Int(1, 9), "c": Categorical("xyz")})
    opt = ABCOptimizer(space, colony_size=6, seed=0)
    cands = opt.ask()
    opt.tell(cands, [0.0] * len(cands))
    for _ in range(300):
        (c,) = opt.ask()
        kind, i = opt._pending[c.id]
        if kind != "scout":
            assert np.count_nonzero(c.x != opt.foods[i]) <= 1
        assert 0.0 <= c.x.min() and c.x.max() <= 1.0
        assert c.params == space.to_dicts(c.x)[0]
        opt.tell([c], [float(c.x.sum())])


def test_fitness_and_probabilities_match_the_original_formulas():
    f = np.array([0.0, 1.0, -2.0, math.inf])
    assert fitness(f).tolist() == [1.0, 0.5, 3.0, 0.0]
    assert [fitness_scalar(v) for v in f] == [1.0, 0.5, 3.0, 0.0]
    assert onlooker_probabilities(np.array([0.5, 1.0])).tolist() == [0.55, 1.0]
    assert onlooker_probabilities(np.zeros(3)).tolist() == [1.0, 1.0, 1.0]


def test_phase_order_and_evaluation_counts():
    space = sphere_space(3)
    opt = ABCOptimizer(space, colony_size=10, limit=1000, seed=1)
    phases = []
    cands = opt.ask()
    assert len(cands) == 5
    opt.tell(cands, [sphere(c.params) for c in cands])
    for _ in range(20):
        phases.append(opt.phase)
        (c,) = opt.ask()
        opt.tell([c], [sphere(c.params)])
    assert phases[:5] == ["employed"] * 5 and phases[5:10] == ["onlooker"] * 5
    assert opt.cycle == 1 and opt.phase == "scout"


def test_scout_replaces_an_exhausted_source():
    space = sphere_space(2)
    opt = ABCOptimizer(space, colony_size=4, limit=2, seed=3)
    cands = opt.ask()
    opt.tell(cands, [1.0, 1.0])
    kinds = []
    for _ in range(30):
        (c,) = opt.ask()
        kinds.append(opt._pending[c.id][0])
        opt.tell([c], [5.0])
    assert "scout" in kinds


def test_same_seed_same_result_and_sphere_converges():
    a = optimize(sphere, sphere_space(), max_evals=2000, seed=7)
    b = optimize(sphere, sphere_space(), max_evals=2000, seed=7)
    assert a.scores.tolist() == b.scores.tolist()
    assert a.best_score < 1e-2


def test_direction_max_and_nan_handling():
    space = Space({"x": Float(0.0, 1.0)})

    def score(p):
        return math.nan if p["x"] < 0.2 else p["x"]

    study = optimize(score, space, direction="max", max_evals=300, seed=0)
    assert study.best_score > 0.95
    assert any(t.failed for t in study.trials)
    assert not study.best_trial.failed


def test_cache_skips_repeated_parameter_sets():
    space = Space({"k": Categorical([1, 2, 3]), "n": Int(1, 2)})
    calls = []

    def f(p):
        calls.append(p)
        return p["k"] * p["n"]

    study = optimize(f, space, max_evals=100, seed=0)
    assert len(study.trials) == 100
    assert study.n_calls == len(calls) <= 6
    assert study.best_score == 1


def test_budget_is_exact_even_inside_the_initial_population():
    study = optimize(sphere, sphere_space(), max_evals=3, seed=0)
    assert len(study.trials) == 3


def test_protocol_errors():
    opt = ABCOptimizer(sphere_space(), seed=0)
    cands = opt.ask()
    opt.tell(cands, [1.0] * len(cands))
    (c,) = opt.ask()
    with pytest.raises(RuntimeError):
        opt.ask()
    with pytest.raises(ValueError):
        opt.tell([cands[0]], [1.0])
    with pytest.raises(ValueError):
        ABCOptimizer(sphere_space(), colony_size=3)
