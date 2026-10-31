"""Hand-written optimize() example behind the cache=False property of test_fuzz_grammar.py.

Oracles: optimize docstring ("with cache=True a repeated parameter set reuses the earlier score"; nothing is
reused otherwise), Study.n_calls docstring ("calls of the user's function; cached repeats are not counted")
and README Behaviour. The positive counterpart, cache=True skipping repeated parameter sets, is
test_cache_skips_repeated_parameter_sets in tests/test_abc.py.

Unspecified, therefore not asserted: which of the two choices the seeded search visits, and in which order.

Input space: Every objective, space, direction, budget, optimizer, seed and cache flag of optimize(), which is
unbounded.
Subset: One hand-written cache=False example behind the generated property.
"""

from metahpo import Categorical, Space, optimize

ANY_SEED = 0


SCORE_OF = {"a": 1.0, "b": 2.0}


def test_optimize_should_call_the_function_for_every_trial_when_cache_is_false_on_two_choices():
    """Given Space({'k': Categorical(['a', 'b'])}) and cache=False, when optimize runs six evaluations, then
    the function is called six times, n_calls is 6 and no trial is cached.

    Oracle: optimize and Study.n_calls docstrings. Two choices give only two distinct parameter sets, so six
    evaluations must repeat one; any reuse with cache=False shows as fewer calls (site: optimize cache flag).
    ANY_SEED is arbitrary: the counts hold for every seed.
    """
    space = Space({"k": Categorical(["a", "b"])})
    calls = []

    def spy_objective(params):
        calls.append(params["k"])
        return SCORE_OF[params["k"]]

    study = optimize(spy_objective, space, max_evals=6, seed=ANY_SEED, cache=False)

    assert (len(calls), study.n_calls) == (6, 6)
    assert [t.cached for t in study.trials] == [False] * 6
