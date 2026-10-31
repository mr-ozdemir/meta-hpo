"""Public API contract of metahpo and smoke tests of its default configuration.

Contract tests for the public interface of metahpo.

When the public interface changes, earlier tests may no longer run as they are. These tests pin the
documented public surface (exported names, constructor and function signatures, the ask/tell members,
dataclass fields and properties), so a change to it fails here, under a clear name, instead of breaking
other tests in ways that are hard to read. Only 'import metahpo' runs at module level and every name is
looked up through the package inside a test, so a renamed or removed export gives named FAILED results
and never a collection error that would stop the rest of the suite.

Oracles: README (Quick start, Ask and tell, Search space table, Behaviour), CHANGELOG 'Added', the
docstrings of facade.optimize, base.Candidate, base.Optimizer, study.Trial.failed and study.Study.n_calls,
and the package __init__ export list.

Characterised, not documented: these values come from the current public signature and are pinned so that
a change shows up as a named failure, not because a spec source states them:
- the defaults max_evals=500, direction='min', seed=None of optimize (cache=True and optimizer=None
  follow from README Behaviour and the facade docstring);
- the defaults direction='min', colony_size=10, limit=15, seed=None, max_tries=100 of ABCOptimizer (the
  README passes colony_size=10 and limit=15 explicitly without calling them defaults);
- the field order of Candidate and Trial, and the Trial field names;
- Study.scores and Study.best_trial as properties (no docstring or README line names them).

Unspecified, and therefore not asserted:
- the value and format of __version__ (CHANGELOG names Semantic Versioning, the value is PEP 440);
- parameter kinds (positional-only, keyword-only), which no spec source states;
- type annotations and return types;
- the parameter names of tell, which no docstring states (the README passes both arguments positionally);
- whether Categorical's 'choices' has a default (the class reports one, yet Categorical() is rejected);
- whether Study.trials is a constructor field or a plain attribute beyond being a dataclass field;
- the constructor signature of the Optimizer base class, which the README does not document.

Smoke tests of the default configuration of metahpo.

Unit tests cannot catch configuration problems of the whole system, so these tests start the library
with its default options only and pass if it starts and produces a result of the documented shape,
without asserting much else. Only 'import metahpo' runs at module level, so a renamed export fails these
tests by name instead of interrupting collection.

Oracles: facade.optimize docstring ('The default optimizer is ABCOptimizer', 'max_evals counts the
calls of function'), README Behaviour ('cache=True (the default)'), CHANGELOG 'Added' ('optimize()
... returning a Study with every trial') and the original ABC.py (FoodNumber = NP/2, initial population
evaluated as one batch).

Characterised, not documented: the defaults max_evals=500, direction='min' and colony_size=10 come from
the current public signature; they are pinned so that a change shows up as a named failure.

Unspecified, and therefore not asserted: the quality of the result of a default run (best_score) and
the number of calls of the objective.

Optimizer contract tests for every optimizer.

OPTIMIZER_FACTORIES lists one factory per optimizer of the library, each called as factory(space, seed).
Today it holds ABCOptimizer only. Every new optimizer (GA, PSO, DE, ...) must be added to this list, so
that it runs the same contract tests that pin defects once found in ABCOptimizer: a candidate of another
optimizer that carries a pending id is rejected; a score that float() rejects raises and keeps the
candidate pending; a failing batch leaves the optimizer unchanged (checked against a twin with the same
seed that never sees the failing call); clearing cand.params between ask and tell does not change the
search; optimize(max_evals=1, optimizer=opt) leaves opt usable for ask and tell; and the median best over
21 seeds on the 3-D sphere scaled by 1e-20, divided by the scale, stays within a factor of 100 of the
unscaled median with the same seeds and the same number of calls.

Oracles: the Optimizer docstring (the contract every optimizer keeps), the optimize docstring (cancel of
the candidates a budget cuts off) and the Candidate docstring.

Input space: The public names, signatures and defaults of metahpo, and every optimizer with every
ask/tell sequence.
Subset: Every documented name, field and default once, plus one smoke run of the default configuration,
and one seeded run per contract rule and optimizer factory.
"""

import dataclasses
import inspect
import math
import sys

import numpy as np
import pytest

import metahpo

DOCUMENTED_EXPORTS = (
    "ABCOptimizer",
    "Candidate",
    "Categorical",
    "Float",
    "Int",
    "Optimizer",
    "Space",
    "Study",
    "Trial",
    "optimize",
)


IRRELEVANT_NUMBER = 0


IRRELEVANT_CACHED = False


ANY_SEED = 0


SMALLEST_COLONY = 4


CONSTANT_SCORE = 2.0


def _trial_with_score(score):
    """Build a Trial through its public constructor where only the score matters."""
    return metahpo.Trial(
        number=IRRELEVANT_NUMBER,
        params={},
        score=score,
        cached=IRRELEVANT_CACHED,
        x=np.zeros(0),
    )


def _params_of(candidates):
    """List the params dicts of the given candidates, in ask order."""
    return [c.params for c in candidates]


def test_metahpo_should_export_exactly_the_documented_names():
    """Given the package, when __all__ is read, then it holds exactly the ten documented names once each.

    Oracle: README Quick start (Space, Float, Int, Categorical, optimize, ABCOptimizer), CHANGELOG 'Added'
    (Study) and the __init__ export list (Candidate, Optimizer, Trial). Set equality is decisive: an added
    name and a removed name both fail, and the length check catches a duplicated entry.
    """
    expected = set(DOCUMENTED_EXPORTS)

    exported = list(metahpo.__all__)

    assert set(exported) == expected
    assert len(exported) == len(expected)


@pytest.mark.parametrize(
    ("name", "expected_kind"),
    [
        pytest.param("ABCOptimizer", inspect.isclass, id="ABCOptimizer"),
        pytest.param("Candidate", inspect.isclass, id="Candidate"),
        pytest.param("Categorical", inspect.isclass, id="Categorical"),
        pytest.param("Float", inspect.isclass, id="Float"),
        pytest.param("Int", inspect.isclass, id="Int"),
        pytest.param("Optimizer", inspect.isclass, id="Optimizer"),
        pytest.param("Space", inspect.isclass, id="Space"),
        pytest.param("Study", inspect.isclass, id="Study"),
        pytest.param("Trial", inspect.isclass, id="Trial"),
        pytest.param("optimize", inspect.isfunction, id="optimize"),
    ],
)
def test_metahpo_should_expose_each_documented_name_as_an_attribute(name, expected_kind):
    """Given a hard-coded documented name, when it is looked up on the package, then it exists and is a class,
    or for optimize a function.

    Oracle: README Quick start uses the classes as mh.X(...) constructors and mh.optimize(...) as a
    function. The names are hard-coded rather than read from __all__, so a rename with a stale __all__ is
    caught. The expected predicate is decisive in both directions: inspect.isclass rejects a class export
    rebound to None, a module or a function, and inspect.isfunction rejects optimize rebound to None, a
    module or a class. A missing name fails with the message 'metahpo.<name> is missing' instead of
    interrupting collection.
    """
    assert hasattr(metahpo, name), f"metahpo.{name} is missing"

    attribute = getattr(metahpo, name)

    assert expected_kind(attribute) is True, f"metahpo.{name} is {attribute!r}"


def test_metahpo_should_define_version_as_a_non_empty_string():
    """Given the package, when __version__ is read, then it is a non-empty str.

    Oracle: the package defines __version__; no spec source fixes its value or format, so only its type
    and non-emptiness are asserted.
    """
    expected_type = str

    version = metahpo.__version__

    assert type(version) is expected_type
    assert version.strip() != ""


def test_optimize_should_declare_parameters_in_documented_order():
    """Given optimize, when its signature is introspected, then the parameter names come in the documented
    order.

    Oracle: facade.optimize docstring ('function', 'space', 'max_evals', 'cache') and README calls
    optimize(train, space, direction=..., max_evals=..., seed=...); order matters because function and
    space are passed positionally in the README. The position of optimizer, seed and cache is characterised
    from the current public signature.
    """
    expected = ["function", "space", "direction", "max_evals", "optimizer", "seed", "cache"]

    names = list(inspect.signature(metahpo.optimize).parameters)

    assert names == expected


@pytest.mark.parametrize("name", ["function", "space"], ids=["function", "space"])
def test_optimize_should_require_parameter_when_it_has_no_documented_default(name):
    """Given optimize, when a required parameter is introspected, then it has no default.

    Oracle: optimize(function, space, ...) in the facade docstring and the README.
    """
    expected = inspect.Parameter.empty

    default = inspect.signature(metahpo.optimize).parameters[name].default

    assert default is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("direction", "min", id="direction-min"),
        pytest.param("max_evals", 500, id="max_evals-500"),
        pytest.param("optimizer", None, id="optimizer-None"),
        pytest.param("seed", None, id="seed-None"),
        pytest.param("cache", True, id="cache-True"),
    ],
)
def test_optimize_should_declare_documented_default(name, expected):
    """Given optimize, when an optional parameter is introspected, then its default equals the pinned value.

    Oracle: README Behaviour ('cache=True (the default)') and facade docstring ('The default optimizer is
    ABCOptimizer', so optimizer defaults to None). direction='min', max_evals=500 and seed=None are
    characterised from the current public signature (see the module docstring). The type check is
    decisive: it rejects 1 for True and 500.0 for 500, which equality alone would accept.
    """
    expected_type = type(expected)

    default = inspect.signature(metahpo.optimize).parameters[name].default

    assert default == expected
    assert type(default) is expected_type


def test_abc_optimizer_should_declare_constructor_parameters_in_documented_order():
    """Given the ABCOptimizer class, when its constructor is introspected, then space comes first and self is
    hidden.

    Oracle: README Ask and tell ABCOptimizer(space, direction=..., colony_size=..., limit=..., seed=...);
    space is passed positionally there, so it must be the first parameter. max_tries is the last keyword
    (the Optimizer docstring: a new parameter set is searched for in `max_tries` attempts), after seed, so
    the README call keeps its meaning.
    """
    expected = ["space", "direction", "colony_size", "limit", "seed", "max_tries"]

    names = list(inspect.signature(metahpo.ABCOptimizer).parameters)

    assert names == expected


def test_abc_optimizer_should_require_space():
    """Given the ABCOptimizer class, when the space parameter is introspected, then it has no default.

    Oracle: README Ask and tell passes the space as the first positional argument.
    """
    expected = inspect.Parameter.empty

    default = inspect.signature(metahpo.ABCOptimizer).parameters["space"].default

    assert default is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("direction", "min", id="direction-min"),
        pytest.param("colony_size", 10, id="colony_size-10"),
        pytest.param("limit", 15, id="limit-15"),
        pytest.param("seed", None, id="seed-None"),
        pytest.param("max_tries", 100, id="max_tries-100"),
    ],
)
def test_abc_optimizer_should_declare_documented_default(name, expected):
    """Given the ABCOptimizer class, when an optional constructor parameter is introspected, then its default
    is the pinned value.

    Oracle: characterisation of the current public signature (see the module docstring); the README
    passes colony_size=10 and limit=15 explicitly without calling them defaults. 10, 15 and 100 are
    discriminating: any other default fails, and the type check rejects 10.0 for 10.
    """
    expected_type = type(expected)

    default = inspect.signature(metahpo.ABCOptimizer).parameters[name].default

    assert default == expected
    assert type(default) is expected_type


@pytest.mark.parametrize("name", ["ask", "tell"], ids=["ask", "tell"])
def test_abc_optimizer_should_expose_ask_and_tell_as_methods(name):
    """Given the ABCOptimizer class, when ask or tell is looked up statically, then it is a plain function,
    that is a method.

    Oracle: README Ask and tell (opt.ask() and opt.tell(candidates, scores) are called) and the
    base.Optimizer docstring ('ask() returns ...; tell() reports ...'). Looked up with
    inspect.getattr_static on the class, so no instance is built. Decisive against a removed member
    (missing attribute) and against ask or tell turned into a property or a plain value.
    """
    assert hasattr(metahpo.ABCOptimizer, name), f"ABCOptimizer.{name} is missing"

    attribute = inspect.getattr_static(metahpo.ABCOptimizer, name)

    assert inspect.isfunction(attribute) is True, f"ABCOptimizer.{name} is {attribute!r}"


@pytest.mark.parametrize("name", ["best_params", "best_score"], ids=["best_params", "best_score"])
def test_abc_optimizer_should_expose_best_params_and_best_score_as_properties(name):
    """Given the ABCOptimizer class, when best_params or best_score is looked up statically, then it is a
    property.

    Oracle: README Ask and tell reads opt.best_params and opt.best_score without a call. Looked up with
    inspect.getattr_static on the class, so no property code runs. Decisive against a removed member
    (missing attribute) and against best_score or best_params turned into a method (the attribute
    becomes a plain function, not a property).
    """
    expected_type = property

    attribute = inspect.getattr_static(metahpo.ABCOptimizer, name)

    assert type(attribute) is expected_type, f"ABCOptimizer.{name} is {attribute!r}"


def test_abc_optimizer_should_accept_documented_ask_then_tell_call_shape():
    """Given an ABCOptimizer with colony_size 4, when the README calls opt.ask() and opt.tell(candidates,
    scores) run once with a constant score, then best_score is that score and best_params is one of the told
    params.

    Oracle: README Ask and tell call shapes (ask with no argument, tell with candidates and a list of scores
    passed positionally, best_params and best_score read afterwards), the base.Optimizer docstring and a
    hand calculation: every told score is 2.0, so the best of them is 2.0 whichever candidate wins (which
    one wins is not asserted). Decisive against a renamed ask or tell: the inherited Optimizer placeholder
    or a missing method makes this named test fail, which the static member check alone cannot see.
    colony_size 4 is the smallest valid colony and keeps the first batch at two candidates.
    """
    space = metahpo.Space({"a": metahpo.Float(0.0, 1.0)})
    opt = metahpo.ABCOptimizer(space, colony_size=SMALLEST_COLONY, seed=ANY_SEED)

    candidates = opt.ask()
    opt.tell(candidates, [CONSTANT_SCORE] * len(candidates))
    best_score = opt.best_score
    best_params = opt.best_params
    told_params = _params_of(candidates)

    assert best_score == CONSTANT_SCORE
    assert best_params in told_params


@pytest.mark.parametrize("cls_name", ["Float", "Int"])
def test_numeric_parameter_should_declare_low_high_log(cls_name):
    """Given Float or Int, when its constructor is introspected, then the parameters are low, high, log in
    that order.

    Oracle: README Search space table Float(low, high, log=False) and Int(low, high, log=False).
    """
    expected = ["low", "high", "log"]

    names = list(inspect.signature(getattr(metahpo, cls_name)).parameters)

    assert names == expected


@pytest.mark.parametrize("cls_name", ["Float", "Int"])
def test_numeric_parameter_should_require_low_and_high(cls_name):
    """Given Float or Int, when low and high are introspected, then neither has a default.

    Oracle: README Search space table Float(low, high, ...) and Int(low, high, ...).
    """
    expected = inspect.Parameter.empty

    params = inspect.signature(getattr(metahpo, cls_name)).parameters

    assert params["low"].default is expected
    assert params["high"].default is expected


@pytest.mark.parametrize("cls_name", ["Float", "Int"])
def test_numeric_parameter_should_default_log_to_false(cls_name):
    """Given Float or Int, when log is introspected, then its default is False itself (not 0 or None).

    Oracle: README Search space table 'log=False'.
    """
    expected = False

    default = inspect.signature(getattr(metahpo, cls_name)).parameters["log"].default

    assert default is expected


def test_categorical_should_declare_choices_as_its_only_parameter():
    """Given Categorical, when its constructor is introspected, then its only parameter is choices.

    Oracle: README Search space table Categorical(choices). Its default is unspecified and not asserted.
    """
    expected = ["choices"]

    names = list(inspect.signature(metahpo.Categorical).parameters)

    assert names == expected


def test_candidate_should_expose_documented_fields_in_order():
    """Given Candidate, when its dataclass fields are listed, then they are id, x, params, key in that order.

    Oracle: base.Candidate docstring (x, params, key), README 'c.params', and the id that tell uses to match
    a candidate; the order is characterised from the current public signature. Order equality catches a
    reordering (positional construction breaks) as well as a rename such as params to parameters.
    """
    expected = ["id", "x", "params", "key"]

    names = [f.name for f in dataclasses.fields(metahpo.Candidate)]

    assert names == expected


def test_trial_should_expose_documented_fields_in_order():
    """Given Trial, when its dataclass fields are listed, then they are number, params, score, cached, x in
    order.

    Oracle: characterisation of the current public signature (see the module docstring); Trial.failed's
    docstring refers to the score and Study.n_calls's docstring to cached repeats.
    """
    expected = ["number", "params", "score", "cached", "x"]

    names = [f.name for f in dataclasses.fields(metahpo.Trial)]

    assert names == expected


def test_trial_should_define_failed_as_a_property():
    """Given the Trial class, when its failed attribute is read on the class, then it is a property.

    Oracle: study.Trial.failed docstring ('True when the user's function returned NaN or an infinite value').
    """
    expected_type = property

    attribute = inspect.getattr_static(metahpo.Trial, "failed")

    assert type(attribute) is expected_type


@pytest.mark.parametrize(
    "score",
    [
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="inf"),
        pytest.param(-math.inf, id="neg_inf"),
    ],
)
def test_trial_failed_should_be_true_when_score_is_non_finite(score):
    """Given a Trial with a non-finite score, when failed is read, then it is True.

    Oracle: Trial.failed docstring and README 'A score of NaN or infinity counts as the worst'. The three
    equivalence classes of non-finite values are covered; -inf is decisive against an implementation
    that only compares with +inf.
    """
    trial = _trial_with_score(score)

    failed = trial.failed

    assert failed is True


@pytest.mark.parametrize(
    "score",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.5, id="negative"),
        pytest.param(sys.float_info.max, id="max_finite"),
    ],
)
def test_trial_failed_should_be_false_when_score_is_finite(score):
    """Given a Trial with a finite score, when failed is read, then it is False.

    Oracle: Trial.failed docstring. Zero and a negative value are ordinary finite classes; the largest
    finite double is the boundary next to infinity.
    """
    trial = _trial_with_score(score)

    failed = trial.failed

    assert failed is False


def test_study_should_declare_direction_and_trials_as_fields():
    """Given the Study class, when its dataclass fields are listed, then direction and trials are among them.

    Oracle: CHANGELOG 'Added' (optimize() 'returning a Study with every trial') and the direction the
    Study is created with. Membership, not exact equality, is used because no source fixes the
    full field list.
    """
    expected = {"direction", "trials"}

    names = {f.name for f in dataclasses.fields(metahpo.Study)}

    assert names >= expected


@pytest.mark.parametrize(
    "name",
    ["scores", "best_trial", "best_params", "best_score", "n_calls"],
    ids=["scores", "best_trial", "best_params", "best_score", "n_calls"],
)
def test_study_should_expose_documented_attribute_as_property(name):
    """Given the Study class, when a documented attribute is read on the class, then it is a property.

    Oracle: README 'study.best_params, study.best_score', CHANGELOG 'Added' ('every trial and the best one')
    and the Study.n_calls docstring; scores and best_trial are characterised (see the module docstring).
    Checked on the class so best_trial's ValueError on an empty study cannot turn a failure into an error.
    """
    expected_type = property

    attribute = inspect.getattr_static(metahpo.Study, name)

    assert type(attribute) is expected_type, f"Study.{name} is not a property"


DEFAULT_MAX_EVALS = 500


DEFAULT_COLONY_SIZE = 10


def test_optimize_should_run_to_the_default_budget_when_only_function_space_and_seed_are_given():
    """Given a two-Float space and a cheap sum of squares, when optimize gets only function, space and seed,
    then it returns a minimising Study of 500 trials and 500 calls.

    Oracle: facade.optimize docstring ('max_evals counts the calls of function'), CHANGELOG 'Added'
    ('returning a Study with every trial') and the characterised defaults direction='min', max_evals=500.
    The seed only makes the run deterministic; any fixed value works. 500 is decisive: a changed default
    budget or a broken budget loop changes the call count. Two Floats never repeat a parameter set, so the
    trial count equals the call count.
    """
    space = metahpo.Space({"a": metahpo.Float(-1.0, 1.0), "b": metahpo.Float(-1.0, 1.0)})

    def dummy_objective(params):
        """Cheap objective whose value is irrelevant to the smoke test."""
        return params["a"] ** 2 + params["b"] ** 2

    study = metahpo.optimize(dummy_objective, space, seed=ANY_SEED)

    assert type(study) is metahpo.Study
    assert len(study.trials) == DEFAULT_MAX_EVALS
    assert study.n_calls == DEFAULT_MAX_EVALS
    assert study.direction == "min"


def test_abc_optimizer_should_ask_half_the_default_colony_as_first_batch_when_built_with_defaults():
    """Given an ABCOptimizer built with only a space and a seed, when ask() is called once, then it returns
    colony_size // 2 = 5 Candidates.

    Oracle: original ABC.py FoodNumber = NP/2 with the initial population evaluated as one batch, and the
    characterised default colony_size=10. 5 is decisive: colony_size 8 would give 4 and 12 would give 6,
    and asking one candidate at a time would give 1.
    """
    space = metahpo.Space({"a": metahpo.Float(-1.0, 1.0), "b": metahpo.Float(-1.0, 1.0)})
    opt = metahpo.ABCOptimizer(space, seed=ANY_SEED)
    expected_batch = DEFAULT_COLONY_SIZE // 2

    candidates = opt.ask()

    assert len(candidates) == expected_batch
    assert {type(c) for c in candidates} == {metahpo.Candidate}


OPTIMIZER_FACTORIES = [
    pytest.param(lambda space, seed: metahpo.ABCOptimizer(space, seed=seed), id="abc"),
]


CONTRACT_SEED = 0


FOREIGN_SEED = 7


CONTRACT_ROUNDS = 40


SPHERE_SCALE = 1e-20


SPHERE_SEEDS = tuple(range(21))


SPHERE_EVALS = 300


SPHERE_MEDIAN_FACTOR = 100.0


def _contract_space():
    """The 3-D box [-5, 5]^3 of three linear Floats, the domain of the sphere."""
    return metahpo.Space({name: metahpo.Float(-5.0, 5.0) for name in ("a", "b", "c")})


def _sphere(params):
    """Sum of the squared parameter values; its minimum 0 lies at the centre of the box."""
    return sum(value * value for value in params.values())


def _sphere_scores(cands):
    """The sphere score of each candidate."""
    return [_sphere(c.params) for c in cands]


def _with_id_of(cand, other):
    """A public Candidate with the id of `cand` and the x, params and key of `other`."""
    return metahpo.Candidate(cand.id, other.x, other.params, other.key)


def _run_rounds(opt, rounds, spoil=False):
    """Asks `rounds` times and tells each batch its sphere scores in one call; returns the handed-out keys,
    copies of the handed-out params and the final best_score. With spoil=True every candidate's params dict
    is cleared and given a stray entry after scoring and before the tell."""
    keys, handed = [], []
    for _ in range(rounds):
        cands = opt.ask()
        keys.extend(c.key for c in cands)
        handed.extend(dict(c.params) for c in cands)
        scores = _sphere_scores(cands)
        for cand in cands:
            if spoil:
                cand.params.clear()
                cand.params["stray"] = None
        opt.tell(cands, scores)
    return keys, handed, opt.best_score


def _bad_last_score(batch, foreign):
    """The whole batch with a last score that float() cannot convert."""
    return batch, _sphere_scores(batch)[:-1] + ["bad"]


def _foreign_last_candidate(batch, foreign):
    """The batch whose last candidate is replaced by a candidate of another optimizer with the same id."""
    return batch[:-1] + [_with_id_of(batch[-1], foreign[-1])], _sphere_scores(batch)


def _first_candidate_twice(batch, foreign):
    """The whole batch plus its first candidate a second time."""
    return batch + batch[:1], _sphere_scores(batch + batch[:1])


def _median_scaled_best(make_optimizer, scale):
    """Runs optimize() on `scale` times the sphere for every seed of SPHERE_SEEDS with SPHERE_EVALS calls;
    returns the median of best_score / scale and the number of calls of each run."""
    space = _contract_space()
    bests, calls = [], []
    for seed in SPHERE_SEEDS:
        study = metahpo.optimize(
            lambda params: scale * _sphere(params),
            space,
            max_evals=SPHERE_EVALS,
            optimizer=make_optimizer(space, seed),
            seed=seed,
        )
        bests.append(study.best_score / scale)
        calls.append(study.n_calls)
    return float(np.median(bests)), calls


@pytest.mark.parametrize("make_optimizer", OPTIMIZER_FACTORIES)
def test_optimizer_tell_should_reject_a_candidate_of_another_optimizer_when_it_carries_a_pending_id(
    make_optimizer,
):
    """Given an optimizer with its first batch pending, when it is told a candidate of another optimizer of
    the same kind that carries the id of a pending candidate, then ValueError is raised, and the own batch
    can still be told, giving the best of its scores.

    Oracle: Optimizer docstring ('tell() accepts only candidates this optimizer handed out'). Decisive input:
    ids are counted per optimizer, so the id alone does not tell a foreign candidate apart. Every optimizer
    must be added to OPTIMIZER_FACTORIES.
    """
    space = _contract_space()
    opt = make_optimizer(space, CONTRACT_SEED)
    own, foreign = opt.ask(), make_optimizer(space, FOREIGN_SEED).ask()

    with pytest.raises(ValueError):
        opt.tell([_with_id_of(own[0], foreign[0])], [CONSTANT_SCORE])
    opt.tell(own, _sphere_scores(own))

    assert opt.best_score == min(_sphere_scores(own))


@pytest.mark.parametrize("make_optimizer", OPTIMIZER_FACTORIES)
@pytest.mark.parametrize("bad_score", [pytest.param("bad", id="string"), pytest.param(None, id="none")])
def test_optimizer_tell_should_keep_the_candidate_pending_when_its_score_does_not_convert(
    make_optimizer, bad_score
):
    """Given an optimizer with its first batch pending, when its first candidate is told a score that float()
    rejects, then the float() error is raised, and the whole batch can then be told, giving the best of
    its scores.

    Oracle: Optimizer docstring ('every score must convert with float()'; 'a failing batch raises and leaves
    the optimizer unchanged'). Decisive: an optimizer that drops the candidate before converting the score
    raises on the second tell. Every optimizer must be added to OPTIMIZER_FACTORIES.
    """
    space = _contract_space()
    opt = make_optimizer(space, CONTRACT_SEED)
    batch = opt.ask()

    with pytest.raises((ValueError, TypeError)):
        opt.tell(batch[:1], [bad_score])
    opt.tell(batch, _sphere_scores(batch))

    assert opt.best_score == min(_sphere_scores(batch))


@pytest.mark.parametrize("make_optimizer", OPTIMIZER_FACTORIES)
@pytest.mark.parametrize(
    "failing_batch",
    [
        pytest.param(_bad_last_score, id="last-score-not-a-number"),
        pytest.param(_foreign_last_candidate, id="last-candidate-foreign"),
        pytest.param(_first_candidate_twice, id="first-candidate-twice"),
    ],
)
def test_optimizer_tell_should_leave_the_optimizer_unchanged_when_the_batch_fails(
    make_optimizer, failing_batch
):
    """Given an optimizer and a twin with the same seed, when the optimizer is first told a batch whose last
    part fails, then ValueError is raised and, once both are told their first batch, they hand out the same
    keys and params and reach the same best_score over CONTRACT_ROUNDS asks.

    Oracle: Optimizer docstring ('The whole batch is checked first'; 'a failing batch raises and leaves the
    optimizer unchanged'). The twin never sees the failing tell, so it is the reference. Every optimizer
    must be added to OPTIMIZER_FACTORIES.
    """
    space = _contract_space()
    opt, twin = make_optimizer(space, CONTRACT_SEED), make_optimizer(space, CONTRACT_SEED)
    batch, twin_batch = opt.ask(), twin.ask()
    cands, scores = failing_batch(batch, make_optimizer(space, FOREIGN_SEED).ask())

    with pytest.raises(ValueError):
        opt.tell(cands, scores)
    opt.tell(batch, _sphere_scores(batch))
    twin.tell(twin_batch, _sphere_scores(twin_batch))

    assert _run_rounds(opt, CONTRACT_ROUNDS) == _run_rounds(twin, CONTRACT_ROUNDS)


@pytest.mark.parametrize("make_optimizer", OPTIMIZER_FACTORIES)
def test_optimizer_should_hand_out_the_same_candidates_when_the_caller_clears_the_params_before_telling(
    make_optimizer,
):
    """Given an optimizer and a twin with the same seed, when the caller clears every candidate's params dict
    after scoring it and before telling it, then both hand out the same keys and params and reach the same
    best_score over CONTRACT_ROUNDS asks.

    Oracle: Optimizer docstring ('changing `cand.params` does not change the search') and the Candidate
    docstring (params as the user's function receives them). Every optimizer must be added to
    OPTIMIZER_FACTORIES.
    """
    space = _contract_space()
    opt, twin = make_optimizer(space, CONTRACT_SEED), make_optimizer(space, CONTRACT_SEED)

    spoiled = _run_rounds(opt, CONTRACT_ROUNDS, spoil=True)
    reference = _run_rounds(twin, CONTRACT_ROUNDS)

    assert spoiled == reference


@pytest.mark.parametrize("make_optimizer", OPTIMIZER_FACTORIES)
def test_optimize_should_leave_the_passed_optimizer_usable_for_ask_and_tell_when_max_evals_is_one(
    make_optimizer,
):
    """Given an optimizer passed to optimize() with max_evals=1, when the caller goes on with CONTRACT_ROUNDS
    rounds of ask and tell, then nothing raises, every round hands out candidates and the best score does
    not get worse than the one evaluated trial.

    Oracle: optimize docstring ('the candidates that are not evaluated are withdrawn with
    `optimizer.cancel()`, so a passed optimizer can still be used with ask/tell afterwards'). Decisive: a
    first batch larger than one candidate is cut to one, and an optimizer that keeps the rest pending
    refuses to ask again. Every optimizer must be added to OPTIMIZER_FACTORIES.
    """
    space = _contract_space()
    opt = make_optimizer(space, CONTRACT_SEED)
    study = metahpo.optimize(_sphere, space, max_evals=1, optimizer=opt, seed=CONTRACT_SEED)

    keys, _, best_score = _run_rounds(opt, CONTRACT_ROUNDS)

    assert len(study.trials) == 1
    assert len(keys) >= CONTRACT_ROUNDS
    assert best_score <= study.best_score


@pytest.mark.parametrize("make_optimizer", OPTIMIZER_FACTORIES)
def test_optimizer_should_converge_as_well_on_a_sphere_scaled_by_1e_minus_20_when_evaluations_are_equal(
    make_optimizer,
):
    """Given the 3-D sphere on [-5, 5]^3 and the same sphere times 1e-20, when optimize() runs 300 calls for
    each of 21 seeds on both, then the runs use the same number of calls and the median of best_score / 1e-20
    on the scaled sphere is at most 100 times the median best on the unscaled one.

    Oracle: Optimizer docstring (scores are only compared, internally everything is minimised), so
    multiplying every score by a positive constant must not ruin the search. Statistic: the median over 21
    seeds is robust against a few unlucky runs. Bound: the two medians differ by a factor of about 0.8 to 2
    for ABC (onlooker probabilities, which use 1 / (1 + f), become uniform at 1e-20); an acceptance rule on
    1 / (1 + f) rejects every neighbour at that scale, because all fitness values round to 1.0, and gives
    medians about 10**4 times worse, so 100 is generous yet still decisive. Every optimizer must be added to
    OPTIMIZER_FACTORIES.
    """
    unscaled, unscaled_calls = _median_scaled_best(make_optimizer, 1.0)
    scaled, scaled_calls = _median_scaled_best(make_optimizer, SPHERE_SCALE)

    assert scaled_calls == unscaled_calls
    assert scaled <= SPHERE_MEDIAN_FACTOR * unscaled
