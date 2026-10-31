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

Input space: The public names, signatures and defaults of metahpo.
Subset: Every documented name, field and default once, plus one smoke run of the default configuration.
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
