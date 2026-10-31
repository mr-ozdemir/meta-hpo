"""Requirement-test traceability matrix and suite health checks.

REQUIREMENTS maps each requirement id to its spec sources and to the tests that verify it. A source item is
'<LOCATION> | <verbatim quote>', where LOCATION is README.md, CHANGELOG.md, CONTRIBUTING.md or the dotted
path of a documented object (its docstring holds the quote), or a note starting with 'derived:' for facts
that only the computed task specification or the original hyp_optimizer ABC.py states (ABC.py lives outside
the repository and cannot be read in CI), or 'untestable:' for a source that is not available at all.
A test is listed as '<module>::<function>' or '<module>::<Class>::<method>'.

The matrix is checked in both directions: every requirement has at least one existing test and every test
in the modules of TRACED_MODULES belongs to at least one requirement. The quoted sentences are re-read from
the files on every run, so a changed README, CHANGELOG or docstring shows up as an outdated requirement, and
every README Behaviour bullet, Search space row and CHANGELOG 'Added' bullet must be quoted by some entry.

The meta-tests of this module are not listed in the matrix; they check the matrix and the suite itself.
Further health checks run over every tests/test_*.py: only pytest, numpy, metahpo, the suite's own
conftest and standard-library imports, no clock or global random module, no np.random call on numpy's
global state, seeded generators, a seed for every ABCOptimizer and optimize call outside pytest.raises, no
comment lines, no placeholder or unconditionally skipped test, no unfinished-work marker, and for every new
test a docstring and a body without if statements, loops or conditional expressions; no two tests have the
same body. Each checker is a pure function with
its own self-test on small snippets.

TRACED_MODULES holds the original test_abc.py and test_space.py and the specification-gap modules
written together with this matrix. The other test modules of the suite trace their tests to the rule ids of
the test-writing rules and to library requirements in docs/test-izlenebilirlik.md and in their own module
docstrings, so the orphan check covers TRACED_MODULES only. When a test is added to one of these modules,
it must be added to REQUIREMENTS; the orphan check fails until it is.

Input space: The requirement matrix, the documents and the test files, which are unbounded.
Subset: The real files as they are on disk, plus small snippets for each checker.
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

import pytest

import metahpo

TESTS_DIR = Path(__file__).resolve().parent
THIS_FILE = Path(__file__).resolve().name
DOC_FILES = ("README.md", "CHANGELOG.md", "CONTRIBUTING.md")
EXEMPT_PREFIXES = ("derived:", "untestable:")
ALLOWED_TOP_MODULES = frozenset({"pytest", "numpy", "metahpo", "conftest"})
FLAKY_MODULES = frozenset({"time", "datetime", "random"})
MARKERS = ("TO" + "DO", "FIX" + "ME", "X" * 3)
ID_PATTERN = re.compile(r"R-[A-Z0-9-]+")
TEST_REF_PATTERN = re.compile(r"test\w*::(Test\w*::)?test\w*")
GENERATOR_API = frozenset(
    {
        "default_rng",
        "Generator",
        "SeedSequence",
        "BitGenerator",
        "PCG64",
        "PCG64DXSM",
        "Philox",
        "SFC64",
        "MT19937",
    }
)
SEED_POSITION = {"ABCOptimizer": 4, "optimize": 5}
TRACED_MODULES = frozenset(
    {
        "test_abc",
        "test_space",
        "test_abc_spec",
        "test_base_spec",
        "test_facade_spec",
        "test_space_spec",
        "test_study_spec",
    }
)
OPTIMIZER_POSITION = 4
CONTROL_FLOW_NODES = (
    ast.If,
    ast.IfExp,
    ast.For,
    ast.AsyncFor,
    ast.While,
)

LEGACY_TESTS = frozenset(
    {
        "test_abc::test_neighbour_changes_one_coordinate_and_stays_in_the_cube",
        "test_abc::test_fitness_and_probabilities_match_the_original_formulas",
        "test_abc::test_phase_order_and_evaluation_counts",
        "test_abc::test_scout_replaces_an_exhausted_source",
        "test_abc::test_same_seed_same_result_and_sphere_converges",
        "test_abc::test_direction_max_and_nan_handling",
        "test_abc::test_cache_skips_repeated_parameter_sets",
        "test_abc::test_budget_is_exact_even_inside_the_initial_population",
        "test_abc::test_protocol_errors",
        "test_space::test_float_linear_and_log_bounds",
        "test_space::test_int_gives_every_value_an_equal_share",
        "test_space::test_categorical_maps_to_choices",
        "test_space::test_to_dicts_returns_python_types",
        "test_space::test_invalid_parameters_are_rejected",
    }
)

REQUIRED_IDS = frozenset(
    {
        "R-SPACE-INT-INCLUSIVE",
        "R-FLOAT-RANGE",
        "R-CATEGORICAL-CHOICES",
        "R-LOG-DECODE",
        "R-DECODE-VECTORISED",
        "R-UNIT-CUBE-SAMPLE",
        "R-SPACE-DIM",
        "R-MOVE-CONSISTENT",
        "R-ROWS-TYPES",
        "R-INVALID-PARAMS",
        "R-NEIGHBOUR-CLIP",
        "R-NEIGHBOUR-ONE-COORD",
        "R-FITNESS",
        "R-PROB",
        "R-DIRECTION",
        "R-DIRECTION-INVALID",
        "R-NAN-WORST",
        "R-ASK-TELL-BEST",
        "R-CANDIDATE",
        "R-SEED",
        "R-ABC-PHASES",
        "R-ABC-GREEDY",
        "R-ONLOOKER-VISIBLE",
        "R-SCOUT-ONE-PER-CYCLE",
        "R-INIT-BATCH-PARTIAL",
        "R-PROTOCOL-ERRORS",
        "R-TRIAL-FAILED",
        "R-STUDY-BEST",
        "R-STUDY-BEST-NO-FINITE",
        "R-NCALLS",
        "R-CACHE",
        "R-BUDGET-EXACT",
        "R-FACADE-MAXEVALS",
        "R-FACADE-MISMATCH",
        "R-FACADE-DEFAULT-AND-CUSTOM-OPT",
        "R-FUNCTION-RECEIVES-DICT",
        "R-USER-EXC-PROPAGATES",
    }
)

REQUIREMENTS = {
    "R-SPACE-INT-INCLUSIVE": {
        "source": [
            "README.md | integers `low..high`, inclusive",
            "metahpo.space.Int | With `log=False` every integer gets an equal share of the unit interval;",
        ],
        "tests": [
            "test_space::test_int_gives_every_value_an_equal_share",
            "test_space_spec::test_int_should_decode_every_unit_value_to_low_when_low_equals_high",
            "test_space_spec::test_int_log_should_decode_to_integers_inside_the_inclusive_bounds",
        ],
    },
    "R-FLOAT-RANGE": {
        "source": ["README.md | real numbers in `[low, high]`"],
        "tests": ["test_space::test_float_linear_and_log_bounds"],
    },
    "R-CATEGORICAL-CHOICES": {
        "source": [
            "README.md | one of `choices`",
            "metahpo.space.Categorical | One of `choices`; every choice gets an equal share of the unit "
            "interval.",
        ],
        "tests": [
            "test_space::test_categorical_maps_to_choices",
            "test_space_spec::test_categorical_should_give_every_choice_an_equal_share",
        ],
    },
    "R-LOG-DECODE": {
        "source": [
            "README.md | `Float(1e-4, 1e-2, log=True)`",
            "metahpo.space | one numpy expression per kind (linear float, log float, integer, log integer, "
            "categorical)",
        ],
        "tests": [
            "test_space::test_float_linear_and_log_bounds",
            "test_space_spec::test_int_log_should_decode_to_integers_inside_the_inclusive_bounds",
            "test_space_spec::test_space_decode_should_match_the_decode_of_each_parameter",
        ],
    },
    "R-DECODE-VECTORISED": {
        "source": [
            "CHANGELOG.md | `Space` with `Float`, `Int` and `Categorical` parameters, decoded from the unit "
            "cube in vectorised numpy.",
            "metahpo.space.Space.decode | Unit-cube points (n, d) to raw values (n, d): integers as whole "
            "numbers, categories as indices.",
        ],
        "tests": ["test_space_spec::test_space_decode_should_match_the_decode_of_each_parameter"],
    },
    "R-UNIT-CUBE-SAMPLE": {
        "source": [
            "README.md | Optimizers search the unit cube `[0, 1]^d`; the space maps each point to typed "
            "parameters.",
            "metahpo.space.Space.sample | n uniform points of the unit cube, shape (n, d).",
            "derived: computed spec, sample(rng, n) -> (n, d) in [0,1).",
        ],
        "tests": ["test_space_spec::test_space_sample_should_return_n_points_of_the_half_open_unit_cube"],
    },
    "R-SPACE-DIM": {
        "source": [
            "metahpo.space.Space | An ordered set of named parameters; dimension d = number of parameters."
        ],
        "tests": ["test_space_spec::test_space_dim_should_equal_number_of_parameters"],
    },
    "R-MOVE-CONSISTENT": {
        "source": [
            "metahpo.space.Space.move | (key, params) of a point differing from a decoded point only in "
            "coordinate j: O(1) decoding.",
            "metahpo.optimizers.abc | so only that coordinate is decoded again; the rest of the decoded "
            "parameters are copied from the source.",
        ],
        "tests": [
            "test_abc::test_neighbour_changes_one_coordinate_and_stays_in_the_cube",
            "test_space_spec::test_space_move_should_match_the_full_decode_of_the_moved_point",
            "test_base_spec::test_candidate_should_carry_key_and_params_decoded_from_its_x",
        ],
    },
    "R-ROWS-TYPES": {
        "source": [
            "metahpo.space.Space.rows | params are the dictionaries the user's function receives, with "
            "Python types.",
            "metahpo.space.Space.rows | A key is the hashable raw row: equal keys mean the user's function "
            "sees equal parameters.",
        ],
        "tests": [
            "test_space::test_to_dicts_returns_python_types",
            "test_space_spec::test_space_rows_should_return_exact_python_types",
            "test_space_spec::test_space_rows_should_give_equal_keys_exactly_when_points_share_every_cell",
        ],
    },
    "R-INVALID-PARAMS": {
        "source": [
            "derived: computed spec, Invalid parameters raise ValueError/TypeError; the checks are in "
            "Float/Int/Categorical __post_init__ and Space.__init__, which have no docstring sentence.",
        ],
        "tests": [
            "test_space::test_invalid_parameters_are_rejected",
            "test_space_spec::test_parameter_should_raise_value_error_when_arguments_are_invalid",
        ],
    },
    "R-NEIGHBOUR-CLIP": {
        "source": [
            "metahpo.operators.neighbour_value | v_ij = x_ij + phi * (x_ij - x_kj), phi in [-1, 1], clipped "
            "to [0, 1] like the original ABC.",
        ],
        "tests": ["test_abc::test_neighbour_changes_one_coordinate_and_stays_in_the_cube"],
    },
    "R-NEIGHBOUR-ONE-COORD": {
        "source": ["metahpo.optimizers.abc | A neighbour differs from its food source in one coordinate"],
        "tests": [
            "test_abc::test_neighbour_changes_one_coordinate_and_stays_in_the_cube",
            "test_abc_spec::test_abc_onlooker_should_build_on_a_source_improved_earlier_in_the_same_phase",
        ],
    },
    "R-FITNESS": {
        "source": [
            "metahpo.operators.fitness | Fitness of minimisation values: 1 / (1 + f) for f >= 0, 1 + |f| "
            "otherwise; +inf gives 0.",
        ],
        "tests": [
            "test_abc::test_fitness_and_probabilities_match_the_original_formulas",
            "test_abc_spec::test_abc_employed_tell_should_reset_trial_only_on_strictly_greater_fitness",
        ],
    },
    "R-PROB": {
        "source": [
            "metahpo.operators.onlooker_probabilities | p_i = 0.9 * fit_i / max(fit) + 0.1; all-zero "
            "fitness gives equal probabilities.",
            "metahpo.optimizers.abc | onlookers chosen by roulette sweep (p = 0.9 * fit / max + 0.1)",
        ],
        "tests": [
            "test_abc::test_fitness_and_probabilities_match_the_original_formulas",
            "test_abc_spec::test_abc_onlookers_should_follow_probabilities_of_the_employed_result",
        ],
    },
    "R-DIRECTION": {
        "source": ["README.md | meta-hpo searches for the parameters that maximise or minimise that score."],
        "tests": [
            "test_abc::test_direction_max_and_nan_handling",
            "test_base_spec::test_optimizer_should_report_best_in_user_direction_when_maximising",
        ],
    },
    "R-DIRECTION-INVALID": {
        "source": [
            "derived: computed spec, Optimizer(space, direction 'min'|'max', seed); Optimizer.__init__ "
            "raises ValueError and has no docstring sentence.",
        ],
        "tests": ["test_base_spec::test_optimizer_should_raise_value_error_when_direction_is_not_min_or_max"],
    },
    "R-NAN-WORST": {
        "source": [
            "README.md | A score of NaN or infinity counts as the worst possible value for the chosen "
            "direction.",
            "CHANGELOG.md | NaN and infinite scores are treated as the worst value for the chosen direction.",
            "metahpo.optimizers.base.Optimizer | a NaN or infinite score becomes the worst possible value "
            "for the direction (-inf when maximising, +inf when minimising).",
        ],
        "tests": [
            "test_abc::test_direction_max_and_nan_handling",
            "test_base_spec::test_optimizer_should_treat_non_finite_score_as_worst",
            "test_study_spec::test_study_best_trial_should_skip_failed_trials_in_the_users_direction",
        ],
    },
    "R-ASK-TELL-BEST": {
        "source": [
            "metahpo.optimizers.base.Optimizer | Scores are given in the user's direction; internally "
            "everything is minimised",
            "derived: computed spec, best_params, best_score.",
        ],
        "tests": [
            "test_base_spec::test_optimizer_should_report_best_in_user_direction_when_maximising",
            "test_base_spec::test_optimizer_should_treat_non_finite_score_as_worst",
        ],
    },
    "R-CANDIDATE": {
        "source": [
            "metahpo.optimizers.base.Candidate | A point to evaluate: `x` in the unit cube, `params` as the "
            "user's function receives them, and `key`, the hashable decoded identity used for caching.",
        ],
        "tests": [
            "test_base_spec::test_candidate_should_carry_key_and_params_decoded_from_its_x",
            "test_base_spec::test_candidate_ids_should_be_unique_over_a_run",
        ],
    },
    "R-SEED": {
        "source": [
            "README.md | The same `seed` reproduces the same search.",
            "CONTRIBUTING.md | Tests are deterministic: pass a `seed` to anything random.",
            "derived: computed spec, RNG: numpy Generator, uniforms drawn in blocks of 4096.",
        ],
        "tests": [
            "test_abc::test_same_seed_same_result_and_sphere_converges",
            "test_base_spec::test_optimizer_should_repeat_candidates_past_a_uniform_block_when_seed_is_equal",
        ],
    },
    "R-ABC-PHASES": {
        "source": [
            "CHANGELOG.md | `ABCOptimizer`: Artificial Bee Colony behind an ask/tell interface, following "
            "the phase order of the original implementation.",
            "metahpo.optimizers.abc | employed bees one food source after another",
        ],
        "tests": [
            "test_abc::test_phase_order_and_evaluation_counts",
            "test_abc_spec::test_abc_cycle_should_increment_when_the_scout_tell_finishes",
            "test_abc_spec::test_abc_onlookers_should_follow_probabilities_of_the_employed_result",
        ],
    },
    "R-ABC-GREEDY": {
        "source": [
            "metahpo.optimizers.abc | The phases run in the same order and with the same rules as the "
            "original hyp_optimizer implementation",
            "derived: hyp_optimizer/ABC.py line 113 'if fitness_sol > self.fitness[i]:', else branch line "
            "121 'self.trial[i] = self.trial[i] + 1'.",
        ],
        "tests": [
            "test_abc_spec::test_abc_employed_tell_should_reset_trial_only_on_strictly_greater_fitness"
        ],
    },
    "R-ONLOOKER-VISIBLE": {
        "source": [
            "metahpo.optimizers.abc | with each improvement visible to the next onlooker",
            "derived: hyp_optimizer/ABC.py send_onlooker_bees writes self.foods[i] inside the loop, lines "
            "140-146.",
        ],
        "tests": [
            "test_abc_spec::test_abc_onlooker_should_build_on_a_source_improved_earlier_in_the_same_phase"
        ],
    },
    "R-SCOUT-ONE-PER-CYCLE": {
        "source": [
            "metahpo.optimizers.abc | and at most one scout per cycle.",
            "derived: hyp_optimizer/ABC.py send_scout_bees lines 151-153 'if np.amax(self.trial) >= "
            "self.conf.LIMIT: self.init(self.trial.argmax(axis=0))'.",
        ],
        "tests": [
            "test_abc::test_scout_replaces_an_exhausted_source",
            "test_abc_spec::test_abc_should_issue_exactly_one_scout_per_cycle_when_limit_is_one",
            "test_abc_spec::test_abc_should_send_a_scout_only_when_max_trial_reaches_limit",
            "test_abc_spec::test_abc_scout_should_reset_the_trial_of_the_first_most_exhausted_source",
            "test_abc_spec::test_abc_scout_should_replace_its_source_even_when_the_scout_score_is_worse",
            "test_abc_spec::test_abc_cycle_should_increment_when_the_scout_tell_finishes",
        ],
    },
    "R-INIT-BATCH-PARTIAL": {
        "source": [
            "metahpo.optimizers.abc | Only the initial population is handed out as a batch, because its "
            "evaluations are independent.",
            "derived: computed spec, init (colony_size//2 sources asked as one batch, may be told "
            "partially).",
        ],
        "tests": [
            "test_abc_spec::test_abc_initial_batch_should_hold_colony_size_floor_half_candidates",
            "test_abc_spec::test_abc_ask_should_raise_runtime_error_while_initial_candidates_are_pending",
            "test_abc_spec::test_abc_should_enter_employed_phase_when_initial_batch_is_told_in_two_parts",
        ],
    },
    "R-PROTOCOL-ERRORS": {
        "source": [
            "metahpo.optimizers.base.Optimizer | ask() returns the candidates that can be evaluated now; "
            "tell() reports their scores.",
            "derived: computed spec, asking again with a pending candidate -> RuntimeError; telling an "
            "unknown/already-told candidate -> ValueError; mismatched lengths -> ValueError; invalid "
            "colony_size/limit -> ValueError.",
        ],
        "tests": [
            "test_abc::test_protocol_errors",
            "test_abc_spec::test_abc_should_raise_value_error_when_limit_is_below_one",
            "test_abc_spec::test_abc_ask_should_raise_runtime_error_while_initial_candidates_are_pending",
            "test_abc_spec::test_abc_tell_should_raise_value_error_when_lengths_differ",
            "test_abc_spec::test_abc_tell_should_raise_value_error_when_candidate_was_never_asked_for",
            "test_base_spec::test_candidate_ids_should_be_unique_over_a_run",
        ],
    },
    "R-TRIAL-FAILED": {
        "source": [
            "metahpo.study.Trial.failed | True when the user's function returned NaN or an infinite value."
        ],
        "tests": [
            "test_abc::test_direction_max_and_nan_handling",
            "test_study_spec::test_trial_failed_should_be_true_exactly_for_non_finite_scores",
        ],
    },
    "R-STUDY-BEST": {
        "source": [
            "CHANGELOG.md | `optimize()` for one-call optimisation, returning a `Study` with every trial "
            "and the best one.",
            "metahpo.study | Results of an optimisation: every evaluated trial and the best one.",
        ],
        "tests": ["test_study_spec::test_study_best_trial_should_skip_failed_trials_in_the_users_direction"],
    },
    "R-STUDY-BEST-NO-FINITE": {
        "source": [
            "derived: computed spec, best_trial (ValueError if no finite trial); Study.best_trial has no "
            "docstring.",
        ],
        "tests": ["test_study_spec::test_study_best_should_raise_value_error_when_no_trial_is_finite"],
    },
    "R-NCALLS": {
        "source": [
            "metahpo.study.Study.n_calls | Calls of the user's function; cached repeats are not counted."
        ],
        "tests": [
            "test_abc::test_cache_skips_repeated_parameter_sets",
            "test_study_spec::test_study_n_calls_should_not_count_cached_trials",
            "test_facade_spec::test_optimize_should_call_function_for_every_trial_when_cache_is_false",
            "test_facade_spec::test_optimize_should_reuse_the_first_score_for_cached_trials",
        ],
    },
    "R-CACHE": {
        "source": [
            "README.md | With `cache=True` (the default), a repeated parameter set reuses its earlier score "
            "instead of training again and does not count against `max_evals`.",
            "CHANGELOG.md | Caching of repeated parameter sets, so a configuration is never trained twice.",
            "metahpo.facade.optimize | with `cache=True` a repeated parameter set reuses the earlier score "
            "and does not use up the budget.",
        ],
        "tests": [
            "test_abc::test_cache_skips_repeated_parameter_sets",
            "test_facade_spec::test_optimize_should_call_function_for_every_trial_when_cache_is_false",
            "test_facade_spec::test_optimize_should_reuse_the_first_score_for_cached_trials",
        ],
    },
    "R-BUDGET-EXACT": {
        "source": [
            "README.md | Its training budget is entirely yours; meta-hpo only counts evaluations.",
            "metahpo.facade.optimize | `max_evals` counts the calls of `function`",
            "derived: computed spec, exact max_evals calls (even inside the initial batch).",
        ],
        "tests": [
            "test_abc::test_budget_is_exact_even_inside_the_initial_population",
            "test_facade_spec::test_optimize_should_run_exactly_max_evals_trials_around_the_initial_batch",
        ],
    },
    "R-FACADE-MAXEVALS": {
        "source": ["derived: computed spec, ValueError for max_evals<1; optimize's docstring is silent."],
        "tests": [
            "test_facade_spec::test_optimize_should_raise_value_error_when_max_evals_is_below_one",
        ],
    },
    "R-FACADE-MISMATCH": {
        "source": ["derived: computed spec, ValueError for mismatched optimizer space/direction."],
        "tests": ["test_facade_spec::test_optimize_should_raise_value_error_when_optimizer_setup_differs"],
    },
    "R-FACADE-DEFAULT-AND-CUSTOM-OPT": {
        "source": [
            "metahpo.facade.optimize | The default optimizer is ABCOptimizer.",
            "derived: computed spec, optimize(function, space, direction='min', max_evals=500, "
            "optimizer=None, seed=None, cache=True).",
        ],
        "tests": [
            "test_facade_spec::test_optimize_should_use_an_abc_optimizer_by_default",
            "test_facade_spec::test_optimize_should_ask_and_tell_the_optimizer_it_is_given",
        ],
    },
    "R-FUNCTION-RECEIVES-DICT": {
        "source": [
            "metahpo.facade.optimize | `function` receives a dict of parameter values and returns a score"
        ],
        "tests": ["test_facade_spec::test_optimize_should_pass_a_dict_keyed_by_the_space_names"],
    },
    "R-USER-EXC-PROPAGATES": {
        "source": [
            "derived: computed spec, user exceptions propagate; README.md and optimize's docstring are "
            "silent."
        ],
        "tests": ["test_facade_spec::test_optimize_should_propagate_a_user_exception_unwrapped"],
    },
}


def _normalise(text):
    """Collapses every run of whitespace to one space, so wrapped lines and CRLF endings compare equal."""
    return " ".join(text.split())


def _doc_file(name):
    """A documentation file next to tests/, or of the repository that holds the imported metahpo package."""
    candidates = [TESTS_DIR.parent / name, Path(metahpo.__file__).resolve().parents[2] / name]
    found = [p for p in candidates if p.is_file()]
    return found[0] if found else candidates[0]


def _docstring_of(dotted):
    """The docstring of a module or of an attribute path below the longest importable module prefix."""
    parts = dotted.split(".")
    module_parts = [i for i in range(len(parts), 0, -1) if _importable(".".join(parts[:i]))]
    obj = sys.modules[".".join(parts[: module_parts[0]])]
    for name in parts[module_parts[0] :]:
        obj = getattr(obj, name)
    return obj.__doc__ or ""


def _importable(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def _source_text(location):
    if location in DOC_FILES:
        return _doc_file(location).read_text(encoding="utf-8")
    return _docstring_of(location)


def _requirements_keys(tree):
    """Keys of the REQUIREMENTS dict literal as written in the module, a key written twice kept twice."""
    literals = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and _dotted(node.targets[0]) == "REQUIREMENTS"
    ]
    return [key.value for key in literals[0].keys]


def _malformed_sources(sources):
    """Source items that are neither an exempt note nor 'LOCATION | quote' with a non-empty quote."""
    return [
        s
        for s in sources
        if not (s.startswith(EXEMPT_PREFIXES) or " | " in s) or not s.split(" | ", 1)[-1].strip()
    ]


def _malformed_references(tests):
    """Test references that are not '<module>::<function>' or '<module>::<Class>::<method>'."""
    return [t for t in tests if not TEST_REF_PATTERN.fullmatch(t)]


def _listed_pairs():
    """Sorted (requirement id, test reference) pairs of the whole matrix."""
    return sorted({(rid, t) for rid, entry in REQUIREMENTS.items() for t in entry["tests"]})


def _defined_tests(stem):
    """Test references defined by the sibling module <stem>.py; empty when that module does not exist."""
    path = TESTS_DIR / f"{stem}.py"
    if not path.is_file():
        return []
    return _test_names(_parse(path), stem)


def _unquoted_spec_items(spec_items, quoted_sources):
    """The spec item texts that no quoted source contains (whitespace normalised)."""
    quoted = _normalise(" ".join(quote for *_, quote in quoted_sources))
    texts = [text for kind_texts in spec_items.values() for text in kind_texts]
    return [text for text in texts if _normalise(text) not in quoted]


def _item_counts(spec_items):
    return {kind: len(texts) for kind, texts in spec_items.items()}


def _quoted_sources():
    """(id, index, location, quote) for every source item that quotes a file or docstring."""
    return [
        (rid, n, *item.split(" | ", 1))
        for rid, entry in REQUIREMENTS.items()
        for n, item in enumerate(entry["source"])
        if not item.startswith(EXEMPT_PREFIXES)
    ]


def _section(lines, heading):
    """Lines after the heading up to the next heading of the same or a higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    start = lines.index(heading) + 1
    ends = [
        i
        for i in range(start, len(lines))
        if lines[i].startswith("#") and len(lines[i]) - len(lines[i].lstrip("#")) <= level
    ]
    return lines[start : ends[0] if ends else len(lines)]


def _bullets(lines):
    """'- ' bullets with indented continuation lines joined by one space."""
    items = []
    for line in lines:
        if line.startswith("- "):
            items.append(line[2:].strip())
        elif line.startswith(" ") and line.strip() and items:
            items[-1] += " " + line.strip()
    return items


def _spec_items():
    """{kind: [texts]} of the README and CHANGELOG sentences that some requirement must quote."""
    readme = _doc_file("README.md").read_text(encoding="utf-8").splitlines()
    changelog = _doc_file("CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    rows = [line for line in _section(readme, "## Search space") if line.startswith("|")]
    return {
        "behaviour": _bullets(_section(readme, "## Behaviour")),
        "search_space": [row.split("|")[2].strip() for row in rows[2:]],
        "unit_cube": [line.strip() for line in readme if "Optimizers search the unit cube" in line],
        "changelog_added": _bullets(_section(changelog, "### Added")),
    }


def _test_files():
    return sorted(TESTS_DIR.glob("test_*.py"))


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _test_functions(tree, stem):
    """[(reference, node)] for top-level test functions and test methods of Test classes."""
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found.append((f"{stem}::{node.name}", node))
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            found += [
                (f"{stem}::{node.name}::{item.name}", item)
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test")
            ]
    return found


def _test_names(tree, stem):
    return sorted(ref for ref, _ in _test_functions(tree, stem))


def _all_test_functions(include_this_file):
    return [
        pair
        for path in _test_files()
        if include_this_file or path.name != THIS_FILE
        for pair in _test_functions(_parse(path), path.stem)
    ]


def _all_test_refs(include_this_file):
    return [ref for ref, _ in _all_test_functions(include_this_file)]


def _traced_test_refs():
    """References of the tests in the modules of TRACED_MODULES."""
    return [ref for ref in _all_test_refs(include_this_file=False) if ref.split("::", 1)[0] in TRACED_MODULES]


def _listed_tests():
    return {t for entry in REQUIREMENTS.values() for t in entry["tests"]}


def _orphans(collected, listed):
    return sorted(set(collected) - set(listed))


def _body_without_docstring(node):
    body = node.body
    has_doc = body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
    return body[1:] if has_doc and isinstance(body[0].value.value, str) else body


def _dotted(node):
    """'a.b.c' for a Name/Attribute chain, '' otherwise."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _is_raises_block(node):
    """A 'with pytest.raises(...)' statement."""
    return isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Call) and _dotted(item.context_expr.func) == "pytest.raises"
        for item in node.items
    )


def _is_assertion(node):
    """An assert statement, a 'with pytest.raises(...)' block or a call of an assert_* helper."""
    if isinstance(node, ast.Assert) or _is_raises_block(node):
        return True
    return isinstance(node, ast.Call) and _dotted(node.func).split(".")[-1].startswith("assert_")


def _comment_lines(source):
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [tok.start[0] for tok in tokens if tok.type == tokenize.COMMENT]


def _marker_lines(source):
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [
        tok.start[0]
        for tok in tokens
        if tok.type in (tokenize.COMMENT, tokenize.STRING) and any(m in tok.string for m in MARKERS)
    ]


def _imported_modules(tree):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
    return names


def _disallowed_imports(tree):
    allowed = ALLOWED_TOP_MODULES | set(sys.stdlib_module_names)
    return sorted(name for name in _imported_modules(tree) if name.split(".")[0] not in allowed)


def _flaky_imports(tree):
    return sorted(name for name in _imported_modules(tree) if name.split(".")[0] in FLAKY_MODULES)


def _unseeded_rng_lines(tree):
    """Lines of default_rng calls without a seed argument or with seed None."""
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _dotted(node.func).split(".")[-1] == "default_rng"
    ]
    seeds = [
        (call, call.args[0] if call.args else next((k.value for k in call.keywords), None)) for call in calls
    ]
    return [
        call.lineno
        for call, seed in seeds
        if seed is None or (isinstance(seed, ast.Constant) and seed.value is None)
    ]


def _global_random_lines(tree):
    """Lines of np.random.<name> / numpy.random.<name> calls other than the seedable Generator API.

    Calls such as np.random.rand, np.random.seed or np.random.normal use numpy's global state.
    """
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _dotted(node.func).rsplit(".", 1)[0] in ("np.random", "numpy.random")
        and _dotted(node.func).rsplit(".", 1)[-1] not in GENERATOR_API
    )


def _nodes_inside_raises_blocks(tree):
    """ids of every node below the body of a 'with pytest.raises(...)' block."""
    return {
        id(node)
        for block in ast.walk(tree)
        if _is_raises_block(block)
        for stmt in block.body
        for node in ast.walk(stmt)
    }


def _is_none(node):
    return isinstance(node, ast.Constant) and node.value is None


def _has_seed(call, position):
    """True when the call passes a seed that is not None by keyword, by position or through **kwargs."""
    seeds = [k.value for k in call.keywords if k.arg == "seed"] + call.args[position : position + 1]
    unpacked = any(k.arg is None for k in call.keywords)
    return unpacked or any(not _is_none(s) for s in seeds)


def _passes_optimizer(call):
    """True when an optimize call hands over its own optimizer, whose seed is then the one that counts."""
    by_keyword = any(k.arg == "optimizer" and not _is_none(k.value) for k in call.keywords)
    return by_keyword or len(call.args) > OPTIMIZER_POSITION


def _search_calls(tree):
    """(name, call) of every ABCOptimizer or optimize call outside a pytest.raises block."""
    inside = _nodes_inside_raises_blocks(tree)
    return [
        (_dotted(node.func).split(".")[-1], node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _dotted(node.func).split(".")[-1] in SEED_POSITION
        and id(node) not in inside
    ]


def _unseeded_search_lines(tree):
    """Lines of ABCOptimizer / optimize calls without a seed, outside pytest.raises.

    An optimize call that passes its own optimizer needs no seed of its own; a call inside pytest.raises fails
    before anything random happens.
    """
    return sorted(
        call.lineno
        for name, call in _search_calls(tree)
        if not _has_seed(call, SEED_POSITION[name]) and not (name == "optimize" and _passes_optimizer(call))
    )


def _control_flow_tests(tree, stem):
    """New tests (not in LEGACY_TESTS) whose body holds an if, a loop or a conditional expression;
    decorators such as parametrize tables are not part of the body."""
    return [
        ref
        for ref, node in _test_functions(tree, stem)
        if ref not in LEGACY_TESTS
        and any(isinstance(n, CONTROL_FLOW_NODES) for stmt in node.body for n in ast.walk(stmt))
    ]


def _placeholder_tests(tree):
    """Tests whose body, without the docstring, is empty or only pass, Ellipsis or constant expressions."""
    return [
        ref
        for ref, node in _test_functions(tree, "m")
        if all(
            isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
            for s in _body_without_docstring(node)
        )
    ]


def _tests_without_assertion(tree):
    return [
        ref for ref, node in _test_functions(tree, "m") if not any(_is_assertion(n) for n in ast.walk(node))
    ]


def _constant_assert_lines(tree):
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Constant)
    ]


def _is_unconditional_skip(decorator):
    name = _dotted(decorator.func if isinstance(decorator, ast.Call) else decorator)
    if name == "pytest.mark.skip":
        return True
    has_condition = isinstance(decorator, ast.Call) and (
        decorator.args or any(k.arg == "condition" for k in decorator.keywords)
    )
    return name == "pytest.mark.xfail" and not has_condition


def _unconditional_skip_tests(tree):
    """Tests with an unconditional skip or xfail decorator, or with a pytest.skip / pytest.xfail call."""
    return [
        ref
        for ref, node in _test_functions(tree, "m")
        if any(_is_unconditional_skip(d) for d in node.decorator_list)
        or any(
            isinstance(n, ast.Call) and _dotted(n.func) in ("pytest.skip", "pytest.xfail")
            for n in ast.walk(node)
        )
    ]


def _tests_without_docstring(tree, stem):
    return [ref for ref, node in _test_functions(tree, stem) if not ast.get_docstring(node)]


def _new_tests_without_docstring(tree, stem):
    """Tests without a docstring, except the original tests of test_abc.py and test_space.py."""
    return [ref for ref in _tests_without_docstring(tree, stem) if ref not in LEGACY_TESTS]


def _fingerprint(node):
    body = ast.dump(ast.Module(body=_body_without_docstring(node), type_ignores=[]))
    return body + "".join(ast.dump(d) for d in node.decorator_list)


def _duplicate_pairs(functions):
    """[(first, second)] of test references whose bodies and decorators are identical."""
    first_by_print = {}
    pairs = []
    for ref, node in functions:
        key = _fingerprint(node)
        if key in first_by_print:
            pairs.append((first_by_print[key], ref))
        first_by_print.setdefault(key, ref)
    return pairs


def _scan_sources(checker):
    """{file name: findings} of a source-text checker over every test module, for files with findings."""
    found = {path.name: checker(path.read_text(encoding="utf-8")) for path in _test_files()}
    return {name: lines for name, lines in found.items() if lines}


def _scan_trees(checker):
    """{file name: findings} of a syntax-tree checker over every test module, for files with findings."""
    found = {path.name: checker(_parse(path)) for path in _test_files()}
    return {name: items for name, items in found.items() if items}


def _scan_modules(checker):
    """{file name: findings} of a checker taking (tree, module stem) over every test module, for files
    with findings."""
    found = {path.name: checker(_parse(path), path.stem) for path in _test_files()}
    return {name: items for name, items in found.items() if items}


def _tree(snippet):
    return ast.parse(snippet)


def test_traceability_matrix_should_contain_every_required_id_exactly_once():
    """Given the REQUIREMENTS literal, when its keys are compared with the required ids and counted in the
    source, then no required id is missing and no key is written twice.

    Oracle: rule P-T8-18 and the requirement list of this work item. A dict literal silently keeps the last
    of two equal keys, so the keys are also counted in this module's syntax tree.
    """
    tree = _parse(Path(__file__))

    written_keys = _requirements_keys(tree)

    assert sorted(REQUIRED_IDS - set(REQUIREMENTS)) == []
    assert len(written_keys) == len(set(written_keys))


@pytest.mark.parametrize("req_id", sorted(REQUIREMENTS))
def test_traceability_entry_should_be_well_formed(req_id):
    """Given one requirement entry, when its fields are read, then the id matches R-[A-Z0-9-]+, every source
    is a non-empty quote or note, and every test reference is 'module::name' without duplicates.

    Oracle: rule P-T8-18 and the matrix format in this module's docstring.
    """
    entry = REQUIREMENTS[req_id]

    bad_sources = _malformed_sources(entry["source"])
    bad_references = _malformed_references(entry["tests"])

    assert ID_PATTERN.fullmatch(req_id), req_id
    assert sorted(entry) == ["source", "tests"]
    assert entry["source"] != []
    assert bad_sources == []
    assert bad_references == []
    assert len(entry["tests"]) == len(set(entry["tests"]))


@pytest.mark.parametrize(
    ("req_id", "index", "location", "quote"),
    [pytest.param(*item, id=f"{item[0]}:{item[1]}") for item in _quoted_sources()],
)
def test_traceability_source_should_still_contain_the_quoted_sentence(req_id, index, location, quote):
    """Given one quoted source of a requirement, when the file or docstring is read now, then it still
    contains the quote (whitespace normalised).

    Oracle: README.md, CHANGELOG.md, CONTRIBUTING.md and the metahpo docstrings as they are on disk. A failing
    case names the requirement whose source changed, which is how the matrix reports outdated requirements
    (rule P-T6-06, rule P-T8-18).
    """
    text = _source_text(location)

    found = _normalise(quote) in _normalise(text)

    assert found, f"{req_id} source {index}: '{quote}' not found in {location}"


@pytest.mark.parametrize("req_id", sorted(REQUIREMENTS))
def test_traceability_requirement_should_have_at_least_one_test(req_id):
    """Given one requirement, when its tests are read, then the list is not empty.

    Oracle: rule P-T8-19 ('Requirement 2 has no test case' must never happen).
    """
    tests = REQUIREMENTS[req_id]["tests"]

    assert tests != [], f"{req_id} icin test senaryosu yok"


@pytest.mark.parametrize(("req_id", "reference"), _listed_pairs())
def test_traceability_listed_test_should_exist(req_id, reference):
    """Given one test reference of a requirement, when the sibling module is parsed, then it defines that
    test function or method.

    Oracle: rule P-T8-19; a listed test that does not exist is no coverage.
    """
    stem = reference.split("::")[0]

    defined = _defined_tests(stem)

    assert reference in defined, f"{req_id}: {reference} does not exist"


def test_traceability_every_test_should_map_to_a_requirement():
    """Given every test of the modules in TRACED_MODULES, when they are compared with the matrix, then no
    test is missing from all requirements.

    Oracle: rule P-T8-20 ('what does test case 11 test?'). A parametrized function counts once. The other
    modules are traced by rule id in docs/test-izlenebilirlik.md (see the module docstring).
    """
    collected = _traced_test_refs()

    orphans = _orphans(collected, _listed_tests())

    assert orphans == [], "Hangi gereksinime ait? " + ", ".join(orphans)


def test_traceability_every_readme_and_changelog_item_should_be_quoted_by_a_requirement():
    """Given the README Behaviour bullets, Search space values, unit-cube sentence and CHANGELOG 'Added'
    bullets, when they are compared with all quoted sources, then each one appears in some source.

    Oracle: rule P-T2-32 (every sentence of the specification gets a requirement and hence a test).
    """
    items = _spec_items()

    missing = _unquoted_spec_items(items, _quoted_sources())

    assert missing == []


def test_traceability_spec_parser_should_find_every_item_of_the_documents():
    """Given README.md and CHANGELOG.md as written, when the spec items are extracted, then 3 Behaviour
    bullets, 3 Search space rows, 1 unit-cube sentence and 5 'Added' bullets are found.

    Oracle: README.md and CHANGELOG.md read by hand. Without this count a parser that finds nothing would
    let the quote check pass vacuously; a new bullet makes this test fail as a reminder to trace it.
    """
    items = _spec_items()

    counts = _item_counts(items)

    assert counts == {"behaviour": 3, "search_space": 3, "unit_cube": 1, "changelog_added": 5}


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param("def test_a():\n    assert 1 == 1\n", ["m::test_a"], id="function"),
        pytest.param(
            "class TestX:\n    def test_b(self):\n        assert 1 == 1\n", ["m::TestX::test_b"], id="method"
        ),
        pytest.param("def helper():\n    return 1\n", [], id="helper_ignored"),
        pytest.param("class Other:\n    def test_c(self):\n        pass\n", [], id="non_test_class_ignored"),
    ],
)
def test_traceability_collector_should_find_test_functions_and_methods(snippet, expected):
    """Given a small module, when its tests are collected, then top-level test functions and test methods of
    Test classes are found and nothing else.

    Oracle: pytest's default collection rules (test prefix, Test classes), which the orphan check relies on
    to report an unmapped test such as 'module::test_orphan'.
    """
    tree = _tree(snippet)

    names = _test_names(tree, "m")

    assert names == expected


@pytest.mark.parametrize(
    ("collected", "listed", "expected"),
    [
        pytest.param(
            ["m::test_a", "m::test_orphan"], ["m::test_a"], ["m::test_orphan"], id="orphan_reported"
        ),
        pytest.param(["m::test_a"], ["m::test_a", "n::test_b"], [], id="all_mapped"),
    ],
)
def test_traceability_orphan_check_should_report_only_unlisted_tests(collected, listed, expected):
    """Given collected and listed tests, when orphans are computed, then exactly the unlisted tests remain.

    Oracle: rule P-T8-20; postcondition: none, a pure set difference.
    """
    orphans = _orphans(collected, listed)

    assert orphans == expected


def test_suite_should_import_only_pytest_numpy_metahpo_and_the_standard_library():
    """Given every test module, when its imports are read, then each top-level module is pytest, numpy,
    metahpo or a standard-library module.

    Oracle: rule P-T2-06 and the project constraint 'pytest only, plus numpy and the Python standard
    library'; a test dependency outside them would make the suite unhealthy on a clean install. The suite's
    own tests/conftest.py, which test_suite_layout.py imports to call its hooks, is part of the suite and
    no dependency.
    """
    violations = _scan_trees(_disallowed_imports)

    assert violations == {}


def test_suite_should_not_import_a_clock_or_global_random_module():
    """Given every test module, when its imports are read, then none imports time, datetime or random.

    Oracle: CONTRIBUTING.md 'Tests are deterministic: pass a `seed` to anything random.' and rule P-T2-06;
    wall-clock values and the global random state make results change from run to run.
    """
    violations = _scan_trees(_flaky_imports)

    assert violations == {}


def test_suite_should_seed_every_default_rng_call():
    """Given every test module, when its default_rng calls are read, then each passes a seed that is not None.

    Oracle: CONTRIBUTING.md 'Tests are deterministic: pass a `seed` to anything random.'
    """
    violations = _scan_trees(_unseeded_rng_lines)

    assert violations == {}


def test_suite_should_make_no_global_numpy_random_call():
    """Given every test module, when its np.random calls are read, then each belongs to the seedable
    Generator API (default_rng, Generator, SeedSequence, bit generators).

    Oracle: CONTRIBUTING.md 'Tests are deterministic: pass a `seed` to anything random.' and the style
    checklist 'no seed=None or global numpy/random state in unit tests'; np.random.rand and its relatives
    draw from numpy's global state, which any other test can move.
    """
    violations = _scan_trees(_global_random_lines)

    assert violations == {}


def test_suite_should_seed_every_optimizer_and_optimize_call():
    """Given every test module, when its ABCOptimizer and optimize calls outside pytest.raises are read, then
    each passes a seed that is not None, or hands optimize an optimizer of its own.

    Oracle: CONTRIBUTING.md 'Tests are deterministic: pass a `seed` to anything random.' and README 'The
    same `seed` reproduces the same search.'; without a seed the search differs from run to run.
    """
    violations = _scan_trees(_unseeded_search_lines)

    assert violations == {}


def test_suite_new_test_bodies_should_hold_no_if_or_loop():
    """Given every test written after the original suite, when its body is read, then it holds no if, loop
    or conditional expression.

    Oracle: the style checklist 'No if/for/while in test bodies', which moves unavoidable sequences into
    straight-line underscore-prefixed helpers; a branch in a test body is untested logic. Comprehensions are
    expressions the checklist does not name, so they are allowed. The original tests of test_abc.py and
    test_space.py are exempt.
    """
    violations = _scan_modules(_control_flow_tests)

    assert violations == {}


def test_suite_should_hold_no_comment_lines():
    """Given every test module, when it is tokenised, then it holds no comment token.

    Oracle: CONTRIBUTING.md 'No `#` comment lines in Python, YAML or TOML files.'
    """
    violations = _scan_sources(_comment_lines)

    assert violations == {}


def test_suite_should_hold_no_placeholder_test():
    """Given every test module, when its tests are read, then no test body is only pass, Ellipsis or a
    constant.

    Oracle: rule P-T2-35; an empty test passes without checking anything.
    """
    violations = _scan_trees(_placeholder_tests)

    assert violations == {}


def test_suite_every_test_should_hold_an_assertion():
    """Given every test module, when its tests are read, then each holds an assert, a pytest.raises block or
    an assert_* call.

    Oracle: rule P-T2-35 and the style checklist, which allow numpy.testing assert_* matchers.
    """
    violations = _scan_trees(_tests_without_assertion)

    assert violations == {}


def test_suite_should_hold_no_assert_on_a_constant():
    """Given every test module, when its assert statements are read, then none asserts a bare constant.

    Oracle: rule P-T2-35; 'assert True' is a placeholder that always passes.
    """
    violations = _scan_trees(_constant_assert_lines)

    assert violations == {}


def test_suite_should_hold_no_unconditional_skip_or_xfail():
    """Given every test module, when its tests are read, then none is skipped or expected to fail without a
    condition.

    Oracle: rule P-T2-35 'all correctly written tests pass'; an unconditional skip hides a test from that
    requirement.
    """
    violations = _scan_trees(_unconditional_skip_tests)

    assert violations == {}


def test_suite_should_hold_no_unfinished_work_marker():
    """Given every test module, when its comments and strings are tokenised, then none holds an
    unfinished-work marker word.

    Oracle: rule P-T2-35 (every placeholder of the skeleton is replaced). The marker words are built in code
    so that this module does not contain them literally.
    """
    violations = _scan_sources(_marker_lines)

    assert violations == {}


def test_suite_every_new_test_should_have_a_docstring():
    """Given every test written after the original suite, when it is read, then it has a docstring.

    Oracle: rule P-T1-36; each new test names in its docstring the class or boundary it adds beyond the
    original tests of test_abc.py and test_space.py, which are exempt.
    """
    violations = _scan_modules(_new_tests_without_docstring)

    assert violations == {}


def test_suite_should_hold_no_two_tests_with_the_same_body():
    """Given every test of every module, when bodies without docstrings and decorators are compared, then no
    two tests are identical.

    Oracle: rule P-T6-06 ('many similar tests covering the same code'); a copied test adds no new behaviour.
    """
    functions = _all_test_functions(include_this_file=True)

    pairs = _duplicate_pairs(functions)

    assert pairs == []


@pytest.mark.parametrize(
    ("checker", "snippet", "expected"),
    [
        pytest.param(
            _disallowed_imports, "import pytest\nimport numpy as np\nimport json\n", [], id="imports_ok"
        ),
        pytest.param(
            _disallowed_imports,
            "import hypothesis\nfrom scipy import stats\n",
            ["hypothesis", "scipy"],
            id="imports_third_party",
        ),
        pytest.param(_disallowed_imports, "from unittest.mock import Mock\n", [], id="imports_mock_ok"),
        pytest.param(
            _flaky_imports,
            "import time\nfrom datetime import date\nimport random\n",
            ["datetime", "random", "time"],
            id="flaky_modules",
        ),
        pytest.param(_flaky_imports, "import math\n", [], id="flaky_none"),
        pytest.param(
            _unseeded_rng_lines, "import numpy as np\nnp.random.default_rng()\n", [2], id="rng_no_seed"
        ),
        pytest.param(_unseeded_rng_lines, "np.random.default_rng(None)\n", [1], id="rng_seed_none"),
        pytest.param(
            _unseeded_rng_lines,
            "np.random.default_rng(3)\nnp.random.default_rng(seed=4)\n",
            [],
            id="rng_seeded",
        ),
        pytest.param(_placeholder_tests, "def test_x():\n    pass\n", ["m::test_x"], id="placeholder_pass"),
        pytest.param(
            _placeholder_tests,
            "def test_y():\n    '''Doc.'''\n    ...\n",
            ["m::test_y"],
            id="placeholder_ellipsis",
        ),
        pytest.param(_placeholder_tests, "def test_z():\n    assert f() == 1\n", [], id="placeholder_none"),
        pytest.param(_tests_without_assertion, "def test_a():\n    f()\n", ["m::test_a"], id="no_assertion"),
        pytest.param(
            _tests_without_assertion,
            "def test_b():\n    with pytest.raises(ValueError):\n        f()\n",
            [],
            id="raises_counts",
        ),
        pytest.param(
            _tests_without_assertion,
            "def test_c():\n    np.testing.assert_array_equal(a, b)\n",
            [],
            id="numpy_assert_counts",
        ),
        pytest.param(_constant_assert_lines, "def test_d():\n    assert True\n", [2], id="assert_true"),
        pytest.param(
            _constant_assert_lines, "def test_e():\n    assert x == 1\n", [], id="assert_expression"
        ),
        pytest.param(
            _unconditional_skip_tests,
            "@pytest.mark.skip\ndef test_f():\n    assert x\n",
            ["m::test_f"],
            id="skip_bare",
        ),
        pytest.param(
            _unconditional_skip_tests,
            "@pytest.mark.xfail(reason='r')\ndef test_g():\n    assert x\n",
            ["m::test_g"],
            id="xfail_without_condition",
        ),
        pytest.param(
            _unconditional_skip_tests, "def test_h():\n    pytest.skip('r')\n", ["m::test_h"], id="skip_call"
        ),
        pytest.param(
            _unconditional_skip_tests,
            "@pytest.mark.skipif(c, reason='r')\ndef test_i():\n    assert x\n",
            [],
            id="skipif_ok",
        ),
        pytest.param(_global_random_lines, "import numpy as np\nnp.random.rand(3)\n", [2], id="global_rand"),
        pytest.param(_global_random_lines, "numpy.random.seed(0)\n", [1], id="global_seed"),
        pytest.param(
            _global_random_lines,
            "rng = np.random.default_rng(0)\nrng.random(3)\nnp.random.default_rng(1).normal()\n",
            [],
            id="generator_api_ok",
        ),
        pytest.param(_unseeded_search_lines, "ABCOptimizer(space)\n", [1], id="optimizer_no_seed"),
        pytest.param(
            _unseeded_search_lines, "ABCOptimizer(space, seed=None)\n", [1], id="optimizer_seed_none"
        ),
        pytest.param(_unseeded_search_lines, "optimize(f, space, max_evals=5)\n", [1], id="optimize_no_seed"),
        pytest.param(
            _unseeded_search_lines,
            "ABCOptimizer(space, seed=0)\nABCOptimizer(space, 'min', 10, 15, 0)\n"
            "optimize(f, space, seed=1)\noptimize(f, space, 'min', 9, None, 2)\n",
            [],
            id="seeded_by_keyword_or_position",
        ),
        pytest.param(
            _unseeded_search_lines,
            "optimize(f, space, max_evals=5, optimizer=opt)\n",
            [],
            id="optimize_with_own_optimizer",
        ),
        pytest.param(
            _unseeded_search_lines,
            "with pytest.raises(ValueError):\n    ABCOptimizer(space, colony_size=3)\n",
            [],
            id="inside_raises_ok",
        ),
    ],
)
def test_suite_tree_checker_should_report_exactly_the_violations_of_a_snippet(checker, snippet, expected):
    """Given a small module, when a syntax-tree checker reads it, then it reports exactly the violations
    built into the snippet and nothing for compliant code.

    Oracle: hand-written snippets; each case pairs a violation of rule P-T2-06 or P-T2-35 with a compliant
    neighbour, which shows that the suite-wide checks above can fail.
    """
    tree = _tree(snippet)

    found = checker(tree)

    assert found == expected


@pytest.mark.parametrize(
    ("checker", "snippet", "expected"),
    [
        pytest.param(_comment_lines, "x = 1\n" + chr(35) + " note\n", [2], id="comment_line"),
        pytest.param(_comment_lines, "x = '" + chr(35) + " not a comment'\n", [], id="hash_in_string"),
        pytest.param(_marker_lines, chr(35) + " " + MARKERS[0] + " later\n", [1], id="marker_in_comment"),
        pytest.param(
            _marker_lines, "def f():\n    '''" + MARKERS[1] + "'''\n", [2], id="marker_in_docstring"
        ),
        pytest.param(_marker_lines, "x = 'done'\n", [], id="no_marker"),
    ],
)
def test_suite_source_checker_should_report_exactly_the_violating_lines(checker, snippet, expected):
    """Given a small source text, when a token checker reads it, then it reports exactly the violating lines.

    Oracle: hand-written snippets for CONTRIBUTING.md 'No `#` comment lines' and rule P-T2-35 markers; the
    hash and marker characters are built in code so that this module stays free of them.
    """
    source = snippet

    found = checker(source)

    assert found == expected


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param(
            "def test_a():\n    '''A.'''\n    assert f() == 1\n\n"
            "def test_b():\n    '''B.'''\n    assert f() == 1\n",
            [("m::test_a", "m::test_b")],
            id="copy_with_other_docstring",
        ),
        pytest.param(
            "def test_a():\n    assert f() == 1\n\ndef test_b():\n    assert f() == 2\n",
            [],
            id="different_literal",
        ),
    ],
)
def test_suite_duplicate_check_should_report_tests_with_identical_bodies(snippet, expected):
    """Given two tests in a snippet, when their bodies are compared, then a renamed copy is reported and a
    test with another literal is not.

    Oracle: rule P-T6-06 (a copied test is redundant); the docstring does not count as a difference.
    """
    functions = _test_functions(_tree(snippet), "m")

    pairs = _duplicate_pairs(functions)

    assert pairs == expected


def test_suite_docstring_check_should_report_a_new_test_without_docstring():
    """Given one test with and one without a docstring, when the docstring check reads them, then only the
    second is reported.

    Oracle: rule P-T1-36; hand-written snippet.
    """
    tree = _tree("def test_a():\n    '''Doc.'''\n    assert x\n\ndef test_b():\n    assert x\n")

    missing = _tests_without_docstring(tree, "m")

    assert missing == ["m::test_b"]


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param("def test_a():\n    if x:\n        assert x\n", ["test_abc::test_a"], id="if_statement"),
        pytest.param(
            "def test_b():\n    assert (1 if x else 2) == 1\n",
            ["test_abc::test_b"],
            id="conditional_expression",
        ),
        pytest.param(
            "def test_c():\n    for v in x:\n        assert v\n", ["test_abc::test_c"], id="for_loop"
        ),
        pytest.param(
            "def test_d():\n    while x:\n        assert f()\n", ["test_abc::test_d"], id="while_loop"
        ),
        pytest.param(
            "def test_e():\n    assert [v for v in x] == []\n", [], id="comprehension_expression_ok"
        ),
        pytest.param(
            "@pytest.mark.parametrize('v', [i for i in range(3)])\ndef test_f(v):\n    assert f(v) == 1\n",
            [],
            id="comprehension_in_decorator_ok",
        ),
        pytest.param(
            "def test_protocol_errors():\n    for v in x:\n        assert v\n", [], id="original_test_exempt"
        ),
    ],
)
def test_suite_control_flow_check_should_report_new_tests_with_branches_or_loops(snippet, expected):
    """Given a small module read as test_abc, when the control-flow check reads it, then exactly the new
    tests with an if, a loop or a conditional expression in the body are reported.

    Oracle: hand-written snippets for the style checklist rule 'No if/for/while in test bodies'; a decorator
    is not body, a comprehension is an expression that the rule does not name, and
    test_abc::test_protocol_errors is one of the exempt original tests.
    """
    tree = _tree(snippet)

    found = _control_flow_tests(tree, "test_abc")

    assert found == expected


@pytest.mark.parametrize(
    ("checker", "items", "expected"),
    [
        pytest.param(
            _malformed_sources,
            ["README.md | a quote", "derived: a note", "", "README.md | ", "no separator"],
            ["", "README.md | ", "no separator"],
            id="sources",
        ),
        pytest.param(
            _malformed_references,
            ["test_a::test_x", "test_a::TestK::test_y", "test_a.test_x", "test_a::helper"],
            ["test_a.test_x", "test_a::helper"],
            id="references",
        ),
    ],
)
def test_traceability_entry_checkers_should_report_exactly_the_malformed_items(checker, items, expected):
    """Given well-formed and malformed source items or test references, when the entry checker reads them,
    then exactly the malformed ones are reported.

    Oracle: the matrix format in this module's docstring ('<LOCATION> | <verbatim quote>' or a 'derived:'
    note; '<module>::<function>' or '<module>::<Class>::<method>'); hand-written cases.
    """
    given = items

    found = checker(given)

    assert found == expected
