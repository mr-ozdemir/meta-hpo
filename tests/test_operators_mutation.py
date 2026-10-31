"""Standard-library mutation-testing harness for metahpo.operators.

Mutation operators, a Python-appropriate subset and not a full catalogue: AOR swaps an arithmetic BinOp
operator for each other member of {+, -, *, /}; ROR swaps a single comparison operator for each other member
of {<, <=, >, >=, ==, !=}; CONST swaps a numeric literal (type int or float, never bool or str) for every
member of {0.0, 1.0, 0.9, 0.1} that is not equal to it, so the int 0 in 'f >= 0' never becomes 0.0. Mutations
are applied only inside the bodies of neighbour_value, fitness, fitness_scalar and onlooker_probabilities,
after their docstrings; imports, module-level statements, annotations, docstrings and string keywords stay
untouched. Each mutant is first order: one operator applied at one location.

Locations: 12 AOR, 6 ROR and 17 CONST locations give 12 * 3 + 6 * 5 + 17 * 3 = 117 mutants. A location key
holds the end column too, because the three BinOps of '0.9 * fit / top + 0.1' start at the same column. Mutant
ids read '<function>-L<lineno>c<col>-<end_col>-<CLASS>-<orig>to<new>' with the real node.lineno of
operators.py; the shorter '<function>-L<lineno>-<CLASS>-<orig>to<new>' form is not unique here (line 16 holds
two 0.0 and two 1.0 literals).

Cost: mutants are generated once at import (pytest needs their ids at collection time) and each one is
compiled once per session through functools.cache. One full analysis executes about 18 checks x 117 mutants,
roughly 2100 small calls. Every use executes the cached code object into a fresh dict, so no mutable namespace
is shared between tests, and neither sys.modules nor operators.py on disk is changed.

Kill criterion: a mutant is killed when at least one oracle check raises or returns a value different from its
hand-computed literal. Oracles: the docstrings of metahpo.operators; the original hyp_optimizer ABC.py
(calculate_neighbour_solution clips to the bounds, calculate_fitness is 1 / (fun + 1) for fun >= 0 else 1 +
abs(fun), calculate_probabilities is 0.9 * fitness / max_fit + 0.1); Karaboga & Basturk (2007) for v_ij = x_ij
+ phi_ij (x_ij - x_kj) and the fitness formula. The 0.9 / 0.1 probability form comes from ABC.py and the
docstring, not from the paper (which uses fit_i / sum fit); the round() of the ABC.py neighbour formula is
deliberately dropped by the library and is not an oracle.

Equivalent mutants carry a written reason in EQUIVALENT; the seventh entry is equivalent only within the
documented domain (fitness values are >= 0). Mutation score = killed / (total - equivalent).

Unspecified, therefore never used as an expected value: NaN inputs of any operator, the sign of a zero result,
onlooker_probabilities for negative fitness and neighbour_value for phi outside [-1, 1]. NaN and -0.0 appear
only in the equivalence boundary test, where the original and the mutant are compared with each other.

Input space: Every mutant of operators.py.
Subset: The mutants of the harness operator table, each run against the operator tests.
"""

import ast
import copy
import functools
import math
import operator
import re
import sys
import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest

import metahpo.operators as operators

OPERATOR_FUNCTIONS = ("neighbour_value", "fitness", "fitness_scalar", "onlooker_probabilities")
AOR_SYMBOLS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
ROR_SYMBOLS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
AOR_BY_SYMBOL = {symbol: node_type for node_type, symbol in AOR_SYMBOLS.items()}
ROR_BY_SYMBOL = {symbol: node_type for node_type, symbol in ROR_SYMBOLS.items()}
CONST_VALUES = (0.0, 1.0, 0.9, 0.1)
MUTATION_CLASSES = ("AOR", "ROR", "CONST")
EXPECTED_MUTANT_COUNT = 117
MUTANT_BUDGET = 150
CHECK_TOLERANCE = 1e-12
EXACT = 0.0
MIN_REASON_LENGTH = 20
ID_FORMAT = re.compile(
    r"(neighbour_value|fitness|fitness_scalar|onlooker_probabilities)-L\d+c\d+-\d+-(AOR|ROR|CONST)-\S+to\S+"
)
REPORT_PREFIX_WIDTH = len("original: ")

SOURCE_PATH = Path(operators.__file__)
SOURCE_BYTES = SOURCE_PATH.read_bytes()
SOURCE_LINES = SOURCE_BYTES.decode("utf-8").splitlines()
ORIGINAL_TREE = ast.parse(SOURCE_BYTES.decode("utf-8"))
ORIGINAL_FUNCTION_OBJECTS = operator.attrgetter(*OPERATOR_FUNCTIONS)(operators)


class Mutant(NamedTuple):
    """One first-order mutant: its id, location key, original token, replacement and mutated module tree."""

    id: str
    function: str
    lineno: int
    col: int
    end_col: int
    cls: str
    original: Any
    replacement: Any
    tree: ast.Module


class Check(NamedTuple):
    """One oracle check: a name, a call on an operators namespace and its hand-computed literal result."""

    name: str
    run: Callable[[dict], Any]
    expected: Any


class Equivalent(NamedTuple):
    """One documented equivalent mutant: id, written reason and boundary inputs where it could differ."""

    mutant_id: str
    reason: str
    boundary_inputs: tuple


def _set_binop(node, symbol):
    node.op = AOR_BY_SYMBOL[symbol]()


def _set_compare(node, symbol):
    node.ops = [ROR_BY_SYMBOL[symbol]()]


def _set_constant(node, value):
    node.value = value


class _Mutator(ast.NodeTransformer):
    """The single generator: collects mutation sites (no target) or applies one replacement at a target."""

    def __init__(self, function, target=None, replacement=None):
        self.function = function
        self.target = target
        self.replacement = replacement
        self.sites = []

    def _site(self, node, cls, original, choices, apply):
        key = (self.function, node.lineno, node.col_offset, node.end_col_offset, cls)
        if self.target is None:
            self.sites.extend((key, original, choice) for choice in choices)
        elif key == self.target:
            apply(node, self.replacement)
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) not in AOR_SYMBOLS:
            return node
        original = AOR_SYMBOLS[type(node.op)]
        choices = [symbol for symbol in AOR_SYMBOLS.values() if symbol != original]
        return self._site(node, "AOR", original, choices, _set_binop)

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1 or type(node.ops[0]) not in ROR_SYMBOLS:
            return node
        original = ROR_SYMBOLS[type(node.ops[0])]
        choices = [symbol for symbol in ROR_SYMBOLS.values() if symbol != original]
        return self._site(node, "ROR", original, choices, _set_compare)

    def visit_Constant(self, node):
        if type(node.value) not in (int, float):
            return node
        choices = [value for value in CONST_VALUES if value != node.value]
        return self._site(node, "CONST", node.value, choices, _set_constant)


def _operator_defs(tree):
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in OPERATOR_FUNCTIONS
    }


def _body_after_docstring(function_def):
    return function_def.body[1:] if ast.get_docstring(function_def) is not None else function_def.body


def _visit_body(function_def, mutator):
    for statement in _body_after_docstring(function_def):
        mutator.visit(statement)


def _token(value):
    return value if isinstance(value, str) else repr(value)


def _mutant_id(key, original, replacement):
    function, lineno, col, end_col, cls = key
    return f"{function}-L{lineno}c{col}-{end_col}-{cls}-{_token(original)}to{_token(replacement)}"


def _generate_mutants():
    mutants = []
    for name, function_def in _operator_defs(ORIGINAL_TREE).items():
        collector = _Mutator(name)
        _visit_body(function_def, collector)
        for key, original, replacement in collector.sites:
            tree = copy.deepcopy(ORIGINAL_TREE)
            _visit_body(_operator_defs(tree)[name], _Mutator(name, key, replacement))
            mutant_id = _mutant_id(key, original, replacement)
            mutants.append(Mutant(mutant_id, *key, original, replacement, tree))
    return tuple(mutants)


MUTANTS = _generate_mutants()
MUTANT_IDS = tuple(mutant.id for mutant in MUTANTS)
MUTANTS_BY_ID = {mutant.id: mutant for mutant in MUTANTS}


@functools.cache
def _mutant_code(mutant_id):
    return compile(MUTANTS_BY_ID[mutant_id].tree, f"<mutant:{mutant_id}>", "exec")


def _mutant_namespace(mutant_id):
    namespace = {"__name__": "operators_mutant"}
    exec(_mutant_code(mutant_id), namespace)
    return namespace


def _original_namespace():
    return dict(vars(operators))


def _readonly(values):
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


CHECKS = (
    Check("nv_interior", lambda ns: ns["neighbour_value"](0.5, 0.25, 0.5), 0.625),
    Check("nv_clip_low", lambda ns: ns["neighbour_value"](0.25, 0.75, 1.0), 0.0),
    Check("nv_clip_high", lambda ns: ns["neighbour_value"](0.75, 0.25, 1.0), 1.0),
    Check("nv_just_above_0", lambda ns: ns["neighbour_value"](0.0625, 0.0625, 0.5), 0.0625),
    Check("nv_just_below_1", lambda ns: ns["neighbour_value"](0.75, 0.5, 0.75), 0.9375),
    Check("fs_zero", lambda ns: ns["fitness_scalar"](0.0), 1.0),
    Check("fs_inf", lambda ns: ns["fitness_scalar"](math.inf), 0.0),
    Check("fs_pos", lambda ns: ns["fitness_scalar"](1.0), 0.5),
    Check("fs_neg", lambda ns: ns["fitness_scalar"](-2.0), 3.0),
    Check("fs_small", lambda ns: ns["fitness_scalar"](0.05), 0.9523809523809523),
    Check("f_zero", lambda ns: ns["fitness"](np.array([0.0])), [1.0]),
    Check("f_inf", lambda ns: ns["fitness"](np.array([math.inf])), [0.0]),
    Check("f_pos", lambda ns: ns["fitness"](np.array([1.0])), [0.5]),
    Check("f_neg", lambda ns: ns["fitness"](np.array([-2.0])), [3.0]),
    Check("f_small", lambda ns: ns["fitness"](np.array([0.05])), [0.9523809523809523]),
    Check("op_zero", lambda ns: ns["onlooker_probabilities"](np.array([0.0, 0.0])), [1.0, 1.0]),
    Check("op_unit", lambda ns: ns["onlooker_probabilities"](np.array([1.0, 0.5])), [1.0, 0.55]),
    Check("op_small", lambda ns: ns["onlooker_probabilities"](np.array([0.01, 0.005])), [1.0, 0.55]),
)
"""Derivation order: each check was added for a mutant that all earlier checks left alive.

Hand calculations: 0.5 + 0.5 * 0.25 = 0.625; 0.25 + 1.0 * (-0.5) = -0.25 -> 0.0; 0.75 + 1.0 * 0.5 = 1.25 ->
1.0; v = 0.0625 lies in (0, 0.1) and kills 'v < 0.1'; 0.75 + 0.75 * 0.25 = 0.9375 lies in (0.9, 1) and kills
'v > 0.9'; 1 / (1 + 0) = 1.0; +inf gives 0; 1 / (1 + 1) = 0.5; 1 + |-2| = 3.0; 1 / 1.05 = 0.9523809523809523
with f in (0, 0.1) kills 'f >= 0.1'; all-zero fitness gives ones; 0.9 * 0.5 / 1 + 0.1 = 0.55; 0.9 * 0.005 /
0.01 + 0.1 = 0.55 with max in (0, 0.1] kills 'top <= 0.1' and '/' -> '*' (hidden when max == 1). The tolerance
1e-12 is needed because 0.9 * 0.01 / 0.01 + 0.1 is 1.0000000000000002, and it is far below the 0.1-sized
constant changes it must detect.
"""

EQUIVALENT = (
    Equivalent(
        "neighbour_value-L16c18-25-ROR-<to<=",
        "At v == 0.0 the mutant returns the literal 0.0 and the original returns v, which is 0.0; "
        "every other v takes the same branch in both, and NaN fails both comparisons. Only the sign "
        "of a -0.0 result could differ, which is unspecified and outside the [0, 1) coordinates the "
        "optimizer produces.",
        ((0.25, 0.5, 1.0), (0.0, 0.0, 0.0), (math.nan, 0.5, 0.5)),
    ),
    Equivalent(
        "neighbour_value-L16c38-45-ROR->to>=",
        "At v == 1.0 the mutant returns the literal 1.0 and the original returns v, which is 1.0; "
        "every other v takes the same branch in both.",
        ((0.75, 0.5, 1.0), (1.0, 1.0, 0.0)),
    ),
    Equivalent(
        "fitness-L23c24-30-ROR->=to>",
        "Only f == 0 (and -0.0) changes branch, and there 1 / (1 + max(f, 0)) == 1 + |f| == 1.0; "
        "NaN takes the 1 + |f| branch in both.",
        ((_readonly([0.0, -0.0, math.nan, math.inf]),),),
    ),
    Equivalent(
        "fitness_scalar-L28c7-13-ROR->=to>",
        "Only f == 0 (and -0.0) changes branch, and there 1.0 / (1.0 + 0) == 1.0 + abs(0) == 1.0; "
        "NaN goes to 1.0 + abs(nan) in both.",
        ((0.0,), (-0.0,), (0,), (math.nan,), (math.inf,)),
    ),
    Equivalent(
        "fitness_scalar-L29c22-35-ROR-==to>=",
        "No float is greater than +inf, so f >= inf holds exactly when f == inf and the branch "
        "choice is unchanged.",
        ((math.inf,), (1e308,), (0.0,)),
    ),
    Equivalent(
        "fitness_scalar-L29c22-35-ROR-==to>",
        "f > inf is never true, so +inf falls through to 1.0 / (1.0 + inf), which is 0.0 like the "
        "literal 0.0 of the original branch; finite f take the same branch in both.",
        ((math.inf,), (1e308,), (0.0,)),
    ),
    Equivalent(
        "onlooker_probabilities-L36c7-17-ROR-<=to==",
        "Domain-limited: fitness() and fitness_scalar() only return values >= 0, so a negative max "
        "fitness is outside the documented domain; for max >= 0 or NaN, 'top <= 0.0' and "
        "'top == 0.0' choose the same branch.",
        ((_readonly([0.0, 0.0]),), (_readonly([0.01, 0.005]),), (_readonly([math.nan, 0.5]),)),
    ),
)
EQUIVALENT_IDS = frozenset(entry.mutant_id for entry in EQUIVALENT)

LOCATION_COUNTS = [
    pytest.param("neighbour_value", 3, 2, 4, id="neighbour_value-3-binops-2-compares-4-literals"),
    pytest.param("fitness", 3, 1, 5, id="fitness-3-binops-1-compare-5-literals"),
    pytest.param("fitness_scalar", 3, 2, 5, id="fitness_scalar-3-binops-2-compares-5-literals"),
    pytest.param(
        "onlooker_probabilities", 3, 1, 3, id="onlooker_probabilities-3-binops-1-compare-3-literals"
    ),
]


def _matches(actual, expected, tolerance):
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        return False
    with np.errstate(invalid="ignore"):
        close = np.abs(actual_array - expected_array) <= tolerance
    same = actual_array == expected_array
    both_nan = np.isnan(actual_array) & np.isnan(expected_array)
    return bool(np.all(close | same | both_nan))


def _check_fails(namespace, check):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return not _matches(check.run(namespace), check.expected, CHECK_TOLERANCE)
        except Exception:
            return True


def _failed_checks(namespace, checks=CHECKS):
    return [check.name for check in checks if _check_fails(namespace, check)]


def _is_killed(namespace, checks=CHECKS):
    return bool(_failed_checks(namespace, checks))


def _killed_by_check(checks):
    killed = {check.name: set() for check in checks}
    for mutant_id in MUTANT_IDS:
        for name in _failed_checks(_mutant_namespace(mutant_id), checks):
            killed[name].add(mutant_id)
    return killed


def _killed_ids(checks):
    return {mutant_id for mutant_id in MUTANT_IDS if _is_killed(_mutant_namespace(mutant_id), checks)}


def _survivor_ids(checks):
    return set(MUTANT_IDS) - _killed_ids(checks)


def _redundant_checks(checks):
    killed_by = _killed_by_check(checks)
    dead = set()
    redundant = []
    for check in checks:
        if not killed_by[check.name] - dead:
            redundant.append(check.name)
        dead |= killed_by[check.name]
    return redundant


def _checks_without(name):
    return tuple(check for check in CHECKS if check.name != name)


def _structural_nodes(tree):
    return [
        node
        for node in ast.walk(tree)
        if not isinstance(node, (ast.operator, ast.cmpop, ast.expr_context, ast.boolop, ast.unaryop))
    ]


def _signature(node):
    if isinstance(node, ast.BinOp):
        return ("BinOp", type(node.op))
    if isinstance(node, ast.Compare):
        return ("Compare", tuple(type(op) for op in node.ops))
    if isinstance(node, ast.Constant):
        return ("Constant", type(node.value), node.value)
    return (type(node).__name__,)


def _node_diff(mutant):
    original_nodes = _structural_nodes(ORIGINAL_TREE)
    mutant_nodes = _structural_nodes(mutant.tree)
    pairs = [
        (a, b) for a, b in zip(original_nodes, mutant_nodes, strict=False) if _signature(a) != _signature(b)
    ]
    return len(original_nodes), len(mutant_nodes), pairs


def _line_diff(mutant):
    original_lines = ast.unparse(ORIGINAL_TREE).splitlines()
    mutant_lines = ast.unparse(mutant.tree).splitlines()
    pairs = [(a, b) for a, b in zip(original_lines, mutant_lines, strict=False) if a != b]
    return len(original_lines), len(mutant_lines), pairs


def _enclosing_functions():
    return {
        id(node): name
        for name, function_def in _operator_defs(ORIGINAL_TREE).items()
        for statement in _body_after_docstring(function_def)
        for node in ast.walk(statement)
    }


def _category(original_node, mutant_node):
    if isinstance(original_node, ast.BinOp) and isinstance(mutant_node, ast.BinOp):
        kinds = {type(original_node.op), type(mutant_node.op)}
        return "AOR" if kinds <= set(AOR_SYMBOLS) and len(kinds) == 2 else "other"
    if isinstance(original_node, ast.Compare) and isinstance(mutant_node, ast.Compare):
        kinds = {type(op) for op in original_node.ops + mutant_node.ops}
        single = len(original_node.ops) == len(mutant_node.ops) == 1
        return "ROR" if single and kinds <= set(ROR_SYMBOLS) and len(kinds) == 2 else "other"
    if isinstance(original_node, ast.Constant) and isinstance(mutant_node, ast.Constant):
        numeric = type(original_node.value) in (int, float) and type(mutant_node.value) in (int, float)
        return "CONST" if numeric else "other"
    return "other"


def _changed_mutation(mutant):
    _, _, pairs = _node_diff(mutant)
    original_node, mutant_node = pairs[0]
    return _enclosing_functions().get(id(original_node)), _category(original_node, mutant_node)


def _location_key(mutant):
    return (mutant.function, mutant.lineno, mutant.col, mutant.end_col, mutant.cls)


def _replacements_by_location():
    grouped = {}
    for mutant in MUTANTS:
        original, replacements = grouped.setdefault(_location_key(mutant), (mutant.original, []))
        replacements.append(mutant.replacement)
    return {key: (original, sorted(map(repr, values))) for key, (original, values) in grouped.items()}


def _rule_replacements(cls, original):
    if cls == "AOR":
        return [symbol for symbol in AOR_SYMBOLS.values() if symbol != original]
    if cls == "ROR":
        return [symbol for symbol in ROR_SYMBOLS.values() if symbol != original]
    return [value for value in CONST_VALUES if value != original]


def _expected_replacements(locations):
    return {
        key: (original, sorted(map(repr, _rule_replacements(key[4], original))))
        for key, (original, _) in locations.items()
    }


def _location_class_counts(function=None):
    keys = {_location_key(mutant) for mutant in MUTANTS if function in (None, mutant.function)}
    return dict(Counter(key[4] for key in keys))


def _unparsed_mutant_sources():
    return [ast.unparse(mutant.tree) for mutant in MUTANTS]


def _missing_operators(namespace):
    return [name for name in OPERATOR_FUNCTIONS if not callable(namespace.get(name))]


def _load_every_mutant():
    return [_mutant_namespace(mutant_id) for mutant_id in MUTANT_IDS]


def _ids_not_matching_format(ids):
    return [mutant_id for mutant_id in ids if ID_FORMAT.fullmatch(mutant_id) is None]


def _first_difference(a, b):
    differing = (index for index, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)
    return next(differing, min(len(a), len(b)))


def _describe(mutant, checks):
    _, _, line_pairs = _line_diff(mutant)
    original_line, mutant_line = (line.strip() for line in line_pairs[0])
    failed = _failed_checks(_mutant_namespace(mutant.id), checks)
    names = ", ".join(check.name for check in checks)
    status = f"killed by {', '.join(failed)}" if failed else "survived every check"
    return "\n".join(
        [
            f"mutant {mutant.id} {status}; checks run: {names}",
            f"original: {original_line}",
            f"mutant:   {mutant_line}",
            " " * (REPORT_PREFIX_WIDTH + _first_difference(original_line, mutant_line)) + "^",
        ]
    )


def _boundary_mismatches(function, entry):
    original = getattr(operators, function)
    mutant = _mutant_namespace(entry.mutant_id)[function]
    mismatches = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for args in entry.boundary_inputs:
            expected, actual = original(*args), mutant(*args)
            if not _matches(actual, expected, EXACT):
                mismatches.append(f"IN={args!r} original={expected!r} mutant={actual!r}")
    return mismatches


@pytest.mark.parametrize("mutant_id", MUTANT_IDS)
def test_operators_mutant_should_compile_and_define_the_four_operators(mutant_id):
    """Given one generated mutant, when its tree is compiled and executed, then all four operators are
    callable.

    Oracle: rule P-T5-06 (a mutant is valid, compilable code) and the four function names of operators.py.
    Postcondition: the namespace is a fresh dict; sys.modules is checked in the isolation test.
    """

    namespace = _mutant_namespace(mutant_id)

    assert _missing_operators(namespace) == []


def test_operators_mutation_should_leave_the_real_module_and_source_file_untouched():
    """Given every mutant loaded, when the real module and file are inspected, then they are the originals.

    Oracle: rule P-T5-06/P-T5-17 harness contract (mutants live only in fresh dicts). Postcondition:
    sys.modules['metahpo.operators'] is the imported module, its four functions are the objects captured at
    import and operators.py has the bytes read at import.
    """

    namespaces = _load_every_mutant()

    assert len(namespaces) == EXPECTED_MUTANT_COUNT
    assert sys.modules["metahpo.operators"] is operators
    assert operator.attrgetter(*OPERATOR_FUNCTIONS)(operators) == ORIGINAL_FUNCTION_OBJECTS
    assert SOURCE_PATH.read_bytes() == SOURCE_BYTES


@pytest.mark.parametrize("mutant_id", MUTANT_IDS)
def test_operators_mutant_should_change_one_numeric_literal_or_operator_inside_an_operator_body(mutant_id):
    """Given one mutant, when its single changed node is located, then it is an AOR, ROR or CONST glitch in a
    body.

    Oracle: rule P-T5-06 (small change resembling a programming glitch). The changed original node must lie in
    the body of one of the four functions after the docstring (so docstrings, imports, annotations and the
    module-level FloatArray alias are excluded), and a str literal change classifies as 'other' and fails. The
    id is checked against the same function and class so reports cannot mislabel a mutant. Postcondition:
    none.
    """
    mutant = MUTANTS_BY_ID[mutant_id]

    function, category = _changed_mutation(mutant)

    assert function in OPERATOR_FUNCTIONS
    assert category in MUTATION_CLASSES
    assert (function, category) == (mutant.function, mutant.cls)


def test_operators_mutation_should_generate_operator_set_minus_original_at_every_location():
    """Given all locations, when replacements are grouped per location, then each gets the set minus its
    original.

    Oracle: rule P-T5-07: AOR yields the 3 other members of {+, -, *, /}, ROR the 5 other members of {<, <=,
    >, >=, ==, !=}, CONST every member of {0.0, 1.0, 0.9, 0.1} not equal to the original. Decisive locations:
    the int 0 at L23 and L28 (0 == 0.0, so 0.0 must not appear) and the three BinOps of L38 that share a start
    column. Postcondition: none.
    """

    generated = _replacements_by_location()

    assert generated == _expected_replacements(generated)


def test_operators_mutation_should_find_the_literal_number_of_locations_per_class():
    """Given operators.py, when locations are counted per class, then there are 12 AOR, 6 ROR and 17 CONST.

    Oracle: hand count over operators.py lines 15, 16, 23, 28-30, 36 and 38 (rule P-T5-07); a generator that
    skips nodes or merges same-column BinOps gives fewer. Postcondition: none.
    """

    counts = _location_class_counts()

    assert counts == {"AOR": 12, "ROR": 6, "CONST": 17}


@pytest.mark.parametrize(("function", "aor", "ror", "const"), LOCATION_COUNTS)
def test_operators_mutation_should_find_the_literal_locations_in_each_function(function, aor, ror, const):
    """Given one operator, when its locations are counted per class, then the hand-counted literals come out.

    Oracle: hand count over operators.py (rule P-T5-07), for example neighbour_value: '+', '*', '-' on line
    15, two comparisons and four literals on line 16. Postcondition: none.
    """

    counts = _location_class_counts(function)

    assert counts == {"AOR": aor, "ROR": ror, "CONST": const}


@pytest.mark.parametrize("mutant_id", MUTANT_IDS)
def test_operators_mutant_should_differ_from_the_original_in_exactly_one_node_and_one_line(mutant_id):
    """Given one mutant, when its tree is walked in parallel with the original, then exactly one node differs.

    Oracle: rule P-T5-08 (first-order mutant, no higher-order mutant). Operator and context nodes are folded
    into their BinOp/Compare so one swapped operator counts once; ast.unparse lines are compared by count only
    because unparse drops blank lines. Postcondition: none.
    """
    mutant = MUTANTS_BY_ID[mutant_id]

    original_nodes, mutant_nodes, node_pairs = _node_diff(mutant)
    original_lines, mutant_lines, line_pairs = _line_diff(mutant)

    assert (mutant_nodes, len(node_pairs)) == (original_nodes, 1)
    assert (mutant_lines, len(line_pairs)) == (original_lines, 1)


def test_operators_mutation_should_produce_distinct_sources_that_differ_from_the_original():
    """Given all mutants, when they are unparsed, then all 117 sources are distinct and none equals the
    original.

    Oracle: rule P-T5-08; this is the regression guard against merged locations (a key without the end column
    collapses the L38 BinOps and yields duplicate sources). Postcondition: none.
    """

    sources = _unparsed_mutant_sources()

    assert len(set(sources)) == EXPECTED_MUTANT_COUNT
    assert ast.unparse(ORIGINAL_TREE) not in sources


@pytest.mark.parametrize("check", CHECKS, ids=[check.name for check in CHECKS])
def test_operators_oracle_check_should_pass_on_the_original_module(check):
    """Given the unmutated operators, when one oracle check runs, then it matches its hand-computed literal.

    Oracle: the operator docstrings, ABC.py (calculate_neighbour_solution, calculate_fitness,
    calculate_probabilities) and Karaboga & Basturk (2007); the literals are worked out in the CHECKS
    docstring. This is the precondition of rule P-T5-10: the test set must pass on the original first. The
    check is called directly, with no exception guard, so a failure shows the actual and desired values and an
    exception keeps its traceback; the exception-to-kill mapping of _check_fails is for mutants only.
    Postcondition: none.
    """

    namespace = _original_namespace()

    actual = check.run(namespace)

    np.testing.assert_allclose(actual, check.expected, rtol=0.0, atol=CHECK_TOLERANCE, strict=True)


@pytest.mark.parametrize("mutant_id", MUTANT_IDS)
def test_operators_mutant_should_be_killed_unless_documented_equivalent(mutant_id):
    """Given one mutant, when every oracle check runs on it, then it is killed exactly when it is not
    equivalent.

    Oracle: rule P-T5-10 (killed = some check gives a different output or raises; an exception such as
    ZeroDivisionError from '1.0 / (1.0 * f)' at f = 0 counts as a kill) and the EQUIVALENT registry. On
    failure the message puts the original and mutated lines side by side with a caret under the change (rule
    P-T5-19). Postcondition: none.
    """
    expected_killed = mutant_id not in EQUIVALENT_IDS

    killed = _is_killed(_mutant_namespace(mutant_id))

    assert killed == expected_killed, _describe(MUTANTS_BY_ID[mutant_id], CHECKS)


def test_operators_mutation_survivors_should_equal_the_documented_equivalents():
    """Given the full check list rerun on the original and on all mutants, when survivors are collected, then
    they are exactly the seven documented equivalent mutants.

    Oracle: rule P-T5-13 final state (original passes, every non-equivalent mutant is killed) and the written
    reasons in EQUIVALENT. Postcondition: none.
    """
    original_failures = _failed_checks(_original_namespace())

    survivors = _survivor_ids(CHECKS)

    assert original_failures == []
    assert survivors == set(EQUIVALENT_IDS)


def test_operators_oracle_checks_should_each_kill_a_mutant_no_earlier_check_killed():
    """Given CHECKS in derivation order, when kills accumulate check by check, then every check adds a new
    kill.

    Oracle: rule P-T5-13 (never add a test for an already dead mutant). The order is the derivation order of
    CHECKS: each check was added for a mutant that the earlier checks left alive; a failure names the
    redundant check. Postcondition: none.
    """

    redundant = _redundant_checks(CHECKS)

    assert redundant == []


def test_operators_mutation_score_should_exclude_equivalents_from_the_denominator():
    """Given all mutants and the registry, when the score is computed, then killed / (total - equivalent) is
    1.0.

    Oracle: rule P-T5-14 (Mutation Score = killed / (total - equivalent)); 117 generated, 7 equivalent, so 110
    must be killed. killed / total is about 0.94 and lower, which shows the corrected denominator is the one
    used. Postcondition: none.
    """
    killed_ids = _killed_ids(CHECKS)
    total = len(MUTANTS)
    equivalent = len(EQUIVALENT_IDS)
    killed = len(killed_ids)

    score = killed / (total - equivalent)

    assert (total, equivalent, killed) == (117, 7, 110)
    assert killed_ids.isdisjoint(EQUIVALENT_IDS)
    assert score == 1.0
    assert killed / total < score


@pytest.mark.parametrize("entry", EQUIVALENT, ids=[entry.mutant_id for entry in EQUIVALENT])
def test_operators_equivalent_entry_should_carry_a_written_reason_and_boundary_inputs(entry):
    """Given one registry entry, when it is read, then it names a generated mutant, a reason and boundary
    inputs.

    Oracle: rule P-T5-16 (an equivalence claim is justified in writing, never decided by sampling); an id that
    no longer matches a generated mutant is a stale entry. A reason must be a sentence, not a placeholder such
    as 'equivalent', so it must be longer than MIN_REASON_LENGTH characters. Postcondition: none.
    """

    reason, inputs = entry.reason, entry.boundary_inputs

    assert entry.mutant_id in MUTANTS_BY_ID
    assert type(reason) is str
    assert len(reason.strip()) > MIN_REASON_LENGTH
    assert len(inputs) >= 1


def test_operators_equivalent_registry_should_list_each_mutant_once_and_mark_the_domain_limited_entry():
    """Given the registry, when its ids and the onlooker entry are read, then ids are unique and the claim is
    scoped.

    Oracle: rule P-T5-16; the onlooker '<=' -> '==' entry is equivalent only for fitness values >= 0, which
    the fitness docstring implies and onlooker_probabilities does not state, so its reason must say 'domain'.
    Postcondition: none.
    """
    domain_entry = EQUIVALENT[-1]

    ids = [entry.mutant_id for entry in EQUIVALENT]

    assert len(set(ids)) == len(ids) == 7
    assert domain_entry.mutant_id == "onlooker_probabilities-L36c7-17-ROR-<=to=="
    assert "domain" in domain_entry.reason.lower()


@pytest.mark.parametrize("entry", EQUIVALENT, ids=[entry.mutant_id for entry in EQUIVALENT])
def test_operators_equivalent_mutant_should_match_the_original_on_its_boundary_inputs(entry):
    """Given one equivalent mutant, when original and mutant run on its boundary inputs, then the outputs
    agree.

    Oracle: rule P-T5-16 (an equivalence claim is refuted by one killing input). Each input sits on the
    mutated relational boundary (v == 0, v == 1, f == 0, f == inf, max == 0), so the changed branch is
    executed; NaN and -0.0 compare as equal values here only, never as expected literals. The entry's reason
    is the failure message. Postcondition: none.
    """
    function = MUTANTS_BY_ID[entry.mutant_id].function

    mismatches = _boundary_mismatches(function, entry)

    assert mismatches == [], entry.reason


def test_operators_mutation_should_stay_within_the_mutant_budget():
    """Given the restricted operator sets, when mutants are counted, then 0 < 117 <= 150.

    Oracle: rule P-T5-17 (plan the cost: operators x locations x checks). Adding '%' or '**' to AOR or more
    literals to CONST would exceed the budget. Postcondition: none.
    """

    count = len(MUTANTS)

    assert 0 < count <= MUTANT_BUDGET
    assert count == EXPECTED_MUTANT_COUNT


def test_operators_mutant_should_be_compiled_once_per_session():
    """Given a mutant id, when its code is requested twice, then the same code object is returned.

    Oracle: rule P-T5-17 ('each mutant needs to be compiled' - once, not once per check or per case).
    Postcondition: the functools cache holds at most one entry per generated mutant.
    """
    mutant_id = MUTANT_IDS[0]

    first, second = _mutant_code(mutant_id), _mutant_code(mutant_id)

    assert first is second
    assert _mutant_code.cache_info().currsize <= EXPECTED_MUTANT_COUNT


def test_operators_mutation_ids_should_be_unique_and_name_function_line_columns_and_change():
    """Given the generated ids, when they are checked, then each is unique and names function, line and
    change.

    Oracle: rule P-T5-19 (report a mutant with its location and change). 'fitness_scalar-L28c7-13-ROR->=to>'
    uses the real file line 28 of 'f >= 0', columns 7-13. Postcondition: none.
    """

    malformed = _ids_not_matching_format(MUTANT_IDS)

    assert malformed == []
    assert len(set(MUTANT_IDS)) == EXPECTED_MUTANT_COUNT
    assert "fitness_scalar-L28c7-13-ROR->=to>" in MUTANT_IDS
    assert SOURCE_LINES[27][7:13] == "f >= 0"


def test_operators_mutant_report_should_show_original_and_mutated_lines_side_by_side_with_a_caret():
    """Given a non-equivalent mutant and the check list without its only killer, when it is described, then
    the report shows both lines, a caret under the changed token and the checks that ran.

    Oracle: rule P-T5-19. 'neighbour_value-L16c22-25-CONST-0.0to0.1' is killed only by nv_just_above_0 (v =
    0.0625 in (0, 0.1)), so removing that check makes it survive; the change is the last digit of '0.0',
    column 20 of the stripped line, which is column 30 after the 10-character prefix. Postcondition: none.
    """
    mutant = MUTANTS_BY_ID["neighbour_value-L16c22-25-CONST-0.0to0.1"]
    reduced = _checks_without("nv_just_above_0")
    assert _is_killed(_mutant_namespace(mutant.id), reduced) is False

    report = _describe(mutant, reduced).splitlines()

    assert report[0].startswith(
        "mutant neighbour_value-L16c22-25-CONST-0.0to0.1 survived every check; checks run:"
    )
    assert "nv_interior" in report[0] and "nv_just_above_0" not in report[0]
    assert report[1:] == [
        "original: return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v",
        "mutant:   return 0.0 if v < 0.1 else 1.0 if v > 1.0 else v",
        " " * 30 + "^",
    ]
