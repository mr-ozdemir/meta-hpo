"""Suite layout: the written test plan in the conftest docstring checked against the files on disk.

Static checks (outside the test triangle, P-T1-48): the tests read tests/conftest.py, tests/test_*.py,
src/ (its import statements parsed with ast) and pyproject.toml, and call the two conftest hooks directly.
No library behaviour is run apart from package discovery with pkgutil and reading metahpo.__version__.

Input space: the tests/ and src/ file trees and the plan text, which are unbounded; the hook calls take any
list of collected items and any environment.
Subset: the real tree and plan as they are on disk (one input each); for the checkers whose verdict the
layout depends on (unit name pattern, import rule, missing unit file) synthetic boundary names, snippets and
a tmp_path copy; for the collection hook lists of size 0, 1 and many with the variable unset and set to '1'.

Oracles: the rule texts of P-T1-10, P-T1-23, P-T1-32, P-T1-47, P-T2-16, P-T2-42, P-T3-20, P-T4-16, P-T5-25,
P-T6-02, P-T6-29, P-T6-30, P-T7-24, P-T7-41, P-T8-10, P-T8-11, P-T8-13 and P-T8-57, pyproject.toml,
.github/workflows/ci.yml and metahpo.__init__.

Unspecified, not asserted: values of METAHPO_TEST_REVERSE other than '1'; the exact HANG_TIMEOUT value
(only its origin, the 4000 ms template, and a margin are specified); the strategy for base (the plan
chooses code-based); whether the static files go under every module in the selection map (the plan lists
them as always selected); that the existing files stay byte-identical (a reviewer step, not pinned here);
which test files reach a module at run time without importing it (the selection is checked at module level
through import statements; a package __init__ that Python runs implicitly for a submodule import is not an
edge, otherwise every row would have to be the whole suite).
"""

import ast
import io
import pkgutil
import platform
import re
import sys
import tokenize
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path

import conftest
import numpy
import pytest

import metahpo

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
SRC = ROOT / "src"
CONFTEST = TESTS_DIR / "conftest.py"
LAYOUT = Path(__file__).resolve()

MODULES = frozenset({"space", "operators", "base", "abc", "study", "facade"})
FEATURES = [
    "search space",
    "ABC operators",
    "optimizer base protocol",
    "ABC optimizer",
    "study results",
    "optimize facade",
    "cross-cutting",
]
CROSS_CUTTING = frozenset(
    {
        "test_end_to_end.py",
        "test_determinism.py",
        "test_regression.py",
        "test_fuzz.py",
        "test_fuzz_grammar.py",
        "test_api_contract.py",
        "test_static.py",
        "test_suite_layout.py",
        "test_traceability.py",
    }
)
UNIT_NAME = re.compile(r"test_(space|operators|base|abc|study|facade)(_[a-z_]+)?\.py")
STATIC_LABEL = "static (outside the triangle, P-T1-48)"
LAYER_OF_LABEL = {"unit": "unit", "integration": "integration", "system": "system", STATIC_LABEL: "static"}
EXEMPT = frozenset({"test_space.py", "test_abc.py"})
INPUT_SPACE_MARKER = "Input space:"
SUBSET_MARKER = "Subset:"
REVERSE_ENV = "METAHPO_TEST_REVERSE"
ALLOWED_PACKAGES = frozenset({"metahpo", "numpy", "pytest", "conftest"})
ALLOWED_METAHPO_MODULES = frozenset({"metahpo", "metahpo.operators"})
NEW_AFTER_PYTHON_3_10 = frozenset({"tomllib"})
TEMPLATE_TIMEOUT_SECONDS = 4.0
EXISTING_SPACE_TESTS = frozenset(
    {
        "test_float_linear_and_log_bounds",
        "test_int_gives_every_value_an_equal_share",
        "test_categorical_maps_to_choices",
        "test_to_dicts_returns_python_types",
        "test_invalid_parameters_are_rejected",
    }
)
EXISTING_ABC_TESTS = frozenset(
    {
        "test_neighbour_changes_one_coordinate_and_stays_in_the_cube",
        "test_fitness_and_probabilities_match_the_original_formulas",
        "test_phase_order_and_evaluation_counts",
        "test_scout_replaces_an_exhausted_source",
        "test_same_seed_same_result_and_sphere_converges",
        "test_direction_max_and_nan_handling",
        "test_cache_skips_repeated_parameter_sets",
        "test_budget_is_exact_even_inside_the_initial_population",
        "test_protocol_errors",
    }
)
ROW = re.compile(r"^ {4}(test_\w+\.py)\s*\|\s*([^|]+?)\s*(?:\|.*)?$")
STRATEGY_ROW = re.compile(r"^ {4}(\w+) \| (.+?) \| (.+)$")
FSM_EDGE = re.compile(
    r"^ {4}(init|employed|onlooker|scout) -> (init|employed|onlooker|scout)(?: \(([^)]*)\))?: (.*)$"
)
SELECTION_ROW = re.compile(r"^ {4}(\w+): (.*)$")
GRAPH_ROW = re.compile(r"^ {4}(\w+) <- (.*)$")
INIT_KEY = "__init__"
BLIND_SPOT = re.compile(r"\((\d)\) ([^:]+):")
V_MODEL_PAIR = re.compile(
    r"(Requirements|Architecture|Detailed design) \(([^)]*)\) (?:is|are) checked by "
    r"(unit|integration|system) tests: "
    r"((?:test_\w+\.py(?:, )?)+)\."
)


def _read(path):
    return path.read_text(encoding="utf-8")


def _plan():
    return ast.get_docstring(ast.parse(_read(CONFTEST)))


def _flat(text):
    return " ".join(text.split())


def _sections():
    lines = _plan().splitlines()
    sections = {}
    title = None
    for i, line in enumerate(lines):
        is_rule = i + 1 < len(lines) and set(lines[i + 1]) == {"-"} and len(lines[i + 1]) == len(line)
        if line and is_rule:
            title = line
            sections[title] = []
        elif title is not None and set(line) != {"-"}:
            sections[title].append(line)
    return {name: "\n".join(body) for name, body in sections.items()}


def _section(title):
    return _sections()[title]


def _sentence_with(text, marker):
    sentences = re.split(r"(?<=\.)\s+", _flat(text))
    found = [sentence for sentence in sentences if marker in sentence]
    return found[0] if found else ""


def _plan_blocks():
    blocks = []
    for line in _section("Plans").splitlines():
        if line.startswith("Plan: "):
            blocks.append({"feature": line[len("Plan: ") :], "groups": "", "rows": [], "in_groups": False})
        elif blocks and line.startswith("Scenario groups:"):
            blocks[-1]["groups"] = line[len("Scenario groups:") :]
            blocks[-1]["in_groups"] = True
        elif blocks and ROW.match(line):
            blocks[-1]["rows"].append(ROW.match(line).groups())
            blocks[-1]["in_groups"] = False
        elif blocks and blocks[-1]["in_groups"] and line.strip():
            blocks[-1]["groups"] += " " + line.strip()
        elif blocks:
            blocks[-1]["in_groups"] = False
    for block in blocks:
        block["groups"] = [group.strip() for group in block["groups"].split(";") if group.strip()]
    return blocks


def _plan_rows():
    return [(block["feature"], name, label) for block in _plan_blocks() for name, label in block["rows"]]


def _plan_features():
    return [block["feature"] for block in _plan_blocks()]


def _lower_groups_of(feature):
    return {group.lower() for group in _plan_blocks()[FEATURES.index(feature)]["groups"]}


def _planned_files_of(feature):
    return {name for owner, name, _ in _plan_rows() if owner == feature}


def _layer_sizes(layers, names):
    return {name: len(layers.get(name, ())) for name in names}


def _v_model_layers(pairs):
    return {level: layer for level, (_, layer, _) in pairs.items()}


def _strategy_modules(rows):
    return sorted(row[0] for row in rows)


def _keyword_sources(call):
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords}


def _strategy_table(rows):
    return {module: (strategy, target) for module, strategy, target in rows}


def _known_hookspecs(pytestconfig, hooks):
    return {hook for hook in hooks if hasattr(pytestconfig.hook, hook)}


def _layer_by_file():
    return {name: LAYER_OF_LABEL.get(label) for _, name, label in _plan_rows()}


def _files_by_layer():
    layers = {}
    for name, layer in _layer_by_file().items():
        layers.setdefault(layer, set()).add(name)
    return layers


def _disk_test_files(directory=TESTS_DIR):
    return {path.name for path in directory.glob("test_*.py")}


def _files_listed_more_than_once():
    counts = Counter(name for _, name, _ in _plan_rows())
    return {name: count for name, count in counts.items() if count != 1}


def _plans_with_fewer_groups_than_files():
    return [
        (block["feature"], len(block["groups"]), len(block["rows"]))
        for block in _plan_blocks()
        if len(block["groups"]) < max(1, len(block["rows"]))
    ]


def _rows_with_unknown_label():
    return [(name, label) for _, name, label in _plan_rows() if label not in LAYER_OF_LABEL]


def _modules_without_unit_row(layers):
    return {module for module in MODULES if layers.get(f"test_{module}.py") != "unit"}


def _v_model_pairs():
    pairs = V_MODEL_PAIR.findall(_flat(_section("V-model")))
    return {level: (sources, layer, files.split(", ")) for level, sources, layer, files in pairs}


def _v_model_files_with_other_layer():
    layers = _layer_by_file()
    return [
        (name, layer, layers.get(name))
        for _, layer, files in _v_model_pairs().values()
        for name in files
        if layers.get(name) != layer
    ]


def _strategy_rows():
    rows = [STRATEGY_ROW.match(line) for line in _section("Strategy and coverage").splitlines()]
    return [match.groups() for match in rows if match and match.group(1) in MODULES]


def _fsm_edges():
    edges = [FSM_EDGE.match(line) for line in _section("Strategy and coverage").splitlines()]
    return {(m.group(1), m.group(2), m.group(3) or ""): m.group(4) for m in edges if m}


def _blind_spots():
    text = _flat(_section("Blind spots"))
    marks = list(BLIND_SPOT.finditer(text))
    ends = [mark.start() for mark in marks[1:]] + [len(text)]
    return {mark.group(2): text[mark.end() : end] for mark, end in zip(marks, ends, strict=True)}


def _scope_statuses(paragraph):
    statuses = set()
    if re.search(r"\bin scope\b", paragraph):
        statuses.add("in scope")
    if "out of scope" in paragraph:
        statuses.add("out of scope")
    return statuses


def _selection_map():
    rows = {}
    key = None
    for line in _section("Selection").splitlines():
        match = SELECTION_ROW.match(line)
        if match:
            key = match.group(1)
            rows[key] = match.group(2)
        elif key is not None and line.startswith(" " * 8):
            rows[key] += " " + line.strip()
        else:
            key = None
    return {name: [item.strip() for item in text.split(",") if item.strip()] for name, text in rows.items()}


def _always_selected():
    line = _sentence_with(_section("Selection"), "Always selected:")
    names = line.split("Always selected:", 1)[1].split("Statement-level", 1)[0]
    return [name.strip(" .") for name in names.split(",")]


def _selected(patterns, names):
    return {name for name in names for pattern in patterns if fnmatch(name, pattern)}


def _unselectable_files():
    names = set(_layer_by_file())
    selected = set(_always_selected())
    for patterns in _selection_map().values():
        selected |= _selected(patterns, names)
    return names - selected


def _patterns_matching_nothing():
    names = set(_layer_by_file())
    return [
        (key, pattern)
        for key, patterns in _selection_map().items()
        for pattern in patterns
        if not _selected([pattern], names)
    ]


def _modules_without_own_pattern(rows):
    return {module for module in MODULES if f"test_{module}*.py" not in rows.get(module, [])}


def _modules_without_unit_file(tests_dir, modules):
    return {module for module in modules if not (tests_dir / f"test_{module}.py").is_file()}


def _discovered_leaf_modules(package_path):
    found = pkgutil.walk_packages(package_path, "metahpo.")
    return {name.rsplit(".", 1)[-1] for _, name, is_package in found if not is_package}


def _src_modules(src=SRC):
    found = {}
    for path in sorted((src / "metahpo").rglob("*.py")):
        parts = list(path.relative_to(src).with_suffix("").parts)
        is_package = parts[-1] == INIT_KEY
        dotted = ".".join(parts[:-1] if is_package else parts)
        found[dotted] = (path, is_package)
    return found


def _module_key(dotted, modules):
    return INIT_KEY if modules[dotted][1] else dotted.rsplit(".", 1)[-1]


def _absolute_module(node, dotted, is_package):
    if node.level == 0:
        return node.module
    anchor = dotted.split(".") if is_package else dotted.split(".")[:-1]
    anchor = anchor[: len(anchor) - node.level + 1]
    return ".".join([*anchor, node.module] if node.module else anchor)


def _imported_modules(tree, dotted, is_package, modules):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names if alias.name in modules}
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_module(node, dotted, is_package)
            for alias in node.names:
                child = f"{base}.{alias.name}"
                found |= {child} if child in modules else ({base} & set(modules))
    return found


def _src_import_graph(src=SRC):
    modules = _src_modules(src)
    edges = set()
    for dotted, (path, is_package) in modules.items():
        tree = ast.parse(_read(path))
        for target in _imported_modules(tree, dotted, is_package, modules):
            edges.add((_module_key(dotted, modules), _module_key(target, modules)))
    return {(dependent, dependency) for dependent, dependency in edges if dependent != dependency}


def _plan_import_graph():
    edges = set()
    for line in _section("Selection").splitlines():
        match = GRAPH_ROW.match(line)
        if match:
            dependents = [name.strip() for name in match.group(2).split(",")]
            edges |= {(dependent, match.group(1)) for dependent in dependents}
    return edges


def _rows_missing_dependent_files(rows, edges, names):
    missing = []
    for dependent, dependency in sorted(edges):
        lacking = _selected(rows.get(dependent, []), names) - _selected(rows.get(dependency, []), names)
        if dependency not in rows or dependent not in rows or lacking:
            missing.append((dependency, dependent, sorted(lacking)))
    return missing


def _public_name_origin(name, package, modules):
    path, _ = modules[package]
    for node in ast.parse(_read(path)).body:
        if isinstance(node, ast.ImportFrom) and name in {alias.asname or alias.name for alias in node.names}:
            base = _absolute_module(node, package, True)
            child = f"{base}.{name}"
            if child in modules:
                return child
            if base not in modules:
                return None
            return _public_name_origin(name, base, modules) if modules[base][1] else base
    return None


def _package_aliases(tree):
    """Local names bound to the metahpo package: 'metahpo' and every alias of 'import metahpo as <alias>'."""
    aliases = {"metahpo"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {alias.asname for alias in node.names if alias.name == "metahpo" and alias.asname}
    return aliases


def _is_metahpo_attribute(node, aliases):
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases


def _imported_from_package(node):
    """Names of 'from metahpo import ...'; a star import binds every name of metahpo.__all__."""
    names = {alias.name for alias in node.names}
    return set(metahpo.__all__) if "*" in names else names


def _metahpo_names_used(tree):
    names = set()
    aliases = _package_aliases(tree)
    for node in ast.walk(tree):
        source = (node.module or "") if isinstance(node, ast.ImportFrom) and node.level == 0 else ""
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[1] for alias in node.names if alias.name.startswith("metahpo.")}
        elif source == "metahpo":
            names |= _imported_from_package(node)
        elif source.startswith("metahpo."):
            names.add(source.split(".")[1])
        elif _is_metahpo_attribute(node, aliases):
            names.add(node.attr)
    return names


def _metahpo_origin(name, modules):
    submodule = f"metahpo.{name}"
    return submodule if submodule in modules else _public_name_origin(name, "metahpo", modules)


def _modules_used_in(source, modules):
    found = set()
    for name in _metahpo_names_used(ast.parse(source)):
        origin = _metahpo_origin(name, modules)
        found |= {_module_key(origin, modules)} if origin else set()
    return found


def _modules_used_by(path, modules):
    return _modules_used_in(_read(path), modules)


def _files_not_selected_for_used_modules(rows, always, directory=TESTS_DIR):
    modules = _src_modules()
    names = set(_layer_by_file())
    missing = []
    for path in sorted(directory.glob("test_*.py")):
        for key in sorted(_modules_used_by(path, modules)):
            selected = _selected(rows.get(key, []), names) | set(always)
            if path.name in names and path.name not in selected:
                missing.append((path.name, key))
    return missing


def _misnamed_files(directory=TESTS_DIR):
    return sorted(
        name
        for name in _disk_test_files(directory)
        if not UNIT_NAME.fullmatch(name) and name not in CROSS_CUTTING
    )


def _import_violations(source):
    violations = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            violations.append(f"relative import from {node.module!r}")
        elif isinstance(node, ast.ImportFrom):
            violations.extend(_module_violations(node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                violations.extend(_module_violations(alias.name))
    return violations


def _module_violations(module):
    top = module.split(".", 1)[0]
    if top == "metahpo" and module not in ALLOWED_METAHPO_MODULES:
        return [f"metahpo import from {module!r}"]
    if top not in ALLOWED_PACKAGES and top not in sys.stdlib_module_names:
        return [f"import of {module!r}"]
    return []


def _suite_import_violations():
    found = {path.name: _import_violations(_read(path)) for path in sorted(TESTS_DIR.glob("*.py"))}
    return {name: violations for name, violations in found.items() if violations}


def _top_level_test_names(path):
    body = ast.parse(_read(path)).body
    return {node.name for node in body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def _fixture_keywords(decorator):
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    is_fixture = ast.unparse(target) == "pytest.fixture"
    keywords = decorator.keywords if isinstance(decorator, ast.Call) else []
    return is_fixture, {kw.arg: _constant_or_source(kw.value) for kw in keywords}


def _constant_or_source(node):
    return node.value if isinstance(node, ast.Constant) else ast.unparse(node)


def _fixtures(tree):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                is_fixture, keywords = _fixture_keywords(decorator)
                if is_fixture:
                    found.append((node, keywords))
    return found


def _wide_fixtures():
    return [
        (path.name, node.name, keywords["scope"])
        for path in sorted(TESTS_DIR.glob("*.py"))
        for node, keywords in _fixtures(ast.parse(_read(path)))
        if keywords.get("scope", "function") != "function"
    ]


def _conftest_tree():
    return ast.parse(_read(CONFTEST))


def _autouse_function_fixtures():
    return [
        node.name
        for node, keywords in _fixtures(_conftest_tree())
        if keywords.get("autouse") is True and keywords.get("scope", "function") == "function"
    ]


def _hang_guard():
    return next(node for node in _conftest_tree().body if getattr(node, "name", None) == "hang_guard")


def _calls_named(nodes, name):
    return [
        node
        for top in nodes
        for node in ast.walk(top)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == name
    ]


def _arm_call():
    return _calls_named(_hang_guard().body, "faulthandler.dump_traceback_later")[0]


def _hang_guard_shape():
    body = _hang_guard().body
    arm_index = next(
        i for i, node in enumerate(body) if _calls_named([node], "faulthandler.dump_traceback_later")
    )
    try_index = next(i for i, node in enumerate(body) if isinstance(node, ast.Try))
    guard = body[try_index]
    yields_in_try = any(isinstance(node, ast.Yield) for top in guard.body for node in ast.walk(top))
    cancels_in_finally = bool(_calls_named(guard.finalbody, "faulthandler.cancel_dump_traceback_later"))
    return {
        "arms before try": arm_index < try_index,
        "yield in try": yields_in_try,
        "cancel in finally": cancels_in_finally,
    }


def _hang_timeout_statement():
    body = _conftest_tree().body
    index = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == ["HANG_TIMEOUT"]
    )
    following = body[index + 1]
    is_string = isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant)
    return body[index].value, following.value.value if is_string else ""


def _comment_lines(path):
    source = _read(path)
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    commented = {token.start[0] for token in tokens if token.type == tokenize.COMMENT}
    starting = {number for number, line in enumerate(source.splitlines(), 1) if line.strip().startswith("#")}
    return sorted(commented | starting)


def _module_level_hooks():
    return {
        node.name
        for node in _conftest_tree().body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("pytest_")
    }


def _modules_declaring_plugins():
    return [
        path.name
        for path in sorted(TESTS_DIR.glob("*.py"))
        for node in ast.parse(_read(path)).body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and "pytest_plugins" in ast.unparse(node).split("=", 1)[0]
    ]


def _new_test_modules():
    return sorted(path for path in TESTS_DIR.glob("test_*.py") if path.name not in EXEMPT)


def _modules_missing_markers():
    missing = []
    for path in _new_test_modules():
        doc = ast.get_docstring(ast.parse(_read(path))) or ""
        missing.extend(
            (path.name, marker) for marker in (INPUT_SPACE_MARKER, SUBSET_MARKER) if marker not in doc
        )
    return missing


def _modules_without_leading_docstring():
    bad = []
    for path in [*_new_test_modules(), CONFTEST]:
        body = ast.parse(_read(path)).body
        first = body[0] if body else None
        leading = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
        text = first.value.value if leading else None
        if not isinstance(text, str) or not text.strip():
            bad.append(path.name)
    return bad


def _post_3_10_imports(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return names & NEW_AFTER_PYTHON_3_10


def _pyproject_version():
    return re.search(r'^version = "([^"]+)"', _read(ROOT / "pyproject.toml"), re.MULTILINE).group(1)


def _report_header_text():
    header = conftest.pytest_report_header(None)
    return header if isinstance(header, str) else "\n".join(header)


def test_plan_should_list_the_feature_plans_top_down_in_the_fixed_order():
    """Given the conftest docstring, when its headings are read in order, then Suite, Features and the seven
    feature plans follow each other top-down.

    Oracle: P-T8-11 (suite first, then the split into features, then one plan per feature, then its files)
    and its feature list. A swapped, missing or duplicated plan shows up as a list difference.
    """
    plan = _plan()

    order = [plan.index("\nSuite\n"), plan.index("\nFeatures\n"), plan.index("\nPlan: ")]
    features = _plan_features()

    assert order == sorted(order)
    assert features == FEATURES


def test_plan_should_list_every_test_file_on_disk_in_exactly_one_file_row():
    """Given the file rows of all feature plans and the test files on disk, when both sets are compared,
    then each file has exactly one row and the two sets are equal.

    Oracle: P-T8-11 acceptance (every tests/test_*.py appears exactly once, nothing in the plan is missing
    from disk) and P-T2-16 (a file-to-layer table). Only file rows count, because the V-model, selection and
    blind-spot sections must repeat file names.
    """
    disk = _disk_test_files()

    planned = set(_layer_by_file())
    repeated = _files_listed_more_than_once()

    assert repeated == {}
    assert planned - disk == set(), "planned but missing on disk"
    assert disk - planned == set(), "on disk but missing from the plan"


def test_plan_should_name_at_least_one_scenario_group_per_file_in_every_feature_plan():
    """Given the seven feature plans, when their scenario groups and file rows are counted, then no plan has
    fewer groups than files and none has zero groups.

    Oracle: P-T8-10 (a plan holds the related scenarios of one feature) and its acceptance (at least one
    group per listed file). That each group belongs to its own feature only is a reviewer check.
    """
    blocks = _plan_blocks()

    short = _plans_with_fewer_groups_than_files()

    assert len(blocks) == len(FEATURES)
    assert short == []


def test_abc_optimizer_plan_should_name_the_example_scenario_groups():
    """Given the ABC optimizer plan, when its scenario groups are read, then they include the seven example
    groups of the rule.

    Oracle: P-T8-10 requirement, example list for the ABC optimizer plan.
    """
    expected = {
        "init batch and partial tell",
        "employed sweep",
        "onlooker roulette",
        "scout and limit",
        "greedy acceptance",
        "protocol errors",
        "fsm model",
    }

    groups = _lower_groups_of("ABC optimizer")

    assert expected - groups == set()


def test_plan_should_give_every_file_row_exactly_one_known_layer_label():
    """Given the file rows, when each label is read, then it is unit, integration, system or the exact
    static wording.

    Oracle: P-T2-16 (layers unit, integration, system, manual; static checks labelled 'static (outside the
    triangle, P-T1-48)'). An unknown label such as 'e2e' or a bare 'static' is reported.
    """
    rows = _plan_rows()

    unknown = _rows_with_unknown_label()

    assert len(rows) == len(_disk_test_files())
    assert unknown == []


def test_plan_should_declare_the_manual_layer_out_of_scope_with_its_reason():
    """Given the Layers section, when the sentence about the manual layer is read, then it states the out of
    scope status and the exploratory-testing reason.

    Oracle: P-T3-20 (manual layer declared out of scope with a reason) and P-T2-16 (manual is the layer of
    qualitative and exploratory tests).
    """
    layers = _section("Layers")

    sentence = _sentence_with(layers, "Manual:")

    assert "out of scope" in sentence
    assert "exploratory" in sentence


def test_plan_should_state_no_numeric_ratio_between_layers():
    """Given the Layers section, when it is searched for percentages and ratios, then none is found.

    Oracle: P-T2-16 and P-T3-20 (the source gives no number or ratio per layer; a pyramid ratio is outside
    knowledge).
    """
    layers = _flat(_section("Layers"))

    numbers = re.findall(r"\d+\s*%|\d+\s*:\s*\d+", layers)

    assert numbers == []


def test_plan_should_keep_the_unit_integration_and_system_layers_non_empty():
    """Given the file-to-layer table, when files are grouped by layer, then unit, integration and system
    each hold at least one file.

    Oracle: P-T3-20 acceptance (each layer has at least one file; larger tests complement unit tests).
    """
    layers = _files_by_layer()

    sizes = _layer_sizes(layers, ("unit", "integration", "system"))

    assert min(sizes.values()) >= 1, sizes


def test_plan_should_give_every_module_its_own_unit_layer_file():
    """Given the six library modules, when test_<module>.py is looked up in the table, then each is labelled
    unit, also test_abc.py, which calls optimize() as well.

    Oracle: P-T3-20 acceptance (every unit has its own unit-layer file besides integration or system
    coverage).
    """
    layers = _layer_by_file()

    missing = _modules_without_unit_row(layers)

    assert missing == set()


def test_plan_should_pair_each_design_level_with_its_test_level():
    """Given the V-model section, when its three pairs are parsed, then requirements map to system,
    architecture to integration and detailed design to unit, each with files of that layer.

    Oracle: P-T8-57 (requirements by system or acceptance tests, architecture by integration tests, detailed
    design by unit tests), with the sources named in its requirement.
    """
    expected = {"Requirements": "system", "Architecture": "integration", "Detailed design": "unit"}

    pairs = _v_model_pairs()

    assert _v_model_layers(pairs) == expected
    assert pairs["Requirements"][0] == "README, CHANGELOG"
    assert "ask/tell Optimizer interface" in pairs["Architecture"][0]
    assert "Karaboga & Basturk 2007" in pairs["Detailed design"][0]
    assert _v_model_files_with_other_layer() == []


def test_plan_should_state_that_the_process_is_iterative():
    """Given the V-model section, when it is searched, then it says the process is iterative.

    Oracle: P-T8-57 (the process is in practice iterative, not straight).
    """
    v_model = _section("V-model")

    sentence = _sentence_with(v_model, "iterative")

    assert "process is iterative" in sentence


def test_plan_should_give_each_unit_exactly_one_strategy_with_its_coverage_target():
    """Given the strategy table, when its rows are read, then each of the six units has exactly one row:
    code-based with branch coverage, except abc, model-based with every FSM transition as target.

    Oracle: P-T7-41 (source code available: code-based with code coverage; model-based with model coverage
    for the ABC protocol). The spec names no strategy for base; the plan's choice is checked only for
    being present once.
    """
    code_based = ("code-based (white box)", "branch coverage")
    expected = {
        "space": code_based,
        "operators": code_based,
        "base": code_based,
        "abc": ("model-based on the phase FSM", "every transition of the phase FSM"),
        "study": code_based,
        "facade": code_based,
    }

    rows = _strategy_rows()

    assert _strategy_modules(rows) == sorted(MODULES)
    assert _strategy_table(rows) == expected


def test_plan_should_write_out_every_transition_of_the_abc_phase_fsm():
    """Given the phase FSM list, when its edges are parsed, then they are exactly the eight transitions of
    the phase model, the skipped scout step and the partial init tell included.

    Oracle: P-T7-41 requirement and the phase rules of the original ABC.py (init, employed, onlooker, scout,
    back to employed; at most one scout per cycle; the initial batch may be told partially).
    """
    expected = {
        ("init", "init", ""),
        ("init", "employed", ""),
        ("employed", "employed", ""),
        ("employed", "onlooker", ""),
        ("onlooker", "onlooker", ""),
        ("onlooker", "scout", ""),
        ("scout", "employed", "scout candidate"),
        ("scout", "employed", "skipped"),
    }

    edges = set(_fsm_edges())

    assert edges == expected


@pytest.mark.parametrize(
    ("edge", "condition"),
    [
        pytest.param(("init", "init", ""), "told partially", id="init-partial-tell"),
        pytest.param(("scout", "employed", "scout candidate"), "max trial >= limit", id="scout-at-limit"),
        pytest.param(("scout", "employed", "skipped"), "max trial < limit", id="scout-skipped-below-limit"),
    ],
)
def test_plan_should_state_the_condition_of_the_named_fsm_edges(edge, condition):
    """Given one of the edges the rule names, when its condition text is read, then it states the
    discriminating condition.

    Oracle: P-T7-41 (the scout step skipped when max trial < limit and init told partially are named); the
    limit comparison follows ABC.py (scout when max trial >= limit).
    """
    edges = _fsm_edges()

    text = edges[edge]

    assert condition in text


def test_plan_should_claim_no_numeric_coverage_level():
    """Given the strategy and coverage section, when it is searched for percentages, then none is found.

    Oracle: P-T7-41 adequacy (both code and model coverage are 'some level of', no numeric level).
    """
    section = _flat(_section("Strategy and coverage"))

    levels = re.findall(r"\d+\s*%", section)

    assert levels == []


@pytest.mark.parametrize(
    ("marker", "phrase"),
    [
        pytest.param("Target criterion:", "branch coverage", id="target-is-branch"),
        pytest.param("statement -> branch -> path", "branch subsumes statement", id="order-and-subsumption"),
        pytest.param("statement -> branch -> path", "often intractable", id="path-intractable"),
        pytest.param("0/1/many", "onlooker roulette while loop", id="path-loop-roulette"),
        pytest.param("0/1/many", "candidate loop in ABCOptimizer.tell", id="path-loop-tell"),
        pytest.param("0/1/many", "optimize while loop", id="path-loop-optimize"),
        pytest.param("All-Uses", "foods", id="all-uses-foods"),
        pytest.param("All-Uses", "trial", id="all-uses-trial"),
        pytest.param("All-Uses", "phase, cycle", id="all-uses-phase-cycle"),
        pytest.param("All-Uses", "`seen` cache", id="all-uses-seen"),
        pytest.param("Measurement:", "no coverage tool may be added", id="no-coverage-tool"),
        pytest.param("Measurement:", "per-module branch inventory in docstrings", id="branch-inventory"),
    ],
)
def test_plan_should_state_the_structural_coverage_policy(marker, phrase):
    """Given the strategy and coverage section, when the sentence holding the marker is read, then it
    contains the required phrase.

    Oracle: P-T4-16 (target criterion named; order statement -> branch -> path; branch stronger at little
    effort, path often intractable) and its requirement (path limited to 0/1/many of three loops, All-Uses
    only for ABC state and `seen`, measurement through a branch inventory because no coverage tool may be
    added).
    """
    section = _section("Strategy and coverage")

    sentence = _sentence_with(section, marker)

    assert phrase in sentence


def test_plan_should_list_the_six_blind_spots_in_order():
    """Given the Blind spots section, when its numbered paragraphs are read, then the six blind spots of the
    rule appear in order.

    Oracle: P-T1-32 (compiler, parallel programming, non-functional, floating point, system level,
    requirement misunderstanding).
    """
    expected = [
        "Compiler",
        "Parallel programming",
        "Non-functional",
        "Floating point",
        "System level",
        "Requirement misunderstanding",
    ]

    titles = list(_blind_spots())

    assert titles == expected


@pytest.mark.parametrize(
    ("title", "status"),
    [
        pytest.param("Compiler", "out of scope", id="compiler"),
        pytest.param("Parallel programming", "out of scope", id="parallel"),
        pytest.param("Non-functional", "out of scope", id="non-functional"),
        pytest.param("Floating point", "in scope", id="floating-point"),
        pytest.param("System level", "out of scope", id="system-level"),
        pytest.param("Requirement misunderstanding", "in scope", id="requirement-misunderstanding"),
    ],
)
def test_blind_spot_should_carry_exactly_one_scope_status(title, status):
    """Given one blind-spot paragraph, when its scope statuses are collected, then it carries exactly the
    planned status and not the other.

    Oracle: P-T1-32 (each blind spot either handled or explicitly out of scope; requirement
    misunderstanding is not among those the source may ignore, so it is in scope).
    """
    paragraph = _blind_spots()[title]

    statuses = _scope_statuses(paragraph)

    assert statuses == {status}


@pytest.mark.parametrize(
    ("title", "phrase"),
    [
        pytest.param("Compiler", "3.10 to 3.14", id="compiler-ci-matrix"),
        pytest.param("Non-functional", "hang guard", id="non-functional-hang-guard"),
        pytest.param("Floating point", "neighbour_value", id="float-clip"),
        pytest.param("Floating point", "NaN", id="float-nan"),
        pytest.param("Floating point", "log decoding", id="float-log"),
        pytest.param("Floating point", "u=1.0", id="float-endpoint"),
        pytest.param(
            "Requirement misunderstanding", "test_traceability.py", id="misunderstanding-traceability"
        ),
        pytest.param("Requirement misunderstanding", "test_api_contract.py", id="misunderstanding-api"),
    ],
)
def test_blind_spot_should_give_the_reason_or_scope_detail(title, phrase):
    """Given one blind-spot paragraph, when it is read, then it names the detail that justifies its status.

    Oracle: P-T1-32 requirement as refined for this suite; the CI matrix comes from .github/workflows/ci.yml
    (3.10 to 3.14 on Ubuntu, 3.14 on Windows and macOS).
    """
    paragraphs = _blind_spots()

    paragraph = paragraphs[title]

    assert phrase in paragraph


def test_new_test_modules_should_state_their_input_space_and_subset():
    """Given every tests/test_*.py except the two existing files, when its docstring is read, then it holds
    both markers 'Input space:' and 'Subset:'.

    Oracle: P-T1-10 acceptance (marker spelling fixed in the conftest docstring; the existing files are
    exempt by name). A misspelling such as 'Input-space:' is reported.
    """
    modules = _new_test_modules()

    missing = _modules_missing_markers()

    assert LAYOUT in modules
    assert missing == []


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param("'Input space:'", id="input-space-marker"),
        pytest.param("'Subset:'", id="subset-marker"),
        pytest.param("Exhaustive testing is not the goal", id="not-exhaustive"),
        pytest.param("three float64 values", id="operators-count"),
        pytest.param("d unit-cube coordinates", id="space-count"),
        pytest.param("unbounded ask/tell sequences", id="abc-count"),
    ],
)
def test_plan_should_define_the_input_space_convention(phrase):
    """Given the Input space convention section, when it is read, then it fixes the marker spelling, gives
    the module-level counts and rejects exhaustive testing.

    Oracle: P-T1-10 (count the input space, accept its size, pick a subset).
    """
    sections = _sections()

    section = _flat(sections["Input space convention"])

    assert phrase in section


def test_test_modules_should_start_with_a_non_empty_docstring():
    """Given conftest.py and every non-exempt test module, when each is parsed, then its first statement is
    a non-empty string.

    Oracle: P-T1-47 second requirement (the module docstring carries input space, subset and layer;
    existing files exempt). A missing docstring, one after an import and a blank one are reported.
    """
    modules = _new_test_modules()

    bad = _modules_without_leading_docstring()

    assert len(modules) == len(_disk_test_files()) - len(EXEMPT)
    assert bad == []


def test_report_header_should_record_python_numpy_platform_and_metahpo_versions():
    """Given the conftest hook, when pytest_report_header(None) is called, then its text holds the Python,
    numpy, platform and metahpo versions.

    Oracle: P-T1-47 (run conditions such as compiler, OS and runtime libraries are part of the test record);
    the expected metahpo version is the [project] version of pyproject.toml.
    """
    version = _pyproject_version()

    header = _report_header_text()

    assert platform.python_version() in header
    assert numpy.__version__ in header
    assert platform.platform() in header
    assert f"metahpo: {version}" in header


def test_conftest_should_define_hang_guard_as_its_only_autouse_function_scoped_fixture():
    """Given the conftest source, when its autouse function-scoped fixtures are listed, then the list is
    exactly ['hang_guard'].

    Oracle: P-T5-25 acceptance (an autouse hang guard fixture). autouse missing or False, or scope
    'session', is reported. Fixtures that are not autouse are allowed, because the rule does not forbid
    shared fixtures in conftest.py.
    """
    expected = ["hang_guard"]

    autouse = _autouse_function_fixtures()

    assert autouse == expected


def test_hang_guard_should_arm_faulthandler_with_the_named_timeout_and_exit():
    """Given the hang_guard fixture, when its faulthandler.dump_traceback_later call is read, then the
    timeout is the name HANG_TIMEOUT and exit is True.

    Oracle: P-T5-25 (named constant, dump_traceback_later(HANG_TIMEOUT, exit=True)). A literal number or a
    missing exit keyword is reported.
    """
    call = _arm_call()

    keywords = _keyword_sources(call)

    assert ast.unparse(call.args[0]) == "HANG_TIMEOUT"
    assert keywords["exit"] == "True"


def test_hang_guard_should_disarm_faulthandler_in_a_finally_block_around_the_yield():
    """Given the hang_guard fixture, when its body is parsed, then it arms before a try whose body yields
    and whose finally block calls faulthandler.cancel_dump_traceback_later.

    Oracle: P-T5-25 (cancel in a finally block after yield). A cancel after a bare yield is reported.
    """
    expected = {"arms before try": True, "yield in try": True, "cancel in finally": True}

    shape = _hang_guard_shape()

    assert shape == expected


def test_hang_timeout_should_be_a_named_literal_traced_to_the_4000_ms_template():
    """Given the module constant HANG_TIMEOUT, when its assignment and the string after it are read, then
    it is a numeric literal of at least the template's 4 seconds and its docstring traces the 4000 ms origin.

    Oracle: P-T5-25 (named constant; origin the 4000 ms template; the source has no general threshold; a
    margin for the slowest CI runner). The exact value is unspecified and not asserted.
    """
    value, doc = _hang_timeout_statement()

    timeout = conftest.HANG_TIMEOUT

    assert isinstance(value, ast.Constant)
    assert timeout >= TEMPLATE_TIMEOUT_SECONDS
    assert "4000 ms" in doc
    assert "no general threshold" in doc


@pytest.mark.parametrize(
    "items",
    [
        pytest.param([], id="zero-items"),
        pytest.param(["a"], id="one-item"),
        pytest.param(["a", "b", "c"], id="many-items"),
    ],
)
def test_collection_hook_should_keep_the_order_when_the_variable_is_unset(monkeypatch, items):
    """Given METAHPO_TEST_REVERSE unset and a list of 0, 1 or many items, when the hook runs, then the list
    keeps its content and order.

    Oracle: P-T8-13 (order unchanged when the variable is not set). delenv makes the case independent of
    a run that sets the variable.
    """
    monkeypatch.delenv(REVERSE_ENV, raising=False)
    expected = list(items)

    conftest.pytest_collection_modifyitems(None, items)

    assert items == expected


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        pytest.param([], [], id="zero-items"),
        pytest.param(["a"], ["a"], id="one-item"),
        pytest.param(["a", "b", "c"], ["c", "b", "a"], id="many-items"),
    ],
)
def test_collection_hook_should_reverse_the_list_in_place_when_the_variable_is_set(
    monkeypatch, items, expected
):
    """Given METAHPO_TEST_REVERSE=1 and a list of 0, 1 or many items, when the hook runs, then the same list
    object is reversed.

    Oracle: P-T8-13 acceptance ([a, b, c] becomes [c, b, a]); the list passed in is checked, so returning a
    new list instead of reversing in place is reported.
    """
    monkeypatch.setenv(REVERSE_ENV, "1")

    conftest.pytest_collection_modifyitems(None, items)

    assert items == expected


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param("share no mutable module state", id="no-shared-state"),
        pytest.param("own Space, optimizer and seed", id="own-preconditions"),
        pytest.param("function-scoped", id="function-scope"),
        pytest.param("METAHPO_TEST_REVERSE=1", id="reverse-variable"),
    ],
)
def test_plan_should_state_the_independence_conventions(phrase):
    """Given the Independence section, when it is read, then it states the independence convention and the
    variable that reverses the order.

    Oracle: P-T8-13 requirement (tests share no mutable module state, build their own preconditions, use
    function-scoped fixtures; the variable is documented).
    """
    sections = _sections()

    section = _flat(sections["Independence"])

    assert phrase in section


def test_fixtures_should_not_be_wider_than_function_scope():
    """Given every tests/*.py, when its pytest.fixture decorators are parsed, then none has a scope other
    than 'function'.

    Oracle: P-T8-13 (fixtures are function-scoped, so no state is shared between tests). Shared mutable
    module state beyond fixtures is a reviewer check.
    """
    files = sorted(TESTS_DIR.glob("*.py"))

    wide = _wide_fixtures()

    assert CONFTEST in files
    assert wide == []


def test_test_files_should_be_named_after_their_unit_or_be_cross_cutting():
    """Given the test files on disk, when each name is matched, then it fits test_<module>[_<aspect>].py or
    is one of the cross-cutting files.

    Oracle: P-T1-23 (name the test file after its unit) and its acceptance regex and cross-cutting list.
    """
    names = _disk_test_files()

    misnamed = _misnamed_files()

    assert CROSS_CUTTING - names == set()
    assert misnamed == []


def test_cross_cutting_plan_should_list_exactly_the_cross_cutting_files():
    """Given the cross-cutting plan, when its file rows are read, then they equal the cross-cutting set.

    Oracle: P-T1-23 acceptance (the cross-cutting set equals the one written in the plan).
    """
    feature = "cross-cutting"

    planned = _planned_files_of(feature)

    assert planned == CROSS_CUTTING


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("test_space.py", id="unit-file"),
        pytest.param("test_abc.py", id="short-unit-name"),
        pytest.param("test_space_partition.py", id="unit-with-aspect"),
        pytest.param("test_abc_phase_model.py", id="aspect-with-underscore"),
    ],
)
def test_unit_name_pattern_should_accept_unit_and_aspect_names(name):
    """Given a name built from a unit and an optional aspect, when it is matched, then it is accepted.

    Oracle: the regex in the P-T1-23 acceptance; the expected verdicts follow from regex semantics.
    """
    pattern = UNIT_NAME

    match = pattern.fullmatch(name)

    assert match.group(0) == name


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("test_spaces.py", id="unit-prefix-collision"),
        pytest.param("test_abcd.py", id="unit-prefix-collision-abc"),
        pytest.param("test_space_.py", id="empty-aspect"),
        pytest.param("test_Space.py", id="upper-case"),
        pytest.param("test_space-x.py", id="hyphen"),
        pytest.param("test_space_x2.py", id="digit-in-aspect"),
        pytest.param("space_test.py", id="suffix-style"),
        pytest.param("test_optimizers.py", id="not-a-leaf-module"),
    ],
)
def test_unit_name_pattern_should_reject_names_just_outside_the_rule(name):
    """Given a name on the wrong side of the naming rule, when it is matched, then it is rejected.

    Oracle: the regex in the P-T1-23 acceptance; each name breaks exactly one part of it.
    """
    pattern = UNIT_NAME

    match = pattern.fullmatch(name)

    assert match is None


def test_package_discovery_should_find_exactly_the_six_library_modules():
    """Given the installed metahpo package, when pkgutil walks it, then the leaf modules are exactly the six
    units, so a module added later must be added to the plan.

    Oracle: P-T7-24 acceptance (discovered leaf modules equal space, operators, base, abc, study, facade).
    """
    package_path = metahpo.__path__

    leaves = _discovered_leaf_modules(package_path)

    assert leaves == MODULES


def test_every_discovered_library_module_should_have_a_unit_test_file():
    """Given the discovered leaf modules, when tests/test_<module>.py is looked up for each, then none is
    missing.

    Oracle: P-T7-24 (no module left without tests; map the module list to the test files).
    """
    modules = _discovered_leaf_modules(metahpo.__path__)

    missing = _modules_without_unit_file(TESTS_DIR, modules)

    assert missing == set()


def test_package_init_modules_should_map_to_the_api_contract_file():
    """Given the selection map, when the __init__ row is read, then it names test_api_contract.py, which is
    planned.

    Oracle: P-T7-24 (the package __init__ modules map to test_api_contract.py).
    """
    rows = _selection_map()

    init_files = rows["__init__"]

    assert init_files == ["test_api_contract.py"]
    assert "test_api_contract.py" in _layer_by_file()


def test_missing_unit_test_file_should_be_reported_for_a_copy_without_test_study(tmp_path):
    """Given a copy of the test file names without test_study.py, when the missing-file check runs on it,
    then exactly study is reported.

    Oracle: P-T7-24 acceptance (renaming test_study.py in a scratch copy makes the check fail), automated
    on tmp_path so the project tree is not touched.
    """
    names = _disk_test_files() - {"test_study.py"}
    _touch_all(tmp_path, names)

    missing = _modules_without_unit_file(tmp_path, MODULES)

    assert missing == {"study"}


def _touch_all(directory, names):
    for name in names:
        (directory / name).write_text("", encoding="utf-8")


def test_production_tree_should_contain_no_test_code():
    """Given the src/ tree, when it is searched for test modules and conftest files, then none is found.

    Oracle: P-T2-42 (production and test code in separate trees) and its acceptance.
    """
    source = SRC

    found = sorted(source.rglob("test_*.py")) + sorted(source.rglob("conftest.py"))

    assert (source / "metahpo" / "__init__.py").is_file()
    assert found == []


def test_test_modules_should_import_only_metahpo_numpy_pytest_and_the_standard_library():
    """Given every tests/*.py, when its imports are classified, then none breaks the import rule.

    Oracle: P-T2-42 acceptance (metahpo names only from 'metahpo' or 'metahpo.operators'; top-level packages
    only metahpo, numpy, pytest, conftest and sys.stdlib_module_names) and P-T4-16 (no coverage tool or
    plugin import).
    """
    files = sorted(TESTS_DIR.glob("*.py"))

    violations = _suite_import_violations()

    assert len(files) >= 2
    assert violations == {}


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("import unittest.mock", id="stdlib-submodule"),
        pytest.param("from __future__ import annotations", id="future"),
        pytest.param("import conftest", id="conftest"),
        pytest.param("from metahpo import Space", id="public-api"),
        pytest.param("from metahpo.operators import fitness", id="operators-exception"),
        pytest.param("import numpy as np", id="numpy"),
    ],
)
def test_import_rule_should_accept_allowed_imports(source):
    """Given an allowed import, when it is classified, then no violation is reported.

    Oracle: P-T2-42 acceptance and the hard constraints (pytest, numpy, standard library, metahpo public API,
    metahpo.operators for operator rules).
    """
    snippet = source

    violations = _import_violations(snippet)

    assert violations == []


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("from metahpo.space import Float", id="deep-metahpo-from"),
        pytest.param("import metahpo.space", id="deep-metahpo-import"),
        pytest.param("import hypothesis", id="third-party"),
        pytest.param("import coverage", id="coverage-tool"),
        pytest.param("import pytest_cov", id="pytest-plugin"),
        pytest.param("from . import helpers", id="relative"),
    ],
)
def test_import_rule_should_reject_forbidden_imports(source):
    """Given a forbidden import, when it is classified, then exactly one violation is reported.

    Oracle: P-T2-42 acceptance and P-T4-16 (no coverage.py or plugin in any import).
    """
    snippet = source

    violations = _import_violations(snippet)

    assert len(violations) == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("test_space.py", EXISTING_SPACE_TESTS, id="existing-space-tests"),
        pytest.param("test_abc.py", EXISTING_ABC_TESTS, id="existing-abc-tests"),
    ],
)
def test_existing_test_file_should_keep_all_its_test_functions(name, expected):
    """Given one of the two existing test files, when its top-level test functions are listed, then all
    original names are still present.

    Oracle: P-T6-29 (keep tests instead of deleting them; overlap is handled by selection) and its
    acceptance with the five and nine original names.
    """
    path = TESTS_DIR / name

    names = _top_level_test_names(path)

    assert expected - names == set()


def test_selection_map_should_use_only_library_modules_and_init_as_keys():
    """Given the selection map, when its keys are read, then they are the six modules plus __init__.

    Oracle: P-T6-30 acceptance (every module in the map exists) and P-T7-24 (__init__ maps to
    test_api_contract.py).
    """
    rows = _selection_map()

    keys = set(rows)

    assert keys == MODULES | {"__init__"}


def test_selection_map_should_select_each_modules_own_test_files():
    """Given the selection map, when each module row is read, then it holds the module's own pattern
    test_<module>*.py.

    Oracle: P-T6-30 requirement (step 2 selects test_<module>*.py plus the listed files).
    """
    rows = _selection_map()

    missing = _modules_without_own_pattern(rows)

    assert missing == set()


def test_selection_map_should_make_every_planned_file_selectable():
    """Given the selection map and the always-selected files, when every pattern is expanded against the
    planned files, then each planned file is selected by at least one row.

    Oracle: P-T6-30 acceptance (every test file appears under at least one module).
    """
    always = _always_selected()

    unselectable = _unselectable_files()

    assert always == ["test_static.py", "test_suite_layout.py", "test_traceability.py"]
    assert unselectable == set()


def test_selection_map_should_have_no_pattern_that_matches_nothing():
    """Given the selection map, when each pattern is expanded against the planned files, then every pattern
    selects at least one file.

    Oracle: P-T6-30 (the map is used for selection, so a dead entry would hide a renamed file).
    """
    rows = _selection_map()

    dead = _patterns_matching_nothing()

    assert len(rows) == len(MODULES) + 1
    assert dead == []


def test_plan_should_write_down_the_import_graph_that_src_has():
    """Given the import graph written in the Selection section and src/metahpo parsed with ast, when both
    edge sets are compared, then they are equal.

    Oracle: P-T6-30 (step 1 identifies the affected code; at module level that is the changed module and
    every module importing it). The src tree is the ground truth; a stale plan graph shows up as a set
    difference.
    """
    planned = _plan_import_graph()

    actual = _src_import_graph()

    assert actual - planned == set(), "imports in src missing from the plan graph"
    assert planned - actual == set(), "plan graph edges with no import in src"


def test_selection_map_should_contain_the_rows_of_all_importing_modules():
    """Given the import graph of src/metahpo and the selection map, when each row is expanded against the
    planned files, then for every edge the imported module's row selects every file of the importing
    module's row.

    Oracle: P-T6-30 adequacy (the selected set equals the modification-traversing tests, P-T6-32); at module
    level a change to space.py can traverse every test of base, abc, study and facade, so a row lacking one
    of them is not a safe selection. Each failure names the imported module, the importing module and the
    files that are not selected. The edge base -> space is asserted first so an empty graph cannot pass.
    """
    edges = _src_import_graph()
    rows = _selection_map()
    names = set(_layer_by_file())

    missing = _rows_missing_dependent_files(rows, edges, names)

    assert ("base", "space") in edges
    assert missing == []


def test_selection_map_should_select_every_test_file_for_the_modules_it_uses():
    """Given every planned test file on disk and the public metahpo names it uses (mapped to their modules
    through the package __init__ files), when the selection map is expanded, then each file is selected for
    every module it uses.

    Oracle: P-T6-30 adequacy (a test that uses a module traverses it, so it belongs to that module's
    selection) and metahpo/__init__.py for the name-to-module mapping.
    """
    rows = _selection_map()
    always = _always_selected()

    missing = _files_not_selected_for_used_modules(rows, always)

    assert missing == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param("from metahpo import Space", {"space"}, id="from-package"),
        pytest.param("import metahpo\nmetahpo.optimize", {"facade"}, id="package-attribute"),
        pytest.param("from metahpo.operators import fitness", {"operators"}, id="from-submodule"),
        pytest.param("import metahpo.operators\nmetahpo.operators.fitness", {"operators"}, id="dotted"),
        pytest.param("import metahpo.operators as ops\nops.fitness", {"operators"}, id="aliased-import"),
        pytest.param("from metahpo import ABCOptimizer, Study", {"abc", "study"}, id="two-names"),
        pytest.param("import metahpo as mh\nmh.optimize", {"facade"}, id="package-alias"),
        pytest.param(
            "from metahpo import *", {"abc", "base", "space", "study", "facade"}, id="star-import-binds-all"
        ),
        pytest.param("import metahpo", set(), id="bare-package-not-followed"),
        pytest.param("import numpy", set(), id="no-metahpo"),
    ],
)
def test_used_module_mapping_should_cover_every_allowed_import_form(source, expected):
    """Given a snippet in one of the import forms the suite's import rule allows, when the metahpo modules it
    uses are mapped, then the result is the module that defines the names it uses.

    Oracle: P-T6-30 (a test that uses a module belongs to that module's selection), P-T2-42 (metahpo names
    come from 'metahpo' or 'metahpo.operators', in any import form) and metahpo/__init__.py for the
    name-to-module mapping. The aliased form 'import metahpo.operators as ops' must map to operators,
    otherwise a file using it outside the operators row would not be selected when operators changes. The
    README's own form 'import metahpo as mh' with mh.optimize must map to facade for the same reason, and
    'from metahpo import *' binds every name of metahpo.__all__, whose defining modules are abc
    (ABCOptimizer), base (Candidate, Optimizer), space (Categorical, Float, Int, Space), study (Study, Trial)
    and facade (optimize). A bare 'import metahpo' maps to nothing because the package __init__ is not
    followed (see the Selection section).
    """
    modules = _src_modules()

    found = _modules_used_in(source, modules)

    assert found == expected


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param("Step 1: identify the affected module", id="step-1"),
        pytest.param("Step 2: select that module's test files test_<module>*.py", id="step-2"),
        pytest.param("the rows of every module that imports it", id="step-2-dependents"),
        pytest.param("closed over the import graph", id="closed-over-imports"),
        pytest.param("package __init__ files, which import every module, are not followed", id="init-scope"),
        pytest.param("(P-T6-32) is out of scope", id="statement-level-out-of-scope"),
    ],
)
def test_plan_should_state_both_selection_steps(phrase):
    """Given the Selection section, when it is read, then it states the two steps and leaves
    statement-level selection out of scope.

    Oracle: P-T6-30 (identify the affected code, then select the tests to run) and P-T6-29 (selection
    instead of deletion).
    """
    sections = _sections()

    section = _flat(sections["Selection"])

    assert phrase in section


def test_pyproject_should_make_the_tests_directory_the_only_test_path(pytestconfig):
    """Given pyproject.toml, when it is read as text and through pytest's parsed ini value, then testpaths is
    exactly ["tests"].

    Oracle: P-T6-02 acceptance (pyproject read as text, without tomllib, for Python 3.10); the parsed value
    catches a format pytest reads differently.
    """
    text = _read(ROOT / "pyproject.toml")

    testpaths = pytestconfig.getini("testpaths")

    assert 'testpaths = ["tests"]' in text
    assert testpaths == ["tests"]


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param("The whole tests/ directory, run by plain `python -m pytest`", id="whole-directory"),
        pytest.param('testpaths = ["tests"], is the regression suite', id="through-testpaths"),
        pytest.param("test_regression.py holds the pinned seeded outcomes", id="pinned-outcomes"),
        pytest.param("Nothing depends on manual steps", id="no-manual-steps"),
    ],
)
def test_plan_should_name_the_tests_directory_as_the_regression_suite(phrase):
    """Given the Regression suite section, when it is read, then it names the suite and the pinned outcomes.

    Oracle: P-T6-02 (regression testing rests on a regression test suite, not on single manual tries).
    """
    sections = _sections()

    section = _flat(sections["Regression suite"])

    assert phrase in section


@pytest.mark.parametrize(
    "path",
    [pytest.param(CONFTEST, id="conftest"), pytest.param(LAYOUT, id="suite-layout")],
)
def test_suite_skeleton_file_should_contain_no_comment_lines(path):
    """Given conftest.py or this module, when its tokens and stripped lines are scanned, then no comment
    and no line starting with '#' is found.

    Oracle: P-T2-42 second requirement and the hard constraint 'no # comment lines anywhere'.
    """
    source = _read(path)

    lines = _comment_lines(path)

    assert source.strip() != ""
    assert lines == []


def test_conftest_should_define_the_two_planned_hooks_and_only_builtin_hookspecs(pytestconfig):
    """Given the conftest source, when its module-level pytest_ functions are listed, then the two planned
    hooks are among them and every one of them is a known pytest hookspec.

    Oracle: P-T2-42 (only built-in hooks, no plugins), P-T1-47 (report header) and P-T8-13 (collection
    order). A misspelled hook such as pytest_report_headers is not a hookspec; further built-in hooks such
    as pytest_configure are allowed, because the rule does not forbid them.
    """
    expected = {"pytest_report_header", "pytest_collection_modifyitems"}

    hooks = _module_level_hooks()

    assert expected - hooks == set()
    assert _known_hookspecs(pytestconfig, hooks) == hooks


def test_test_modules_should_not_declare_pytest_plugins():
    """Given every tests/*.py, when module-level assignments are listed, then none assigns pytest_plugins.

    Oracle: P-T2-42 (no plugins) and the hard constraint (no pytest plugins).
    """
    files = sorted(TESTS_DIR.glob("*.py"))

    declaring = _modules_declaring_plugins()

    assert CONFTEST in files
    assert declaring == []


@pytest.mark.parametrize(
    "path",
    [pytest.param(CONFTEST, id="conftest"), pytest.param(LAYOUT, id="suite-layout")],
)
def test_suite_skeleton_file_should_parse_with_the_python_3_10_grammar(path):
    """Given conftest.py or this module, when it is parsed with feature_version (3, 10), then it parses and
    imports no module added after 3.10.

    Oracle: P-T2-42 second requirement and pyproject requires-python >= 3.10. feature_version is best
    effort; the CI job on Python 3.10 is the full check.
    """
    source = _read(path)

    tree = ast.parse(source, feature_version=(3, 10))

    assert _post_3_10_imports(tree) == set()
