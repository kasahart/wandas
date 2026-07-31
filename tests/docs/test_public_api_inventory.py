from __future__ import annotations

import importlib
import inspect
from collections import Counter, defaultdict
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from types import ModuleType

import pytest

from wandas._public_api import (
    CLASSIFICATIONS,
    DEPRECATED_COMPATIBILITY,
    PRIVATE_INTERNAL,
    PUBLIC_API_INVENTORY,
    STABLE_PUBLIC,
    SYMBOL_KINDS,
    TRACKED_PACKAGE_SURFACES,
    ApiSymbol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOVERNED_UNDERSCORED_NAMES = {
    "wandas": frozenset({"_LAZY_EXPORTS"}),
    "wandas.processing": frozenset({"_OPERATION_MODULES", "_OPERATION_REGISTRY"}),
}
DYNAMIC_EXPORT_MAPPINGS = {
    "wandas": ("_LAZY_EXPORTS",),
    "wandas.processing": ("_LAZY_OPERATION_CLASSES",),
}
STANDARD_MODULE_DUNDERS = frozenset(
    {
        "__all__",
        "__annotations__",
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "__warningregistry__",
    }
)
PROJECTION_BEGIN = "<!-- public-api-inventory:begin -->"
PROJECTION_END = "<!-- public-api-inventory:end -->"
PROJECTION_COLUMNS = ("Surface", "Symbol", "Kind", "Stability", "In __all__", "Replacement", "Support")
PROJECTION_HEADER = f"| {' | '.join(PROJECTION_COLUMNS)} |"
PROJECTION_SEPARATOR = f"| {' | '.join('---' for _ in PROJECTION_COLUMNS)} |"
ProjectionEntry = tuple[str, str, str, str, str, str, str]
ProjectionRecord = tuple[str | None, ProjectionEntry]


def _inventory_projection(module_name: str, symbol: ApiSymbol) -> ProjectionRecord:
    """Project every compatibility-significant inventory field exactly once."""
    return (
        symbol.documentation,
        (
            module_name,
            symbol.name,
            symbol.kind,
            symbol.classification,
            "yes" if symbol.in_all else "no",
            symbol.replacement or "—",
            symbol.support or "—",
        ),
    )


def _documented_api_projection(documentation: str) -> tuple[tuple[ProjectionEntry, ...], tuple[str, ...]]:
    """Parse one fail-closed, human-readable public-API projection table."""

    errors: list[str] = []
    if documentation.count(PROJECTION_BEGIN) != 1 or documentation.count(PROJECTION_END) != 1:
        return (), ("expected exactly one public API projection block",)
    begin = documentation.index(PROJECTION_BEGIN)
    end = documentation.index(PROJECTION_END)
    if end < begin:
        return (), ("public API projection markers are out of order",)
    block = documentation[begin + len(PROJECTION_BEGIN) : end]
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return (), ("public API projection table is incomplete",)
    if lines[0] != PROJECTION_HEADER:
        errors.append(f"invalid projection header {lines[0]!r}")
    if lines[1] != PROJECTION_SEPARATOR:
        errors.append(f"invalid projection separator {lines[1]!r}")

    entries: list[ProjectionEntry] = []
    for line_number, line in enumerate(lines[2:], start=3):
        columns = line.split("|")
        if len(columns) != len(PROJECTION_COLUMNS) + 2 or columns[0].strip() or columns[-1].strip():
            errors.append(f"projection row {line_number} must have exactly {len(PROJECTION_COLUMNS)} columns")
            continue
        surface_cell, symbol_cell, kind, classification, exported, replacement, support = (
            column.strip() for column in columns[1:-1]
        )
        if not all((surface_cell, symbol_cell, kind, classification, exported, replacement, support)):
            errors.append(f"projection row {line_number} contains an empty value")
            continue
        if not (surface_cell.startswith("`") and surface_cell.endswith("`") and len(surface_cell) > 2):
            errors.append(f"projection row {line_number} has invalid surface {surface_cell!r}")
            continue
        if not (symbol_cell.startswith("`") and symbol_cell.endswith("`") and len(symbol_cell) > 2):
            errors.append(f"projection row {line_number} has invalid symbol {symbol_cell!r}")
            continue
        if exported not in {"yes", "no"}:
            errors.append(f"projection row {line_number} has invalid In __all__ value {exported!r}")
            continue
        entries.append((surface_cell[1:-1], symbol_cell[1:-1], kind, classification, exported, replacement, support))
    return tuple(entries), tuple(errors)


def _runtime_symbol_kind(value: object) -> str:
    if inspect.isclass(value):
        return "class"
    if isinstance(value, Mapping):
        return "mapping"
    if callable(value):
        return "function"
    return "attribute"


def _wandas_api_candidates(module: ModuleType) -> set[str]:
    """Return governed names visible on a Wandas package module."""

    candidates = {name for name in GOVERNED_UNDERSCORED_NAMES.get(module.__name__, ()) if hasattr(module, name)}
    for name, value in vars(module).items():
        if name.startswith("__") and name.endswith("__"):
            if name not in STANDARD_MODULE_DUNDERS:
                candidates.add(name)
            continue
        if name.startswith("_"):
            continue
        if isinstance(value, ModuleType) and value.__name__ == f"{module.__name__}.{name}":
            continue
        candidates.add(name)
    for mapping_name in DYNAMIC_EXPORT_MAPPINGS.get(module.__name__, ()):
        dynamic_exports = vars(module).get(mapping_name, {})
        if isinstance(dynamic_exports, Mapping):
            candidates.update(dynamic_exports)
    return candidates


def _inventory_errors(
    inventory: Mapping[str, tuple[ApiSymbol, ...]],
    documentation_overrides: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_by_document: defaultdict[str, list[ProjectionEntry]] = defaultdict(list)
    expected_surfaces = set(TRACKED_PACKAGE_SURFACES)
    actual_surfaces = set(inventory)
    missing_surfaces = sorted(expected_surfaces - actual_surfaces)
    extra_surfaces = sorted(actual_surfaces - expected_surfaces)
    if missing_surfaces:
        errors.append(f"inventory is missing tracked surfaces {missing_surfaces!r}")
    if extra_surfaces:
        errors.append(f"inventory has unknown surfaces {extra_surfaces!r}")

    for module_name in TRACKED_PACKAGE_SURFACES:
        module = importlib.import_module(module_name)
        symbols = inventory.get(module_name)
        if symbols is None:
            errors.append(f"{module_name}: missing inventory")
            continue

        names = [symbol.name for symbol in symbols]
        if len(names) != len(set(names)):
            errors.append(f"{module_name}: duplicate inventory names")

        expected_all = [symbol.name for symbol in symbols if symbol.in_all]
        module_all = getattr(module, "__all__", None)
        if module_all is None:
            errors.append(f"{module_name}: missing explicit __all__")
            actual_all = []
        else:
            actual_all = list(module_all)
        if actual_all != expected_all:
            errors.append(f"{module_name}: __all__ {actual_all!r} != {expected_all!r}")

        undeclared = _wandas_api_candidates(module) - set(names)
        if undeclared:
            errors.append(f"{module_name}: unclassified visible names {sorted(undeclared)!r}")

        for symbol in symbols:
            qualified_name = f"{module_name}.{symbol.name}"
            if symbol.kind not in SYMBOL_KINDS:
                errors.append(f"{qualified_name}: unknown symbol kind")
            if symbol.classification not in CLASSIFICATIONS:
                errors.append(f"{qualified_name}: unknown classification")
            if symbol.classification == PRIVATE_INTERNAL and symbol.in_all:
                errors.append(f"{qualified_name}: private symbol is in __all__")
            if symbol.classification == DEPRECATED_COMPATIBILITY:
                if not symbol.replacement:
                    errors.append(f"{qualified_name}: deprecated symbol has no replacement")
                if not symbol.support:
                    errors.append(f"{qualified_name}: deprecated symbol has no support window")
            elif symbol.replacement or symbol.support:
                errors.append(f"{qualified_name}: non-deprecated symbol has deprecation metadata")
            if not hasattr(module, symbol.name):
                errors.append(f"{qualified_name}: inventory symbol is not importable")
            elif symbol.kind in SYMBOL_KINDS:
                runtime_kind = _runtime_symbol_kind(getattr(module, symbol.name))
                if runtime_kind != symbol.kind:
                    errors.append(f"{qualified_name}: runtime kind {runtime_kind!r} != {symbol.kind!r}")
            if symbol.classification != PRIVATE_INTERNAL:
                documentation_path, projection_entry = _inventory_projection(module_name, symbol)
                if not documentation_path:
                    errors.append(f"{qualified_name}: public symbol has no documentation path")
                else:
                    expected_by_document[documentation_path].append(projection_entry)

    documents_to_validate = set(expected_by_document)
    for document in (REPO_ROOT / "docs/src").rglob("*.md"):
        relative_path = document.relative_to(REPO_ROOT).as_posix()
        if documentation_overrides is not None and relative_path in documentation_overrides:
            documentation = documentation_overrides[relative_path]
        else:
            documentation = document.read_text(encoding="utf-8")
        if any(marker in documentation for marker in (PROJECTION_BEGIN, PROJECTION_END, PROJECTION_HEADER)):
            documents_to_validate.add(relative_path)

    for relative_path in sorted(documents_to_validate):
        expected_entries = expected_by_document.get(relative_path, [])
        document = REPO_ROOT / relative_path
        if not document.is_file():
            errors.append(f"{relative_path}: missing documentation file")
            continue
        if documentation_overrides is not None and relative_path in documentation_overrides:
            documentation = documentation_overrides[relative_path]
        else:
            documentation = document.read_text(encoding="utf-8")
        actual_entries, parse_errors = _documented_api_projection(documentation)
        errors.extend(f"{relative_path}: {error}" for error in parse_errors)
        expected_counter = Counter(expected_entries)
        actual_counter = Counter(actual_entries)
        missing = list((expected_counter - actual_counter).elements())
        extra = list((actual_counter - expected_counter).elements())
        if missing:
            errors.append(f"{relative_path}: missing projection rows {missing!r}")
        if extra:
            errors.append(f"{relative_path}: extra projection rows {extra!r}")
    return tuple(errors)


def test_canonical_inventory_matches_exports_and_api_documentation() -> None:
    assert _inventory_errors(PUBLIC_API_INVENTORY) == ()


def test_documentation_projection_parser_is_fail_closed() -> None:
    valid = f"""{PROJECTION_BEGIN}
{PROJECTION_HEADER}
{PROJECTION_SEPARATOR}
| `wandas` | `read` | function | stable public | yes | — | — |
{PROJECTION_END}"""
    assert _documented_api_projection(valid) == (
        (("wandas", "read", "function", "stable public", "yes", "—", "—"),),
        (),
    )

    for broken in (
        valid.replace("Surface", "Module"),
        valid.replace("| yes | — | — |", "| yes | — | — | extra |"),
        valid.replace("| function |", "|  |"),
        valid.replace("| yes |", "| maybe |"),
        valid.replace(PROJECTION_END, ""),
        f"{PROJECTION_END}\n{valid.replace(PROJECTION_END, '')}",
    ):
        _, parse_errors = _documented_api_projection(broken)
        assert parse_errors


def test_inventory_is_structurally_immutable() -> None:
    import wandas._public_api as inventory_module

    assert not hasattr(inventory_module, "_INVENTORY")

    with pytest.raises(TypeError):
        PUBLIC_API_INVENTORY["wandas"] = ()  # ty: ignore[invalid-assignment]

    with pytest.raises(AttributeError):
        PUBLIC_API_INVENTORY["wandas"].append(  # ty: ignore[unresolved-attribute]
            ApiSymbol("drift", "function", PRIVATE_INTERNAL, False)
        )


def test_drift_gate_detects_a_deliberately_mutated_export() -> None:
    mutated: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
    top_level = list(mutated["wandas"])
    index = next(index for index, symbol in enumerate(top_level) if symbol.name == "supported_formats")
    top_level[index] = top_level[index]._replace(name="supported_formats_drift")
    mutated["wandas"] = tuple(top_level)

    errors = _inventory_errors(mutated)

    assert any("wandas: __all__" in error for error in errors)
    assert any("wandas.supported_formats_drift" in error for error in errors)


def test_inventory_rejects_unknown_or_missing_surface_keys() -> None:
    with_unknown: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
    with_unknown["wandas.untracked"] = (ApiSymbol("Invented", "banana", STABLE_PUBLIC, True),)
    assert "inventory has unknown surfaces ['wandas.untracked']" in _inventory_errors(with_unknown)

    missing = dict(PUBLIC_API_INVENTORY)
    del missing["wandas.datasets"]
    assert "inventory is missing tracked surfaces ['wandas.datasets']" in _inventory_errors(missing)


def test_public_inventory_requires_documentation_and_runtime_kind() -> None:
    channel_frame = next(symbol for symbol in PUBLIC_API_INVENTORY["wandas"] if symbol.name == "ChannelFrame")
    for mutated_symbol, expected_error in (
        (channel_frame._replace(documentation=None), "public symbol has no documentation path"),
        (channel_frame._replace(kind="function"), "runtime kind 'class' != 'function'"),
    ):
        mutated: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
        top_level = list(mutated["wandas"])
        index = next(index for index, symbol in enumerate(top_level) if symbol.name == "ChannelFrame")
        top_level[index] = mutated_symbol
        mutated["wandas"] = tuple(top_level)

        assert any(expected_error in error for error in _inventory_errors(mutated))


def test_documentation_projection_detects_classification_drift() -> None:
    mutated: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
    top_level = list(mutated["wandas"])
    index = next(index for index, symbol in enumerate(top_level) if symbol.name == "ChannelFrame")
    top_level[index] = top_level[index]._replace(classification="experimental public")
    mutated["wandas"] = tuple(top_level)

    errors = _inventory_errors(mutated)

    assert any("missing projection rows" in error and "experimental public" in error for error in errors)
    assert any("extra projection rows" in error and "stable public" in error for error in errors)


def test_documentation_projection_detects_export_membership_drift() -> None:
    mutated: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
    top_level = list(mutated["wandas"])
    index = next(index for index, symbol in enumerate(top_level) if symbol.name == "read_wav")
    top_level[index] = top_level[index]._replace(in_all=True)
    mutated["wandas"] = tuple(top_level)

    errors = _inventory_errors(mutated)

    assert any("wandas: __all__" in error for error in errors)
    assert any("missing projection rows" in error and "read_wav" in error and "yes" in error for error in errors)
    assert any("extra projection rows" in error and "read_wav" in error and "no" in error for error in errors)


def test_stale_projection_page_is_checked_after_its_last_public_entry_is_removed() -> None:
    mutated: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
    utilities = list(mutated["wandas.utils"])
    index = next(index for index, symbol in enumerate(utilities) if symbol.name == "validate_sampling_rate")
    utilities[index] = utilities[index]._replace(classification=PRIVATE_INTERNAL, documentation=None)
    mutated["wandas.utils"] = tuple(utilities)

    errors = _inventory_errors(mutated)

    assert any(
        "docs/src/api/utils.md" in error and "extra projection rows" in error and "validate_sampling_rate" in error
        for error in errors
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_documentation_projection_is_bidirectional(mutation: str) -> None:
    relative_path = "docs/src/api/index.md"
    documentation = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    row = "| `wandas` | `ChannelFrame` | class | stable public | yes | — | — |"
    if mutation == "missing":
        documentation = documentation.replace(f"{row}\n", "", 1)
        expected_error = "missing projection rows"
    elif mutation == "extra":
        extra = "| `wandas` | `UnexpectedApi` | function | stable public | yes | — | — |"
        documentation = documentation.replace(PROJECTION_END, f"{extra}\n{PROJECTION_END}", 1)
        expected_error = "extra projection rows"
    else:
        documentation = documentation.replace(PROJECTION_END, f"{row}\n{PROJECTION_END}", 1)
        expected_error = "extra projection rows"

    errors = _inventory_errors(PUBLIC_API_INVENTORY, {relative_path: documentation})

    assert any(relative_path in error and expected_error in error for error in errors)


def test_projected_documentation_pages_are_in_mkdocs_navigation() -> None:
    nav = (REPO_ROOT / "docs/mkdocs.yml").read_text(encoding="utf-8")
    projected_paths = {
        symbol.documentation.removeprefix("docs/src/")
        for symbols in PUBLIC_API_INVENTORY.values()
        for symbol in symbols
        if symbol.classification != PRIVATE_INTERNAL and symbol.documentation is not None
    }
    for relative_path in projected_paths:
        assert f": {relative_path}" in nav


def test_documented_version_attribute_is_stable_and_drift_checked() -> None:
    import wandas

    version_symbol = next(symbol for symbol in PUBLIC_API_INVENTORY["wandas"] if symbol.name == "__version__")
    assert version_symbol.classification == STABLE_PUBLIC
    assert version_symbol.in_all is False
    assert isinstance(wandas.__version__, str)
    assert wandas.__version__

    mutated: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
    mutated["wandas"] = tuple(symbol for symbol in mutated["wandas"] if symbol.name != "__version__")

    errors = _inventory_errors(mutated)

    assert "wandas: unclassified visible names ['__version__']" in errors


def test_new_nonstandard_dunder_requires_an_explicit_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    import wandas

    monkeypatch.setattr(wandas, "__build__", "local", raising=False)

    errors = _inventory_errors(PUBLIC_API_INVENTORY)

    assert "wandas: unclassified visible names ['__build__']" in errors


def test_new_package_data_attribute_requires_an_explicit_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    import wandas.utils as utils

    monkeypatch.setattr(utils, "NEW_PUBLIC_CONSTANT", 42, raising=False)

    errors = _inventory_errors(PUBLIC_API_INVENTORY)

    assert "wandas.utils: unclassified visible names ['NEW_PUBLIC_CONSTANT']" in errors


def test_new_callable_package_attribute_requires_an_explicit_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wandas.utils as utils

    monkeypatch.setattr(utils, "NEW_PUBLIC_FACTORY", partial(int, base=10), raising=False)

    errors = _inventory_errors(PUBLIC_API_INVENTORY)

    assert "wandas.utils: unclassified visible names ['NEW_PUBLIC_FACTORY']" in errors


def test_new_module_alias_requires_an_explicit_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import wandas.utils as utils

    monkeypatch.setattr(utils, "json_alias", json, raising=False)

    errors = _inventory_errors(PUBLIC_API_INVENTORY)

    assert "wandas.utils: unclassified visible names ['json_alias']" in errors


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("replacement", "read"),
        ("support", "Deprecated immediately."),
    ],
)
def test_deprecated_metadata_is_projected_exactly(field: str, value: str) -> None:
    mutated: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
    top_level = list(mutated["wandas"])
    index = next(index for index, symbol in enumerate(top_level) if symbol.name == "from_ndarray")
    if field == "replacement":
        top_level[index] = top_level[index]._replace(replacement=value)
    else:
        top_level[index] = top_level[index]._replace(support=value)
    mutated["wandas"] = tuple(top_level)

    errors = _inventory_errors(mutated)

    assert any(
        "docs/src/api/index.md" in error and "missing projection rows" in error and value in error for error in errors
    )
    assert any("docs/src/api/index.md" in error and "extra projection rows" in error for error in errors)


@pytest.mark.parametrize(
    ("symbol_name", "field", "value", "expected_error"),
    [
        ("from_ndarray", "replacement", None, "deprecated symbol has no replacement"),
        ("from_ndarray", "support", None, "deprecated symbol has no support window"),
        ("ChannelFrame", "replacement", "read", "non-deprecated symbol has deprecation metadata"),
        ("ChannelFrame", "support", "Forever.", "non-deprecated symbol has deprecation metadata"),
    ],
)
def test_deprecation_payload_matches_classification(
    symbol_name: str,
    field: str,
    value: str | None,
    expected_error: str,
) -> None:
    mutated: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
    top_level = list(mutated["wandas"])
    index = next(index for index, symbol in enumerate(top_level) if symbol.name == symbol_name)
    if field == "replacement":
        top_level[index] = top_level[index]._replace(replacement=value)
    else:
        top_level[index] = top_level[index]._replace(support=value)
    mutated["wandas"] = tuple(top_level)

    assert any(expected_error in error for error in _inventory_errors(mutated))


@pytest.mark.parametrize("name", ["_OPERATION_MODULES", "_OPERATION_REGISTRY"])
def test_governed_processing_registries_are_drift_checked(name: str) -> None:
    mutated: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
    mutated["wandas.processing"] = tuple(symbol for symbol in mutated["wandas.processing"] if symbol.name != name)

    errors = _inventory_errors(mutated)

    assert f"wandas.processing: unclassified visible names ['{name}']" in errors


def test_trim_support_window_and_extension_workflow_match_policy() -> None:
    processing_symbols = PUBLIC_API_INVENTORY["wandas.processing"]
    trim = next(symbol for symbol in processing_symbols if symbol.name == "Trim")
    assert trim.support is not None
    assert "through 0.7.x" in trim.support
    assert "no earlier than 0.8.0" in trim.support

    processing_docs = (REPO_ROOT / "docs/src/api/processing.md").read_text(encoding="utf-8")
    stability_docs = (REPO_ROOT / "docs/src/explanation/public-api-stability.md").read_text(encoding="utf-8")
    extension_guide = (REPO_ROOT / "docs/src/contributing/frame-operation-extensions.md").read_text(encoding="utf-8")
    normalized_processing_docs = " ".join(processing_docs.split())
    normalized_stability_docs = " ".join(stability_docs.split())
    normalized_extension_guide = " ".join(extension_guide.split())
    assert "retained through 0.7.x" in normalized_processing_docs
    assert "no earlier than 0.8.0" in normalized_processing_docs
    assert "remains supported through 0.7.x" in normalized_stability_docs
    assert "removable no earlier than 0.8.0" in normalized_stability_docs
    assert "PUBLIC_API_INVENTORY" in normalized_extension_guide
    assert "top-level `wandas.__all__`" in normalized_extension_guide
    assert "`kind`" in normalized_extension_guide
    assert "documentation path" in normalized_extension_guide
    assert "projection" in normalized_extension_guide


def test_drift_gate_detects_an_unclassified_lazy_processing_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wandas.processing as processing

    monkeypatch.setitem(
        processing._LAZY_OPERATION_CLASSES,
        "UnclassifiedLazyOperation",
        ("unclassified_lazy_operation", "wandas.processing.spectral"),
    )

    errors = _inventory_errors(PUBLIC_API_INVENTORY)

    assert any(
        "wandas.processing: unclassified visible names ['UnclassifiedLazyOperation']" in error for error in errors
    )


def test_drift_gate_detects_a_removed_top_level_lazy_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wandas

    mutated: dict[str, tuple[ApiSymbol, ...]] = dict(PUBLIC_API_INVENTORY)
    mutated["wandas"] = tuple(symbol for symbol in mutated["wandas"] if symbol.name != "ChannelFrameDataset")
    monkeypatch.setattr(wandas, "__all__", [name for name in wandas.__all__ if name != "ChannelFrameDataset"])
    relative_path = "docs/src/api/index.md"
    documentation = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    projection_row = "| `wandas` | `ChannelFrameDataset` | class | stable public | yes | — | — |\n"
    documentation = documentation.replace(projection_row, "", 1)

    errors = _inventory_errors(mutated, {relative_path: documentation})

    assert "wandas: unclassified visible names ['ChannelFrameDataset']" in errors


def test_internal_registry_and_utils_helpers_stay_outside_all() -> None:
    import wandas.processing as processing
    import wandas.utils as utils

    assert {
        "_OPERATION_MODULES",
        "_OPERATION_REGISTRY",
        "__getattr__",
        "apply_channel_factors",
        "register_lazy_operation",
    }.isdisjoint(processing.__all__)
    assert {
        "accepted_kwargs",
        "filter_kwargs",
        "require_dependency",
        "require_dependency_attr",
        "require_optional_dependency",
        "require_optional_attr",
    }.isdisjoint(utils.__all__)

    # Direct imports continue to resolve for Wandas internals and migration code.
    assert callable(processing.register_lazy_operation)
    assert callable(utils.filter_kwargs)
    assert callable(utils.require_optional_dependency)


def test_stable_generate_sin_uses_the_public_wrapper() -> None:
    import wandas
    from wandas.utils.generate_sample import generate_sin

    assert wandas.generate_sin is generate_sin
    assert "generate_sin" in wandas.__all__


def test_datasets_namespace_does_not_promise_assets() -> None:
    import wandas.datasets as datasets
    import wandas.datasets.sample_data as sample_data

    documentation = (REPO_ROOT / "docs/src/api/datasets.md").read_text(encoding="utf-8")
    overview = (REPO_ROOT / "docs/src/api/index.md").read_text(encoding="utf-8")

    assert datasets.__all__ == []
    assert sample_data.__all__ == []
    assert "exports no sample datasets, catalog, or packaged audio assets" in " ".join(documentation.split())
    assert "use the stable top-level `wd.generate_sin()` helper" in " ".join(documentation.split())
    assert "stableなtop-level helper `wd.generate_sin()`" in " ".join(documentation.split())
    assert "Use stable `wd.generate_sin()` for a known signal" in " ".join(overview.split())
    assert "既知信号にはstableな`wd.generate_sin()`" in " ".join(overview.split())
    assert "provides sample data for testing and demonstrations" not in overview
