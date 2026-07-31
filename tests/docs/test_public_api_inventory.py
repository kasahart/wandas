from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest

from wandas._public_api import (
    CLASSIFICATIONS,
    DEPRECATED_COMPATIBILITY,
    PRIVATE_INTERNAL,
    PUBLIC_API_INVENTORY,
    STABLE_PUBLIC,
    ApiSymbol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_EXPORT_MODULES = (
    "wandas",
    "wandas.frames",
    "wandas.frames.mixins",
    "wandas.processing",
    "wandas.utils",
    "wandas.datasets",
    "wandas.datasets.sample_data",
)


def _wandas_api_candidates(module: ModuleType) -> set[str]:
    """Return non-private Wandas callables/classes visible on a package module."""

    candidates: set[str] = set()
    if module.__name__ == "wandas" and hasattr(module, "__version__"):
        candidates.add("__version__")
    for name, value in vars(module).items():
        if name.startswith("_") or not callable(value):
            continue
        owner = getattr(value, "__module__", "")
        if isinstance(owner, str) and (owner == "wandas" or owner.startswith("wandas.")):
            candidates.add(name)
    if module.__name__ == "wandas.processing":
        lazy_operations = vars(module).get("_LAZY_OPERATION_CLASSES", {})
        candidates.update(lazy_operations)
    return candidates


def _inventory_errors(
    inventory: Mapping[str, tuple[ApiSymbol, ...]],
) -> tuple[str, ...]:
    errors: list[str] = []
    for module_name in PACKAGE_EXPORT_MODULES:
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
            if symbol.classification not in CLASSIFICATIONS:
                errors.append(f"{qualified_name}: unknown classification")
            if symbol.classification == PRIVATE_INTERNAL and symbol.in_all:
                errors.append(f"{qualified_name}: private symbol is in __all__")
            if symbol.classification == DEPRECATED_COMPATIBILITY:
                if not symbol.replacement:
                    errors.append(f"{qualified_name}: deprecated symbol has no replacement")
                if not symbol.support:
                    errors.append(f"{qualified_name}: deprecated symbol has no support window")
            if not hasattr(module, symbol.name):
                errors.append(f"{qualified_name}: inventory symbol is not importable")
            if symbol.documentation is not None:
                document = REPO_ROOT / symbol.documentation
                if not document.is_file():
                    errors.append(f"{qualified_name}: missing documentation file")
                elif symbol.name not in document.read_text(encoding="utf-8"):
                    errors.append(f"{qualified_name}: missing from {symbol.documentation}")
    return tuple(errors)


def test_canonical_inventory_matches_exports_and_api_documentation() -> None:
    assert _inventory_errors(PUBLIC_API_INVENTORY) == ()


def test_inventory_is_structurally_immutable() -> None:
    import wandas._public_api as inventory_module

    assert not hasattr(inventory_module, "_INVENTORY")

    with pytest.raises(TypeError):
        PUBLIC_API_INVENTORY["wandas"] = ()  # ty: ignore[invalid-assignment]

    with pytest.raises(AttributeError):
        PUBLIC_API_INVENTORY["wandas"].append(  # ty: ignore[unresolved-attribute]
            ApiSymbol("drift", PRIVATE_INTERNAL, False)
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


def test_internal_registry_and_utils_helpers_stay_outside_all() -> None:
    import wandas.processing as processing
    import wandas.utils as utils

    assert {
        "_OPERATION_MODULES",
        "_OPERATION_REGISTRY",
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
