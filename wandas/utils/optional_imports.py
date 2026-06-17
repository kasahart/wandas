from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

DependencyKey = Literal[
    "pandas",
    "matplotlib_pyplot",
    "matplotlib_gridspec",
    "matplotlib_axes",
    "matplotlib_figure",
    "matplotlib_lines",
    "h5py",
    "librosa",
    "librosa_display",
    "librosa_effects",
    "mosqito_sq_metrics",
    "mosqito_sound_level_meter",
    "mosqito_center_freq",
    "ipython_display",
    "torch",
    "tensorflow",
]


@dataclass(frozen=True)
class DependencySpec:
    module: str
    extra: str
    distribution: str | None = None
    project_dependency: str | None = None
    install_hint: str | None = None

    @property
    def package_name(self) -> str:
        return self.project_dependency or self.distribution or self.module.partition(".")[0]


DEPENDENCY_REGISTRY: dict[DependencyKey, DependencySpec] = {
    "pandas": DependencySpec(module="pandas", extra="core"),
    "matplotlib_pyplot": DependencySpec(
        module="matplotlib.pyplot", distribution="matplotlib", extra="core"
    ),
    "matplotlib_gridspec": DependencySpec(
        module="matplotlib.gridspec", distribution="matplotlib", extra="core"
    ),
    "matplotlib_axes": DependencySpec(module="matplotlib.axes", distribution="matplotlib", extra="core"),
    "matplotlib_figure": DependencySpec(module="matplotlib.figure", distribution="matplotlib", extra="core"),
    "matplotlib_lines": DependencySpec(module="matplotlib.lines", distribution="matplotlib", extra="core"),
    "h5py": DependencySpec(module="h5py", extra="io"),
    "librosa": DependencySpec(module="librosa", extra="viz"),
    "librosa_display": DependencySpec(module="librosa", distribution="librosa", extra="viz"),
    "librosa_effects": DependencySpec(module="librosa.effects", distribution="librosa", extra="viz"),
    "mosqito_sq_metrics": DependencySpec(
        module="mosqito.sq_metrics",
        distribution="mosqito",
        extra="psychoacoustic",
    ),
    "mosqito_sound_level_meter": DependencySpec(
        module="mosqito.sound_level_meter",
        distribution="mosqito",
        extra="psychoacoustic",
    ),
    "mosqito_center_freq": DependencySpec(
        module="mosqito.sound_level_meter.noct_spectrum._center_freq",
        distribution="mosqito",
        extra="psychoacoustic",
    ),
    "ipython_display": DependencySpec(
        module="IPython.display",
        distribution="IPython",
        project_dependency="ipykernel",
        extra="notebook",
        install_hint='pip install "wandas[notebook]"',
    ),
    "torch": DependencySpec(module="torch", extra="ml"),
    "tensorflow": DependencySpec(module="tensorflow", extra="ml"),
}


def _is_requested_module_or_parent(missing_name: str | None, module_name: str) -> bool:
    if missing_name is None:
        return False
    return module_name == missing_name or module_name.startswith(f"{missing_name}.")


def _dependency_label(extra: str) -> str:
    return "core dependency" if extra == "core" else "optional dependency"


def _install_command(extra: str) -> str:
    return 'pip install "wandas"' if extra == "core" else f'pip install "wandas[{extra}]"'


def _install_hint(spec: DependencySpec) -> str:
    if spec.install_hint is not None:
        return spec.install_hint
    return _install_command(spec.extra)


def _dependency_label_for_spec(spec: DependencySpec) -> str:
    return _dependency_label(spec.extra)


def _missing_dependency_error(
    spec: DependencySpec,
    feature: str,
    error: ModuleNotFoundError,
    *,
    dependency_name: str | None = None,
) -> ImportError:
    name = dependency_name or spec.package_name
    return ImportError(
        f"{feature} requires {_dependency_label_for_spec(spec)} {name!r}.\n"
        f"Install it with: {_install_hint(spec)}"
    )


def require_dependency(key: DependencyKey, *, feature: str) -> ModuleType:
    """Import a registered dependency or raise an actionable install error."""
    spec = DEPENDENCY_REGISTRY[key]
    try:
        return importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        if not _is_requested_module_or_parent(exc.name, spec.module):
            raise
        raise _missing_dependency_error(spec, feature, exc) from exc


def require_dependency_attr(key: DependencyKey, attr_name: str, *, feature: str) -> Any:
    """Import an attribute from a registered dependency module."""
    module = require_dependency(key, feature=feature)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        spec = DEPENDENCY_REGISTRY[key]
        raise ImportError(
            f"{feature} requires {spec.module!r} to provide attribute {attr_name!r}.\n"
            f"Install or update it with: {_install_hint(spec)}"
        ) from exc


def require_optional_dependency(module_name: str, *, extra: str, feature: str) -> ModuleType:
    """Import a dependency or raise an actionable install error.

    Kept for compatibility with external callers; new Wandas call sites should
    prefer require_dependency() so dependency metadata stays centralized.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not _is_requested_module_or_parent(exc.name, module_name):
            raise
        spec = DependencySpec(module=module_name, extra=extra)
        raise _missing_dependency_error(spec, feature, exc, dependency_name=module_name) from exc


def require_optional_attr(module_name: str, attr_name: str, *, extra: str, feature: str) -> Any:
    """Import an attribute from an optional dependency with the same error style."""
    module = require_optional_dependency(module_name, extra=extra, feature=feature)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"{feature} requires {module_name!r} to provide attribute {attr_name!r}.\n"
            f"Install or update it with: {_install_command(extra)}"
        ) from exc
