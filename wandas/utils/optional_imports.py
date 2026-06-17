from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

DependencyKey = Literal[
    "pandas",
    "h5py",
    "librosa",
    "librosa_display",
    "librosa_effects",
    "librosa_util",
    "mosqito_sq_metrics",
    "mosqito_sound_level_meter",
    "mosqito_center_freq",
    "ipython_display",
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
    "h5py": DependencySpec(module="h5py", extra="io"),
    "librosa": DependencySpec(module="librosa", extra="viz"),
    "librosa_display": DependencySpec(module="librosa.display", distribution="librosa", extra="viz"),
    "librosa_effects": DependencySpec(module="librosa.effects", distribution="librosa", extra="viz"),
    "librosa_util": DependencySpec(module="librosa.util", distribution="librosa", extra="viz"),
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
}


def _install_hint(spec: DependencySpec) -> str:
    if spec.install_hint is not None:
        return spec.install_hint
    if spec.extra == "core":
        return 'pip install "wandas"'
    return f'pip install "wandas[{spec.extra}]"'


def _missing_dependency_error(spec: DependencySpec, feature: str, error: ImportError) -> ImportError:
    return ImportError(
        f"Missing optional dependency for {feature}\n"
        f"  Module: {spec.module}\n"
        f"  Package: {spec.package_name}\n"
        f"Install it with: {_install_hint(spec)}"
    ).with_traceback(error.__traceback__)


def require_dependency(key: DependencyKey, *, feature: str) -> ModuleType:
    """Import a registered dependency or raise a Wandas install hint."""
    spec = DEPENDENCY_REGISTRY[key]
    try:
        return importlib.import_module(spec.module)
    except ImportError as error:
        raise _missing_dependency_error(spec, feature, error) from error


def require_dependency_attr(key: DependencyKey, attr_name: str, *, feature: str) -> Any:
    """Import an attribute from a registered dependency module."""
    module = require_dependency(key, feature=feature)
    try:
        return getattr(module, attr_name)
    except AttributeError as error:
        spec = DEPENDENCY_REGISTRY[key]
        raise ImportError(
            f"Dependency attribute is unavailable for {feature}\n"
            f"  Module: {spec.module}\n"
            f"  Attribute: {attr_name}\n"
            f"  Package: {spec.package_name}\n"
            f"Install it with: {_install_hint(spec)}"
        ) from error


def require_optional_dependency(module_name: str, *, extra: str, feature: str) -> ModuleType:
    """Compatibility wrapper for legacy optional import call sites."""
    spec = DependencySpec(module=module_name, extra=extra)
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise _missing_dependency_error(spec, feature, error) from error
