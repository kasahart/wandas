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
    "librosa_effects",
    "mosqito_sq_metrics",
    "mosqito_sound_level_meter",
    "mosqito_center_freq",
    "ipython_display",
    "sklearn_base",
    "torch",
    "tensorflow",
]


@dataclass(frozen=True)
class DependencySpec:
    import_name: str
    extra: str
    display_name: str | None = None
    install_hint: str | None = None

    @property
    def error_name(self) -> str:
        return self.display_name or self.import_name


DEPENDENCY_REGISTRY: dict[DependencyKey, DependencySpec] = {
    "pandas": DependencySpec(import_name="pandas", extra="core"),
    "matplotlib_pyplot": DependencySpec(import_name="matplotlib.pyplot", extra="core"),
    "matplotlib_gridspec": DependencySpec(import_name="matplotlib.gridspec", extra="core"),
    "matplotlib_axes": DependencySpec(import_name="matplotlib.axes", extra="core"),
    "matplotlib_figure": DependencySpec(import_name="matplotlib.figure", extra="core"),
    "matplotlib_lines": DependencySpec(import_name="matplotlib.lines", extra="core"),
    "h5py": DependencySpec(import_name="h5py", extra="io"),
    "librosa": DependencySpec(import_name="librosa", extra="effects"),
    "librosa_effects": DependencySpec(import_name="librosa.effects", extra="effects"),
    "mosqito_sq_metrics": DependencySpec(import_name="mosqito.sq_metrics", extra="psychoacoustic"),
    "mosqito_sound_level_meter": DependencySpec(
        import_name="mosqito.sound_level_meter",
        extra="psychoacoustic",
    ),
    "mosqito_center_freq": DependencySpec(
        import_name="mosqito.sound_level_meter.noct_spectrum._center_freq",
        extra="psychoacoustic",
    ),
    "ipython_display": DependencySpec(
        import_name="IPython.display",
        extra="marimo",
        install_hint='pip install "wandas[marimo]"',
    ),
    "sklearn_base": DependencySpec(import_name="sklearn.base", extra="sklearn"),
    "torch": DependencySpec(import_name="torch", extra="ml"),
    "tensorflow": DependencySpec(import_name="tensorflow", extra="ml"),
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
    return spec.install_hint or _install_command(spec.extra)


def _missing_dependency_error(
    *,
    import_name: str,
    extra: str,
    feature: str,
    display_name: str | None = None,
    install_hint: str | None = None,
) -> ImportError:
    spec = DependencySpec(
        import_name=import_name,
        extra=extra,
        display_name=display_name,
        install_hint=install_hint,
    )
    return ImportError(
        f"{feature} requires {_dependency_label(spec.extra)} {spec.error_name!r}.\n"
        f"Install it with: {_install_hint(spec)}"
    )


def require_dependency(key: DependencyKey, *, feature: str) -> ModuleType:
    """Import a registered dependency or raise an actionable install error."""
    spec = DEPENDENCY_REGISTRY[key]
    try:
        return importlib.import_module(spec.import_name)
    except ModuleNotFoundError as exc:
        if not _is_requested_module_or_parent(exc.name, spec.import_name):
            raise
        raise _missing_dependency_error(
            import_name=spec.import_name,
            extra=spec.extra,
            feature=feature,
            display_name=spec.display_name,
            install_hint=spec.install_hint,
        ) from exc


def require_dependency_attr(key: DependencyKey, attr_name: str, *, feature: str) -> Any:
    """Import an attribute from a registered dependency module."""
    module = require_dependency(key, feature=feature)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        spec = DEPENDENCY_REGISTRY[key]
        raise ImportError(
            f"{feature} requires {spec.error_name!r} to provide attribute {attr_name!r}.\n"
            f"Install or update it with: {_install_hint(spec)}"
        ) from exc


def require_pandas(feature: str) -> ModuleType:
    return require_dependency("pandas", feature=feature)


def require_matplotlib_pyplot(feature: str) -> ModuleType:
    return require_dependency("matplotlib_pyplot", feature=feature)


def require_matplotlib_gridspec(feature: str) -> ModuleType:
    return require_dependency("matplotlib_gridspec", feature=feature)


def require_matplotlib_axes_type(feature: str) -> Any:
    return require_dependency_attr("matplotlib_axes", "Axes", feature=feature)


def require_matplotlib_figure_type(feature: str) -> Any:
    return require_dependency_attr("matplotlib_figure", "Figure", feature=feature)


def require_matplotlib_line2d_type(feature: str) -> Any:
    return require_dependency_attr("matplotlib_lines", "Line2D", feature=feature)


def require_h5py(feature: str) -> ModuleType:
    return require_dependency("h5py", feature=feature)


def require_librosa_effects(feature: str) -> ModuleType:
    return require_dependency("librosa_effects", feature=feature)


def require_mosqito_sound_level_meter(feature: str) -> ModuleType:
    return require_dependency("mosqito_sound_level_meter", feature=feature)


def require_mosqito_sq_metric(name: str, feature: str) -> Any:
    return require_dependency_attr("mosqito_sq_metrics", name, feature=feature)


def require_mosqito_center_freq(feature: str) -> Any:
    return require_dependency_attr("mosqito_center_freq", "_center_freq", feature=feature)


def require_ipython_display(feature: str) -> tuple[Any, Any]:
    display_module = require_dependency("ipython_display", feature=feature)
    return display_module.display, display_module.Audio


def require_optional_dependency(module_name: str, *, extra: str, feature: str) -> ModuleType:
    """Import a dependency or raise an actionable install error.

    Kept for external compatibility. Wandas internals should use
    require_dependency() so dependency-to-extra metadata stays centralized.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not _is_requested_module_or_parent(exc.name, module_name):
            raise
        raise _missing_dependency_error(
            import_name=module_name,
            extra=extra,
            feature=feature,
        ) from exc


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
