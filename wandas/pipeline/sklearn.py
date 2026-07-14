"""Scikit-learn-compatible adapters for declared Wandas Frame operations."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from wandas.pipeline.decorators import recipe_definition
from wandas.utils.optional_imports import require_dependency_attr

_SKLEARN_IMPORT_ERROR: ImportError | None = None

try:
    BaseEstimator = require_dependency_attr(
        "sklearn_base",
        "BaseEstimator",
        feature="Wandas sklearn transformers",
    )
    TransformerMixin = require_dependency_attr(
        "sklearn_base",
        "TransformerMixin",
        feature="Wandas sklearn transformers",
    )
except ImportError as exc:
    _SKLEARN_IMPORT_ERROR = exc

    class BaseEstimator:  # type: ignore[no-redef]
        """Import-time placeholder used only to define unavailable adapters."""

        pass

    class TransformerMixin:  # type: ignore[no-redef]
        """Import-time placeholder used only to define unavailable adapters."""

        pass


def _require_sklearn() -> None:
    """Raise the captured optional-dependency error when sklearn is unavailable."""
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(str(_SKLEARN_IMPORT_ERROR)) from _SKLEARN_IMPORT_ERROR


class WandasOperationTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Stateless sklearn transformer for a declared Wandas Frame operation.

    Args:
        operation: Public Frame method name decorated with ``@recipe_operation``.
        **params: Keyword arguments forwarded to that method by :meth:`transform`.

    Raises:
        ImportError: If scikit-learn is not installed.
    """

    _param_names: tuple[str, ...] | None = None

    def __init__(self, operation: str, **params: Any) -> None:
        """Store operation identity and estimator parameters without fitting state."""
        _require_sklearn()
        self.operation = operation
        if self._param_names is None:
            self._params = dict(params)

    def _resolved_params(self) -> dict[str, Any]:
        """Return current estimator parameters for the wrapped Frame call."""
        if self._param_names is None:
            return dict(self._params)
        return {name: getattr(self, name) for name in self._param_names}

    def fit(self, X: Any, y: Any = None) -> WandasOperationTransformer:  # noqa: N803
        """Return this stateless transformer without inspecting training data.

        Args:
            X: Accepted for sklearn estimator compatibility and left untouched.
            y: Optional target accepted for sklearn estimator compatibility.

        Returns:
            This already-fitted stateless transformer.
        """
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Report that this stateless transformer requires no fitted attributes."""
        return True

    def transform(self, X: Any) -> Any:  # noqa: N803
        """Apply the declared public operation to a Wandas Frame.

        Args:
            X: Frame exposing the configured declared public operation.

        Returns:
            The Frame result returned by that operation.

        Raises:
            ValueError: If ``operation`` does not name a declared Recipe-capable
                public method on ``X``.
        """
        declaration = inspect.getattr_static(X, self.operation, None)
        try:
            recipe_definition(declaration)
        except TypeError as exc:
            raise ValueError(
                f"operation must name a declared public Recipe Frame method, got {self.operation!r}"
            ) from exc
        method = getattr(X, self.operation)
        return method(**self._resolved_params())

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return estimator parameters using sklearn's cloning contract.

        Args:
            deep: Accepted for sklearn compatibility; this adapter has no nested
                estimators.

        Returns:
            A fresh parameter dictionary.
        """
        params = self._resolved_params()
        if self._param_names is None:
            return {"operation": self.operation, **params}
        return params

    def set_params(self, **params: Any) -> WandasOperationTransformer:
        """Update estimator parameters in place using sklearn conventions.

        Args:
            **params: Parameter values to update.

        Returns:
            This transformer.

        Raises:
            TypeError: If a generic ``operation`` update is not a string.
            ValueError: If a parameter is invalid for the current adapter configuration.
        """
        if self._param_names is None and "operation" in params:
            operation = params.pop("operation")
            if not isinstance(operation, str):
                raise TypeError(f"operation must be a string, got {type(operation).__name__}")
            self.operation = operation
            self._params = dict(params)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                valid_names = ", ".join(sorted(valid_params))
                raise ValueError(f"Invalid parameter {key!r}. Valid parameters are: {valid_names}")
            if self._param_names is None:
                self._params[key] = value
            else:
                setattr(self, key, value)
        return self


class HighPassFilter(WandasOperationTransformer):
    """Stateless sklearn adapter for ``Frame.high_pass_filter``.

    Args:
        cutoff: High-pass cutoff frequency in hertz.
        order: Filter order.
    """

    _param_names = ("cutoff", "order")

    def __init__(self, cutoff: float, order: int = 4) -> None:
        """Initialize high-pass filter parameters."""
        self.cutoff = cutoff
        self.order = order
        super().__init__("high_pass_filter")


class LowPassFilter(WandasOperationTransformer):
    """Stateless sklearn adapter for ``Frame.low_pass_filter``.

    Args:
        cutoff: Low-pass cutoff frequency in hertz.
        order: Filter order.
    """

    _param_names = ("cutoff", "order")

    def __init__(self, cutoff: float, order: int = 4) -> None:
        """Initialize low-pass filter parameters."""
        self.cutoff = cutoff
        self.order = order
        super().__init__("low_pass_filter")


class BandPassFilter(WandasOperationTransformer):
    """Stateless sklearn adapter for ``Frame.band_pass_filter``.

    Args:
        low_cutoff: Lower cutoff frequency in hertz.
        high_cutoff: Upper cutoff frequency in hertz.
        order: Filter order.
    """

    _param_names = ("low_cutoff", "high_cutoff", "order")

    def __init__(self, low_cutoff: float, high_cutoff: float, order: int = 4) -> None:
        """Initialize band-pass filter parameters."""
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order
        super().__init__("band_pass_filter")


class Normalize(WandasOperationTransformer):
    """Stateless sklearn adapter for ``Frame.normalize``.

    Args:
        norm: Norm used for amplitude normalization.
        axis: Axis normalized independently, or ``None`` for global normalization.
        threshold: Optional magnitude threshold treated as zero.
        fill: Optional zero-norm fill behavior forwarded to the Frame operation.
    """

    _param_names = ("norm", "axis", "threshold", "fill")

    def __init__(
        self,
        norm: float | None = np.inf,
        axis: int | None = -1,
        threshold: float | None = None,
        fill: bool | None = None,
    ) -> None:
        """Initialize normalization parameters."""
        self.norm = norm
        self.axis = axis
        self.threshold = threshold
        self.fill = fill
        super().__init__("normalize")


class RemoveDC(WandasOperationTransformer):
    """Stateless sklearn adapter for ``Frame.remove_dc``."""

    _param_names: tuple[str, ...] = ()

    def __init__(self) -> None:
        """Initialize the parameter-free DC-removal transformer."""
        super().__init__("remove_dc")


__all__ = [
    "BandPassFilter",
    "HighPassFilter",
    "LowPassFilter",
    "Normalize",
    "RemoveDC",
    "WandasOperationTransformer",
]
