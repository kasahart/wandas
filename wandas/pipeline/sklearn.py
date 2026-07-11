from __future__ import annotations

from typing import Any

import numpy as np

from wandas.pipeline import OperationSpec
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
        pass

    class TransformerMixin:  # type: ignore[no-redef]
        pass


def _require_sklearn() -> None:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(str(_SKLEARN_IMPORT_ERROR)) from _SKLEARN_IMPORT_ERROR


class WandasOperationTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Minimal sklearn-compatible wrapper for a Wandas frame operation."""

    _operation_name: str | None = None
    _param_names: tuple[str, ...] | None = None

    def __init__(self, operation: str, **params: Any) -> None:
        _require_sklearn()
        self.operation = operation
        self._params = dict(params)

    def _resolved_operation(self) -> str:
        return self._operation_name or self.operation

    def _resolved_params(self) -> dict[str, Any]:
        if self._param_names is None:
            return dict(self._params)
        return {name: getattr(self, name) for name in self._param_names}

    def fit(self, X: Any, y: Any = None) -> WandasOperationTransformer:  # noqa: N803
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def transform(self, X: Any) -> Any:  # noqa: N803
        return X.apply_operation(self._resolved_operation(), **self._resolved_params())

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = self._resolved_params()
        if self._param_names is None:
            return {"operation": self.operation, **params}
        return params

    def set_params(self, **params: Any) -> WandasOperationTransformer:
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

    def to_spec(self) -> OperationSpec:
        return OperationSpec(self._resolved_operation(), self._resolved_params())


class HighPassFilter(WandasOperationTransformer):
    _operation_name = "highpass_filter"
    _param_names = ("cutoff", "order")

    def __init__(self, cutoff: float, order: int = 4) -> None:
        self.cutoff = cutoff
        self.order = order
        super().__init__(self._operation_name, cutoff=cutoff, order=order)


class LowPassFilter(WandasOperationTransformer):
    _operation_name = "lowpass_filter"
    _param_names = ("cutoff", "order")

    def __init__(self, cutoff: float, order: int = 4) -> None:
        self.cutoff = cutoff
        self.order = order
        super().__init__(self._operation_name, cutoff=cutoff, order=order)


class BandPassFilter(WandasOperationTransformer):
    _operation_name = "bandpass_filter"
    _param_names = ("low_cutoff", "high_cutoff", "order")

    def __init__(self, low_cutoff: float, high_cutoff: float, order: int = 4) -> None:
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order
        super().__init__(
            self._operation_name,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            order=order,
        )


class Normalize(WandasOperationTransformer):
    _operation_name = "normalize"
    _param_names = ("norm", "axis", "threshold", "fill")

    def __init__(
        self,
        norm: float | None = np.inf,
        axis: int | None = -1,
        threshold: float | None = None,
        fill: bool | None = None,
    ) -> None:
        self.norm = norm
        self.axis = axis
        self.threshold = threshold
        self.fill = fill
        super().__init__(
            self._operation_name,
            norm=norm,
            axis=axis,
            threshold=threshold,
            fill=fill,
        )


class RemoveDC(WandasOperationTransformer):
    _operation_name = "remove_dc"
    _param_names: tuple[str, ...] = ()

    def __init__(self) -> None:
        super().__init__(self._operation_name)


__all__ = [
    "BandPassFilter",
    "HighPassFilter",
    "LowPassFilter",
    "Normalize",
    "RemoveDC",
    "WandasOperationTransformer",
]
