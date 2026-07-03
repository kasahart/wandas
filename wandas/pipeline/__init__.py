from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast


class _SupportsApplyOperation(Protocol):
    def apply_operation(self, operation_name: str, **params: Any) -> Any: ...


T_Frame = TypeVar("T_Frame", bound=_SupportsApplyOperation)


class RecipeExtractionError(ValueError):
    """Raised when a frame lineage cannot be represented by RecipeSpec."""


_REPLAYABLE_APPLY_OPERATIONS = frozenset(
    {
        "bandpass_filter",
        "highpass_filter",
        "lowpass_filter",
        "normalize",
        "remove_dc",
        "resampling",
        "trim",
    }
)


def _snapshot_param_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        frozen_float = float(value)
        if math.isnan(frozen_float):
            raise TypeError(
                "OperationSpec params must not contain NaN\n"
                "  NaN does not compare equal to itself, so recipe equality becomes unstable."
            )
        return frozen_float
    raise TypeError(
        "OperationSpec params must be flat recipe-literal values\n"
        f"  Got: {type(value).__name__}\n"
        "  Supported values: None, bool, int, float, and str."
    )


def _snapshot_params(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    frozen: list[tuple[str, Any]] = []
    for key, value in params.items():
        if not isinstance(key, str):
            raise TypeError(
                "OperationSpec params mapping keys must be strings\n"
                f"  Got: {type(key).__name__}\n"
                "  Recipe params use string keys so equality and serialization stay predictable."
            )
        frozen.append((key, _snapshot_param_value(value)))
    return tuple(sorted(frozen))


def _restore_history_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"type", "value"} and value.get("type") == "float":
            float_value = value["value"]
            if float_value == "inf":
                return float("inf")
            if float_value == "-inf":
                return float("-inf")
            if float_value == "nan":
                return float("nan")
        return {key: _restore_history_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_history_value(item) for item in value]
    return value


def _validate_replayable_operation(operation: str) -> None:
    try:
        from wandas.processing import get_operation

        operation_class = get_operation(operation)
    except ValueError as exc:
        raise RecipeExtractionError(
            "Operation is outside the Stage 1 recipe allowlist\n"
            f"  Operation: {operation}\n"
            "  Current RecipeSpec can only replay selected single-input Wandas operations. "
            "Graph, method, domain-transition, and callable recipes require the next recipe model."
        ) from exc

    expected_input_count = getattr(operation_class, "_expected_input_count", 1)
    if isinstance(expected_input_count, int) and expected_input_count != 1:
        raise RecipeExtractionError(
            "Graph operation requires graph recipe support\n"
            f"  Operation: {operation}\n"
            f"  Runtime inputs: {expected_input_count}\n"
            "  Current RecipeSpec can only replay single-input linear operations."
        )
    if operation not in _REPLAYABLE_APPLY_OPERATIONS:
        raise RecipeExtractionError(
            "Operation is outside the Stage 1 recipe allowlist\n"
            f"  Operation: {operation}\n"
            "  Current RecipeSpec can only replay selected single-input Wandas operations. "
            "Graph, method, domain-transition, and callable recipes require the next recipe model."
        )


def _steps_from_graph(graph: Mapping[str, Any]) -> tuple[OperationSpec, ...]:
    operation = str(graph["operation"])
    inputs = tuple(graph.get("inputs", ()))
    if len(inputs) > 1:
        raise RecipeExtractionError(
            "Graph operation requires graph recipe support\n"
            f"  Operation: {operation}\n"
            f"  Parent count: {len(inputs)}\n"
            "  Current RecipeSpec can only replay one linear parent chain."
        )
    _validate_replayable_operation(operation)

    params = cast(Mapping[str, Any], _restore_history_value(graph.get("params", {})))
    parent_steps = _steps_from_graph(cast(Mapping[str, Any], inputs[0])) if inputs else ()
    return (*parent_steps, OperationSpec(operation, params))


@dataclass(frozen=True, init=False)
class OperationSpec:
    """Replayable single-frame operation call."""

    operation: str
    _params: tuple[tuple[str, Any], ...]

    def __init__(self, operation: str, params: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "_params", _snapshot_params(params or {}))

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "params": self.params}


@dataclass(frozen=True, init=False)
class RecipeSpec:
    """Serial recipe of replayable Wandas frame operations."""

    steps: tuple[OperationSpec, ...]

    def __init__(self, steps: Iterable[OperationSpec]) -> None:
        object.__setattr__(self, "steps", tuple(steps))

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_frame(cls, frame: Any) -> RecipeSpec:
        from wandas.core.base_frame import BaseFrame

        if not isinstance(frame, BaseFrame):
            raise RecipeExtractionError(
                "Recipe extraction requires a Wandas frame\n"
                f"  Got: {type(frame).__name__}\n"
                "  Pass a processed Wandas frame with operation_graph lineage."
            )
        graph = frame.operation_graph
        if graph is None:
            return cls(())
        return cls(_steps_from_graph(cast(Mapping[str, Any], graph)))

    def apply(self, frame: T_Frame) -> T_Frame:
        result: Any = frame
        for step in self.steps:
            result = result.apply_operation(step.operation, **step.params)
        return cast(T_Frame, result)


__all__ = ["OperationSpec", "RecipeExtractionError", "RecipeSpec"]
