from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _FrozenSequence:
    items: tuple[Any, ...]


@dataclass(frozen=True)
class _FrozenMapping:
    items: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True)
class _BooleanMask:
    values: tuple[bool, ...]


def _snapshot_param_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str):
        return value
    if type(value).__module__ == "numpy" and type(value).__name__ in {"bool", "bool_"}:
        return bool(value)
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
    if isinstance(value, list | tuple):
        return _FrozenSequence(tuple(_snapshot_sequence_item(item) for item in value))
    raise TypeError(
        "OperationSpec params must be flat recipe-literal values\n"
        f"  Got: {type(value).__name__}\n"
        "  Supported values: None, bool, int, float, str, and shallow list/tuple of those values."
    )


def _snapshot_sequence_item(value: Any) -> Any:
    if isinstance(value, list | tuple | Mapping):
        raise TypeError(
            "OperationSpec params must be flat recipe-literal values\n"
            f"  Got nested value: {type(value).__name__}\n"
            "  Sequence and mapping params are intentionally shallow so equality and serialization stay predictable."
        )
    return _snapshot_param_value(value)


def _snapshot_rename_mapping_key(value: Any) -> int | str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or (type(value).__module__ == "numpy" and type(value).__name__ in {"bool", "bool_"}):
        raise TypeError(
            "rename_channels mapping keys must be int or str\n"
            f"  Got: {type(value).__name__}\n"
            "  Supported mapping keys: int and str."
        )
    if isinstance(value, numbers.Integral):
        return int(value)
    raise TypeError(
        "rename_channels mapping keys must be int or str\n"
        f"  Got: {type(value).__name__}\n"
        "  Supported mapping keys: int and str."
    )


def _snapshot_rename_mapping_value(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"rename_channels mapping values must be strings\n  Got: {type(value).__name__}")
    return value


def _snapshot_rename_channels_params(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    if set(params) != {"mapping"}:
        return _snapshot_params(params)
    mapping = params["mapping"]
    if not isinstance(mapping, Mapping):
        raise TypeError(f"rename_channels mapping must be a mapping\n  Got: {type(mapping).__name__}")
    frozen_mapping = tuple(
        (_snapshot_rename_mapping_key(key), _snapshot_rename_mapping_value(value)) for key, value in mapping.items()
    )
    return (("mapping", _FrozenMapping(frozen_mapping)),)


def _is_bool_scalar(value: Any) -> bool:
    return isinstance(value, bool) or (type(value).__module__ == "numpy" and type(value).__name__ in {"bool", "bool_"})


def _snapshot_bool_mask_param(value: Any, *, context: str) -> _FrozenSequence:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{context} must be a shallow sequence of bool values\n  Got: {type(value).__name__}")
    if not all(_is_bool_scalar(item) for item in value):
        raise TypeError(f"{context} must contain only bool values\n  Got: {value!r}")
    return _FrozenSequence(tuple(bool(item) for item in value))


def _snapshot_get_channel_param_value(key: str, value: Any) -> Any:
    if key == "channel_mask":
        return _snapshot_bool_mask_param(value, context="get_channel channel_mask")
    return _snapshot_param_value(value)


def _snapshot_get_channel_query_params(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    query = params.get("query")
    if not isinstance(query, Mapping):
        if "channel_mask" in params:
            frozen: list[tuple[str, Any]] = []
            for key, value in params.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "OperationSpec params mapping keys must be strings\n"
                        f"  Got: {type(key).__name__}\n"
                        "  Recipe params use string keys so equality and serialization stay predictable."
                    )
                frozen.append((key, _snapshot_get_channel_param_value(key, value)))
            return tuple(sorted(frozen))
        return _snapshot_params(params)
    frozen: list[tuple[str, Any]] = []
    for key, value in params.items():
        if key == "query":
            query_items: list[tuple[Any, Any]] = []
            for query_key, query_value in query.items():
                if not isinstance(query_key, str):
                    raise TypeError(f"get_channel query keys must be strings\n  Got: {type(query_key).__name__}")
                query_items.append((query_key, _snapshot_param_value(query_value)))
            frozen.append((key, _FrozenMapping(tuple(sorted(query_items)))))
        else:
            if not isinstance(key, str):
                raise TypeError(
                    "OperationSpec params mapping keys must be strings\n"
                    f"  Got: {type(key).__name__}\n"
                    "  Recipe params use string keys so equality and serialization stay predictable."
                )
            frozen.append((key, _snapshot_get_channel_param_value(key, value)))
    return tuple(sorted(frozen))


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


def _params_to_public_dict(params: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {key: _param_to_public_value(value) for key, value in params}


def _param_to_public_value(value: Any) -> Any:
    if isinstance(value, _FrozenSequence):
        return [_param_to_public_value(item) for item in value.items]
    if isinstance(value, _FrozenMapping):
        return {key: _param_to_public_value(item) for key, item in value.items}
    return value


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


restore_history_value = _restore_history_value
