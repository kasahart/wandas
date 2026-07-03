from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast


class RecipeExtractionError(ValueError):
    """Raised when a frame lineage cannot be represented by RecipeSpec."""


@dataclass(frozen=True)
class _FrozenSequence:
    items: tuple[Any, ...]


@dataclass(frozen=True)
class _FrozenMapping:
    items: tuple[tuple[Any, Any], ...]


_REPLAYABLE_APPLY_OPERATIONS = frozenset(
    {
        "a_weighting",
        "abs",
        "bandpass_filter",
        "fade",
        "hpss_harmonic",
        "hpss_percussive",
        "highpass_filter",
        "lowpass_filter",
        "loudness_zwtv",
        "normalize",
        "power",
        "remove_dc",
        "roughness_dw",
        "rms_trend",
        "resampling",
        "sharpness_din",
        "sound_level",
        "trim",
    }
)
_REPLAYABLE_METHOD_OPERATIONS = {
    "channel_difference": ("channel_difference", {"other_channel": "other_channel"}),
    "fix_length": ("fix_length", {"target_length": "length"}),
    "get_channel": (
        "get_channel",
        {
            "channel_idx": "channel_idx",
            "query": "query",
            "validate_query_keys": "validate_query_keys",
        },
    ),
    "mean": ("mean", {}),
    "remove_channel": ("remove_channel", {"key": "key"}),
    "rename_channels": ("rename_channels", {"mapping_items": "mapping"}),
    "sum": ("sum", {}),
}
_REPLAYABLE_METHOD_NAMES = frozenset(method for method, _param_names in _REPLAYABLE_METHOD_OPERATIONS.values())
_REPLAYABLE_TYPED_METHOD_OPERATIONS = {
    "coherence": ("coherence", None),
    "csd": ("csd", None),
    "fft": ("fft", None),
    "get_frame_at": ("get_frame_at", {"time_idx": "time_idx"}),
    "ifft": ("ifft", {}),
    "istft": ("istft", {}),
    "noct_spectrum": ("noct_spectrum", None),
    "noct_synthesis": ("noct_synthesis", None),
    "roughness_dw_spec": ("roughness_dw_spec", None),
    "stft": ("stft", None),
    "transfer_function": ("transfer_function", None),
    "welch": (
        "welch",
        {
            "n_fft": "n_fft",
            "hop_length": "hop_length",
            "win_length": "win_length",
            "window": "window",
            "average": "average",
        },
    ),
}
_REPLAYABLE_TYPED_METHOD_NAMES = frozenset(
    method for method, _param_names in _REPLAYABLE_TYPED_METHOD_OPERATIONS.values()
)
_REPLAYABLE_SCALAR_OPERATIONS = frozenset({"+", "-", "*", "/", "**"})
_REPLAYABLE_GETITEM_INDEXING = frozenset({"channel_slice", "label_list", "multidimensional_slice"})


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


def _method_params(params: Mapping[str, Any], param_names: Mapping[str, str] | None) -> dict[str, Any]:
    if param_names is None:
        return dict(params)
    return {param_names[key]: value for key, value in params.items() if key in param_names}


def _method_step_from_graph(operation: str, params: Mapping[str, Any]) -> MethodStep:
    if operation == "get_channel" and "query_kind" in params:
        raise RecipeExtractionError(
            "Channel selection recipe extraction only supports explicit indices or exact label queries\n"
            f"  Query kind: {params['query_kind']!r}\n"
            "  Callable, regex, and dict channel queries need a selection recipe model that can preserve query intent."
        )
    if operation == "rename_channels":
        return MethodStep("rename_channels", {"mapping": _rename_mapping_from_params(params)})
    method, param_names = _REPLAYABLE_METHOD_OPERATIONS[operation]
    return MethodStep(method, _method_params(params, param_names))


def _rename_mapping_from_params(params: Mapping[str, Any]) -> dict[int | str, str]:
    raw_items = params.get("mapping_items")
    if not isinstance(raw_items, list | tuple):
        raise RecipeExtractionError(
            f"rename_channels recipe extraction requires typed mapping items\n  Got: {type(raw_items).__name__}"
        )
    mapping: dict[int | str, str] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, list | tuple) or len(raw_item) != 2:
            raise RecipeExtractionError(
                f"rename_channels recipe extraction requires key/value mapping items\n  Got: {raw_item!r}"
            )
        key, value = raw_item
        if not isinstance(key, int | str) or not isinstance(value, str):
            raise RecipeExtractionError(
                "rename_channels recipe extraction only supports int/str keys and str labels\n"
                f"  Got key/value types: {type(key).__name__}/{type(value).__name__}"
            )
        mapping[key] = value
    return mapping


def _typed_method_step_from_graph(operation: str, params: Mapping[str, Any], kind: str | None) -> TypedMethodStep:
    if operation == "welch" and params.get("detrend", "constant") != "constant":
        raise RecipeExtractionError(
            "Welch recipe extraction only supports public welch parameters\n"
            f"  Operation detrend: {params.get('detrend')!r}\n"
            "  ChannelFrame.welch() does not expose detrend, so non-default values cannot be replayed safely."
        )
    if kind != "method":
        raise RecipeExtractionError(
            "Typed operation requires frame method lineage\n"
            f"  Operation: {operation}\n"
            "  Call the typed frame method, such as frame.fft(), instead of generic apply_operation(). "
            "Recipe replay delegates typed domain transitions to frame methods so output frame metadata stays correct."
        )
    method, param_names = _REPLAYABLE_TYPED_METHOD_OPERATIONS[operation]
    return TypedMethodStep(method, _method_params(params, param_names))


def _scalar_operand_from_params(operation: str, params: Mapping[str, Any]) -> int | float:
    if params.get("operand_kind") != "operand":
        raise RecipeExtractionError(
            "Graph operation requires graph recipe support\n"
            f"  Operation: {operation}\n"
            "  ScalarOperationStep can only replay a single numeric operand stored in the operation graph."
        )

    operand = params.get("operand")
    if isinstance(operand, int | float) and not isinstance(operand, bool):
        return operand
    if not isinstance(operand, Mapping) or set(operand) != {"type", "value"}:
        raise RecipeExtractionError(
            f"Scalar operation requires a numeric scalar operand\n  Operation: {operation}\n  Operand: {operand!r}"
        )

    operand_type = operand["type"]
    value = operand["value"]
    if operand_type == "bool" or isinstance(value, bool):
        raise RecipeExtractionError(
            "Scalar operation requires a numeric scalar operand\n"
            f"  Operation: {operation}\n"
            f"  Operand type: {operand_type!r}"
        )
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    raise RecipeExtractionError(
        "Scalar operation requires a numeric scalar operand\n"
        f"  Operation: {operation}\n"
        f"  Operand type: {operand_type!r}"
    )


def _scalar_step_from_graph(operation: str, params: Mapping[str, Any]) -> ScalarOperationStep:
    symbol = params.get("symbol", operation)
    if symbol != operation:
        raise RecipeExtractionError(
            f"Scalar operation graph has inconsistent operator metadata\n  Operation: {operation}\n  Symbol: {symbol!r}"
        )
    try:
        return ScalarOperationStep(operation, _scalar_operand_from_params(operation, params))
    except TypeError as exc:
        raise RecipeExtractionError(
            "Scalar operation requires a stable numeric scalar operand\n"
            f"  Operation: {operation}\n"
            "  NaN operands are not replayable because recipe equality must remain stable."
        ) from exc


def _optional_int_from_params(params: Mapping[str, Any], key: str) -> int | None:
    if key not in params:
        raise RecipeExtractionError(
            f"Channel slice recipe extraction requires explicit slice params\n  Missing parameter: {key}"
        )
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise RecipeExtractionError(
            "Channel slice recipe extraction requires integer slice bounds\n"
            f"  Parameter: {key}\n"
            f"  Got: {type(value).__name__}"
        )
    return int(value)


def _slice_from_serialized(value: Any, *, context: str) -> slice:
    if not isinstance(value, Mapping):
        raise RecipeExtractionError(f"{context} requires serialized slice objects\n  Got: {type(value).__name__}")
    if not {"start", "stop", "step"}.issubset(value):
        raise RecipeExtractionError(f"{context} requires explicit start/stop/step slice params\n  Got: {value!r}")
    return slice(
        _optional_int_from_params(value, "start"),
        _optional_int_from_params(value, "stop"),
        _optional_int_from_params(value, "step"),
    )


def _axis_slices_from_params(params: Mapping[str, Any]) -> tuple[slice, ...]:
    raw_axis_slices = params.get("axis_slices")
    if not isinstance(raw_axis_slices, list | tuple) or not raw_axis_slices:
        raise RecipeExtractionError(
            f"Multidimensional slice recipe extraction requires non-empty axis_slices\n  Got: {raw_axis_slices!r}"
        )
    return tuple(
        _slice_from_serialized(raw_axis_slice, context="Multidimensional slice recipe extraction")
        for raw_axis_slice in raw_axis_slices
    )


def _channel_key_from_getitem_params(params: Mapping[str, Any]) -> slice | list[str]:
    indexing = params.get("indexing")
    if indexing == "channel_slice":
        return slice(
            _optional_int_from_params(params, "start"),
            _optional_int_from_params(params, "stop"),
            _optional_int_from_params(params, "step"),
        )
    if indexing == "label_list":
        labels = params.get("labels")
        if isinstance(labels, list | tuple) and all(isinstance(label, str) for label in labels):
            return list(labels)
    raise RecipeExtractionError(
        "Multidimensional indexing recipe extraction only supports slice or label-list channel selectors\n"
        f"  Parent indexing kind: {indexing!r}"
    )


def _channel_key_from_parent_graph(parent: Mapping[str, Any]) -> int | slice | list[str]:
    operation = str(parent["operation"])
    params = cast(Mapping[str, Any], _restore_history_value(parent.get("params", {})))
    if operation == "get_channel":
        channel_idx = params.get("channel_idx")
        if isinstance(channel_idx, numbers.Integral) and not isinstance(channel_idx, bool):
            return int(channel_idx)
        raise RecipeExtractionError(
            "Multidimensional indexing recipe extraction only supports single integer get_channel parents\n"
            f"  Parent params: {params!r}"
        )
    if operation == "__getitem__":
        return _channel_key_from_getitem_params(params)
    raise RecipeExtractionError(
        "Multidimensional indexing recipe extraction requires a replayable channel-selection parent\n"
        f"  Parent operation: {operation}"
    )


def _getitem_step_from_graph(params: Mapping[str, Any]) -> IndexingStep:
    indexing = params.get("indexing")
    if indexing not in _REPLAYABLE_GETITEM_INDEXING:
        raise RecipeExtractionError(
            "Indexing recipe extraction only supports channel-only slice, label list, "
            "and multidimensional slice selection\n"
            f"  Indexing kind: {indexing!r}\n"
            "  Multidimensional, callable, regex, dict, and array indexing need a selection recipe model "
            "that can preserve full indexing intent."
        )
    if indexing == "channel_slice":
        return IndexingStep(
            slice(
                _optional_int_from_params(params, "start"),
                _optional_int_from_params(params, "stop"),
                _optional_int_from_params(params, "step"),
            )
        )
    if indexing == "multidimensional_slice":
        return IndexingStep((slice(None), *_axis_slices_from_params(params)))
    labels = params.get("labels")
    if not isinstance(labels, list | tuple) or not all(isinstance(label, str) for label in labels):
        raise RecipeExtractionError(f"Label-list indexing recipe extraction requires string labels\n  Got: {labels!r}")
    return IndexingStep(list(labels))


def _multidimensional_steps_from_graph(
    params: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> tuple[RecipeStep, ...]:
    axis_slices = _axis_slices_from_params(params)
    parent_inputs = tuple(parent.get("inputs", ()))
    if len(parent_inputs) > 1:
        raise RecipeExtractionError(
            "Graph operation requires graph recipe support\n"
            f"  Operation: {parent['operation']}\n"
            f"  Parent count: {len(parent_inputs)}\n"
            "  Current RecipeSpec can only replay one linear parent chain."
        )
    grandparent_steps = _steps_from_graph(cast(Mapping[str, Any], parent_inputs[0])) if parent_inputs else ()
    return (*grandparent_steps, IndexingStep((_channel_key_from_parent_graph(parent), *axis_slices)))


def _step_from_graph(operation: str, params: Mapping[str, Any], kind: str | None) -> RecipeStep:
    if operation == "__getitem__":
        return _getitem_step_from_graph(params)
    if operation in _REPLAYABLE_METHOD_OPERATIONS:
        return _method_step_from_graph(operation, params)
    if operation in _REPLAYABLE_TYPED_METHOD_OPERATIONS:
        return _typed_method_step_from_graph(operation, params, kind)
    if operation in _REPLAYABLE_SCALAR_OPERATIONS:
        return _scalar_step_from_graph(operation, params)
    _validate_replayable_operation(operation)
    return OperationSpec(operation, params)


def _steps_from_graph(graph: Mapping[str, Any]) -> tuple[RecipeStep, ...]:
    operation = str(graph["operation"])
    kind = cast(str | None, graph.get("kind"))
    inputs = tuple(graph.get("inputs", ()))
    if len(inputs) > 1:
        raise RecipeExtractionError(
            "Graph operation requires graph recipe support\n"
            f"  Operation: {operation}\n"
            f"  Parent count: {len(inputs)}\n"
            "  Current RecipeSpec can only replay one linear parent chain."
        )

    params = cast(Mapping[str, Any], _restore_history_value(graph.get("params", {})))
    if operation == "__getitem__" and params.get("indexing") == "multidimensional_slice":
        if len(inputs) != 1:
            raise RecipeExtractionError(
                "Multidimensional indexing recipe extraction requires one channel-selection parent\n"
                f"  Parent count: {len(inputs)}"
            )
        return _multidimensional_steps_from_graph(params, cast(Mapping[str, Any], inputs[0]))
    parent_steps = _steps_from_graph(cast(Mapping[str, Any], inputs[0])) if inputs else ()
    return (*parent_steps, _step_from_graph(operation, params, kind))


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
        return _params_to_public_dict(self._params)

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "params": _params_to_public_dict(self._params)}


@dataclass(frozen=True, init=False)
class MethodStep:
    """Replayable frame method call."""

    method: str
    _params: tuple[tuple[str, Any], ...]

    def __init__(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        if method not in _REPLAYABLE_METHOD_NAMES:
            valid_methods = ", ".join(sorted(_REPLAYABLE_METHOD_NAMES))
            raise ValueError(
                "MethodStep method is outside the replayable method allowlist\n"
                f"  Method: {method}\n"
                f"  Valid methods: {valid_methods}"
            )
        object.__setattr__(self, "method", method)
        frozen_params = (
            _snapshot_rename_channels_params(params or {})
            if method == "rename_channels"
            else _snapshot_params(params or {})
        )
        object.__setattr__(self, "_params", frozen_params)

    @property
    def params(self) -> dict[str, Any]:
        return _params_to_public_dict(self._params)

    def to_dict(self) -> dict[str, Any]:
        params = _params_to_public_dict(self._params)
        if self.method == "rename_channels" and isinstance(params.get("mapping"), Mapping):
            return {
                "method": self.method,
                "params": {"mapping_items": [[key, value] for key, value in params["mapping"].items()]},
            }
        return {"method": self.method, "params": params}

    def apply(self, frame: Any) -> Any:
        return getattr(frame, self.method)(**self.params)


@dataclass(frozen=True, init=False)
class TypedMethodStep:
    """Replayable frame method call that may change the frame type."""

    method: str
    _params: tuple[tuple[str, Any], ...]

    def __init__(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        if method not in _REPLAYABLE_TYPED_METHOD_NAMES:
            valid_methods = ", ".join(sorted(_REPLAYABLE_TYPED_METHOD_NAMES))
            raise ValueError(
                "TypedMethodStep method is outside the replayable typed-method allowlist\n"
                f"  Method: {method}\n"
                f"  Valid methods: {valid_methods}"
            )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "_params", _snapshot_params(params or {}))

    @property
    def params(self) -> dict[str, Any]:
        return _params_to_public_dict(self._params)

    def to_dict(self) -> dict[str, Any]:
        return {"typed_method": self.method, "params": _params_to_public_dict(self._params)}

    def apply(self, frame: Any) -> Any:
        return getattr(frame, self.method)(**self.params)


@dataclass(frozen=True, init=False)
class ScalarOperationStep:
    """Replayable frame operation with a single numeric scalar operand."""

    symbol: str
    operand: int | float

    def __init__(self, symbol: str, operand: int | float) -> None:
        if symbol not in _REPLAYABLE_SCALAR_OPERATIONS:
            valid_operations = ", ".join(sorted(_REPLAYABLE_SCALAR_OPERATIONS))
            raise ValueError(
                "ScalarOperationStep operation is outside the replayable scalar allowlist\n"
                f"  Operation: {symbol}\n"
                f"  Valid operations: {valid_operations}"
            )
        if isinstance(operand, bool) or not isinstance(operand, int | float):
            raise TypeError(f"ScalarOperationStep operand must be an int or float\n  Got: {type(operand).__name__}")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "operand", _snapshot_param_value(operand))

    def to_dict(self) -> dict[str, Any]:
        return {"scalar_operation": self.symbol, "operand": self.operand}

    def apply(self, frame: Any) -> Any:
        if self.symbol == "+":
            return frame + self.operand
        if self.symbol == "-":
            return frame - self.operand
        if self.symbol == "*":
            return frame * self.operand
        if self.symbol == "/":
            return frame / self.operand
        if self.symbol == "**":
            return frame**self.operand
        raise AssertionError(f"Unhandled scalar operation: {self.symbol}")


@dataclass(frozen=True, init=False)
class IndexingStep:
    """Replayable channel-only indexing call."""

    _key: slice | tuple[str, ...] | tuple[int | slice | tuple[str, ...], ...]

    def __init__(self, key: slice | list[str] | tuple[str, ...] | tuple[int | slice | list[str], ...]) -> None:
        if isinstance(key, slice):
            object.__setattr__(
                self,
                "_key",
                self._snapshot_slice(key),
            )
            return
        if self._is_multidimensional_key(key):
            object.__setattr__(self, "_key", self._snapshot_multidimensional_key(cast(tuple[Any, ...], key)))
            return
        if not isinstance(key, list | tuple) or not key or not all(isinstance(label, str) for label in key):
            raise TypeError(
                f"IndexingStep key must be a channel slice or non-empty label list\n  Got: {type(key).__name__}"
            )
        object.__setattr__(self, "_key", tuple(key))

    @staticmethod
    def _is_multidimensional_key(key: object) -> bool:
        return isinstance(key, tuple) and len(key) >= 2 and all(isinstance(item, slice) for item in key[1:])

    @classmethod
    def _snapshot_multidimensional_key(cls, key: tuple[Any, ...]) -> tuple[int | slice | tuple[str, ...], ...]:
        channel_key = key[0]
        if isinstance(channel_key, slice):
            frozen_channel_key: int | slice | tuple[str, ...] = cls._snapshot_slice(channel_key)
        elif isinstance(channel_key, numbers.Integral) and not isinstance(channel_key, bool):
            frozen_channel_key = int(channel_key)
        elif isinstance(channel_key, list) and channel_key and all(isinstance(label, str) for label in channel_key):
            frozen_channel_key = tuple(channel_key)
        else:
            raise TypeError(
                "IndexingStep multidimensional channel key must be an int, slice, or non-empty label list\n"
                f"  Got: {type(channel_key).__name__}"
            )
        return (frozen_channel_key, *(cls._snapshot_slice(axis_slice) for axis_slice in key[1:]))

    @classmethod
    def _snapshot_slice(cls, key: slice) -> slice:
        return slice(
            cls._snapshot_slice_value(key.start, "start"),
            cls._snapshot_slice_value(key.stop, "stop"),
            cls._snapshot_slice_value(key.step, "step"),
        )

    @staticmethod
    def _snapshot_slice_value(value: Any, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(
                "IndexingStep slice bounds must be integers or None\n"
                f"  Parameter: {name}\n"
                f"  Got: {type(value).__name__}"
            )
        return int(value)

    @property
    def key(self) -> slice | list[str] | tuple[int | slice | list[str], ...]:
        if isinstance(self._key, slice):
            return self._key
        if self._is_multidimensional_key(self._key):
            channel_key = self._key[0]
            public_channel_key: int | slice | list[str]
            if isinstance(channel_key, tuple):
                public_channel_key = list(channel_key)
            else:
                public_channel_key = cast(int | slice, channel_key)
            return cast(tuple[int | slice | list[str], ...], (public_channel_key, *self._key[1:]))
        return list(self._key)

    @staticmethod
    def _slice_to_dict(key: slice) -> dict[str, int | None]:
        return {"start": key.start, "stop": key.stop, "step": key.step}

    @classmethod
    def _channel_key_to_dict(cls, key: int | slice | tuple[str, ...]) -> dict[str, Any]:
        if isinstance(key, slice):
            return {"type": "slice", **cls._slice_to_dict(key)}
        if isinstance(key, int):
            return {"type": "index", "value": key}
        return {"type": "label_list", "labels": list(key)}

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self._key, slice):
            return {
                "getitem": {
                    "type": "channel_slice",
                    **self._slice_to_dict(self._key),
                }
            }
        if self._is_multidimensional_key(self._key):
            return {
                "getitem": {
                    "type": "multidimensional_slice",
                    "channel": self._channel_key_to_dict(cast(int | slice | tuple[str, ...], self._key[0])),
                    "axis_slices": [self._slice_to_dict(axis_slice) for axis_slice in self._key[1:]],
                }
            }
        return {"getitem": {"type": "label_list", "labels": list(self._key)}}

    def apply(self, frame: Any) -> Any:
        return frame[self.key]


RecipeStep = OperationSpec | MethodStep | TypedMethodStep | ScalarOperationStep | IndexingStep


@dataclass(frozen=True, init=False)
class RecipeSpec:
    """Serial recipe of replayable Wandas frame operations."""

    steps: tuple[RecipeStep, ...]

    def __init__(self, steps: Iterable[RecipeStep]) -> None:
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

    def apply(self, frame: Any) -> Any:
        result: Any = frame
        for step in self.steps:
            if isinstance(step, MethodStep | TypedMethodStep | ScalarOperationStep | IndexingStep):
                result = step.apply(result)
            else:
                result = result.apply_operation(step.operation, **step.params)
        return result


__all__ = [
    "IndexingStep",
    "MethodStep",
    "OperationSpec",
    "RecipeExtractionError",
    "RecipeSpec",
    "ScalarOperationStep",
    "TypedMethodStep",
]
