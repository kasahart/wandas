from __future__ import annotations

import importlib
import inspect
import numbers
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from dask.array.core import Array as DaArray

from wandas.pipeline.params import (
    _BooleanMask,
    _params_to_public_dict,
    _snapshot_get_channel_query_params,
    _snapshot_param_value,
    _snapshot_params,
    _snapshot_rename_channels_params,
)
from wandas.pipeline.registry import (
    _REPLAYABLE_BINARY_FRAME_OPERATIONS,
    _REPLAYABLE_METHOD_NAMES,
    _REPLAYABLE_SCALAR_OPERATIONS,
    _REPLAYABLE_TERMINAL_METHODS,
    _REPLAYABLE_TERMINAL_PROPERTIES,
    _REPLAYABLE_TYPED_METHOD_NAMES,
)


def _load_importable_function(path: str) -> Callable[..., Any]:
    module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"CustomFunctionStep function must be a module-level import path\n  Got: {path!r}")
    if module_name == "__main__" or attr_name == "<lambda>":
        raise TypeError(f"CustomFunctionStep import path must resolve to a module-level function\n  Path: {path!r}")
    module = importlib.import_module(module_name)
    func = getattr(module, attr_name)
    if not inspect.isfunction(func) or func.__qualname__ != func.__name__ or func.__closure__ is not None:
        raise TypeError(
            "CustomFunctionStep import path must resolve to a module-level function\n"
            f"  Path: {path!r}\n"
            f"  Got: {type(func).__name__}"
        )
    if func.__module__ != module_name or getattr(module, func.__name__, None) is not func:
        raise TypeError(f"CustomFunctionStep import path must resolve to its module-level function\n  Path: {path!r}")
    return cast(Callable[..., Any], func)


def _load_importable_frame_class(path: str) -> type[Any]:
    module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"CustomFunctionStep output_frame_class must be a module-level import path\n  Got: {path!r}")
    if module_name == "__main__":
        raise TypeError(
            f"CustomFunctionStep output_frame_class must resolve to an importable BaseFrame subclass\n  Path: {path!r}"
        )
    module = importlib.import_module(module_name)
    frame_class = getattr(module, attr_name)

    from wandas.core.base_frame import BaseFrame

    if not inspect.isclass(frame_class) or not issubclass(frame_class, BaseFrame):
        raise TypeError(
            "CustomFunctionStep output_frame_class must resolve to an importable BaseFrame subclass\n"
            f"  Path: {path!r}\n"
            f"  Got: {type(frame_class).__name__}"
        )
    if frame_class.__module__ != module_name or getattr(module, frame_class.__name__, None) is not frame_class:
        raise TypeError(
            f"CustomFunctionStep output_frame_class must resolve to its module-level class\n  Path: {path!r}"
        )
    return cast(type[Any], frame_class)


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
        if method == "rename_channels":
            frozen_params = _snapshot_rename_channels_params(params or {})
        elif method == "get_channel":
            frozen_params = _snapshot_get_channel_query_params(params or {})
        else:
            frozen_params = _snapshot_params(params or {})
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
        if self.method == "get_channel":
            params = self.params
            mask = params.pop("channel_mask", None)
            if mask is not None:
                return frame.get_channel(np.array(mask, dtype=bool), **params)
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
class CustomFunctionStep:
    """Replayable same-frame custom function by import path."""

    function: str
    output_shape_function: str | None
    dask_pure: bool
    output_frame_class: str | None
    _output_frame_kwargs: tuple[tuple[str, Any], ...]
    _params: tuple[tuple[str, Any], ...]

    def __init__(
        self,
        function: str,
        params: Mapping[str, Any] | None = None,
        *,
        output_shape_function: str | None = None,
        dask_pure: bool = True,
        output_frame_class: str | None = None,
        output_frame_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(function, str) or not function:
            raise TypeError("CustomFunctionStep function must be a non-empty import path string")
        if "." not in function:
            raise ValueError(f"CustomFunctionStep function must be a module import path\n  Got: {function!r}")
        if output_shape_function is not None and (
            not isinstance(output_shape_function, str) or not output_shape_function
        ):
            raise TypeError("CustomFunctionStep output_shape_function must be None or a non-empty import path string")
        if output_shape_function is not None and "." not in output_shape_function:
            raise ValueError(
                "CustomFunctionStep output_shape_function must be a module import path\n"
                f"  Got: {output_shape_function!r}"
            )
        if not isinstance(dask_pure, bool):
            raise TypeError(f"CustomFunctionStep dask_pure must be a bool\n  Got: {type(dask_pure).__name__}")
        if output_frame_class is not None and (not isinstance(output_frame_class, str) or not output_frame_class):
            raise TypeError("CustomFunctionStep output_frame_class must be None or a non-empty import path string")
        if output_frame_class is not None and "." not in output_frame_class:
            raise ValueError(
                f"CustomFunctionStep output_frame_class must be a module import path\n  Got: {output_frame_class!r}"
            )
        if output_frame_class is None and output_frame_kwargs:
            raise TypeError("CustomFunctionStep output_frame_kwargs require output_frame_class")
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "output_shape_function", output_shape_function)
        object.__setattr__(self, "dask_pure", dask_pure)
        object.__setattr__(self, "output_frame_class", output_frame_class)
        object.__setattr__(self, "_output_frame_kwargs", _snapshot_params(output_frame_kwargs or {}))
        object.__setattr__(self, "_params", _snapshot_params(params or {}))

    @property
    def params(self) -> dict[str, Any]:
        return _params_to_public_dict(self._params)

    @property
    def output_frame_kwargs(self) -> dict[str, Any]:
        return _params_to_public_dict(self._output_frame_kwargs)

    def to_dict(self) -> dict[str, Any]:
        step_dict = {
            "custom_function": self.function,
            "output_shape_function": self.output_shape_function,
            "dask_pure": self.dask_pure,
            "params": self.params,
        }
        if self.output_frame_class is not None:
            step_dict["output_frame_class"] = self.output_frame_class
            step_dict["output_frame_kwargs"] = self.output_frame_kwargs
        return step_dict

    def apply(self, frame: Any) -> Any:
        func = _load_importable_function(self.function)
        output_shape_func = (
            None if self.output_shape_function is None else _load_importable_function(self.output_shape_function)
        )
        output_frame_class = (
            None if self.output_frame_class is None else _load_importable_frame_class(self.output_frame_class)
        )
        return frame.apply(
            func,
            output_shape_func=output_shape_func,
            output_frame_class=output_frame_class,
            output_frame_kwargs=self.output_frame_kwargs or None,
            dask_pure=self.dask_pure,
            **self.params,
        )


@dataclass(frozen=True, init=False)
class ScalarOperationStep:
    """Replayable frame operation with a single numeric scalar operand."""

    symbol: str
    operand: int | float
    reverse: bool

    def __init__(self, symbol: str, operand: int | float, *, reverse: bool = False) -> None:
        if symbol not in _REPLAYABLE_SCALAR_OPERATIONS:
            valid_operations = ", ".join(sorted(_REPLAYABLE_SCALAR_OPERATIONS))
            raise ValueError(
                "ScalarOperationStep operation is outside the replayable scalar allowlist\n"
                f"  Operation: {symbol}\n"
                f"  Valid operations: {valid_operations}"
            )
        if isinstance(operand, bool) or not isinstance(operand, int | float):
            raise TypeError(f"ScalarOperationStep operand must be an int or float\n  Got: {type(operand).__name__}")
        if not isinstance(reverse, bool):
            raise TypeError(f"ScalarOperationStep reverse must be a bool\n  Got: {type(reverse).__name__}")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "operand", _snapshot_param_value(operand))
        object.__setattr__(self, "reverse", reverse)

    def to_dict(self) -> dict[str, Any]:
        step_dict: dict[str, Any] = {"scalar_operation": self.symbol, "operand": self.operand}
        if self.reverse:
            step_dict["reverse"] = True
        return step_dict

    def apply(self, frame: Any) -> Any:
        if self.reverse:
            if self.symbol == "+":
                return self.operand + frame
            if self.symbol == "-":
                return self.operand - frame
            if self.symbol == "*":
                return self.operand * frame
            if self.symbol == "/":
                return self.operand / frame
            if self.symbol == "**":
                return self.operand**frame
            raise AssertionError(f"Unhandled reverse scalar operation: {self.symbol}")
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

    _key: (
        slice
        | _BooleanMask
        | tuple[int, ...]
        | tuple[str, ...]
        | tuple[int | slice | _BooleanMask | tuple[int, ...] | tuple[str, ...], ...]
    )

    def __init__(self, key: Any) -> None:
        if isinstance(key, slice):
            object.__setattr__(
                self,
                "_key",
                self._snapshot_slice(key),
            )
            return
        if isinstance(key, _BooleanMask):
            object.__setattr__(self, "_key", self._snapshot_boolean_mask(key.values))
            return
        if self._is_boolean_mask_key(key):
            object.__setattr__(self, "_key", self._snapshot_boolean_mask(key))
            return
        if self._is_multidimensional_key(key):
            object.__setattr__(self, "_key", self._snapshot_multidimensional_key(cast(tuple[Any, ...], key)))
            return
        if (
            isinstance(key, list | tuple)
            and key
            and all(isinstance(index, numbers.Integral) and not isinstance(index, bool) for index in key)
        ):
            object.__setattr__(self, "_key", tuple(int(index) for index in key))
            return
        if not isinstance(key, list | tuple) or not key or not all(isinstance(label, str) for label in key):
            raise TypeError(
                "IndexingStep key must be a channel slice, non-empty integer list, or non-empty label list\n"
                f"  Got: {type(key).__name__}"
            )
        object.__setattr__(self, "_key", tuple(key))

    @staticmethod
    def _is_multidimensional_key(key: object) -> bool:
        return isinstance(key, tuple) and len(key) >= 2 and all(isinstance(item, slice) for item in key[1:])

    @staticmethod
    def _is_boolean_mask_key(key: object) -> bool:
        return isinstance(key, np.ndarray) and key.dtype in (bool, np.bool_) and key.ndim == 1 and key.size > 0

    @staticmethod
    def _snapshot_boolean_mask(key: Any) -> _BooleanMask:
        if isinstance(key, np.ndarray):
            values = key.tolist()
        else:
            values = list(key)
        return _BooleanMask(tuple(bool(value) for value in values))

    @classmethod
    def _snapshot_multidimensional_key(
        cls, key: tuple[Any, ...]
    ) -> tuple[int | slice | _BooleanMask | tuple[int, ...] | tuple[str, ...], ...]:
        channel_key = key[0]
        if isinstance(channel_key, slice):
            frozen_channel_key: int | slice | _BooleanMask | tuple[int, ...] | tuple[str, ...] = cls._snapshot_slice(
                channel_key
            )
        elif isinstance(channel_key, _BooleanMask):
            frozen_channel_key = cls._snapshot_boolean_mask(channel_key.values)
        elif cls._is_boolean_mask_key(channel_key):
            frozen_channel_key = cls._snapshot_boolean_mask(channel_key)
        elif isinstance(channel_key, numbers.Integral) and not isinstance(channel_key, bool):
            frozen_channel_key = int(channel_key)
        elif (
            isinstance(channel_key, list)
            and channel_key
            and all(isinstance(index, numbers.Integral) and not isinstance(index, bool) for index in channel_key)
        ):
            frozen_channel_key = tuple(int(index) for index in channel_key)
        elif isinstance(channel_key, list) and channel_key and all(isinstance(label, str) for label in channel_key):
            frozen_channel_key = tuple(channel_key)
        else:
            raise TypeError(
                "IndexingStep multidimensional channel key must be an int, slice, non-empty integer list, "
                "or non-empty label list\n"
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
    def key(self) -> slice | np.ndarray[Any, np.dtype[np.bool_]] | list[int] | list[str] | tuple[Any, ...]:
        if isinstance(self._key, slice):
            return self._key
        if isinstance(self._key, _BooleanMask):
            return np.array(self._key.values, dtype=bool)
        if self._is_multidimensional_key(self._key):
            channel_key = self._key[0]
            public_channel_key: int | slice | np.ndarray[Any, np.dtype[np.bool_]] | list[int] | list[str]
            if isinstance(channel_key, tuple):
                public_channel_key = list(channel_key)
            elif isinstance(channel_key, _BooleanMask):
                public_channel_key = np.array(channel_key.values, dtype=bool)
            else:
                public_channel_key = cast(int | slice, channel_key)
            return (public_channel_key, *self._key[1:])
        return list(self._key)

    @staticmethod
    def _slice_to_dict(key: slice) -> dict[str, int | None]:
        return {"start": key.start, "stop": key.stop, "step": key.step}

    @classmethod
    def _channel_key_to_dict(
        cls, key: int | slice | _BooleanMask | tuple[int, ...] | tuple[str, ...]
    ) -> dict[str, Any]:
        if isinstance(key, slice):
            return {"type": "slice", **cls._slice_to_dict(key)}
        if isinstance(key, int):
            return {"type": "index", "value": key}
        if isinstance(key, _BooleanMask):
            return {"type": "boolean_mask", "mask": list(key.values)}
        if all(isinstance(index, int) for index in key):
            return {"type": "integer_list", "indices": list(key)}
        return {"type": "label_list", "labels": list(key)}

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self._key, slice):
            return {
                "getitem": {
                    "type": "channel_slice",
                    **self._slice_to_dict(self._key),
                }
            }
        if isinstance(self._key, _BooleanMask):
            return {"getitem": {"type": "boolean_mask", "mask": list(self._key.values)}}
        if self._is_multidimensional_key(self._key):
            return {
                "getitem": {
                    "type": "multidimensional_slice",
                    "channel": self._channel_key_to_dict(
                        cast(int | slice | _BooleanMask | tuple[int, ...] | tuple[str, ...], self._key[0])
                    ),
                    "axis_slices": [self._slice_to_dict(axis_slice) for axis_slice in self._key[1:]],
                }
            }
        if self._key and all(isinstance(index, int) for index in self._key):
            return {"getitem": {"type": "integer_list", "indices": list(self._key)}}
        return {"getitem": {"type": "label_list", "labels": list(self._key)}}

    def apply(self, frame: Any) -> Any:
        return frame[self.key]


@dataclass(frozen=True, init=False)
class TerminalStep:
    """Replayable terminal frame metric that returns a non-frame value."""

    metric: str
    _params: tuple[tuple[str, Any], ...]

    def __init__(self, metric: str, params: Mapping[str, Any] | None = None) -> None:
        if metric not in _REPLAYABLE_TERMINAL_PROPERTIES | _REPLAYABLE_TERMINAL_METHODS:
            valid_metrics = ", ".join(sorted(_REPLAYABLE_TERMINAL_PROPERTIES | _REPLAYABLE_TERMINAL_METHODS))
            raise ValueError(
                "TerminalStep metric is outside the replayable terminal allowlist\n"
                f"  Metric: {metric}\n"
                f"  Valid metrics: {valid_metrics}"
            )
        if metric in _REPLAYABLE_TERMINAL_PROPERTIES and params:
            raise TypeError(
                "TerminalStep metric does not accept params\n"
                f"  Metric: {metric}\n"
                "  Current terminal recipe support is limited to zero-argument frame properties."
            )
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "_params", _snapshot_params(params or {}))

    @property
    def params(self) -> dict[str, Any]:
        return _params_to_public_dict(self._params)

    def to_dict(self) -> dict[str, Any]:
        return {"terminal": self.metric, "params": self.params}

    def apply(self, frame: Any) -> Any:
        terminal = getattr(frame, self.metric)
        if self.metric in _REPLAYABLE_TERMINAL_PROPERTIES:
            if callable(terminal):
                raise TypeError(
                    "TerminalStep expected a terminal property but found a callable attribute\n"
                    f"  Metric: {self.metric}\n"
                    f"  Frame type: {type(frame).__name__}"
                )
            return terminal
        return terminal(**self.params)


@dataclass(frozen=True, init=False)
class BinaryFrameStep:
    """Replayable binary operation over two named frame inputs."""

    operation: str
    left: str
    right: str
    _params: tuple[tuple[str, Any], ...]

    def __init__(self, operation: str, left: str, right: str, params: Mapping[str, Any] | None = None) -> None:
        if operation not in _REPLAYABLE_BINARY_FRAME_OPERATIONS:
            valid_operations = ", ".join(sorted(_REPLAYABLE_BINARY_FRAME_OPERATIONS))
            raise ValueError(
                "BinaryFrameStep operation is outside the replayable binary-frame allowlist\n"
                f"  Operation: {operation}\n"
                f"  Valid operations: {valid_operations}"
            )
        if not left or not right:
            raise ValueError("BinaryFrameStep left and right input names must be non-empty strings")
        if operation in _REPLAYABLE_SCALAR_OPERATIONS and params:
            raise TypeError(f"BinaryFrameStep frame operator does not accept params\n  Operation: {operation}")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "_params", _snapshot_params(params or {}))

    @property
    def params(self) -> dict[str, Any]:
        return _params_to_public_dict(self._params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "binary_frame": {
                "operation": self.operation,
                "left": self.left,
                "right": self.right,
                "params": self.params,
            }
        }

    def apply(self, frames: Mapping[str, Any]) -> Any:
        try:
            left_frame = frames[self.left]
            right_frame = frames[self.right]
        except KeyError as exc:
            raise KeyError(
                "GraphRecipeSpec input is missing\n"
                f"  Missing input: {exc.args[0]!r}\n"
                f"  Available inputs: {sorted(frames)}"
            ) from exc
        if self.operation == "+":
            return left_frame + right_frame
        if self.operation == "-":
            return left_frame - right_frame
        if self.operation == "*":
            return left_frame * right_frame
        if self.operation == "/":
            return left_frame / right_frame
        if self.operation == "**":
            return left_frame**right_frame
        return left_frame.add(right_frame, **self.params)


@dataclass(frozen=True, init=False)
class BinaryOperandStep:
    """Replayable binary operation over a frame and a named external operand."""

    operation: str
    frame: str
    operand: str

    def __init__(self, operation: str, frame: str, operand: str) -> None:
        if operation not in _REPLAYABLE_SCALAR_OPERATIONS:
            valid_operations = ", ".join(sorted(_REPLAYABLE_SCALAR_OPERATIONS))
            raise ValueError(
                "BinaryOperandStep operation is outside the replayable operand allowlist\n"
                f"  Operation: {operation}\n"
                f"  Valid operations: {valid_operations}"
            )
        if not frame or not operand:
            raise ValueError("BinaryOperandStep frame and operand input names must be non-empty strings")
        if frame == operand:
            raise ValueError(f"BinaryOperandStep frame and operand inputs must be distinct\n  Input: {frame!r}")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "operand", operand)

    def to_dict(self) -> dict[str, Any]:
        return {
            "binary_operand": {
                "operation": self.operation,
                "frame": self.frame,
                "operand": self.operand,
            }
        }

    def apply(self, inputs: Mapping[str, Any]) -> Any:
        try:
            frame = inputs[self.frame]
            operand = inputs[self.operand]
        except KeyError as exc:
            raise KeyError(
                "NodeGraphRecipeSpec binary operand input is missing\n"
                f"  Missing input: {exc.args[0]!r}\n"
                f"  Available inputs: {sorted(inputs)}"
            ) from exc
        if not isinstance(operand, np.ndarray | DaArray):
            raise TypeError(
                "BinaryOperandStep operand input must be a NumPy or Dask array\n"
                f"  Operand input: {self.operand!r}\n"
                f"  Got: {type(operand).__name__}"
            )
        if self.operation == "+":
            return frame + operand
        if self.operation == "-":
            return frame - operand
        if self.operation == "*":
            return frame * operand
        if self.operation == "/":
            return frame / operand
        if self.operation == "**":
            return frame**operand
        raise AssertionError(f"Unhandled binary operand operation: {self.operation}")


@dataclass(frozen=True, init=False)
class AddChannelStep:
    """Replayable add_channel call over two named frame inputs."""

    base: str
    added: str
    _params: tuple[tuple[str, Any], ...]

    def __init__(self, base: str, added: str, params: Mapping[str, Any] | None = None) -> None:
        if not base or not added:
            raise ValueError("AddChannelStep base and added input names must be non-empty strings")
        unsupported_params = set(params or ()) - {"align", "label", "suffix_on_dup"}
        if unsupported_params:
            raise TypeError(
                "AddChannelStep params only support public add_channel frame-input options\n"
                f"  Unsupported params: {sorted(unsupported_params)}"
            )
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "added", added)
        object.__setattr__(self, "_params", _snapshot_params(params or {}))

    @property
    def params(self) -> dict[str, Any]:
        return _params_to_public_dict(self._params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "add_channel": {
                "base": self.base,
                "added": self.added,
                "params": self.params,
            }
        }

    def apply(self, frames: Mapping[str, Any]) -> Any:
        try:
            base_frame = frames[self.base]
            added_frame = frames[self.added]
        except KeyError as exc:
            raise KeyError(
                "NodeGraphRecipeSpec add_channel input is missing\n"
                f"  Missing input: {exc.args[0]!r}\n"
                f"  Available inputs: {sorted(frames)}"
            ) from exc
        return base_frame.add_channel(added_frame, **self.params)


@dataclass(frozen=True, init=False)
class AddChannelDataStep:
    """Replayable add_channel call over a frame and named raw channel data."""

    base: str
    data: str
    _params: tuple[tuple[str, Any], ...]

    def __init__(self, base: str, data: str, params: Mapping[str, Any] | None = None) -> None:
        if not base or not data:
            raise ValueError("AddChannelDataStep base and data input names must be non-empty strings")
        if base == data:
            raise ValueError(f"AddChannelDataStep base and data inputs must be distinct\n  Input: {base!r}")
        unsupported_params = set(params or ()) - {"align", "label", "suffix_on_dup", "source_time_offset"}
        if unsupported_params:
            raise TypeError(
                "AddChannelDataStep params only support public add_channel raw-data options\n"
                f"  Unsupported params: {sorted(unsupported_params)}"
            )
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "_params", _snapshot_params(params or {}))

    @property
    def params(self) -> dict[str, Any]:
        return _params_to_public_dict(self._params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "add_channel_data": {
                "base": self.base,
                "data": self.data,
                "params": self.params,
            }
        }

    def apply(self, inputs: Mapping[str, Any]) -> Any:
        try:
            base_frame = inputs[self.base]
            data = inputs[self.data]
        except KeyError as exc:
            raise KeyError(
                "NodeGraphRecipeSpec add_channel data input is missing\n"
                f"  Missing input: {exc.args[0]!r}\n"
                f"  Available inputs: {sorted(inputs)}"
            ) from exc
        if not isinstance(data, np.ndarray | DaArray):
            raise TypeError(
                "AddChannelDataStep data input must be a NumPy or Dask array\n"
                f"  Data input: {self.data!r}\n"
                f"  Got: {type(data).__name__}"
            )
        return base_frame.add_channel(data, **self.params)


RecipeStep = (
    OperationSpec
    | MethodStep
    | TypedMethodStep
    | CustomFunctionStep
    | ScalarOperationStep
    | IndexingStep
    | TerminalStep
)
GraphStep = RecipeStep | BinaryFrameStep | BinaryOperandStep | AddChannelStep | AddChannelDataStep


def _apply_recipe_step(step: RecipeStep, frame: Any) -> Any:
    if isinstance(
        step,
        MethodStep | TypedMethodStep | CustomFunctionStep | ScalarOperationStep | IndexingStep | TerminalStep,
    ):
        return step.apply(frame)
    return frame.apply_operation(step.operation, **step.params)


@dataclass(frozen=True, init=False)
class GraphNodeSpec:
    """Replayable node in a tree-shaped frame computation graph."""

    id: str
    step: GraphStep
    inputs: tuple[str, ...]

    def __init__(self, id: str, step: GraphStep, inputs: Iterable[str]) -> None:
        if not isinstance(id, str) or not id:
            raise TypeError("GraphNodeSpec id must be a non-empty string")
        frozen_inputs = tuple(inputs)
        if not frozen_inputs or not all(isinstance(input_id, str) and input_id for input_id in frozen_inputs):
            raise TypeError("GraphNodeSpec inputs must be non-empty strings")
        if isinstance(step, BinaryFrameStep):
            if frozen_inputs != (step.left, step.right):
                raise ValueError(
                    "GraphNodeSpec binary step inputs must match BinaryFrameStep references\n"
                    f"  Node inputs: {list(frozen_inputs)}\n"
                    f"  Binary inputs: {[step.left, step.right]}"
                )
        elif isinstance(step, BinaryOperandStep):
            if frozen_inputs != (step.frame, step.operand):
                raise ValueError(
                    "GraphNodeSpec binary operand inputs must match BinaryOperandStep references\n"
                    f"  Node inputs: {list(frozen_inputs)}\n"
                    f"  Binary operand inputs: {[step.frame, step.operand]}"
                )
        elif isinstance(step, AddChannelStep):
            if frozen_inputs != (step.base, step.added):
                raise ValueError(
                    "GraphNodeSpec add_channel step inputs must match AddChannelStep references\n"
                    f"  Node inputs: {list(frozen_inputs)}\n"
                    f"  add_channel inputs: {[step.base, step.added]}"
                )
        elif isinstance(step, AddChannelDataStep):
            if frozen_inputs != (step.base, step.data):
                raise ValueError(
                    "GraphNodeSpec add_channel data inputs must match AddChannelDataStep references\n"
                    f"  Node inputs: {list(frozen_inputs)}\n"
                    f"  add_channel data inputs: {[step.base, step.data]}"
                )
        elif len(frozen_inputs) != 1:
            raise ValueError(
                f"GraphNodeSpec unary step requires exactly one input\n  Node inputs: {list(frozen_inputs)}"
            )
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "inputs", frozen_inputs)

    def to_dict(self) -> dict[str, Any]:
        node_dict: dict[str, Any] = {"id": self.id, "step": self.step.to_dict()}
        if len(self.inputs) == 1:
            node_dict["input"] = self.inputs[0]
        else:
            node_dict["inputs"] = list(self.inputs)
        return node_dict
