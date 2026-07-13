"""Edge-free executable call families for Recipe v2."""

from __future__ import annotations

import importlib
import inspect
import math
import numbers
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import numpy as np
from dask.array.core import Array as DaArray

from wandas.pipeline.decorators import MultiInputHandler, multi_input_handler
from wandas.pipeline.errors import RecipeSerializationError
from wandas.processing.semantic import (
    OperationContract,
    ReplayTargetContract,
    ReplayValue,
    freeze_replay_value,
    thaw_replay_value,
)

CallLoader = Callable[[Mapping[str, Any]], "CanonicalCall"]
_LOADERS: dict[str, tuple[CallLoader, frozenset[str]]] = {}


def register_call(call_type: str, loader: CallLoader, fields: frozenset[str]) -> None:
    if call_type in _LOADERS:
        raise RuntimeError(f"Recipe call type is already registered: {call_type}")
    _LOADERS[call_type] = (loader, fields)


def load_call(payload: Mapping[str, Any]) -> CanonicalCall:
    call_type = payload.get("type")
    if not isinstance(call_type, str) or call_type not in _LOADERS:
        raise RecipeSerializationError(f"Unknown Recipe call type\n  Type: {call_type!r}")
    loader, fields = _LOADERS[call_type]
    if set(payload) != fields:
        raise RecipeSerializationError(f"Recipe call fields do not match type\n  Type: {call_type!r}")
    return loader(payload)


def _version(value: object, expected: int | None = 1) -> int:
    version = cast(Mapping[str, Any], value).get("version") if isinstance(value, Mapping) else value
    if type(version) is not int or version < 1 or expected is not None and version != expected:
        raise RecipeSerializationError(f"Unsupported Recipe operation version\n  Got: {version!r}")
    return version


def _operation(payload: Mapping[str, Any]) -> str:
    operation = payload.get("operation")
    if not isinstance(operation, str) or not operation:
        raise RecipeSerializationError("Recipe operation must be a non-empty string")
    return operation


def _freeze_params(params: Mapping[str, Any]) -> ReplayValue:
    if not isinstance(params, Mapping) or not all(isinstance(key, str) for key in params):
        raise RecipeSerializationError("Recipe params must be a string-keyed mapping")
    if _contains_array(params):
        raise RecipeSerializationError("External arrays must be named Recipe inputs, not params")
    try:
        return freeze_replay_value(dict(params))
    except (TypeError, ValueError) as exc:
        raise RecipeSerializationError(f"Invalid Recipe params\n  Cause: {exc}") from exc


def _contains_array(value: Any) -> bool:
    if isinstance(value, np.ndarray | DaArray):
        return True
    if isinstance(value, Mapping):
        return any(_contains_array(item) for item in value.values())
    if isinstance(value, tuple | list | set | frozenset):
        return any(_contains_array(item) for item in value)
    return False


def _thaw_params(params: ReplayValue) -> dict[str, Any]:
    return dict(thaw_replay_value(params))


def _normalize_params(value: Mapping[str, Any] | ReplayValue) -> ReplayValue:
    params = thaw_replay_value(cast(ReplayValue, value)) if _is_frozen_mapping(value) else value
    return _freeze_params(cast(Mapping[str, Any], params))


def _load_path(path: object) -> Any:
    if not isinstance(path, str) or "<locals>" in path or path.startswith("__main__."):
        raise RecipeSerializationError(f"Recipe callable path is not stable\n  Path: {path!r}")
    parts = path.split(".")
    for split_at in range(len(parts) - 1, 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:split_at]))
            for part in parts[split_at:]:
                value = getattr(value, part)
            return value
        except (ImportError, AttributeError):
            continue
    raise RecipeSerializationError(f"Recipe callable path is not importable\n  Path: {path!r}")


def _load_stable_function(path: object) -> Callable[..., Any]:
    value = _load_path(path)
    if not inspect.isfunction(value) or value.__module__ == "__main__" or "<locals>" in value.__qualname__:
        raise RecipeSerializationError(f"Recipe callable must be a module-level function\n  Path: {path!r}")
    if f"{value.__module__}.{value.__qualname__}" != path:
        raise RecipeSerializationError(f"Recipe callable path is not canonical\n  Path: {path!r}")
    return value


class CanonicalCall:
    """Nominal marker for calls validated by their constructors."""

    arity: int
    output_kind: str

    def accepts_input_kinds(self, kinds: tuple[str, ...]) -> bool:
        raise NotImplementedError

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        raise NotImplementedError

    def to_payload(self) -> dict[str, Any]:
        raise NotImplementedError


class FrameCall(CanonicalCall):
    output_kind: ClassVar[str] = "frame"

    def accepts_input_kinds(self, kinds: tuple[str, ...]) -> bool:
        return all(kind == "frame" for kind in kinds)


def _audio_operation(operation: str, version: int) -> type[Any]:
    from wandas.processing import get_operation

    try:
        operation_class = get_operation(operation)
    except ValueError as exc:
        raise RecipeSerializationError(f"Unknown Recipe operation\n  Operation: {operation!r}") from exc
    if getattr(operation_class, "operation_version", 1) != version:
        raise RecipeSerializationError("Recipe operation version does not match runtime operation")
    if not getattr(operation_class, "supports_generic_replay", False):
        raise RecipeSerializationError("Audio operation has not opted into generic Recipe replay")
    return operation_class


def _target_member(operation: str, target: str, version: int, output_kind: str) -> Any:
    member_value = _load_path(target)
    owner_path, _, member = target.rpartition(".")
    owner = _load_path(owner_path)
    if not isinstance(owner, type) or owner.__dict__.get(member) is not member_value or member.startswith("_"):
        raise RecipeSerializationError("Recipe target must be a directly owned public member")
    if operation != member:
        raise RecipeSerializationError("Recipe operation and target member must match")
    contract_owner = member_value.fget if isinstance(member_value, property) else member_value
    if getattr(contract_owner, "__wandas_replay_target__", None) != ReplayTargetContract(
        operation, version, cast(Any, output_kind)
    ):
        raise RecipeSerializationError(f"Recipe {output_kind} target contract mismatch")
    return member_value


@dataclass(frozen=True)
class AudioCall(FrameCall):
    operation: str
    params: ReplayValue
    version: int = 1
    arity: ClassVar[int] = 1

    def __init__(self, operation: str, params: Mapping[str, Any] | None = None, version: int = 1) -> None:
        _version(version, None)
        _audio_operation(operation, version)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "params", _freeze_params(params or {}))
        object.__setattr__(self, "version", version)

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        _audio_operation(self.operation, self.version)
        return inputs[0].apply_operation(self.operation, **_thaw_params(self.params))

    def to_payload(self) -> dict[str, Any]:
        return _payload("audio", self.operation, self.version, self.params)


@dataclass(frozen=True)
class MethodCall(FrameCall):
    operation: str
    target: str
    params: ReplayValue
    version: int = 1
    arity: ClassVar[int] = 1

    def __init__(self, operation: str, target: str, params: Mapping[str, Any] | None = None, version: int = 1) -> None:
        _target_member(operation, target, version, "frame")
        _version(version, None)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "params", _freeze_params(params or {}))
        object.__setattr__(self, "version", version)

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        member = _target_member(self.operation, self.target, self.version, "frame")
        return member.__get__(inputs[0], type(inputs[0]))(**_thaw_params(self.params))

    def to_payload(self) -> dict[str, Any]:
        payload = _payload("method", self.operation, self.version, self.params)
        payload["target"] = self.target
        return payload


_OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "**": operator.pow}


def _operator(symbol: str, left: Any, right: Any) -> Any:
    if symbol not in _OPERATORS:
        raise RecipeSerializationError(f"Unknown Recipe operator: {symbol!r}")
    return _OPERATORS[symbol](left, right)


@dataclass(frozen=True)
class ScalarCall(FrameCall):
    operation: str
    operand: bool | int | float | complex
    reverse: bool = False
    version: int = 1
    arity: ClassVar[int] = 1

    def __post_init__(self) -> None:
        _version(self.version)
        if self.operation not in _OPERATORS:
            raise RecipeSerializationError("Unknown scalar Recipe operation")
        if not isinstance(self.operand, numbers.Number):
            raise RecipeSerializationError("Scalar Recipe operand must be numeric")
        value = complex(self.operand)
        if type(self.reverse) is not bool:
            raise RecipeSerializationError("Scalar Recipe operand and direction are invalid")
        normalized = (
            bool(self.operand)
            if isinstance(self.operand, bool | np.bool_)
            else int(self.operand)
            if isinstance(self.operand, numbers.Integral)
            else float(self.operand)
            if isinstance(self.operand, numbers.Real)
            else value
        )
        object.__setattr__(self, "operand", normalized)

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        left, right = (self.operand, inputs[0]) if self.reverse else (inputs[0], self.operand)
        return _operator(self.operation, left, right)

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "scalar",
            "operation": self.operation,
            "version": self.version,
            "params": _freeze_params({}),
            "operand": freeze_replay_value(self.operand)
            if isinstance(self.operand, complex)
            or isinstance(self.operand, numbers.Real)
            and not math.isfinite(float(self.operand))
            else self.operand,
            "reverse": self.reverse,
        }


@dataclass(frozen=True)
class BinaryCall(FrameCall):
    operation: str
    version: int = 1
    arity: ClassVar[int] = 2

    def __post_init__(self) -> None:
        _version(self.version)
        if self.operation not in _OPERATORS:
            raise RecipeSerializationError("Unknown binary Recipe operation")

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        return _operator(self.operation, inputs[0], inputs[1])

    def to_payload(self) -> dict[str, Any]:
        return _payload("binary", self.operation, self.version, _freeze_params({}))


@dataclass(frozen=True)
class ExternalArrayCall(CanonicalCall):
    operation: str
    array_index: int
    version: int = 1
    arity: ClassVar[int] = 2
    output_kind: ClassVar[str] = "frame"

    def __post_init__(self) -> None:
        _version(self.version)
        if self.operation not in _OPERATORS or type(self.array_index) is not int or self.array_index not in {0, 1}:
            raise RecipeSerializationError("Invalid external-array Recipe call")

    def accepts_input_kinds(self, kinds: tuple[str, ...]) -> bool:
        return len(kinds) == 2 and kinds[self.array_index] == "array" and kinds[1 - self.array_index] == "frame"

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        if not isinstance(inputs[self.array_index], np.ndarray | DaArray):
            raise TypeError("External Recipe operand must be NumPy or Dask")
        frame = inputs[1 - self.array_index]
        return frame._binary_operand_op(
            inputs[self.array_index],
            _OPERATORS[self.operation],
            self.operation,
            reverse=self.array_index == 0,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = _payload("external_array", self.operation, self.version, _freeze_params({}))
        payload["array_index"] = self.array_index
        return payload


@dataclass(frozen=True)
class IndexCall(FrameCall):
    key: ReplayValue
    version: int = 1
    arity: ClassVar[int] = 1

    def __init__(self, key: Any, version: int = 1) -> None:
        _version(version)
        if not _valid_index(key):
            raise RecipeSerializationError("Recipe index is not a supported selection intent")
        object.__setattr__(self, "key", freeze_replay_value(key))
        object.__setattr__(self, "version", version)

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        return inputs[0][thaw_replay_value(self.key)]

    def to_payload(self) -> dict[str, Any]:
        return {"type": "index", "operation": "__getitem__", "version": self.version, "key": self.key}


def _valid_index(key: Any) -> bool:
    if isinstance(key, bool | np.bool_):
        return False
    if isinstance(key, numbers.Integral | str | slice):
        return True
    if isinstance(key, np.ndarray):
        return key.ndim == 1 and (np.issubdtype(key.dtype, np.integer) or np.issubdtype(key.dtype, np.bool_))
    if isinstance(key, list):
        return bool(key) and (all(isinstance(item, str) for item in key) or all(type(item) is int for item in key))
    if isinstance(key, tuple):
        return bool(key) and _valid_index(key[0]) and all(isinstance(item, slice) for item in key[1:])
    return False


def _is_frozen_mapping(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and value[0] == "mapping" and isinstance(value[1], tuple)


@dataclass(frozen=True)
class AddChannelCall(CanonicalCall):
    input_kind: Literal["frame", "array"]
    params: ReplayValue
    version: int = 1
    arity: ClassVar[int] = 2
    output_kind: ClassVar[str] = "frame"

    def __init__(self, input_kind: Literal["frame", "array"], params: Mapping[str, Any], version: int = 1) -> None:
        _version(version)
        if input_kind not in {"frame", "array"}:
            raise RecipeSerializationError("Invalid add-channel input kind")
        object.__setattr__(self, "input_kind", input_kind)
        object.__setattr__(self, "params", _freeze_params(params))
        object.__setattr__(self, "version", version)

    def accepts_input_kinds(self, kinds: tuple[str, ...]) -> bool:
        return kinds == ("frame", self.input_kind)

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        return inputs[0].add_channel(inputs[1], **_thaw_params(self.params))

    def to_payload(self) -> dict[str, Any]:
        payload = _payload("add_channel", "add_channel", self.version, self.params)
        payload["input_kind"] = self.input_kind
        return payload


@dataclass(frozen=True)
class CustomCall(FrameCall):
    function: str
    params: ReplayValue
    output_shape_function: str | None = None
    output_frame_class: str | None = None
    output_frame_kwargs: ReplayValue = ("mapping", ())
    pure: bool = True
    version: int = 1
    arity: ClassVar[int] = 1

    def __init__(
        self,
        function: str,
        params: Mapping[str, Any] | ReplayValue,
        output_shape_function: str | None = None,
        output_frame_class: str | None = None,
        output_frame_kwargs: Mapping[str, Any] | ReplayValue | None = None,
        pure: bool = True,
        version: int = 1,
    ) -> None:
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "params", _normalize_params(params))
        object.__setattr__(self, "output_shape_function", output_shape_function)
        object.__setattr__(self, "output_frame_class", output_frame_class)
        kwargs = output_frame_kwargs or {}
        object.__setattr__(
            self,
            "output_frame_kwargs",
            _normalize_params(kwargs),
        )
        object.__setattr__(self, "pure", pure)
        object.__setattr__(self, "version", version)
        self.__post_init__()

    def __post_init__(self) -> None:
        _version(self.version)
        if type(self.pure) is not bool:
            raise RecipeSerializationError("Invalid custom Recipe function contract")
        _load_stable_function(self.function)
        if self.output_shape_function is not None:
            _load_stable_function(self.output_shape_function)
        if self.output_frame_class is not None:
            from wandas.core.base_frame import BaseFrame

            frame_class = _load_path(self.output_frame_class)
            if not isinstance(frame_class, type) or not issubclass(frame_class, BaseFrame):
                raise RecipeSerializationError("Custom Recipe output class must be a Wandas frame class")

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        frame_class = None if self.output_frame_class is None else _load_path(self.output_frame_class)
        return inputs[0].apply(
            _load_stable_function(self.function),
            output_shape_func=None
            if self.output_shape_function is None
            else _load_stable_function(self.output_shape_function),
            output_frame_class=frame_class,
            output_frame_kwargs=_thaw_params(self.output_frame_kwargs) or None,
            dask_pure=self.pure,
            **_thaw_params(self.params),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "custom",
            "operation": "custom",
            "version": self.version,
            "function": self.function,
            "output_shape_function": self.output_shape_function,
            "output_frame_class": self.output_frame_class,
            "output_frame_kwargs": self.output_frame_kwargs,
            "pure": self.pure,
            "params": self.params,
        }


@dataclass(frozen=True)
class TerminalCall(FrameCall):
    operation: str
    target: str
    version: int = 1
    arity: ClassVar[int] = 1
    output_kind: ClassVar[str] = "terminal"

    def __post_init__(self) -> None:
        _version(self.version)
        _target_member(self.operation, self.target, self.version, "terminal")

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        member = _target_member(self.operation, self.target, self.version, "terminal")
        target = member.__get__(inputs[0], type(inputs[0]))
        return target() if callable(target) else target

    def to_payload(self) -> dict[str, Any]:
        payload = _payload("terminal", self.operation, self.version, _freeze_params({}))
        payload["target"] = self.target
        return payload


@dataclass(frozen=True)
class MultiInputCall(FrameCall):
    operation: str
    roles: tuple[str, ...]
    target: str
    params: ReplayValue
    version: int = 1
    input_kinds: tuple[Literal["frame", "array"], ...] = ()

    def __init__(
        self,
        operation: str,
        roles: tuple[str, ...],
        target: str,
        params: Mapping[str, Any] | ReplayValue,
        version: int = 1,
        *,
        input_kinds: tuple[Literal["frame", "array"], ...] | None = None,
    ) -> None:
        normalized_roles = tuple(roles)
        if not normalized_roles or len(set(normalized_roles)) != len(normalized_roles):
            raise RecipeSerializationError("Multi-input roles must be non-empty and unique")
        if not all(isinstance(role, str) and role.strip() for role in normalized_roles):
            raise RecipeSerializationError("Multi-input roles must be non-blank strings")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "roles", normalized_roles)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "params", _normalize_params(params))
        object.__setattr__(self, "version", version)
        kinds = ("frame",) * len(normalized_roles) if input_kinds is None else tuple(input_kinds)
        object.__setattr__(self, "input_kinds", kinds)
        self.__post_init__()

    @property
    def arity(self) -> int:
        return len(self.roles)

    def __post_init__(self) -> None:
        _version(self.version, None)
        if len(self.input_kinds) != len(self.roles) or any(kind not in {"frame", "array"} for kind in self.input_kinds):
            raise RecipeSerializationError("Multi-input kinds must match roles")
        self._handler()

    def accepts_input_kinds(self, kinds: tuple[str, ...]) -> bool:
        return kinds == self.input_kinds

    def _handler(self) -> MultiInputHandler:
        handler = _load_stable_function(self.target)
        if getattr(handler, "__wandas_multi_input_contract__", None) != (
            OperationContract(self.operation, self.version, True, ()),
            self.roles,
        ):
            raise RecipeSerializationError("Multi-input handler contract mismatch")
        return handler

    def invoke(self, inputs: tuple[Any, ...]) -> Any:
        return self._handler()(inputs, _thaw_params(self.params))

    def to_payload(self) -> dict[str, Any]:
        payload = _payload("multi_input", self.operation, self.version, self.params)
        payload.update({"roles": list(self.roles), "input_kinds": list(self.input_kinds), "target": self.target})
        return payload


def _payload(kind: str, operation: str, version: int, params: ReplayValue) -> dict[str, Any]:
    return {"type": kind, "operation": operation, "version": version, "params": params}


def _tree(value: Any) -> ReplayValue:
    if not isinstance(value, list) or not value or not isinstance(value[0], str):
        raise RecipeSerializationError("Recipe value tree is malformed")
    kind = value[0]
    if kind == "mapping":
        if len(value) != 2:
            raise RecipeSerializationError("Recipe mapping value is malformed")
        pairs = value[1]
        if not isinstance(pairs, list):
            raise RecipeSerializationError("Recipe mapping value is malformed")
        seen: set[Any] = set()
        items: list[tuple[Any, ReplayValue]] = []
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise RecipeSerializationError("Recipe mapping entry is malformed")
            key = pair[0] if isinstance(pair[0], str) else _tree(pair[0])
            try:
                runtime_key = key if isinstance(key, str) else thaw_replay_value(key)
                hash(runtime_key)
            except (TypeError, ValueError) as exc:
                raise RecipeSerializationError("Recipe mapping key must be a supported hashable value") from exc
            if runtime_key in seen:
                raise RecipeSerializationError("Recipe mapping keys must be unique")
            seen.add(runtime_key)
            items.append((key, _tree(pair[1])))
        return ("mapping", tuple(items))
    if kind in {"list", "tuple", "set", "frozenset"}:
        if len(value) != 2:
            raise RecipeSerializationError("Recipe sequence value is malformed")
        if not isinstance(value[1], list):
            raise RecipeSerializationError("Recipe sequence value is malformed")
        return (kind, tuple(_tree(item) for item in value[1]))
    if kind in {"complex", "slice"}:
        if len(value) != (3 if kind == "complex" else 4):
            raise RecipeSerializationError(f"Recipe {kind} value is malformed")
        return (kind, *(_tree(item) for item in value[1:]))
    if kind == "ndarray":
        if len(value) != 4:
            raise RecipeSerializationError("Recipe ndarray value is malformed")
        return (kind, value[1], tuple(value[2]), _tree(value[3]))
    if kind == "dask-descriptor":
        if len(value) != 2:
            raise RecipeSerializationError("Recipe Dask descriptor is malformed")
        return (kind, _tree(value[1]))
    if kind == "none" and len(value) == 1:
        return (kind,)
    if kind in {"str", "bool", "int", "float", "opaque"} and len(value) == 2:
        return (kind, value[1])
    raise RecipeSerializationError(f"Unknown Recipe value kind\n  Kind: {kind!r}")


def _params_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _thaw_params(_tree(payload.get("params")))


def _empty_params(payload: Mapping[str, Any]) -> None:
    if _params_payload(payload):
        raise RecipeSerializationError("Recipe call does not accept params")


_BASE = frozenset({"type", "operation", "version", "params"})
register_call(
    "audio",
    lambda value: AudioCall(_operation(value), _params_payload(value), _version(value, None)),
    _BASE,
)
register_call(
    "method",
    lambda value: MethodCall(_operation(value), str(value["target"]), _params_payload(value), _version(value, None)),
    _BASE | {"target"},
)


def _load_scalar(value: Mapping[str, Any]) -> ScalarCall:
    _empty_params(value)
    operand = value.get("operand")
    if isinstance(operand, list):
        operand = thaw_replay_value(_tree(operand))
    return ScalarCall(_operation(value), cast(Any, operand), cast(Any, value.get("reverse")), _version(value))


def _load_binary(value: Mapping[str, Any]) -> BinaryCall:
    _empty_params(value)
    return BinaryCall(_operation(value), _version(value))


def _load_external(value: Mapping[str, Any]) -> ExternalArrayCall:
    _empty_params(value)
    return ExternalArrayCall(_operation(value), cast(Any, value.get("array_index")), _version(value))


register_call("scalar", _load_scalar, _BASE | {"operand", "reverse"})
register_call("binary", _load_binary, _BASE)
register_call("external_array", _load_external, _BASE | {"array_index"})


def _load_index(value: Mapping[str, Any]) -> IndexCall:
    if _operation(value) != "__getitem__":
        raise RecipeSerializationError("Index Recipe operation must be '__getitem__'")
    return IndexCall(thaw_replay_value(_tree(value["key"])), _version(value))


def _load_add_channel(value: Mapping[str, Any]) -> AddChannelCall:
    if _operation(value) != "add_channel":
        raise RecipeSerializationError("Add-channel Recipe operation must be 'add_channel'")
    return AddChannelCall(value["input_kind"], _params_payload(value), _version(value))  # type: ignore[arg-type]


register_call("index", _load_index, frozenset({"type", "operation", "version", "key"}))
register_call("add_channel", _load_add_channel, _BASE | {"input_kind"})


def _load_custom(value: Mapping[str, Any]) -> CustomCall:
    if _operation(value) != "custom":
        raise RecipeSerializationError("Custom Recipe operation must be 'custom'")
    return CustomCall(
        str(value["function"]),
        _tree(value["params"]),
        value.get("output_shape_function"),  # type: ignore[arg-type]
        value.get("output_frame_class"),  # type: ignore[arg-type]
        _tree(value["output_frame_kwargs"]),
        cast(Any, value.get("pure")),
        _version(value),
    )


register_call(
    "custom",
    _load_custom,
    frozenset(
        {
            "type",
            "operation",
            "version",
            "function",
            "output_shape_function",
            "output_frame_class",
            "output_frame_kwargs",
            "pure",
            "params",
        }
    ),
)
register_call(
    "terminal",
    lambda value: TerminalCall(_operation(value), str(value["target"]), _version(value))
    if not _params_payload(value)
    else _reject_params(),
    _BASE | {"target"},
)


def _reject_params() -> Any:
    raise RecipeSerializationError("Recipe call does not accept params")


register_call(
    "multi_input",
    lambda value: MultiInputCall(
        _operation(value),
        tuple(value["roles"]),
        str(value["target"]),
        _tree(value["params"]),
        _version(value, None),
        input_kinds=tuple(value["input_kinds"]),
    ),
    _BASE | {"roles", "input_kinds", "target"},
)


@multi_input_handler("add", version=1, roles=("signal", "operand"))
def apply_add(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
    return inputs[0].add(inputs[1], **params)


@multi_input_handler("add_with_snr", version=1, roles=("signal", "noise"))
def apply_add_with_snr(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
    return inputs[0].add(inputs[1], **params)
