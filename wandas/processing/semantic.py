"""Immutable semantic replay contracts captured by runtime lineage."""

from __future__ import annotations

import inspect
import math
import numbers
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeAlias, TypeVar, cast

import numpy as np
from dask.array.core import Array as DaArray

InputKind = Literal["frame", "array", "scalar"]
ReplayValue: TypeAlias = tuple[Any, ...]
P = ParamSpec("P")
R = TypeVar("R")


_semantic_capture: ContextVar[Any | None] = ContextVar("wandas_semantic_capture", default=None)


def active_semantic_lineage() -> Any | None:
    """Return the public-operation lineage selected at the current call boundary."""
    return _semantic_capture.get()


@contextmanager
def semantic_lineage(lineage: Any) -> Any:
    """Make an already-final semantic node authoritative for nested helpers."""
    token = _semantic_capture.set(lineage)
    try:
        yield
    finally:
        _semantic_capture.reset(token)


def _has_frame_lineage_contract(value: object) -> bool:
    """Return whether a result structurally exposes Wandas Frame lineage."""
    frame_contract_members = {name for base in type(value).__mro__ for name in base.__dict__}
    return {"_lineage_or_source", "lineage"} <= frame_contract_members


def _invoke_semantic(method: Callable[P, R], args: tuple[Any, ...], kwargs: Mapping[str, Any], lineage: Any) -> R:
    token = _semantic_capture.set(lineage)
    try:
        result = method(*args, **kwargs)
        result_lineage = getattr(result, "lineage", lineage)
        if _has_frame_lineage_contract(result) and result_lineage is not lineage:
            raise RuntimeError("Public operation did not preserve semantic lineage")
        return result
    finally:
        _semantic_capture.reset(token)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


@dataclass(frozen=True)
class InputBinding:
    role: str
    kind: InputKind

    def __post_init__(self) -> None:
        _identifier(self.role, "Replay input role")
        if self.kind not in {"frame", "array", "scalar"}:
            raise ValueError("Replay input kind must be 'frame', 'array', or 'scalar'")


@dataclass(frozen=True)
class OperationContract:
    operation_id: str
    version: int
    pure: bool
    bindings: tuple[InputBinding, ...]

    def __post_init__(self) -> None:
        _identifier(self.operation_id, "Replay operation id")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Replay operation version must be a positive integer")
        if type(self.pure) is not bool:
            raise TypeError("Replay operation purity must be boolean")
        if not isinstance(self.bindings, tuple) or not all(isinstance(item, InputBinding) for item in self.bindings):
            raise TypeError("Replay bindings must be a tuple of InputBinding values")
        roles = tuple(item.role for item in self.bindings)
        if len(set(roles)) != len(roles):
            raise ValueError("Replay input roles must be unique")


@dataclass(frozen=True)
class ReplayTargetContract:
    operation_id: str
    version: int
    output_kind: Literal["frame", "terminal"]

    def __post_init__(self) -> None:
        _identifier(self.operation_id, "Replay target operation id")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Replay target version must be positive")


def replay_method(
    operation_id: str | None = None,
    *,
    version: int = 1,
    params: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(method: Callable[P, R]) -> Callable[P, R]:
        operation = getattr(method, "__name__") if operation_id is None else operation_id
        signature = inspect.signature(method)

        @wraps(method)
        def semantic_call(*args: P.args, **kwargs: P.kwargs) -> R:
            if _semantic_capture.get() is not None:
                return method(*args, **kwargs)
            if not args or not hasattr(args[0], "_lineage_or_source"):
                return method(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            captured_params = dict(bound.arguments)
            captured_params.pop(next(iter(signature.parameters)), None)
            for name, parameter in signature.parameters.items():
                if parameter.kind is inspect.Parameter.VAR_KEYWORD and name in captured_params:
                    captured_params.update(cast(Mapping[str, Any], captured_params.pop(name)))
            if params is not None:
                captured_params = dict(params(captured_params))
            from wandas.processing.base import FrameMethodOperation, LineageNode

            receiver = cast(Any, args[0])
            target = f"{semantic_call.__module__}.{semantic_call.__qualname__}"
            lineage = LineageNode(
                FrameMethodOperation(operation, captured_params, target, version),
                (receiver._lineage_or_source(),),
            )
            return _invoke_semantic(method, args, kwargs, lineage)

        setattr(semantic_call, "__wandas_replay_target__", ReplayTargetContract(operation, version, "frame"))
        return semantic_call

    return decorate


def semantic_index(method: Callable[P, R]) -> Callable[P, R]:
    """Capture one public indexing intent before helper selections execute."""

    @wraps(method)
    def semantic_call(*args: P.args, **kwargs: P.kwargs) -> R:
        if _semantic_capture.get() is not None or not args:
            return method(*args, **kwargs)
        bound = inspect.signature(method).bind(*args, **kwargs)
        key = bound.arguments.get("key")
        from wandas.processing.base import IndexOperation, LineageNode

        receiver = cast(Any, args[0])
        lineage = LineageNode(
            IndexOperation(receiver._semantic_index_params(key)),
            (receiver._lineage_or_source(),),
        )
        return _invoke_semantic(method, args, kwargs, lineage)

    return semantic_call


def terminal_method(operation_id: str | None = None, *, version: int = 1) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(method: Callable[P, R]) -> Callable[P, R]:
        operation = getattr(method, "__name__") if operation_id is None else operation_id
        setattr(method, "__wandas_replay_target__", ReplayTargetContract(operation, version, "terminal"))
        return method

    return decorate


def freeze_replay_value(value: Any, *, allow_opaque: bool = False) -> ReplayValue:
    """Freeze supported semantic state into a collision-proof tuple tree."""
    if isinstance(value, np.bool_):
        value = bool(value)
    if value is None:
        return ("none",)
    if isinstance(value, str | bool):
        return (type(value).__name__, value)
    if isinstance(value, numbers.Integral):
        return ("int", int(value))
    if isinstance(value, numbers.Real):
        number = float(value)
        encoded = number if math.isfinite(number) else "nan" if math.isnan(number) else "inf" if number > 0 else "-inf"
        return ("float", encoded)
    if isinstance(value, numbers.Complex):
        return ("complex", freeze_replay_value(value.real), freeze_replay_value(value.imag))
    if isinstance(value, slice):
        return (
            "slice",
            freeze_replay_value(value.start, allow_opaque=allow_opaque),
            freeze_replay_value(value.stop, allow_opaque=allow_opaque),
            freeze_replay_value(value.step, allow_opaque=allow_opaque),
        )
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            str(value.dtype),
            tuple(value.shape),
            freeze_replay_value(value.tolist(), allow_opaque=allow_opaque),
        )
    if isinstance(value, DaArray):
        descriptor = {
            "type": "dask.array",
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "chunks": value.chunks,
        }
        return ("dask-descriptor", freeze_replay_value(descriptor))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (
                    key if isinstance(key, str) else freeze_replay_value(key),
                    freeze_replay_value(item, allow_opaque=allow_opaque),
                )
                for key, item in value.items()
            ),
        )
    if isinstance(value, list):
        return ("list", tuple(freeze_replay_value(item, allow_opaque=allow_opaque) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(freeze_replay_value(item, allow_opaque=allow_opaque) for item in value))
    if isinstance(value, set | frozenset):
        kind = "frozenset" if isinstance(value, frozenset) else "set"
        items = tuple(sorted((freeze_replay_value(item, allow_opaque=allow_opaque) for item in value), key=repr))
        return (kind, items)
    if allow_opaque:
        return ("opaque", type(value).__name__)
    raise TypeError(f"Unsupported replay parameter value: {type(value).__name__}")


def thaw_replay_value(value: ReplayValue) -> Any:
    """Return a fresh runtime value from an immutable replay value tree."""
    kind, *payload = value
    if kind == "none":
        return None
    if kind in {"str", "bool", "int"}:
        return payload[0]
    if kind == "float":
        if isinstance(payload[0], str):
            return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[payload[0]]
        return payload[0]
    if kind == "complex":
        return complex(thaw_replay_value(payload[0]), thaw_replay_value(payload[1]))
    if kind == "slice":
        return slice(*(thaw_replay_value(item) for item in payload))
    if kind == "mapping":
        return {
            key if isinstance(key, str) else thaw_replay_value(key): thaw_replay_value(item) for key, item in payload[0]
        }
    if kind in {"list", "tuple", "set", "frozenset"}:
        items = [thaw_replay_value(item) for item in payload[0]]
        if kind == "list":
            return items
        if kind == "tuple":
            return tuple(items)
        return set(items) if kind == "set" else frozenset(items)
    if kind == "ndarray":
        return np.array(thaw_replay_value(payload[2]), dtype=payload[0]).reshape(payload[1])
    if kind == "dask-descriptor":
        return thaw_replay_value(payload[0])
    if kind == "opaque":
        return {"type": "opaque", "name": payload[0]}
    raise TypeError(f"Unknown replay value kind: {kind!r}")


@dataclass(frozen=True)
class ReplayDescriptor:
    contract: OperationContract
    params: ReplayValue

    def __post_init__(self) -> None:
        if self.params[0] != "mapping":
            raise TypeError("Replay descriptor params must be a frozen mapping")

    @property
    def semantic_name(self) -> str:
        """Return the single operation identity owned by the replay contract."""
        return self.contract.operation_id

    def thaw_params(self) -> dict[str, Any]:
        return dict(thaw_replay_value(self.params))


@dataclass(frozen=True)
class SourceReplay(ReplayDescriptor):
    pass


@dataclass(frozen=True)
class AudioReplay(ReplayDescriptor):
    pass


@dataclass(frozen=True)
class MethodReplay(ReplayDescriptor):
    target: str | None = None


@dataclass(frozen=True)
class IndexReplay(ReplayDescriptor):
    """A public selection intent, independent of its internal array slicing."""


@dataclass(frozen=True)
class AddChannelReplay(ReplayDescriptor):
    def __post_init__(self) -> None:
        super().__post_init__()
        input_kinds = tuple(binding.kind for binding in self.contract.bindings)
        if input_kinds not in {("frame", "frame"), ("frame", "array")}:
            raise ValueError("add-channel replay requires ordered frame and frame-or-array bindings")

    @property
    def input_kind(self) -> Literal["frame", "array"]:
        return cast(Literal["frame", "array"], self.contract.bindings[1].kind)


@dataclass(frozen=True)
class TerminalReplay(ReplayDescriptor):
    target: str


@dataclass(frozen=True)
class BinaryReplay(ReplayDescriptor):
    def __post_init__(self) -> None:
        super().__post_init__()
        input_kinds = tuple(binding.kind for binding in self.contract.bindings)
        if len(input_kinds) != 2 or input_kinds.count("frame") not in {1, 2}:
            raise ValueError("binary replay requires two bindings with one or two frame inputs")

    @property
    def scalar_operand(self) -> bool | int | float | complex | np.number[Any] | None:
        """Restore the scalar execution value from canonical replay operand params."""
        operand = self.thaw_params().get("operand")
        if not isinstance(operand, Mapping):
            return None
        operand_type = operand.get("type")
        if not isinstance(operand_type, str):
            return None
        value = operand.get("value")
        builtin_type = {"bool": bool, "int": int, "float": float}.get(operand_type)
        if builtin_type is not None:
            return value if type(value) is builtin_type else None
        if operand_type == "complex":
            real = operand.get("real")
            imaginary = operand.get("imag")
            if isinstance(real, numbers.Real) and isinstance(imaginary, numbers.Real):
                return complex(real, imaginary)
            return None
        try:
            dtype = np.dtype(operand_type)
        except TypeError:
            return None
        if not np.issubdtype(dtype, np.number):
            return None
        # Operand descriptors snapshot NumPy float and complex values through
        # Python scalars, so wider dtypes cannot be restored losslessly.
        if np.issubdtype(dtype, np.floating) and dtype.itemsize > np.dtype(np.float64).itemsize:
            return None
        if np.issubdtype(dtype, np.complexfloating) and dtype.itemsize > np.dtype(np.complex128).itemsize:
            return None
        if "value" in operand:
            return cast(Any, dtype.type(value))
        real = operand.get("real")
        imaginary = operand.get("imag")
        if isinstance(real, numbers.Real) and isinstance(imaginary, numbers.Real):
            return cast(Any, dtype.type(complex(real, imaginary)))
        return None


@dataclass(frozen=True)
class CustomReplay(ReplayDescriptor):
    function: str | None
    output_shape_function: str | None
    output_frame_class: str | None
    output_frame_kwargs: ReplayValue


@dataclass(frozen=True)
class MultiInputReplay(ReplayDescriptor):
    handler: str


@dataclass(frozen=True)
class UnsupportedReplay(ReplayDescriptor):
    reason: str


def frozen_params(params: Mapping[str, Any], *, allow_opaque: bool = False) -> ReplayValue:
    return freeze_replay_value(dict(params), allow_opaque=allow_opaque)
