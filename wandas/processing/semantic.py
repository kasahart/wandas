"""Immutable semantic replay contracts captured by runtime lineage."""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, ParamSpec, TypeAlias, TypeVar

import numpy as np
from dask.array.core import Array as DaArray

InputKind = Literal["frame", "array", "scalar"]
ReplayValue: TypeAlias = tuple[Any, ...]
P = ParamSpec("P")
R = TypeVar("R")


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


def replay_method(operation_id: str | None = None, *, version: int = 1) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(method: Callable[P, R]) -> Callable[P, R]:
        operation = getattr(method, "__name__") if operation_id is None else operation_id
        setattr(method, "__wandas_replay_target__", ReplayTargetContract(operation, version, "frame"))
        return method

    return decorate


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
    semantic_name: str

    def __post_init__(self) -> None:
        _identifier(self.semantic_name, "Replay semantic name")
        if self.params[0] != "mapping":
            raise TypeError("Replay descriptor params must be a frozen mapping")

    def thaw_params(self) -> dict[str, Any]:
        return dict(thaw_replay_value(self.params))


@dataclass(frozen=True)
class SourceReplay(ReplayDescriptor):
    pass


@dataclass(frozen=True)
class AudioReplay(ReplayDescriptor):
    generic: bool


@dataclass(frozen=True)
class MethodReplay(ReplayDescriptor):
    target: str | None = None
    call_params: ReplayValue | None = None


@dataclass(frozen=True)
class IndexReplay(ReplayDescriptor):
    """A public selection intent, independent of its internal array slicing."""


@dataclass(frozen=True)
class AddChannelReplay(ReplayDescriptor):
    input_kind: Literal["frame", "array"]


@dataclass(frozen=True)
class TerminalReplay(ReplayDescriptor):
    target: str


@dataclass(frozen=True)
class BinaryReplay(ReplayDescriptor):
    symbol: str
    operand_kind: Literal["frame", "scalar", "array"]
    operand_position: Literal["left", "right"]
    scalar_operand: bool | int | float | complex | None = None


@dataclass(frozen=True)
class CustomReplay(ReplayDescriptor):
    function: str | None
    output_shape_function: str | None
    output_frame_class: str | None
    output_frame_kwargs: ReplayValue


@dataclass(frozen=True)
class MultiInputReplay(ReplayDescriptor):
    handler: str
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        binding_roles = tuple(binding.role for binding in self.contract.bindings if binding.kind != "scalar")
        if not self.roles or len(set(self.roles)) != len(self.roles) or self.roles != binding_roles:
            raise ValueError("Multi-input roles must exactly match ordered non-scalar bindings")


@dataclass(frozen=True)
class UnsupportedReplay(ReplayDescriptor):
    reason: str


def frozen_params(params: Mapping[str, Any], *, allow_opaque: bool = False) -> ReplayValue:
    return freeze_replay_value(dict(params), allow_opaque=allow_opaque)


def method_replay_params(operation: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Apply declarative adapters for legacy runtime/public signature differences."""
    if operation in {"ifft", "istft"}:
        return {}
    result = dict(params)
    policy = {
        "rename_channels": ({"mapping_items": "mapping"}, ()),
        "get_channel": ({}, ("query_kind",)),
        "welch": ({}, ("detrend",)),
    }.get(operation)
    if policy is None:
        return result
    renames, drops = policy
    for old, new in renames.items():
        if old in result:
            value = result.pop(old)
            result[new] = dict(value) if old == "mapping_items" else value
    if operation == "get_channel" and "channel_mask" in result:
        mask = result.pop("channel_mask")
        result["channel_idx"] = [index for index, selected in enumerate(mask) if selected]
    for name in drops:
        result.pop(name, None)
    return result
