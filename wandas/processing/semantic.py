"""Immutable semantic provenance and canonical Recipe values."""

from __future__ import annotations

import copy
import json
import struct
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np

InputKind = Literal["frame", "array"]


@dataclass(frozen=True)
class FrozenList:
    """Canonical immutable sequence used in semantic parameters.

    Args:
        items: Already-frozen values in caller order.
    """

    items: tuple[CanonicalValue, ...]


@dataclass(frozen=True)
class FrozenMap:
    """Canonical immutable string-keyed mapping.

    Args:
        entries: Key-value pairs sorted lexicographically by their unique string keys.
    """

    entries: tuple[tuple[str, CanonicalValue], ...]

    def __post_init__(self) -> None:
        """Validate unique, sorted string keys."""
        keys = tuple(key for key, _ in self.entries)
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("Canonical map keys must be strings")
        if len(set(keys)) != len(keys):
            raise ValueError("Canonical map keys must be unique")
        if keys != tuple(sorted(keys)):
            raise ValueError("Canonical map keys must be sorted")


@dataclass(frozen=True)
class FrozenNumber:
    """Bit-preserving Python or NumPy numeric scalar.

    Args:
        kind: Scalar representation family.
        data: Big-endian Python scalar bytes or native NumPy scalar bytes as hex.
        dtype: NumPy dtype string, required only when ``kind`` is ``"numpy"``.
    """

    kind: Literal["python-float", "python-complex", "numpy"]
    data: str
    dtype: str | None = None

    def __post_init__(self) -> None:
        """Validate the scalar tag, dtype relationship, and hex payload."""
        if self.kind == "numpy" and not self.dtype:
            raise ValueError("Canonical NumPy numbers require a dtype")
        if self.kind != "numpy" and self.dtype is not None:
            raise ValueError("Canonical Python numbers do not have a NumPy dtype")
        try:
            bytes.fromhex(self.data)
        except ValueError as exc:
            raise ValueError("Canonical number data must be hexadecimal") from exc


CanonicalValue: TypeAlias = None | bool | int | str | FrozenNumber | FrozenList | FrozenMap


def freeze_value(value: Any) -> CanonicalValue:
    """Freeze a portable parameter without retaining caller-owned containers.

    Args:
        value: Scalar, string-keyed mapping, list, or tuple to snapshot.

    Returns:
        An immutable canonical value that preserves supported numeric bits.

    Raises:
        TypeError: If the value or a nested value is outside the portable grammar.
    """
    if value is None or isinstance(value, str | bool):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return FrozenNumber("python-float", struct.pack(">d", value).hex())
    if type(value) is complex:
        return FrozenNumber("python-complex", struct.pack(">dd", value.real, value.imag).hex())
    if isinstance(value, np.generic):
        if not isinstance(value, np.number | np.bool_):
            raise TypeError(f"Unsupported NumPy Recipe scalar: {type(value).__name__}")
        scalar = np.asarray(value)
        if scalar.dtype.hasobject:
            raise TypeError("Object NumPy scalars are not portable Recipe values")
        return FrozenNumber("numpy", scalar.tobytes().hex(), scalar.dtype.str)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Recipe parameter mappings require string keys")
        return FrozenMap(tuple((key, freeze_value(value[key])) for key in sorted(value)))
    if isinstance(value, tuple | list):
        return FrozenList(tuple(freeze_value(item) for item in value))
    raise TypeError(f"Unsupported Recipe parameter value: {type(value).__name__}")


def freeze_params(params: Mapping[str, Any]) -> FrozenMap:
    """Freeze an operation parameter mapping.

    Args:
        params: Public call parameters to snapshot.

    Returns:
        A canonical immutable mapping.

    Raises:
        TypeError: If the mapping contains non-string keys or unsupported values.
    """
    frozen = freeze_value(params)
    if not isinstance(frozen, FrozenMap):
        raise TypeError("Recipe params must be a mapping")
    return frozen


def thaw_value(value: CanonicalValue) -> Any:
    """Decode a canonical value into a fresh runtime value.

    Args:
        value: Canonical value to decode.

    Returns:
        A scalar or freshly allocated mutable Python container.

    Raises:
        ValueError: If a canonical numeric payload disagrees with its kind or dtype.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, FrozenList):
        return [thaw_value(item) for item in value.items]
    if isinstance(value, FrozenMap):
        return {key: thaw_value(item) for key, item in value.entries}
    raw = bytes.fromhex(value.data)
    if value.kind == "python-float":
        if len(raw) != struct.calcsize(">d"):
            raise ValueError("Canonical Python float data does not match its kind")
        return struct.unpack(">d", raw)[0]
    if value.kind == "python-complex":
        if len(raw) != struct.calcsize(">dd"):
            raise ValueError("Canonical Python complex data does not match its kind")
        real, imaginary = struct.unpack(">dd", raw)
        return complex(real, imaginary)
    dtype = np.dtype(cast(str, value.dtype))
    if dtype.hasobject or len(raw) != dtype.itemsize:
        raise ValueError("Canonical NumPy scalar data does not match its dtype")
    return np.frombuffer(raw, dtype=dtype, count=1)[0]


def thaw_params(params: FrozenMap) -> dict[str, Any]:
    """Decode canonical operation parameters into a fresh dictionary."""
    return cast(dict[str, Any], thaw_value(params))


def value_to_json(value: CanonicalValue) -> Any:
    """Encode a canonical value using a collision-proof tagged JSON grammar.

    Args:
        value: Canonical semantic value.

    Returns:
        A strict-JSON-compatible scalar, mapping, or list representation.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, FrozenNumber):
        payload: dict[str, Any] = {"$type": "number", "kind": value.kind, "data": value.data}
        if value.dtype is not None:
            payload["dtype"] = value.dtype
        return payload
    if isinstance(value, FrozenList):
        return {"$type": "list", "items": [value_to_json(item) for item in value.items]}
    return {
        "$type": "map",
        "entries": [[key, value_to_json(item)] for key, item in value.entries],
    }


def value_from_json(value: Any) -> CanonicalValue:
    """Decode and strictly validate the canonical JSON value grammar.

    Args:
        value: Decoded JSON-like value.

    Returns:
        The corresponding immutable canonical value.

    Raises:
        ValueError: If tags, fields, ordering, or numeric payloads are malformed.
    """
    if value is None or isinstance(value, str | bool) or type(value) is int:
        return value
    if not isinstance(value, Mapping) or not isinstance(value.get("$type"), str):
        raise ValueError("Recipe value is outside the canonical grammar")
    value_type = value["$type"]
    if value_type == "number":
        expected = {"$type", "kind", "data"} | ({"dtype"} if "dtype" in value else set())
        if set(value) != expected:
            raise ValueError("Canonical number fields are malformed")
        kind = value.get("kind")
        data = value.get("data")
        dtype = value.get("dtype")
        if kind not in {"python-float", "python-complex", "numpy"} or not isinstance(data, str):
            raise ValueError("Canonical number kind or data is malformed")
        if dtype is not None and not isinstance(dtype, str):
            raise ValueError("Canonical NumPy dtype is malformed")
        number = FrozenNumber(kind, data, dtype)
        thaw_value(number)
        return number
    if value_type == "list":
        if set(value) != {"$type", "items"} or not isinstance(value.get("items"), list):
            raise ValueError("Canonical list fields are malformed")
        return FrozenList(tuple(value_from_json(item) for item in value["items"]))
    if value_type == "map":
        if set(value) != {"$type", "entries"} or not isinstance(value.get("entries"), list):
            raise ValueError("Canonical map fields are malformed")
        entries: list[tuple[str, CanonicalValue]] = []
        for entry in value["entries"]:
            if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
                raise ValueError("Canonical map entry is malformed")
            entries.append((entry[0], value_from_json(entry[1])))
        return FrozenMap(tuple(entries))
    raise ValueError(f"Unknown Recipe value tag: {value_type!r}")


def _display_value(value: CanonicalValue) -> Any:
    """Return a strict-JSON display value without exposing container backends."""
    runtime = thaw_value(value)
    if runtime is None or isinstance(runtime, str | bool) or type(runtime) is int:
        return runtime
    if isinstance(runtime, np.generic):
        return value_to_json(value)
    if type(runtime) is float:
        if np.isfinite(runtime):
            return runtime
        return value_to_json(value)
    if type(runtime) is complex:
        return value_to_json(value)
    if isinstance(value, FrozenList):
        return [_display_value(item) for item in value.items]
    if isinstance(value, FrozenMap):
        return {key: _display_value(item) for key, item in value.entries}
    raise TypeError(f"Unsupported display value: {type(value).__name__}")


def params_to_display(params: FrozenMap) -> dict[str, Any]:
    """Return a defensive, strict-JSON-safe history parameter mapping."""
    return {key: _display_value(value) for key, value in params.entries}


def _identifier(value: object, label: str) -> str:
    """Validate and return a non-blank semantic identifier."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


@dataclass(frozen=True)
class InputBinding:
    """Role and runtime kind of one ordered semantic operation input.

    Args:
        role: Unique human-readable role within an operation invocation.
        kind: ``"frame"`` for lineage-bearing inputs or ``"array"`` for external
            NumPy/Dask inputs.
    """

    role: str
    kind: InputKind

    def __post_init__(self) -> None:
        """Validate the role and supported input kind."""
        _identifier(self.role, "Semantic input role")
        if self.kind not in {"frame", "array"}:
            raise ValueError("Semantic input kind must be 'frame' or 'array'")


@dataclass(frozen=True)
class SemanticOperation:
    """Complete immutable replay intent for one public operation call.

    Args:
        operation_id: Stable identifier shared with a Recipe registry definition.
        version: Positive operation contract version.
        bindings: Ordered input roles and runtime kinds selected by the call.
        params: Canonical snapshot of portable public parameters.
    """

    operation_id: str
    version: int
    bindings: tuple[InputBinding, ...]
    params: FrozenMap

    def __post_init__(self) -> None:
        """Validate operation identity, bindings, and canonical parameters."""
        _identifier(self.operation_id, "Semantic operation id")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Semantic operation version must be a positive integer")
        if not isinstance(self.bindings, tuple) or not all(isinstance(item, InputBinding) for item in self.bindings):
            raise TypeError("Semantic bindings must be a tuple of InputBinding values")
        roles = tuple(binding.role for binding in self.bindings)
        if len(set(roles)) != len(roles):
            raise ValueError("Semantic input roles must be unique")
        if not isinstance(self.params, FrozenMap):
            raise TypeError("Semantic operation params must be canonical")


@dataclass(frozen=True)
class HistoryRecord:
    """JSON-projectable display history for one completed public operation.

    Args:
        operation: Stable operation identifier.
        version: Positive operation contract version.
        params: Canonical parameter snapshot without runtime container details.
    """

    operation: str
    version: int
    params: FrozenMap

    def __post_init__(self) -> None:
        """Validate operation identity and canonical parameters."""
        _identifier(self.operation, "History operation id")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("History operation version must be a positive integer")
        if not isinstance(self.params, FrozenMap):
            raise TypeError("History params must be canonical")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh strict-JSON-safe display record."""
        return {
            "operation": self.operation,
            "version": self.version,
            "params": params_to_display(self.params),
        }


@dataclass(frozen=True)
class LineageNode:
    """Source or operation node that is the sole Frame provenance state.

    Args:
        operation: Immutable public operation intent, or ``None`` for a source.
        inputs: Ordered lineage parents; external array bindings use ``None``.
        history_prefix: Persisted display-only records allowed on source nodes.
        recipe_error: Optional atomic Recipe-extraction rejection reason attached to
            an otherwise valid public operation node.
    """

    operation: SemanticOperation | None = None
    inputs: tuple[LineageNode | None, ...] = ()
    history_prefix: tuple[HistoryRecord, ...] = ()
    recipe_error: str | None = None

    def __post_init__(self) -> None:
        """Validate source and operation-node invariants."""
        if not isinstance(self.inputs, tuple):
            raise TypeError("Lineage inputs must be a tuple")
        if self.operation is None:
            if self.inputs:
                raise ValueError("Source lineage cannot have input edges")
            if self.recipe_error is not None:
                raise ValueError("Source lineage cannot carry a Recipe error")
            if not all(isinstance(record, HistoryRecord) for record in self.history_prefix):
                raise TypeError("Source history prefix must contain HistoryRecord values")
            return
        if self.history_prefix:
            raise ValueError("Only source lineage may carry a display-history prefix")
        if len(self.inputs) != len(self.operation.bindings):
            raise ValueError("Semantic lineage edges must match operation bindings")
        for binding, parent in zip(self.operation.bindings, self.inputs):
            if binding.kind == "frame" and not isinstance(parent, LineageNode):
                raise TypeError(f"Frame binding {binding.role!r} requires a lineage parent")
            if binding.kind == "array" and parent is not None:
                raise TypeError(f"Array binding {binding.role!r} cannot have a lineage parent")
        if self.recipe_error is not None and (not isinstance(self.recipe_error, str) or not self.recipe_error):
            raise TypeError("Lineage Recipe errors must be non-empty strings")


def source_lineage(history: Sequence[Mapping[str, Any]] = ()) -> LineageNode:
    """Create source lineage with an optional persisted display prefix.

    Args:
        history: Strict operation-history mappings loaded from persistence.

    Returns:
        A source node containing defensive canonical copies of the records.

    Raises:
        ValueError: If a persisted record does not match the display-history contract.
    """
    records: list[HistoryRecord] = []
    for raw_record in history:
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"operation", "version", "params"}:
            raise ValueError("Persisted operation history record is malformed")
        params = raw_record["params"]
        if not isinstance(params, Mapping):
            raise ValueError("Persisted operation history params must be a mapping")
        records.append(
            HistoryRecord(
                cast(str, raw_record["operation"]),
                cast(int, raw_record["version"]),
                freeze_params(cast(Mapping[str, Any], copy.deepcopy(params))),
            )
        )
    return LineageNode(history_prefix=tuple(records))


def lineage_history(lineage: LineageNode) -> list[dict[str, Any]]:
    """Project lineage to deterministic depth-first display history.

    Shared lineage nodes are emitted once, parents precede consumers, and the current
    operation is last.

    Args:
        lineage: Output lineage root to project.

    Returns:
        A fresh strict-JSON-compatible list of operation records.
    """
    records: list[HistoryRecord] = []
    seen: set[int] = set()

    def visit(node: LineageNode) -> None:
        """Append each reachable source or operation record once."""
        identity = id(node)
        if identity in seen:
            return
        seen.add(identity)
        if node.operation is None:
            records.extend(node.history_prefix)
            return
        for parent in node.inputs:
            if parent is not None:
                visit(parent)
        records.append(HistoryRecord(node.operation.operation_id, node.operation.version, node.operation.params))

    visit(lineage)
    result = [record.to_dict() for record in records]
    json.dumps(result, allow_nan=False)
    return result


_semantic_capture: ContextVar[LineageNode | None] = ContextVar("wandas_semantic_capture", default=None)


def active_semantic_lineage() -> LineageNode | None:
    """Return the authoritative public-operation node for the current context."""
    return _semantic_capture.get()


@contextmanager
def semantic_lineage(lineage: LineageNode) -> Any:
    """Make an already-final semantic node authoritative for nested helpers.

    Args:
        lineage: Node that every Frame result created in the context must preserve.

    Yields:
        Control to the public operation or Recipe handler.
    """
    token = _semantic_capture.set(lineage)
    try:
        yield
    finally:
        _semantic_capture.reset(token)


def has_frame_lineage_contract(value: object) -> bool:
    """Return whether a value structurally exposes the BaseFrame lineage contract."""
    members = {name for base in type(value).__mro__ for name in base.__dict__}
    return {"lineage", "_create_new_instance"} <= members


def invoke_semantic(call: Any, lineage: LineageNode, *args: Any, **kwargs: Any) -> Any:
    """Invoke a public operation atomically under authoritative lineage.

    Args:
        call: Public operation implementation to invoke.
        lineage: Final node that any Frame-like result must preserve by identity.
        *args: Positional arguments forwarded to ``call``.
        **kwargs: Keyword arguments forwarded to ``call``.

    Returns:
        The operation result.

    Raises:
        RuntimeError: If a Frame-like result does not preserve ``lineage``.
    """
    with semantic_lineage(lineage):
        result = call(*args, **kwargs)
    if has_frame_lineage_contract(result) and getattr(result, "lineage", None) is not lineage:
        raise RuntimeError("Public operation did not preserve its authoritative semantic lineage")
    return result
