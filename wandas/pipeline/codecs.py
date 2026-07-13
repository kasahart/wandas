"""Typed ReplayDescriptor codecs used by the generic lineage compiler."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, cast

import numpy as np

from wandas.pipeline.calls import (
    AddChannelCall,
    AudioCall,
    BinaryCall,
    CustomCall,
    ExternalArrayCall,
    IndexCall,
    MethodCall,
    MultiInputCall,
    ScalarCall,
    TerminalCall,
)
from wandas.pipeline.errors import RecipeExtractionError
from wandas.pipeline.model import RecipeCall
from wandas.processing.base import LineageNode
from wandas.processing.semantic import (
    AddChannelReplay,
    AudioReplay,
    BinaryReplay,
    CustomReplay,
    IndexReplay,
    MethodReplay,
    MultiInputReplay,
    ReplayDescriptor,
    TerminalReplay,
    UnsupportedReplay,
)


@dataclass(frozen=True)
class BoundInput:
    role: str
    kind: Literal["frame", "array"]
    lineage: LineageNode | None = None


@dataclass(frozen=True)
class CodecResult:
    call: RecipeCall
    bindings: tuple[BoundInput, ...]


ReplayCodec = Callable[[ReplayDescriptor, tuple[LineageNode, ...]], CodecResult]


class ReplayCodecRegistry:
    def __init__(self) -> None:
        self._codecs: dict[type[ReplayDescriptor], ReplayCodec] = {}
        self._frozen = False

    def register(self, descriptor_type: type[ReplayDescriptor], codec: ReplayCodec) -> None:
        if self._frozen:
            raise TypeError("Frozen Recipe codec registries cannot be modified")
        if descriptor_type in self._codecs:
            raise ValueError(f"Recipe codec is already registered: {descriptor_type.__name__}")
        self._codecs[descriptor_type] = codec

    def freeze(self) -> ReplayCodecRegistry:
        self._frozen = True
        return self

    def encode(self, descriptor: ReplayDescriptor, lineage_inputs: tuple[LineageNode, ...]) -> CodecResult:
        if isinstance(descriptor, UnsupportedReplay):
            raise RecipeExtractionError(descriptor.reason)
        codec = self._codecs.get(type(descriptor))
        if codec is None:
            raise RecipeExtractionError(f"No Recipe codec for descriptor: {type(descriptor).__name__}")
        expected_frame_count = sum(binding.kind == "frame" for binding in descriptor.contract.bindings)
        if len(lineage_inputs) != expected_frame_count:
            raise RecipeExtractionError(
                "Recipe descriptor frame bindings and lineage inputs disagree\n"
                f"  Expected: {expected_frame_count}; got: {len(lineage_inputs)}"
            )
        result = codec(descriptor, lineage_inputs)
        expected_bindings = tuple(
            (binding.role, binding.kind) for binding in descriptor.contract.bindings if binding.kind != "scalar"
        )
        codec_bindings = tuple((binding.role, binding.kind) for binding in result.bindings)
        if expected_bindings != codec_bindings or len(codec_bindings) != result.call.arity:
            raise RecipeExtractionError(
                f"Recipe codec bindings disagree with descriptor\n  Expected: {expected_bindings}"
            )
        codec_frame_lineages = tuple(binding.lineage for binding in result.bindings if binding.kind == "frame")
        frame_lineage_mismatch = any(
            codec_lineage is not descriptor_lineage
            for codec_lineage, descriptor_lineage in zip(codec_frame_lineages, lineage_inputs, strict=True)
        )
        array_has_lineage = any(binding.lineage is not None for binding in result.bindings if binding.kind == "array")
        if frame_lineage_mismatch or array_has_lineage:
            raise RecipeExtractionError("Recipe codec input lineage disagrees with descriptor inputs")
        return result


@cache
def default_codec_registry() -> ReplayCodecRegistry:
    registry = ReplayCodecRegistry()
    registry.register(AudioReplay, _audio)
    registry.register(MethodReplay, _method)
    registry.register(IndexReplay, _index)
    registry.register(AddChannelReplay, _add_channel)
    registry.register(TerminalReplay, _terminal)
    registry.register(BinaryReplay, _binary)
    registry.register(CustomReplay, _custom)
    registry.register(MultiInputReplay, _multi)
    return registry.freeze()


def _frame(role: str, lineage: LineageNode) -> BoundInput:
    return BoundInput(role, "frame", lineage)


def _audio(descriptor: ReplayDescriptor, lineage_inputs: tuple[LineageNode, ...]) -> CodecResult:
    descriptor = cast(AudioReplay, descriptor)
    return CodecResult(
        AudioCall(descriptor.contract.operation_id, descriptor.thaw_params(), descriptor.contract.version),
        (_frame(descriptor.contract.bindings[0].role, lineage_inputs[0]),),
    )


def _binary(descriptor: ReplayDescriptor, lineage_inputs: tuple[LineageNode, ...]) -> CodecResult:
    descriptor = cast(BinaryReplay, descriptor)
    operation = descriptor.contract.operation_id
    bindings = descriptor.contract.bindings
    if all(binding.kind == "frame" for binding in bindings):
        return CodecResult(
            BinaryCall(operation, descriptor.contract.version),
            tuple(_frame(binding.role, lineage) for binding, lineage in zip(bindings, lineage_inputs, strict=True)),
        )
    frame_binding = next(binding for binding in bindings if binding.kind == "frame")
    frame_lineage = lineage_inputs[0]
    if any(binding.kind == "scalar" for binding in bindings):
        scalar_operand = descriptor.scalar_operand
        if scalar_operand is None:
            raise RecipeExtractionError("Unsupported scalar Recipe operand")
        scalar_is_left_operand = bindings[0].kind == "scalar"
        return CodecResult(
            ScalarCall(
                operation,
                scalar_operand,
                scalar_is_left_operand,
                descriptor.contract.version,
            ),
            (_frame(frame_binding.role, frame_lineage),),
        )
    array_index = next(index for index, binding in enumerate(bindings) if binding.kind == "array")
    result_bindings = tuple(
        BoundInput(binding.role, "array") if binding.kind == "array" else _frame(binding.role, frame_lineage)
        for binding in bindings
    )
    return CodecResult(ExternalArrayCall(operation, array_index, descriptor.contract.version), result_bindings)


def _slice(value: Mapping[str, Any]) -> slice:
    return slice(value.get("start"), value.get("stop"), value.get("step"))


def _selector(value: Mapping[str, Any]) -> Any:
    kind = value.get("type", value.get("indexing"))
    if kind == "integer":
        return int(value["index"])
    if kind == "label":
        return value["label"]
    if kind == "channel_slice":
        return _slice(value)
    if kind in {"integer_list", "integer_array"}:
        items = [int(item) for item in value["indices"]]
        return items if kind == "integer_list" else np.array(items, dtype=int)
    if kind == "label_list":
        return list(value["labels"])
    if kind == "boolean_mask":
        return np.array(value["mask"], dtype=bool)
    raise RecipeExtractionError(f"Unsupported Recipe selector: {kind!r}")


def _method(descriptor: ReplayDescriptor, lineage_inputs: tuple[LineageNode, ...]) -> CodecResult:
    descriptor = cast(MethodReplay, descriptor)
    operation = descriptor.contract.operation_id
    params = descriptor.thaw_params()
    if descriptor.target is None:
        raise RecipeExtractionError(f"Recipe method has no stable target: {operation!r}")
    return CodecResult(
        MethodCall(operation, descriptor.target, params, descriptor.contract.version),
        (_frame(descriptor.contract.bindings[0].role, lineage_inputs[0]),),
    )


def _index(descriptor: ReplayDescriptor, lineage_inputs: tuple[LineageNode, ...]) -> CodecResult:
    descriptor = cast(IndexReplay, descriptor)
    params = descriptor.thaw_params()
    if params.get("indexing") == "multidimensional_slice":
        key = (_selector(params["channel"]), *(_slice(item) for item in params["axis_slices"]))
    elif params.get("indexing") == "multidimensional":
        raise RecipeExtractionError("Unsupported multidimensional Recipe indexing")
    else:
        key = _selector(params)
    return CodecResult(IndexCall(key, descriptor.contract.version), (_frame("frame", lineage_inputs[0]),))


def _add_channel(descriptor: ReplayDescriptor, lineage_inputs: tuple[LineageNode, ...]) -> CodecResult:
    descriptor = cast(AddChannelReplay, descriptor)
    bindings = descriptor.contract.bindings
    input_kind = descriptor.input_kind
    if input_kind == "frame":
        resolved = tuple(
            _frame(binding.role, lineage) for binding, lineage in zip(bindings, lineage_inputs, strict=True)
        )
    else:
        resolved = (
            _frame(bindings[0].role, lineage_inputs[0]),
            BoundInput(bindings[1].role, "array"),
        )
    return CodecResult(AddChannelCall(input_kind, descriptor.thaw_params(), descriptor.contract.version), resolved)


def _terminal(descriptor: ReplayDescriptor, lineage_inputs: tuple[LineageNode, ...]) -> CodecResult:
    descriptor = cast(TerminalReplay, descriptor)
    return CodecResult(
        TerminalCall(descriptor.contract.operation_id, descriptor.target, descriptor.contract.version),
        (_frame(descriptor.contract.bindings[0].role, lineage_inputs[0]),),
    )


def _custom(descriptor: ReplayDescriptor, lineage_inputs: tuple[LineageNode, ...]) -> CodecResult:
    if not isinstance(descriptor, CustomReplay) or descriptor.function is None:
        raise RecipeExtractionError("Custom Recipe replay requires stable callable paths")
    return CodecResult(
        CustomCall(
            descriptor.function,
            descriptor.params,
            descriptor.output_shape_function,
            descriptor.output_frame_class,
            descriptor.output_frame_kwargs,
            descriptor.contract.pure,
            descriptor.contract.version,
        ),
        (_frame(descriptor.contract.bindings[0].role, lineage_inputs[0]),),
    )


def _multi(descriptor: ReplayDescriptor, lineage_inputs: tuple[LineageNode, ...]) -> CodecResult:
    if not isinstance(descriptor, MultiInputReplay) or not descriptor.handler:
        raise RecipeExtractionError("Multi-input Recipe replay requires a stable handler")
    executable_bindings = tuple(binding for binding in descriptor.contract.bindings if binding.kind != "scalar")
    roles = tuple(binding.role for binding in executable_bindings)
    lineage_iterator = iter(lineage_inputs)
    bindings = tuple(
        _frame(binding.role, next(lineage_iterator)) if binding.kind == "frame" else BoundInput(binding.role, "array")
        for binding in executable_bindings
    )
    return CodecResult(
        MultiInputCall(
            descriptor.contract.operation_id,
            roles,
            descriptor.handler,
            descriptor.params,
            descriptor.contract.version,
            input_kinds=cast(
                tuple[Literal["frame", "array"], ...],
                tuple(binding.kind for binding in executable_bindings),
            ),
        ),
        bindings,
    )
