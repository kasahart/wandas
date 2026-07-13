from dataclasses import dataclass

import pytest

from wandas.pipeline.calls import AudioCall, ScalarCall
from wandas.pipeline.codecs import BoundInput, CodecResult, ReplayCodecRegistry, default_codec_registry
from wandas.pipeline.errors import RecipeExtractionError
from wandas.processing.base import AddChannelOperation, FrameSourceOperation, LineageNode
from wandas.processing.semantic import (
    AddChannelReplay,
    BinaryReplay,
    InputBinding,
    MultiInputReplay,
    OperationContract,
    ReplayDescriptor,
    frozen_params,
)


@dataclass(frozen=True)
class ProbeReplay(ReplayDescriptor):
    pass


def _probe() -> ProbeReplay:
    return ProbeReplay(
        OperationContract("normalize", 1, True, (InputBinding("frame", "frame"),)),
        frozen_params({}),
    )


def _source_lineage() -> LineageNode:
    return LineageNode(FrameSourceOperation())


def test_custom_registry_must_be_built_before_freeze() -> None:
    registry = ReplayCodecRegistry()

    def codec(_descriptor: ReplayDescriptor, inputs: tuple[LineageNode, ...]) -> CodecResult:
        return CodecResult(AudioCall("normalize"), (BoundInput("frame", "frame", inputs[0]),))

    registry.register(
        ProbeReplay,
        codec,
    )
    registry.freeze()

    assert isinstance(registry.encode(_probe(), (_source_lineage(),)).call, AudioCall)
    with pytest.raises(TypeError, match="Frozen"):
        registry.register(
            ReplayDescriptor,
            codec,
        )


def test_duplicate_codec_registration_is_rejected() -> None:
    registry = ReplayCodecRegistry()

    def codec(_descriptor, _inputs):
        return CodecResult(AudioCall("normalize"), (BoundInput("frame", "frame"),))

    registry.register(ProbeReplay, codec)
    with pytest.raises(ValueError, match="already"):
        registry.register(ProbeReplay, codec)


@pytest.mark.parametrize("lineage_count", [0, 2])
def test_registry_rejects_missing_or_extra_frame_lineage_before_codec(lineage_count: int) -> None:
    registry = ReplayCodecRegistry()

    def fail_codec(_descriptor: ReplayDescriptor, _inputs: tuple[LineageNode, ...]) -> CodecResult:
        raise AssertionError("codec must not run for malformed lineage")

    registry.register(ProbeReplay, fail_codec)
    lineage_inputs = tuple(_source_lineage() for _ in range(lineage_count))

    with pytest.raises(RecipeExtractionError, match="frame bindings and lineage inputs disagree"):
        registry.encode(_probe(), lineage_inputs)


def test_registry_rejects_codec_that_substitutes_frame_lineage() -> None:
    expected_lineage = _source_lineage()
    substituted_lineage = _source_lineage()
    registry = ReplayCodecRegistry()
    registry.register(
        ProbeReplay,
        lambda _descriptor, _inputs: CodecResult(
            AudioCall("normalize"),
            (BoundInput("frame", "frame", substituted_lineage),),
        ),
    )

    with pytest.raises(RecipeExtractionError, match="codec input lineage disagrees"):
        registry.encode(_probe(), (expected_lineage,))


def test_add_channel_replay_rejects_malformed_binding_shape() -> None:
    with pytest.raises(ValueError, match="ordered frame and frame-or-array bindings"):
        AddChannelReplay(
            OperationContract("add_channel", 1, True, (InputBinding("base", "frame"),)),
            frozen_params({}),
        )


def test_binary_replay_rejects_malformed_binding_shape() -> None:
    with pytest.raises(ValueError, match="two bindings with one or two frame inputs"):
        BinaryReplay(
            OperationContract("+", 1, True, (InputBinding("frame", "frame"),)),
            frozen_params({}),
        )


def test_binary_replay_derives_operation_kind_and_order_from_contract() -> None:
    descriptor = BinaryReplay(
        OperationContract(
            "-",
            1,
            True,
            (InputBinding("operand", "scalar"), InputBinding("frame", "frame")),
        ),
        frozen_params({"operand": {"type": "int", "value": 2}}),
    )

    call = default_codec_registry().encode(descriptor, (_source_lineage(),)).call

    assert isinstance(call, ScalarCall)
    assert call.operation == "-"
    assert call.operand == 2
    assert call.reverse is True
    assert descriptor.thaw_params() == {"operand": {"type": "int", "value": 2}}
    assert (
        not {"semantic_name", "symbol", "operand_kind", "operand_position", "scalar_operand"} & vars(descriptor).keys()
    )


def test_replay_descriptors_do_not_store_contract_derived_input_state() -> None:
    multi_input = MultiInputReplay(
        OperationContract(
            "mix",
            1,
            True,
            (InputBinding("signal", "frame"), InputBinding("noise", "array")),
        ),
        frozen_params({}),
        "tests.pipeline.test_recipe_codecs.handler",
    )
    add_channel = AddChannelOperation({}, "array").replay_descriptor()

    assert "roles" not in vars(multi_input)
    assert "input_kind" not in vars(add_channel)
    assert multi_input.semantic_name == "mix"
    assert "semantic_name" not in vars(multi_input)
