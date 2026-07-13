from dataclasses import dataclass

import pytest

from wandas.pipeline.calls import AudioCall
from wandas.pipeline.codecs import BoundInput, CodecResult, ReplayCodecRegistry
from wandas.processing.semantic import InputBinding, OperationContract, ReplayDescriptor, frozen_params


@dataclass(frozen=True)
class ProbeReplay(ReplayDescriptor):
    pass


def _probe() -> ProbeReplay:
    return ProbeReplay(
        OperationContract("normalize", 1, True, (InputBinding("frame", "frame"),)),
        frozen_params({}),
        "normalize",
    )


def test_custom_registry_must_be_built_before_freeze() -> None:
    registry = ReplayCodecRegistry()
    registry.register(
        ProbeReplay,
        lambda _descriptor, _inputs: CodecResult(AudioCall("normalize"), (BoundInput("frame", "frame"),)),
    )
    registry.freeze()

    assert isinstance(registry.encode(_probe(), ()).call, AudioCall)
    with pytest.raises(TypeError, match="Frozen"):
        registry.register(
            ReplayDescriptor,
            lambda _descriptor, _inputs: CodecResult(AudioCall("normalize"), (BoundInput("frame", "frame"),)),
        )


def test_duplicate_codec_registration_is_rejected() -> None:
    registry = ReplayCodecRegistry()

    def codec(_descriptor, _inputs):
        return CodecResult(AudioCall("normalize"), (BoundInput("frame", "frame"),))

    registry.register(ProbeReplay, codec)
    with pytest.raises(ValueError, match="already"):
        registry.register(ProbeReplay, codec)
