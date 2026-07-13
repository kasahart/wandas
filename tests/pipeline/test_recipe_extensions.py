"""Executable probes for the three Recipe extension KPI families."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import wandas.processing.base as processing_base
from wandas.frames.channel import ChannelFrame
from wandas.pipeline.calls import AudioCall, MethodCall, MultiInputCall
from wandas.pipeline.codecs import default_codec_registry
from wandas.pipeline.decorators import multi_input_handler
from wandas.processing.base import AudioOperation, LineageNode
from wandas.processing.semantic import (
    AudioReplay,
    InputBinding,
    MethodReplay,
    MultiInputReplay,
    OperationContract,
    frozen_params,
)


class RecipeProbeUnary(AudioOperation[Any, Any]):
    name = "recipe_probe_unary"
    supports_generic_replay = True

    def _process(self, data: Any) -> Any:
        return data


class ProbeTransitionOwner:
    def probe_transition(self) -> None:
        return None


@multi_input_handler("recipe_probe_multi", version=1, roles=("signal", "noise"))
def probe_multi_handler(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
    return inputs[0] + inputs[1] * params["gain"]


def test_new_unary_same_frame_uses_existing_audio_family(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(processing_base._OPERATION_REGISTRY, RecipeProbeUnary.name, RecipeProbeUnary)
    descriptor = AudioReplay(
        OperationContract("recipe_probe_unary", 1, True, (InputBinding("frame", "frame"),)),
        frozen_params({}),
        "recipe_probe_unary",
        True,
    )

    result = default_codec_registry().encode(descriptor, (LineageNode.__new__(LineageNode),))

    assert isinstance(result.call, AudioCall) and result.call.operation == "recipe_probe_unary"


def test_new_typed_transition_uses_existing_method_family() -> None:
    target = f"{__name__}.ProbeTransitionOwner.probe_transition"
    descriptor = MethodReplay(
        OperationContract("probe_transition", 1, True, (InputBinding("frame", "frame"),)),
        frozen_params({}),
        "probe_transition",
        target,
    )

    result = default_codec_registry().encode(descriptor, (LineageNode.__new__(LineageNode),))

    assert isinstance(result.call, MethodCall)


def test_new_true_multi_input_uses_existing_multi_family() -> None:
    roles = ("signal", "noise")
    descriptor = MultiInputReplay(
        OperationContract(
            "recipe_probe_multi",
            1,
            True,
            tuple(InputBinding(role, "frame") for role in roles),
        ),
        frozen_params({"gain": 0.5}),
        "recipe_probe_multi",
        f"{__name__}.probe_multi_handler",
        roles,
    )
    fake = (LineageNode.__new__(LineageNode), LineageNode.__new__(LineageNode))

    result = default_codec_registry().encode(descriptor, fake)
    source = ChannelFrame.from_numpy(np.ones((1, 8)), sampling_rate=8000)

    assert isinstance(result.call, MultiInputCall)
    np.testing.assert_allclose(result.call.invoke((source, source)).compute(), 1.5)
