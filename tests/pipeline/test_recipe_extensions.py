from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import (
    OperationCapture,
    RecipePlan,
    RecipeRegistry,
    default_recipe_registry,
    recipe_definition,
    recipe_operation,
)
from wandas.processing.base import AudioOperation
from wandas.processing.semantic import InputBinding
from wandas.utils.types import NDArrayReal


class ProbeGain(AudioOperation[NDArrayReal, NDArrayReal]):
    """Test-only numerical operation used by an extension probe."""

    name = "test_gain"

    def __init__(self, sampling_rate: float, gain: float) -> None:
        super().__init__(sampling_rate, gain=gain)

    def process(self, data: Any, *inputs: Any) -> Any:
        return data * cast(float, self._config_value("gain"))


class TypedChannelFrame(ChannelFrame):
    """Distinct output family for the typed-transition probe."""


def _capture_frame_pair(args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
    left = cast("ExtensionChannelFrame", args[0])
    right = params["other"]
    if not isinstance(right, ChannelFrame):
        raise TypeError("other must be a ChannelFrame")
    return OperationCapture(
        (InputBinding("left", "frame"), InputBinding("right", "frame")),
        (left.lineage, right.lineage),
        {key: value for key, value in params.items() if key != "other"},
    )


def _difference_handler(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
    return inputs[0].difference(inputs[1], **dict(params))


class ExtensionChannelFrame(ChannelFrame):
    @recipe_operation("tests.audio.gain", version=2)
    def test_gain(self, gain: float) -> ExtensionChannelFrame:
        operation = ProbeGain(self.sampling_rate, gain)
        return cast(
            "ExtensionChannelFrame",
            self._apply_operation_instance(operation, operation_name="test_gain"),
        )

    @recipe_operation("tests.transition.typed")
    def to_typed(self, scale: float = 1.0) -> TypedChannelFrame:
        return TypedChannelFrame(
            data=self._data * scale,
            sampling_rate=self.sampling_rate,
            label=self.label,
            metadata=self.metadata,
            channel_metadata=self.channels.to_list(),
            channel_ids=self._channel_ids,
            source_time_offset=self.source_time_offset,
            lineage=self._required_semantic_lineage(),
            previous=self,
        )

    @recipe_operation(
        "tests.frame.difference",
        bindings=(InputBinding("left", "frame"), InputBinding("right", "frame")),
        capture=_capture_frame_pair,
        handler=_difference_handler,
    )
    def difference(self, other: ChannelFrame, *, scale: float = 1.0) -> ExtensionChannelFrame:
        return self._create_new_instance(
            data=(self._data - other._data) * scale,
            lineage=self._required_semantic_lineage(),
        )


def _registry() -> RecipeRegistry:
    registry = default_recipe_registry()
    for method in (
        ExtensionChannelFrame.test_gain,
        ExtensionChannelFrame.to_typed,
        ExtensionChannelFrame.difference,
    ):
        registry = registry.with_operation(recipe_definition(method))
    return registry


def _frame(value: float) -> ExtensionChannelFrame:
    return cast(
        ExtensionChannelFrame,
        ExtensionChannelFrame.from_numpy(np.full((1, 16), value), sampling_rate=8000),
    )


def _roundtrip(processed: ChannelFrame, inputs: dict[str, ChannelFrame]) -> ChannelFrame:
    registry = _registry()
    plan = RecipePlan.from_frame(processed, input_names=tuple(inputs), registry=registry)
    loaded = RecipePlan.from_dict(plan.to_dict(), registry=registry)
    return cast(ChannelFrame, loaded.apply(inputs, registry=registry))


def test_unary_audio_operation_extension_runs_complete_public_path() -> None:
    source = _frame(2.0)
    replayed = _roundtrip(source.test_gain(3.0), {"signal": source})

    assert type(replayed) is ExtensionChannelFrame
    np.testing.assert_allclose(replayed.compute(), 6.0)


def test_typed_frame_transition_extension_runs_complete_public_path() -> None:
    source = _frame(2.0)
    replayed = _roundtrip(source.to_typed(scale=4.0), {"signal": source})

    assert type(replayed) is TypedChannelFrame
    np.testing.assert_allclose(replayed.compute(), 8.0)


def test_true_multi_frame_extension_preserves_binding_order_end_to_end() -> None:
    left = _frame(5.0)
    right = _frame(2.0)
    replayed = _roundtrip(left.difference(right, scale=2.0), {"left": left, "right": right})

    assert type(replayed) is ExtensionChannelFrame
    np.testing.assert_allclose(replayed.compute(), 6.0)


def test_extension_registry_does_not_mutate_default_registry() -> None:
    registry = _registry()

    assert registry.require("tests.audio.gain", 2).operation_id == "tests.audio.gain"
    try:
        default_recipe_registry().require("tests.audio.gain", 2)
    except KeyError:
        pass
    else:  # pragma: no cover - makes the absence contract explicit
        raise AssertionError("test extension leaked into the default registry")
