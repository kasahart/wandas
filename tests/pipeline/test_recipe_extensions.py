from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import (
    OperationCapture,
    RecipeOperation,
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
        "tests.audio.signal-role-copy",
        bindings=(InputBinding("signal", "frame"),),
    )
    def signal_role_copy(self) -> ExtensionChannelFrame:
        return self._create_new_instance(
            data=self._data,
            lineage=self._required_semantic_lineage(),
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
        ExtensionChannelFrame.signal_role_copy,
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
    np.testing.assert_allclose(channel_first_values(replayed), 6.0)


def test_typed_frame_transition_extension_runs_complete_public_path() -> None:
    source = _frame(2.0)
    replayed = _roundtrip(source.to_typed(scale=4.0), {"signal": source})

    assert type(replayed) is TypedChannelFrame
    np.testing.assert_allclose(channel_first_values(replayed), 8.0)


def test_default_capture_uses_declared_unary_frame_role() -> None:
    source = _frame(2.0)
    processed = source.signal_role_copy()
    replayed = _roundtrip(processed, {"signal": source})

    assert processed.lineage.operation is not None
    assert processed.lineage.operation.bindings == (InputBinding("signal", "frame"),)
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(source))


def test_true_multi_frame_extension_preserves_binding_order_end_to_end() -> None:
    left = _frame(5.0)
    right = _frame(2.0)
    replayed = _roundtrip(left.difference(right, scale=2.0), {"left": left, "right": right})

    assert type(replayed) is ExtensionChannelFrame
    np.testing.assert_allclose(channel_first_values(replayed), 6.0)


def test_extension_registry_does_not_mutate_default_registry() -> None:
    registry = _registry()

    assert registry.require("tests.audio.gain", 2).operation_id == "tests.audio.gain"
    try:
        default_recipe_registry().require("tests.audio.gain", 2)
    except KeyError:
        pass
    else:  # pragma: no cover - makes the absence contract explicit
        raise AssertionError("test extension leaked into the default registry")


def test_explicit_falsy_registry_is_used_for_extraction() -> None:
    class FalsyRegistry(RecipeRegistry):
        def __bool__(self) -> bool:
            return False

    source = _frame(2.0)
    registry = FalsyRegistry(_registry().operations)

    plan = RecipePlan.from_frame(source.test_gain(3.0), registry=registry)

    assert plan.nodes[0].operation == "tests.audio.gain"


def test_parameter_validator_runs_once_per_public_plan_phase() -> None:
    calls: list[dict[str, Any]] = []

    def validate(params: Mapping[str, Any]) -> None:
        calls.append(dict(params))

    base = recipe_definition(ExtensionChannelFrame.test_gain)
    validated = RecipeOperation(
        base.operation_id,
        base.version,
        base.binding_patterns,
        base.handler,
        validate,
    )
    registry = default_recipe_registry().with_operation(validated)
    source = _frame(2.0)
    processed = source.test_gain(3.0)

    plan = RecipePlan.from_frame(processed, input_names=("signal",), registry=registry)
    assert calls == [{"gain": 3.0}]

    loaded = RecipePlan.from_dict(plan.to_dict(), registry=registry)
    assert calls == [{"gain": 3.0}, {"gain": 3.0}]

    replayed = loaded.apply({"signal": source}, registry=registry)
    assert calls == [{"gain": 3.0}, {"gain": 3.0}, {"gain": 3.0}]
    np.testing.assert_allclose(channel_first_values(replayed), 6.0)


def test_falsy_extension_hooks_are_not_discarded() -> None:
    calls: list[str] = []

    class FalsyCapture:
        def __bool__(self) -> bool:
            return False

        def __call__(self, args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
            calls.append("capture")
            frame = cast(ChannelFrame, args[0])
            return OperationCapture(
                (InputBinding("frame", "frame"),),
                (frame.lineage,),
                params,
            )

    class FalsyHandler:
        def __bool__(self) -> bool:
            return False

        def __call__(self, inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
            calls.append("handler")
            return inputs[0].falsy_scale(**dict(params))

    class FalsyValidator:
        def __bool__(self) -> bool:
            return False

        def __call__(self, _params: Mapping[str, Any]) -> None:
            calls.append("validate")

    capture = FalsyCapture()
    handler = FalsyHandler()
    validator = FalsyValidator()

    class FalsyHookFrame(ChannelFrame):
        @recipe_operation(
            "tests.audio.falsy-hooks",
            capture=capture,
            handler=handler,
            validate_params=validator,
        )
        def falsy_scale(self, gain: float) -> FalsyHookFrame:
            return self._create_new_instance(
                data=self._data * gain,
                lineage=self._required_semantic_lineage(),
            )

    source = cast(
        FalsyHookFrame,
        FalsyHookFrame.from_numpy(np.full((1, 16), 2.0), sampling_rate=8000),
    )
    processed = source.falsy_scale(3.0)
    definition = recipe_definition(FalsyHookFrame.falsy_scale)
    registry = default_recipe_registry().with_operation(definition)
    plan = RecipePlan.from_frame(processed, input_names=("signal",), registry=registry)
    replayed = plan.apply({"signal": source}, registry=registry)

    assert definition.handler is handler
    assert definition.validate_params is validator
    assert calls.count("capture") == 1
    assert calls.count("handler") == 1
    assert calls.count("validate") == 2
    np.testing.assert_allclose(channel_first_values(replayed), 6.0)
