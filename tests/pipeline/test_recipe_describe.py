from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import (
    RecipeOperation,
    RecipeOperationDescription,
    RecipeParameterDescription,
    RecipePlan,
    RecipeRegistry,
    default_recipe_registry,
    recipe_definition,
    recipe_operation,
)
from wandas.processing.semantic import InputBinding


class _DescribedFrame(ChannelFrame):
    @recipe_operation("tests.describe.gain")
    def gain(
        self,
        factor: float,
        *,
        mode: Literal["peak", "rms"] = "peak",
        limit: Literal["soft"] | None = None,
        label: str | None = None,
        ref: Unresolvable = None,  # noqa: F821  # ty: ignore[unresolved-reference]
    ) -> _DescribedFrame:
        return self * factor


def _identity(inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
    return inputs[0]


def _by_name(description: RecipeOperationDescription) -> dict[str, RecipeParameterDescription]:
    assert description.parameters is not None
    return {parameter.name: parameter for parameter in description.parameters}


def test_describe_reports_public_parameters_and_literal_choices() -> None:
    description = recipe_definition(_DescribedFrame.gain).describe()

    assert description.operation_id == "tests.describe.gain"
    assert description.version == 1
    assert description.method == "gain"
    assert description.binding_patterns == ((InputBinding("frame", "frame"),),)
    assert _by_name(description) == {
        "factor": RecipeParameterDescription("factor", "POSITIONAL_OR_KEYWORD", True, None, "float", None),
        "mode": RecipeParameterDescription(
            "mode", "KEYWORD_ONLY", False, "peak", "Literal['peak', 'rms']", ("peak", "rms")
        ),
        "limit": RecipeParameterDescription(
            "limit", "KEYWORD_ONLY", False, None, "Literal['soft'] | None", ("soft", None)
        ),
        "label": RecipeParameterDescription("label", "KEYWORD_ONLY", False, None, "str | None", None),
        "ref": RecipeParameterDescription("ref", "KEYWORD_ONLY", False, None, "Unresolvable", None),
    }


def test_describe_does_not_change_capture_or_replay() -> None:
    source = cast(_DescribedFrame, _DescribedFrame.from_numpy(np.ones((1, 8)), sampling_rate=8000))
    result = source.gain(2.0, mode="rms")
    registry = default_recipe_registry().with_operation(recipe_definition(_DescribedFrame.gain))
    plan = RecipePlan.from_frame(result, input_names=("signal",), registry=registry)

    assert [node["operation"] for node in plan.to_dict()["nodes"]] == ["tests.describe.gain"]
    np.testing.assert_allclose(plan.apply({"signal": source}, registry=registry).data, result.data)


def test_operation_without_method_has_no_parameter_description() -> None:
    operation = RecipeOperation("tests.describe.identity", 1, ((InputBinding("frame", "frame"),),), _identity)

    assert operation.describe() == RecipeOperationDescription(
        "tests.describe.identity", 1, None, ((InputBinding("frame", "frame"),),), None
    )


def test_operation_rejects_non_callable_method() -> None:
    with pytest.raises(TypeError, match="method must be callable"):
        RecipeOperation(
            "tests.describe.invalid", 1, ((InputBinding("frame", "frame"),),), _identity, method=cast(Any, object())
        )


def test_registry_describe_follows_registration_order() -> None:
    first = RecipeOperation("tests.describe.first", 1, ((InputBinding("frame", "frame"),),), _identity)
    second = recipe_definition(_DescribedFrame.gain)

    described = RecipeRegistry((first,)).with_operation(second).describe()

    assert [item.operation_id for item in described] == ["tests.describe.first", "tests.describe.gain"]


def test_builtin_descriptions_match_public_method_signatures() -> None:
    described = default_recipe_registry().describe()

    assert len(described) == len(default_recipe_registry().operations)
    assert all(item.parameters is not None for item in described)
    fft = next(item for item in described if (item.operation_id, item.version) == ("wandas.audio.fft", 2))
    expected = list(inspect.signature(ChannelFrame.fft).parameters)[1:]
    assert fft.method == "fft"
    assert [parameter.name for parameter in fft.parameters or ()] == expected
    assert _by_name(fft)["window"].default == "hann"


def test_public_calls_record_operations_with_public_method_names() -> None:
    source = ChannelFrame.from_numpy(np.ones((1, 64)), sampling_rate=8000)
    plan = RecipePlan.from_frame(source.trim(0.0, 0.004).fft(), input_names=("signal",))
    methods = {(item.operation_id, item.version): item.method for item in default_recipe_registry().describe()}

    recorded = [methods[(node["operation"], node["version"])] for node in plan.to_dict()["nodes"]]

    assert recorded == ["trim", "fft"]
    assert methods[("wandas.audio.trim", 1)] == "_trim_recipe_v1"


def _broken_annotation() -> Any:
    raise KeyError("Mode")


class _EdgeFrame(ChannelFrame):
    @recipe_operation("tests.describe.edge")
    def edge(
        self,
        bands: list[float] = [1.0, 2.0],
        mode: _broken_annotation() = "a",  # ty: ignore[invalid-type-form]
    ) -> _EdgeFrame:
        return self


def test_describe_copies_defaults_and_tolerates_failing_annotations() -> None:
    operation = recipe_definition(_EdgeFrame.edge)
    bands = _by_name(operation.describe())["bands"]
    assert bands.default is not None
    bands.default.append(3.0)

    parameters = _by_name(operation.describe())
    assert parameters["bands"].default == [1.0, 2.0]
    assert parameters["mode"].annotation == "_broken_annotation()"
    assert parameters["mode"].choices is None
