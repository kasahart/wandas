from __future__ import annotations

import inspect
import threading
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
    fft = next(item for item in described if (item.operation_id, item.version) == ("wandas.audio.fft", 3))
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


_LOCK = threading.Lock()


class _EdgeFrame(ChannelFrame):
    @recipe_operation("tests.describe.edge")
    def edge(
        self,
        bands: list[float] = [1.0, 2.0],
        mode: _broken_annotation() = "a",  # ty: ignore[invalid-type-form]
        lock: object = _LOCK,
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
    assert parameters["lock"].default is _LOCK


@pytest.mark.parametrize(
    ("operation_id", "version", "parameter", "choices"),
    [
        ("wandas.audio.welch", 2, "average", ("mean", "median")),
        ("wandas.audio.stft", 2, "detrend", ("constant", "linear", None)),
        ("wandas.audio.csd", 2, "scaling", ("spectrum", "density")),
        ("wandas.audio.csd", 2, "average", ("mean", "median")),
        ("wandas.audio.transfer_function", 2, "scaling", ("spectrum", "density")),
        ("wandas.audio.transfer_function", 2, "average", ("mean", "median")),
        ("wandas.spectrogram.csd", 1, "scaling", ("spectrum", "density")),
        ("wandas.audio.mix", 1, "align", ("strict", "pad", "truncate")),
        ("wandas.channel.add_channel", 2, "align", ("strict", "pad", "truncate")),
        ("wandas.channel.concat_frame", 1, "align", ("strict", "pad", "truncate")),
        ("wandas.audio.sharpness_din", 1, "weighting", ("din", "aures", "bismarck", "fastl")),
        ("wandas.audio.sharpness_din", 1, "field_type", ("free", "diffuse")),
        ("wandas.audio.loudness_zwtv", 1, "field_type", ("free", "diffuse")),
        ("wandas.audio.sound_level", 2, "freq_weighting", ("A", "C", "Z", None)),
        ("wandas.audio.sound_level", 2, "time_weighting", ("Fast", "Slow")),
    ],
)
def test_builtin_fixed_choice_parameters_report_choices(
    operation_id: str, version: int, parameter: str, choices: tuple[Any, ...]
) -> None:
    description = default_recipe_registry().require(operation_id, version).describe()

    assert _by_name(description)[parameter].choices == choices


def _frame_classes() -> list[type[Any]]:
    from wandas.core.base_frame import BaseFrame

    classes: list[type[Any]] = []
    pending: list[type[Any]] = [BaseFrame]
    while pending:
        cls = pending.pop(0)
        if cls not in classes:
            classes.append(cls)
            pending.extend(cls.__subclasses__())
    return classes


def _builtin(operation_id: str, version: int | None = None) -> RecipeOperationDescription:
    matches = [
        description
        for description in default_recipe_registry().describe()
        if description.operation_id == operation_id and (version is None or description.version == version)
    ]
    return max(matches, key=lambda description: description.version)


def test_describe_reports_receiver_returns_and_extras_of_builtins() -> None:
    from wandas.core.base_frame import BaseFrame
    from wandas.frames.noct import NOctFrame
    from wandas.frames.spectral import SpectralFrame
    from wandas.frames.spectrogram import SpectrogramFrame

    welch = _builtin("wandas.audio.welch")
    assert (welch.receiver, welch.returns, welch.extras) == ((ChannelFrame,), SpectralFrame, ())

    synthesis = _builtin("wandas.spectral.noct_synthesis")
    assert (synthesis.receiver, synthesis.returns, synthesis.extras) == (
        (SpectralFrame,),
        NOctFrame,
        ("psychoacoustic",),
    )

    hpss = _builtin("wandas.audio.hpss_harmonic")
    assert (hpss.receiver, hpss.returns, hpss.extras) == ((ChannelFrame,), "receiver", ("effects",))

    add = _builtin("wandas.operator.add")
    assert (add.receiver, add.returns) == ((BaseFrame,), "receiver")

    get_frame_at = _builtin("wandas.spectrogram.get_frame_at")
    assert (get_frame_at.receiver, get_frame_at.returns) == ((SpectrogramFrame,), SpectralFrame)

    # A callable's result type is not declared, so no Frame type is reported.
    assert _builtin("wandas.custom.apply").returns is None


def test_described_receiver_matches_methods_of_every_frame_class() -> None:
    for description in default_recipe_registry().describe():
        assert description.method is not None, description.operation_id
        assert description.receiver, description.operation_id
        operation = default_recipe_registry().require(description.operation_id, description.version)
        # Test modules define Frame subclasses that may override methods for their own checks.
        for cls in (cls for cls in _frame_classes() if cls.__module__.startswith("wandas.")):
            member = getattr(cls, description.method, None)
            records = getattr(member, "__wandas_recipe_operation__", None) is operation
            assert records == issubclass(cls, description.receiver), (description.operation_id, cls.__name__)


def test_described_extras_are_installable_wandas_extras() -> None:
    from wandas.utils.optional_imports import DEPENDENCY_REGISTRY

    extras = {spec.extra for spec in DEPENDENCY_REGISTRY.values()} - {"core"}
    described = {extra for description in default_recipe_registry().describe() for extra in description.extras}

    assert described == {"effects", "psychoacoustic"}
    assert described <= extras


def test_describe_reports_extension_frame_receiver_and_declared_extras() -> None:
    @recipe_operation("tests.describe.extras", extras=("effects",))
    def stretched(frame: ChannelFrame) -> ChannelFrame:
        return frame

    description = recipe_definition(_DescribedFrame.gain).describe()
    assert (description.receiver, description.returns, description.extras) == ((_DescribedFrame,), _DescribedFrame, ())

    standalone = recipe_definition(stretched).describe()
    assert (standalone.receiver, standalone.returns, standalone.extras) == ((), ChannelFrame, ("effects",))


def test_operation_rejects_blank_extra() -> None:
    with pytest.raises(ValueError, match="extras must be non-blank strings"):
        RecipeOperation("tests.blank-extra", 1, ((InputBinding("frame", "frame"),),), _identity, extras=(" ",))


@pytest.mark.parametrize("extras", ["effects", ("effects")])
def test_operation_rejects_extras_given_as_one_string(extras: Any) -> None:
    with pytest.raises(TypeError, match="must be a tuple of strings, not a str"):
        RecipeOperation("tests.string-extra", 1, ((InputBinding("frame", "frame"),),), _identity, extras=extras)


def test_builtin_returns_ignore_same_named_frame_class_elsewhere() -> None:
    from wandas.frames.spectral import SpectralFrame as WandasSpectralFrame

    namespace: dict[str, Any] = {}
    # A subclass from another package that reuses a built-in Frame class name.
    exec("class SpectralFrame(base):\n    pass", {"base": WandasSpectralFrame}, namespace)  # noqa: S102
    assert namespace["SpectralFrame"].__module__ != "wandas.frames.spectral"

    assert _builtin("wandas.audio.fft").returns is WandasSpectralFrame
    assert _builtin("wandas.spectrogram.get_frame_at").returns is WandasSpectralFrame
