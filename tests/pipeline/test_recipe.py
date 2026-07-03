import json
import re
from collections.abc import Callable
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.frames.noct import NOctFrame
from wandas.frames.roughness import RoughnessFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.pipeline import (
    IndexingStep,
    MethodStep,
    OperationSpec,
    RecipeExtractionError,
    RecipeSpec,
    ScalarOperationStep,
    TypedMethodStep,
)


def _frame() -> ChannelFrame:
    sampling_rate = 16000
    time = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = (0.25 + np.sin(2 * np.pi * 1000 * time)).reshape(1, -1)
    return ChannelFrame.from_numpy(data, sampling_rate=sampling_rate, label="recipe-source")


def _two_channel_frame_with_refs() -> ChannelFrame:
    base = _frame()
    return ChannelFrame(
        data=da.from_array(np.vstack([base.data, base.data * 0.5]), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="left", unit="Pa", ref=2.0),
            ChannelMetadata(label="right", unit="Pa", ref=4.0),
        ],
    )


def _patch_hpss_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import wandas.processing.effects as effects_module

    fake_effects = SimpleNamespace(
        harmonic=lambda data, **_: data,
        percussive=lambda data, **_: data,
    )
    monkeypatch.setattr(effects_module, "require_librosa_effects", lambda _feature: fake_effects)


def _patch_psychoacoustic_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import wandas.processing.psychoacoustic as psychoacoustic_module

    def fake_loudness(
        signal: np.ndarray,
        sampling_rate: float,
        *,
        field_type: str,
    ) -> tuple[np.ndarray, None, None, None]:
        del sampling_rate, field_type
        return np.linspace(1.0, 2.0, max(1, signal.shape[-1] // 96)), None, None, None

    def fake_roughness(
        signal: np.ndarray,
        sampling_rate: float,
        *,
        overlap: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        del sampling_rate, overlap
        total = np.linspace(0.1, 0.2, max(1, signal.shape[-1] // 7200))
        specific = np.ones((47, total.shape[0]), dtype=np.float64)
        bark_axis = np.linspace(0.5, 23.5, 47)
        return total, specific, bark_axis, None

    def fake_sharpness(
        signal: np.ndarray,
        sampling_rate: float,
        *,
        weighting: str,
        field_type: str,
        skip: int,
    ) -> tuple[np.ndarray, None]:
        del sampling_rate, weighting, field_type, skip
        return np.linspace(0.3, 0.4, max(1, signal.shape[-1] // 96)), None

    psychoacoustic_module.RoughnessDwSpec._bark_axis_cache.clear()
    monkeypatch.setattr(psychoacoustic_module, "loudness_zwtv_mosqito", fake_loudness)
    monkeypatch.setattr(psychoacoustic_module, "roughness_dw_mosqito", fake_roughness)
    monkeypatch.setattr(psychoacoustic_module, "sharpness_din_tv_mosqito", fake_sharpness)


def _fake_center_freq(*, fmin: float, fmax: float, n: int, **_: object) -> tuple[np.ndarray, np.ndarray]:
    bands = max(1, int(np.ceil(np.log2(fmax / fmin) * n)))
    indices = np.arange(bands, dtype=np.float64)
    return indices, fmin * 2.0 ** (indices / n)


def _fake_noct_spectrum(
    *,
    sig: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    n: int,
    fr: int,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    del fs, fr
    _, fpref = _fake_center_freq(fmin=fmin, fmax=fmax, n=n, **kwargs)
    data = np.asarray(sig)
    n_channels = 1 if data.ndim == 1 else data.shape[-1]
    spectrum = np.ones((fpref.shape[0], n_channels), dtype=np.float64)
    if n_channels == 1:
        return spectrum[:, 0], fpref
    return spectrum, fpref


def _fake_noct_synthesis(
    *,
    spectrum: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    n: int,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    del freqs
    _, fpref = _fake_center_freq(fmin=fmin, fmax=fmax, n=n, **kwargs)
    data = np.asarray(spectrum)
    n_channels = 1 if data.ndim == 1 else data.shape[-1]
    return np.ones((fpref.shape[0], n_channels), dtype=np.float64), fpref


def _patch_noct_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import wandas.frames.noct as noct_frame_module
    import wandas.processing.spectral as spectral_module

    monkeypatch.setattr(spectral_module, "require_mosqito_center_freq", lambda _feature: _fake_center_freq)
    monkeypatch.setattr(spectral_module, "_center_freq", _fake_center_freq)
    monkeypatch.setattr(spectral_module, "noct_spectrum", _fake_noct_spectrum)
    monkeypatch.setattr(spectral_module, "noct_synthesis", _fake_noct_synthesis)
    monkeypatch.setattr(noct_frame_module, "_center_freq", _fake_center_freq)


def test_recipe_apply_runs_steps_in_order_and_preserves_source_frame() -> None:
    frame = _frame()
    source_history = list(frame.operation_history)
    source_data = frame.data.copy()
    recipe = RecipeSpec(
        steps=(
            OperationSpec("highpass_filter", {"cutoff": 100.0, "order": 2}),
            OperationSpec("normalize", {"norm": 2.0}),
        )
    )

    result = recipe.apply(frame)

    assert result is not frame
    np.testing.assert_array_equal(frame.data, source_data)
    assert frame.operation_history == source_history
    assert [record["operation"] for record in result.operation_history] == ["highpass_filter", "normalize"]
    assert result.operation_history[0]["params"] == {"cutoff": 100.0, "order": 2}
    assert result.operation_history[1]["params"] == {
        "axis": -1,
        "fill": None,
        "norm": 2.0,
        "threshold": None,
    }


def test_recipe_spec_snapshots_mutable_step_params() -> None:
    params = {"cutoff": 100.0, "order": 2}
    operation = OperationSpec("highpass_filter", params)
    params["cutoff"] = 200.0
    recipe = RecipeSpec([operation])
    returned_params = operation.params
    returned_params["cutoff"] = 300.0

    assert operation.params["cutoff"] == 100.0
    assert type(operation.to_dict()["params"]) is dict
    assert operation.to_dict() == {
        "operation": "highpass_filter",
        "params": {"cutoff": 100.0, "order": 2},
    }
    assert recipe.steps == (operation,)
    assert recipe.to_dict() == {
        "steps": [
            {
                "operation": "highpass_filter",
                "params": {"cutoff": 100.0, "order": 2},
            }
        ]
    }


def test_operation_spec_snapshots_shallow_sequence_params() -> None:
    values = [1, np.float64(2.0), np.bool_(True), None, "hann"]
    operation = OperationSpec("hpss_harmonic", {"value": values})
    values[0] = 99
    returned_params = operation.params
    returned_params["value"] = [3]

    assert operation.params == {"value": [1, 2.0, True, None, "hann"]}
    assert operation.to_dict() == {
        "operation": "hpss_harmonic",
        "params": {"value": [1, 2.0, True, None, "hann"]},
    }
    assert operation == OperationSpec("hpss_harmonic", {"value": (1, 2.0, True, None, "hann")})


@pytest.mark.parametrize(
    "value",
    [
        np.array([1.0, 2.0]),
        object(),
        b"bytes",
        1 + 2j,
        {"nested": 1},
        {1, 2},
        frozenset({1, 2}),
        [[1, 2]],
        [{"nested": 1}],
        [object()],
        [b"bytes"],
        [1 + 2j],
        [np.array([1.0, 2.0])],
    ],
)
def test_operation_spec_rejects_non_flat_literal_params(value: object) -> None:
    with pytest.raises(TypeError, match="OperationSpec params must be flat recipe-literal values"):
        OperationSpec("normalize", {"value": value})


def test_operation_spec_rejects_nested_mapping_params() -> None:
    with pytest.raises(TypeError, match="OperationSpec params must be flat recipe-literal values"):
        OperationSpec("normalize", {"value": {"outer": {"nested": 1}}})


def test_operation_spec_rejects_nan_params() -> None:
    with pytest.raises(TypeError, match="OperationSpec params must not contain NaN"):
        OperationSpec("normalize", {"norm": float("nan")})


@pytest.mark.parametrize("value", [[float("nan")], [np.float64("nan")]])
def test_operation_spec_rejects_nan_inside_sequence_params(value: object) -> None:
    with pytest.raises(TypeError, match="OperationSpec params must not contain NaN"):
        OperationSpec("normalize", {"value": value})


def test_operation_spec_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(TypeError, match="OperationSpec params mapping keys must be strings"):
        OperationSpec("normalize", {object(): "value"})  # ty: ignore[invalid-argument-type]


def test_rename_channels_method_step_serializes_mapping_as_items() -> None:
    mapping = {"right": "front-right", 0: "left"}
    step = MethodStep("rename_channels", {"mapping": mapping})
    mapping[0] = "changed"

    assert step.params == {"mapping": {"right": "front-right", 0: "left"}}
    serialized_step = step.to_dict()
    assert serialized_step == {
        "method": "rename_channels",
        "params": {"mapping_items": [["right", "front-right"], [0, "left"]]},
    }
    assert json.loads(json.dumps(serialized_step)) == serialized_step


def test_method_step_rejects_methods_outside_replay_allowlist() -> None:
    with pytest.raises(ValueError, match="MethodStep method is outside the replayable method allowlist"):
        MethodStep("plot")


def test_typed_method_step_rejects_methods_outside_replay_allowlist() -> None:
    with pytest.raises(ValueError, match="TypedMethodStep method is outside the replayable typed-method allowlist"):
        TypedMethodStep("plot")


def test_scalar_operation_step_rejects_operations_outside_replay_allowlist() -> None:
    with pytest.raises(ValueError, match="ScalarOperationStep operation is outside the replayable scalar allowlist"):
        ScalarOperationStep("%", 2)


def test_scalar_operation_step_rejects_non_numeric_operands() -> None:
    with pytest.raises(TypeError, match="ScalarOperationStep operand must be an int or float"):
        ScalarOperationStep("+", "2")  # ty: ignore[invalid-argument-type]


def test_recipe_apply_preserves_dask_laziness(monkeypatch: pytest.MonkeyPatch) -> None:
    sampling_rate = 16000
    time = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = da.from_array(np.sin(2 * np.pi * 1000 * time).reshape(1, -1), chunks=(1, -1))
    frame = ChannelFrame(data=data, sampling_rate=sampling_rate)
    compute_calls: list[tuple[object, ...]] = []

    original_compute = DaArray.compute

    def record_compute(self: DaArray, *args: object, **kwargs: object) -> object:
        compute_calls.append(args)
        return original_compute(self, *args, **kwargs)

    monkeypatch.setattr(DaArray, "compute", record_compute)

    result = RecipeSpec(
        [
            OperationSpec("highpass_filter", {"cutoff": 100.0, "order": 2}),
            OperationSpec("normalize"),
        ]
    ).apply(frame)

    assert isinstance(result._data, DaArray)
    assert compute_calls == []


def test_recipe_from_frame_extracts_linear_apply_operation_replayable_chain() -> None:
    frame = _frame()
    processed = frame.remove_dc().trim(start=0.1, end=0.5).resampling(target_sr=8000).normalize(norm=2.0)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        OperationSpec("remove_dc"),
        OperationSpec("trim", {"start": 0.1, "end": 0.5}),
        OperationSpec("resampling", {"target_sr": 8000.0}),
        OperationSpec("normalize", {"norm": 2.0, "axis": -1, "threshold": None, "fill": None}),
    )
    assert [record["operation"] for record in replayed.operation_history] == [
        "remove_dc",
        "trim",
        "resampling",
        "normalize",
    ]
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)


@pytest.mark.parametrize(
    ("build_frame", "expected_step"),
    [
        (lambda frame: frame.abs(), OperationSpec("abs")),
        (lambda frame: frame.power(exponent=3.0), OperationSpec("power", {"exponent": 3.0})),
        (lambda frame: frame.a_weighting(), OperationSpec("a_weighting")),
        (lambda frame: frame.fade(fade_ms=10.0), OperationSpec("fade", {"fade_ms": 10.0})),
    ],
)
def test_recipe_from_frame_extracts_additional_single_input_apply_operations(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    expected_step: OperationSpec,
) -> None:
    frame = _frame()
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (expected_step,)
    assert replayed.operation_history == processed.operation_history
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape


@pytest.mark.parametrize(
    ("build_frame", "expected_step"),
    [
        (
            lambda frame: frame.rms_trend(frame_length=512, hop_length=128, dB=True, Aw=False),
            OperationSpec(
                "rms_trend",
                {"frame_length": 512, "hop_length": 128, "dB": True, "Aw": False, "ref": [2.0, 4.0]},
            ),
        ),
        (
            lambda frame: frame.sound_level(freq_weighting="Z", time_weighting="Fast", dB=True),
            OperationSpec(
                "sound_level",
                {"ref": [2.0, 4.0], "freq_weighting": "Z", "time_weighting": "Fast", "dB": True},
            ),
        ),
    ],
)
def test_recipe_from_frame_extracts_ref_bearing_apply_operations(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    expected_step: OperationSpec,
) -> None:
    frame = _two_channel_frame_with_refs()
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (expected_step,)
    assert replayed.operation_history == processed.operation_history
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape


@pytest.mark.parametrize(
    ("build_frame", "expected_step"),
    [
        (
            lambda frame: frame.hpss_harmonic(kernel_size=(31, 31), margin=(1.0, 2.0)),
            OperationSpec(
                "hpss_harmonic",
                {
                    "kernel_size": [31, 31],
                    "power": 2,
                    "margin": [1.0, 2.0],
                    "n_fft": 2048,
                    "hop_length": None,
                    "win_length": None,
                    "window": "hann",
                    "center": True,
                    "pad_mode": "constant",
                },
            ),
        ),
        (
            lambda frame: frame.hpss_percussive(kernel_size=(31, 31), margin=(1.0, 2.0)),
            OperationSpec(
                "hpss_percussive",
                {
                    "kernel_size": [31, 31],
                    "power": 2,
                    "margin": [1.0, 2.0],
                    "n_fft": 2048,
                    "hop_length": None,
                    "win_length": None,
                    "window": "hann",
                    "center": True,
                    "pad_mode": "constant",
                },
            ),
        ),
    ],
)
def test_recipe_from_frame_extracts_hpss_apply_operations(
    monkeypatch: pytest.MonkeyPatch,
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    expected_step: OperationSpec,
) -> None:
    _patch_hpss_backend(monkeypatch)
    frame = _frame()
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (expected_step,)
    assert replayed.operation_history == processed.operation_history
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape


@pytest.mark.parametrize(
    ("build_frame", "expected_step"),
    [
        (
            lambda frame: frame.loudness_zwtv(field_type="diffuse"),
            OperationSpec("loudness_zwtv", {"field_type": "diffuse"}),
        ),
        (
            lambda frame: frame.roughness_dw(overlap=0.25),
            OperationSpec("roughness_dw", {"overlap": 0.25}),
        ),
        (
            lambda frame: frame.sharpness_din(weighting="din", field_type="diffuse"),
            OperationSpec("sharpness_din", {"weighting": "din", "field_type": "diffuse"}),
        ),
    ],
)
def test_recipe_from_frame_extracts_psychoacoustic_apply_operations(
    monkeypatch: pytest.MonkeyPatch,
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    expected_step: OperationSpec,
) -> None:
    _patch_psychoacoustic_backend(monkeypatch)
    sampling_rate = 48000
    time = np.linspace(0, 1, sampling_rate, endpoint=False)
    frame = ChannelFrame.from_numpy(
        np.sin(2 * np.pi * 1000 * time).reshape(1, -1),
        sampling_rate=sampling_rate,
        label="psychoacoustic-source",
    )
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (expected_step,)
    assert replayed.operation_history == processed.operation_history
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape


def test_recipe_from_frame_extracts_roughness_spec_typed_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_psychoacoustic_backend(monkeypatch)
    sampling_rate = 48000
    time = np.linspace(0, 1, sampling_rate, endpoint=False)
    frame = ChannelFrame.from_numpy(
        np.sin(2 * np.pi * 1000 * time).reshape(1, -1),
        sampling_rate=sampling_rate,
        label="roughness-spec-source",
    )
    processed = frame.roughness_dw_spec(overlap=0.25)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (TypedMethodStep("roughness_dw_spec", {"overlap": 0.25}),)
    assert isinstance(replayed, RoughnessFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    np.testing.assert_allclose(replayed.bark_axis, processed.bark_axis)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape
    assert replayed.overlap == processed.overlap


def test_recipe_from_frame_extracts_method_aware_linear_steps() -> None:
    frame = ChannelFrame.from_numpy(
        np.vstack([_frame().data, _frame().data * 0.5]),
        sampling_rate=_frame().sampling_rate,
        ch_labels=["left", "right"],
    )
    processed = frame.fix_length(length=8000).sum().mean()

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        MethodStep("fix_length", {"length": 8000}),
        MethodStep("sum"),
        MethodStep("mean"),
    )
    assert recipe.to_dict() == {
        "steps": [
            {"method": "fix_length", "params": {"length": 8000}},
            {"method": "sum", "params": {}},
            {"method": "mean", "params": {}},
        ]
    }
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape


def test_recipe_from_frame_extracts_channel_difference_method_step() -> None:
    base = _frame()
    frame = ChannelFrame.from_numpy(
        np.vstack([base.data, base.data * 0.5]),
        sampling_rate=base.sampling_rate,
        ch_labels=["left", "right"],
    )
    processed = frame.channel_difference(other_channel="left")

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (MethodStep("channel_difference", {"other_channel": 0}),)
    assert recipe.to_dict() == {"steps": [{"method": "channel_difference", "params": {"other_channel": 0}}]}
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape


@pytest.mark.parametrize(
    ("build_frame", "expected_step"),
    [
        (lambda frame: frame.get_channel(1), MethodStep("get_channel", {"channel_idx": 1})),
        (lambda frame: frame.get_channel([0, 2]), MethodStep("get_channel", {"channel_idx": [0, 2]})),
        (lambda frame: frame["right"], MethodStep("get_channel", {"channel_idx": 1})),
        (
            lambda frame: frame.get_channel(query="right"),
            MethodStep("get_channel", {"query": "right", "validate_query_keys": True}),
        ),
    ],
)
def test_recipe_from_frame_extracts_channel_selection_method_step(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    expected_step: MethodStep,
) -> None:
    base = _frame()
    frame = ChannelFrame.from_numpy(
        np.vstack([base.data, base.data * 0.5, base.data * 0.25]),
        sampling_rate=base.sampling_rate,
        ch_labels=["left", "right", "rear"],
    )
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (expected_step,)
    assert replayed.operation_history == processed.operation_history
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape


@pytest.mark.parametrize(
    ("build_frame", "expected_step"),
    [
        (lambda frame: frame.remove_channel(0), MethodStep("remove_channel", {"key": 0})),
        (lambda frame: frame.remove_channel("right"), MethodStep("remove_channel", {"key": "right"})),
    ],
)
def test_recipe_from_frame_extracts_remove_channel_method_step(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    expected_step: MethodStep,
) -> None:
    base = _frame()
    frame = ChannelFrame(
        data=da.from_array(np.vstack([base.data, base.data * 0.5, base.data * 0.25]), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="right"),
            ChannelMetadata(label="rear"),
        ],
        source_time_offset=[0.0, 0.1, 0.2],
    )
    source_history = list(frame.operation_history)
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert frame.operation_history == source_history
    assert recipe.steps == (expected_step,)
    assert replayed.operation_history == processed.operation_history
    np.testing.assert_allclose(replayed.data, processed.data)
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)
    assert replayed.labels == processed.labels
    assert replayed.shape == processed.shape


@pytest.mark.parametrize(
    ("mapping", "expected_step"),
    [
        (
            {0: "front-left", 1: "front-right"},
            MethodStep("rename_channels", {"mapping": {0: "front-left", 1: "front-right"}}),
        ),
        (
            {"left": "front-left", "right": "front-right"},
            MethodStep("rename_channels", {"mapping": {"left": "front-left", "right": "front-right"}}),
        ),
        (
            {"right": "front-right", 0: "front-left"},
            MethodStep("rename_channels", {"mapping": {"right": "front-right", 0: "front-left"}}),
        ),
    ],
)
def test_recipe_from_frame_extracts_rename_channels_method_step(
    mapping: dict[int | str, str],
    expected_step: MethodStep,
) -> None:
    base = _frame()
    frame = ChannelFrame(
        data=da.from_array(np.vstack([base.data, base.data * 0.5]), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="left", unit="Pa", ref=2.0),
            ChannelMetadata(label="right", unit="V", ref=1.0),
        ],
        source_time_offset=[0.0, 0.1],
    )
    processed = frame.rename_channels(mapping)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (expected_step,)
    serialized_step = recipe.to_dict()["steps"][0]
    assert serialized_step == {
        "method": "rename_channels",
        "params": {"mapping_items": [[key, value] for key, value in expected_step.params["mapping"].items()]},
    }
    assert json.loads(json.dumps(serialized_step)) == serialized_step
    assert replayed.operation_history == processed.operation_history
    assert replayed.labels == processed.labels
    assert replayed.channels[0].unit == processed.channels[0].unit
    assert replayed.channels[1].ref == processed.channels[1].ref
    np.testing.assert_allclose(replayed.data, processed.data)
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)


@pytest.mark.parametrize(
    "build_frame",
    [
        lambda frame: frame.get_channel(query=lambda channel: channel.label == "right"),
        lambda frame: frame.get_channel(query=re.compile("right")),
        lambda frame: frame.get_channel(query={"label": "right"}),
    ],
)
def test_recipe_from_frame_rejects_non_literal_channel_queries(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
) -> None:
    base = _frame()
    frame = ChannelFrame.from_numpy(
        np.vstack([base.data, base.data * 0.5]),
        sampling_rate=base.sampling_rate,
        ch_labels=["left", "right"],
    )
    processed = build_frame(frame)

    with pytest.raises(RecipeExtractionError, match="Channel selection recipe extraction only supports"):
        RecipeSpec.from_frame(processed)


def test_recipe_from_frame_rejects_integer_list_channel_and_time_indexing_boundary() -> None:
    frame = _two_channel_frame_with_refs()
    processed = frame[[0, 1], 10:20]

    with pytest.raises(RecipeExtractionError, match="Multidimensional indexing recipe extraction only supports"):
        RecipeSpec.from_frame(processed)


@pytest.mark.parametrize(
    "build_frame",
    [
        lambda frame: frame[[0, 1]],
        lambda frame: frame[np.array([0, 1])],
        lambda frame: frame[np.array([True, True])],
    ],
)
def test_recipe_from_frame_rejects_getitem_index_arrays_and_integer_lists(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
) -> None:
    frame = _two_channel_frame_with_refs()
    processed = build_frame(frame)

    with pytest.raises(RecipeExtractionError, match="Indexing recipe extraction only supports channel-only"):
        RecipeSpec.from_frame(processed)


def test_getitem_label_list_recipe_extraction_snapshots_labels() -> None:
    frame = _two_channel_frame_with_refs()
    labels = ["left", "right"]
    processed = frame[labels]
    labels[0] = "mutated"

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.to_dict() == {"steps": [{"getitem": {"type": "label_list", "labels": ["left", "right"]}}]}
    assert replayed.operation_history == processed.operation_history
    assert replayed.labels == ["left", "right"]


def test_recipe_from_frame_rejects_legacy_getitem_channel_slice_without_bounds() -> None:
    frame = _two_channel_frame_with_refs()
    processed = frame[0:2]
    assert processed.lineage is not None
    processed._lineage = frame._lineage_with_method("__getitem__", {"indexing": "channel_slice"})

    with pytest.raises(RecipeExtractionError, match="Channel slice recipe extraction requires explicit slice params"):
        RecipeSpec.from_frame(processed)


@pytest.mark.parametrize(
    ("build_frame", "expected_step"),
    [
        (
            lambda frame: frame[0:2],
            {"getitem": {"type": "channel_slice", "start": 0, "stop": 2, "step": None}},
        ),
        (
            lambda frame: frame[["left", "right"]],
            {"getitem": {"type": "label_list", "labels": ["left", "right"]}},
        ),
    ],
)
def test_recipe_from_frame_extracts_getitem_channel_selection(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    expected_step: dict[str, object],
) -> None:
    frame = _two_channel_frame_with_refs()
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.to_dict() == {"steps": [expected_step]}
    assert replayed.operation_history == processed.operation_history
    assert replayed.labels == processed.labels
    np.testing.assert_allclose(replayed.data, processed.data)
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)


@pytest.mark.parametrize(
    ("build_frame", "expected_step"),
    [
        (
            lambda frame: frame[:, 100:400],
            {
                "getitem": {
                    "type": "multidimensional_slice",
                    "channel": {"type": "slice", "start": None, "stop": None, "step": None},
                    "axis_slices": [{"start": 100, "stop": 400, "step": None}],
                }
            },
        ),
        (
            lambda frame: frame["right", 200:600],
            {
                "getitem": {
                    "type": "multidimensional_slice",
                    "channel": {"type": "index", "value": 1},
                    "axis_slices": [{"start": 200, "stop": 600, "step": None}],
                }
            },
        ),
    ],
)
def test_recipe_from_frame_extracts_multidimensional_slice_indexing(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    expected_step: dict[str, object],
) -> None:
    frame = _two_channel_frame_with_refs()
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.to_dict() == {"steps": [expected_step]}
    assert replayed.operation_history == processed.operation_history
    assert replayed.labels == processed.labels
    np.testing.assert_allclose(replayed.data, processed.data)
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)


def test_recipe_from_frame_extracts_multidimensional_slice_after_operation() -> None:
    frame = _two_channel_frame_with_refs()
    processed = frame.normalize(norm=2.0)[:, 100:400]

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.to_dict() == {
        "steps": [
            {
                "operation": "normalize",
                "params": {"axis": -1, "fill": None, "norm": 2.0, "threshold": None},
            },
            {
                "getitem": {
                    "type": "multidimensional_slice",
                    "channel": {"type": "slice", "start": None, "stop": None, "step": None},
                    "axis_slices": [{"start": 100, "stop": 400, "step": None}],
                }
            },
        ]
    }
    assert replayed.operation_history == processed.operation_history
    np.testing.assert_allclose(replayed.data, processed.data)
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)


def test_indexing_step_normalizes_multidimensional_slices() -> None:
    key = (slice(None), slice(np.int64(100), np.int64(400)))
    step = IndexingStep(key)

    assert step.key == (slice(None), slice(100, 400))
    assert step.to_dict() == {
        "getitem": {
            "type": "multidimensional_slice",
            "channel": {"type": "slice", "start": None, "stop": None, "step": None},
            "axis_slices": [{"start": 100, "stop": 400, "step": None}],
        }
    }


def test_recipe_from_frame_extracts_scalar_operation_chain() -> None:
    frame = _frame()
    processed = frame.normalize(norm=2.0) + 0.25

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        OperationSpec("normalize", {"norm": 2.0, "axis": -1, "threshold": None, "fill": None}),
        ScalarOperationStep("+", 0.25),
    )
    assert recipe.to_dict() == {
        "steps": [
            {
                "operation": "normalize",
                "params": {"axis": -1, "fill": None, "norm": 2.0, "threshold": None},
            },
            {"scalar_operation": "+", "operand": 0.25},
        ]
    }
    assert [record["operation"] for record in replayed.operation_history] == ["normalize", "+"]
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape


def test_recipe_from_frame_extracts_value_bearing_numpy_scalar_operation() -> None:
    frame = _frame()
    processed = frame + np.float64(0.25)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (ScalarOperationStep("+", 0.25),)
    np.testing.assert_allclose(replayed.data, processed.data)


def test_recipe_from_frame_reports_scalar_nan_operand_boundary() -> None:
    processed = _frame() + float("nan")

    with pytest.raises(RecipeExtractionError, match="Scalar operation requires a stable numeric scalar operand"):
        RecipeSpec.from_frame(processed)


@pytest.mark.parametrize(
    ("build_frame", "step"),
    [
        (lambda frame: frame - 0.5, ScalarOperationStep("-", 0.5)),
        (lambda frame: frame * 2, ScalarOperationStep("*", 2)),
        (lambda frame: frame / 2, ScalarOperationStep("/", 2)),
        (lambda frame: frame**2, ScalarOperationStep("**", 2)),
    ],
)
def test_recipe_from_frame_extracts_scalar_operation_symbols(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    step: ScalarOperationStep,
) -> None:
    frame = _frame()
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (step,)
    np.testing.assert_allclose(replayed.data, processed.data)


def test_recipe_from_frame_extracts_fft_typed_transition() -> None:
    frame = _frame()
    processed = frame.fft(n_fft=1024, window="hann")

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (TypedMethodStep("fft", {"n_fft": 1024, "window": "hann"}),)
    assert recipe.to_dict() == {"steps": [{"typed_method": "fft", "params": {"n_fft": 1024, "window": "hann"}}]}
    assert isinstance(replayed, SpectralFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.n_fft == processed.n_fft
    assert replayed.window == processed.window
    assert replayed.sampling_rate == processed.sampling_rate


def test_recipe_from_frame_rejects_direct_apply_operation_typed_transition() -> None:
    frame = _frame()
    processed = frame.apply_operation("fft", n_fft=1024, window="hann")

    with pytest.raises(RecipeExtractionError, match="Typed operation requires frame method lineage"):
        RecipeSpec.from_frame(processed)


def test_recipe_from_frame_extracts_fft_ifft_typed_transition_chain() -> None:
    frame = _frame()
    processed = frame.fft(n_fft=1024, window="hann").ifft()

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        TypedMethodStep("fft", {"n_fft": 1024, "window": "hann"}),
        TypedMethodStep("ifft"),
    )
    assert isinstance(replayed, ChannelFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.sampling_rate == processed.sampling_rate


def test_recipe_from_frame_extracts_stft_istft_typed_transition_chain() -> None:
    frame = _frame()
    processed = frame.stft(n_fft=512, hop_length=128, win_length=512, window="hann").istft()

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        TypedMethodStep(
            "stft",
            {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann"},
        ),
        TypedMethodStep("istft"),
    )
    assert isinstance(replayed, ChannelFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.sampling_rate == processed.sampling_rate


def test_recipe_from_frame_extracts_stft_typed_transition() -> None:
    frame = _frame()
    processed = frame.stft(n_fft=512, hop_length=128, win_length=512, window="hann")

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        TypedMethodStep(
            "stft",
            {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann"},
        ),
    )
    assert isinstance(replayed, SpectrogramFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.n_fft == processed.n_fft
    assert replayed.hop_length == processed.hop_length
    assert replayed.win_length == processed.win_length
    assert replayed.window == processed.window


def test_recipe_from_frame_extracts_stft_get_frame_at_typed_transition_chain() -> None:
    frame = _frame()
    processed = frame.stft(n_fft=512, hop_length=128, win_length=512, window="hann").get_frame_at(2)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        TypedMethodStep(
            "stft",
            {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann"},
        ),
        TypedMethodStep("get_frame_at", {"time_idx": 2}),
    )
    assert isinstance(replayed, SpectralFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)
    assert replayed.n_fft == processed.n_fft
    assert replayed.window == processed.window


def test_recipe_from_frame_extracts_welch_typed_transition() -> None:
    frame = _frame()
    processed = frame.welch(n_fft=512, hop_length=128, win_length=512, window="hann", average="mean")

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        TypedMethodStep(
            "welch",
            {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann", "average": "mean"},
        ),
    )
    assert isinstance(replayed, SpectralFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.n_fft == processed.n_fft
    assert replayed.window == processed.window
    assert replayed.sampling_rate == processed.sampling_rate


def test_recipe_from_frame_extracts_noct_spectrum_typed_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_noct_backend(monkeypatch)
    frame = _frame()
    processed = frame.noct_spectrum(fmin=125, fmax=8000, n=3, G=10, fr=1000)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        TypedMethodStep(
            "noct_spectrum",
            {"fmin": 125, "fmax": 8000, "n": 3, "G": 10, "fr": 1000},
        ),
    )
    assert isinstance(replayed, NOctFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape
    assert replayed.fmin == processed.fmin
    assert replayed.fmax == processed.fmax
    assert replayed.n == processed.n
    assert replayed.G == processed.G
    assert replayed.fr == processed.fr


def test_recipe_from_frame_extracts_noct_synthesis_typed_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_noct_backend(monkeypatch)
    sampling_rate = 48000
    time = np.linspace(0, 1, sampling_rate, endpoint=False)
    frame = ChannelFrame.from_numpy(
        np.sin(2 * np.pi * 1000 * time).reshape(1, -1),
        sampling_rate=sampling_rate,
        label="noct-synthesis-source",
    )
    processed = frame.fft(n_fft=2048).noct_synthesis(fmin=125, fmax=8000, n=3, G=10, fr=1000)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        TypedMethodStep("fft", {"n_fft": 2048, "window": "hann"}),
        TypedMethodStep(
            "noct_synthesis",
            {"fmin": 125, "fmax": 8000, "n": 3, "G": 10, "fr": 1000},
        ),
    )
    assert isinstance(replayed, NOctFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape
    assert replayed.fmin == processed.fmin
    assert replayed.fmax == processed.fmax
    assert replayed.n == processed.n
    assert replayed.G == processed.G
    assert replayed.fr == processed.fr


def test_recipe_from_frame_rejects_welch_with_non_public_detrend() -> None:
    frame = _frame()
    processed = frame.apply_operation(
        "welch",
        n_fft=512,
        hop_length=128,
        win_length=512,
        window="hann",
        average="mean",
        detrend="linear",
    )

    with pytest.raises(RecipeExtractionError, match="Welch recipe extraction only supports public welch parameters"):
        RecipeSpec.from_frame(processed)


@pytest.mark.parametrize(
    ("build_frame", "expected_step"),
    [
        (
            lambda frame: frame.coherence(n_fft=512, hop_length=128, win_length=512, window="hann"),
            TypedMethodStep(
                "coherence",
                {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann", "detrend": "constant"},
            ),
        ),
        (
            lambda frame: frame.csd(n_fft=512, hop_length=128, win_length=512, window="hann"),
            TypedMethodStep(
                "csd",
                {
                    "n_fft": 512,
                    "hop_length": 128,
                    "win_length": 512,
                    "window": "hann",
                    "detrend": "constant",
                    "scaling": "spectrum",
                    "average": "mean",
                },
            ),
        ),
        (
            lambda frame: frame.transfer_function(n_fft=512, hop_length=128, win_length=512, window="hann"),
            TypedMethodStep(
                "transfer_function",
                {
                    "n_fft": 512,
                    "hop_length": 128,
                    "win_length": 512,
                    "window": "hann",
                    "detrend": "constant",
                    "scaling": "spectrum",
                    "average": "mean",
                },
            ),
        ),
    ],
)
def test_recipe_from_frame_extracts_cross_channel_typed_transitions(
    build_frame: Callable[[ChannelFrame], SpectralFrame],
    expected_step: TypedMethodStep,
) -> None:
    base = _frame()
    frame = ChannelFrame.from_numpy(
        np.vstack([base.data, base.data * 0.5]),
        sampling_rate=base.sampling_rate,
        ch_labels=["left", "right"],
    )
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (expected_step,)
    assert isinstance(replayed, SpectralFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)


def test_recipe_from_frame_extracts_spectrogram_abs_chain() -> None:
    frame = _frame()
    processed = frame.stft(n_fft=512, hop_length=128, win_length=512, window="hann").abs()

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        TypedMethodStep(
            "stft",
            {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann"},
        ),
        OperationSpec("abs"),
    )
    assert isinstance(replayed, SpectrogramFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape


def test_recipe_from_frame_extracts_spectrogram_to_channel_frame_as_istft() -> None:
    frame = _frame()
    processed = frame.stft(n_fft=512, hop_length=128, win_length=512, window="hann").to_channel_frame()

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        TypedMethodStep(
            "stft",
            {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann"},
        ),
        TypedMethodStep("istft"),
    )
    assert isinstance(replayed, ChannelFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.sampling_rate == processed.sampling_rate


def test_recipe_from_frame_empty_history_returns_empty_recipe() -> None:
    assert RecipeSpec.from_frame(_frame()).steps == ()


@pytest.mark.parametrize(
    ("operation_name", "build_frame", "message"),
    [
        (
            "binary frame operation",
            lambda frame: frame + _frame().normalize(),
            "Graph operation requires graph recipe support",
        ),
        (
            "array operand operation",
            lambda frame: frame + np.ones(frame.shape),
            "Scalar operation requires a numeric scalar operand",
        ),
        (
            "custom apply operation",
            lambda frame: frame.apply(
                lambda data, gain: data * gain,
                output_shape_func=lambda shape: shape,
                gain=2.0,
            ),
            "Operation is outside the Stage 1 recipe allowlist",
        ),
        (
            "registered operation outside current allowlist",
            lambda frame: frame.apply_operation("loudness_zwst", field_type="free"),
            "Operation is outside the Stage 1 recipe allowlist",
        ),
    ],
)
def test_recipe_from_frame_reports_current_boundary_for_non_replayable_operations(
    operation_name: str,
    build_frame: Callable[[ChannelFrame], object],
    message: str,
) -> None:
    frame = _frame()
    processed = build_frame(frame)

    with pytest.raises(RecipeExtractionError, match=message):
        RecipeSpec.from_frame(processed)


def test_recipe_from_frame_rejects_non_frame_input() -> None:
    with pytest.raises(RecipeExtractionError, match="Recipe extraction requires a Wandas frame"):
        RecipeSpec.from_frame(object())


def test_recipe_from_frame_rejects_fake_operation_graph_holder() -> None:
    fake_frame = type("FakeFrame", (), {"operation_graph": None})()

    with pytest.raises(RecipeExtractionError, match="Recipe extraction requires a Wandas frame"):
        RecipeSpec.from_frame(fake_frame)


def test_recipe_from_frame_reports_graph_recipe_boundary_for_multi_input_operation() -> None:
    frame = _frame()
    noise = _frame().low_pass_filter(cutoff=1000.0)
    processed = frame.normalize().add(noise, snr=6.0)

    with pytest.raises(RecipeExtractionError, match="Graph operation requires graph recipe support"):
        RecipeSpec.from_frame(processed)


@pytest.mark.parametrize(
    "build_added",
    [
        lambda frame: frame.add_channel(np.zeros(frame.n_samples), label="raw"),
        lambda frame: frame.add_channel(frame.get_channel(0).rename_channels({0: "copied"})),
    ],
)
def test_recipe_from_frame_rejects_add_channel_boundary(
    build_added: Callable[[ChannelFrame], ChannelFrame],
) -> None:
    frame = _two_channel_frame_with_refs().normalize(norm=2.0)
    processed = build_added(frame)

    assert processed.operation_history[-1]["operation"] == "add_channel"
    with pytest.raises(RecipeExtractionError, match="add_channel recipe extraction requires external input support"):
        RecipeSpec.from_frame(processed)
