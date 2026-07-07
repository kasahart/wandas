import json
import re
from collections.abc import Callable
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

import tests.pipeline.custom_recipe_fixtures as custom_recipe_fixtures
from tests.pipeline.custom_recipe_fixtures import callable_scale, custom_rfft, custom_scale, rfft_shape, same_shape
from wandas.core.metadata import ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.frames.noct import NOctFrame
from wandas.frames.roughness import RoughnessFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.pipeline import (
    AddChannelDataStep,
    AddChannelStep,
    BinaryFrameStep,
    BinaryOperandStep,
    CustomFunctionStep,
    GraphNodeSpec,
    GraphRecipeSpec,
    IndexingStep,
    MethodStep,
    NodeGraphRecipeSpec,
    OperationSpec,
    RecipeExtractionError,
    RecipeSpec,
    ScalarOperationStep,
    TerminalStep,
    TypedMethodStep,
)
from wandas.pipeline.extraction import (
    _add_channel_data_step_from_graph,
    _add_channel_step_from_graph,
    _axis_slices_from_params,
    _binary_frame_step_from_graph,
    _binary_operand_step_from_graph,
    _channel_key_from_parent_graph,
    _custom_function_step_from_graph,
    _getitem_step_from_graph,
    _indices_from_params,
    _mask_from_params,
    _rename_mapping_from_params,
    _scalar_step_from_graph,
    _slice_from_serialized,
    _steps_from_graph,
    _validate_replayable_operation,
)
from wandas.pipeline.params import _BooleanMask, _restore_history_value, _snapshot_get_channel_query_params
from wandas.pipeline.steps import _load_importable_frame_class, _load_importable_function
from wandas.processing.base import _OPERATION_REGISTRY, AudioOperation
from wandas.utils.types import NDArrayReal


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


def _two_channel_frame_with_reordered_refs() -> ChannelFrame:
    base = _frame()
    return ChannelFrame(
        data=da.from_array(np.vstack([base.data * 0.5, base.data]), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="right", unit="Pa", ref=8.0),
            ChannelMetadata(label="left", unit="Pa", ref=2.0),
        ],
    )


def _register_recipe_boundary_operation(monkeypatch: pytest.MonkeyPatch) -> str:
    class RecipeBoundaryNoop(AudioOperation[NDArrayReal, NDArrayReal]):
        name = "_recipe_boundary_noop"

        def _process(self, x: NDArrayReal) -> NDArrayReal:
            return x

    monkeypatch.setitem(_OPERATION_REGISTRY, RecipeBoundaryNoop.name, RecipeBoundaryNoop)
    return RecipeBoundaryNoop.name


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

    def fake_loudness_zwst(signal: np.ndarray, sampling_rate: float, *, field_type: str) -> tuple[float, None, None]:
        del sampling_rate
        scale = 2.0 if field_type == "diffuse" else 1.0
        return float(np.mean(np.abs(signal)) * scale), None, None

    def fake_sharpness_din_st(signal: np.ndarray, sampling_rate: float, *, weighting: str, field_type: str) -> float:
        del sampling_rate
        scale = 2.0 if field_type == "diffuse" else 1.0
        weighting_scale = 1.5 if weighting == "aures" else 1.0
        return float(np.max(np.abs(signal)) * scale * weighting_scale)

    psychoacoustic_module.RoughnessDwSpec._bark_axis_cache.clear()
    monkeypatch.setattr(psychoacoustic_module, "require_mosqito_sq_metric", lambda _feature, _name: object())
    monkeypatch.setattr(psychoacoustic_module, "loudness_zwtv_mosqito", fake_loudness)
    monkeypatch.setattr(psychoacoustic_module, "roughness_dw_mosqito", fake_roughness)
    monkeypatch.setattr(psychoacoustic_module, "sharpness_din_tv_mosqito", fake_sharpness)
    monkeypatch.setattr(psychoacoustic_module, "loudness_zwst_mosqito", fake_loudness_zwst)
    monkeypatch.setattr(psychoacoustic_module, "sharpness_din_st_mosqito", fake_sharpness_din_st)


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


def test_recipe_apply_supports_terminal_rms_metric() -> None:
    frame = _frame()
    recipe = RecipeSpec([OperationSpec("remove_dc"), TerminalStep("rms")])

    result = recipe.apply(frame)
    expected = frame.remove_dc().rms

    np.testing.assert_allclose(result, expected)
    assert recipe.to_dict() == {
        "steps": [
            {"operation": "remove_dc", "params": {}},
            {"terminal": "rms", "params": {}},
        ]
    }


def test_recipe_apply_supports_terminal_crest_factor_metric() -> None:
    frame = _frame()
    recipe = RecipeSpec([TerminalStep("crest_factor")])

    result = recipe.apply(frame)

    np.testing.assert_allclose(result, frame.crest_factor)


@pytest.mark.parametrize("metric", ["time", "source_time"])
def test_recipe_apply_supports_channel_terminal_axis_properties(metric: str) -> None:
    frame = _frame()
    recipe = RecipeSpec([OperationSpec("trim", {"start": 0.1, "end": 0.2}), TerminalStep(metric)])

    result = recipe.apply(frame)
    expected = getattr(frame.trim(start=0.1, end=0.2), metric)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("metric", ["magnitude", "phase", "power", "dB", "dBA", "freqs", "unwrapped_phase"])
def test_recipe_apply_supports_spectral_terminal_properties(metric: str) -> None:
    frame = _frame()
    recipe = RecipeSpec([TypedMethodStep("fft", {"n_fft": 1024, "window": "hann"}), TerminalStep(metric)])

    result = recipe.apply(frame)
    expected = getattr(frame.fft(n_fft=1024, window="hann"), metric)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("metric", ["magnitude", "phase", "power", "dB", "dBA", "freqs", "times", "source_times"])
def test_recipe_apply_supports_spectrogram_terminal_properties(metric: str) -> None:
    frame = _frame()
    params = {"n_fft": 1024, "hop_length": 256, "win_length": 1024, "window": "hann"}
    recipe = RecipeSpec([TypedMethodStep("stft", params), TerminalStep(metric)])

    result = recipe.apply(frame)
    expected = getattr(frame.stft(**params), metric)

    np.testing.assert_allclose(result, expected)


def test_recipe_apply_supports_roughness_terminal_bark_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_psychoacoustic_backend(monkeypatch)
    frame = _frame()
    recipe = RecipeSpec([TypedMethodStep("roughness_dw_spec", {"overlap": 0.5}), TerminalStep("bark_axis")])

    result = recipe.apply(frame)
    expected = frame.roughness_dw_spec(overlap=0.5).bark_axis

    np.testing.assert_allclose(result, expected)


def test_terminal_property_step_rejects_callable_attribute_on_frame() -> None:
    with pytest.raises(TypeError, match="TerminalStep expected a terminal property"):
        TerminalStep("power").apply(_frame())


def test_terminal_step_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="TerminalStep metric is outside the replayable terminal allowlist"):
        TerminalStep("plot")


def test_terminal_property_step_rejects_params() -> None:
    with pytest.raises(TypeError, match="TerminalStep metric does not accept params"):
        TerminalStep("rms", {"axis": -1})


def test_recipe_apply_supports_terminal_loudness_zwst_method(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_psychoacoustic_backend(monkeypatch)
    frame = _two_channel_frame_with_refs()
    recipe = RecipeSpec([OperationSpec("remove_dc"), TerminalStep("loudness_zwst", {"field_type": "diffuse"})])

    result = recipe.apply(frame)
    expected = frame.remove_dc().loudness_zwst(field_type="diffuse")

    np.testing.assert_allclose(result, expected)
    assert recipe.to_dict() == {
        "steps": [
            {"operation": "remove_dc", "params": {}},
            {"terminal": "loudness_zwst", "params": {"field_type": "diffuse"}},
        ]
    }


def test_recipe_apply_supports_terminal_sharpness_din_st_method(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_psychoacoustic_backend(monkeypatch)
    frame = _two_channel_frame_with_refs()
    recipe = RecipeSpec([TerminalStep("sharpness_din_st", {"weighting": "aures", "field_type": "diffuse"})])

    result = recipe.apply(frame)
    expected = frame.sharpness_din_st(weighting="aures", field_type="diffuse")

    np.testing.assert_allclose(result, expected)


def test_recipe_from_frame_extracts_importable_custom_function() -> None:
    frame = _frame()
    processed = frame.apply(custom_scale, output_shape_func=same_shape, gain=2.0)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        CustomFunctionStep(
            "tests.pipeline.custom_recipe_fixtures.custom_scale",
            {"gain": 2.0},
            output_shape_function="tests.pipeline.custom_recipe_fixtures.same_shape",
        ),
    )
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
    assert processed.operation_history[-1] == {"operation": "custom", "params": {"gain": 2.0}}
    assert processed.operation_graph is not None
    assert processed.operation_graph["custom"] == {
        "function": "tests.pipeline.custom_recipe_fixtures.custom_scale",
        "output_shape_function": "tests.pipeline.custom_recipe_fixtures.same_shape",
        "dask_pure": True,
        "output_frame_class": None,
    }


def test_recipe_from_frame_preserves_importable_custom_dask_pure_flag() -> None:
    frame = _frame()
    processed = frame.apply(custom_scale, output_shape_func=same_shape, dask_pure=False, gain=2.0)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        CustomFunctionStep(
            "tests.pipeline.custom_recipe_fixtures.custom_scale",
            {"gain": 2.0},
            output_shape_function="tests.pipeline.custom_recipe_fixtures.same_shape",
            dask_pure=False,
        ),
    )
    assert replayed.lineage is not None
    assert replayed.lineage.operation.pure is False
    np.testing.assert_allclose(replayed.data, processed.data)


def test_recipe_from_frame_rejects_custom_lambda_boundary() -> None:
    frame = _frame()
    processed = frame.apply(lambda data, gain: data * gain, output_shape_func=lambda shape: shape, gain=2.0)

    with pytest.raises(RecipeExtractionError, match="Custom operation recipe extraction requires importable"):
        RecipeSpec.from_frame(processed)


def test_recipe_from_frame_extracts_custom_domain_transition() -> None:
    frame = _frame()
    processed = frame.apply(
        custom_rfft,
        output_shape_func=rfft_shape,
        output_frame_class=SpectralFrame,
        output_frame_kwargs={"n_fft": frame.n_samples, "window": "hann"},
    )

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (
        CustomFunctionStep(
            "tests.pipeline.custom_recipe_fixtures.custom_rfft",
            {},
            output_shape_function="tests.pipeline.custom_recipe_fixtures.rfft_shape",
            output_frame_class="wandas.frames.spectral.SpectralFrame",
            output_frame_kwargs={"n_fft": frame.n_samples, "window": "hann"},
        ),
    )
    assert recipe.to_dict() == {
        "steps": [
            {
                "custom_function": "tests.pipeline.custom_recipe_fixtures.custom_rfft",
                "output_shape_function": "tests.pipeline.custom_recipe_fixtures.rfft_shape",
                "dask_pure": True,
                "output_frame_class": "wandas.frames.spectral.SpectralFrame",
                "output_frame_kwargs": {"n_fft": frame.n_samples, "window": "hann"},
                "params": {},
            }
        ]
    }
    assert isinstance(replayed, SpectralFrame)
    assert replayed.n_fft == processed.n_fft == frame.n_samples
    assert replayed.window == processed.window == "hann"
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
    assert processed.operation_graph is not None
    assert processed.operation_graph["custom"] == {
        "function": "tests.pipeline.custom_recipe_fixtures.custom_rfft",
        "output_shape_function": "tests.pipeline.custom_recipe_fixtures.rfft_shape",
        "dask_pure": True,
        "output_frame_class": "wandas.frames.spectral.SpectralFrame",
        "output_frame_kwargs": {"n_fft": frame.n_samples, "window": "hann"},
    }


def test_recipe_from_frame_rejects_non_importable_custom_output_frame_class() -> None:
    frame = _frame()

    class LocalSpectralFrame(SpectralFrame):
        pass

    processed = frame.apply(
        custom_rfft,
        output_shape_func=rfft_shape,
        output_frame_class=LocalSpectralFrame,
        output_frame_kwargs={"n_fft": frame.n_samples, "window": "hann"},
    )

    with pytest.raises(RecipeExtractionError, match="importable output frame class"):
        RecipeSpec.from_frame(processed)


def test_recipe_from_frame_rejects_non_literal_custom_output_frame_kwargs() -> None:
    frame = _frame()
    processed = frame.apply(
        custom_rfft,
        output_shape_func=rfft_shape,
        output_frame_class=SpectralFrame,
        output_frame_kwargs={"n_fft": np.array([frame.n_samples]), "window": "hann"},
    )

    assert processed.operation_graph is not None
    assert "custom" not in processed.operation_graph
    with pytest.raises(RecipeExtractionError, match="Custom operation recipe extraction requires importable"):
        RecipeSpec.from_frame(processed)


@pytest.mark.parametrize(
    "function_path",
    [
        "tests.pipeline.custom_recipe_fixtures.CallableScale",
        "tests.pipeline.custom_recipe_fixtures.callable_scale",
        "tests.pipeline.custom_recipe_fixtures.partial_scale",
    ],
)
def test_custom_function_step_rejects_non_function_import_targets(function_path: str) -> None:
    frame = _frame()
    step = CustomFunctionStep(function_path, {"gain": 2.0}, output_shape_function=None)

    with pytest.raises(TypeError, match="module-level function"):
        step.apply(frame)


def test_recipe_from_frame_rejects_custom_callable_object_boundary() -> None:
    frame = _frame()
    processed = frame.apply(callable_scale, output_shape_func=same_shape, gain=2.0)

    with pytest.raises(RecipeExtractionError, match="Custom operation recipe extraction requires importable"):
        RecipeSpec.from_frame(processed)


def test_graph_recipe_applies_named_input_recipes_and_frame_addition() -> None:
    base = _frame()
    signal = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="noise")
    graph_recipe = GraphRecipeSpec(
        input_recipes={
            "signal": RecipeSpec([OperationSpec("normalize")]),
            "noise": RecipeSpec([OperationSpec("lowpass_filter", {"cutoff": 2000.0})]),
        },
        output=BinaryFrameStep("+", left="signal", right="noise"),
    )

    result = graph_recipe.apply({"signal": signal, "noise": noise})
    expected = signal.normalize() + noise.low_pass_filter(cutoff=2000.0)

    np.testing.assert_allclose(result.data, expected.data)
    assert result.operation_history == expected.operation_history
    assert graph_recipe.to_dict() == {
        "inputs": {
            "signal": {"steps": [{"operation": "normalize", "params": {}}]},
            "noise": {"steps": [{"operation": "lowpass_filter", "params": {"cutoff": 2000.0}}]},
        },
        "output": {"binary_frame": {"operation": "+", "left": "signal", "right": "noise", "params": {}}},
    }


def test_graph_recipe_applies_add_with_snr() -> None:
    base = _frame()
    signal = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise = ChannelFrame.from_numpy(np.flip(base.data), sampling_rate=base.sampling_rate, label="noise")
    graph_recipe = GraphRecipeSpec(
        input_recipes={
            "signal": RecipeSpec([OperationSpec("normalize")]),
            "noise": RecipeSpec([OperationSpec("remove_dc")]),
        },
        output=BinaryFrameStep("add_with_snr", left="signal", right="noise", params={"snr": 6.0}),
    )

    result = graph_recipe.apply({"signal": signal, "noise": noise})
    expected = signal.normalize().add(noise.remove_dc(), snr=6.0)

    np.testing.assert_allclose(result.data, expected.data)
    assert result.operation_history == expected.operation_history


def test_binary_frame_step_requires_numeric_snr_for_add_with_snr() -> None:
    with pytest.raises(TypeError, match="requires a numeric snr"):
        BinaryFrameStep("add_with_snr", left="signal", right="noise")
    with pytest.raises(TypeError, match="requires a numeric snr"):
        BinaryFrameStep("add_with_snr", left="signal", right="noise", params={"snr": "6"})
    with pytest.raises(TypeError, match="requires a numeric snr"):
        BinaryFrameStep("add_with_snr", left="signal", right="noise", params={"snr": True})
    with pytest.raises(TypeError, match="only accepts the snr parameter"):
        BinaryFrameStep("add_with_snr", left="signal", right="noise", params={"snr": 6.0, "gain": 2.0})


def test_graph_recipe_from_frame_extracts_root_add_with_snr_with_input_names() -> None:
    base = _frame()
    signal_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(np.flip(base.data), sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.normalize().add(noise_source.low_pass_filter(cutoff=1200.0), snr=6.0)

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("signal", "noise"))
    replayed = graph_recipe.apply({"signal": signal_source, "noise": noise_source})

    assert graph_recipe.to_dict() == {
        "inputs": {
            "signal": {
                "steps": [
                    {
                        "operation": "normalize",
                        "params": {"axis": -1, "fill": None, "norm": float("inf"), "threshold": None},
                    }
                ]
            },
            "noise": {"steps": [{"operation": "lowpass_filter", "params": {"cutoff": 1200.0, "order": 4}}]},
        },
        "output": {
            "binary_frame": {
                "operation": "add_with_snr",
                "left": "signal",
                "right": "noise",
                "params": {"snr": 6.0},
            }
        },
    }
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_from_frame_extracts_root_frame_addition_with_input_names() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="left")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="right")
    processed = left_source.remove_dc() + right_source.high_pass_filter(cutoff=500.0)

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    replayed = graph_recipe.apply({"left": left_source, "right": right_source})

    assert graph_recipe.output == BinaryFrameStep("+", "left", "right")
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_from_frame_does_not_self_recommend_for_external_operand_parent() -> None:
    frame = _frame()
    other = _frame().remove_dc()
    processed = (frame + np.ones(frame.shape)) + other

    with pytest.raises(RecipeExtractionError) as exc_info:
        GraphRecipeSpec.from_frame(processed, input_names=("left", "right"))

    message = str(exc_info.value)
    assert "RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe" not in message
    assert "Use GraphRecipeSpec.from_frame(...)" not in message
    assert "NodeGraphRecipeSpec.from_frame(...)" in message


def test_graph_recipe_from_frame_uses_numbered_default_input_names() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="noise")
    processed = left_source.remove_dc() + right_source.high_pass_filter(cutoff=500.0)

    graph_recipe = GraphRecipeSpec.from_frame(processed)
    replayed = graph_recipe.apply({"input_0": left_source, "input_1": right_source})

    assert graph_recipe.input_recipes == (
        ("input_0", RecipeSpec([OperationSpec("remove_dc")])),
        ("input_1", RecipeSpec([OperationSpec("highpass_filter", {"cutoff": 500.0, "order": 4})])),
    )
    assert graph_recipe.output == BinaryFrameStep("+", "input_0", "input_1")
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_from_frame_extracts_raw_left_binary_parent() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="left")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="right")
    processed = left_source + right_source.remove_dc()

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    replayed = graph_recipe.apply({"left": left_source, "right": right_source})

    assert graph_recipe.input_recipes == (
        ("left", RecipeSpec(())),
        ("right", RecipeSpec([OperationSpec("remove_dc")])),
    )
    assert graph_recipe.output == BinaryFrameStep("+", "left", "right")
    assert [record["operation"] for record in processed.operation_history] == ["remove_dc", "+"]
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_from_frame_extracts_raw_right_binary_parent() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="left")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="right")
    processed = left_source.normalize() + right_source

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    replayed = graph_recipe.apply({"left": left_source, "right": right_source})
    left_recipe = RecipeSpec(
        [OperationSpec("normalize", {"axis": -1, "fill": None, "norm": float("inf"), "threshold": None})]
    )

    assert graph_recipe.input_recipes == (
        ("left", left_recipe),
        ("right", RecipeSpec(())),
    )
    assert graph_recipe.output == BinaryFrameStep("+", "left", "right")
    assert [record["operation"] for record in processed.operation_history] == ["normalize", "+"]
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_from_frame_extracts_raw_add_with_snr_parents() -> None:
    base = _frame()
    signal_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(np.flip(base.data), sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.add(noise_source, snr=6.0)

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("signal", "noise"))
    replayed = graph_recipe.apply({"signal": signal_source, "noise": noise_source})

    assert graph_recipe.input_recipes == (("signal", RecipeSpec(())), ("noise", RecipeSpec(())))
    assert graph_recipe.output == BinaryFrameStep("add_with_snr", "signal", "noise", {"snr": 6.0})
    assert processed.operation_history == [{"operation": "add_with_snr", "params": {"snr": 6.0}}]
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_from_frame_does_not_bake_source_length_into_add_with_snr() -> None:
    base = _frame()
    data = base.data.reshape(1, -1)
    signal_source = ChannelFrame.from_numpy(data[:, :100], sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(data[:, :200], sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.add(noise_source, snr=6.0)

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("signal", "noise"))

    assert graph_recipe.input_recipes == (("signal", RecipeSpec(())), ("noise", RecipeSpec(())))

    replay_signal = ChannelFrame.from_numpy(data[:, :150], sampling_rate=base.sampling_rate, label="signal")
    replay_noise = ChannelFrame.from_numpy(data[:, :200], sampling_rate=base.sampling_rate, label="noise")
    replayed = graph_recipe.apply({"signal": replay_signal, "noise": replay_noise})
    expected = replay_signal.add(replay_noise, snr=6.0)

    np.testing.assert_allclose(replayed.data, expected.data)
    assert replayed.n_samples == replay_signal.n_samples
    assert replayed.operation_history == expected.operation_history


def test_graph_recipe_add_with_snr_serializes_snr_without_implicit_fix_length_step() -> None:
    base = _frame()
    data = base.data.reshape(1, -1)
    signal_source = ChannelFrame.from_numpy(data[:, :100], sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(data[:, :200], sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.add(noise_source, snr=6.0)

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("signal", "noise"))

    assert graph_recipe.to_dict() == {
        "inputs": {
            "signal": {"steps": []},
            "noise": {"steps": []},
        },
        "output": {
            "binary_frame": {
                "operation": "add_with_snr",
                "left": "signal",
                "right": "noise",
                "params": {"snr": 6.0},
            }
        },
    }


def test_graph_recipe_from_frame_uses_numbered_default_names_for_raw_add_with_snr() -> None:
    base = _frame()
    signal_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(np.flip(base.data), sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.add(noise_source, snr=6.0)

    graph_recipe = GraphRecipeSpec.from_frame(processed)
    replayed = graph_recipe.apply({"input_0": signal_source, "input_1": noise_source})

    assert graph_recipe.input_recipes == (("input_0", RecipeSpec(())), ("input_1", RecipeSpec(())))
    assert graph_recipe.output == BinaryFrameStep("add_with_snr", "input_0", "input_1", {"snr": 6.0})
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_numbered_default_names_preserve_binary_operand_order() -> None:
    base = _frame()
    signal_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.remove_dc() - noise_source.normalize()

    graph_recipe = GraphRecipeSpec.from_frame(processed)
    replayed = graph_recipe.apply({"input_0": signal_source, "input_1": noise_source})

    assert graph_recipe.input_recipes == (
        ("input_0", RecipeSpec([OperationSpec("remove_dc")])),
        (
            "input_1",
            RecipeSpec(
                [OperationSpec("normalize", {"axis": -1, "fill": None, "norm": float("inf"), "threshold": None})]
            ),
        ),
    )
    assert graph_recipe.output == BinaryFrameStep("-", "input_0", "input_1")
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_applies_single_merge_with_linear_tail() -> None:
    base = _frame()
    signal = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="noise")
    graph_recipe = GraphRecipeSpec(
        input_recipes={
            "signal": RecipeSpec([OperationSpec("remove_dc")]),
            "noise": RecipeSpec([OperationSpec("lowpass_filter", {"cutoff": 1200.0})]),
        },
        output=BinaryFrameStep("+", left="signal", right="noise"),
        tail_recipe=RecipeSpec([OperationSpec("normalize")]),
    )

    result = graph_recipe.apply({"signal": signal, "noise": noise})
    expected = (signal.remove_dc() + noise.low_pass_filter(cutoff=1200.0)).normalize()

    np.testing.assert_allclose(result.data, expected.data)
    assert result.operation_history == expected.operation_history
    assert graph_recipe.to_dict() == {
        "inputs": {
            "signal": {"steps": [{"operation": "remove_dc", "params": {}}]},
            "noise": {"steps": [{"operation": "lowpass_filter", "params": {"cutoff": 1200.0}}]},
        },
        "output": {"binary_frame": {"operation": "+", "left": "signal", "right": "noise", "params": {}}},
        "tail": {"steps": [{"operation": "normalize", "params": {}}]},
    }


def test_graph_recipe_from_frame_extracts_single_merge_with_linear_tail() -> None:
    base = _frame()
    signal_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="noise")
    processed = (signal_source.remove_dc() + noise_source.low_pass_filter(cutoff=1200.0)).normalize()

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("signal", "noise"))
    replayed = graph_recipe.apply({"signal": signal_source, "noise": noise_source})

    assert graph_recipe.output == BinaryFrameStep("+", "signal", "noise")
    assert graph_recipe.tail_recipe == RecipeSpec(
        [OperationSpec("normalize", {"axis": -1, "fill": None, "norm": float("inf"), "threshold": None})]
    )
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_from_frame_extracts_add_with_snr_with_linear_tail() -> None:
    base = _frame()
    signal_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(np.flip(base.data), sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.normalize().add(noise_source.remove_dc(), snr=6.0).trim(start=0.1, end=0.5)

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("signal", "noise"))
    replayed = graph_recipe.apply({"signal": signal_source, "noise": noise_source})

    assert graph_recipe.output == BinaryFrameStep("add_with_snr", "signal", "noise", {"snr": 6.0})
    assert graph_recipe.tail_recipe == RecipeSpec([OperationSpec("trim", {"start": 0.1, "end": 0.5})])
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_from_frame_extracts_single_merge_with_typed_tail() -> None:
    base = _frame()
    signal_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="noise")
    processed = (signal_source.remove_dc() + noise_source.remove_dc()).stft(
        n_fft=512,
        hop_length=128,
        win_length=512,
        window="hann",
    )

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("signal", "noise"))
    replayed = graph_recipe.apply({"signal": signal_source, "noise": noise_source})

    assert graph_recipe.tail_recipe == RecipeSpec(
        [TypedMethodStep("stft", {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann"})]
    )
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
    assert isinstance(replayed, SpectrogramFrame)


def test_graph_recipe_from_frame_uses_numbered_default_names_with_typed_tail() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="noise")
    processed = (left_source + right_source).stft(
        n_fft=512,
        hop_length=128,
        win_length=512,
        window="hann",
    )

    graph_recipe = GraphRecipeSpec.from_frame(processed)
    replayed = graph_recipe.apply({"input_0": left_source, "input_1": right_source})

    assert graph_recipe.input_recipes == (("input_0", RecipeSpec(())), ("input_1", RecipeSpec(())))
    assert graph_recipe.output == BinaryFrameStep("+", "input_0", "input_1")
    assert graph_recipe.tail_recipe == RecipeSpec(
        [TypedMethodStep("stft", {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann"})]
    )
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_from_frame_rejects_without_binary_merge() -> None:
    processed = _frame().remove_dc().normalize()

    with pytest.raises(RecipeExtractionError, match="GraphRecipeSpec extraction requires one binary merge"):
        GraphRecipeSpec.from_frame(processed, input_names=("signal", "noise"))


def test_graph_recipe_from_frame_rejects_wrong_input_name_count() -> None:
    base = _frame()
    signal_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="signal")
    noise_source = ChannelFrame.from_numpy(np.flip(base.data), sampling_rate=base.sampling_rate, label="noise")
    processed = signal_source.normalize().add(noise_source.low_pass_filter(cutoff=1200.0), snr=6.0)

    with pytest.raises(RecipeExtractionError, match="GraphRecipeSpec extraction requires one input name per parent"):
        GraphRecipeSpec.from_frame(processed, input_names=("signal",))


def test_node_graph_recipe_from_frame_extracts_two_binary_merges() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="left")
    middle_source = ChannelFrame.from_numpy(base.data * 0.5, sampling_rate=base.sampling_rate, label="middle")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="right")
    processed = (left_source.normalize() + middle_source.remove_dc()) + right_source.high_pass_filter(cutoff=500.0)

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("left", "middle", "right"))
    replayed = recipe.apply({"left": left_source, "middle": middle_source, "right": right_source})

    assert recipe.inputs == ("left", "middle", "right")
    assert [node.id for node in recipe.nodes] == ["n0", "n1", "n2", "n3", "n4"]
    assert recipe.output == "n4"
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_node_graph_recipe_from_frame_extracts_custom_function_branch() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="left")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="right")
    processed = left_source.apply(custom_scale, output_shape_func=same_shape, gain=2.0) + right_source.remove_dc()

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    replayed = recipe.apply({"left": left_source, "right": right_source})

    assert recipe.nodes[0] == GraphNodeSpec(
        "n0",
        CustomFunctionStep(
            "tests.pipeline.custom_recipe_fixtures.custom_scale",
            {"gain": 2.0},
            output_shape_function="tests.pipeline.custom_recipe_fixtures.same_shape",
        ),
        ("left",),
    )
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_node_graph_recipe_from_frame_uses_default_input_names() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="left")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="right")
    processed = left_source.normalize() + right_source.remove_dc()

    recipe = NodeGraphRecipeSpec.from_frame(processed)
    replayed = recipe.apply({"input_0": left_source, "input_1": right_source})

    assert recipe.inputs == ("input_0", "input_1")
    assert recipe.nodes == (
        GraphNodeSpec(
            "n0",
            OperationSpec("normalize", {"axis": -1, "fill": None, "norm": float("inf"), "threshold": None}),
            ("input_0",),
        ),
        GraphNodeSpec("n1", OperationSpec("remove_dc"), ("input_1",)),
        GraphNodeSpec("n2", BinaryFrameStep("+", "n0", "n1"), ("n0", "n1")),
    )
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_node_graph_recipe_from_frame_rejects_duplicate_input_names() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="left")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="right")
    processed = left_source + right_source

    with pytest.raises(RecipeExtractionError, match="requires distinct input names"):
        NodeGraphRecipeSpec.from_frame(processed, input_names=("same", "same"))


def test_node_graph_recipe_rejects_input_output_when_nodes_exist() -> None:
    with pytest.raises(ValueError, match="output must reference a graph node"):
        NodeGraphRecipeSpec(
            inputs=("signal",),
            nodes=(GraphNodeSpec("n0", OperationSpec("normalize"), ("signal",)),),
            output="signal",
        )


def test_node_graph_recipe_rejects_non_final_node_output() -> None:
    with pytest.raises(ValueError, match="output must reference the final graph node"):
        NodeGraphRecipeSpec(
            inputs=("signal",),
            nodes=(
                GraphNodeSpec("n0", OperationSpec("remove_dc"), ("signal",)),
                GraphNodeSpec("n1", OperationSpec("normalize"), ("n0",)),
            ),
            output="n0",
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: NodeGraphRecipeSpec(inputs=(), nodes=(), output="n0"), "inputs must be non-empty strings"),
        (
            lambda: NodeGraphRecipeSpec(inputs=("signal", "signal"), nodes=(), output="n0"),
            "inputs must be unique",
        ),
        (
            lambda: NodeGraphRecipeSpec(inputs=("signal",), nodes=(), output="n0"),
            "requires at least one node",
        ),
        (
            lambda: NodeGraphRecipeSpec(
                inputs=("signal",),
                nodes=(GraphNodeSpec("n0", OperationSpec("normalize"), ("signal",)),),
                output="",
            ),
            "output must be a non-empty string",
        ),
        (
            lambda: NodeGraphRecipeSpec(
                inputs=("signal",),
                nodes=(cast(Any, TerminalStep("rms")),),
                output="n0",
            ),
            "nodes must be GraphNodeSpec instances",
        ),
        (
            lambda: NodeGraphRecipeSpec(
                inputs=("signal",),
                nodes=(GraphNodeSpec("signal", OperationSpec("normalize"), ("signal",)),),
                output="signal",
            ),
            "node id duplicates an existing reference",
        ),
        (
            lambda: NodeGraphRecipeSpec(
                inputs=("signal",),
                nodes=(GraphNodeSpec("n0", OperationSpec("normalize"), ("missing",)),),
                output="n0",
            ),
            "node references unknown inputs",
        ),
        (
            lambda: NodeGraphRecipeSpec(
                inputs=("signal",),
                nodes=(GraphNodeSpec("n0", OperationSpec("normalize"), ("signal",)),),
                output="missing",
            ),
            "output references unknown node or input",
        ),
    ],
)
def test_node_graph_recipe_constructor_rejects_invalid_structure(factory: Callable[[], object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_node_graph_recipe_from_frame_rejects_non_frame_or_missing_graph() -> None:
    with pytest.raises(RecipeExtractionError, match="requires a Wandas frame"):
        NodeGraphRecipeSpec.from_frame(object())

    with pytest.raises(RecipeExtractionError, match="requires operation_graph lineage"):
        NodeGraphRecipeSpec.from_frame(_frame())


@pytest.mark.parametrize(
    ("graph", "input_names", "message"),
    [
        (
            {"operation": "+", "params": {"operand_kind": "frame"}, "inputs": [{"kind": "source"}, {"kind": "source"}]},
            ("left",),
            "requires one input name per source leaf",
        ),
        (
            {"operation": "remove_dc", "params": {}, "inputs": []},
            ("",),
            "input names must be non-empty strings",
        ),
        (
            {"operation": "add_channel", "params": {"input_kind": "ndarray"}, "inputs": [{}, {}, {}]},
            ("a", "b", "c", "data"),
            "add_channel data recipe extraction requires at most one frame parent",
        ),
        (
            {"operation": "add_channel", "params": {"input_kind": "unknown"}, "inputs": [{}]},
            ("signal",),
            "add_channel recipe extraction only supports",
        ),
        (
            {
                "operation": "+",
                "params": {"operand_kind": "operand", "operand": {"type": "ndarray"}},
                "inputs": [{}, {}],
            },
            ("signal", "offset", "extra"),
            "Binary operand recipe extraction requires at most one frame parent",
        ),
        (
            {
                "operation": "__getitem__",
                "params": {
                    "indexing": "multidimensional_slice",
                    "axis_slices": [{"start": 0, "stop": None, "step": 1}],
                },
                "inputs": [],
            },
            ("signal",),
            "requires one channel-selection parent",
        ),
        (
            {
                "operation": "__getitem__",
                "params": {
                    "indexing": "multidimensional_slice",
                    "axis_slices": [{"start": 0, "stop": None, "step": 1}],
                },
                "inputs": [{"operation": "get_channel", "params": {"query": "left"}, "inputs": [{}, {}]}],
            },
            ("left", "right", "extra"),
            "Multidimensional indexing requires one replayable parent chain",
        ),
        (
            {"operation": "remove_dc", "params": {}, "inputs": [{}, {}, {}]},
            ("a", "b", "c"),
            "only supports unary and binary frame graph nodes",
        ),
        (
            {"operation": "remove_dc", "params": {}, "inputs": []},
            ("signal", "extra"),
            "requires one input name per source leaf",
        ),
    ],
)
def test_node_graph_recipe_from_frame_rejects_invalid_graph_shapes(
    graph: dict[str, object],
    input_names: tuple[str, ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ChannelFrame, "operation_graph", property(lambda _self: graph))

    with pytest.raises(RecipeExtractionError, match=message):
        NodeGraphRecipeSpec.from_frame(_frame(), input_names=input_names)


def test_node_graph_recipe_from_frame_extracts_typed_tail_after_merge() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="left")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="right")
    processed = (
        (left_source + right_source)
        .normalize()
        .stft(
            n_fft=512,
            hop_length=128,
            win_length=512,
            window="hann",
        )
    )

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    replayed = recipe.apply({"left": left_source, "right": right_source})

    assert recipe.inputs == ("left", "right")
    assert isinstance(replayed, SpectrogramFrame)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_node_graph_recipe_from_frame_replays_duplicated_shared_branch_with_same_source() -> None:
    base = _frame()
    shared = base.normalize()
    signal = shared.low_pass_filter(cutoff=3000.0)
    noise = shared.high_pass_filter(cutoff=300.0)
    processed = signal.add(noise, snr=3.0)

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("base_signal", "base_noise"))
    replayed = recipe.apply({"base_signal": base, "base_noise": base})

    assert recipe.inputs == ("base_signal", "base_noise")
    assert recipe.nodes == (
        GraphNodeSpec(
            "n0",
            OperationSpec("normalize", {"axis": -1, "fill": None, "norm": float("inf"), "threshold": None}),
            ("base_signal",),
        ),
        GraphNodeSpec("n1", OperationSpec("lowpass_filter", {"cutoff": 3000.0, "order": 4}), ("n0",)),
        GraphNodeSpec(
            "n2",
            OperationSpec("normalize", {"axis": -1, "fill": None, "norm": float("inf"), "threshold": None}),
            ("base_noise",),
        ),
        GraphNodeSpec("n3", OperationSpec("highpass_filter", {"cutoff": 300.0, "order": 4}), ("n2",)),
        GraphNodeSpec("n4", BinaryFrameStep("add_with_snr", "n1", "n3", {"snr": 3.0}), ("n1", "n3")),
    )
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_node_graph_recipe_from_frame_extracts_multidimensional_indexing_branch() -> None:
    left = _two_channel_frame_with_refs()
    right = ChannelFrame(
        data=da.from_array(left.data.copy(), chunks=(1, -1)),
        sampling_rate=left.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="right", unit="Pa", ref=2.0),
            ChannelMetadata(label="rear", unit="Pa", ref=4.0),
        ],
    )
    processed = left[[0], 100:400] + right[["right"], 100:400]

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    replayed = recipe.apply({"left": left, "right": right})

    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_node_graph_recipe_from_frame_extracts_numpy_operand_as_external_input() -> None:
    frame = _frame()
    operand = np.ones(frame.shape)
    processed = frame + operand

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("signal", "offset"))
    replayed = recipe.apply({"signal": frame, "offset": operand})

    assert recipe.inputs == ("signal", "offset")
    assert recipe.nodes == (GraphNodeSpec("n0", BinaryOperandStep("+", "signal", "offset"), ("signal", "offset")),)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_node_graph_recipe_from_frame_extracts_dask_operand_after_processed_parent() -> None:
    frame = _frame()
    operand = da.ones(frame.shape, chunks=frame.shape)
    processed = frame.normalize() * operand

    recipe = NodeGraphRecipeSpec.from_frame(processed)
    replayed = recipe.apply({"input_0": frame, "input_1": operand})

    assert recipe.inputs == ("input_0", "input_1")
    assert recipe.nodes == (
        GraphNodeSpec(
            "n0",
            OperationSpec("normalize", {"axis": -1, "fill": None, "norm": float("inf"), "threshold": None}),
            ("input_0",),
        ),
        GraphNodeSpec("n1", BinaryOperandStep("*", "n0", "input_1"), ("n0", "input_1")),
    )
    assert isinstance(replayed._data, DaArray)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_node_graph_recipe_binary_operand_rejects_missing_operand_input() -> None:
    frame = _frame()
    recipe = NodeGraphRecipeSpec.from_frame(frame + np.ones(frame.shape), input_names=("signal", "offset"))

    with pytest.raises(KeyError, match="NodeGraphRecipeSpec input is missing"):
        recipe.apply({"signal": frame})


def test_binary_operand_step_rejects_same_frame_and_operand_ref() -> None:
    with pytest.raises(ValueError, match="BinaryOperandStep frame and operand inputs must be distinct"):
        BinaryOperandStep("+", "signal", "signal")


def test_binary_operand_step_rejects_non_array_runtime_operand() -> None:
    frame = _frame()
    step = BinaryOperandStep("+", "signal", "operand")

    with pytest.raises(TypeError, match="BinaryOperandStep operand input must be a NumPy or Dask array"):
        step.apply({"signal": frame, "operand": 1.0})


def test_binary_operand_step_rejects_unknown_operation_and_missing_runtime_inputs() -> None:
    with pytest.raises(ValueError, match="operation is outside the replayable operand allowlist"):
        BinaryOperandStep("@", "signal", "operand")

    with pytest.raises(KeyError, match="binary operand input is missing"):
        BinaryOperandStep("+", "signal", "operand").apply({"signal": _frame()})


def test_node_graph_recipe_binary_operand_rejects_inconsistent_symbol_metadata() -> None:
    graph = {
        "operation": "+",
        "params": {"symbol": "-", "operand_kind": "operand", "operand": {"type": "ndarray", "shape": [8]}},
        "inputs": [],
    }

    class GraphFrame(ChannelFrame):
        @property
        def operation_graph(self) -> dict[str, Any]:
            return graph

    graph_frame = GraphFrame(data=da.from_array(np.arange(8.0), chunks=-1), sampling_rate=8000)

    with pytest.raises(RecipeExtractionError, match="Binary operand graph has inconsistent operator metadata"):
        NodeGraphRecipeSpec.from_frame(graph_frame)


def test_node_graph_recipe_from_frame_extracts_add_channel_frame_inputs() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, ch_labels=["left"])
    right_source = ChannelFrame.from_numpy(base.data * 0.5, sampling_rate=base.sampling_rate, ch_labels=["right"])
    processed = left_source.add_channel(right_source, label="ref")

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    replayed = recipe.apply({"left": left_source, "right": right_source})

    assert recipe.inputs == ("left", "right")
    assert recipe.nodes == (
        GraphNodeSpec(
            "n0",
            AddChannelStep(
                "left",
                "right",
                {"align": "strict", "label": "ref", "suffix_on_dup": None},
            ),
            ("left", "right"),
        ),
    )
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
    assert replayed.labels == processed.labels


def test_node_graph_recipe_add_channel_frame_input_omits_raw_source_time_offset_option() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, ch_labels=["left"])
    right_source = ChannelFrame(
        data=da.from_array((base.data * 0.5).reshape(1, -1), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[ChannelMetadata(label="right")],
        source_time_offset=2.5,
    )
    processed = left_source.add_channel(right_source, label="ref")

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    serialized = recipe.to_dict()

    assert serialized["nodes"][0]["step"] == {
        "add_channel": {
            "base": "left",
            "added": "right",
            "params": {"align": "strict", "label": "ref", "suffix_on_dup": None},
        }
    }
    assert "source_time_offset" not in serialized["nodes"][0]["step"]["add_channel"]["params"]


def test_node_graph_recipe_from_frame_extracts_add_channel_with_processed_parents() -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, ch_labels=["left"])
    right_source = ChannelFrame.from_numpy(base.data * 0.5, sampling_rate=base.sampling_rate, ch_labels=["right"])
    processed = left_source.normalize().add_channel(right_source.remove_dc(), label="ref").normalize()

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    replayed = recipe.apply({"left": left_source, "right": right_source})

    assert [node.id for node in recipe.nodes] == ["n0", "n1", "n2", "n3"]
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
    assert replayed.labels == processed.labels


def test_node_graph_recipe_from_frame_extracts_add_channel_numpy_data_input() -> None:
    frame = _frame()
    raw = np.zeros(frame.n_samples)
    processed = frame.add_channel(raw, label="raw", source_time_offset=1.25)

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("signal", "raw"))
    replayed = recipe.apply({"signal": frame, "raw": raw})

    assert recipe.inputs == ("signal", "raw")
    assert recipe.nodes == (
        GraphNodeSpec(
            "n0",
            AddChannelDataStep(
                "signal",
                "raw",
                {"align": "strict", "label": "raw", "suffix_on_dup": None, "source_time_offset": 1.25},
            ),
            ("signal", "raw"),
        ),
    )
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)


def test_node_graph_recipe_add_channel_data_serializes_input_name_not_array_values() -> None:
    frame = _frame()
    raw = np.arange(frame.n_samples, dtype=float)
    processed = frame.add_channel(raw, label="raw", source_time_offset=1.25)

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("signal", "raw"))
    serialized = recipe.to_dict()

    assert serialized["inputs"] == ["signal", "raw"]
    assert serialized["nodes"] == [
        {
            "id": "n0",
            "step": {
                "add_channel_data": {
                    "base": "signal",
                    "data": "raw",
                    "params": {
                        "align": "strict",
                        "label": "raw",
                        "suffix_on_dup": None,
                        "source_time_offset": 1.25,
                    },
                }
            },
            "inputs": ["signal", "raw"],
        }
    ]
    assert "array" not in repr(serialized).lower()
    assert "arange" not in repr(serialized).lower()


def test_node_graph_recipe_from_frame_extracts_add_channel_dask_data_after_processed_parent() -> None:
    frame = _frame()
    raw = da.ones(frame.n_samples + 2, chunks=4)
    processed = frame.normalize().add_channel(raw, label="raw", align="truncate", source_time_offset=[2.0])

    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("signal", "raw"))
    replayed = recipe.apply({"signal": frame, "raw": raw})

    assert recipe.inputs == ("signal", "raw")
    assert [node.id for node in recipe.nodes] == ["n0", "n1"]
    assert isinstance(replayed._data, DaArray)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)


def test_add_channel_data_step_rejects_non_array_runtime_data() -> None:
    frame = _frame()
    step = AddChannelDataStep("signal", "raw", {"label": "raw"})

    with pytest.raises(TypeError, match="AddChannelDataStep data input must be a NumPy or Dask array"):
        step.apply({"signal": frame, "raw": 1.0})


@pytest.mark.parametrize(
    ("step", "inputs", "message"),
    [
        (BinaryFrameStep("+", "left", "right"), {"left": _frame()}, "GraphRecipeSpec input is missing"),
        (AddChannelStep("base", "added"), {"base": _frame()}, "NodeGraphRecipeSpec add_channel input is missing"),
        (
            AddChannelDataStep("base", "raw"),
            {"raw": np.ones(_frame().n_samples)},
            "NodeGraphRecipeSpec add_channel data input is missing",
        ),
    ],
)
def test_graph_steps_report_missing_runtime_inputs(
    step: BinaryFrameStep | AddChannelStep | AddChannelDataStep,
    inputs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(KeyError, match=message):
        step.apply(inputs)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: BinaryFrameStep("+", "", "right"), "left and right input names must be non-empty"),
        (lambda: BinaryFrameStep("+", "left", "right", {"unexpected": 1}), "does not accept params"),
        (lambda: BinaryFrameStep("add_with_snr", "left", "right", {"gain": 1}), "only accepts the snr parameter"),
        (lambda: BinaryFrameStep("add_with_snr", "left", "right", {"snr": True}), "requires a numeric snr"),
        (lambda: BinaryOperandStep("+", "", "operand"), "frame and operand input names must be non-empty"),
        (lambda: AddChannelStep("", "added"), "base and added input names must be non-empty"),
        (lambda: AddChannelStep("base", "added", {"source_time_offset": 1.0}), "params only support"),
        (lambda: AddChannelDataStep("", "raw"), "base and data input names must be non-empty"),
        (lambda: AddChannelDataStep("raw", "raw"), "base and data inputs must be distinct"),
        (lambda: AddChannelDataStep("base", "raw", {"gain": 1}), "params only support"),
    ],
)
def test_graph_step_constructors_reject_invalid_inputs(factory: Callable[[], object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


@pytest.mark.parametrize("operator", ["-", "/", "**"])
def test_binary_operand_step_applies_non_additive_array_operators(operator: str) -> None:
    frame = _frame()
    operand = np.full(frame.shape, 2.0 if operator == "**" else 0.5)
    step = BinaryOperandStep(operator, "signal", "operand")

    result = step.apply({"signal": frame, "operand": operand})
    expected = {
        "-": frame - operand,
        "/": frame / operand,
        "**": frame**operand,
    }[operator]

    np.testing.assert_allclose(result.data, expected.data)
    assert result.operation_history == expected.operation_history


@pytest.mark.parametrize(
    "step",
    [
        GraphNodeSpec("n0", BinaryFrameStep("+", "left", "right"), ("left", "right")),
        GraphNodeSpec("n1", BinaryOperandStep("*", "signal", "operand"), ("signal", "operand")),
        GraphNodeSpec("n2", AddChannelStep("base", "added"), ("base", "added")),
        GraphNodeSpec("n3", AddChannelDataStep("base", "raw"), ("base", "raw")),
    ],
)
def test_graph_node_spec_serializes_multi_input_nodes(step: GraphNodeSpec) -> None:
    serialized = step.to_dict()

    assert "inputs" in serialized
    assert "input" not in serialized


def test_graph_node_spec_serializes_unary_input_key() -> None:
    assert GraphNodeSpec("n0", OperationSpec("abs"), ("signal",)).to_dict() == {
        "id": "n0",
        "step": {"operation": "abs", "params": {}},
        "input": "signal",
    }


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: GraphNodeSpec("", OperationSpec("normalize"), ("signal",)), "id must be a non-empty string"),
        (lambda: GraphNodeSpec("n0", OperationSpec("normalize"), ()), "inputs must be non-empty strings"),
        (
            lambda: GraphNodeSpec("n0", BinaryFrameStep("+", "left", "right"), ("right", "left")),
            "binary step inputs must match",
        ),
        (
            lambda: GraphNodeSpec("n0", BinaryOperandStep("+", "signal", "operand"), ("operand", "signal")),
            "binary operand inputs must match",
        ),
        (
            lambda: GraphNodeSpec("n0", AddChannelStep("base", "added"), ("added", "base")),
            "add_channel step inputs must match",
        ),
        (
            lambda: GraphNodeSpec("n0", AddChannelDataStep("base", "raw"), ("raw", "base")),
            "add_channel data inputs must match",
        ),
        (
            lambda: GraphNodeSpec("n0", OperationSpec("normalize"), ("left", "right")),
            "unary step requires exactly one input",
        ),
    ],
)
def test_graph_node_spec_rejects_invalid_identity_and_inputs(factory: Callable[[], object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


@pytest.mark.parametrize("operator", ["-", "*", "/", "**"])
def test_graph_recipe_applies_named_input_recipes_and_frame_binary_operator(operator: str) -> None:
    base = _frame()
    left = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="left")
    right = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="right")
    graph_recipe = GraphRecipeSpec(
        input_recipes={
            "left": RecipeSpec([OperationSpec("abs"), ScalarOperationStep("+", 1.0)]),
            "right": RecipeSpec([OperationSpec("abs"), ScalarOperationStep("+", 1.0)]),
        },
        output=BinaryFrameStep(operator, left="left", right="right"),
    )

    result = graph_recipe.apply({"left": left, "right": right})
    left_processed = left.abs() + 1.0
    right_processed = right.abs() + 1.0
    expected = {
        "-": left_processed - right_processed,
        "*": left_processed * right_processed,
        "/": left_processed / right_processed,
        "**": left_processed**right_processed,
    }[operator]

    np.testing.assert_allclose(result.data, expected.data)
    assert result.operation_history == expected.operation_history


@pytest.mark.parametrize("operator", ["-", "*", "/", "**"])
def test_graph_recipe_from_frame_extracts_root_frame_binary_operator(operator: str) -> None:
    base = _frame()
    left_source = ChannelFrame.from_numpy(base.data, sampling_rate=base.sampling_rate, label="left")
    right_source = ChannelFrame.from_numpy(base.data * 0.25, sampling_rate=base.sampling_rate, label="right")
    left_processed = left_source.abs() + 1.0
    right_processed = right_source.abs() + 1.0
    processed = {
        "-": left_processed - right_processed,
        "*": left_processed * right_processed,
        "/": left_processed / right_processed,
        "**": left_processed**right_processed,
    }[operator]

    graph_recipe = GraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    replayed = graph_recipe.apply({"left": left_source, "right": right_source})

    assert graph_recipe.output == BinaryFrameStep(operator, "left", "right")
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history


def test_graph_recipe_rejects_missing_input() -> None:
    graph_recipe = GraphRecipeSpec(
        input_recipes={"signal": RecipeSpec(()), "noise": RecipeSpec(())},
        output=BinaryFrameStep("+", left="signal", right="noise"),
    )

    with pytest.raises(KeyError, match="GraphRecipeSpec input is missing"):
        graph_recipe.apply({"signal": _frame()})


def test_graph_recipe_rejects_output_reference_outside_named_inputs() -> None:
    with pytest.raises(ValueError, match="GraphRecipeSpec output references unknown input"):
        GraphRecipeSpec(
            input_recipes={"signal": RecipeSpec(())},
            output=BinaryFrameStep("+", left="signal", right="noise"),
        )


def test_graph_recipe_rejects_extra_input_recipe_outside_binary_merge() -> None:
    with pytest.raises(ValueError, match="GraphRecipeSpec input recipes must exactly match output inputs"):
        GraphRecipeSpec(
            input_recipes={"signal": RecipeSpec(()), "noise": RecipeSpec(()), "unused": RecipeSpec(())},
            output=BinaryFrameStep("+", left="signal", right="noise"),
        )


def test_graph_recipe_rejects_reused_binary_input_name() -> None:
    with pytest.raises(ValueError, match="GraphRecipeSpec output requires two distinct inputs"):
        GraphRecipeSpec(
            input_recipes={"signal": RecipeSpec(())},
            output=BinaryFrameStep("+", left="signal", right="signal"),
        )


def test_graph_recipe_rejects_non_binary_output() -> None:
    with pytest.raises(TypeError, match="GraphRecipeSpec output must be a BinaryFrameStep"):
        GraphRecipeSpec(input_recipes={"signal": RecipeSpec(())}, output=cast(Any, TerminalStep("rms")))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: GraphRecipeSpec({}, BinaryFrameStep("+", "left", "right")),
            "requires at least one named input recipe",
        ),
        (
            lambda: GraphRecipeSpec(
                input_recipes={"left": RecipeSpec(()), "right": RecipeSpec(())},
                output=BinaryFrameStep("+", "left", "right"),
                tail_recipe=cast(Any, TerminalStep("rms")),
            ),
            "tail_recipe must be a RecipeSpec",
        ),
        (
            lambda: GraphRecipeSpec(
                input_recipes={"": RecipeSpec(()), "right": RecipeSpec(())},
                output=BinaryFrameStep("+", "left", "right"),
            ),
            "input names must be non-empty strings",
        ),
        (
            lambda: GraphRecipeSpec(
                input_recipes={"left": cast(Any, TerminalStep("rms")), "right": RecipeSpec(())},
                output=BinaryFrameStep("+", "left", "right"),
            ),
            "input recipes must be RecipeSpec instances",
        ),
    ],
)
def test_graph_recipe_constructor_rejects_invalid_structure(factory: Callable[[], object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_graph_recipe_from_frame_rejects_non_frame_or_missing_graph() -> None:
    with pytest.raises(RecipeExtractionError, match="requires a Wandas frame"):
        GraphRecipeSpec.from_frame(object())

    with pytest.raises(RecipeExtractionError, match="requires operation_graph lineage"):
        GraphRecipeSpec.from_frame(_frame())


def test_graph_recipe_from_frame_rejects_duplicate_input_names() -> None:
    base = _frame()
    processed = base + ChannelFrame.from_numpy(base.data * 0.5, sampling_rate=base.sampling_rate)

    with pytest.raises(RecipeExtractionError, match="requires distinct input names"):
        GraphRecipeSpec.from_frame(processed, input_names=("same", "same"))


def test_binary_frame_step_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError, match="BinaryFrameStep operation is outside the replayable binary-frame allowlist"):
        BinaryFrameStep("@", left="signal", right="noise")


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


@pytest.mark.parametrize(
    ("loader", "path", "message"),
    [
        (_load_importable_function, "not_import_path", "function must be a module-level import path"),
        (_load_importable_function, "__main__.custom_scale", "import path must resolve to a module-level function"),
        (
            _load_importable_function,
            "tests.pipeline.custom_recipe_fixtures.callable_scale",
            "import path must resolve to a module-level function",
        ),
        (_load_importable_frame_class, "not_import_path", "output_frame_class must be a module-level import path"),
        (
            _load_importable_frame_class,
            "__main__.ChannelFrame",
            "output_frame_class must resolve to an importable BaseFrame subclass",
        ),
        (
            _load_importable_frame_class,
            "tests.pipeline.custom_recipe_fixtures.custom_scale",
            "output_frame_class must resolve to an importable BaseFrame subclass",
        ),
    ],
)
def test_custom_function_import_loaders_reject_non_importable_targets(
    loader: Callable[[str], object],
    path: str,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        loader(path)


def test_custom_function_import_loaders_reject_non_module_identity_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def local_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        return shape

    class LocalChannelFrame(ChannelFrame):
        pass

    monkeypatch.setattr(custom_recipe_fixtures, "same_shape", local_shape)
    monkeypatch.setattr(custom_recipe_fixtures, "LocalChannelFrame", LocalChannelFrame, raising=False)

    with pytest.raises(TypeError, match="function"):
        _load_importable_function("tests.pipeline.custom_recipe_fixtures.same_shape")
    with pytest.raises(TypeError, match="module-level class"):
        _load_importable_frame_class("tests.pipeline.custom_recipe_fixtures.LocalChannelFrame")


def test_custom_function_loader_rejects_module_level_name_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(custom_scale, "__module__", "math")
    monkeypatch.setattr(custom_scale, "__name__", "sqrt")
    monkeypatch.setattr(custom_scale, "__qualname__", "sqrt")

    with pytest.raises(TypeError, match="module-level function"):
        _load_importable_function("tests.pipeline.custom_recipe_fixtures.custom_scale")


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: CustomFunctionStep(cast(Any, "")), "function must be a non-empty import path string"),
        (lambda: CustomFunctionStep("not_import_path"), "function must be a module import path"),
        (
            lambda: CustomFunctionStep("tests.pipeline.custom_recipe_fixtures.custom_scale", output_shape_function=""),
            "output_shape_function must be None or a non-empty import path string",
        ),
        (
            lambda: CustomFunctionStep(
                "tests.pipeline.custom_recipe_fixtures.custom_scale",
                output_shape_function="not_import_path",
            ),
            "output_shape_function must be a module import path",
        ),
        (
            lambda: CustomFunctionStep(
                "tests.pipeline.custom_recipe_fixtures.custom_scale",
                dask_pure=cast(Any, "yes"),
            ),
            "dask_pure must be a bool",
        ),
        (
            lambda: CustomFunctionStep("tests.pipeline.custom_recipe_fixtures.custom_scale", output_frame_class=""),
            "output_frame_class must be None or a non-empty import path string",
        ),
        (
            lambda: CustomFunctionStep(
                "tests.pipeline.custom_recipe_fixtures.custom_scale",
                output_frame_class="NotImportPath",
            ),
            "output_frame_class must be a module import path",
        ),
        (
            lambda: CustomFunctionStep(
                "tests.pipeline.custom_recipe_fixtures.custom_scale",
                output_frame_kwargs={"freqs": [1.0]},
            ),
            "output_frame_kwargs require output_frame_class",
        ),
    ],
)
def test_custom_function_step_rejects_invalid_metadata(factory: Callable[[], object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_custom_function_step_serializes_output_frame_class_metadata() -> None:
    step = CustomFunctionStep(
        "tests.pipeline.custom_recipe_fixtures.custom_rfft",
        output_shape_function="tests.pipeline.custom_recipe_fixtures.rfft_shape",
        output_frame_class="wandas.frames.spectral.SpectralFrame",
        output_frame_kwargs={"freqs": [0.0, 1.0]},
    )

    assert step.to_dict() == {
        "custom_function": "tests.pipeline.custom_recipe_fixtures.custom_rfft",
        "output_shape_function": "tests.pipeline.custom_recipe_fixtures.rfft_shape",
        "dask_pure": True,
        "params": {},
        "output_frame_class": "wandas.frames.spectral.SpectralFrame",
        "output_frame_kwargs": {"freqs": [0.0, 1.0]},
    }


@pytest.mark.parametrize(
    ("key", "expected_channel"),
    [
        ((slice(0, 2), slice(10, 20)), {"type": "slice", "start": 0, "stop": 2, "step": None}),
        ((np.array([True, False]), slice(10, 20)), {"type": "boolean_mask", "mask": [True, False]}),
        ((1, slice(10, 20)), {"type": "index", "value": 1}),
        ((["left", "right"], slice(10, 20)), {"type": "label_list", "labels": ["left", "right"]}),
        (([0, 1], slice(10, 20)), {"type": "integer_list", "indices": [0, 1]}),
    ],
)
def test_indexing_step_serializes_multidimensional_channel_key_variants(
    key: tuple[object, slice],
    expected_channel: dict[str, object],
) -> None:
    step = IndexingStep(key)

    assert step.to_dict() == {
        "getitem": {
            "type": "multidimensional_slice",
            "channel": expected_channel,
            "axis_slices": [{"start": 10, "stop": 20, "step": None}],
        }
    }


def test_indexing_step_restores_boolean_mask_wrapper_to_public_numpy_mask() -> None:
    step = IndexingStep((_BooleanMask((True, False)), slice(10, 20)))

    key = step.key

    assert isinstance(key, tuple)
    np.testing.assert_array_equal(key[0], np.array([True, False]))


def test_indexing_step_snapshots_direct_boolean_mask_key() -> None:
    step = IndexingStep(np.array([True, False]))

    np.testing.assert_array_equal(step.key, np.array([True, False]))
    assert step.to_dict() == {"getitem": {"type": "boolean_mask", "mask": [True, False]}}


def test_get_channel_query_snapshot_rejects_non_string_key_with_channel_mask() -> None:
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        _snapshot_get_channel_query_params(cast(Any, {1: "left", "channel_mask": [True]}))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: IndexingStep((object(), slice(0, 1))), "multidimensional channel key"),
        (lambda: IndexingStep(slice(0.1, 1)), "slice bounds must be integers or None"),
        (lambda: IndexingStep([]), "key must be a channel label"),
    ],
)
def test_indexing_step_rejects_invalid_keys(factory: Callable[[], object], message: str) -> None:
    with pytest.raises(TypeError, match=message):
        factory()


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (None, "requires importable module-level functions"),
        ({"function": object()}, "requires an importable function path"),
        (
            {
                "function": "tests.pipeline.custom_recipe_fixtures.custom_scale",
                "output_shape_function": object(),
            },
            "requires an importable output_shape_func path",
        ),
        (
            {"function": "tests.pipeline.custom_recipe_fixtures.custom_scale", "dask_pure": "yes"},
            "requires a boolean dask_pure flag",
        ),
        (
            {
                "function": "tests.pipeline.custom_recipe_fixtures.custom_scale",
                "output_frame_class": object(),
            },
            "requires an importable output frame class path",
        ),
        (
            {
                "function": "tests.pipeline.custom_recipe_fixtures.custom_scale",
                "output_frame_class": "tests.pipeline.custom_recipe_fixtures.custom_scale",
            },
            "requires an importable output frame class",
        ),
        (
            {
                "function": "tests.pipeline.custom_recipe_fixtures.custom_scale",
                "output_frame_kwargs": object(),
            },
            "requires output_frame_kwargs mapping",
        ),
        (
            {
                "function": "tests.pipeline.custom_recipe_fixtures.custom_scale",
            },
            "requires recipe-literal params",
        ),
    ],
)
def test_custom_function_extraction_rejects_unreplayable_metadata(
    metadata: dict[str, object] | None,
    message: str,
) -> None:
    params = {"bad": {"nested": 1}} if message == "requires recipe-literal params" else {}

    with pytest.raises(RecipeExtractionError, match=message):
        _custom_function_step_from_graph(params, metadata)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({}, "requires typed mapping items"),
        ({"mapping_items": [("ok",)]}, "requires key/value mapping items"),
        ({"mapping_items": [(object(), "label")]}, "only supports int/str keys"),
    ],
)
def test_rename_mapping_extraction_rejects_invalid_serialized_items(
    params: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RecipeExtractionError, match=message):
        _rename_mapping_from_params(params)


@pytest.mark.parametrize(
    ("operation", "params", "message"),
    [
        ("+", {"operand_kind": "frame"}, "explicit node graph recipe"),
        ("+", {"operand_kind": "operand", "operand": {"type": "int"}}, "numeric scalar operand"),
        ("+", {"operand_kind": "operand", "operand": {"type": "bool", "value": True}}, "numeric scalar operand"),
        ("+", {"operand_kind": "operand", "operand": {"type": "str", "value": "x"}}, "numeric scalar operand"),
        ("+", {"symbol": "-", "operand_kind": "operand", "operand": 1.0}, "inconsistent operator metadata"),
        ("+", {"operand_position": "middle", "operand_kind": "operand", "operand": 1.0}, "invalid operand position"),
    ],
)
def test_scalar_step_extraction_rejects_unstable_operand_metadata(
    operation: str,
    params: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RecipeExtractionError, match=message):
        _scalar_step_from_graph(operation, params)


def test_scalar_step_extraction_accepts_direct_numeric_operand() -> None:
    assert _scalar_step_from_graph("+", {"operand_kind": "operand", "operand": 2}) == ScalarOperationStep("+", 2)


@pytest.mark.parametrize(
    ("func", "args", "message"),
    [
        (_slice_from_serialized, (object(),), "requires serialized slice objects"),
        (_slice_from_serialized, ({"start": 0},), "requires explicit start/stop/step"),
        (
            _slice_from_serialized,
            ({"start": True, "stop": None, "step": 1},),
            "requires integer slice bounds",
        ),
        (_axis_slices_from_params, ({},), "requires non-empty axis_slices"),
        (_indices_from_params, ({},), "requires indices"),
        (_indices_from_params, ({"indices": [True]},), "requires integer indices"),
        (_mask_from_params, ({},), "requires mask values"),
        (_mask_from_params, ({"mask": [True, 1]},), "requires bool values"),
    ],
)
def test_indexing_extraction_rejects_invalid_serialized_selectors(
    func: Callable[..., object],
    args: tuple[object, ...],
    message: str,
) -> None:
    kwargs = {"context": "test"} if func is _slice_from_serialized else {}

    with pytest.raises(RecipeExtractionError, match=message):
        func(*args, **kwargs)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"indexing": "label", "label": 1}, "requires a string label"),
        ({"indexing": "label_list", "labels": [1]}, "requires string labels"),
        ({"indexing": "callable"}, "Indexing recipe extraction only supports"),
    ],
)
def test_getitem_step_extraction_rejects_unreplayable_indexing(
    params: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RecipeExtractionError, match=message):
        _getitem_step_from_graph(params)


def test_getitem_step_extraction_accepts_multidimensional_slice() -> None:
    step = _getitem_step_from_graph(
        {"indexing": "multidimensional_slice", "axis_slices": [{"start": 0, "stop": None, "step": 2}]}
    )

    assert step == IndexingStep((slice(None), slice(0, None, 2)))


@pytest.mark.parametrize(
    ("parent", "message"),
    [
        ({"operation": "get_channel", "params": {"query": {"label": "left"}}}, "only supports single integer"),
        ({"operation": "__getitem__", "params": {"indexing": "callable"}}, "only supports label"),
        ({"operation": "remove_dc", "params": {}}, "requires a replayable channel-selection parent"),
    ],
)
def test_channel_key_extraction_rejects_unreplayable_parent_graph(
    parent: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RecipeExtractionError, match=message):
        _channel_key_from_parent_graph(parent)


@pytest.mark.parametrize(
    ("parent", "expected"),
    [
        ({"operation": "get_channel", "params": {"query": "left"}}, "left"),
        ({"operation": "get_channel", "params": {"channel_idx": np.int64(1)}}, 1),
    ],
)
def test_channel_key_extraction_accepts_public_get_channel_selectors(
    parent: dict[str, object],
    expected: str | int,
) -> None:
    assert _channel_key_from_parent_graph(parent) == expected


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (
            {
                "operation": "__getitem__",
                "params": {
                    "indexing": "multidimensional_slice",
                    "axis_slices": [{"start": 0, "stop": None, "step": 1}],
                },
                "inputs": [],
            },
            "requires one channel-selection parent",
        ),
        (
            {
                "operation": "__getitem__",
                "params": {
                    "indexing": "multidimensional_slice",
                    "axis_slices": [{"start": 0, "stop": None, "step": 1}],
                },
                "inputs": [{"operation": "get_channel", "params": {"query": "left"}, "inputs": [{}, {}]}],
            },
            "explicit node graph recipe",
        ),
    ],
)
def test_steps_from_graph_rejects_invalid_multidimensional_parent_shapes(
    graph: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RecipeExtractionError, match=message):
        _steps_from_graph(graph)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _binary_frame_step_from_graph("+", {"operand_kind": "operand"}, "left", "right"),
            "only supports root binary frame operations",
        ),
        (
            lambda: _binary_operand_step_from_graph("+", {"operand_kind": "frame"}, "frame", "operand"),
            "only supports external ndarray",
        ),
        (
            lambda: _add_channel_step_from_graph({"input_kind": "ndarray"}, "base", "added"),
            "only supports ChannelFrame inputs",
        ),
        (
            lambda: _add_channel_data_step_from_graph({"input_kind": "frame"}, "base", "raw"),
            "only supports external ndarray",
        ),
    ],
)
def test_graph_step_extraction_rejects_unreplayable_binary_metadata(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(RecipeExtractionError, match=message):
        factory()


def test_validate_replayable_operation_rejects_unregistered_operation() -> None:
    with pytest.raises(RecipeExtractionError, match="outside the Stage 1 recipe allowlist"):
        _validate_replayable_operation("definitely_missing_operation")


def test_validate_replayable_operation_rejects_multi_input_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MultiInputOperation(AudioOperation[NDArrayReal, NDArrayReal]):
        name = "_recipe_multi_input"
        _expected_input_count = 2

        def _process(self, x: NDArrayReal) -> NDArrayReal:
            return x

    monkeypatch.setitem(_OPERATION_REGISTRY, MultiInputOperation.name, MultiInputOperation)

    with pytest.raises(RecipeExtractionError, match=r"NodeGraphRecipeSpec\.from_frame"):
        _validate_replayable_operation(MultiInputOperation.name)


def test_validate_replayable_operation_reports_runtime_inputs_in_recipe_spec_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MultiInputOperation(AudioOperation[NDArrayReal, NDArrayReal]):
        name = "_recipe_multi_input_recipe_spec_context"
        _expected_input_count = 2

        def _process(self, x: NDArrayReal) -> NDArrayReal:
            return x

    monkeypatch.setitem(_OPERATION_REGISTRY, MultiInputOperation.name, MultiInputOperation)

    with pytest.raises(RecipeExtractionError) as exc_info:
        _validate_replayable_operation(MultiInputOperation.name, recipe_spec_context=True)

    message = str(exc_info.value)
    assert "RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe" in message
    assert "Runtime inputs: 2" in message


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


def test_rename_channels_method_step_allows_non_mapping_params_through_generic_snapshot() -> None:
    step = MethodStep("rename_channels", {"mapping_items": ["left", "front-left"]})

    assert step.params == {"mapping_items": ["left", "front-left"]}


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"mapping": object()}, "rename_channels mapping must be a mapping"),
        ({"mapping": {True: "bad"}}, "mapping keys must be int or str"),
        ({"mapping": {object(): "bad"}}, "mapping keys must be int or str"),
        ({"mapping": {"left": 1}}, "mapping values must be strings"),
    ],
)
def test_rename_channels_method_step_rejects_invalid_mapping_params(
    params: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        MethodStep("rename_channels", params)


def test_get_channel_method_step_snapshots_boolean_mask_and_query_mapping() -> None:
    step = MethodStep(
        "get_channel",
        {"query": {"label": "left", "index": 0}, "channel_mask": [np.bool_(True), False]},
    )

    assert step.params == {"query": {"index": 0, "label": "left"}, "channel_mask": [True, False]}


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"channel_mask": object()}, "channel_mask must be a shallow sequence"),
        ({"channel_mask": [True, 1]}, "channel_mask must contain only bool values"),
        ({"query": {object(): "left"}}, "get_channel query keys must be strings"),
        ({"query": {"label": "left"}, object(): "bad"}, "OperationSpec params mapping keys must be strings"),
    ],
)
def test_get_channel_method_step_rejects_invalid_query_params(
    params: dict[object, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        MethodStep("get_channel", params)  # ty: ignore[invalid-argument-type]


def test_restore_history_value_handles_negative_infinity_and_nested_nan() -> None:
    restored = _restore_history_value(
        {
            "floor": {"type": "float", "value": "-inf"},
            "values": [{"type": "float", "value": "nan"}],
        }
    )

    assert restored["floor"] == float("-inf")
    assert np.isnan(restored["values"][0])


def test_method_step_rejects_methods_outside_replay_allowlist() -> None:
    with pytest.raises(ValueError, match="MethodStep method is outside the replayable method allowlist"):
        MethodStep("plot")


def test_method_step_rejects_inplace_mutation_params() -> None:
    with pytest.raises(TypeError, match="cannot replay in-place frame mutations"):
        MethodStep("remove_channel", {"key": 0, "inplace": True})


def test_typed_method_step_rejects_methods_outside_replay_allowlist() -> None:
    with pytest.raises(ValueError, match="TypedMethodStep method is outside the replayable typed-method allowlist"):
        TypedMethodStep("plot")


def test_scalar_operation_step_rejects_operations_outside_replay_allowlist() -> None:
    with pytest.raises(ValueError, match="ScalarOperationStep operation is outside the replayable scalar allowlist"):
        ScalarOperationStep("%", 2)


def test_scalar_operation_step_rejects_non_numeric_operands() -> None:
    with pytest.raises(TypeError, match="ScalarOperationStep operand must be an int or float"):
        ScalarOperationStep("+", "2")  # ty: ignore[invalid-argument-type]


def test_scalar_operation_step_rejects_non_bool_reverse_flag_and_serializes_reverse() -> None:
    with pytest.raises(TypeError, match="reverse must be a bool"):
        ScalarOperationStep("+", 2, reverse=cast(Any, "yes"))

    assert ScalarOperationStep("-", 2, reverse=True).to_dict() == {
        "scalar_operation": "-",
        "operand": 2,
        "reverse": True,
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_scalar_operation_step_defensive_assertion_for_unreachable_operation(reverse: bool) -> None:
    step = object.__new__(ScalarOperationStep)
    object.__setattr__(step, "symbol", "%")
    object.__setattr__(step, "operand", 2)
    object.__setattr__(step, "reverse", reverse)

    with pytest.raises(AssertionError, match="Unhandled"):
        step.apply(_frame())


def test_binary_operand_step_defensive_assertion_for_unreachable_operation() -> None:
    step = object.__new__(BinaryOperandStep)
    object.__setattr__(step, "operation", "%")
    object.__setattr__(step, "frame", "signal")
    object.__setattr__(step, "operand", "operand")

    with pytest.raises(AssertionError, match="Unhandled binary operand operation"):
        step.apply({"signal": _frame(), "operand": np.ones(_frame().shape)})


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
        (
            lambda frame: frame.high_pass_filter(cutoff=100.0, order=2),
            OperationSpec("highpass_filter", {"cutoff": 100.0, "order": 2}),
        ),
        (
            lambda frame: frame.low_pass_filter(cutoff=1000.0, order=3),
            OperationSpec("lowpass_filter", {"cutoff": 1000.0, "order": 3}),
        ),
        (
            lambda frame: frame.band_pass_filter(low_cutoff=100.0, high_cutoff=1000.0, order=2),
            OperationSpec("bandpass_filter", {"low_cutoff": 100.0, "high_cutoff": 1000.0, "order": 2}),
        ),
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
            MethodStep("rms_trend", {"frame_length": 512, "hop_length": 128, "dB": True, "Aw": False}),
        ),
        (
            lambda frame: frame.sound_level(freq_weighting="Z", time_weighting="Fast", dB=True),
            MethodStep("sound_level", {"freq_weighting": "Z", "time_weighting": "Fast", "dB": True}),
        ),
    ],
)
def test_recipe_from_frame_extracts_ref_bearing_frame_methods(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    expected_step: MethodStep,
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
    "build_frame",
    [
        lambda frame: frame.rms_trend(frame_length=512, hop_length=128, dB=True, Aw=False),
        lambda frame: frame.sound_level(freq_weighting="Z", time_weighting="Fast", dB=True),
    ],
)
def test_ref_bearing_frame_method_recipes_recompute_target_refs(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
) -> None:
    source = _two_channel_frame_with_refs()
    target = _two_channel_frame_with_reordered_refs()
    processed = build_frame(source)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(target)
    expected = build_frame(target)

    assert replayed.operation_history == expected.operation_history
    np.testing.assert_allclose(replayed.data, expected.data)


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


def test_fix_length_duration_recipe_extraction_replays_computed_length() -> None:
    frame = _frame()
    original_shape = frame.shape
    processed = frame.fix_length(duration=0.25)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (MethodStep("fix_length", {"length": 4000}),)
    assert recipe.to_dict() == {"steps": [{"method": "fix_length", "params": {"length": 4000}}]}
    assert replayed.operation_history == processed.operation_history
    assert replayed.shape == processed.shape == (4000,)
    assert frame.shape == original_shape
    assert isinstance(replayed._data, DaArray)


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

    assert recipe.steps == (MethodStep("channel_difference", {"other_channel": "left"}),)
    assert recipe.to_dict() == {"steps": [{"method": "channel_difference", "params": {"other_channel": "left"}}]}
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.labels == processed.labels
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.shape == processed.shape


def test_channel_difference_recipe_replays_label_intent_on_reordered_channels() -> None:
    source = _two_channel_frame_with_refs()
    target = _two_channel_frame_with_reordered_refs()
    processed = source.channel_difference(other_channel="right")

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(target)
    expected = target.channel_difference(other_channel="right")

    assert recipe.steps == (MethodStep("channel_difference", {"other_channel": "right"}),)
    np.testing.assert_allclose(replayed.data, expected.data)
    assert replayed.labels == expected.labels


@pytest.mark.parametrize(
    ("build_frame", "expected_step"),
    [
        (lambda frame: frame.get_channel(1), MethodStep("get_channel", {"channel_idx": 1})),
        (lambda frame: frame.get_channel([0, 2]), MethodStep("get_channel", {"channel_idx": [0, 2]})),
        (lambda frame: frame.get_channel((2, 0)), MethodStep("get_channel", {"channel_idx": [2, 0]})),
        (lambda frame: frame.get_channel(np.array([2, 0])), MethodStep("get_channel", {"channel_idx": [2, 0]})),
        (
            lambda frame: frame.get_channel(query="right"),
            MethodStep("get_channel", {"query": "right", "validate_query_keys": True}),
        ),
        (
            lambda frame: frame.get_channel(query={"label": "right"}),
            MethodStep("get_channel", {"query": {"label": "right"}, "validate_query_keys": True}),
        ),
        (
            lambda frame: frame.get_channel(query={"unit": "Pa", "ref": 2.0}),
            MethodStep("get_channel", {"query": {"unit": "Pa", "ref": 2.0}, "validate_query_keys": True}),
        ),
    ],
)
def test_recipe_from_frame_extracts_channel_selection_method_step(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    expected_step: MethodStep,
) -> None:
    base = _frame()
    frame = ChannelFrame(
        data=da.from_array(np.vstack([base.data, base.data * 0.5, base.data * 0.25]), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="left", unit="Pa", ref=2.0),
            ChannelMetadata(label="right", unit="Pa", ref=2.0),
            ChannelMetadata(label="rear", unit="V", ref=1.0),
        ],
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


def test_get_channel_boolean_mask_recipe_extraction_snapshots_mask() -> None:
    base = _frame()
    frame = ChannelFrame(
        data=da.from_array(np.vstack([base.data, base.data * 0.5]), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="right"),
        ],
    )
    mask = np.array([False, True])
    processed = frame.get_channel(mask)
    mask[1] = False

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (MethodStep("get_channel", {"channel_mask": [False, True]}),)
    assert recipe.to_dict() == {"steps": [{"method": "get_channel", "params": {"channel_mask": [False, True]}}]}
    assert replayed.operation_history == processed.operation_history
    assert isinstance(replayed._data, DaArray)
    assert replayed.labels == ["right"]


def test_get_channel_boolean_mask_step_rejects_non_bool_values() -> None:
    with pytest.raises(TypeError, match="get_channel channel_mask"):
        MethodStep("get_channel", {"channel_mask": [0, 2, 0]})


def test_get_channel_dict_query_recipe_extraction_snapshots_query() -> None:
    base = _frame()
    frame = ChannelFrame(
        data=da.from_array(np.vstack([base.data, base.data * 0.5]), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="left", unit="V"),
            ChannelMetadata(label="right", unit="Pa"),
        ],
    )
    query = {"unit": "Pa"}
    processed = frame.get_channel(query=query)
    query["unit"] = "V"

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.to_dict() == {
        "steps": [{"method": "get_channel", "params": {"query": {"unit": "Pa"}, "validate_query_keys": True}}]
    }
    assert replayed.operation_history == processed.operation_history
    assert replayed.labels == ["right"]


def test_get_channel_dict_query_recipe_extraction_rejects_non_builtin_real() -> None:
    base = _frame()
    ratio = Fraction(1, 3)
    frame = ChannelFrame(
        data=da.from_array(np.vstack([base.data, base.data * 0.5]), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="left", extra={"ratio": Fraction(1, 2)}),
            ChannelMetadata(label="right", extra={"ratio": ratio}),
        ],
    )

    processed = frame.get_channel(query={"ratio": ratio})

    with pytest.raises(RecipeExtractionError, match="Channel selection recipe extraction only supports"):
        RecipeSpec.from_frame(processed)


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
        lambda frame: frame.get_channel(query={"label": re.compile("right")}),
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


def test_recipe_from_frame_extracts_boolean_mask_channel_and_time_indexing() -> None:
    frame = _two_channel_frame_with_refs()
    mask = np.array([False, True])
    processed = frame[mask, 10:20]
    mask[1] = False

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.to_dict() == {
        "steps": [
            {
                "getitem": {
                    "type": "multidimensional_slice",
                    "channel": {"type": "boolean_mask", "mask": [False, True]},
                    "axis_slices": [{"start": 10, "stop": 20, "step": None}],
                }
            }
        ]
    }
    assert replayed.operation_history == processed.operation_history
    assert replayed.labels == ["right"]
    np.testing.assert_allclose(replayed.data, processed.data)


def test_getitem_boolean_mask_recipe_extraction_snapshots_mask() -> None:
    frame = _two_channel_frame_with_refs()
    mask = np.array([False, True])
    processed = frame[mask]
    mask[1] = False

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.to_dict() == {"steps": [{"getitem": {"type": "boolean_mask", "mask": [False, True]}}]}
    assert replayed.operation_history == processed.operation_history
    assert isinstance(replayed._data, DaArray)
    assert replayed.labels == ["right"]


def test_getitem_label_recipe_replays_single_label_getitem_semantics_with_duplicates() -> None:
    base = _frame()
    source = ChannelFrame(
        data=da.from_array(np.vstack([base.data, base.data * 0.5]), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="left"),
        ],
    )
    target = ChannelFrame(
        data=da.from_array(np.vstack([base.data * 0.25, base.data * 0.75]), chunks=(1, -1)),
        sampling_rate=base.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="left"),
        ],
    )
    processed = source["left"]

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(target)
    expected = target["left"]

    assert recipe.steps == (IndexingStep("left"),)
    assert recipe.to_dict() == {"steps": [{"getitem": {"type": "label", "label": "left"}}]}
    assert replayed.n_channels == 1
    assert replayed.labels == expected.labels == ["left"]
    np.testing.assert_allclose(replayed.data, expected.data)


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[True, False]]),
        np.array([[False], [True]]),
    ],
)
def test_getitem_boolean_mask_recipe_extraction_rejects_multidimensional_mask(mask: np.ndarray[Any, Any]) -> None:
    frame = _two_channel_frame_with_refs()

    with pytest.raises(ValueError, match="Boolean mask"):
        _ = frame[mask]


def test_indexing_step_rejects_python_boolean_list() -> None:
    with pytest.raises(TypeError, match="IndexingStep key must be"):
        IndexingStep([True, False])


@pytest.mark.parametrize(
    "key",
    [
        [True, False],
        ([True, False], slice(100, 200)),
    ],
)
def test_recipe_from_frame_rejects_python_boolean_list_indexing(key: Any) -> None:
    frame = _two_channel_frame_with_refs()

    with pytest.raises(TypeError, match="List must contain all str or all int"):
        _ = frame[key]


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


def test_getitem_integer_list_recipe_extraction_snapshots_indices() -> None:
    frame = _two_channel_frame_with_refs()
    indices = [1, 0]
    processed = frame[indices]
    indices[0] = 0

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.to_dict() == {"steps": [{"getitem": {"type": "integer_list", "indices": [1, 0]}}]}
    assert replayed.operation_history == processed.operation_history
    assert replayed.labels == ["right", "left"]


def test_getitem_integer_array_recipe_extraction_snapshots_indices() -> None:
    frame = _two_channel_frame_with_refs()
    indices = np.array([1, 0])
    processed = frame[indices]
    indices[0] = 0

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.to_dict() == {"steps": [{"getitem": {"type": "integer_list", "indices": [1, 0]}}]}
    assert replayed.operation_history == processed.operation_history
    assert isinstance(replayed._data, DaArray)
    assert replayed.labels == ["right", "left"]


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
            lambda frame: frame["right"],
            {"getitem": {"type": "label", "label": "right"}},
        ),
        (
            lambda frame: frame[0:2],
            {"getitem": {"type": "channel_slice", "start": 0, "stop": 2, "step": None}},
        ),
        (
            lambda frame: frame[["left", "right"]],
            {"getitem": {"type": "label_list", "labels": ["left", "right"]}},
        ),
        (
            lambda frame: frame[[1, 0]],
            {"getitem": {"type": "integer_list", "indices": [1, 0]}},
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
                    "channel": {"type": "label", "label": "right"},
                    "axis_slices": [{"start": 200, "stop": 600, "step": None}],
                }
            },
        ),
        (
            lambda frame: frame[[1, 0], 200:600],
            {
                "getitem": {
                    "type": "multidimensional_slice",
                    "channel": {"type": "integer_list", "indices": [1, 0]},
                    "axis_slices": [{"start": 200, "stop": 600, "step": None}],
                }
            },
        ),
        (
            lambda frame: frame[np.array([1, 0]), 200:600],
            {
                "getitem": {
                    "type": "multidimensional_slice",
                    "channel": {"type": "integer_list", "indices": [1, 0]},
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


def test_recipe_from_frame_rejects_non_slice_multidimensional_indexing_lineage() -> None:
    frame = _two_channel_frame_with_refs()
    processed = frame[:, 100:400]
    assert processed.lineage is not None
    processed._lineage = frame._lineage_with_method("__getitem__", {"indexing": "multidimensional"})

    with pytest.raises(RecipeExtractionError, match="Indexing recipe extraction only supports"):
        RecipeSpec.from_frame(processed)


def test_multidimensional_label_indexing_recipe_replays_label_intent_on_reordered_channels() -> None:
    source = _two_channel_frame_with_refs()
    target = _two_channel_frame_with_reordered_refs()
    processed = source["right", 200:600]

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(target)
    expected = target["right", 200:600]

    assert recipe.to_dict() == {
        "steps": [
            {
                "getitem": {
                    "type": "multidimensional_slice",
                    "channel": {"type": "label", "label": "right"},
                    "axis_slices": [{"start": 200, "stop": 600, "step": None}],
                }
            }
        ]
    }
    assert replayed.labels == expected.labels
    np.testing.assert_allclose(replayed.data, expected.data)


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


@pytest.mark.parametrize("operand", [np.float64(0.25), np.float32(0.25), np.int64(2)])
def test_recipe_from_frame_extracts_value_bearing_numpy_scalar_operation(operand: Any) -> None:
    frame = _frame()
    processed = frame + operand

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    expected_operand = float(operand) if isinstance(operand, np.floating) else int(operand)
    assert recipe.steps == (ScalarOperationStep("+", expected_operand),)
    np.testing.assert_allclose(replayed.data, processed.data)


@pytest.mark.parametrize(
    ("build_frame", "step"),
    [
        (lambda frame: 0.5 + frame, ScalarOperationStep("+", 0.5, reverse=True)),
        (lambda frame: 0.5 - frame, ScalarOperationStep("-", 0.5, reverse=True)),
        (lambda frame: 2.0 * frame, ScalarOperationStep("*", 2.0, reverse=True)),
        (lambda frame: 2.0 / frame, ScalarOperationStep("/", 2.0, reverse=True)),
        (lambda frame: 2.0**frame, ScalarOperationStep("**", 2.0, reverse=True)),
        (lambda frame: np.float64(0.5) + frame, ScalarOperationStep("+", 0.5, reverse=True)),
        (lambda frame: np.float64(0.5) - frame, ScalarOperationStep("-", 0.5, reverse=True)),
        (lambda frame: np.float64(2.0) * frame, ScalarOperationStep("*", 2.0, reverse=True)),
        (lambda frame: np.float64(2.0) / frame, ScalarOperationStep("/", 2.0, reverse=True)),
        (lambda frame: np.float64(2.0) ** frame, ScalarOperationStep("**", 2.0, reverse=True)),
    ],
)
def test_recipe_from_frame_extracts_reverse_numeric_scalar_operations(
    build_frame: Callable[[ChannelFrame], ChannelFrame],
    step: ScalarOperationStep,
) -> None:
    frame = _frame()
    processed = build_frame(frame)

    recipe = RecipeSpec.from_frame(processed)
    replayed = recipe.apply(frame)

    assert recipe.steps == (step,)
    np.testing.assert_allclose(replayed.data, processed.data)
    assert replayed.operation_history == processed.operation_history
    assert replayed.labels == processed.labels


def test_numpy_array_left_operator_remains_outside_frame_recipe_boundary() -> None:
    frame = _frame()

    result = np.ones(frame.shape) + frame

    assert isinstance(result, np.ndarray)


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


def test_recipe_from_frame_empty_history_returns_linear_recipe() -> None:
    recipe = RecipeSpec.from_frame(_frame())

    assert isinstance(recipe, RecipeSpec)
    assert recipe.steps == ()


@pytest.mark.parametrize(
    ("operation_name", "build_frame", "message"),
    [
        (
            "binary frame operation",
            lambda frame: frame + _frame().normalize(),
            "RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe",
        ),
        (
            "array operand operation",
            lambda frame: frame + np.ones(frame.shape),
            "NodeGraphRecipeSpec.from_frame(...)",
        ),
        (
            "custom apply operation",
            lambda frame: frame.apply(
                lambda data, gain: data * gain,
                output_shape_func=lambda shape: shape,
                gain=2.0,
            ),
            "Custom operation recipe extraction requires importable",
        ),
        (
            "registered operation outside current allowlist",
            lambda frame: frame.apply_operation("_recipe_boundary_noop"),
            "Operation is outside the Stage 1 recipe allowlist",
        ),
    ],
)
def test_recipe_from_frame_reports_current_boundary_for_non_replayable_operations(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
    build_frame: Callable[[ChannelFrame], object],
    message: str,
) -> None:
    if operation_name == "registered operation outside current allowlist":
        _register_recipe_boundary_operation(monkeypatch)
    frame = _frame()
    processed = build_frame(frame)

    with pytest.raises(RecipeExtractionError, match=re.escape(message)):
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

    with pytest.raises(RecipeExtractionError, match=r"GraphRecipeSpec\.from_frame"):
        RecipeSpec.from_frame(processed)


def test_recipe_from_frame_reports_explicit_graph_extractors_for_binary_merge() -> None:
    frame = _frame()
    noise = _frame().remove_dc()
    processed = frame.normalize() + noise

    with pytest.raises(RecipeExtractionError) as exc_info:
        RecipeSpec.from_frame(processed)

    message = str(exc_info.value)
    assert "RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe" in message
    assert "RecipeSpec.from_frame(...) only supports single-input linear recipes" in message
    assert "GraphRecipeSpec.from_frame(...)" in message
    assert "NodeGraphRecipeSpec.from_frame(...)" in message
    assert isinstance(GraphRecipeSpec.from_frame(processed), GraphRecipeSpec)


def test_recipe_from_frame_reports_node_graph_extractor_for_external_operand() -> None:
    frame = _frame()
    processed = frame + np.ones(frame.shape)

    with pytest.raises(RecipeExtractionError) as exc_info:
        RecipeSpec.from_frame(processed)

    message = str(exc_info.value)
    assert "RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe" in message
    assert "NodeGraphRecipeSpec.from_frame(...)" in message
    assert "external operands" in message
    assert isinstance(
        NodeGraphRecipeSpec.from_frame(processed, input_names=("signal", "offset")),
        NodeGraphRecipeSpec,
    )


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
    with pytest.raises(RecipeExtractionError, match=r"NodeGraphRecipeSpec\.from_frame"):
        RecipeSpec.from_frame(processed)
