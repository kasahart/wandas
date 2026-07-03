from collections.abc import Callable

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.pipeline import (
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


@pytest.mark.parametrize(
    "value",
    [
        np.array([1.0, 2.0]),
        object(),
        b"bytes",
        1 + 2j,
        {1, 2},
        frozenset({1, 2}),
        {"nested": 1},
        [1, 2],
        (1, 2),
    ],
)
def test_operation_spec_rejects_non_flat_literal_params(value: object) -> None:
    with pytest.raises(TypeError, match="OperationSpec params must be flat recipe-literal values"):
        OperationSpec("normalize", {"value": value})


def test_operation_spec_rejects_nan_params() -> None:
    with pytest.raises(TypeError, match="OperationSpec params must not contain NaN"):
        OperationSpec("normalize", {"norm": float("nan")})


def test_operation_spec_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(TypeError, match="OperationSpec params mapping keys must be strings"):
        OperationSpec("normalize", {object(): "value"})  # ty: ignore[invalid-argument-type]


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
            "metadata-aware registered operation outside current allowlist",
            lambda frame: ChannelFrame.from_numpy(
                np.vstack([frame.data, frame.data * 0.5]),
                sampling_rate=frame.sampling_rate,
                ch_labels=["left", "right"],
            ).channel_difference(other_channel=0),
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
