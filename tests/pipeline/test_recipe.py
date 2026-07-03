from collections.abc import Callable

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import MethodStep, OperationSpec, RecipeExtractionError, RecipeSpec


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


def test_recipe_from_frame_empty_history_returns_empty_recipe() -> None:
    assert RecipeSpec.from_frame(_frame()).steps == ()


@pytest.mark.parametrize(
    ("operation_name", "build_frame", "message"),
    [
        (
            "fft domain transition",
            lambda frame: frame.fft(),
            "Operation is outside the Stage 1 recipe allowlist",
        ),
        (
            "stft domain transition",
            lambda frame: frame.stft(n_fft=512, hop_length=128),
            "Operation is outside the Stage 1 recipe allowlist",
        ),
        (
            "binary scalar operation",
            lambda frame: frame + 0.1,
            "Operation is outside the Stage 1 recipe allowlist",
        ),
        (
            "binary frame operation",
            lambda frame: frame + _frame().normalize(),
            "Operation is outside the Stage 1 recipe allowlist",
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
            "registered operation outside stage one",
            lambda frame: frame.abs(),
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
