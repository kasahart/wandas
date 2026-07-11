import numpy as np
import pytest

sklearn_pipeline = pytest.importorskip("sklearn.pipeline")
Pipeline = sklearn_pipeline.Pipeline

from wandas.frames.channel import ChannelFrame  # noqa: E402
from wandas.pipeline import OperationSpec  # noqa: E402
from wandas.pipeline.sklearn import (  # noqa: E402
    BandPassFilter,
    HighPassFilter,
    LowPassFilter,
    Normalize,
    RemoveDC,
    WandasOperationTransformer,
)


def _frame() -> ChannelFrame:
    sampling_rate = 16000
    time = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = (0.25 + np.sin(2 * np.pi * 50 * time) + np.sin(2 * np.pi * 1000 * time)).reshape(1, -1)
    return ChannelFrame.from_numpy(data, sampling_rate=sampling_rate)


def test_operation_transformer_fit_transform_and_to_spec() -> None:
    frame = _frame()
    transformer = WandasOperationTransformer("highpass_filter", cutoff=100.0, order=2)

    assert transformer.fit(frame) is transformer
    result = transformer.transform(frame)

    assert result is not frame
    assert result.operation_history[-1]["operation"] == "highpass_filter"
    assert transformer.to_spec() == OperationSpec("highpass_filter", {"cutoff": 100.0, "order": 2})


def test_named_transformer_get_params_set_params_and_to_spec() -> None:
    transformer = HighPassFilter(cutoff=100.0)

    assert transformer.get_params() == {"cutoff": 100.0, "order": 4}

    returned = transformer.set_params(cutoff=200.0, order=6)

    assert returned is transformer
    assert transformer.get_params() == {"cutoff": 200.0, "order": 6}
    assert transformer.to_spec() == OperationSpec("highpass_filter", {"cutoff": 200.0, "order": 6})


def test_generic_operation_transformer_get_params_set_params_and_rejects_unknown_param() -> None:
    transformer = WandasOperationTransformer("normalize", norm=2.0)

    assert transformer.get_params() == {"operation": "normalize", "norm": 2.0}
    transformer.set_params(norm=3.0)
    assert transformer.get_params() == {"operation": "normalize", "norm": 3.0}

    returned = transformer.set_params(operation="remove_dc")

    assert returned is transformer
    assert transformer.get_params() == {"operation": "remove_dc"}
    assert transformer.to_spec() == OperationSpec("remove_dc", {})

    transformer.set_params(operation="normalize", norm=1.0)
    assert transformer.get_params() == {"operation": "normalize", "norm": 1.0}
    assert transformer.to_spec() == OperationSpec("normalize", {"norm": 1.0})
    with pytest.raises(ValueError, match="Invalid parameter"):
        transformer.set_params(missing=True)

    with pytest.raises(TypeError, match="operation must be a string"):
        transformer.set_params(operation=42)


def test_sklearn_pipeline_transform_applies_wandas_operations_in_order() -> None:
    frame = _frame()
    pipeline = Pipeline(
        [
            ("hp", HighPassFilter(cutoff=100.0, order=2)),
            ("norm", Normalize()),
        ]
    )

    result = pipeline.transform(frame)

    assert [record["operation"] for record in result.operation_history] == ["highpass_filter", "normalize"]


def test_named_transformers_emit_expected_operation_specs() -> None:
    assert LowPassFilter(cutoff=1000.0).to_spec() == OperationSpec("lowpass_filter", {"cutoff": 1000.0, "order": 4})
    assert BandPassFilter(low_cutoff=100.0, high_cutoff=1000.0, order=2).to_spec() == OperationSpec(
        "bandpass_filter", {"low_cutoff": 100.0, "high_cutoff": 1000.0, "order": 2}
    )
    assert Normalize(norm=np.inf, axis=-1, threshold=None, fill=None).to_spec() == OperationSpec(
        "normalize", {"norm": np.inf, "axis": -1, "threshold": None, "fill": None}
    )
    assert RemoveDC().to_spec() == OperationSpec("remove_dc", {})
