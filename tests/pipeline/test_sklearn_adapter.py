import numpy as np
import pytest

sklearn_pipeline = pytest.importorskip("sklearn.pipeline")
Pipeline = sklearn_pipeline.Pipeline

from wandas.frames.channel import ChannelFrame  # noqa: E402
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


def test_operation_transformer_fit_and_transform() -> None:
    frame = _frame()
    transformer = WandasOperationTransformer("highpass_filter", cutoff=100.0, order=2)

    assert transformer.fit(frame) is transformer
    result = transformer.transform(frame)

    assert result is not frame
    assert result.operation_history[-1]["operation"] == "wandas.audio.highpass_filter"


def test_named_transformer_get_params_and_set_params() -> None:
    transformer = HighPassFilter(cutoff=100.0)

    assert transformer.get_params() == {"cutoff": 100.0, "order": 4}

    returned = transformer.set_params(cutoff=200.0, order=6)

    assert returned is transformer
    assert transformer.get_params() == {"cutoff": 200.0, "order": 6}


def test_generic_operation_transformer_get_params_set_params_and_rejects_unknown_param() -> None:
    transformer = WandasOperationTransformer("normalize", norm=2.0)

    assert transformer.get_params() == {"operation": "normalize", "norm": 2.0}
    transformer.set_params(norm=3.0)
    assert transformer.get_params() == {"operation": "normalize", "norm": 3.0}

    returned = transformer.set_params(operation="remove_dc")

    assert returned is transformer
    assert transformer.get_params() == {"operation": "remove_dc"}

    transformer.set_params(operation="normalize", norm=1.0)
    assert transformer.get_params() == {"operation": "normalize", "norm": 1.0}
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

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.highpass_filter",
        "wandas.audio.normalize",
    ]


def test_named_transformers_expose_expected_sklearn_params() -> None:
    assert LowPassFilter(cutoff=1000.0).get_params() == {"cutoff": 1000.0, "order": 4}
    assert BandPassFilter(low_cutoff=100.0, high_cutoff=1000.0, order=2).get_params() == {
        "low_cutoff": 100.0,
        "high_cutoff": 1000.0,
        "order": 2,
    }
    assert Normalize(norm=np.inf, axis=-1, threshold=None, fill=None).get_params() == {
        "norm": np.inf,
        "axis": -1,
        "threshold": None,
        "fill": None,
    }
    assert RemoveDC().get_params() == {}
