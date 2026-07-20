from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from dask.array.core import Array as DaArray

sklearn_pipeline = pytest.importorskip("sklearn.pipeline")
Pipeline = sklearn_pipeline.Pipeline

from tests.frame_helpers import channel_first_values  # noqa: E402
from wandas.frames.channel import ChannelFrame  # noqa: E402
from wandas.pipeline import RecipePlan  # noqa: E402
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
    transformer = WandasOperationTransformer("high_pass_filter", cutoff=100.0, order=2)

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


def test_named_transformer_uses_declared_public_recipe_operation() -> None:
    source = _frame()
    result = HighPassFilter(cutoff=100.0, order=2).transform(source)
    plan = RecipePlan.from_frame(result, input_names=("signal",))

    replayed = plan.apply({"signal": source})

    assert type(replayed) is ChannelFrame
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(result))


def test_generic_transformer_dispatches_declared_public_recipe_operation() -> None:
    result = WandasOperationTransformer("fft", n_fft=16).transform(_frame())
    plan = RecipePlan.from_frame(result, input_names=("signal",))

    replayed = plan.apply({"signal": _frame()})

    assert type(replayed) is type(result)
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(result))


def test_generic_transformer_rejects_non_public_operation_name() -> None:
    with pytest.raises(ValueError, match="declared public Recipe Frame method"):
        WandasOperationTransformer("highpass_filter", cutoff=100.0).transform(_frame())


@pytest.mark.parametrize("method_name", ["compute", "persist"])
def test_generic_transformer_rejects_undeclared_eager_methods(method_name: str) -> None:
    with patch.object(DaArray, method_name, autospec=True, side_effect=AssertionError("unexpected side effect")):
        with pytest.raises(ValueError, match="declared public Recipe Frame method"):
            WandasOperationTransformer(method_name).transform(_frame())


def test_generic_transformer_does_not_evaluate_undeclared_properties() -> None:
    property_reads = 0

    class TrapFrame(ChannelFrame):
        @property
        def trap(self) -> Any:
            nonlocal property_reads
            property_reads += 1
            return self.normalize

    frame = TrapFrame.from_numpy(np.ones((1, 8)), sampling_rate=8000)

    with pytest.raises(ValueError, match="declared public Recipe Frame method"):
        WandasOperationTransformer("trap").transform(frame)

    assert property_reads == 0
