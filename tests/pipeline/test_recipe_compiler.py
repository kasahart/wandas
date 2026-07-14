from __future__ import annotations

import re
from typing import Any
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipeExtractionError, RecipePlan


def _frame(value: float = 1.0) -> ChannelFrame:
    data = np.full((2, 16), value, dtype=float)
    return ChannelFrame.from_numpy(data, sampling_rate=8000, ch_labels=["left", "right"])


def test_compiler_preserves_shared_source_identity_across_branches() -> None:
    source = _frame()
    processed = source.normalize() + source.remove_dc()

    payload = RecipePlan.from_frame(processed, input_names=("signal",)).to_dict()

    assert payload["inputs"] == [{"id": "input-0", "name": "signal", "kind": "frame"}]
    assert [node["operation"] for node in payload["nodes"]] == [
        "wandas.audio.normalize",
        "wandas.audio.remove_dc",
        "wandas.operator.add",
    ]
    assert payload["nodes"][-1]["inputs"] == ["node-0", "node-1"]


def test_compiler_preserves_independent_frame_input_order() -> None:
    left = _frame(1.0)
    right = _frame(2.0)
    plan = RecipePlan.from_frame(left - right, input_names=("left", "right"))

    assert [item["name"] for item in plan.to_dict()["inputs"]] == ["left", "right"]
    replayed = plan.apply({"left": left, "right": right})
    np.testing.assert_allclose(replayed.compute(), -1.0)


@pytest.mark.parametrize(
    "operand",
    [
        np.arange(16.0),
        da.from_array(np.arange(16.0), chunks=4),
    ],
)
def test_compiler_models_numpy_and_dask_as_one_external_array_kind(operand: np.ndarray | DaArray) -> None:
    source = _frame()
    plan = RecipePlan.from_frame(source + operand, input_names=("signal", "operand"))
    payload = plan.to_dict()

    assert [item["kind"] for item in payload["inputs"]] == ["frame", "array"]
    assert "numpy" not in str(payload).lower()
    assert "dask" not in str(payload).lower()
    replayed = plan.apply({"signal": source, "operand": operand})
    assert isinstance(replayed._data, DaArray)


def test_compiler_preserves_omitted_arguments_as_omitted() -> None:
    plan = RecipePlan.from_frame(_frame().normalize())
    params = dict(plan.nodes[0].params.entries)

    assert params == {}


def test_compiler_rejects_custom_callable_without_registered_public_operation() -> None:
    source = _frame()
    processed = source.apply(lambda data: data)

    with pytest.raises(RecipeExtractionError, match="rejected a public operation"):
        RecipePlan.from_frame(processed)


def test_generic_apply_operation_entrypoint_is_absent() -> None:
    assert not hasattr(_frame(), "apply_operation")


@pytest.mark.parametrize(
    "query",
    [re.compile("left"), lambda channel: channel.label == "left"],
)
def test_compiler_rejects_nonportable_channel_queries(query: Any) -> None:
    selected = _frame().get_channel(query=query)

    with pytest.raises(RecipeExtractionError, match="not portable"):
        RecipePlan.from_frame(selected)


def test_extraction_and_serialization_do_not_compute_dask_graph() -> None:
    source = _frame()
    processed = source.normalize() + da.ones((2, 16), chunks=(1, 4))

    with patch.object(DaArray, "compute", autospec=True, side_effect=AssertionError("unexpected compute")):
        payload = RecipePlan.from_frame(processed).to_dict()

    assert payload["nodes"][-1]["operation"] == "wandas.operator.add"


def test_compiler_requires_one_name_per_runtime_input() -> None:
    processed = _frame() + _frame(2.0)

    with pytest.raises(RecipeExtractionError, match="one name per runtime input"):
        RecipePlan.from_frame(processed, input_names=("only-one",))
