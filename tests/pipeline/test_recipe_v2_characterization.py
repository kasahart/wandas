"""Breaking baseline for the Recipe v2 implementation.

Tests in this module deliberately separate behavior that v2 preserves from semantic
contracts that v2 changes. The latter assertions are updated in the semantic-lineage
commit, making the intended break reviewable rather than accidental.
"""

import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan
from wandas.processing.base import IndexOperation


def _frame() -> ChannelFrame:
    frame = ChannelFrame.from_numpy(
        np.arange(64.0).reshape(2, 32),
        sampling_rate=8000,
        ch_labels=["left", "right"],
    )
    frame.metadata["contract"] = "preserved"
    frame.source_time_offset = [0.25, 0.5]
    return frame


def test_multidimensional_indexing_preserves_frame_semantics() -> None:
    source = _frame()
    result = source[:, 2:10]

    assert type(result) is type(source)
    assert result.shape == (2, 8)
    assert result.labels == source.labels
    assert result.sampling_rate == source.sampling_rate
    assert result.metadata == source.metadata
    np.testing.assert_allclose(result.source_time_offset, [0.25025, 0.50025])


def test_multidimensional_indexing_is_one_semantic_operation() -> None:
    result = _frame()[:, 2:10]

    assert [record["operation"] for record in result.operation_history] == ["__getitem__"]


@pytest.mark.parametrize(
    "key",
    [
        0,
        "left",
        slice(None),
        [0, 1],
        ["left", "right"],
        np.array([0, 1]),
        np.array([True, False]),
        (slice(None), slice(2, 10)),
    ],
    ids=("integer", "label", "slice", "integer-list", "label-list", "integer-array", "mask", "multidim"),
)
def test_each_public_indexing_form_owns_one_lineage_node(key: object) -> None:
    source = _frame()

    result = source[key]  # ty: ignore[invalid-argument-type]
    plan = RecipePlan.from_frame(result)

    assert result.lineage is not None
    assert isinstance(result.lineage.operation, IndexOperation)
    assert result.lineage.inputs == (source._lineage_or_source(),)
    assert [record["operation"] for record in result.operation_history] == ["__getitem__"]
    assert len(plan.nodes) == 1
    assert result.metadata == source.metadata
    assert len(result.source_time_offset) == result.n_channels


def test_empty_integer_array_preserves_array_intent() -> None:
    result = _frame()[np.array([], dtype=int)]

    assert result.operation_history[-1]["params"]["indexing"] == "integer_array"


def test_shared_branch_identity_is_reported_once() -> None:
    source = _frame()
    branch = source.normalize()
    result = branch + branch

    assert result.lineage is not None
    assert result.lineage.inputs[0] is result.lineage.inputs[1]
    assert [record["operation"] for record in result.operation_history] == ["normalize", "+"]


def test_existing_graph_replay_preserves_non_commutative_order() -> None:
    left = _frame()
    right = ChannelFrame.from_numpy(np.ones((2, 32)), sampling_rate=8000)
    processed = left - right
    recipe = RecipePlan.from_frame(processed, input_names=("left", "right"))
    replayed = recipe.apply({"left": left, "right": right})

    np.testing.assert_allclose(replayed._data.compute(), processed._data.compute())


def test_external_dask_operand_is_lazy_during_recipe_extraction(monkeypatch) -> None:
    source = _frame()
    processed = source + da.ones(source.shape, chunks=(1, 8))

    def fail_compute(*_args, **_kwargs):
        raise AssertionError("Recipe extraction must not compute Dask inputs")

    monkeypatch.setattr(da.Array, "compute", fail_compute)
    recipe = RecipePlan.from_frame(processed, input_names=("signal", "operand"))

    assert len(recipe.inputs) == 2
