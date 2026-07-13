"""Breaking baseline for the Recipe v2 implementation.

Tests in this module deliberately separate behavior that v2 preserves from semantic
contracts that v2 changes. The latter assertions are updated in the semantic-lineage
commit, making the intended break reviewable rather than accidental.
"""

import dask.array as da
import numpy as np

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import NodeGraphRecipeSpec


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
    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("left", "right"))
    replayed = recipe.apply({"left": left, "right": right})

    np.testing.assert_allclose(replayed._data.compute(), processed._data.compute())


def test_external_dask_operand_is_lazy_during_recipe_extraction(monkeypatch) -> None:
    source = _frame()
    processed = source + da.ones(source.shape, chunks=(1, 8))

    def fail_compute(*_args, **_kwargs):
        raise AssertionError("Recipe extraction must not compute Dask inputs")

    monkeypatch.setattr(da.Array, "compute", fail_compute)
    recipe = NodeGraphRecipeSpec.from_frame(processed, input_names=("signal", "operand"))

    assert len(recipe.inputs) == 2
