from typing import cast

import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan
from wandas.pipeline.calls import BinaryCall, ExternalArrayCall, IndexCall, ScalarCall


def _frame(value: float = 1.0) -> ChannelFrame:
    return ChannelFrame.from_numpy(np.full((2, 32), value), sampling_rate=8000)


def test_linear_audio_recipe_replays() -> None:
    source = _frame()
    processed = source.normalize().remove_dc()
    plan = RecipePlan.from_frame(processed, input_names=("signal",))

    replayed = plan.apply({"signal": source})

    np.testing.assert_allclose(replayed.compute(), processed.compute())
    assert len(plan.inputs) == 1 and len(plan.nodes) == 2


def test_shared_dag_identity_is_preserved() -> None:
    source = _frame()
    branch = source.normalize()
    plan = RecipePlan.from_frame(branch + branch)

    assert len(plan.inputs) == 1
    assert len(plan.nodes) == 2
    assert isinstance(plan.nodes[-1].call, BinaryCall)
    assert plan.nodes[-1].inputs == (plan.nodes[0].id, plan.nodes[0].id)


def test_equal_but_distinct_branches_are_not_collapsed() -> None:
    source = _frame()
    plan = RecipePlan.from_frame(source.normalize() + source.normalize())

    assert len(plan.nodes) == 3


def test_scalar_and_reflected_scalar_preserve_order() -> None:
    source = _frame(2.0)
    direct = RecipePlan.from_frame(source - 3)
    reflected = RecipePlan.from_frame(3 - source)

    assert isinstance(direct.nodes[0].call, ScalarCall)
    assert direct.nodes[0].call.reverse is False
    assert cast(ScalarCall, reflected.nodes[0].call).reverse is True
    np.testing.assert_allclose(reflected.apply({"input_0": source}).compute(), (3 - source).compute())


def test_external_dask_array_is_named_input_and_stays_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _frame()
    operand = da.ones(source.shape, chunks=(1, 8))
    processed = source + operand

    def fail_compute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("compile/apply must not compute")

    monkeypatch.setattr(da.Array, "compute", fail_compute)
    plan = RecipePlan.from_frame(processed, input_names=("signal", "operand"))
    replayed = plan.apply({"signal": source, "operand": operand})

    assert [item.kind for item in plan.inputs] == ["frame", "array"]
    assert isinstance(plan.nodes[0].call, ExternalArrayCall)
    assert replayed.shape == source.shape


def test_raw_add_channel_retains_base_source_lineage() -> None:
    source = _frame()
    processed = source.add_channel(np.ones(source.n_samples), label="added")

    assert processed.lineage is not None and len(processed.lineage.inputs) == 1
    plan = RecipePlan.from_frame(processed, input_names=("signal", "added"))
    replayed = plan.apply({"signal": source, "added": np.ones(source.n_samples)})
    assert replayed.labels[-1] == "added"


def test_add_channel_reuses_shared_root_identity() -> None:
    source = _frame()
    processed = source.add_channel(source, label="same")

    assert processed.lineage is not None
    assert processed.lineage.inputs[0] is processed.lineage.inputs[1]
    assert len(RecipePlan.from_frame(processed).inputs) == 1


def test_multidimensional_indexing_is_one_call() -> None:
    source = _frame()
    processed = source[:, 2:10]
    plan = RecipePlan.from_frame(processed)

    assert len(plan.nodes) == 1
    assert isinstance(plan.nodes[0].call, IndexCall)
    replayed = plan.apply({"input_0": source})
    assert replayed.shape == processed.shape


def test_singleton_tuple_indexing_uses_the_same_canonical_selector() -> None:
    source = _frame()
    direct = RecipePlan.from_frame(source[0])
    singleton = RecipePlan.from_frame(source[(0,)])

    assert singleton.nodes[0].call.to_payload() == direct.nodes[0].call.to_payload()
    assert len(source[(0,)].operation_history) == 1


def test_input_name_count_is_validated() -> None:
    with pytest.raises(Exception, match="one name|too few"):
        RecipePlan.from_frame(_frame() + _frame(2), input_names=("only",))
