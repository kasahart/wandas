import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.processing.base import FrameMethodOperation, FrameSourceOperation, IndexOperation, LineageNode
from wandas.processing.semantic import InputBinding, OperationContract, frozen_params


def test_runtime_contract_rejects_invalid_versions_and_duplicate_roles() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        OperationContract("normalize", 0, True, ())
    with pytest.raises(ValueError, match="unique"):
        OperationContract(
            "merge",
            1,
            True,
            (InputBinding("signal", "frame"), InputBinding("signal", "frame")),
        )


def test_lineage_owns_an_immutable_descriptor_snapshot() -> None:
    values = [1, 2]
    operation = FrameMethodOperation("probe", {"values": values})
    lineage = LineageNode(operation, (LineageNode(FrameSourceOperation()),))

    values.append(3)
    returned_values = operation.method_params["values"]
    returned_values.append(4)

    assert operation.to_params() == {"values": [1, 2]}
    assert lineage.summary == {"operation": "probe", "params": {"values": [1, 2]}}
    assert lineage.replay.thaw_params() == {"values": [1, 2]}


def test_index_operation_exposes_only_defensive_parameter_values() -> None:
    operation = IndexOperation(
        {
            "indexing": "multidimensional_slice",
            "channel": {"type": "integer", "index": 0},
            "axis_slices": ({"start": 2, "stop": 6, "step": None},),
        }
    )
    source_lineage = LineageNode(FrameSourceOperation())
    lineage = LineageNode(operation, (source_lineage,))

    returned_channel_selector = operation.params["channel"]
    returned_axis_slices = operation.params["axis_slices"]
    returned_channel_selector["index"] = 9
    returned_axis_slices[0]["start"] = 99

    assert operation.to_params()["channel"]["index"] == 0
    assert operation.to_params()["axis_slices"][0]["start"] == 2
    assert lineage.replay.thaw_params()["channel"]["index"] == 0
    assert lineage.inputs == (source_lineage,)


def test_replay_params_return_defensive_values() -> None:
    descriptor = FrameMethodOperation("probe", {"nested": {"values": [1]}}).replay_descriptor()
    first = descriptor.thaw_params()
    first["nested"]["values"].append(2)

    assert descriptor.thaw_params() == {"nested": {"values": [1]}}


def test_source_identity_is_shared_by_branches_from_the_same_frame() -> None:
    source = ChannelFrame.from_numpy(np.ones((1, 32)), sampling_rate=8000)
    left = source.normalize()
    right = source.remove_dc()

    assert left.lineage is not None and right.lineage is not None
    assert left.lineage.inputs[0] is right.lineage.inputs[0]


def test_descriptor_capture_does_not_compute_dask_values(monkeypatch: pytest.MonkeyPatch) -> None:
    value = da.ones((1, 8), chunks=(1, 4))

    def fail_compute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("semantic capture must remain lazy")

    monkeypatch.setattr(da.Array, "compute", fail_compute)

    assert frozen_params({"value": value})[0] == "mapping"
