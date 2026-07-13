import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.processing.base import FrameMethodOperation, FrameSourceOperation, LineageNode
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
    operation.method_params["values"].append(4)

    assert lineage.replay.thaw_params() == {"values": [1, 2]}


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
