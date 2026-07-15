from __future__ import annotations

from typing import Any
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.processing.semantic import (
    FrozenMap,
    InputBinding,
    LineageNode,
    SemanticOperation,
    freeze_params,
    params_to_display,
    source_lineage,
    thaw_params,
    value_from_json,
    value_to_json,
)


def test_semantic_operation_rejects_invalid_versions_and_duplicate_roles() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SemanticOperation("tests.invalid", 0, (), FrozenMap(()))
    with pytest.raises(ValueError, match="unique"):
        SemanticOperation(
            "tests.duplicate-role",
            1,
            (InputBinding("signal", "frame"), InputBinding("signal", "array")),
            FrozenMap(()),
        )


def test_canonical_params_own_immutable_nested_snapshot() -> None:
    values = [1, 2]
    config: dict[str, Any] = {"values": values}
    frozen = freeze_params({"config": config})

    values.append(3)
    config["extra"] = True
    first = thaw_params(frozen)
    first["config"]["values"].append(4)

    assert thaw_params(frozen) == {"config": {"values": [1, 2]}}


def test_canonical_json_roundtrip_is_collision_proof() -> None:
    frozen = freeze_params({"value": {"$type": "number", "items": [1, 2]}})

    assert value_from_json(value_to_json(frozen)) == frozen


def test_canonical_sequences_preserve_list_and_tuple_kinds() -> None:
    frozen = freeze_params(
        {
            "list": [1, (2, 3)],
            "tuple": (4, [5, 6]),
        }
    )

    loaded = value_from_json(value_to_json(frozen))
    assert isinstance(loaded, FrozenMap)
    thawed = thaw_params(loaded)

    assert type(thawed["list"]) is list
    assert type(thawed["list"][1]) is tuple
    assert type(thawed["tuple"]) is tuple
    assert type(thawed["tuple"][1]) is list
    assert params_to_display(frozen) == {
        "list": [1, [2, 3]],
        "tuple": [4, [5, 6]],
    }


def test_canonical_tuple_rejects_malformed_items() -> None:
    with pytest.raises(ValueError, match="Canonical tuple fields are malformed"):
        value_from_json({"$type": "tuple"})


def test_lineage_requires_one_parent_per_declared_binding() -> None:
    operation = SemanticOperation(
        "tests.merge",
        1,
        (InputBinding("left", "frame"), InputBinding("right", "array")),
        FrozenMap(()),
    )
    source = source_lineage()

    assert LineageNode(operation, (source, None)).inputs == (source, None)
    with pytest.raises(TypeError, match="Frame binding"):
        LineageNode(operation, (None, None))
    with pytest.raises(TypeError, match="Array binding"):
        LineageNode(operation, (source, source))


def test_source_identity_is_shared_by_branches_from_same_frame() -> None:
    source = ChannelFrame.from_numpy(np.ones((1, 32)), sampling_rate=8000)
    left = source.normalize()
    right = source.remove_dc()

    assert left.lineage.inputs[0] is source.lineage
    assert right.lineage.inputs[0] is source.lineage


def test_canonical_capture_rejects_dask_array_without_computing() -> None:
    value = da.ones((1, 8), chunks=(1, 4))

    with patch.object(DaArray, "compute", autospec=True, side_effect=AssertionError("unexpected compute")):
        with pytest.raises(TypeError, match="Unsupported Recipe parameter"):
            freeze_params({"value": value})
