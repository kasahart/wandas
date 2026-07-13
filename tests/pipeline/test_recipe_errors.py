import copy

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan, RecipeSerializationError


def _payload() -> dict:
    source = ChannelFrame.from_numpy(np.ones((1, 16)), sampling_rate=8000)
    return RecipePlan.from_frame(source.normalize()).to_dict()


def test_loader_rejects_duplicate_node_ids() -> None:
    payload = _payload()
    payload["nodes"].append(copy.deepcopy(payload["nodes"][0]))

    with pytest.raises(RecipeSerializationError, match="Invalid Recipe graph"):
        RecipePlan.from_dict(payload)


def test_loader_rejects_dead_node_injected_into_payload() -> None:
    payload = _payload()
    dead = copy.deepcopy(payload["nodes"][0])
    dead["id"] = "dead"
    payload["nodes"].append(dead)

    with pytest.raises(RecipeSerializationError, match="Invalid Recipe graph"):
        RecipePlan.from_dict(payload)


def test_loader_rejects_malformed_value_tree() -> None:
    payload = _payload()
    payload["nodes"][0]["call"]["params"] = ["mapping", [["x", ["unknown", 1]]]]

    with pytest.raises(RecipeSerializationError, match="Unknown Recipe value"):
        RecipePlan.from_dict(payload)
