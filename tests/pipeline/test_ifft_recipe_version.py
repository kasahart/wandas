"""Recipe compatibility for versioned IFFT amplitude scaling."""

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from tests.frame_helpers import channel_first_values
from wandas.frames.spectral import SpectralFrame
from wandas.pipeline import RecipePlan


def test_released_ifft_recipe_v1_replays_legacy_amplitude_scaling() -> None:
    spectrum = SpectralFrame(
        da.from_array(np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.complex128), chunks=(1, 5)),
        sampling_rate=8,
        n_fft=8,
        window="boxcar",
    )
    released_payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": "wandas.spectral.ifft",
                "version": 1,
                "inputs": ["input-0"],
                "params": {"$type": "map", "entries": []},
            }
        ],
        "output": "node-0",
    }

    plan = RecipePlan.from_dict(released_payload)
    replayed = plan.apply({"signal": spectrum})

    assert isinstance(replayed._data, DaArray)
    np.testing.assert_allclose(channel_first_values(replayed), np.full((1, 8), 1.0 / 8.0))
    assert replayed.operation_history == [{"operation": "wandas.spectral.ifft", "version": 1, "params": {}}]
