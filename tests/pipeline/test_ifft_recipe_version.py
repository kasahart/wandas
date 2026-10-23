"""Recipe compatibility for versioned IFFT amplitude scaling."""

from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from numpy.typing import NDArray

from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame
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


def _replay_with_ifft_version(frame: Any, source: ChannelFrame, version: int) -> Any:
    payload = RecipePlan.from_frame(frame, input_names=("signal",)).to_dict()
    for node in payload["nodes"]:
        if node["operation"] == "wandas.spectral.ifft":
            node["version"] = version
    return RecipePlan.from_dict(payload).apply({"signal": source})


def _released_ifft(spectrum: SpectralFrame) -> NDArray[Any]:
    """Released v2 IFFT: the frame without win_length uses the n_fft window gain."""
    full_window = SpectralFrame(
        spectrum._data,
        sampling_rate=spectrum.sampling_rate,
        n_fft=spectrum.n_fft,
        window=spectrum.window,
    )
    return channel_first_values(full_window.ifft())


@pytest.mark.parametrize("source_kind", ["welch", "stft-frame"])
def test_released_ifft_recipe_v2_ignores_short_analysis_windows(source_kind: str) -> None:
    rng = np.random.default_rng(7)
    source = ChannelFrame.from_numpy(rng.standard_normal((1, 4096)), 8000.0)
    if source_kind == "welch":
        spectrum = source.welch(n_fft=1024, win_length=256)
    else:
        spectrum = source.stft(n_fft=1024, win_length=256).get_frame_at(2)
    assert spectrum.win_length == 256

    replayed = _replay_with_ifft_version(spectrum.ifft(), source, version=2)

    np.testing.assert_allclose(channel_first_values(replayed), _released_ifft(spectrum))
    assert replayed.operation_history[-1]["version"] == 2
