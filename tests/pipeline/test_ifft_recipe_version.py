"""Recipe compatibility for versioned IFFT amplitude scaling."""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.signal import get_window

from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame
from wandas.pipeline import RecipePlan


def _replay_with_ifft_version(frame: Any, source: ChannelFrame, version: int) -> Any:
    payload = RecipePlan.from_frame(frame, input_names=("signal",)).to_dict()
    for node in payload["nodes"]:
        if node["operation"] == "wandas.spectral.ifft":
            node["version"] = version
    return RecipePlan.from_dict(payload).apply({"signal": source})


def _released_ifft(spectrum: SpectralFrame) -> NDArray[Any]:
    """Released v2 IFFT: undo the peak-amplitude scaling with the n_fft window gain."""
    n_fft = spectrum.n_fft
    gain = np.sum(get_window(spectrum.window, n_fft))
    one_sided = np.asarray(spectrum.data, dtype=np.complex128).reshape(1, -1).copy()
    one_sided[..., 1:-1] /= 2
    return np.fft.irfft(one_sided * gain, n=n_fft, axis=-1)


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
