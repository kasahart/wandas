"""FFT/STFT analysis parameters remain immutable across public workflows."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import wandas as wd
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.pipeline import RecipePlan


def _analysis(spectrogram: bool) -> tuple[ChannelFrame, SpectralFrame | SpectrogramFrame]:
    source = ChannelFrame.from_numpy(np.arange(32.0), sampling_rate=8, metadata={"sensor": "left"})
    source = source.with_source_time_offset(2.5)
    if spectrogram:
        return source, source.stft(n_fft=8, hop_length=2, win_length=6, window="boxcar")
    return source, source.fft(n_fft=8, window="boxcar")


@pytest.mark.parametrize(
    ("spectrogram", "name", "value"),
    [
        (False, "n_fft", 10),
        (False, "window", "hann"),
        (True, "n_fft", 10),
        (True, "hop_length", 3),
        (True, "win_length", 8),
        (True, "window", "hann"),
    ],
)
def test_analysis_parameters_reject_assignment(spectrogram: bool, name: str, value: Any) -> None:
    _, frame = _analysis(spectrogram)
    state = frame._get_additional_init_kwargs()
    history = frame.operation_history
    with pytest.raises(AttributeError):
        setattr(frame, name, value)
    assert frame._get_additional_init_kwargs() == state
    assert frame.operation_history == history
    assert len(frame.freqs) == 5


@pytest.mark.parametrize("spectrogram", [False, True])
def test_analysis_state_survives_reconstruction_wdf_and_recipe(tmp_path: Path, spectrogram: bool) -> None:
    source, frame = _analysis(spectrogram)
    path = tmp_path / "analysis.wdf"
    frame.save(path)
    # The source includes an offset operation; replay starts from its original receiver.
    assert source.previous is not None
    plan = RecipePlan.from_dict(RecipePlan.from_frame(frame).to_dict())
    replayed = plan.apply({"input_0": source.previous})
    for result in [frame.with_label("annotated"), frame[0], frame * 1, frame.cache(), wd.load(path), replayed]:
        assert isinstance(result, (SpectralFrame, SpectrogramFrame))
        assert result._get_additional_init_kwargs() == frame._get_additional_init_kwargs()
        np.testing.assert_allclose(result.data, frame.data)
        np.testing.assert_array_equal(result.freqs, frame.freqs)
        np.testing.assert_array_equal(result.source_time_offset, frame.source_time_offset)
        assert result.metadata == frame.metadata
