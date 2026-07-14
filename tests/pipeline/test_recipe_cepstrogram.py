"""Recipe round-trip coverage for the public cepstrogram workflow."""

from unittest.mock import patch

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.frames.cepstrogram import CepstrogramFrame
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.pipeline import RecipePlan

_SAMPLING_RATE = 8_000


def _source(sample_count: int, *, offset: float = 0.0) -> ChannelFrame:
    time = np.arange(sample_count, dtype=float) / _SAMPLING_RATE
    data = (np.sin(2 * np.pi * 500 * time) + 0.25 * np.cos(2 * np.pi * 1_000 * time))[None, :]
    return ChannelFrame(
        data=da.from_array(data, chunks=(1, -1)),
        sampling_rate=_SAMPLING_RATE,
        label="recipe-source",
        metadata={"recording": "speech"},
        source_time_offset=offset,
    )


def _workflow(source: ChannelFrame) -> SpectrogramFrame:
    return (
        source.stft(n_fft=16, hop_length=4, win_length=16, window="boxcar")
        .cepstrum(floor=1e-9)
        .lifter(cutoff=2 / _SAMPLING_RATE)
        .to_spectral_envelope()
    )


def test_cepstrogram_workflow_serializes_and_replays_without_compute() -> None:
    source = _source(64, offset=0.25)
    replay_source = _source(96, offset=1.5)

    with patch.object(DaArray, "compute", autospec=True, side_effect=AssertionError("unexpected compute")):
        processed = _workflow(source)
        plan = RecipePlan.from_frame(processed, input_names=("signal",))
        payload = plan.to_dict()
        loaded = RecipePlan.from_dict(payload)
        replayed = loaded.apply({"signal": replay_source})
        expected = _workflow(replay_source)

    assert loaded.to_dict() == payload
    assert isinstance(replayed, SpectrogramFrame)
    assert isinstance(replayed._data, da.Array)
    assert replayed.metadata == replay_source.metadata
    np.testing.assert_array_equal(replayed.source_time_offset, replay_source.source_time_offset)
    np.testing.assert_allclose(replayed.compute(), expected.compute(), rtol=1e-12, atol=1e-12)
    assert [entry["operation"] for entry in replayed.operation_history[-3:]] == [
        "wandas.spectrogram.cepstrum",
        "wandas.cepstrogram.lifter",
        "wandas.cepstrogram.to_spectral_envelope",
    ]


def test_recipe_intermediate_output_keeps_cepstrogram_frame_type() -> None:
    source = _source(64)
    processed = source.stft(n_fft=16, hop_length=4, window="boxcar").cepstrum()

    replayed = RecipePlan.from_dict(RecipePlan.from_frame(processed, input_names=("signal",)).to_dict()).apply(
        {"signal": source}
    )

    assert isinstance(replayed, CepstrogramFrame)
    assert replayed.n_fft == 16
    assert replayed.hop_length == 4
    np.testing.assert_array_equal(replayed.quefrencies, np.arange(16) / _SAMPLING_RATE)
