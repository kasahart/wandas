"""SpectrogramFrame.csd() reuses an STFT for the cross-spectral matrix."""

from __future__ import annotations

import json
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy import signal as ss

import wandas as wd
from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame
from wandas.frames.pairwise import CrossSpectralFrame, TransferFunctionFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.pipeline import RecipePlan

_FS = 8000.0


def _signal(n_samples: int = 4096) -> ChannelFrame:
    rng = np.random.default_rng(0)
    values = rng.standard_normal((2, n_samples))
    values[1] = np.convolve(values[0], [0.5, 0.3, -0.2], "same") + 0.1 * values[1]
    return wd.from_numpy(values, sampling_rate=_FS, ch_labels=["mic", "acc"], ch_units=["Pa", "m/s^2"])


def _interior_frames(spec: SpectrogramFrame, n_samples: int) -> tuple[int, int, int, int]:
    """Return the frame range whose segments lie inside the signal, and its sample span."""
    assert spec.frame_time_origin is not None
    p_min = round(spec.frame_time_origin * spec.sampling_rate / spec.hop_length)
    starts = (np.arange(spec.n_frames) + p_min) * spec.hop_length - spec.win_length // 2
    keep = np.flatnonzero((starts >= 0) & (starts + spec.win_length <= n_samples))
    first, last = int(keep[0]), int(keep[-1]) + 1
    return first, last, int(starts[first]), int(starts[last - 1]) + spec.win_length


@pytest.mark.parametrize(
    ("n_fft", "win_length", "hop_length"),
    [(256, 256, 64), (256, 256, 128), (255, 255, 127), (256, 200, 50)],
)
@pytest.mark.parametrize("scaling", ["spectrum", "density"])
def test_spectrogram_csd_matches_scipy_csd_over_interior_frames(
    n_fft: int, win_length: int, hop_length: int, scaling: str
) -> None:
    source = _signal()
    values = channel_first_values(source)
    spec = source.stft(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    first, last, lo, hi = _interior_frames(spec, values.shape[-1])

    result = spec[:, :, first:last].csd(scaling=scaling)

    _, expected = ss.csd(
        values[np.newaxis, :, lo:hi],
        values[:, np.newaxis, lo:hi],
        fs=_FS,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        detrend=False,
        scaling=scaling,
    )
    np.testing.assert_allclose(channel_first_values(result), expected.reshape(4, -1), rtol=1e-10, atol=1e-14)


def test_spectrogram_csd_is_lazy_and_matches_channel_frame_csd_state() -> None:
    source = _signal()
    spec = source.stft(n_fft=256, hop_length=64)
    first, last, lo, hi = _interior_frames(spec, source.n_samples)
    interior = spec[:, :, first:last]

    with patch.object(DaArray, "compute", autospec=True) as compute:
        result = interior.csd(scaling="density")
        compute.assert_not_called()

    direct = source[:, lo:hi].csd(n_fft=256, hop_length=64, detrend=False, scaling="density")
    assert type(result) is CrossSpectralFrame
    assert isinstance(result._data, DaArray)
    np.testing.assert_allclose(channel_first_values(result), channel_first_values(direct), rtol=1e-10, atol=1e-14)
    assert result.scaling == "density"
    assert result.pair_state == direct.pair_state
    assert result._channel_metadata == direct._channel_metadata
    assert result.labels == direct.labels
    np.testing.assert_allclose(result.source_time_offset, direct.source_time_offset, atol=1e-12)
    assert result.n_fft == 256
    assert result.window == "hann"
    assert result.previous is interior
    assert result.operation_history[-1]["operation"] == "wandas.spectrogram.csd"


def test_spectrogram_csd_feeds_transfer_function_and_coherence() -> None:
    source = _signal()
    spec = source.stft(n_fft=256, hop_length=128)
    first, last, lo, hi = _interior_frames(spec, source.n_samples)
    csd = spec[:, :, first:last].csd()
    segment = source[:, lo:hi]

    transfer = csd.transfer_function().select_pair(1, 0)
    coherence = csd.coherence().select_pair(1, 0)

    direct_transfer = segment.transfer_function(n_fft=256, hop_length=128, detrend=False).select_pair(1, 0)
    direct_coherence = segment.coherence(n_fft=256, hop_length=128, detrend=False).select_pair(1, 0)
    assert type(transfer) is TransferFunctionFrame
    np.testing.assert_allclose(transfer.data, direct_transfer.data, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(coherence.data, direct_coherence.data, rtol=1e-10, atol=1e-14)


def test_spectrogram_csd_rejects_invalid_scaling() -> None:
    spec = _signal(1024).stft(n_fft=128, hop_length=32)
    with pytest.raises(ValueError, match="Invalid scaling for spectrogram CSD"):
        spec.csd(scaling="power")  # ty: ignore[invalid-argument-type]


def test_spectrogram_csd_rejects_magnitude_spectrogram() -> None:
    spec = _signal(1024).stft(n_fft=128, hop_length=32).abs()
    with pytest.raises(ValueError, match="complex STFT values with phase"):
        spec.csd()


def test_spectrogram_csd_rejects_empty_frame_range() -> None:
    spec = _signal(1024).stft(n_fft=128, hop_length=32)
    with pytest.raises(ValueError, match="at least one time frame"):
        spec[:, :, 3:3].csd()


def test_spectrogram_csd_recipe_roundtrip_replays_on_new_source() -> None:
    source = _signal()
    selected = source.stft(n_fft=256, hop_length=64).csd(scaling="density").transfer_function().select_pair(1, 0)

    payload = RecipePlan.from_frame(selected, input_names=("signal",)).to_dict()
    assert [node["operation"] for node in payload["nodes"]] == [
        "wandas.audio.stft",
        "wandas.spectrogram.csd",
        "wandas.cross_spectral.transfer_function",
        "wandas.frame.select_pair",
    ]

    replayed = RecipePlan.from_dict(json.loads(json.dumps(payload))).apply({"signal": _signal()})
    assert type(replayed) is TransferFunctionFrame
    np.testing.assert_allclose(replayed.data, selected.data, rtol=1e-12, atol=1e-14)


def test_spectrogram_csd_keeps_offset_when_frame_origin_is_unknown() -> None:
    spec = _signal(1024).stft(n_fft=128, hop_length=32)
    unknown = SpectrogramFrame(
        data=spec._data,
        sampling_rate=spec.sampling_rate,
        n_fft=spec.n_fft,
        hop_length=spec.hop_length,
        source_time_offset=0.5,
    )

    np.testing.assert_array_equal(unknown.csd().source_time_offset, np.full(4, 0.5))


def test_spectrogram_csd_operation_rejects_incomplete_frequency_axis() -> None:
    from wandas.processing import SpectrogramCSD

    operation = SpectrogramCSD(_FS, n_fft=128)
    with pytest.raises(ValueError, match=r"Expected: \(channel, 65, time\)"):
        operation.calculate_output_shape((2, 64, 5))


def test_spectrogram_csd_accepts_short_windows() -> None:
    spec = _signal(64).stft(n_fft=4, win_length=2, hop_length=1)

    assert spec.csd().shape == (4, 3)


@pytest.mark.parametrize("detrend", ["constant", "linear"])
def test_detrended_stft_csd_matches_scipy_csd_with_the_same_detrend(detrend: Literal["constant", "linear"]) -> None:
    source = _signal()
    values = channel_first_values(source) + np.linspace(1.0, 4.0, source.n_samples)
    source = wd.from_numpy(values, sampling_rate=_FS)
    spec = source.stft(n_fft=256, hop_length=64, detrend=detrend)
    first, last, lo, hi = _interior_frames(spec, values.shape[-1])

    result = spec[:, :, first:last].csd(scaling="density")

    _, expected = ss.csd(
        values[np.newaxis, :, lo:hi],
        values[:, np.newaxis, lo:hi],
        fs=_FS,
        nperseg=256,
        noverlap=192,
        detrend=detrend,
        scaling="density",
    )
    np.testing.assert_allclose(channel_first_values(result), expected.reshape(4, -1), rtol=1e-10, atol=1e-14)
