"""Explicit test inventory for concrete public built-in Frame families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dask.array as da
import numpy as np

from tests.pairwise_test_helpers import make_pairwise_source
from wandas.core.base_frame import BaseFrame
from wandas.frames.cepstral import CepstralFrame
from wandas.frames.cepstrogram import CepstrogramFrame
from wandas.frames.channel import ChannelFrame
from wandas.frames.noct import NOctFrame
from wandas.frames.pairwise import CoherenceFrame, CrossSpectralFrame, TransferFunctionFrame
from wandas.frames.roughness import RoughnessFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame


@dataclass(frozen=True)
class BuiltinFrameCase:
    """One concrete Frame type and an independent representative factory."""

    frame_type: type[BaseFrame[Any]]
    factory: Callable[[], BaseFrame[Any]]

    @property
    def id(self) -> str:
        """Return the stable pytest parameter identifier."""
        return self.frame_type.__name__


def _channel_frame() -> ChannelFrame:
    return ChannelFrame.from_numpy(
        np.arange(16.0, dtype=np.float32).reshape(2, 8),
        sampling_rate=8.0,
        label="source",
        metadata={"recording": "fixture"},
        ch_labels=["left", "right"],
        ch_units=["Pa", "Pa"],
    )


def _cepstrogram_frame() -> CepstrogramFrame:
    return CepstrogramFrame(
        da.from_array(np.arange(48.0).reshape(2, 8, 3), chunks=(1, -1, -1)),
        sampling_rate=8.0,
        n_fft=8,
        hop_length=2,
        win_length=8,
        window="hann",
        channel_metadata=[{"label": "left", "unit": "Pa"}, {"label": "right", "unit": "Pa"}],
    )


def _noct_frame() -> NOctFrame:
    return NOctFrame(
        da.arange(8.0, chunks=8).reshape(2, 4),
        sampling_rate=8.0,
        fmin=1.0,
        fmax=4.0,
        channel_metadata=[{"label": "left"}, {"label": "right"}],
    )


def _roughness_frame() -> RoughnessFrame:
    return RoughnessFrame(
        da.arange(282.0, chunks=282).reshape(2, 47, 3),
        sampling_rate=8.0,
        bark_axis=np.linspace(0.5, 23.5, 47),
        overlap=0.5,
        channel_metadata=[{"label": "left"}, {"label": "right"}],
    )


def _coherence_frame() -> CoherenceFrame:
    return make_pairwise_source(n_channels=2).coherence(
        n_fft=8,
        win_length=8,
        hop_length=4,
        window="boxcar",
    )


def _cross_spectral_frame() -> CrossSpectralFrame:
    return make_pairwise_source(n_channels=2).csd(
        n_fft=8,
        win_length=8,
        hop_length=4,
        window="boxcar",
        scaling="density",
    )


def _transfer_function_frame() -> TransferFunctionFrame:
    return make_pairwise_source(n_channels=2).transfer_function(
        n_fft=8,
        win_length=8,
        hop_length=4,
        window="boxcar",
        scaling="spectrum",
    )


BUILTIN_FRAME_CASES = (
    BuiltinFrameCase(ChannelFrame, _channel_frame),
    BuiltinFrameCase(CepstralFrame, lambda: _channel_frame().cepstrum(n_fft=8)),
    BuiltinFrameCase(CepstrogramFrame, _cepstrogram_frame),
    BuiltinFrameCase(SpectralFrame, lambda: _channel_frame().fft(n_fft=8)),
    BuiltinFrameCase(SpectrogramFrame, lambda: _channel_frame().stft(n_fft=8, hop_length=2)),
    BuiltinFrameCase(CoherenceFrame, _coherence_frame),
    BuiltinFrameCase(CrossSpectralFrame, _cross_spectral_frame),
    BuiltinFrameCase(TransferFunctionFrame, _transfer_function_frame),
    BuiltinFrameCase(NOctFrame, _noct_frame),
    BuiltinFrameCase(RoughnessFrame, _roughness_frame),
)
