"""Shared fixtures for I/O tests.

Provides standard test signals and factory fixtures per the I/O Test Policy.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from wandas.frames.channel import ChannelFrame


@pytest.fixture
def known_signal_frame() -> ChannelFrame:
    """2-channel frame for round-trip verification.

    SR=44100 Hz, ch_labels=["left", "right"], ch_units=["Pa", "Pa"].
    Uses seeded random data (np.random.default_rng(42)) for reproducibility.
    """
    rng = np.random.default_rng(42)
    sr = 44100
    data = rng.standard_normal((2, sr)).astype(np.float64)
    return ChannelFrame.from_numpy(
        data,
        sampling_rate=sr,
        ch_labels=["left", "right"],
        ch_units=["Pa", "Pa"],
    )


@pytest.fixture
def create_test_wav(tmp_path: Path):
    """Factory fixture that creates int16 PCM WAV files.

    Args:
        sr: Sampling rate in Hz.
        n_channels: Number of channels (1 for mono, 2 for stereo).
        n_samples: Number of samples per channel.

    Returns:
        Path to the created WAV file.
    """

    def _factory(sr: int = 44100, n_channels: int = 2, n_samples: int = 44100) -> Path:
        rng = np.random.default_rng(0)
        filepath = tmp_path / f"test_{sr}_{n_channels}ch_{n_samples}s.wav"
        # Generate int16 PCM data in valid range
        raw = rng.integers(-16384, 16384, size=(n_samples, n_channels), dtype=np.int16)
        if n_channels == 1:
            # scipy.io.wavfile expects 1D array for mono
            raw = raw.squeeze(axis=1)
        wavfile.write(str(filepath), sr, raw)
        return filepath

    return _factory
