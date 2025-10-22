"""Fixtures for benchmark tests."""

import pytest

import wandas as wd
from wandas.frames.channel import ChannelFrame


@pytest.fixture
def benchmark_signal() -> ChannelFrame:
    """Generate a benchmark signal for performance testing.

    Returns a 1-second, 2-channel signal at 44100 Hz sampling rate
    containing 440 Hz and 880 Hz sine waves.

    Returns:
        ChannelFrame: A sample audio signal for benchmarking.
    """
    return wd.generate_sin(freqs=[440, 880], duration=1.0, sampling_rate=44100)
