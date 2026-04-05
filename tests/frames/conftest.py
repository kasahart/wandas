"""Standard fixtures for frame tests.

Provides deterministic, analytically predictable signals as required
by the Wandas Test Grand Policy (Pillar 4: Numerical Validity).
"""

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame

# ---------------------------------------------------------------------------
# Constants – shared across fixtures so tests can reference them directly.
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16000
"""Default sample rate (Hz) for frame-level test fixtures."""

DURATION: float = 1.0
"""Duration (seconds) of the standard test signals."""

N_SAMPLES: int = int(SAMPLE_RATE * DURATION)
"""Number of samples in the standard test signals."""


# ---------------------------------------------------------------------------
# Deterministic signal helpers
# ---------------------------------------------------------------------------


def _sine(freq: float, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """Generate a 1-D sine wave with known analytical properties."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# ChannelFrame fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def channel_frame() -> ChannelFrame:
    """Standard 2-channel frame with deterministic 440 Hz + 1000 Hz sinusoids.

    Channel 0: 440 Hz sine  (A4 concert pitch)
    Channel 1: 1000 Hz sine (commonly used reference tone)

    Both signals have amplitude 1.0 and are analytically predictable.
    """
    data = np.stack([_sine(440.0), _sine(1000.0)])  # shape (2, N_SAMPLES)
    return ChannelFrame.from_numpy(data, sampling_rate=SAMPLE_RATE)


@pytest.fixture
def mono_frame() -> ChannelFrame:
    """Single-channel frame with a deterministic 440 Hz sinusoid."""
    data = _sine(440.0).reshape(1, -1)
    return ChannelFrame.from_numpy(data, sampling_rate=SAMPLE_RATE)


@pytest.fixture
def composite_frame() -> ChannelFrame:
    """Composite-tone: 100 Hz + 500 Hz + 1500 Hz for filter tests."""
    data = np.stack(
        [
            _sine(100.0) + _sine(500.0) + _sine(1500.0),
            _sine(200.0) + _sine(1000.0),
        ]
    )
    return ChannelFrame.from_numpy(data, sampling_rate=SAMPLE_RATE)


@pytest.fixture
def impulse_frame() -> ChannelFrame:
    """Unit impulse for filter impulse response tests."""
    data = np.zeros((1, N_SAMPLES))
    data[0, 0] = 1.0
    return ChannelFrame.from_numpy(data, sampling_rate=SAMPLE_RATE)
