"""Shared fixtures for visualization tests.

Sets the non-interactive Agg backend before any matplotlib import
to prevent GUI window creation during CI/headless runs.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import wandas as wd

# ---------------------------------------------------------------------------
# Standard deterministic signals (known analytical solutions)
# ---------------------------------------------------------------------------

# Sampling rate common to all visualization fixtures
_SAMPLING_RATE = 16_000
# Duration kept short — visual tests only need structure, not long signals
_DURATION = 0.05  # 50 ms
# Pure-tone frequency with analytically predictable FFT peak
_FREQ_HZ = 1_000


@pytest.fixture
def channel_frame() -> wd.ChannelFrame:
    """Standard mono ChannelFrame for plot tests.

    Uses a 1 kHz pure sine at 16 kHz sampling rate so that the FFT peak
    position is analytically predictable (bin index = freq * N / sr).
    """
    return wd.generate_sin(
        freqs=[_FREQ_HZ],
        duration=_DURATION,
        sampling_rate=_SAMPLING_RATE,
    )


@pytest.fixture
def stereo_frame() -> wd.ChannelFrame:
    """Multi-channel (stereo) ChannelFrame for plot tests.

    Channel 0: 1 kHz sine (amplitude 1.0)
    Channel 1: 1 kHz sine (amplitude 0.5)
    Both are deterministic and analytically predictable.
    """
    n_samples = int(_SAMPLING_RATE * _DURATION)
    t = np.linspace(0, _DURATION, n_samples, endpoint=False)
    ch0 = np.sin(2 * np.pi * _FREQ_HZ * t)
    ch1 = 0.5 * np.sin(2 * np.pi * _FREQ_HZ * t)
    data = np.stack([ch0, ch1], axis=0)
    return wd.ChannelFrame.from_numpy(
        data,
        sampling_rate=_SAMPLING_RATE,
        ch_labels=["ch0", "ch1"],
    )


# ---------------------------------------------------------------------------
# Automatic figure cleanup — prevents memory leaks (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Clear and close all matplotlib figures after every test.

    Uses fig.clf() before plt.close() to fully release internal state,
    as plt.close("all") alone is insufficient for preventing memory leaks.
    """
    yield
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.clf()
    plt.close("all")
