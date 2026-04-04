"""Tests for Axes reuse behavior — provided ax must be drawn into directly.

Visualization policy: Axes Return Type Tests — verify that when an external
Axes is passed to plot(), it is used as‑is instead of creating a new figure.
"""

import matplotlib.pyplot as plt

import wandas as wd

# ---------------------------------------------------------------------------
# Deterministic signal shared by both tests
# ---------------------------------------------------------------------------
_FREQ_HZ = 440  # Pure‑tone frequency — analytically predictable FFT peak
_DURATION = 0.05  # 50 ms — short signal sufficient for plot structure
_SAMPLING_RATE = 8_000


def _make_signal() -> wd.ChannelFrame:
    """Return a deterministic single‑channel sine ChannelFrame."""
    return wd.generate_sin(
        freqs=[_FREQ_HZ],
        duration=_DURATION,
        sampling_rate=_SAMPLING_RATE,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_waveform_plot_draws_into_provided_ax() -> None:
    """ChannelFrame.plot(ax=ax) must return the *same* Axes and draw ≥1 line."""
    signal = _make_signal()
    fig, ax = plt.subplots()

    out = signal.plot(ax=ax)

    assert out is ax, "plot() should return the same Axes it was given"
    assert len(ax.lines) >= 1, "At least one Line2D should be drawn"


def test_frequency_plot_draws_into_provided_ax() -> None:
    """SpectralFrame.plot(ax=ax) must return the *same* Axes and draw ≥1 line."""
    signal = _make_signal()
    spec = signal.fft()

    fig, ax = plt.subplots()
    out = spec.plot(ax=ax)

    assert out is ax, "plot() should return the same Axes it was given"
    assert len(ax.lines) >= 1, "At least one Line2D should be drawn"
