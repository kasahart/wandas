import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wandas as wd


def test_waveform_plot_uses_provided_ax() -> None:
    """When an Axes is passed to ChannelFrame.plot(), it must draw into it."""
    np.random.seed(0)
    signal = wd.generate_sin(freqs=[440], duration=0.05, sampling_rate=8000)

    fig, ax = plt.subplots()
    out = signal.plot(ax=ax)

    # Should return the same Axes object
    assert out is ax

    # Axes should contain at least one Line2D after plotting
    assert len(ax.lines) >= 1


def test_frequency_plot_uses_provided_ax() -> None:
    """When an Axes is passed to SpectralFrame.plot(), it must draw into it."""
    np.random.seed(1)
    signal = wd.generate_sin(freqs=[440], duration=0.05, sampling_rate=8000)
    spec = signal.fft()

    fig, ax = plt.subplots()
    out = spec.plot(ax=ax)

    # Should return the same Axes object
    assert out is ax

    # Frequency plot should produce at least one line
    assert len(ax.lines) >= 1
