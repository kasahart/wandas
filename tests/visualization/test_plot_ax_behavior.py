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


def test_waveform_plot_creates_new_ax_when_not_provided() -> None:
    """When no Axes is provided, plot() should create a new figure and axes."""
    np.random.seed(2)
    signal = wd.generate_sin(freqs=[440], duration=0.05, sampling_rate=8000)

    ax = signal.plot()

    # Should return an Axes object
    assert ax is not None
    assert isinstance(ax, plt.Axes)

    # Should contain plotted data
    assert len(ax.lines) >= 1


def test_frequency_plot_creates_new_ax_when_not_provided() -> None:
    """When no Axes is provided to SpectralFrame.plot(), it should create new figure."""
    np.random.seed(3)
    signal = wd.generate_sin(freqs=[440], duration=0.05, sampling_rate=8000)
    spec = signal.fft()

    ax = spec.plot()

    # Should return an Axes object
    assert ax is not None
    assert isinstance(ax, plt.Axes)

    # Should contain plotted data
    assert len(ax.lines) >= 1


def test_multichannel_plot_uses_provided_ax() -> None:
    """Multi-channel ChannelFrame.plot() should use provided axes."""
    np.random.seed(4)
    signal1 = wd.generate_sin(freqs=[440], duration=0.05, sampling_rate=8000)
    signal2 = wd.generate_sin(freqs=[880], duration=0.05, sampling_rate=8000)
    multichannel = wd.ChannelFrame.concatenate([signal1, signal2], axis=0)

    fig, ax = plt.subplots()
    out = multichannel.plot(ax=ax)

    # Should return the same Axes object
    assert out is ax

    # Should have multiple lines (one per channel)
    assert len(ax.lines) >= 2


def test_plot_with_overlay_parameter() -> None:
    """Test plotting with overlay=True parameter."""
    np.random.seed(5)
    signal = wd.generate_sin(freqs=[440], duration=0.05, sampling_rate=8000)

    fig, ax = plt.subplots()
    # First plot
    signal.plot(ax=ax, overlay=True)
    initial_line_count = len(ax.lines)

    # Second plot with overlay
    signal.plot(ax=ax, overlay=True)
    final_line_count = len(ax.lines)

    # Should have more lines after second overlay
    assert final_line_count > initial_line_count


def test_plot_without_overlay_creates_new_figure() -> None:
    """Test plotting without overlay parameter creates new figure."""
    np.random.seed(6)
    signal = wd.generate_sin(freqs=[440], duration=0.05, sampling_rate=8000)

    ax1 = signal.plot()
    ax2 = signal.plot()

    # Should create different axes when overlay is not specified
    assert ax1 is not ax2


def test_spectrogram_plot_uses_provided_ax() -> None:
    """Test that SpectrogramFrame.plot() uses provided axes."""
    np.random.seed(7)
    signal = wd.generate_sin(freqs=[440, 880], duration=0.1, sampling_rate=8000)
    spectrogram = signal.spectrogram()

    fig, ax = plt.subplots()
    out = spectrogram.plot(ax=ax)

    # Should return the same Axes object
    assert out is ax

    # Spectrogram should create an image
    assert len(ax.images) >= 1 or len(ax.collections) >= 1


def test_noct_plot_uses_provided_ax() -> None:
    """Test that NOctFrame.plot() uses provided axes."""
    np.random.seed(8)
    signal = wd.generate_sin(freqs=[440, 880], duration=0.1, sampling_rate=8000)
    noct = signal.to_noct()

    fig, ax = plt.subplots()
    out = noct.plot(ax=ax)

    # Should return the same Axes object
    assert out is ax

    # Should contain plotted data
    assert len(ax.lines) >= 1


def test_plot_with_custom_labels() -> None:
    """Test plotting with custom axis labels."""
    np.random.seed(9)
    signal = wd.generate_sin(freqs=[440], duration=0.05, sampling_rate=8000)

    ax = signal.plot(xlabel="Custom Time [s]", ylabel="Custom Amplitude")

    # Check that labels were set
    assert ax.get_xlabel() == "Custom Time [s]"
    assert ax.get_ylabel() == "Custom Amplitude"


def test_plot_with_xlim_and_ylim() -> None:
    """Test plotting with custom axis limits."""
    np.random.seed(10)
    signal = wd.generate_sin(freqs=[440], duration=1.0, sampling_rate=8000)

    xlim = (0.2, 0.8)
    ylim = (-0.5, 0.5)
    ax = signal.plot(xlim=xlim, ylim=ylim)

    # Check that limits were set
    actual_xlim = ax.get_xlim()
    actual_ylim = ax.get_ylim()

    assert abs(actual_xlim[0] - xlim[0]) < 0.1
    assert abs(actual_xlim[1] - xlim[1]) < 0.1
    assert abs(actual_ylim[0] - ylim[0]) < 0.1
    assert abs(actual_ylim[1] - ylim[1]) < 0.1


def test_plot_preserves_figure_reference() -> None:
    """Test that plotting preserves the figure reference when ax is provided."""
    np.random.seed(11)
    signal = wd.generate_sin(freqs=[440], duration=0.05, sampling_rate=8000)

    fig, ax = plt.subplots()
    original_fig = ax.figure

    signal.plot(ax=ax)

    # Figure reference should remain the same
    assert ax.figure is original_fig
