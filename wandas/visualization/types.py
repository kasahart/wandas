"""Type definitions for visualization parameters."""

from typing import Any, TypedDict


class WaveformConfig(TypedDict, total=False):
    """Configuration for waveform plot in describe view.

    This corresponds to the time-domain plot shown at the top of the
    describe view.
    """

    xlabel: str
    ylabel: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]


class SpectralConfig(TypedDict, total=False):
    """Configuration for spectral plot in describe view.

    This corresponds to the frequency-domain plot (Welch) shown on the
    right side.
    """

    xlabel: str
    ylabel: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]


class DescribeParams(TypedDict, total=False):
    """Parameters for the describe visualization method.

    This visualization creates a comprehensive view with three plots:
    1. Time-domain waveform (top)
    2. Spectrogram (bottom-left)
    3. Frequency spectrum via Welch method (bottom-right)

    Attributes:
        fmin: Minimum frequency to display in the spectrogram (Hz).
            Default: 0
        fmax: Maximum frequency to display in the spectrogram (Hz).
            Default: Nyquist frequency
        cmap: Colormap for the spectrogram. Default: 'jet'
        vmin: Minimum value for spectrogram color scale (dB).
            Auto-calculated if None.
        vmax: Maximum value for spectrogram color scale (dB).
            Auto-calculated if None.
        xlim: Time axis limits (seconds) for all time-based plots.
        ylim: Frequency axis limits (Hz) for frequency-based plots.
        Aw: Apply A-weighting to the frequency analysis. Default: False
        waveform: Additional configuration dict for waveform subplot.
        spectral: Additional configuration dict for spectral subplot.
        normalize: Normalize audio data for playback. Default: True
        is_close: Close the figure after displaying. Default: True

    Deprecated (for backward compatibility):
        axis_config: Old configuration format.
            Use specific parameters instead.
        cbar_config: Old colorbar configuration. Use vmin/vmax instead.

    Examples:
        >>> cf = ChannelFrame.read_wav("audio.wav")
        >>> # Basic usage
        >>> cf.describe()
        >>>
        >>> # Custom frequency range
        >>> cf.describe(fmin=100, fmax=5000)
        >>>
        >>> # Custom color scale
        >>> cf.describe(vmin=-80, vmax=-20, cmap="viridis")
        >>>
        >>> # A-weighted analysis
        >>> cf.describe(Aw=True)
        >>>
        >>> # Custom time range
        >>> cf.describe(xlim=(0, 5))  # Show first 5 seconds
    """

    # Spectrogram parameters
    fmin: float
    fmax: float | None
    cmap: str
    vmin: float | None
    vmax: float | None

    # Axis limits
    xlim: tuple[float, float] | None
    ylim: tuple[float, float] | None

    # Weighting
    Aw: bool

    # Subplot configurations
    waveform: WaveformConfig
    spectral: SpectralConfig

    # Display options
    normalize: bool
    is_close: bool

    # Deprecated (backward compatibility)
    axis_config: dict[str, Any]
    cbar_config: dict[str, Any]
