# spectral_frame.py
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

import dask
import dask.array as da
import librosa
import numpy as np
import pandas as pd
from dask.array.core import Array as DaArray
from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.visualization.plotting import PlotStrategy


dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]
da_from_array = da.from_array  # type: ignore [unused-ignore]

logger = logging.getLogger(__name__)

S = TypeVar("S", bound="BaseFrame[Any]")


class NOctFrame(BaseFrame[NDArrayReal]):
    """
    Class for handling N-octave band analysis data.

    This class represents frequency data analyzed in fractional octave bands,
    typically used in acoustic and vibration analysis. It handles real-valued
    data representing energy or power in each frequency band, following standard
    acoustical band definitions.

    Parameters
    ----------
    data : DaArray
        The N-octave band data. Must be a dask array with shape:
        - (channels, frequency_bins) for multi-channel data
        - (frequency_bins,) for single-channel data, which will be
          reshaped to (1, frequency_bins)
    sampling_rate : float
        The sampling rate of the original time-domain signal in Hz.
    fmin : float, default=0
        Lower frequency bound in Hz.
    fmax : float, default=0
        Upper frequency bound in Hz.
    n : int, default=3
        Number of bands per octave (e.g., 3 for third-octave bands).
    G : int, default=10
        Reference band number according to IEC 61260-1:2014.
    fr : int, default=1000
        Reference frequency in Hz, typically 1000 Hz for acoustic analysis.
    label : str, optional
        A label for the frame.
    metadata : dict, optional
        Additional metadata for the frame.
    operation_history : list[dict], optional
        History of operations performed on this frame.
    channel_metadata : list[ChannelMetadata], optional
        Metadata for each channel in the frame.
    previous : BaseFrame, optional
        The frame that this frame was derived from.

    Attributes
    ----------
    freqs : NDArrayReal
        The center frequencies of each band in Hz, calculated according to
        the standard fractional octave band definitions.
    dB : NDArrayReal
        The spectrum in decibels relative to channel reference values.
    dBA : NDArrayReal
        The A-weighted spectrum in decibels, applying frequency weighting
        for better correlation with perceived loudness.
    fmin : float
        Lower frequency bound in Hz.
    fmax : float
        Upper frequency bound in Hz.
    n : int
        Number of bands per octave.
    G : int
        Reference band number.
    fr : int
        Reference frequency in Hz.

    Examples
    --------
    Create an N-octave band spectrum from a time-domain signal:
    >>> signal = ChannelFrame.from_wav("audio.wav")
    >>> spectrum = signal.noct_spectrum(fmin=20, fmax=20000, n=3)

    Plot the N-octave band spectrum:
    >>> spectrum.plot()

    Plot with A-weighting applied:
    >>> spectrum.plot(Aw=True)

    Notes
    -----
    - Binary operations (addition, multiplication, etc.) are not currently
      supported for N-octave band data.
    - The actual frequency bands are determined by the parameters n, G, and fr
      according to IEC 61260-1:2014 standard for fractional octave band filters.
    - The class follows acoustic standards for band definitions and analysis,
      making it suitable for noise measurements and sound level analysis.
    - A-weighting is available for better correlation with human hearing
      perception, following IEC 61672-1:2013.
    """

    fmin: float
    fmax: float
    n: int
    G: int
    fr: int

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        fmin: float = 0,
        fmax: float = 0,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[list[ChannelMetadata]] = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ) -> None:
        """
        Initialize a NOctFrame instance.

        Sets up N-octave band analysis parameters and prepares the frame for
        storing band-filtered data. Data shape is validated to ensure compatibility
        with N-octave band analysis.

        See class docstring for parameter descriptions.
        """
        self.n = n
        self.G = G
        self.fr = fr
        self.fmin = fmin
        self.fmax = fmax
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata,
            previous=previous,
        )

    @property
    def dB(self) -> NDArrayReal:  # noqa: N802
        """
        Get the spectrum in decibels relative to each channel's reference value.

        The reference value for each channel is specified in its metadata.
        A minimum value of -120 dB is enforced to avoid numerical issues.

        Returns
        -------
        NDArrayReal
            The spectrum in decibels. Shape matches the input data shape:
            (channels, frequency_bins).
        """
        # Collect dB reference values from _channel_metadata
        ref = np.array([ch.ref for ch in self._channel_metadata])
        # Convert to dB
        # Use either the maximum value or 1e-12 to avoid division by zero
        level: NDArrayReal = 20 * np.log10(
            np.maximum(self.data / ref[..., np.newaxis], 1e-12)
        )
        return level

    @property
    def dBA(self) -> NDArrayReal:  # noqa: N802
        """
        Get the A-weighted spectrum in decibels.

        A-weighting applies a frequency-dependent weighting filter that approximates
        the human ear's response to different frequencies. This is particularly useful
        for analyzing noise and acoustic measurements as it provides a better
        correlation with perceived loudness.

        The weighting is applied according to IEC 61672-1:2013 standard.

        Returns
        -------
        NDArrayReal
            The A-weighted spectrum in decibels. Shape matches the input data shape:
            (channels, frequency_bins).
        """
        # Collect dB reference values from _channel_metadata
        weighted: NDArrayReal = librosa.A_weighting(frequencies=self.freqs, min_db=None)
        return self.dB + weighted

    @property
    def _n_channels(self) -> int:
        """
        Get the number of channels in the data.

        Returns
        -------
        int
            The number of channels in the N-octave band data.
        """
        return int(self._data.shape[-2])

    @property
    def freqs(self) -> NDArrayReal:
        """
        Get the center frequencies of each band in Hz.

        These frequencies are calculated based on the N-octave band parameters
        (n, G, fr) and the frequency bounds (fmin, fmax) according to
        IEC 61260-1:2014 standard for fractional octave band filters.

        Returns
        -------
        NDArrayReal
            Array of center frequencies for each frequency band.

        Raises
        ------
        ValueError
            If the center frequencies cannot be calculated or the result
            is not a numpy array.
        """
        _, freqs = _center_freq(
            fmax=self.fmax,
            fmin=self.fmin,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        if isinstance(freqs, np.ndarray):
            return freqs
        else:
            raise ValueError("freqs is not numpy array.")

    def _binary_op(
        self: S,
        other: Union[S, int, float, NDArrayReal, DaArray],
        op: Callable[[DaArray, Any], DaArray],
        symbol: str,
    ) -> S:
        """
        Binary operations are not currently supported for N-octave band data.

        Parameters
        ----------
        other : Union[S, int, float, NDArrayReal, DaArray]
            The right operand of the operation.
        op : callable
            Function to execute the operation.
        symbol : str
            String representation of the operation (e.g., '+', '-', '*', '/').

        Raises
        ------
        NotImplementedError
            Always raises this error as operations are not implemented
            for N-octave band data.
        """
        raise NotImplementedError(
            f"Operation {symbol} is not implemented for NOctFrame."
        )
        return self

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        """
        Apply operations using lazy evaluation.
        """
        # Apply operations using lazy evaluation
        raise NotImplementedError(
            f"Operation {operation_name} is not implemented for NOctFrame."
        )
        return self

    def plot(
        self,
        plot_type: str = "noct",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        alpha: float = 1.0,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        Aw: bool = False,  # noqa: N803
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """
        Plot the N-octave band data using various visualization strategies.

        Supports standard plotting configurations for acoustic analysis,
        including decibel scales and A-weighting.

        Parameters
        ----------
        plot_type : str, default="noct"
            Type of plot to create. The default "noct" type creates a step plot
            suitable for displaying N-octave band data.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.
        title : str, optional
            Title for the plot. If None, uses a default title with band specification.
        overlay : bool, default=False
            Whether to overlay all channels on a single plot (True)
            or create separate subplots for each channel (False).
        xlabel : str, optional
            Label for the x-axis. If None, uses default "Center frequency [Hz]".
        ylabel : str, optional
            Label for the y-axis. If None, uses default based on data type.
        alpha : float, default=1.0
            Transparency level for the plot lines (0.0 to 1.0).
        xlim : tuple[float, float], optional
            Limits for the x-axis as (min, max) tuple.
        ylim : tuple[float, float], optional
            Limits for the y-axis as (min, max) tuple.
        Aw : bool, default=False
            Whether to apply A-weighting to the data.
        **kwargs : dict
            Additional matplotlib Line2D parameters
            (e.g., color, linewidth, linestyle).

        Returns
        -------
        Union[Axes, Iterator[Axes]]
            The matplotlib axes containing the plot, or an iterator of axes
            for multi-plot outputs.

        Examples
        --------
        >>> noct = spectrum.noct(n=3)
        >>> # Basic 1/3-octave plot
        >>> noct.plot()
        >>> # Overlay with A-weighting
        >>> noct.plot(overlay=True, Aw=True)
        >>> # Custom styling
        >>> noct.plot(title="1/3-Octave Spectrum", color="blue", linewidth=2)
        """
        from wandas.visualization.plotting import create_operation

        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # Get plot strategy
        plot_strategy: PlotStrategy[NOctFrame] = create_operation(plot_type)

        # Build kwargs for plot strategy
        plot_kwargs = {
            "title": title,
            "overlay": overlay,
            "Aw": Aw,
            **kwargs,
        }
        if xlabel is not None:
            plot_kwargs["xlabel"] = xlabel
        if ylabel is not None:
            plot_kwargs["ylabel"] = ylabel
        if alpha != 1.0:
            plot_kwargs["alpha"] = alpha
        if xlim is not None:
            plot_kwargs["xlim"] = xlim
        if ylim is not None:
            plot_kwargs["ylim"] = ylim

        # Execute plot
        _ax = plot_strategy.plot(self, ax=ax, **plot_kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        Get additional initialization arguments for NOctFrame.

        This internal method provides the additional initialization arguments
        required by NOctFrame beyond those required by BaseFrame. These include
        the N-octave band analysis parameters that define the frequency bands.

        Returns
        -------
        dict[str, Any]
            Additional initialization arguments specific to NOctFrame:
            - n: Number of bands per octave
            - G: Reference band number
            - fr: Reference frequency
            - fmin: Lower frequency bound
            - fmax: Upper frequency bound
        """
        return {
            "n": self.n,
            "G": self.G,
            "fr": self.fr,
            "fmin": self.fmin,
            "fmax": self.fmax,
        }

    def _get_dataframe_columns(self) -> list[str]:
        """Get channel labels as DataFrame columns."""
        return [ch.label for ch in self._channel_metadata]

    def _get_dataframe_index(self) -> "pd.Index[Any]":
        """Get frequency index for DataFrame."""
        return pd.Index(self.freqs, name="frequency")
