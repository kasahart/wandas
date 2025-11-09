# spectral_frame.py
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, cast

import dask
import dask.array as da
import librosa
import numpy as np
import pandas as pd
from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayComplex, NDArrayReal

from ..core.base_frame import BaseFrame
from ..core.metadata import ChannelMetadata

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..visualization.plotting import PlotStrategy
    from .channel import ChannelFrame
    from .noct import NOctFrame


dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]
da_from_array = da.from_array  # type: ignore [unused-ignore]

logger = logging.getLogger(__name__)

S = TypeVar("S", bound="BaseFrame[Any]")


class SpectralFrame(BaseFrame[NDArrayComplex]):
    """
    Class for handling frequency-domain signal data.

    This class represents spectral data, providing methods for spectral analysis,
    manipulation, and visualization. It handles complex-valued frequency domain data
    obtained through operations like FFT.

    Parameters
    ----------
    data : DaArray
        The spectral data. Must be a dask array with shape:
        - (channels, frequency_bins) for multi-channel data
        - (frequency_bins,) for single-channel data, which will be
          reshaped to (1, frequency_bins)
    sampling_rate : float
        The sampling rate of the original time-domain signal in Hz.
    n_fft : int
        The FFT size used to generate this spectral data.
    window : str, default="hann"
        The window function used in the FFT.
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
    magnitude : NDArrayReal
        The magnitude spectrum of the data.
    phase : NDArrayReal
        The phase spectrum in radians.
    unwrapped_phase : NDArrayReal
        The unwrapped phase spectrum in radians.
    power : NDArrayReal
        The power spectrum (magnitude squared).
    dB : NDArrayReal
        The spectrum in decibels relative to channel reference values.
    dBA : NDArrayReal
        The A-weighted spectrum in decibels.
    freqs : NDArrayReal
        The frequency axis values in Hz.

    Examples
    --------
    Create a SpectralFrame from FFT:
    >>> signal = ChannelFrame.from_numpy(data, sampling_rate=44100)
    >>> spectrum = signal.fft(n_fft=2048)

    Plot the magnitude spectrum:
    >>> spectrum.plot()

    Perform binary operations:
    >>> scaled = spectrum * 2.0
    >>> summed = spectrum1 + spectrum2  # Must have matching sampling rates

    Convert back to time domain:
    >>> time_signal = spectrum.ifft()

    Notes
    -----
    - All operations are performed lazily using dask arrays for efficient memory usage.
    - Binary operations (+, -, *, /) can be performed between SpectralFrames or with
      scalar values.
    - The class maintains the processing history and metadata through all operations.
    """

    n_fft: int
    window: str

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        n_fft: int,
        window: str = "hann",
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[
            Union[list[ChannelMetadata], list[dict[str, Any]]]
        ] = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ) -> None:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(
                f"Data must be 1-dimensional or 2-dimensional. Shape: {data.shape}"
            )
        self.n_fft = n_fft
        self.window = window
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
    def magnitude(self) -> NDArrayReal:
        """
        Get the magnitude spectrum.

        Returns
        -------
        NDArrayReal
            The absolute values of the complex spectrum.
        """
        return np.abs(self.data)

    @property
    def phase(self) -> NDArrayReal:
        """
        Get the phase spectrum.

        Returns
        -------
        NDArrayReal
            The phase angles of the complex spectrum in radians.
        """
        return np.angle(self.data)

    @property
    def unwrapped_phase(self) -> NDArrayReal:
        """
        Get the unwrapped phase spectrum.

        The unwrapped phase removes discontinuities of 2π radians, providing
        continuous phase values across frequency bins.

        Returns
        -------
        NDArrayReal
            The unwrapped phase angles of the complex spectrum in radians.
        """
        return np.unwrap(np.angle(self.data))

    @property
    def power(self) -> NDArrayReal:
        """
        Get the power spectrum.

        Returns
        -------
        NDArrayReal
            The squared magnitude spectrum.
        """
        return self.magnitude**2

    @property
    def dB(self) -> NDArrayReal:  # noqa: N802
        """
        Get the spectrum in decibels.

        The reference values are taken from channel metadata. If no reference
        is specified, uses 1.0.

        Returns
        -------
        NDArrayReal
            The spectrum in dB relative to channel references.
        """
        mag: NDArrayReal = self.magnitude
        ref_values: NDArrayReal = np.array([ch.ref for ch in self._channel_metadata])
        level: NDArrayReal = 20 * np.log10(
            np.maximum(mag / ref_values[:, np.newaxis], 1e-12)
        )

        return level

    @property
    def dBA(self) -> NDArrayReal:  # noqa: N802
        """
        Get the A-weighted spectrum in decibels.

        Applies A-weighting filter to the spectrum for better correlation with
        perceived loudness.

        Returns
        -------
        NDArrayReal
            The A-weighted spectrum in dB.
        """
        weighted: NDArrayReal = librosa.A_weighting(frequencies=self.freqs, min_db=None)
        return self.dB + weighted

    @property
    def _n_channels(self) -> int:
        """
        Get the number of channels in the data.

        Returns
        -------
        int
            The number of channels.
        """
        return int(self._data.shape[-2])

    @property
    def freqs(self) -> NDArrayReal:
        """
        Get the frequency axis values in Hz.

        Returns
        -------
        NDArrayReal
            Array of frequency values corresponding to each frequency bin.
        """
        return np.fft.rfftfreq(self.n_fft, 1.0 / self.sampling_rate)

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        """
        Implementation of operation application for spectral data.

        This internal method handles the application of various operations to
        spectral data, maintaining lazy evaluation through dask.

        Parameters
        ----------
        operation_name : str
            Name of the operation to apply.
        **params : Any
            Parameters for the operation.

        Returns
        -------
        S
            A new instance with the operation applied.
        """
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from ..processing import create_operation

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)

        # Apply processing to data
        processed_data = operation.process(self._data)

        # Update metadata
        operation_metadata = {"operation": operation_name, "params": params}
        new_history = self.operation_history.copy()
        new_history.append(operation_metadata)
        new_metadata = {**self.metadata}
        new_metadata[operation_name] = params

        logger.debug(
            f"Created new ChannelFrame with operation {operation_name} added to graph"
        )
        return self._create_new_instance(
            data=processed_data,
            metadata=new_metadata,
            operation_history=new_history,
        )

    def _binary_op(
        self,
        other: Union[
            "SpectralFrame", int, float, complex, NDArrayComplex, NDArrayReal, "DaArray"
        ],
        op: Callable[["DaArray", Any], "DaArray"],
        symbol: str,
    ) -> "SpectralFrame":
        """
        Common implementation for binary operations.

        This method handles binary operations between SpectralFrames and various types
        of operands, maintaining lazy evaluation through dask arrays.

        Parameters
        ----------
        other : Union[SpectralFrame, int, float, complex,
                        NDArrayComplex, NDArrayReal, DaArray]
            The right operand of the operation.
        op : callable
            Function to execute the operation (e.g., lambda a, b: a + b)
        symbol : str
            String representation of the operation (e.g., '+')

        Returns
        -------
        SpectralFrame
            A new SpectralFrame containing the result of the operation.

        Raises
        ------
        ValueError
            If attempting to operate with a SpectralFrame
            with a different sampling rate.
        """
        logger.debug(f"Setting up {symbol} operation (lazy)")

        # Handle potentially None metadata and operation_history
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        # Check if other is a ChannelFrame - improved type checking
        if isinstance(other, SpectralFrame):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    "Sampling rates do not match. Cannot perform operation."
                )

            # Directly operate on dask arrays (maintaining lazy execution)
            result_data = op(self._data, other._data)

            # Combine channel metadata
            merged_channel_metadata = []
            for self_ch, other_ch in zip(
                self._channel_metadata, other._channel_metadata
            ):
                ch = self_ch.model_copy(deep=True)
                ch["label"] = f"({self_ch['label']} {symbol} {other_ch['label']})"
                merged_channel_metadata.append(ch)

            operation_history.append({"operation": symbol, "with": other.label})

            return SpectralFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                n_fft=self.n_fft,
                window=self.window,
                label=f"({self.label} {symbol} {other.label})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=merged_channel_metadata,
                previous=self,
            )

        # Operation with scalar, NumPy array, or other types
        else:
            # Apply operation directly to dask array (maintaining lazy execution)
            result_data = op(self._data, other)

            # String representation of operand for display
            if isinstance(other, (int, float)):
                other_str = str(other)
            elif isinstance(other, complex):
                other_str = f"complex({other.real}, {other.imag})"
            elif isinstance(other, np.ndarray):
                other_str = f"ndarray{other.shape}"
            elif hasattr(other, "shape"):  # Check for dask.array.Array
                other_str = f"dask.array{other.shape}"
            else:
                other_str = str(type(other).__name__)

            # Update channel metadata
            updated_channel_metadata: list[ChannelMetadata] = []
            for self_ch in self._channel_metadata:
                ch = self_ch.model_copy(deep=True)
                ch["label"] = f"({self_ch.label} {symbol} {other_str})"
                updated_channel_metadata.append(ch)

            operation_history.append({"operation": symbol, "with": other_str})

            return SpectralFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                n_fft=self.n_fft,
                window=self.window,
                label=f"({self.label} {symbol} {other_str})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=updated_channel_metadata,
            )

    def plot(
        self,
        plot_type: str = "frequency",
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
        Plot the spectral data using various visualization strategies.

        Parameters
        ----------
        plot_type : str, default="frequency"
            Type of plot to create. Options include:
            - "frequency": Standard frequency plot
            - "matrix": Matrix plot for comparing channels
            - Other types as defined by available plot strategies
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.
        title : str, optional
            Title for the plot. If None, uses the frame label.
        overlay : bool, default=False
            Whether to overlay all channels on a single plot (True)
            or create separate subplots for each channel (False).
        xlabel : str, optional
            Label for the x-axis. If None, uses default "Frequency [Hz]".
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
        >>> spectrum = cf.fft()
        >>> # Basic frequency plot
        >>> spectrum.plot()
        >>> # Overlay with A-weighting
        >>> spectrum.plot(overlay=True, Aw=True)
        >>> # Custom styling
        >>> spectrum.plot(title="Frequency Spectrum", color="red", linewidth=2)
        """
        from wandas.visualization.plotting import create_operation

        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # Get plot strategy
        plot_strategy: PlotStrategy[SpectralFrame] = create_operation(plot_type)

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

    def ifft(self) -> "ChannelFrame":
        """
        Compute the Inverse Fast Fourier Transform (IFFT) to return to time domain.

        This method transforms the frequency-domain data back to the time domain using
        the inverse FFT operation. The window function used in the forward FFT is
        taken into account to ensure proper reconstruction.

        Returns
        -------
        ChannelFrame
            A new ChannelFrame containing the time-domain signal.
        """
        from ..processing import IFFT, create_operation
        from .channel import ChannelFrame

        params = {"n_fft": self.n_fft, "window": self.window}
        operation_name = "ifft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("IFFT", operation)
        # Apply processing to data
        time_series = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        # Create new instance
        return ChannelFrame(
            data=time_series,
            sampling_rate=self.sampling_rate,
            label=f"ifft({self.label})",
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=self._channel_metadata,
        )

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        Provide additional initialization arguments required for SpectralFrame.

        Returns
        -------
        dict[str, Any]
            Additional initialization arguments for SpectralFrame.
        """
        return {
            "n_fft": self.n_fft,
            "window": self.window,
        }

    def _get_dataframe_columns(self) -> list[str]:
        """Get channel labels as DataFrame columns."""
        return [ch.label for ch in self._channel_metadata]

    def _get_dataframe_index(self) -> "pd.Index[Any]":
        """Get frequency index for DataFrame."""
        return pd.Index(self.freqs, name="frequency")

    def noct_synthesis(
        self,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> "NOctFrame":
        """
        Synthesize N-octave band spectrum.

        This method combines frequency components into N-octave bands according to
        standard acoustical band definitions. This is commonly used in noise and
        vibration analysis.

        Parameters
        ----------
        fmin : float
            Lower frequency bound in Hz.
        fmax : float
            Upper frequency bound in Hz.
        n : int, default=3
            Number of bands per octave (e.g., 3 for third-octave bands).
        G : int, default=10
            Reference band number.
        fr : int, default=1000
            Reference frequency in Hz.

        Returns
        -------
        NOctFrame
            A new NOctFrame containing the N-octave band spectrum.

        Raises
        ------
        ValueError
            If the sampling rate is not 48000 Hz, which is required for this operation.
        """
        if self.sampling_rate != 48000:
            raise ValueError(
                "noct_synthesis can only be used with a sampling rate of 48000 Hz."
            )
        from ..processing import NOctSynthesis
        from .noct import NOctFrame

        params = {"fmin": fmin, "fmax": fmax, "n": n, "G": G, "fr": fr}
        operation_name = "noct_synthesis"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from ..processing import create_operation

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("NOctSynthesis", operation)
        # Apply processing to data
        spectrum_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        return NOctFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            fmin=fmin,
            fmax=fmax,
            n=n,
            G=G,
            fr=fr,
            label=f"1/{n}Oct of {self.label}",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {
                    "operation": "noct_synthesis",
                    "params": params,
                },
            ],
            channel_metadata=self._channel_metadata,
            previous=self,
        )

    def plot_matrix(
        self,
        plot_type: str = "matrix",
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """
        Plot channel relationships in matrix format.

        This method creates a matrix plot showing relationships between channels,
        such as coherence, transfer functions, or cross-spectral density.

        Parameters
        ----------
        plot_type : str, default="matrix"
            Type of matrix plot to create.
        **kwargs : dict
            Additional plot parameters:
            - vmin, vmax: Color scale limits
            - cmap: Colormap name
            - title: Plot title

        Returns
        -------
        Union[Axes, Iterator[Axes]]
            The matplotlib axes containing the plot.
        """
        from wandas.visualization.plotting import create_operation

        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # Get plot strategy
        plot_strategy: PlotStrategy[SpectralFrame] = create_operation(plot_type)

        # Execute plot
        _ax = plot_strategy.plot(self, **kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def info(self) -> None:
        """Display comprehensive information about the SpectralFrame.

        This method prints a summary of the frame's properties including:
        - Number of channels
        - Sampling rate
        - FFT size
        - Frequency range
        - Number of frequency bins
        - Frequency resolution (ΔF)
        - Channel labels

        This is a convenience method to view all key properties at once,
        similar to pandas DataFrame.info().

        Examples
        --------
        >>> spectrum = cf.fft()
        >>> spectrum.info()
        SpectralFrame Information:
          Channels: 2
          Sampling rate: 44100 Hz
          FFT size: 2048
          Frequency range: 0.0 - 22050.0 Hz
          Frequency bins: 1025
          Frequency resolution (ΔF): 21.5 Hz
          Channel labels: ['ch0', 'ch1']
          Operations Applied: 1
        """
        # Calculate frequency resolution (ΔF)
        delta_f = self.sampling_rate / self.n_fft

        print("SpectralFrame Information:")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  FFT size: {self.n_fft}")
        print(f"  Frequency range: {self.freqs[0]:.1f} - {self.freqs[-1]:.1f} Hz")
        print(f"  Frequency bins: {len(self.freqs)}")
        print(f"  Frequency resolution (ΔF): {delta_f:.1f} Hz")
        print(f"  Channel labels: {self.labels}")
        self._print_operation_history()
