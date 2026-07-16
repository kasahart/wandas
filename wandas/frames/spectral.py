# spectral_frame.py
from __future__ import annotations

import logging
import numbers
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr
from dask.array.core import Array as DaArray

from wandas.pipeline.decorators import recipe_operation
from wandas.utils import validate_sampling_rate
from wandas.utils.optional_imports import require_pandas
from wandas.utils.types import NDArrayComplex, NDArrayReal

from ..core.base_frame import BaseFrame
from ..core.metadata import ChannelMetadata
from .mixins.spectral_properties_mixin import SpectralPropertiesMixin

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from ..visualization.plotting import PlotStrategy
    from .channel import ChannelFrame
    from .noct import NOctFrame


logger = logging.getLogger(__name__)


class SpectralFrame(SpectralPropertiesMixin, BaseFrame[NDArrayComplex]):
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
        Required. The FFT size used to generate this spectral data. Must be a
        positive integer and must be the same FFT size that was used to create
        the spectrum (for example, 512 or 1024).
    window : str, default="hann"
        The window function used in the FFT.
    label : str, optional
        A label for the frame.
    metadata : dict, optional
        Additional metadata for the frame.
    lineage : LineageNode, optional
        Constructor override for the runtime lineage. When omitted, a source node is
        created. ``operation_history`` is its public derived projection.
    channel_metadata : list[ChannelMetadata], optional
        Metadata for each channel in the frame.
    previous : BaseFrame, optional
        Compatibility/debug pointer to the immediate prior frame; not the
        provenance source of truth.

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
    - ``sampling_rate``, ``n_fft``, and ``window`` are immutable analysis state;
      construct a new frame to represent a different frequency grid.
    - The class maintains runtime lineage and metadata through all operations.
    """

    _xarray_dim_suffix = ("channel", "frequency")

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        n_fft: int,
        window: str = "hann",
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]] | None = None,
        channel_ids: list[str] | None = None,
        previous: BaseFrame[Any] | None = None,
        source_time_offset: float | Sequence[float] | NDArrayReal = 0.0,
        lineage: Any | None = None,
        operation_history_prefix: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(f"Data must be 1-dimensional or 2-dimensional. Shape: {data.shape}")
        validate_sampling_rate(sampling_rate)
        if isinstance(n_fft, bool) or not isinstance(n_fft, numbers.Integral):
            raise TypeError(
                "Invalid n_fft for SpectralFrame\n"
                f"  Got: {type(n_fft).__name__}\n"
                "  Expected: a positive integer\n"
                "Pass the FFT size used to produce this spectrum."
            )
        normalized_n_fft = int(n_fft)
        if normalized_n_fft <= 0:
            raise ValueError(
                "Invalid n_fft for SpectralFrame\n"
                f"  Got: {normalized_n_fft}\n"
                "  Expected: a positive integer\n"
                "Pass the FFT size used to produce this spectrum."
            )
        if not isinstance(window, str) or not window.strip():
            raise TypeError("SpectralFrame window must be a non-empty string.")
        expected_bins = normalized_n_fft // 2 + 1
        if int(data.shape[-1]) > expected_bins:
            raise ValueError(
                "Invalid frequency bin count for SpectralFrame\n"
                f"  Got: {data.shape[-1]} bins\n"
                f"  Maximum: {expected_bins} bins for n_fft={normalized_n_fft}\n"
                "Use a one-sided spectrum or a slice of its represented frequency axis."
            )
        self._n_fft = normalized_n_fft
        self._window = window
        self._pending_sampling_rate = float(sampling_rate)
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            channel_metadata=channel_metadata,
            channel_ids=channel_ids,
            source_time_offset=source_time_offset,
            lineage=lineage,
            operation_history_prefix=operation_history_prefix,
            previous=previous,
        )
        del self._pending_sampling_rate

    @property
    def n_fft(self) -> int:
        """Return the immutable FFT size defining the frequency axis."""
        return self._n_fft

    @property
    def window(self) -> str:
        """Return the immutable originating analysis-window name."""
        return self._window

    @property
    def sampling_rate(self) -> float:
        """Return the immutable rate defining the frequency axis."""
        return float(self._xr.attrs["sampling_rate"])

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        validate_sampling_rate(value)
        current = self._xr.attrs.get("sampling_rate")
        if current is not None and float(current) != float(value):
            raise AttributeError("SpectralFrame sampling_rate is immutable because it defines the frequency axis.")
        self._xr.attrs["sampling_rate"] = float(value)

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
    def freqs(self) -> NDArrayReal:
        """
        Get the frequency axis values in Hz.

        Returns
        -------
        NDArrayReal
            Array of frequency values corresponding to each frequency bin.

        """
        return np.asarray(self._xr.coords["frequency"].values, dtype=float).copy()

    def _xarray_coords(self, data: DaArray) -> dict[str, Any]:
        """Build channel and represented-frequency coordinates lazily."""
        coords = super()._xarray_coords(data)
        if "frequency" in self._xarray_dims(data):
            sampling_rate = getattr(self, "_pending_sampling_rate", None)
            if sampling_rate is None:
                sampling_rate = self.sampling_rate
            full_axis = np.fft.rfftfreq(self.n_fft, 1.0 / sampling_rate)
            coords["frequency"] = ("frequency", full_axis[: int(data.shape[-1])])
        return coords

    def _require_complete_frequency_axis(self, operation_name: str) -> None:
        """Reject kernels that cannot interpret sliced represented frequencies."""
        expected_frequencies = np.fft.rfftfreq(self.n_fft, 1.0 / self.sampling_rate)
        if int(self._data.shape[-1]) == len(expected_frequencies) and np.array_equal(self.freqs, expected_frequencies):
            return
        represented_range = "empty" if len(self.freqs) == 0 else f"{self.freqs[0]} to {self.freqs[-1]} Hz"
        raise ValueError(
            f"Cannot run {operation_name} on a partial-frequency SpectralFrame\n"
            f"  Got: {self._data.shape[-1]} represented bins ({represented_range})\n"
            f"  Expected: the complete {len(expected_frequencies)}-bin one-sided axis "
            f"from {expected_frequencies[0]} to {expected_frequencies[-1]} Hz\n"
            f"{operation_name} requires every one-sided frequency bin; use the unsliced SpectralFrame."
        )

    def _handle_multidim_indexing(self, key: tuple[Any, ...]) -> SpectralFrame:
        """Preserve selected frequency coordinates during public slicing."""
        result = cast("SpectralFrame", super()._handle_multidim_indexing(key))
        if len(key) > 1:
            selected = np.asarray(self.freqs[key[1]], dtype=float)
            result._xr = result._xr.assign_coords(frequency=("frequency", selected))
        return result

    def to_xarray(self) -> xr.DataArray:
        """Return an isolated xarray view with copied frequency coordinates."""
        exported = super().to_xarray()
        coordinate = exported.coords["frequency"]
        return exported.assign_coords(frequency=(coordinate.dims, coordinate.values.copy()))

    def plot(
        self,
        plot_type: str = "frequency",
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        alpha: float = 1.0,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        Aw: bool = False,  # noqa: N803
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
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

    @recipe_operation("wandas.spectral.ifft")
    def ifft(self) -> ChannelFrame:
        """
        Compute the Inverse Fast Fourier Transform (IFFT) to return to time domain.

        This method transforms the frequency-domain data back to the time domain using
        the inverse FFT operation. The window function used in the forward FFT is
        taken into account to ensure proper reconstruction.

        Returns
        -------
        ChannelFrame
            A new ChannelFrame containing the time-domain signal.

        Raises
        ------
        ValueError
            If this frame does not represent the complete, zero-origin one-sided
            frequency axis required by the inverse transform.
        """
        self._require_complete_frequency_axis("IFFT")

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

        logger.debug(f"Created new SpectralFrame with operation {operation_name} added to graph")

        # Create new instance
        lineage = self._required_semantic_lineage()
        return ChannelFrame(
            data=time_series,
            sampling_rate=self.sampling_rate,
            label=f"ifft({self.label})",
            metadata=self.metadata,
            channel_metadata=self.channels.to_list(),
            channel_ids=self._channel_ids,
            source_time_offset=self.source_time_offset,
            lineage=lineage,
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

    def _get_dataframe_index(self) -> pd.Index[Any]:
        """Get frequency index for DataFrame."""
        pd = require_pandas("SpectralFrame.to_dataframe")
        return pd.Index(self.freqs, name="frequency")

    @recipe_operation("wandas.spectral.noct_synthesis")
    def noct_synthesis(
        self,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> NOctFrame:
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
            If the sampling rate is not 48000 Hz or this frame does not carry the
            complete canonical one-sided frequency axis required for synthesis.
        """
        if self.sampling_rate != 48000:
            raise ValueError("noct_synthesis can only be used with a sampling rate of 48000 Hz.")
        self._require_complete_frequency_axis("N-octave synthesis")
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

        logger.debug(f"Created new SpectralFrame with operation {operation_name} added to graph")

        lineage = self._required_semantic_lineage()
        return NOctFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            fmin=fmin,
            fmax=fmax,
            n=n,
            G=G,
            fr=fr,
            label=f"1/{n}Oct of {self.label}",
            metadata=self.metadata,
            channel_metadata=self.channels.to_list(),
            channel_ids=self._channel_ids,
            source_time_offset=self.source_time_offset,
            lineage=lineage,
            previous=self,
        )

    def plot_matrix(
        self,
        plot_type: str = "matrix",
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
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
