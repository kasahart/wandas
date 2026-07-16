import logging
import numbers
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import dask.array as da
import numpy as np
import xarray as xr
from dask.array.core import Array as DaArray

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata
from wandas.frames.mixins.spectral_properties_mixin import SpectralPropertiesMixin
from wandas.pipeline.decorators import recipe_operation
from wandas.utils import validate_sampling_rate
from wandas.utils.types import NDArrayComplex, NDArrayReal

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from wandas.frames.cepstrogram import CepstrogramFrame
    from wandas.frames.channel import ChannelFrame
    from wandas.frames.spectral import SpectralFrame
    from wandas.visualization.plotting import PlotStrategy

logger = logging.getLogger(__name__)


class SpectrogramFrame(SpectralPropertiesMixin, BaseFrame[NDArrayComplex]):
    """
    Class for handling time-frequency domain data (spectrograms).

    This class represents spectrogram data obtained through
    Short-Time Fourier Transform (STFT)
    or similar time-frequency analysis methods. It provides methods for visualization,
    manipulation, and conversion back to time domain.

    Parameters
    ----------
    data : DaArray
        The spectrogram data. Must be a dask array with shape:
        - (channels, frequency_bins, time_frames) for multi-channel data
        - (frequency_bins, time_frames) for single-channel data, which will be
          reshaped to (1, frequency_bins, time_frames)
    sampling_rate : float
        The sampling rate of the original time-domain signal in Hz.
    n_fft : int
        The FFT size used to generate this spectrogram.
    hop_length : int
        Number of samples between successive frames.
    win_length : int, optional
        The window length in samples. If None, defaults to n_fft.
    window : str, default="hann"
        The window function to use (e.g., "hann", "hamming", "blackman").
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
        The magnitude spectrogram.
    phase : NDArrayReal
        The phase spectrogram in radians.
    power : NDArrayReal
        The power spectrogram.
    dB : NDArrayReal
        The spectrogram in decibels relative to channel reference values.
    dBA : NDArrayReal
        The A-weighted spectrogram in decibels.
    n_frames : int
        Number of time frames.
    n_freq_bins : int
        Number of frequency bins.
    freqs : NDArrayReal
        The frequency axis values in Hz.
    times : NDArrayReal
        The time axis values in seconds.

    Examples
    --------
    Create a spectrogram from a time-domain signal:
    >>> signal = ChannelFrame.from_wav("audio.wav")
    >>> spectrogram = signal.stft(n_fft=2048, hop_length=512)

    Extract a specific time frame:
    >>> frame_at_1s = spectrogram.get_frame_at(int(1.0 * sampling_rate / hop_length))

    Convert back to time domain:
    >>> reconstructed = spectrogram.to_channel_frame()

    Plot the spectrogram:
    >>> spectrogram.plot()

    Notes
    -----
    ``sampling_rate``, ``n_fft``, ``hop_length``, ``win_length``, and ``window``
    are immutable analysis state so the represented frequency and local-time axes
    cannot drift out of synchronization.
    """

    _xarray_dim_suffix = ("channel", "frequency", "time")

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
        window: str = "hann",
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        lineage: Any | None = None,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]] | None = None,
        channel_ids: list[str] | None = None,
        previous: "BaseFrame[Any] | None" = None,
        source_time_offset: float | Sequence[float] | NDArrayReal = 0.0,
        operation_history_prefix: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if data.ndim == 2:
            data = da.expand_dims(data, axis=0)
        elif data.ndim != 3:
            raise ValueError(
                f"Invalid data dimensions\n"
                f"  Got: {data.ndim}D array with shape {data.shape}\n"
                f"  Expected: 2D or 3D array\n"
                f"Spectrograms require 2D (freq x time) or "
                f"3D (channel x freq x time) data."
            )
        validate_sampling_rate(sampling_rate)
        normalized_n_fft = self._positive_integer(n_fft, name="n_fft")
        normalized_hop_length = self._positive_integer(hop_length, name="hop_length")
        normalized_win_length = (
            normalized_n_fft if win_length is None else self._positive_integer(win_length, name="win_length")
        )
        if normalized_win_length > normalized_n_fft:
            raise ValueError(
                "Invalid win_length for SpectrogramFrame\n"
                f"  Got: {normalized_win_length} for n_fft={normalized_n_fft}\n"
                "  Expected: win_length <= n_fft\n"
                "Use the analysis state of the source signal."
            )
        if normalized_hop_length > normalized_win_length:
            raise ValueError(
                "Invalid hop_length for SpectrogramFrame\n"
                f"  Got: {normalized_hop_length} for win_length={normalized_win_length}\n"
                "  Expected: hop_length <= win_length\n"
                "Use the analysis state of the source signal."
            )
        if not isinstance(window, str) or not window.strip():
            raise TypeError("SpectrogramFrame window must be a non-empty string.")
        expected_bins = normalized_n_fft // 2 + 1
        if int(data.shape[-2]) > expected_bins:
            raise ValueError(
                f"Invalid frequency bin count\n"
                f"  Got: {data.shape[-2]} bins\n"
                f"  Maximum: {expected_bins} bins (n_fft={normalized_n_fft})\n"
                "Use a one-sided spectrogram or a slice of its represented frequency axis."
            )

        self._n_fft = normalized_n_fft
        self._hop_length = normalized_hop_length
        self._win_length = normalized_win_length
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

    @staticmethod
    def _positive_integer(value: int, *, name: str) -> int:
        """Return one normalized positive analysis-state integer."""
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(
                f"Invalid {name} for SpectrogramFrame\n"
                f"  Got: {type(value).__name__}\n"
                "  Expected: a positive integer\n"
                "Use the analysis state of the source signal."
            )
        normalized = int(value)
        if normalized <= 0:
            raise ValueError(
                f"Invalid {name} for SpectrogramFrame\n"
                f"  Got: {normalized}\n"
                "  Expected: a positive integer\n"
                "Use the analysis state of the source signal."
            )
        return normalized

    @property
    def n_fft(self) -> int:
        """Return the immutable FFT size defining the frequency axis."""
        return self._n_fft

    @property
    def hop_length(self) -> int:
        """Return the immutable sample spacing defining the time axis."""
        return self._hop_length

    @property
    def win_length(self) -> int:
        """Return the immutable originating analysis-window length."""
        return self._win_length

    @property
    def window(self) -> str:
        """Return the immutable originating analysis-window name."""
        return self._window

    @property
    def sampling_rate(self) -> float:
        """Return the immutable rate defining the frequency and time axes."""
        return float(self._xr.attrs["sampling_rate"])

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        validate_sampling_rate(value)
        current = self._xr.attrs.get("sampling_rate")
        if current is not None and float(current) != float(value):
            raise AttributeError("SpectrogramFrame sampling_rate is immutable because it defines both axes.")
        self._xr.attrs["sampling_rate"] = float(value)

    @property
    def n_frames(self) -> int:
        """
        Get the number of time frames.

        Returns
        -------
        int
            The number of time frames in the spectrogram.
        """
        return self.shape[-1]

    @property
    def n_freq_bins(self) -> int:
        """
        Get the number of frequency bins.

        Returns
        -------
        int
            The number of frequency bins (n_fft // 2 + 1).
        """
        return self.shape[-2]

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

    @property
    def times(self) -> NDArrayReal:
        """
        Get the time axis values in seconds.

        Returns
        -------
        NDArrayReal
            Array of time values corresponding to each time frame.
        """
        return np.asarray(self._xr.coords["time"].values, dtype=float).copy()

    def _xarray_coords(self, data: DaArray) -> dict[str, Any]:
        """Build channel, frequency, and local-time coordinates lazily."""
        coords = super()._xarray_coords(data)
        dims = self._xarray_dims(data)
        sampling_rate = getattr(self, "_pending_sampling_rate", None)
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if "frequency" in dims:
            full_axis = np.fft.rfftfreq(self.n_fft, 1.0 / sampling_rate)
            coords["frequency"] = ("frequency", full_axis[: int(data.shape[-2])])
        if "time" in dims:
            coords["time"] = (
                "time",
                np.arange(int(data.shape[-1]), dtype=float) * self.hop_length / sampling_rate,
            )
        return coords

    def _handle_multidim_indexing(self, key: tuple[Any, ...]) -> "SpectrogramFrame":
        """Preserve frequency slices while time slices reset to local zero."""
        result = cast("SpectrogramFrame", super()._handle_multidim_indexing(key))
        if len(key) > 1:
            selected = np.asarray(self.freqs[key[1]], dtype=float)
            result._xr = result._xr.assign_coords(frequency=("frequency", selected))
        return result

    def to_xarray(self) -> xr.DataArray:
        """Return an isolated xarray view with copied represented axes."""
        exported = super().to_xarray()
        for coordinate_name in ("frequency", "time"):
            coordinate = exported.coords[coordinate_name]
            exported = exported.assign_coords({coordinate_name: (coordinate.dims, coordinate.values.copy())})
        return exported

    @property
    def source_times(self) -> NDArrayReal:
        """Get frame times relative to the original source timeline."""
        return self.source_time_offset[:, None] + self.times[None, :]

    def plot(
        self,
        plot_type: str = "spectrogram",
        ax: "Axes | None" = None,
        title: str | None = None,
        cmap: str = "jet",
        vmin: float | None = None,
        vmax: float | None = None,
        fmin: float = 0,
        fmax: float | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        Aw: bool = False,  # noqa: N803
        overlay: bool = False,
        **kwargs: Any,
    ) -> "Axes | Iterator[Axes]":
        """
        Plot the spectrogram using various visualization strategies.

        Parameters
        ----------
        plot_type : str, default="spectrogram"
            Type of plot to create.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.
        title : str, optional
            Title for the plot. If None, uses the frame label.
        cmap : str, default="jet"
            Colormap name for the spectrogram visualization.
        vmin : float, optional
            Minimum value for colormap scaling (dB). Auto-calculated if None.
        vmax : float, optional
            Maximum value for colormap scaling (dB). Auto-calculated if None.
        fmin : float, default=0
            Minimum frequency to display (Hz).
        fmax : float, optional
            Maximum frequency to display (Hz). If None, uses Nyquist frequency.
        xlim : tuple[float, float], optional
            Time axis limits as (start_time, end_time) in seconds.
        ylim : tuple[float, float], optional
            Frequency axis limits as (min_freq, max_freq) in Hz.
        Aw : bool, default=False
            Whether to apply A-weighting to the spectrogram.
        overlay : bool, default=False
            Whether to overlay channels on a single axes.
        **kwargs : dict
            Additional keyword arguments passed to Matplotlib plotting methods.

        Returns
        -------
        Union[Axes, Iterator[Axes]]
            The matplotlib axes containing the plot, or an iterator of axes
            for multi-plot outputs.

        Examples
        --------
        >>> stft = cf.stft()
        >>> # Basic spectrogram
        >>> stft.plot()
        >>> # Custom color scale and frequency range
        >>> stft.plot(vmin=-80, vmax=-20, fmin=100, fmax=5000)
        >>> # A-weighted spectrogram
        >>> stft.plot(Aw=True, cmap="viridis")
        """
        from wandas.visualization.plotting import create_operation

        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # Get plot strategy
        plot_strategy: PlotStrategy[SpectrogramFrame] = create_operation(plot_type)

        # Build kwargs for plot strategy
        plot_kwargs = {
            "title": title,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "fmin": fmin,
            "fmax": fmax,
            "Aw": Aw,
            **kwargs,
        }
        if xlim is not None:
            plot_kwargs["xlim"] = xlim
        if ylim is not None:
            plot_kwargs["ylim"] = ylim

        # Execute plot
        _ax = plot_strategy.plot(self, ax=ax, overlay=overlay, **plot_kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def plot_Aw(  # noqa: N802
        self,
        plot_type: str = "spectrogram",
        ax: "Axes | None" = None,
        **kwargs: Any,
    ) -> "Axes | Iterator[Axes]":
        """
        Plot the A-weighted spectrogram.

        A convenience method that calls plot() with Aw=True, applying A-weighting
        to the spectrogram before plotting.

        Parameters
        ----------
        plot_type : str, default="spectrogram"
            Type of plot to create.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.
        **kwargs : dict
            Additional keyword arguments passed to plot().
            Accepts all parameters from plot() except Aw (which is set to True).

        Returns
        -------
        Union[Axes, Iterator[Axes]]
            The matplotlib axes containing the plot.

        Examples
        --------
        >>> stft = cf.stft()
        >>> # A-weighted spectrogram with custom settings
        >>> stft.plot_Aw(vmin=-60, vmax=-10, cmap="magma")
        """
        return self.plot(plot_type=plot_type, ax=ax, Aw=True, **kwargs)

    @recipe_operation("wandas.spectrogram.cepstrum")
    def cepstrum(self, floor: float = 1e-12) -> "CepstrogramFrame":
        """Calculate a real cepstrum independently at every time frame.

        Parameters
        ----------
        floor : float, default=1e-12
            Positive finite floor applied to normalized STFT magnitude before
            taking the logarithm.

        Returns
        -------
        CepstrogramFrame
            New lazy coefficients shaped ``(channel, quefrency, time)``. The
            source FFT size, hop length, window state, channels, metadata, and
            source-time offsets are preserved.

        Raises
        ------
        TypeError
            If ``floor`` is not a real number.
        ValueError
            If ``floor`` is non-positive or non-finite.

        Notes
        -----
        The source ``SpectrogramFrame`` already contains normalized one-sided
        STFT amplitudes. This method computes
        ``irfft(log(max(abs(stft), floor)))`` along its frequency axis without
        recomputing the time-domain STFT. It only builds a Dask graph.

        Examples
        --------
        >>> cepstrogram = frame.stft(n_fft=2048).cepstrum()
        >>> envelope = cepstrogram.lifter(0.002).to_spectral_envelope()
        """
        from wandas.frames.cepstrogram import CepstrogramFrame
        from wandas.processing import SpectrogramCepstrum, create_operation

        operation = cast(
            "SpectrogramCepstrum",
            create_operation(
                "spectrogram_cepstrum",
                self.sampling_rate,
                n_fft=self.n_fft,
                floor=floor,
            ),
        )
        return CepstrogramFrame(
            data=operation.process(self._data),
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            label=f"Cepstrogram of {self.label}",
            metadata=self.metadata,
            channel_metadata=self.channels.to_list(),
            channel_ids=self._channel_ids,
            previous=self,
            source_time_offset=self.source_time_offset,
            lineage=self._required_semantic_lineage(),
        )

    @recipe_operation("wandas.spectrogram.absolute")
    def abs(self) -> "SpectrogramFrame":
        """
        Compute the absolute value (magnitude) of the complex spectrogram.

        This method calculates the magnitude of each complex value in the
        spectrogram, converting the complex-valued data to real-valued magnitude data.
        The result remains a SpectrogramFrame but carries a real numeric dtype.

        Returns
        -------
        SpectrogramFrame
            A new SpectrogramFrame containing real-valued magnitudes.

        Examples
        --------
        >>> signal = ChannelFrame.from_wav("audio.wav")
        >>> spectrogram = signal.stft(n_fft=2048, hop_length=512)
        >>> magnitude_spectrogram = spectrogram.abs()
        >>> # The magnitude can be accessed via the magnitude property or data
        >>> print(magnitude_spectrogram.magnitude.shape)
        """
        logger.debug("Computing absolute value (magnitude) of spectrogram")

        new_metadata = self._updated_metadata("abs", {})
        from wandas.processing import create_operation

        operation = create_operation("abs", self.sampling_rate)
        magnitude_data = operation.process(self._data)

        logger.debug("Created new SpectrogramFrame with abs operation added to graph")

        return self._create_new_instance(
            data=magnitude_data,
            label=f"abs({self.label})",
            metadata=new_metadata,
            lineage=self._required_semantic_lineage(),
        )

    @recipe_operation("wandas.spectrogram.get_frame_at")
    def get_frame_at(self, time_idx: int) -> "SpectralFrame":
        """
        Extract spectral data at a specific time frame.

        Parameters
        ----------
        time_idx : int
            Index of the time frame to extract.

        Returns
        -------
        SpectralFrame
            A new SpectralFrame containing the spectral data at the specified time.

        Raises
        ------
        IndexError
            If time_idx is out of range.
        """
        from wandas.frames.spectral import SpectralFrame

        if time_idx < 0 or time_idx >= self.n_frames:
            raise IndexError(
                f"Time index out of range\n"
                f"  Got: {time_idx}\n"
                f"  Expected: 0 to {self.n_frames - 1}\n"
                f"Use an index within the valid range for this spectrogram."
            )

        frame_data = self._data[..., time_idx]

        lineage = self._required_semantic_lineage()
        result = SpectralFrame(
            data=frame_data,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            window=self.window,
            label=f"{self.label} (Frame {time_idx}, Time {self.times[time_idx]:.3f}s)",
            metadata=self.metadata,
            channel_metadata=self.channels.to_list(),
            channel_ids=self._channel_ids,
            source_time_offset=self.source_time_offset + float(self.times[time_idx]),
            lineage=lineage,
        )
        result._xr = result._xr.assign_coords(frequency=("frequency", self.freqs))
        return result

    @recipe_operation("wandas.spectrogram.to_channel_frame")
    def to_channel_frame(self) -> "ChannelFrame":
        """
        Convert the spectrogram back to time domain using inverse STFT.

        This method performs an inverse Short-Time Fourier Transform (ISTFT) to
        reconstruct the time-domain signal from the spectrogram.

        Returns
        -------
        ChannelFrame
            A new ChannelFrame containing the reconstructed time-domain signal.

        See Also
        --------
        istft : Alias for this method with more intuitive naming.
        """
        from wandas.frames.channel import ChannelFrame
        from wandas.processing import ISTFT, create_operation

        expected_frequencies = np.fft.rfftfreq(self.n_fft, 1.0 / self.sampling_rate)
        if self.n_freq_bins != len(expected_frequencies) or not np.array_equal(self.freqs, expected_frequencies):
            represented_range = "empty" if self.n_freq_bins == 0 else f"{self.freqs[0]} to {self.freqs[-1]} Hz"
            raise ValueError(
                "Cannot invert a partial-frequency SpectrogramFrame\n"
                f"  Got: {self.n_freq_bins} represented bins ({represented_range})\n"
                f"  Expected: the complete {len(expected_frequencies)}-bin one-sided axis "
                f"from {expected_frequencies[0]} to {expected_frequencies[-1]} Hz\n"
                "ISTFT requires every one-sided frequency bin; use the unsliced SpectrogramFrame for inversion."
            )

        params = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
        }
        operation_name = "istft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("ISTFT", operation)
        # Apply processing to data
        time_series = operation.process(self._data)

        logger.debug(f"Created new ChannelFrame with operation {operation_name} added to graph")

        # Create new instance
        lineage = self._required_semantic_lineage()
        return ChannelFrame(
            data=time_series,
            sampling_rate=self.sampling_rate,
            label=f"istft({self.label})",
            metadata=self.metadata,
            channel_metadata=self.channels.to_list(),
            channel_ids=self._channel_ids,
            source_time_offset=self.source_time_offset,
            lineage=lineage,
        )

    def istft(self) -> "ChannelFrame":
        """
        Convert the spectrogram back to time domain using inverse STFT.

        This is an alias for `to_channel_frame()` with a more intuitive name.
        It performs an inverse Short-Time Fourier Transform (ISTFT) to
        reconstruct the time-domain signal from the spectrogram.

        Returns
        -------
        ChannelFrame
            A new ChannelFrame containing the reconstructed time-domain signal.

        See Also
        --------
        to_channel_frame : The underlying implementation.

        Examples
        --------
        >>> signal = ChannelFrame.from_wav("audio.wav")
        >>> spectrogram = signal.stft(n_fft=2048, hop_length=512)
        >>> reconstructed = spectrogram.istft()
        """
        return self.to_channel_frame()

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        Get additional initialization arguments for SpectrogramFrame.

        This internal method provides the additional initialization arguments
        required by SpectrogramFrame beyond those required by BaseFrame.

        Returns
        -------
        dict[str, Any]
            Additional initialization arguments.
        """
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
        }

    def _get_dataframe_index(self) -> "pd.Index[Any]":
        """DataFrame index is not supported for SpectrogramFrame."""
        raise NotImplementedError("DataFrame index is not supported for SpectrogramFrame.")

    def to_dataframe(self) -> "pd.DataFrame":
        """DataFrame conversion is not supported for SpectrogramFrame.

        SpectrogramFrame contains 3D data (channels, frequency_bins, time_frames)
        which cannot be directly converted to a 2D DataFrame. Consider using
        get_frame_at() to extract a specific time frame as a SpectralFrame,
        then convert that to a DataFrame.

        Raises
        ------
        NotImplementedError
            Always raised as DataFrame conversion is not supported.
        """
        raise NotImplementedError(
            "DataFrame conversion is not supported for SpectrogramFrame. "
            "Use get_frame_at() to extract a specific time frame as SpectralFrame, "
            "then convert that to a DataFrame."
        )

    def info(self) -> None:
        """Display comprehensive information about the SpectrogramFrame.

        This method prints a summary of the frame's properties including:
        - Number of channels
        - Sampling rate
        - FFT size
        - Hop length
        - Window length
        - Window function
        - Frequency range
        - Number of frequency bins
        - Frequency resolution (ΔF)
        - Number of time frames
        - Time resolution (ΔT)
        - Total duration
        - Channel labels
        - Number of operations applied

        This is a convenience method to view all key properties at once,
        similar to pandas DataFrame.info().

        Examples
        --------
        >>> signal = ChannelFrame.from_wav("audio.wav")
        >>> spectrogram = signal.stft(n_fft=2048, hop_length=512)
        >>> spectrogram.info()
        SpectrogramFrame Information:
          Channels: 2
          Sampling rate: 44100 Hz
          FFT size: 2048
          Hop length: 512 samples
          Window length: 2048 samples
          Window: hann
          Frequency range: 0.0 - 22050.0 Hz
          Frequency bins: 1025
          Frequency resolution (ΔF): 21.5 Hz
          Time frames: 100
          Time resolution (ΔT): 11.6 ms
          Total duration: 1.16 s
          Channel labels: ['ch0', 'ch1']
          Operations Applied: 1
        """
        # Calculate frequency resolution (ΔF) and time resolution (ΔT)
        delta_f = self.sampling_rate / self.n_fft
        delta_t_ms = (self.hop_length / self.sampling_rate) * 1000
        total_duration = (self.n_frames * self.hop_length) / self.sampling_rate

        print("SpectrogramFrame Information:")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  FFT size: {self.n_fft}")
        print(f"  Hop length: {self.hop_length} samples")
        print(f"  Window length: {self.win_length} samples")
        print(f"  Window: {self.window}")
        print(f"  Frequency range: {self.freqs[0]:.1f} - {self.freqs[-1]:.1f} Hz")
        print(f"  Frequency bins: {self.n_freq_bins}")
        print(f"  Frequency resolution (ΔF): {delta_f:.1f} Hz")
        print(f"  Time frames: {self.n_frames}")
        print(f"  Time resolution (ΔT): {delta_t_ms:.1f} ms")
        print(f"  Total duration: {total_duration:.2f} s")
        print(f"  Channel labels: {self.labels}")
        self._print_operation_history()

    @classmethod
    def from_numpy(
        cls,
        data: NDArrayComplex,
        sampling_rate: float,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
        window: str = "hann",
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        lineage: Any | None = None,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]] | None = None,
        channel_ids: list[str] | None = None,
        previous: "BaseFrame[Any] | None" = None,
    ) -> "SpectrogramFrame":
        """Create a SpectrogramFrame from a NumPy array.

        Args:
            data: NumPy array containing spectrogram data.
                Shape should be (n_channels, n_freq_bins, n_time_frames) or
                (n_freq_bins, n_time_frames) for single channel.
            sampling_rate: The sampling rate in Hz.
            n_fft: The FFT size used to generate this spectrogram.
            hop_length: Number of samples between successive frames.
            win_length: The window length in samples. If None, defaults to n_fft.
            window: The window function used (e.g., "hann", "hamming").
            label: A label for the frame.
            metadata: Optional metadata dictionary.
            lineage: Runtime operation lineage for this frame.
            channel_metadata: Metadata for each channel.
            previous: Reference to the previous frame in the processing chain.

        Returns:
            A new SpectrogramFrame containing the NumPy data.
        """

        # Normalize shape: support 2D single-channel inputs by expanding
        # to channel-first 3D shape. Reject 1D inputs as invalid for
        # spectrograms.
        if data.ndim not in (2, 3):
            raise ValueError(
                f"Invalid data shape\n"
                f"  Got: {data.shape}\n"
                f"  Expected: 2D (freq, time) or 3D (channel, freq, time) array\n"
                f"Provide a 2D or 3D array to represent time-frequency data."
            )
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        # Convert NumPy array to dask array
        # Use channel-wise chunking for spectrograms (1, -1, -1).
        # Use shared helper to avoid chunking typing issues
        from wandas.utils.dask_helpers import da_from_array as _da_from_array

        dask_data = _da_from_array(data, chunks=(1, -1, -1))
        sf = cls(
            data=dask_data,
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            label=label or "numpy_spectrogram",
            metadata=metadata,
            lineage=lineage,
            channel_metadata=channel_metadata,
            channel_ids=channel_ids,
            previous=previous,
        )
        return sf
