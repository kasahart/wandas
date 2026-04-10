import logging
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar, cast

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.array.core import Array as DaArray
from dask.array.core import concatenate
from IPython.display import Audio, display
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from wandas.utils import validate_sampling_rate
from wandas.utils.dask_helpers import da_from_array as _da_from_array
from wandas.utils.types import NDArrayReal

from ..core.base_frame import BaseFrame
from ..core.metadata import ChannelMetadata, FrameMetadata
from ..io.readers import get_file_reader
from .mixins import ChannelProcessingMixin, ChannelTransformMixin

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

dask_delayed = dask.delayed
da_from_delayed = da.from_delayed


S = TypeVar("S", bound="BaseFrame[Any]")


def _align_to_length(arr: DaArray, target_len: int, align: str, source_len: int) -> DaArray:
    """Pad or truncate *arr* along axis=1 to *target_len* samples.

    Raises ``ValueError`` when *align* is ``"strict"`` and lengths differ.
    """
    if arr.shape[1] == target_len:
        return arr
    if align not in ("pad", "truncate"):
        raise ValueError(
            f"Data length mismatch\n"
            f"  Existing frame: {target_len} samples\n"
            f"  Channel to add: {source_len} samples\n"
            f"Use align='pad' or align='truncate' to handle "
            f"length differences."
        )
    # Both modes: truncate if longer, pad if shorter
    arr = arr[:, :target_len]
    diff = target_len - arr.shape[1]
    if diff > 0:
        pad = _da_from_array(
            np.zeros((arr.shape[0], diff), dtype=arr.dtype),
            chunks=(1, -1),
        )
        return concatenate([arr, pad], axis=1)
    return arr


def _resolve_channels(channel: int | list[int] | None, n_channels: int) -> list[int]:
    """Normalise a channel specification into a validated list of indices."""
    if channel is None:
        return list(range(n_channels))
    if isinstance(channel, int):
        if channel < 0 or channel >= n_channels:
            raise ValueError(f"Channel specification is out of range: {channel} (valid range: 0-{n_channels - 1})")
        return [channel]
    if isinstance(channel, (list, tuple)):
        for ch in channel:
            if ch < 0 or ch >= n_channels:
                raise ValueError(f"Channel specification is out of range: {ch} (valid range: 0-{n_channels - 1})")
        return list(channel)
    raise TypeError("channel must be int, list, or None")


def _download_url(
    url: str,
    file_type: str | None,
    source_name: str | None,
    timeout: float,
) -> tuple[bytes, str | None, str | None]:
    """Download *url* and return ``(bytes, file_type, source_name)``."""
    import urllib.error
    import urllib.parse
    import urllib.request
    from pathlib import PurePosixPath

    url_path_part = urllib.parse.urlparse(url).path
    url_ext = PurePosixPath(url_path_part).suffix.lower() or None
    if file_type is None and url_ext:
        file_type = url_ext
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data: bytes = resp.read()
    except urllib.error.URLError as exc:
        raise OSError(
            f"Failed to download audio from URL\n"
            f"  URL: {url}\n"
            f"  Error: {exc}\n"
            f"Verify the URL is accessible and try again."
        ) from exc
    if source_name is None:
        source_name = url
    return data, file_type, source_name


def _is_file_like(obj: object) -> bool:
    """Return True if *obj* looks like a readable file-like object (not a path)."""
    return hasattr(obj, "read") and not isinstance(obj, (str, Path))


def _resolve_source(
    path: str | Path | bytes | bytearray | memoryview | BinaryIO,
    file_type: str | None,
) -> tuple[Any, Path | None, Any, str | None]:
    """Resolve *path* into ``(source_obj, path_obj, reader, normalized_file_type)``.

    Handles in-memory buffers, file-like objects, and filesystem paths.
    """

    is_in_memory = isinstance(path, (bytes, bytearray, memoryview)) or _is_file_like(path)
    if is_in_memory and file_type is None:
        raise ValueError(
            "File type is required when the extension is missing\n"
            "  Cannot determine format without an extension\n"
            "  Provide file_type like '.wav' or '.csv'"
        )

    normalized_file_type: str | None = None
    if file_type is not None:
        normalized_file_type = file_type.lower()
        if not normalized_file_type.startswith("."):
            normalized_file_type = f".{normalized_file_type}"

    if is_in_memory:
        if _is_file_like(path):
            file_obj = cast(BinaryIO, path)
            if hasattr(file_obj, "seek"):
                try:
                    file_obj.seek(0)
                except Exception as exc:
                    logger.debug(
                        "Failed to rewind file-like object before read: %r",
                        exc,
                    )
            source: bytes = file_obj.read()
        else:
            source = bytes(cast(bytes | bytearray | memoryview, path))
        reader = get_file_reader(normalized_file_type or "", file_type=normalized_file_type)
        return source, None, reader, normalized_file_type

    path_obj = Path(cast(str | Path, path))
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Audio file not found\n"
            f"  Path: {path_obj.absolute()}\n"
            f"  Current directory: {Path.cwd()}\n"
            f"Please check:\n"
            f"  - File path is correct\n"
            f"  - File exists at the specified location\n"
            f"  - You have read permissions for the file"
        )
    reader = get_file_reader(path_obj)
    return path_obj, path_obj, reader, normalized_file_type


class ChannelFrame(BaseFrame[NDArrayReal], ChannelProcessingMixin, ChannelTransformMixin):
    """Channel-based data frame for handling audio signals and time series data.

    This frame represents channel-based data such as audio signals and time series data,
    with each channel containing data samples in the time domain.
    """

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        label: str | None = None,
        metadata: "FrameMetadata | dict[str, Any] | None" = None,
        operation_history: list[dict[str, Any]] | None = None,
        channel_metadata: list[ChannelMetadata] | list[dict[str, Any]] | None = None,
        previous: "BaseFrame[Any] | None" = None,
    ) -> None:
        """Initialize a ChannelFrame.

        Args:
            data: Dask array containing channel data.
                Shape should be (n_channels, n_samples).
            sampling_rate: The sampling rate of the data in Hz.
                Must be a positive value.
            label: A label for the frame.
            metadata: Optional metadata dictionary.
            operation_history: History of operations applied to the frame.
            channel_metadata: Metadata for each channel.
            previous: Reference to the previous frame in the processing chain.

        Raises:
            ValueError: If data has more than 2 dimensions, or if
                sampling_rate is not positive.
        """
        # Validate sampling rate
        validate_sampling_rate(sampling_rate)

        # Validate and reshape data
        if data.ndim == 1:
            data = da.reshape(data, (1, -1))
        elif data.ndim > 2:
            raise ValueError(
                f"Invalid data shape for ChannelFrame\n"
                f"  Got: {data.shape} ({data.ndim}D)\n"
                f"  Expected: 1D (samples,) or 2D (channels, samples)\n"
                f"If you have a 1D array, it will be automatically reshaped to\n"
                f"  (1, n_samples).\n"
                f"For higher-dimensional data, reshape it before creating\n"
                f"  ChannelFrame:\n"
                f"  Example: data.reshape(n_channels, -1)"
            )
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata,
            previous=previous,
        )

    def _set_channel_labels(self, ch_labels: list[str]) -> None:
        """Overwrite channel labels after construction.

        Raises ``ValueError`` if the list length does not match ``n_channels``.
        """
        if len(ch_labels) != self.n_channels:
            raise ValueError("Number of channel labels does not match the number of channels")
        for i, lbl in enumerate(ch_labels):
            self._channel_metadata[i].label = lbl

    def _set_channel_units(self, ch_units: list[str]) -> None:
        """Overwrite channel units after construction.

        Raises ``ValueError`` if the list length does not match ``n_channels``.
        """
        if len(ch_units) != self.n_channels:
            raise ValueError("Number of channel units does not match the number of channels")
        for i, unit in enumerate(ch_units):
            self._channel_metadata[i].unit = unit

    def _finalize_channel_update(
        self,
        new_data: DaArray,
        new_chmeta: list[ChannelMetadata],
        inplace: bool,
    ) -> "ChannelFrame":
        """Apply *new_data* and *new_chmeta* either in-place or as a new frame."""
        if inplace:
            self._data = new_data
            self._channel_metadata = new_chmeta
            return self
        return ChannelFrame(
            data=new_data,
            sampling_rate=self.sampling_rate,
            label=self.label,
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=new_chmeta,
            previous=self,
        )

    @property
    def time(self) -> NDArrayReal:
        """Get time array for the signal.

        The time array represents the start time of each sample, calculated as
        sample_index / sampling_rate. This provides a uniform, evenly-spaced
        time axis that is consistent across all frame types in wandas.

        For frames resulting from windowed analysis operations (e.g., FFT,
        loudness, roughness), each time point corresponds to the start of
        the analysis window, not the center. This differs from some libraries
        (e.g., MoSQITo) which use window center times, but does not affect
        the calculated values themselves.

        Returns:
            Array of time points in seconds, starting from 0.0.

        Examples:
            >>> import wandas as wd
            >>> signal = wd.read_wav("audio.wav")
            >>> time = signal.time
            >>> print(f"Duration: {time[-1]:.3f}s")
            >>> print(f"Time step: {time[1] - time[0]:.6f}s")
        """
        return np.arange(self.n_samples) / self.sampling_rate

    @property
    def n_samples(self) -> int:
        """Returns the number of samples."""
        n: int = self._data.shape[-1]
        return n

    @property
    def duration(self) -> float:
        """Returns the duration in seconds."""
        return self.n_samples / self.sampling_rate

    @property
    def _float_data(self) -> DaArray:
        """Return data cast to float64 if not already floating-point.

        Prevents integer overflow when squaring (e.g. int16 samples).
        """
        data = self._data
        if not np.issubdtype(data.dtype, np.floating):
            return data.astype(np.float64)
        return data

    @property
    def rms(self) -> NDArrayReal:
        """Calculate RMS (Root Mean Square) value for each channel.

        This is a scalar reduction: it computes one value per channel and
        triggers immediate computation of the underlying Dask graph.  The
        result is a plain NumPy array and does **not** produce a new frame,
        so no operation history is recorded.

        The RMS is defined as::

            rms[i] = sqrt(mean(x[i] ** 2))

        where ``x[i]`` is the sample array for channel ``i``.

        Returns:
            NDArrayReal of shape ``(n_channels,)`` containing the RMS value
            for each channel.

        Examples:
            >>> cf = ChannelFrame.read_wav("audio.wav")
            >>> rms_values = cf.rms
            >>> print(f"RMS values: {rms_values}")
            >>> # Select channels with RMS > threshold
            >>> active_channels = cf[cf.rms > 0.5]
        """
        # Compute RMS per channel.  axis=1 is the sample axis for data of
        # shape (channels, samples).  .compute() materialises the Dask graph
        # and np.array() ensures the result is a concrete NumPy ndarray.
        data = self._float_data
        rms_values = da.sqrt((data**2).mean(axis=1))
        return np.array(rms_values.compute())

    @property
    def crest_factor(self) -> NDArrayReal:
        """Calculate the crest factor (peak-to-RMS ratio) for each channel.

        This is a scalar reduction: it computes one value per channel and
        triggers immediate computation of the underlying Dask graph.  The
        result is a plain NumPy array and does **not** produce a new frame,
        so no operation history is recorded.

        The crest factor is defined as::

            crest_factor[i] = max(|x[i]|) / sqrt(mean(x[i] ** 2))

        where ``x[i]`` is the sample array for channel ``i``.

        For a pure sine wave the theoretical continuous-time crest factor is
        sqrt(2) ≈ 1.414; in discrete-time this implementation typically
        yields a value close to this, and exactly equal only when the sampled
        waveform contains its true peaks. Channels with zero RMS (all-zero
        signals) return 1.0 (defined by convention; no division by zero is
        performed).

        Returns:
            NDArrayReal of shape ``(n_channels,)`` containing the crest factor
            for each channel.  All-zero channels yield 1.0.

        Examples:
            >>> cf = ChannelFrame.read_wav("audio.wav")
            >>> cf_values = cf.crest_factor
            >>> print(f"Crest factors: {cf_values}")
            >>> # Select channels with crest factor above threshold
            >>> impulsive_channels = cf[cf.crest_factor > 3.0]
        """
        data = self._float_data
        peak = da.max(da.abs(data), axis=1)
        rms_vals = da.sqrt((data**2).mean(axis=1))
        # Use a safe denominator so the division never sees a zero RMS value,
        # then replace the result for zero-RMS channels with 1.0 by convention.
        safe_rms = da.where(rms_vals == 0, 1.0, rms_vals)
        crest = da.where(rms_vals != 0, peak / safe_rms, 1.0)
        return np.array(crest.compute())

    def info(self) -> None:
        """Display comprehensive information about the ChannelFrame.

        This method prints a summary of the frame's properties including:
        - Number of channels
        - Sampling rate
        - Duration
        - Number of samples
        - Channel labels

        This is a convenience method to view all key properties at once,
        similar to pandas DataFrame.info().

        Examples
        --------
        >>> cf = ChannelFrame.read_wav("audio.wav")
        >>> cf.info()
        Channels: 2
        Sampling rate: 44100 Hz
        Duration: 1.0 s
        Samples: 44100
        Channel labels: ['ch0', 'ch1']
        """
        print("ChannelFrame Information:")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Duration: {self.duration:.1f} s")
        print(f"  Samples: {self.n_samples}")
        print(f"  Channel labels: {self.labels}")
        self._print_operation_history()

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from ..processing import create_operation

        operation = create_operation(operation_name, self.sampling_rate, **params)
        return self._apply_operation_instance(operation, operation_name=operation_name)

    def add(
        self,
        other: "ChannelFrame | int | float | NDArrayReal",
        snr: float | None = None,
    ) -> "ChannelFrame":
        """Add another signal or value to the current signal.

        If SNR is specified, performs addition with consideration for
        signal-to-noise ratio.

        Args:
            other: Signal or value to add.
            snr: Signal-to-noise ratio (dB). If specified, adjusts the scale of the
                other signal based on this SNR.
                self is treated as the signal, and other as the noise.

        Returns:
            A new channel frame containing the addition result (lazy execution).
        """
        logger.debug(f"Setting up add operation with SNR={snr} (lazy)")

        if isinstance(other, ChannelFrame):
            # Check if sampling rates match
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    f"Sampling rate mismatch\n"
                    f"  Signal: {self.sampling_rate} Hz\n"
                    f"  Other: {other.sampling_rate} Hz\n"
                    f"Resample both frames to the same rate before adding."
                )

        elif isinstance(other, np.ndarray):
            other = ChannelFrame.from_numpy(cast(NDArrayReal, other), self.sampling_rate, label="array_data")
        elif isinstance(other, int | float):
            return self + other
        else:
            raise TypeError(f"Addition target with SNR must be a ChannelFrame or NumPy array: {type(other)}")

        # If SNR is specified, adjust the length of the other signal
        if other.duration != self.duration:
            other = other.fix_length(length=self.n_samples)

        if snr is None:
            return self + other
        return self.apply_operation("add_with_snr", other=other._data, snr=snr)

    def plot(
        self,
        plot_type: str = "waveform",
        ax: "Axes | None" = None,
        title: str | None = None,
        overlay: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        alpha: float = 1.0,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Plot the frame data.

        Args:
            plot_type: Type of plot. Default is "waveform".
            ax: Optional matplotlib axes for plotting.
            title: Title for the plot. If None, uses the frame label.
            overlay: Whether to overlay all channels on a single plot (True)
                or create separate subplots for each channel (False).
            xlabel: Label for the x-axis. If None, uses default based on plot type.
            ylabel: Label for the y-axis. If None, uses default based on plot type.
            alpha: Transparency level for the plot lines (0.0 to 1.0).
            xlim: Limits for the x-axis as (min, max) tuple.
            ylim: Limits for the y-axis as (min, max) tuple.
            **kwargs: Additional matplotlib Line2D parameters
                (e.g., color, linewidth, linestyle).
                These are passed to the underlying matplotlib plot functions.

        Returns:
            Single Axes object or iterator of Axes objects.

        Examples:
            >>> cf = ChannelFrame.read_wav("audio.wav")
            >>> # Basic plot
            >>> cf.plot()
            >>> # Overlay all channels
            >>> cf.plot(overlay=True, alpha=0.7)
            >>> # Custom styling
            >>> cf.plot(title="My Signal", ylabel="Voltage [V]", color="red")
        """
        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # Get plot strategy
        from ..visualization.plotting import create_operation

        plot_strategy = create_operation(plot_type)

        # Build kwargs for plot strategy
        plot_kwargs = {
            "title": title,
            "overlay": overlay,
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

    def rms_plot(
        self,
        ax: "Axes | None" = None,
        title: str | None = None,
        overlay: bool = True,
        Aw: bool = False,  # noqa: N803
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Generate an RMS plot.

        Args:
            ax: Optional matplotlib axes for plotting.
            title: Title for the plot.
            overlay: Whether to overlay the plot on the existing axis.
            Aw: Apply A-weighting.
            **kwargs: Additional arguments passed to the plot() method.
                Accepts the same arguments as plot() including xlabel, ylabel,
                alpha, xlim, ylim, and matplotlib Line2D parameters.

        Returns:
            Single Axes object or iterator of Axes objects.

        Examples:
            >>> cf = ChannelFrame.read_wav("audio.wav")
            >>> # Basic RMS plot
            >>> cf.rms_plot()
            >>> # With A-weighting
            >>> cf.rms_plot(Aw=True)
            >>> # Custom styling
            >>> cf.rms_plot(ylabel="RMS [V]", alpha=0.8, color="blue")
        """
        kwargs = kwargs or {}
        ylabel = kwargs.pop("ylabel", "RMS")
        rms_ch: ChannelFrame = self.rms_trend(Aw=Aw, dB=True)
        return rms_ch.plot(ax=ax, ylabel=ylabel, title=title, overlay=overlay, **kwargs)

    @staticmethod
    def _apply_deprecated_describe_kwargs(plot_kwargs: dict[str, Any]) -> None:
        """Migrate deprecated ``axis_config`` / ``cbar_config`` into *plot_kwargs*."""
        if "axis_config" in plot_kwargs:
            logger.warning("axis_config is retained for backward compatibility but will be deprecated in the future.")
            axis_config = plot_kwargs["axis_config"]
            if "time_plot" in axis_config:
                plot_kwargs["waveform"] = axis_config["time_plot"]
            if "freq_plot" in axis_config:
                if "xlim" in axis_config["freq_plot"]:
                    vlim = axis_config["freq_plot"]["xlim"]
                    plot_kwargs["vmin"] = vlim[0]
                    plot_kwargs["vmax"] = vlim[1]
                if "ylim" in axis_config["freq_plot"]:
                    plot_kwargs["ylim"] = axis_config["freq_plot"]["ylim"]

        if "cbar_config" in plot_kwargs:
            logger.warning("cbar_config is retained for backward compatibility but will be deprecated in the future.")
            cbar_config = plot_kwargs["cbar_config"]
            if "vmin" in cbar_config:
                plot_kwargs["vmin"] = cbar_config["vmin"]
            if "vmax" in cbar_config:
                plot_kwargs["vmax"] = cbar_config["vmax"]

    def describe(
        self,
        normalize: bool = True,
        is_close: bool = True,
        *,
        fmin: float = 0,
        fmax: float | None = None,
        cmap: str = "jet",
        vmin: float | None = None,
        vmax: float | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        Aw: bool = False,  # noqa: N803
        waveform: dict[str, Any] | None = None,
        spectral: dict[str, Any] | None = None,
        image_save: str | Path | None = None,
        **kwargs: Any,
    ) -> list[Figure] | None:
        """Display visual and audio representation of the frame.

        This method creates a comprehensive visualization with three plots:
        1. Time-domain waveform (top)
        2. Spectrogram (bottom-left)
        3. Frequency spectrum via Welch method (bottom-right)

        Args:
            normalize: Whether to normalize the audio data for playback.
                Default: True
            is_close: Whether to close the figure after displaying.
                Default: True
            fmin: Minimum frequency to display in the spectrogram (Hz).
                Default: 0
            fmax: Maximum frequency to display in the spectrogram (Hz).
                Default: Nyquist frequency (sampling_rate / 2)
            cmap: Colormap for the spectrogram.
                Default: 'jet'
            vmin: Minimum value for spectrogram color scale (dB).
                Auto-calculated if None.
            vmax: Maximum value for spectrogram color scale (dB).
                Auto-calculated if None.
            xlim: Time axis limits (seconds) for all time-based plots.
                Format: (start_time, end_time)
            ylim: Frequency axis limits (Hz) for frequency-based plots.
                Format: (min_freq, max_freq)
            Aw: Apply A-weighting to the frequency analysis.
                Default: False
            waveform: Additional configuration dict for waveform subplot.
                Can include 'xlabel', 'ylabel', 'xlim', 'ylim'.
            spectral: Additional configuration dict for spectral subplot.
                Can include 'xlabel', 'ylabel', 'xlim', 'ylim'.
            image_save: Path to save the figure as an image file. If provided,
                the figure will be saved before closing. File format is determined
                from the extension (e.g., '.png', '.jpg', '.pdf'). For multi-channel
                frames, the channel index is appended to the filename stem
                (e.g., 'output_0.png', 'output_1.png'). Default: None.
            **kwargs: Deprecated parameters for backward compatibility only.
                - axis_config: Old configuration format (use waveform/spectral instead)
                - cbar_config: Old colorbar configuration (use vmin/vmax instead)

        Returns:
            None (default). When `is_close=False`, returns a list of matplotlib Figure
            objects created for each channel. The list length equals the number of
            channels in the frame.

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
            >>>
            >>> # Custom waveform subplot settings
            >>> cf.describe(waveform={"ylabel": "Custom Label"})
            >>>
            >>> # Save the figure to a file
            >>> cf.describe(image_save="output.png")
            >>>
            >>> # Get Figure objects for further manipulation (is_close=False)
            >>> figures = cf.describe(is_close=False)
            >>> fig = figures[0]
            >>> fig.savefig("custom_output.png")  # Custom save with modifications
        """
        # Prepare kwargs with explicit parameters
        plot_kwargs: dict[str, Any] = {
            "fmin": fmin,
            "fmax": fmax,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "xlim": xlim,
            "ylim": ylim,
            "Aw": Aw,
            "waveform": waveform or {},
            "spectral": spectral or {},
        }
        # Merge with additional kwargs
        plot_kwargs.update(kwargs)

        self._apply_deprecated_describe_kwargs(plot_kwargs)

        figures: list[Figure] = []

        for ch_idx, ch in enumerate(self):
            _ax = ch.plot("describe", title=f"{ch.label} {ch.labels[0]}", **plot_kwargs)
            if isinstance(_ax, Axes):
                ax = _ax
            elif isinstance(_ax, Iterator):
                ax = cast(Axes, next(_ax))
            else:
                raise TypeError(f"Unexpected type for plot result: {type(_ax)}. Expected Axes or Iterator[Axes].")
            # Extract figure from axes (existing pattern)
            fig = getattr(ax, "figure", None)

            if fig is not None and not is_close:
                figures.append(fig)

            # Save image before closing if requested
            if image_save is not None and fig is not None:
                if self.n_channels > 1:
                    save_path = Path(image_save)
                    ch_path = save_path.parent / f"{save_path.stem}_{ch_idx}{save_path.suffix}"
                    fig.savefig(ch_path, bbox_inches="tight")
                else:
                    fig.savefig(image_save, bbox_inches="tight")

            if fig is not None:
                display(fig)
            if is_close and fig is not None:
                fig.clf()  # Clear the figure to free memory
                plt.close(fig)

            # Play audio for each channel
            display(Audio(ch.data, rate=ch.sampling_rate, normalize=normalize))

        # Return figures only when is_close=False
        if is_close:
            return None
        return figures

    @classmethod
    def from_numpy(
        cls,
        data: NDArrayReal,
        sampling_rate: float,
        label: str | None = None,
        metadata: "FrameMetadata | dict[str, Any] | None" = None,
        ch_labels: list[str] | None = None,
        ch_units: list[str] | str | None = None,
    ) -> "ChannelFrame":
        """Create a ChannelFrame from a NumPy array.

        Args:
            data: NumPy array containing channel data.
            sampling_rate: The sampling rate in Hz.
            label: A label for the frame.
            metadata: Optional metadata dictionary.
            ch_labels: Labels for each channel.
            ch_units: Units for each channel.

        Returns:
            A new ChannelFrame containing the NumPy data.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(f"Data must be 1-dimensional or 2-dimensional. Shape: {data.shape}")

        # Convert NumPy array to dask array. Use channel-wise chunks so
        # the 0th axis (channels) is chunked per-channel and the sample
        # axis remains un-chunked by default.
        dask_data = _da_from_array(data, chunks=(1, -1))
        cf = cls(
            data=dask_data,
            sampling_rate=sampling_rate,
            label=label or "numpy_data",
            metadata=metadata,
        )
        if ch_labels is not None:
            cf._set_channel_labels(ch_labels)
        if ch_units is not None:
            if isinstance(ch_units, str):
                ch_units = [ch_units] * cf.n_channels
            cf._set_channel_units(ch_units)

        return cf

    @classmethod
    def from_ndarray(
        cls,
        array: NDArrayReal,
        sampling_rate: float,
        labels: list[str] | None = None,
        unit: list[str] | str | None = None,
        frame_label: str | None = None,
        metadata: "FrameMetadata | dict[str, Any] | None" = None,
    ) -> "ChannelFrame":
        """Create a ChannelFrame from a NumPy array.

        This method is deprecated. Use from_numpy instead.

        Args:
            array: Signal data. Each row corresponds to a channel.
            sampling_rate: Sampling rate (Hz).
            labels: Labels for each channel.
            unit: Unit of the signal.
            frame_label: Label for the frame.
            metadata: Optional metadata dictionary.

        Returns:
            A new ChannelFrame containing the data.
        """
        warnings.warn(
            "from_ndarray is deprecated. Use from_numpy instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_numpy(
            data=array,
            sampling_rate=sampling_rate,
            label=frame_label,
            metadata=metadata,
            ch_labels=labels,
            ch_units=unit,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path | bytes | bytearray | memoryview | BinaryIO,
        channel: int | list[int] | None = None,
        start: float | None = None,
        end: float | None = None,
        # NOTE: chunk_size removed — chunking is handled internally as
        # channel-wise (1, -1). This simplifies the API and prevents
        # users from accidentally breaking channel-wise parallelism.
        ch_labels: list[str] | None = None,
        # CSV-specific parameters
        time_column: int | str = 0,
        delimiter: str = ",",
        header: int | None = 0,
        file_type: str | None = None,
        source_name: str | None = None,
        normalize: bool = False,
        timeout: float = 10.0,
    ) -> "ChannelFrame":
        """Create a ChannelFrame from an audio file or URL.

        Note:
            The `chunk_size` parameter has been removed. ChannelFrame uses
            channel-wise chunking by default (chunks=(1, -1)). Use `.rechunk(...)`
            on the returned frame for custom sample-axis chunking.

        Args:
            path: Path to the audio file, in-memory bytes/stream, or an HTTP/HTTPS
                URL. When a URL is given it is downloaded in full before processing.
                The file extension is inferred from the URL path; supply `file_type`
                explicitly when the URL has no recognisable extension.
            channel: Channel(s) to load. None loads all channels.
            start: Start time in seconds.
            end: End time in seconds.
            ch_labels: Labels for each channel.
            time_column: For CSV files, index or name of the time column.
                Default is 0 (first column).
            delimiter: For CSV files, delimiter character. Default is ",".
            header: For CSV files, row number to use as header.
                Default is 0 (first row). Set to None if no header.
            file_type: File extension for in-memory data or URLs without a
                recognisable extension (e.g. ".wav", ".csv").
            source_name: Optional source name for in-memory data. Used in metadata.
            normalize: When False (default) and the effective file type is WAV
                (local path or URL), return raw integer PCM samples cast to float32
                (magnitudes preserved, e.g. 16384 stays 16384.0). When True,
                normalize to float32 in [-1.0, 1.0]. Non-WAV formats always use
                soundfile (normalized).
            timeout: Timeout in seconds for HTTP/HTTPS URL downloads. Default is
                10.0 seconds. Has no effect for local files or in-memory data.

        Returns:
            A new ChannelFrame containing the loaded audio data.

        Raises:
            ValueError: If channel specification is invalid or file cannot be read.
                Error message includes absolute path, current directory, and
                troubleshooting suggestions.

        Examples:
            >>> # Load WAV file (raw integer samples cast to float32 by default)
            >>> cf = ChannelFrame.from_file("audio.wav")
            >>> # Load WAV file normalized to float32 [-1.0, 1.0]
            >>> cf = ChannelFrame.from_file("audio.wav", normalize=True)
            >>> # Load specific channels
            >>> cf = ChannelFrame.from_file("audio.wav", channel=[0, 2])
            >>> # Load CSV file
            >>> cf = ChannelFrame.from_file("data.csv", time_column=0, delimiter=",", header=0)
            >>> # Load from a URL
            >>> cf = ChannelFrame.from_file("https://example.com/audio.wav")
        """
        from .channel import ChannelFrame

        # Detect and handle URL paths — download to bytes before any path logic.
        if isinstance(path, str) and path.startswith(("http://", "https://")):
            path, file_type, source_name = _download_url(path, file_type, source_name, timeout)

        source_obj, path_obj, reader, normalized_file_type = _resolve_source(path, file_type)

        # Build kwargs for reader
        reader_kwargs: dict[str, Any] = {}
        is_wav_file = (path_obj is not None and path_obj.suffix.lower() == ".wav") or (normalized_file_type == ".wav")
        if (path_obj is not None and path_obj.suffix.lower() == ".csv") or (normalized_file_type == ".csv"):
            reader_kwargs["time_column"] = time_column
            reader_kwargs["delimiter"] = delimiter
            if header is not None:
                reader_kwargs["header"] = header
        if is_wav_file:
            reader_kwargs["normalize"] = normalize

        info = reader.get_file_info(source_obj, **reader_kwargs)
        sr = info["samplerate"]
        n_channels = info["channels"]
        n_frames = info["frames"]
        ch_labels = ch_labels or info.get("ch_labels", None)

        logger.debug(f"File info: sr={sr}, channels={n_channels}, frames={n_frames}")

        # Channel selection processing
        channels_to_load = _resolve_channels(channel, n_channels)

        # Index calculation
        start_idx = 0 if start is None else max(0, int(start * sr))
        end_idx = n_frames if end is None else min(n_frames, int(end * sr))
        frames_to_read = end_idx - start_idx

        logger.debug(
            f"Setting up lazy load from file={path!r}, frames={frames_to_read}, "
            f"start_idx={start_idx}, end_idx={end_idx}"
        )

        # Settings for lazy loading
        expected_shape = (len(channels_to_load), frames_to_read)

        # Define the loading function using the file reader
        def _load_audio() -> NDArrayReal:
            logger.debug(">>> EXECUTING DELAYED LOAD <<<")
            # Use the reader to get audio data with parameters
            out = reader.get_data(
                source_obj,
                channels_to_load,
                start_idx,
                frames_to_read,
                **reader_kwargs,
            )
            if not isinstance(out, np.ndarray):
                raise ValueError("Unexpected data type after reading file")
            return out

        logger.debug(f"Creating delayed dask task with expected shape: {expected_shape}")

        # Create delayed operation
        delayed_data = dask_delayed(_load_audio)()
        logger.debug("Wrapping delayed function in dask array")

        # Create dask array from delayed computation and ensure channel-wise
        # chunks. The sample axis (1) uses -1 by default to avoid forcing
        # a sample chunk length here.
        dask_array = da_from_delayed(delayed_data, shape=expected_shape, dtype=np.float32)

        # Ensure channel-wise chunks
        dask_array = dask_array.rechunk((1, -1))

        logger.debug("ChannelFrame setup complete - actual file reading will occur on compute()")

        if source_name is not None:
            try:
                frame_label = Path(source_name).stem
            except (TypeError, ValueError, OSError):
                logger.debug(
                    "Using raw source_name as frame label because Path(source_name) failed; source_name=%r",
                    source_name,
                )
                frame_label = source_name
        elif path_obj is not None:
            frame_label = path_obj.stem
        else:
            frame_label = None
        source_file: str | None = None
        if path_obj is not None:
            source_file = str(path_obj.resolve())
        elif source_name is not None:
            source_file = source_name

        cf = ChannelFrame(
            data=dask_array,
            sampling_rate=sr,
            label=frame_label,
            metadata=FrameMetadata(source_file=source_file),
        )
        if ch_labels is not None:
            cf._set_channel_labels(ch_labels)
        return cf

    @classmethod
    def read_wav(
        cls,
        filename: str | Path | bytes | bytearray | memoryview | BinaryIO,
        labels: list[str] | None = None,
        normalize: bool = False,
    ) -> "ChannelFrame":
        """Utility method to read a WAV file.

        Args:
            filename: Path to the WAV file or in-memory bytes/stream.
            labels: Labels to set for each channel.
            normalize: When False (default) and the source is a WAV file path,
                return raw integer PCM samples cast to float32 (magnitudes preserved).
                For in-memory sources, always uses soundfile (normalized float32).
                When True, normalize to float32 in [-1.0, 1.0].

        Returns:
            A new ChannelFrame containing the data (lazy loading).
        """
        from .channel import ChannelFrame

        is_in_memory = isinstance(filename, (bytes, bytearray, memoryview)) or _is_file_like(filename)
        source_name: str | None = None
        if is_in_memory and _is_file_like(filename):
            source_name = getattr(filename, "name", None)
        cf = ChannelFrame.from_file(
            filename,
            ch_labels=labels,
            normalize=normalize,
            file_type=".wav" if is_in_memory else None,
            source_name=source_name,
        )
        return cf

    @classmethod
    def read_csv(
        cls,
        filename: str,
        time_column: int | str = 0,
        labels: list[str] | None = None,
        delimiter: str = ",",
        header: int | None = 0,
    ) -> "ChannelFrame":
        """Utility method to read a CSV file.

        Args:
            filename: Path to the CSV file.
            time_column: Index or name of the time column.
            labels: Labels to set for each channel.
            delimiter: Delimiter character.
            header: Row number to use as header.

        Returns:
            A new ChannelFrame containing the data (lazy loading).

        Examples:
            >>> # Read CSV with default settings
            >>> cf = ChannelFrame.read_csv("data.csv")
            >>> # Read CSV with custom delimiter
            >>> cf = ChannelFrame.read_csv("data.csv", delimiter=";")
            >>> # Read CSV without header
            >>> cf = ChannelFrame.read_csv("data.csv", header=None)
        """
        from .channel import ChannelFrame

        cf = ChannelFrame.from_file(
            filename,
            ch_labels=labels,
            time_column=time_column,
            delimiter=delimiter,
            header=header,
        )
        return cf

    def to_wav(self, path: str | Path, format: str | None = None) -> None:
        """Save the audio data to a WAV file.

        Args:
            path: Path to save the file.
            format: File format. If None, determined from file extension.
        """
        from wandas.io.wav_io import write_wav

        write_wav(str(path), self, format=format)

    def save(
        self,
        path: str | Path,
        *,
        format: str = "hdf5",
        compress: str | None = "gzip",
        overwrite: bool = False,
        dtype: str | np.dtype[Any] | None = None,
    ) -> None:
        """Save the ChannelFrame to a WDF (Wandas Data File) format.

        This saves the complete frame including all channel data and metadata
        in a format that can be loaded back with full fidelity.

        Args:
            path: Path to save the file. '.wdf' extension will be added if not present.
            format: Format to use (currently only 'hdf5' is supported)
            compress: Compression method ('gzip' by default, None for no compression)
            overwrite: Whether to overwrite existing file
            dtype: Optional data type conversion before saving (e.g. 'float32')

        Raises:
            FileExistsError: If the file exists and overwrite=False.
            NotImplementedError: For unsupported formats.

        Example:
            >>> cf = ChannelFrame.read_wav("audio.wav")
            >>> cf.save("audio_analysis.wdf")
        """
        from ..io.wdf_io import save as wdf_save

        wdf_save(
            self,
            path,
            format=format,
            compress=compress,
            overwrite=overwrite,
            dtype=dtype,
        )

    @classmethod
    def load(cls, path: str | Path, *, format: str = "hdf5") -> "ChannelFrame":
        """Load a ChannelFrame from a WDF (Wandas Data File) file.

        This loads data saved with the save() method, preserving all channel data,
        metadata, labels, and units.

        Args:
            path: Path to the WDF file
            format: Format of the file (currently only 'hdf5' is supported)

        Returns:
            A new ChannelFrame with all data and metadata loaded

        Raises:
            FileNotFoundError: If the file doesn't exist
            NotImplementedError: For unsupported formats

        Example:
            >>> cf = ChannelFrame.load("audio_analysis.wdf")
        """
        from ..io.wdf_io import load as wdf_load

        return wdf_load(path, format=format)

    def add_channel(
        self,
        data: "np.ndarray[Any, Any] | DaArray | ChannelFrame",
        label: str | None = None,
        align: str = "strict",
        suffix_on_dup: str | None = None,
        inplace: bool = False,
    ) -> "ChannelFrame":
        """Add a new channel to the frame.

        Args:
            data: Data to add as a new channel. Can be:
                - numpy array (1D or 2D)
                - dask array (1D or 2D)
                - ChannelFrame (channels will be added)
            label: Label for the new channel. If None, generates a default label.
                When data is a ChannelFrame, acts as a prefix: each channel in
                the input frame is renamed to ``"{label}_{original_label}"``.
                If None (the default), the original channel labels are used as-is.
            align: How to handle length mismatches:
                - "strict": Raise error if lengths don't match
                - "pad": Pad shorter data with zeros
                - "truncate": Truncate longer data to match
            suffix_on_dup: Suffix to add to duplicate labels. If None, raises error.
            inplace: If True, modifies the frame in place.
                Otherwise returns a new frame.

        Returns:
            Modified ChannelFrame (self if inplace=True, new frame otherwise).

        Raises:
            ValueError: If data length doesn't match and align="strict",
                or if label is duplicate and suffix_on_dup is None.
            TypeError: If data type is not supported.

        Examples:
            >>> cf = ChannelFrame.read_wav("audio.wav")
            >>> # Add a numpy array as a new channel
            >>> new_data = np.sin(2 * np.pi * 440 * cf.time)
            >>> cf_new = cf.add_channel(new_data, label="sine_440Hz")
            >>> # Add another ChannelFrame's channels
            >>> cf2 = ChannelFrame.read_wav("audio2.wav")
            >>> cf_combined = cf.add_channel(cf2)
        """
        # Handle ndarray/dask/same-type Frame
        if isinstance(data, ChannelFrame):
            if self.sampling_rate != data.sampling_rate:
                raise ValueError("sampling_rate mismatch")
            arr = _align_to_length(data._data, self.n_samples, align, data.n_samples)
            labels = self.labels
            new_labels: list[str] = []
            new_metadata_list: list[ChannelMetadata] = []
            for chmeta in data._channel_metadata:
                new_label = f"{label}_{chmeta.label}" if label is not None else chmeta.label
                if new_label in labels or new_label in new_labels:
                    if suffix_on_dup:
                        new_label += suffix_on_dup
                    else:
                        raise ValueError(
                            f"Duplicate channel label\n"
                            f"  Label: '{new_label}'\n"
                            f"  Existing labels: {labels + new_labels}\n"
                            f"Use suffix_on_dup parameter to automatically "
                            f"rename duplicates."
                        )
                new_labels.append(new_label)
                # Copy the entire channel_metadata and update only the label
                new_ch_meta = chmeta.model_copy(deep=True)
                new_ch_meta.label = new_label
                new_metadata_list.append(new_ch_meta)
            new_data = concatenate([self._data, arr], axis=0)

            new_chmeta = self._channel_metadata + new_metadata_list
            return self._finalize_channel_update(new_data, new_chmeta, inplace)
        if isinstance(data, np.ndarray):
            arr = _da_from_array(data.reshape(1, -1), chunks=(1, -1))
        elif isinstance(data, DaArray):
            arr = data[None, ...] if data.ndim == 1 else data
            if arr.shape[0] != 1:
                arr = arr.reshape((1, -1))
        else:
            raise TypeError("add_channel: ndarray/dask/ChannelFrame")
        arr = _align_to_length(arr, self.n_samples, align, arr.shape[1])
        labels = self.labels
        new_label = label or f"ch{len(labels)}"
        if new_label in labels:
            if suffix_on_dup:
                new_label += suffix_on_dup
            else:
                raise ValueError(
                    f"Duplicate channel label\n"
                    f"  Label: '{new_label}'\n"
                    f"  Existing labels: {labels}\n"
                    f"Use suffix_on_dup parameter to automatically "
                    f"rename duplicates."
                )
        new_data = concatenate([self._data, arr], axis=0)

        new_chmeta = [*self._channel_metadata, ChannelMetadata(label=new_label)]
        return self._finalize_channel_update(new_data, new_chmeta, inplace)

    def remove_channel(self, key: int | str, inplace: bool = False) -> "ChannelFrame":
        if isinstance(key, int):
            if not (0 <= key < self.n_channels):
                raise IndexError(f"index {key} out of range")
            idx = key
        else:
            labels = self.labels
            if key not in labels:
                raise KeyError(f"label {key} not found")
            idx = labels.index(key)
        new_data = self._data[[i for i in range(self.n_channels) if i != idx], :]
        new_chmeta = [ch for i, ch in enumerate(self._channel_metadata) if i != idx]
        return self._finalize_channel_update(new_data, new_chmeta, inplace)

    def rename_channels(
        self,
        mapping: dict[int | str, str],
        inplace: bool = False,
    ) -> "ChannelFrame":
        """Rename channels using a mapping dictionary.

        Args:
            mapping: Dictionary mapping old names to new names.
                Keys can be:
                - int: channel index (e.g., {0: "left"})
                - str: channel label (e.g., {"old_name": "new_name"})
            inplace: If True, modifies the frame in place.

        Returns:
            Modified ChannelFrame (self if inplace=True, new frame otherwise).

        Raises:
            KeyError: If a key in mapping doesn't exist.
            ValueError: If duplicate labels would be created.

        Examples:
            >>> cf = ChannelFrame.read_wav("audio.wav")
            >>> # Rename by index
            >>> cf_renamed = cf.rename_channels({0: "left", 1: "right"})
            >>> # Rename by label
            >>> cf_renamed = cf.rename_channels({"ch0": "vocals"})
        """
        labels = self.labels
        new_labels = labels.copy()

        # Resolve all keys to their target labels and validate
        resolved_mappings: list[tuple[int, str]] = []
        for old_key, new_label in mapping.items():
            if isinstance(old_key, int):
                # Index-based rename
                if not (0 <= old_key < self.n_channels):
                    raise KeyError(
                        f"Channel index out of range\n  Index: {old_key}\n  Total channels: {self.n_channels}"
                    )
                resolved_mappings.append((old_key, new_label))
            else:
                # Label-based rename
                if old_key not in labels:
                    raise KeyError(f"Channel label not found\n  Label: '{old_key}'\n  Existing labels: {labels}")
                idx = labels.index(old_key)
                resolved_mappings.append((idx, new_label))

        # Detect duplicate target indices in mapping
        seen_indices: dict[int, str] = {}
        for idx, new_label in resolved_mappings:
            if idx in seen_indices:
                prev_label = seen_indices[idx]
                raise ValueError(
                    "Duplicate channel rename mapping for the same index\n"
                    f"  Channel index: {idx}\n"
                    f"  Original label: '{labels[idx]}'\n"
                    f"  First new label: '{prev_label}'\n"
                    f"  Second new label: '{new_label}'\n"
                    "Provide at most one new label per channel index in mapping."
                )
            seen_indices[idx] = new_label
        # Apply mappings
        for idx, new_label in resolved_mappings:
            new_labels[idx] = new_label

        # Check for duplicate labels after all renames have been applied
        if len(set(new_labels)) != len(new_labels):
            # Identify duplicates for a more informative error
            seen: set[str] = set()
            duplicates: set[str] = set()
            for lbl in new_labels:
                if lbl in seen:
                    duplicates.add(lbl)
                else:
                    seen.add(lbl)
            raise ValueError(
                "Duplicate channel label after rename\n"
                f"  Final labels: {new_labels}\n"
                f"  Duplicates: {sorted(duplicates)}\n"
                "Ensure new channel labels are unique."
            )
        # Create updated channel_metadata list
        new_chmeta = []
        for i, ch_meta in enumerate(self._channel_metadata):
            new_ch_meta = ch_meta.model_copy(deep=True)
            new_ch_meta.label = new_labels[i]
            new_chmeta.append(new_ch_meta)

        if inplace:
            self._channel_metadata = new_chmeta
            return self
        return self._finalize_channel_update(self._data, new_chmeta, inplace=False)

    def _get_dataframe_index(self) -> "pd.Index[Any]":
        """Get time index for DataFrame."""
        return pd.Index(self.time, name="time")
