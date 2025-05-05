import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, cast

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from IPython.display import Audio, display
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from librosa._typing import (
        _FloatLike_co,
        _IntLike_co,
        _PadModeSTFT,
        _WindowSpec,
    )

    from .noct import NOctFrame
    from .spectral import SpectralFrame
    from .spectrogram import SpectrogramFrame


from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayReal

from ..core.base_frame import BaseFrame
from ..core.metadata import ChannelMetadata
from ..io.readers import get_file_reader
from ..visualization.plotting import create_operation

logger = logging.getLogger(__name__)

dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]
da_from_array = da.from_array  # type: ignore [unused-ignore]


S = TypeVar("S", bound="BaseFrame[Any]")


class ChannelFrame(BaseFrame[NDArrayReal]):
    """Channel-based data frame for handling audio signals and time series data.

    This frame represents channel-based data such as audio signals and time series data,
    with each channel containing data samples in the time domain.
    """

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[list[ChannelMetadata]] = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ) -> None:
        """Initialize a ChannelFrame.

        Args:
            data: Dask array containing channel data.
            Shape should be (n_channels, n_samples).
            sampling_rate: The sampling rate of the data in Hz.
            label: A label for the frame.
            metadata: Optional metadata dictionary.
            operation_history: History of operations applied to the frame.
            channel_metadata: Metadata for each channel.
            previous: Reference to the previous frame in the processing chain.

        Raises:
            ValueError: If data has more than 2 dimensions.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(
                f"Data must be 1-dimensional or 2-dimensional. Shape: {data.shape}"
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

    @property
    def _n_channels(self) -> int:
        """Returns the number of channels."""
        return self.shape[-2]

    @property
    def time(self) -> NDArrayReal:
        """Get time array for the signal.

        Returns:
            Array of time points in seconds.
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

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
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
        if operation_name == "resampling":
            # For resampling, update sampling rate
            return self._create_new_instance(
                sampling_rate=params["target_sr"],
                data=processed_data,
                metadata=new_metadata,
                operation_history=new_history,
            )
        return self._create_new_instance(
            data=processed_data,
            metadata=new_metadata,
            operation_history=new_history,
        )

    def _binary_op(
        self,
        other: Union["ChannelFrame", int, float, NDArrayReal, "DaArray"],
        op: Callable[["DaArray", Any], "DaArray"],
        symbol: str,
    ) -> "ChannelFrame":
        """
        Common implementation for binary operations
        - utilizing dask's lazy evaluation.

        Args:
            other: Right operand for the operation.
            op: Function to execute the operation (e.g., lambda a, b: a + b).
            symbol: Symbolic representation of the operation (e.g., '+').

        Returns:
            A new channel containing the operation result (lazy execution).
        """
        from .channel import ChannelFrame

        logger.debug(f"Setting up {symbol} operation (lazy)")

        # Handle potentially None metadata and operation_history
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        # Check if other is a ChannelFrame - improved type checking
        if isinstance(other, ChannelFrame):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    "Sampling rates do not match. Cannot perform operation."
                )

            # Perform operation directly on dask array (maintaining lazy execution)
            result_data = op(self._data, other._data)

            # Merge channel metadata
            merged_channel_metadata = []
            for self_ch, other_ch in zip(
                self._channel_metadata, other._channel_metadata
            ):
                ch = self_ch.model_copy(deep=True)
                ch["label"] = f"({self_ch['label']} {symbol} {other_ch['label']})"
                merged_channel_metadata.append(ch)

            operation_history.append({"operation": symbol, "with": other.label})

            return ChannelFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                label=f"({self.label} {symbol} {other.label})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=merged_channel_metadata,
                previous=self,
            )

        # Perform operation with scalar, NumPy array, or other types
        else:
            # Apply operation directly on dask array (maintaining lazy execution)
            result_data = op(self._data, other)

            # Operand display string
            if isinstance(other, (int, float)):
                other_str = str(other)
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

            return ChannelFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                label=f"({self.label} {symbol} {other_str})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=updated_channel_metadata,
                previous=self,
            )

    def add(
        self,
        other: Union["ChannelFrame", int, float, NDArrayReal, "DaArray"],
        snr: Optional[float] = None,
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

        # Special processing when SNR is specified
        if snr is not None:
            # First convert other to ChannelFrame if it's not
            if not isinstance(other, ChannelFrame):
                if isinstance(other, np.ndarray):
                    other = ChannelFrame.from_numpy(
                        other, self.sampling_rate, label="array_data"
                    )
                elif isinstance(other, (int, float)):
                    # For scalar values, simply add (ignore SNR)
                    return self + other
                else:
                    raise TypeError(
                        "Addition target with SNR must be a ChannelFrame or "
                        f"NumPy array: {type(other)}"
                    )

            # Check if sampling rates match
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    "Sampling rates do not match. Cannot perform operation."
                )

            # Apply addition operation with SNR adjustment
            return self.apply_operation("add_with_snr", other=other._data, snr=snr)

        # Execute normal addition if SNR is not specified
        return self + other

    def plot(
        self, plot_type: str = "waveform", ax: Optional["Axes"] = None, **kwargs: Any
    ) -> Union["Axes", Iterator["Axes"]]:
        """Plot the frame data.

        Args:
            plot_type: Type of plot. Default is "waveform".
            ax: Optional matplotlib axes for plotting.
            **kwargs: Additional arguments passed to the plot function.

        Returns:
            Single Axes object or iterator of Axes objects.
        """
        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # Get plot strategy
        plot_strategy = create_operation(plot_type)

        # Execute plot
        _ax = plot_strategy.plot(self, ax=ax, **kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def rms_plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = True,
        Aw: bool = False,  # noqa: N803
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """Generate an RMS plot.

        Args:
            ax: Optional matplotlib axes for plotting.
            title: Title for the plot.
            overlay: Whether to overlay the plot on the existing axis.
            Aw: Apply A-weighting.
            **kwargs: Additional arguments passed to the plot function.

        Returns:
            Single Axes object or iterator of Axes objects.
        """
        kwargs = kwargs or {}
        ylabel = kwargs.pop("ylabel", "RMS")
        rms_ch: ChannelFrame = self.rms_trend(Aw=Aw, dB=True)
        return rms_ch.plot(ax=ax, ylabel=ylabel, title=title, overlay=overlay, **kwargs)

    def describe(self, normalize: bool = True, **kwargs: Any) -> None:
        """Display visual and audio representation of the frame.

        Args:
            normalize: Whether to normalize the audio data for playback.
            **kwargs: Additional parameters for visualization.
        """
        if "axis_config" in kwargs:
            logger.warning(
                "axis_config is retained for backward compatibility but will be deprecated in the future."  # noqa: E501
            )
            axis_config = kwargs["axis_config"]
            if "time_plot" in axis_config:
                kwargs["waveform"] = axis_config["time_plot"]
            if "freq_plot" in axis_config:
                if "xlim" in axis_config["freq_plot"]:
                    vlim = axis_config["freq_plot"]["xlim"]
                    kwargs["vmin"] = vlim[0]
                    kwargs["vmax"] = vlim[1]
                if "ylim" in axis_config["freq_plot"]:
                    ylim = axis_config["freq_plot"]["ylim"]
                    kwargs["ylim"] = ylim

        if "cbar_config" in kwargs:
            logger.warning(
                "cbar_config is retained for backward compatibility but will be deprecated in the future."  # noqa: E501
            )
            cbar_config = kwargs["cbar_config"]
            if "vmin" in cbar_config:
                kwargs["vmin"] = cbar_config["vmin"]
            if "vmax" in cbar_config:
                kwargs["vmax"] = cbar_config["vmax"]

        for ch in self:
            ax: Axes
            _ax = ch.plot("describe", title=f"{ch.label} {ch.labels[0]}", **kwargs)
            if isinstance(_ax, Iterator):
                ax = next(iter(_ax))
            elif isinstance(_ax, Axes):
                ax = _ax
            else:
                raise TypeError(
                    f"Unexpected type for plot result: {type(_ax)}. Expected Axes or Iterator[Axes]."  # noqa: E501
                )
            # Ignore type checks for display and Audio
            display(ax.figure)  # type: ignore
            plt.close(ax.figure)  # type: ignore
            display(Audio(ch.data, rate=ch.sampling_rate, normalize=normalize))  # type: ignore

    @classmethod
    def from_numpy(
        cls,
        data: NDArrayReal,
        sampling_rate: float,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        ch_labels: Optional[list[str]] = None,
        ch_units: Optional[Union[list[str], str]] = None,
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
            raise ValueError(
                f"Data must be 1-dimensional or 2-dimensional. Shape: {data.shape}"
            )

        # Convert NumPy array to dask array
        dask_data = da_from_array(data)
        cf = ChannelFrame(
            data=dask_data,
            sampling_rate=sampling_rate,
            label=label or "numpy_data",
        )
        if metadata is not None:
            cf.metadata = metadata
        if ch_labels is not None:
            if len(ch_labels) != cf.n_channels:
                raise ValueError(
                    "Number of channel labels does not match the number of channels"
                )
            for i in range(len(ch_labels)):
                cf._channel_metadata[i].label = ch_labels[i]
        if ch_units is not None:
            if isinstance(ch_units, str):
                ch_units = [ch_units] * cf.n_channels

            if len(ch_units) != cf.n_channels:
                raise ValueError(
                    "Number of channel units does not match the number of channels"
                )
            for i in range(len(ch_units)):
                cf._channel_metadata[i].unit = ch_units[i]

        return cf

    @classmethod
    def from_ndarray(
        cls,
        array: NDArrayReal,
        sampling_rate: float,
        labels: Optional[list[str]] = None,
        unit: Optional[Union[list[str], str]] = None,
        frame_label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
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
        # Redirect to from_numpy for compatibility
        # However, from_ndarray is deprecated
        logger.warning("from_ndarray is deprecated. Use from_numpy instead.")
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
        path: Union[str, Path],
        channel: Optional[Union[int, list[int]]] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        chunk_size: Optional[int] = None,
        ch_labels: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "ChannelFrame":
        """Create a ChannelFrame from an audio file.

        Args:
            path: Path to the audio file.
            channel: Channel(s) to load.
            start: Start time in seconds.
            end: End time in seconds.
            chunk_size: Chunk size for processing.
            Specifies the splitting size for lazy processing.
            ch_labels: Labels for each channel.
            **kwargs: Additional arguments passed to the file reader.

        Returns:
            A new ChannelFrame containing the loaded audio data.

        Raises:
            ValueError: If channel selection is invalid.
            TypeError: If channel parameter type is invalid.
            FileNotFoundError: If the file doesn't exist.
        """
        from .channel import ChannelFrame

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Get file reader
        reader = get_file_reader(path)

        # Get file info
        info = reader.get_file_info(path, **kwargs)
        sr = info["samplerate"]
        n_channels = info["channels"]
        n_frames = info["frames"]
        ch_labels = ch_labels or info.get("ch_labels", None)

        logger.debug(f"File info: sr={sr}, channels={n_channels}, frames={n_frames}")

        # Channel selection processing
        all_channels = list(range(n_channels))

        if channel is None:
            channels_to_load = all_channels
            logger.debug(f"Will load all channels: {channels_to_load}")
        elif isinstance(channel, int):
            if channel < 0 or channel >= n_channels:
                raise ValueError(
                    f"Channel specification is out of range: {channel} (valid range: 0-{n_channels - 1})"  # noqa: E501
                )
            channels_to_load = [channel]
            logger.debug(f"Will load single channel: {channel}")
        elif isinstance(channel, (list, tuple)):
            for ch in channel:
                if ch < 0 or ch >= n_channels:
                    raise ValueError(
                        f"Channel specification is out of range: {ch} (valid range: 0-{n_channels - 1})"  # noqa: E501
                    )
            channels_to_load = list(channel)
            logger.debug(f"Will load specific channels: {channels_to_load}")
        else:
            raise TypeError("channel must be int, list, or None")

        # Index calculation
        start_idx = 0 if start is None else max(0, int(start * sr))
        end_idx = n_frames if end is None else min(n_frames, int(end * sr))
        frames_to_read = end_idx - start_idx

        logger.debug(
            f"Setting up lazy load from file={path}, frames={frames_to_read}, "
            f"start_idx={start_idx}, end_idx={end_idx}"
        )

        # Settings for lazy loading
        expected_shape = (len(channels_to_load), frames_to_read)

        # Define the loading function using the file reader
        def _load_audio() -> NDArrayReal:
            logger.debug(">>> EXECUTING DELAYED LOAD <<<")
            # Use the reader to get audio data with parameters
            out = reader.get_data(path, channels_to_load, start_idx, frames_to_read)
            if not isinstance(out, np.ndarray):
                raise ValueError("Unexpected data type after reading file")
            return out

        logger.debug(
            f"Creating delayed dask task with expected shape: {expected_shape}"
        )

        # Create delayed operation
        delayed_data = dask_delayed(_load_audio)()
        logger.debug("Wrapping delayed function in dask array")

        # Create dask array from delayed computation
        dask_array = da_from_delayed(
            delayed_data, shape=expected_shape, dtype=np.float32
        )

        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError("Chunk size must be a positive integer")
            logger.debug(f"Setting chunk size: {chunk_size} for sample axis")
            dask_array = dask_array.rechunk({0: -1, 1: chunk_size})

        logger.debug(
            "ChannelFrame setup complete - actual file reading will occur on compute()"  # noqa: E501
        )

        cf = ChannelFrame(
            data=dask_array,
            sampling_rate=sr,
            label=path.stem,
            metadata={
                "filename": str(path),
            },
        )
        if ch_labels is not None:
            if len(ch_labels) != len(cf):
                raise ValueError(
                    "Number of channel labels does not match the number of specified channels"  # noqa: E501
                )
            for i in range(len(ch_labels)):
                cf._channel_metadata[i].label = ch_labels[i]
        return cf

    @classmethod
    def read_wav(
        cls, filename: str, labels: Optional[list[str]] = None
    ) -> "ChannelFrame":
        """Utility method to read a WAV file.

        Args:
            filename: Path to the WAV file.
            labels: Labels to set for each channel.

        Returns:
            A new ChannelFrame containing the data (lazy loading).
        """
        from .channel import ChannelFrame

        cf = ChannelFrame.from_file(filename, ch_labels=labels)
        return cf

    @classmethod
    def read_csv(
        cls,
        filename: str,
        time_column: Union[int, str] = 0,
        labels: Optional[list[str]] = None,
        delimiter: str = ",",
        header: Optional[int] = 0,
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

    def save(self, path: Union[str, Path], format: Optional[str] = None) -> None:
        """Save the audio data to a file.

        Args:
            path: Path to save the file.
            format: File format. If None, determined from file extension.
        """
        logger.debug(f"Saving audio data to file: {path} (will compute now)")
        data = self.compute()
        data = data.T
        if data.shape[1] == 1:
            data = data.squeeze(axis=1)
        sf.write(str(path), data, int(self.sampling_rate), format=format)
        logger.debug(f"Save complete: {path}")

    def high_pass_filter(self, cutoff: float, order: int = 4) -> "ChannelFrame":
        """Apply a high-pass filter to the signal.

        Args:
            cutoff: The cutoff frequency of the filter in Hz.
            order: The order of the filter. Default is 4.

        Returns:
            A new ChannelFrame with the filtered signal.
        """
        logger.debug(
            f"Setting up highpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        return self.apply_operation("highpass_filter", cutoff=cutoff, order=order)

    def low_pass_filter(self, cutoff: float, order: int = 4) -> "ChannelFrame":
        """Apply a low-pass filter to the signal.

        Args:
            cutoff: The cutoff frequency of the filter in Hz.
            order: The order of the filter. Default is 4.

        Returns:
            A new ChannelFrame with the filtered signal.
        """
        logger.debug(
            f"Setting up lowpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        return self.apply_operation("lowpass_filter", cutoff=cutoff, order=order)

    def normalize(
        self, target_level: float = -20, channel_wise: bool = True
    ) -> "ChannelFrame":
        """Normalize the signal level.

        This method adjusts the signal amplitude to reach a target RMS level.

        Args:
            target_level: Target RMS level in dB. Default is -20.
            channel_wise: If True, normalize each channel independently.
                If False, apply the same scaling to all channels.

        Returns:
            A new ChannelFrame containing the normalized signal.
        """
        logger.debug(
            f"Setting up normalize: target_level={target_level}, channel_wise={channel_wise} (lazy)"  # noqa: E501
        )
        return self.apply_operation(
            "normalize", target_level=target_level, channel_wise=channel_wise
        )

    def a_weighting(self) -> "ChannelFrame":
        """Apply A-weighting filter to the signal.

        A-weighting adjusts the frequency response to approximate human hearing
        perception, following IEC 61672-1:2013 standard.

        Returns:
            A new ChannelFrame with A-weighted signal.
        """
        return self.apply_operation("a_weighting")

    def hpss_harmonic(
        self,
        kernel_size: Union[
            "_IntLike_co", tuple["_IntLike_co", "_IntLike_co"], list["_IntLike_co"]
        ] = 31,
        power: float = 2,
        margin: Union[
            "_FloatLike_co",
            tuple["_FloatLike_co", "_FloatLike_co"],
            list["_FloatLike_co"],
        ] = 1,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "_WindowSpec" = "hann",
        center: bool = True,
        pad_mode: "_PadModeSTFT" = "constant",
    ) -> "ChannelFrame":
        """
        Extract harmonic components using HPSS
         (Harmonic-Percussive Source Separation).
        """
        return self.apply_operation(
            "hpss_harmonic",
            kernel_size=kernel_size,
            power=power,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )

    def hpss_percussive(
        self,
        kernel_size: Union[
            "_IntLike_co", tuple["_IntLike_co", "_IntLike_co"], list["_IntLike_co"]
        ] = 31,
        power: float = 2,
        margin: Union[
            "_FloatLike_co",
            tuple["_FloatLike_co", "_FloatLike_co"],
            list["_FloatLike_co"],
        ] = 1,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "_WindowSpec" = "hann",
        center: bool = True,
        pad_mode: "_PadModeSTFT" = "constant",
    ) -> "ChannelFrame":
        """
        Extract percussive components using HPSS
        (Harmonic-Percussive Source Separation).

        This method separates the percussive (tonal) components from the signal.

        Args:
            kernel_size: Median filter size for HPSS.
            power: Exponent for the Weiner filter used in HPSS.
            margin: Margin size for the separation.

        Returns:
            A new ChannelFrame containing the harmonic components.
        """
        return self.apply_operation(
            "hpss_percussive",
            kernel_size=kernel_size,
            power=power,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )

    def resampling(
        self,
        target_sr: float,
        **kwargs: Any,
    ) -> "ChannelFrame":
        """
        Resample audio data.

        Parameters
        ----------
        target_sr : float
            Target sampling rate (Hz)
        resample_type : str, optional
            Resampling method ('soxr_hq', 'linear', 'sinc', 'fft', etc.)
        **kwargs : dict
            Additional resampling parameters

        Returns
        -------
        ChannelFrame
            Resampled channel frame
        """
        return self.apply_operation(
            "resampling",
            target_sr=target_sr,
            **kwargs,
        )

    def abs(self) -> "ChannelFrame":
        """Calculate the absolute value of the signal.

        Returns:
            A new ChannelFrame containing the absolute values.
        """
        return self.apply_operation("abs")

    def power(self, exponent: float = 2.0) -> "ChannelFrame":
        """Calculate the power of the signal.

        Args:
            exponent: The exponent to raise the signal to. Default is 2.0.

        Returns:
            A new ChannelFrame containing the signal raised to the power.
        """
        return self.apply_operation("power", exponent=exponent)

    def trim(
        self,
        start: float = 0,
        end: Optional[float] = None,
    ) -> "ChannelFrame":
        """Trim the signal to specified time range.

        Args:
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            A new ChannelFrame with trimmed signal.

        Raises:
            ValueError: If end time is before start time.
        """
        if end is None:
            end = self.duration
        if start > end:
            raise ValueError("start must be less than end")
        # Apply trim operation
        return self.apply_operation("trim", start=start, end=end)

    def rms_trend(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        dB: bool = False,  # noqa: N803
        Aw: bool = False,  # noqa: N803
    ) -> "ChannelFrame":
        """Calculate the RMS trend of the signal.

        This method computes the root mean square value over sliding windows.

        Args:
            frame_length: The size of the sliding window in samples. Default is 2048.
            hop_length: The hop length between windows in samples. Default is 512.
            dB: Whether to return the RMS values in decibels. Default is False.
            Aw: Whether to apply A-weighting. Default is False.

        Returns:
            A new ChannelFrame containing the RMS trend.
        """
        cf = self.apply_operation(
            "rms_trend",
            frame_length=frame_length,
            hop_length=hop_length,
            ref=[ch.ref for ch in self._channel_metadata],
            dB=dB,
            Aw=Aw,
        )
        cf.sampling_rate = self.sampling_rate / hop_length
        return cf

    def sum(self) -> "ChannelFrame":
        """Sum all channels.

        Returns:
            A new ChannelFrame with summed signal.
        """
        return self.apply_operation("sum")

    def mean(self) -> "ChannelFrame":
        """Average all channels.

        Returns:
            A new ChannelFrame with averaged signal.
        """
        return self.apply_operation("mean")

    def channel_difference(self, other_channel: Union[int, str] = 0) -> "ChannelFrame":
        """Calculate channel differences relative to a reference channel.

        Args:
            other_channel: Reference channel index or label. Default is 0.

        Returns:
            A new ChannelFrame with channel differences.
        """
        if isinstance(other_channel, str):
            return self.apply_operation(
                "channel_difference", other_channel=self.label2index(other_channel)
            )
        return self.apply_operation("channel_difference", other_channel=other_channel)

    def fft(self, n_fft: Optional[int] = None, window: str = "hann") -> "SpectralFrame":
        """Compute Fast Fourier Transform.

        Args:
            n_fft: Number of FFT points. Default is next power of 2 of data length.
            window: Window type. Default is "hann".

        Returns:
            A SpectralFrame containing the FFT results.
        """
        from ..processing import FFT
        from .spectral import SpectralFrame

        params = {"n_fft": n_fft, "window": window}
        operation_name = "fft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from ..processing import create_operation

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("FFT", operation)
        # Apply processing to data
        spectrum_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        if n_fft is None:
            is_even = spectrum_data.shape[-1] % 2 == 0
            _n_fft = (
                spectrum_data.shape[-1] * 2 - 2
                if is_even
                else spectrum_data.shape[-1] * 2 - 1
            )
        else:
            _n_fft = n_fft

        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=_n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata={**self.metadata, "window": window, "n_fft": _n_fft},
            operation_history=[
                *self.operation_history,
                {"operation": "fft", "params": {"n_fft": _n_fft, "window": window}},
            ],
            channel_metadata=self._channel_metadata,
            previous=self,
        )

    def welch(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        average: str = "mean",
    ) -> "SpectralFrame":
        """Compute power spectral density using Welch's method.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between successive frames.
            Default is n_fft//4.
            win_length: Length of window. Default is n_fft.
            window: Window type. Default is "hann".
            average: Method for averaging segments. Default is "mean".

        Returns:
            A SpectralFrame containing the power spectral density.
        """
        from ..processing import Welch
        from .spectral import SpectralFrame

        params = dict(
            n_fft=n_fft or win_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            average=average,
        )
        operation_name = "welch"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from ..processing import create_operation

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("Welch", operation)
        # Apply processing to data
        spectrum_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {"operation": "welch", "params": params},
            ],
            channel_metadata=self._channel_metadata,
            previous=self,
        )

    def noct_spectrum(
        self,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> "NOctFrame":
        """Compute N-octave band spectrum.

        Args:
            fmin: Minimum center frequency in Hz. Default is 20 Hz.
            fmax: Maximum center frequency in Hz. Default is 20000 Hz.
            n: Band division (1 for octave, 3 for 1/3 octave). Default is 3.
            G: Reference gain in dB. Default is 10 dB.
            fr: Reference frequency in Hz. Default is 1000 Hz.

        Returns:
            A NOctFrame containing the N-octave band spectrum.
        """
        from ..processing import NOctSpectrum
        from .noct import NOctFrame

        params = {"fmin": fmin, "fmax": fmax, "n": n, "G": G, "fr": fr}
        operation_name = "noct_spectrum"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from ..processing import create_operation

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("NOctSpectrum", operation)
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
                    "operation": "noct_spectrum",
                    "params": params,
                },
            ],
            channel_metadata=self._channel_metadata,
            previous=self,
        )

    def stft(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
    ) -> "SpectrogramFrame":
        """Compute Short-Time Fourier Transform.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between successive frames.
            Default is n_fft//4.
            win_length: Length of window. Default is n_fft.
            window: Window type. Default is "hann".

        Returns:
            A SpectrogramFrame containing the STFT results.
        """
        from ..processing import STFT, create_operation
        from .spectrogram import SpectrogramFrame

        # Set hop length and window length
        _hop_length = hop_length if hop_length is not None else n_fft // 4
        _win_length = win_length if win_length is not None else n_fft

        params = {
            "n_fft": n_fft,
            "hop_length": _hop_length,
            "win_length": _win_length,
            "window": window,
        }
        operation_name = "stft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("STFT", operation)

        # Apply processing to data
        spectrogram_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectrogramFrame with operation {operation_name} added to graph"  # noqa: E501
        )

        # Create new instance
        return SpectrogramFrame(
            data=spectrogram_data,
            sampling_rate=self.sampling_rate,
            n_fft=n_fft,
            hop_length=_hop_length,
            win_length=_win_length,
            window=window,
            label=f"stft({self.label})",
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=self._channel_metadata,
        )

    def coherence(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        detrend: str = "constant",
    ) -> "SpectralFrame":
        """Compute magnitude squared coherence.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between successive frames.
            Default is n_fft//4.
            win_length: Length of window. Default is n_fft.
            window: Window type. Default is "hann".
            detrend: Detrending method. Options: "constant", "linear", None.

        Returns:
            A SpectralFrame containing the magnitude squared coherence.
        """
        from ..processing import Coherence, create_operation
        from .spectral import SpectralFrame

        params = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": window,
            "detrend": detrend,
        }
        operation_name = "coherence"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("Coherence", operation)

        # Apply processing to data
        coherence_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"  # noqa: E501
        )
        # Create new channel metadata
        channel_metadata = []
        for in_ch in self._channel_metadata:
            for out_ch in self._channel_metadata:
                meta = ChannelMetadata()
                meta.label = f"$\\gamma_{{{in_ch.label}, {out_ch.label}}}$"
                meta.unit = ""
                meta.ref = 1
                meta["metadata"] = dict(
                    in_ch=in_ch["metadata"], out_ch=out_ch["metadata"]
                )
                channel_metadata.append(meta)

        # Create new instance
        return SpectralFrame(
            data=coherence_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"Coherence of {self.label}",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {"operation": operation_name, "params": params},
            ],
            channel_metadata=channel_metadata,
            previous=self,
        )

    def csd(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "SpectralFrame":
        """Compute cross-spectral density matrix.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between successive frames.
            Default is n_fft//4.
            win_length: Length of window. Default is n_fft.
            window: Window type. Default is "hann".
            detrend: Detrending method. Options: "constant", "linear", None.
            scaling: Scaling method. Options: "spectrum", "density".
            average: Method for averaging segments. Default is "mean".

        Returns:
            A SpectralFrame containing the cross-spectral density matrix.
        """
        from ..processing import CSD, create_operation
        from .spectral import SpectralFrame

        params = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": window,
            "detrend": detrend,
            "scaling": scaling,
            "average": average,
        }
        operation_name = "csd"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("CSD", operation)

        # Apply processing to data
        csd_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"  # noqa: E501
        )
        # Create new channel metadata
        channel_metadata = []
        for in_ch in self._channel_metadata:
            for out_ch in self._channel_metadata:
                meta = ChannelMetadata()
                meta.label = f"{operation_name}({in_ch.label}, {out_ch.label})"
                meta.unit = ""
                meta.ref = 1
                meta["metadata"] = dict(
                    in_ch=in_ch["metadata"], out_ch=out_ch["metadata"]
                )
                channel_metadata.append(meta)

        # Create new instance
        return SpectralFrame(
            data=csd_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"$C_{{{in_ch.label}, {out_ch.label}}}$",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {"operation": operation_name, "params": params},
            ],
            channel_metadata=channel_metadata,
            previous=self,
        )

    def transfer_function(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "SpectralFrame":
        """Compute transfer function matrix.

        The transfer function characterizes the signal transmission properties
        between channels in the frequency domain, representing the input-output
        relationship of a system.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between successive frames.
            Default is n_fft//4.
            win_length: Length of window. Default is n_fft.
            window: Window type. Default is "hann".
            detrend: Detrending method. Options: "constant", "linear", None.
            scaling: Scaling method. Options: "spectrum", "density".
            average: Method for averaging segments. Default is "mean".

        Returns:
            A SpectralFrame containing the transfer function matrix.
        """
        from ..processing import TransferFunction, create_operation
        from .spectral import SpectralFrame

        params = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": window,
            "detrend": detrend,
            "scaling": scaling,
            "average": average,
        }
        operation_name = "transfer_function"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("TransferFunction", operation)

        # Apply processing to data
        tf_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"  # noqa: E501
        )
        # Create new channel metadata
        channel_metadata = []
        for in_ch in self._channel_metadata:
            for out_ch in self._channel_metadata:
                meta = ChannelMetadata()
                meta.label = f"$H_{{{in_ch.label}, {out_ch.label}}}$"
                meta.unit = ""
                meta.ref = 1
                meta["metadata"] = dict(
                    in_ch=in_ch["metadata"], out_ch=out_ch["metadata"]
                )
                channel_metadata.append(meta)

        # Create new instance
        return SpectralFrame(
            data=tf_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"Transfer function of {self.label}",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {"operation": operation_name, "params": params},
            ],
            channel_metadata=channel_metadata,
            previous=self,
        )

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """Provide additional initialization arguments required for ChannelFrame."""
        return {}
