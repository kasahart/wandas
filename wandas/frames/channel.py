import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from dask.array.core import Array as DaArray
from IPython.display import Audio, display
from matplotlib.axes import Axes

from wandas.utils.types import NDArrayReal

from ..core.base_frame import BaseFrame
from ..core.metadata import ChannelMetadata
from ..io.readers import get_file_reader
from ..visualization.plotting import create_operation
from .mixins import ChannelProcessingMixin, ChannelTransformMixin

logger = logging.getLogger(__name__)

dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]
da_from_array = da.from_array  # type: ignore [unused-ignore]


S = TypeVar("S", bound="BaseFrame[Any]")


class ChannelFrame(
    BaseFrame[NDArrayReal], ChannelProcessingMixin, ChannelTransformMixin
):
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
        other: Union["ChannelFrame", int, float, NDArrayReal],
        snr: Optional[float] = 1,
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
                    "Sampling rates do not match. Cannot perform operation."
                )

        elif isinstance(other, np.ndarray):
            other = ChannelFrame.from_numpy(
                other, self.sampling_rate, label="array_data"
            )
        elif isinstance(other, (int, float)):
            return self + other
        else:
            raise TypeError(
                "Addition target with SNR must be a ChannelFrame or "
                f"NumPy array: {type(other)}"
            )

        # If SNR is specified, adjust the length of the other signal
        if other.duration != self.duration:
            other = other.fix_length(length=self.n_samples)

        if snr is None:
            return self + other

        return self.apply_operation("add_with_snr", other=other._data, snr=snr)

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

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """Provide additional initialization arguments required for ChannelFrame."""
        return {}
