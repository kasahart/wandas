import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union, cast

import dask.array as da
import librosa
import numpy as np
import pandas as pd
from dask.array.core import Array as DaArray

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata
from wandas.utils.types import NDArrayComplex, NDArrayReal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.frames.channel import ChannelFrame
    from wandas.frames.spectral import SpectralFrame
    from wandas.visualization.plotting import PlotStrategy

logger = logging.getLogger(__name__)

S = TypeVar("S", bound="BaseFrame[Any]")


class SpectrogramFrame(BaseFrame[NDArrayComplex]):
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
    operation_history : list[dict], optional
        History of operations performed on this frame.
    channel_metadata : list[ChannelMetadata], optional
        Metadata for each channel in the frame.
    previous : BaseFrame, optional
        The frame that this frame was derived from.

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
    """

    n_fft: int
    hop_length: int
    win_length: int
    window: str

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
        operation_history: list[dict[str, Any]] | None = None,
        channel_metadata: list[ChannelMetadata] | list[dict[str, Any]] | None = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ) -> None:
        if data.ndim == 2:
            data = da.expand_dims(data, axis=0)  # type: ignore [unused-ignore]
        elif data.ndim != 3:
            raise ValueError(
                f"Invalid data dimensions\n"
                f"  Got: {data.ndim}D array with shape {data.shape}\n"
                f"  Expected: 2D or 3D array\n"
                f"Spectrograms require 2D (freq x time) or "
                f"3D (channel x freq x time) data."
            )
        if not data.shape[-2] == n_fft // 2 + 1:
            raise ValueError(
                f"Invalid frequency bin count\n"
                f"  Got: {data.shape[-2]} bins\n"
                f"  Expected: {n_fft // 2 + 1} bins (n_fft={n_fft})\n"
                f"Ensure data shape matches the specified n_fft parameter."
            )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
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
        Get the magnitude spectrogram.

        Returns
        -------
        NDArrayReal
            The absolute values of the complex spectrogram.
        """
        return np.abs(self.data)

    @property
    def phase(self) -> NDArrayReal:
        """
        Get the phase spectrogram.

        Returns
        -------
        NDArrayReal
            The phase angles of the complex spectrogram in radians.
        """
        return np.angle(self.data)

    @property
    def power(self) -> NDArrayReal:
        """
        Get the power spectrogram.

        Returns
        -------
        NDArrayReal
            The squared magnitude of the complex spectrogram.
        """
        return np.abs(self.data) ** 2

    @property
    def dB(self) -> NDArrayReal:  # noqa: N802
        """
        Get the spectrogram in decibels relative to each channel's reference value.

        The reference value for each channel is specified in its metadata.
        A minimum value of -120 dB is enforced to avoid numerical issues.

        Returns
        -------
        NDArrayReal
            The spectrogram in decibels.
        """
        # dB規定値を_channel_metadataから収集
        ref = np.array([ch.ref for ch in self._channel_metadata])
        # dB変換
        # 0除算を避けるために、最大値と1e-12のいずれかを使用
        level: NDArrayReal = 20 * np.log10(
            np.maximum(self.magnitude / ref[..., np.newaxis, np.newaxis], 1e-12)
        )
        return level

    @property
    def dBA(self) -> NDArrayReal:  # noqa: N802
        """
        Get the A-weighted spectrogram in decibels.

        A-weighting applies a frequency-dependent weighting filter that approximates
        the human ear's response. This is particularly useful for analyzing noise
        and acoustic measurements.

        Returns
        -------
        NDArrayReal
            The A-weighted spectrogram in decibels.
        """
        weighted: NDArrayReal = librosa.A_weighting(frequencies=self.freqs, min_db=None)
        return self.dB + weighted[:, np.newaxis]  # 周波数軸に沿ってブロードキャスト

    @property
    def _n_channels(self) -> int:
        """
        Get the number of channels in the data.

        Returns
        -------
        int
            The number of channels.
        """
        return int(self._data.shape[-3])

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
        return np.fft.rfftfreq(self.n_fft, 1.0 / self.sampling_rate)

    @property
    def times(self) -> NDArrayReal:
        """
        Get the time axis values in seconds.

        Returns
        -------
        NDArrayReal
            Array of time values corresponding to each time frame.
        """
        return np.arange(self.n_frames) * self.hop_length / self.sampling_rate

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        """
        Implementation of operation application for spectrogram data.

        This internal method handles the application of various operations to
        spectrogram data, maintaining lazy evaluation through dask.

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
        from wandas.processing import create_operation

        operation = create_operation(operation_name, self.sampling_rate, **params)
        processed_data = operation.process(self._data)

        operation_metadata = {"operation": operation_name, "params": params}
        new_history = self.operation_history.copy()
        new_history.append(operation_metadata)
        new_metadata = {**self.metadata}
        new_metadata[operation_name] = params

        logger.debug(
            f"Created new SpectrogramFrame with operation {operation_name} added to graph"  # noqa: E501
        )
        return self._create_new_instance(
            data=processed_data,
            metadata=new_metadata,
            operation_history=new_history,
        )

    def _binary_op(
        self,
        other: Union[
            "SpectrogramFrame",
            int,
            float,
            complex,
            NDArrayComplex,
            NDArrayReal,
            "DaArray",
        ],
        op: Callable[["DaArray", Any], "DaArray"],
        symbol: str,
    ) -> "SpectrogramFrame":
        """
        Common implementation for binary operations.

        This method handles binary operations between
        SpectrogramFrames and various types
        of operands, maintaining lazy evaluation through dask arrays.

        Parameters
        ----------
        other : Union[SpectrogramFrame, int, float, complex,
            NDArrayComplex, NDArrayReal, DaArray]
            The right operand of the operation.
        op : callable
            Function to execute the operation (e.g., lambda a, b: a + b)
        symbol : str
            String representation of the operation (e.g., '+')

        Returns
        -------
        SpectrogramFrame
            A new SpectrogramFrame containing the result of the operation.

        Raises
        ------
        ValueError
            If attempting to operate with a SpectrogramFrame
            with a different sampling rate.
        """
        logger.debug(f"Setting up {symbol} operation (lazy)")

        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        if isinstance(other, SpectrogramFrame):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    f"Sampling rate mismatch\n"
                    f"  Got: {self.sampling_rate} Hz and {other.sampling_rate} Hz\n"
                    f"  Expected: matching sampling rates\n"
                    f"Resample one frame to match the other before "
                    f"performing operations."
                )

            result_data = op(self._data, other._data)

            merged_channel_metadata = []
            for self_ch, other_ch in zip(
                self._channel_metadata, other._channel_metadata
            ):
                ch = self_ch.model_copy(deep=True)
                ch["label"] = f"({self_ch['label']} {symbol} {other_ch['label']})"
                merged_channel_metadata.append(ch)

            operation_history.append({"operation": symbol, "with": other.label})

            return SpectrogramFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                label=f"({self.label} {symbol} {other.label})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=merged_channel_metadata,
                previous=self,
            )
        else:
            result_data = op(self._data, other)

            if isinstance(other, int | float):
                other_str = str(other)
            elif isinstance(other, complex):
                other_str = f"complex({other.real}, {other.imag})"
            elif isinstance(other, np.ndarray):
                other_str = f"ndarray{other.shape}"
            elif hasattr(other, "shape"):
                other_str = f"dask.array{other.shape}"
            else:
                other_str = str(type(other).__name__)

            updated_channel_metadata: list[ChannelMetadata] = []
            for self_ch in self._channel_metadata:
                ch = self_ch.model_copy(deep=True)
                ch["label"] = f"({self_ch.label} {symbol} {other_str})"
                updated_channel_metadata.append(ch)

            operation_history.append({"operation": symbol, "with": other_str})

            return SpectrogramFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                label=f"({self.label} {symbol} {other_str})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=updated_channel_metadata,
            )

    def plot(
        self,
        plot_type: str = "spectrogram",
        ax: Optional["Axes"] = None,
        title: str | None = None,
        cmap: str = "jet",
        vmin: float | None = None,
        vmax: float | None = None,
        fmin: float = 0,
        fmax: float | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        Aw: bool = False,  # noqa: N803
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
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
        **kwargs : dict
            Additional keyword arguments passed to librosa.display.specshow().

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

        # プロット戦略を取得
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

        # プロット実行
        _ax = plot_strategy.plot(self, ax=ax, **plot_kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def plot_Aw(  # noqa: N802
        self,
        plot_type: str = "spectrogram",
        ax: Optional["Axes"] = None,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
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

    def abs(self) -> "SpectrogramFrame":
        """
        Compute the absolute value (magnitude) of the complex spectrogram.

        This method calculates the magnitude of each complex value in the
        spectrogram, converting the complex-valued data to real-valued magnitude data.
        The result is stored in a new SpectrogramFrame with complex dtype to maintain
        compatibility with other spectrogram operations.

        Returns
        -------
        SpectrogramFrame
            A new SpectrogramFrame containing the magnitude values as complex numbers
            (with zero imaginary parts).

        Examples
        --------
        >>> signal = ChannelFrame.from_wav("audio.wav")
        >>> spectrogram = signal.stft(n_fft=2048, hop_length=512)
        >>> magnitude_spectrogram = spectrogram.abs()
        >>> # The magnitude can be accessed via the magnitude property or data
        >>> print(magnitude_spectrogram.magnitude.shape)
        """
        logger.debug("Computing absolute value (magnitude) of spectrogram")

        # Compute the absolute value using dask for lazy evaluation
        magnitude_data = da.absolute(self._data)

        # Update operation history
        operation_metadata = {"operation": "abs", "params": {}}
        new_history = self.operation_history.copy()
        new_history.append(operation_metadata)
        new_metadata = {**self.metadata}
        new_metadata["abs"] = {}

        logger.debug("Created new SpectrogramFrame with abs operation added to graph")

        return SpectrogramFrame(
            data=magnitude_data,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            label=f"abs({self.label})",
            metadata=new_metadata,
            operation_history=new_history,
            channel_metadata=self._channel_metadata,
            previous=self,
        )

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

        return SpectralFrame(
            data=frame_data,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            window=self.window,
            label=f"{self.label} (Frame {time_idx}, Time {self.times[time_idx]:.3f}s)",
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=self._channel_metadata,
        )

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

        params = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
        }
        operation_name = "istft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("ISTFT", operation)
        # データに処理を適用
        time_series = operation.process(self._data)

        logger.debug(
            f"Created new ChannelFrame with operation {operation_name} added to graph"
        )

        # 新しいインスタンスを作成
        return ChannelFrame(
            data=time_series,
            sampling_rate=self.sampling_rate,
            label=f"istft({self.label})",
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=self._channel_metadata,
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

    def _get_dataframe_columns(self) -> list[str]:
        """Get channel labels as DataFrame columns."""
        return [ch.label for ch in self._channel_metadata]

    def _get_dataframe_index(self) -> "pd.Index[Any]":
        """DataFrame index is not supported for SpectrogramFrame."""
        raise NotImplementedError(
            "DataFrame index is not supported for SpectrogramFrame."
        )

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
        operation_history: list[dict[str, Any]] | None = None,
        channel_metadata: list[ChannelMetadata] | list[dict[str, Any]] | None = None,
        previous: Optional["BaseFrame[Any]"] = None,
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
            operation_history: History of operations applied to the frame.
            channel_metadata: Metadata for each channel.
            previous: Reference to the previous frame in the processing chain.

        Returns:
            A new SpectrogramFrame containing the NumPy data.
        """

        # Normalize shape: support 2D single-channel inputs by expanding
        # to channel-first 3D shape. Reject 1D inputs as invalid for
        # spectrograms.
        if data.ndim == 1:
            raise ValueError(
                f"Invalid data shape\n"
                f"  Got: {data.shape}\n"
                f"  Expected: 2D (freq, time) or 3D (channel, freq, time) array\n"
                f"Provide a 2D or 3D array to represent time-frequency data."
            )
        if data.ndim >= 4:
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
        # Use shared helper to avoid mypy chunking typing issues
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
            operation_history=operation_history,
            channel_metadata=channel_metadata,
            previous=previous,
        )
        return sf
