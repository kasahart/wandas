import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, cast

import dask.array as da
import librosa
import numpy as np
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
        win_length: Optional[int] = None,
        window: str = "hann",
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[list[ChannelMetadata]] = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ) -> None:
        if data.ndim == 2:
            data = da.expand_dims(data, axis=0)  # type: ignore [unused-ignore]
        elif data.ndim != 3:
            raise ValueError(
                f"データは2次元または3次元である必要があります。形状: {data.shape}"
            )
        if not data.shape[-2] == n_fft // 2 + 1:
            raise ValueError(
                f"データの形状が無効です。周波数ビン数は {n_fft // 2 + 1} である必要があります。"  # noqa: E501
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
                    "サンプリングレートが一致していません。演算できません。"
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

            if isinstance(other, (int, float)):
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
        self, plot_type: str = "spectrogram", ax: Optional["Axes"] = None, **kwargs: Any
    ) -> Union["Axes", Iterator["Axes"]]:
        """
        Plot the spectrogram using various visualization strategies.

        Parameters
        ----------
        plot_type : str, default="spectrogram"
            Type of plot to create.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new axes.
        **kwargs : dict
            Additional keyword arguments passed to the plot strategy.
            Common options include:
            - vmin, vmax: Colormap scaling
            - cmap: Colormap name
            - dB: Whether to plot in decibels
            - Aw: Whether to apply A-weighting

        Returns
        -------
        Union[Axes, Iterator[Axes]]
            The matplotlib axes containing the plot, or an iterator of axes
            for multi-plot outputs.
        """
        from wandas.visualization.plotting import create_operation

        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # プロット戦略を取得
        plot_strategy: PlotStrategy[SpectrogramFrame] = create_operation(plot_type)

        # プロット実行
        _ax = plot_strategy.plot(self, ax=ax, **kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def plot_Aw(  # noqa: N802
        self, plot_type: str = "spectrogram", ax: Optional["Axes"] = None, **kwargs: Any
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

        Returns
        -------
        Union[Axes, Iterator[Axes]]
            The matplotlib axes containing the plot.
        """
        return self.plot(plot_type=plot_type, ax=ax, Aw=True, **kwargs)

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
                f"時間インデックス {time_idx} が範囲外です。有効範囲: 0-{self.n_frames - 1}"  # noqa: E501
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
