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

    from .noct_frame import NOctFrame
    from .spectral_frame import SpectralFrame
    from .spectrogram_frame import SpectrogramFrame


from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayReal

from .base_frame import BaseFrame
from .channel_metadata import ChannelMetadata
from .file_readers import get_file_reader
from .plotting import create_operation

logger = logging.getLogger(__name__)

dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]
da_from_array = da.from_array  # type: ignore [unused-ignore]


S = TypeVar("S", bound="BaseFrame[Any]")


class ChannelFrame(BaseFrame[NDArrayReal]):
    """
    Wrapper class for audio channels.
    Data shape: (channels, samples) or (1, samples) for a single channel.
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
        """
        Returns the time data.
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
        from .time_series_operation import create_operation

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
        Common implementation for binary operations - utilizing dask's lazy evaluation

        Parameters
        ----------
        other : ChannelFrame, int, float, ndarray, dask.array
            Right operand for the operation
        op : callable
            Function to execute the operation (e.g., lambda a, b: a + b)
        symbol : str
            Symbolic representation of the operation (e.g., '+')

        Returns
        -------
        ChannelFrame
            A new channel containing the operation result (lazy execution)
        """
        from .channel_frame import ChannelFrame

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

            # dask arrayを直接演算（遅延実行を維持）
            result_data = op(self._data, other._data)

            # チャネルメタデータを結合
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

        # スカラー、NumPy配列、または他のタイプとの演算
        else:
            # dask arrayに直接演算を適用（遅延実行を維持）
            result_data = op(self._data, other)

            # オペランドの表示用文字列
            if isinstance(other, (int, float)):
                other_str = str(other)
            elif isinstance(other, np.ndarray):
                other_str = f"ndarray{other.shape}"
            elif hasattr(other, "shape"):  # dask.array.Arrayのチェック
                other_str = f"dask.array{other.shape}"
            else:
                other_str = str(type(other).__name__)

            # チャネルメタデータを更新
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
        """
        Add another signal or value to the current signal.
        If SNR is specified, performs addition with consideration for
        signal-to-noise ratio.

        Parameters
        ----------
        other : ChannelFrame, int, float, ndarray, dask.array
            Signal or value to add
        snr : float, optional
            Signal-to-noise ratio (dB). If specified, adjusts the scale of the
            other signal based on this SNR.
            self is treated as the signal, and other as the noise.

        Returns
        -------
        ChannelFrame
            A new channel frame containing the addition result (lazy execution)
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
        """
        Various types of plots (using Strategy pattern)

        Parameters
        ----------
        plot_type : str
            Plot type such as 'waveform', 'spectrogram', etc.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new axis
        **kwargs : dict
            Plot-specific parameters
        """

        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # プロット戦略を取得
        plot_strategy = create_operation(plot_type)

        # プロット実行
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
        """
        Generate an RMS plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new axis
        title : str, optional
            Title for the plot
        overlay : bool, default=True
            Whether to overlay the plot on the existing axis
        Aw : bool, optional
            Apply A-weighting.
        **kwargs : dict
            Plot-specific parameters
        """
        kwargs = kwargs or {}
        ylabel = kwargs.pop("ylabel", "RMS")
        rms_ch: ChannelFrame = self.rms_trend(Aw=Aw, dB=True)
        return rms_ch.plot(ax=ax, ylabel=ylabel, title=title, overlay=overlay, **kwargs)

    def describe(self, normalize: bool = True, **kwargs: Any) -> None:
        """
        Describes and displays the audio data with waveform visualization and playback.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to normalize the audio data for playback
        **kwargs : dict
            Additional parameters for visualization
        """
        if "axis_config" in kwargs:
            logger.warning(
                "axis_configは前方互換性のために残されていますが、今後は非推奨となります。"  # noqa: E501
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
                "cbar_configは前方互換性のために残されていますが、今後は非推奨となります。"  # noqa: E501
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
            # displayとAudioの型チェックを無視する
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
        """
        Create a channel frame from a NumPy array.

        Parameters
        ----------
        data : numpy.ndarray
            Audio data. Shape can be:
            (batch, channels, samples) or (channels, samples) or (samples,)
        sampling_rate : float
            Sampling rate (Hz)
        label : str, optional
            Label for the channel
        metadata : dict, optional
            Additional metadata
        ch_labels : list of str, optional
            Labels for each channel
        ch_units : Union[list[str], str], optional
            Units for each channel

        Returns
        -------
        ChannelFrame
            A new channel frame containing the data
        """

        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(
                f"Data must be 1-dimensional or 2-dimensional. Shape: {data.shape}"
            )

        # NumPy配列をdask配列に変換
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
        """
        Create a channel frame from a NumPy array.

        This method is deprecated. Use from_numpy instead.

        Parameters
        ----------
        array : NDArrayReal
            Signal data. Each row corresponds to a channel.
        sampling_rate : int
            Sampling rate (Hz).
        labels : list[str], optional
            Labels for each channel.
        unit : Union[list[str], str], optional
            Unit of the signal.
        frame_label : str, optional
            Label for the frame.
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        ChannelFrame
            A new channel frame containing the data
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
        """
        Creates a channel with lazy loading from a file.
        Automatically detects and supports various file formats (WAV, CSV, etc.).

        Parameters
        ----------
        path : str or Path
            Path to the file to read
        channel : int or list of int, optional
            Channel number(s) to read. If None, all channels are read
        start : float, optional
            Start position in seconds. If None, starts from the beginning
        end : float, optional
            End position in seconds. If None, reads until the end of file
        chunk_size : int, optional
            Chunk size for processing. Specifies the splitting size for lazy processing
        ch_labels : list of str, optional
            Labels to set for each channel
        **kwargs :
            Additional file-specific parameters

        Returns
        -------
        ChannelFrame
            A new channel frame containing the data (lazy loading)
        """
        from .channel_frame import ChannelFrame

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")

        # ファイルリーダー取得
        reader = get_file_reader(path)

        # ファイル情報取得
        info = reader.get_file_info(path, **kwargs)
        sr = info["samplerate"]
        n_channels = info["channels"]
        n_frames = info["frames"]
        ch_labels = ch_labels or info.get("ch_labels", None)

        logger.debug(f"File info: sr={sr}, channels={n_channels}, frames={n_frames}")

        # チャネル選択処理
        all_channels = list(range(n_channels))

        if channel is None:
            channels_to_load = all_channels
            logger.debug(f"Will load all channels: {channels_to_load}")
        elif isinstance(channel, int):
            if channel < 0 or channel >= n_channels:
                raise ValueError(
                    f"チャネル指定が範囲外です: {channel} (有効範囲: 0-{n_channels - 1})"  # noqa: E501
                )
            channels_to_load = [channel]
            logger.debug(f"Will load single channel: {channel}")
        elif isinstance(channel, (list, tuple)):
            for ch in channel:
                if ch < 0 or ch >= n_channels:
                    raise ValueError(
                        f"チャネル指定が範囲外です: {ch} (有効範囲: 0-{n_channels - 1})"
                    )
            channels_to_load = list(channel)
            logger.debug(f"Will load specific channels: {channels_to_load}")
        else:
            raise TypeError("channel は int, list, または None である必要があります")

        # インデックス計算
        start_idx = 0 if start is None else max(0, int(start * sr))
        end_idx = n_frames if end is None else min(n_frames, int(end * sr))
        frames_to_read = end_idx - start_idx

        logger.debug(
            f"Setting up lazy load from file={path}, frames={frames_to_read}, "
            f"start_idx={start_idx}, end_idx={end_idx}"
        )

        # 遅延読み込み用の設定
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
                    "チャネルラベルの数が指定されたチャネル数と一致しません"
                )
            for i in range(len(ch_labels)):
                cf._channel_metadata[i].label = ch_labels[i]
        return cf

    @classmethod
    def read_wav(
        cls, filename: str, labels: Optional[list[str]] = None
    ) -> "ChannelFrame":
        """
        Utility method to read a WAV file.

        Parameters
        ----------
        filename : str
            Path to the WAV file
        labels : list of str, optional
            Labels to set for each channel

        Returns
        -------
        ChannelFrame
            A new channel frame containing the data (lazy loading)
        """
        from .channel_frame import ChannelFrame

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
        """
        Utility method to read a CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV file
        time_column : int or str, optional
            Index or name of the time column
        labels : list of str, optional
            Labels to set for each channel
        delimiter : str, optional
            Delimiter character
        header : int, optional
            Row number to use as header

        Returns
        -------
        ChannelFrame
            A new channel frame containing the data (lazy loading)
        """
        from .channel_frame import ChannelFrame

        cf = ChannelFrame.from_file(
            filename,
            ch_labels=labels,
            time_column=time_column,
            delimiter=delimiter,
            header=header,
        )
        return cf

    def save(self, path: Union[str, Path], format: Optional[str] = None) -> None:
        """
        Save audio data to a file.

        Parameters
        ----------
        path : str or Path
            Path to save the file
        format : str, optional
            File format. If None, determined from file extension
        """
        logger.debug(f"Saving audio data to file: {path} (will compute now)")
        data = self.compute()
        data = data.T
        if data.shape[1] == 1:
            data = data.squeeze(axis=1)
        sf.write(str(path), data, int(self.sampling_rate), format=format)
        logger.debug(f"Save complete: {path}")

    def high_pass_filter(self, cutoff: float, order: int = 4) -> "ChannelFrame":
        """ハイパスフィルターを適用します。"""
        logger.debug(
            f"Setting up highpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        return self.apply_operation("highpass_filter", cutoff=cutoff, order=order)

    def low_pass_filter(self, cutoff: float, order: int = 4) -> "ChannelFrame":
        """ローパスフィルターを適用します。"""
        logger.debug(
            f"Setting up lowpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        return self.apply_operation("lowpass_filter", cutoff=cutoff, order=order)

    def normalize(
        self, target_level: float = -20, channel_wise: bool = True
    ) -> "ChannelFrame":
        """信号レベルを正規化します。"""
        logger.debug(
            f"Setting up normalize: target_level={target_level}, channel_wise={channel_wise} (lazy)"  # noqa: E501
        )
        return self.apply_operation(
            "normalize", target_level=target_level, channel_wise=channel_wise
        )

    def a_weighting(self) -> "ChannelFrame":
        """A加重フィルタを適用します。"""
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
        """HPSS（Harmonic-Percussive Source Separation）の調波成分を抽出します。"""
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
        """HPSS（Harmonic-Percussive Source Separation）の打撃音成分を抽出します。"""
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
        音声データをリサンプリングします。

        Parameters
        ----------
        target_sr : float
            目標サンプリングレート (Hz)
        resample_type : str, optional
            リサンプリング方法 ('linear', 'sinc', 'fft'など)
        window : str, optional
            窓関数の種類 ('hann', 'hamming'など)
        **kwargs : dict
            追加のリサンプリングパラメータ

        Returns
        -------
        ChannelFrame
            リサンプリングされたチャネルフレーム
        """
        return self.apply_operation(
            "resampling",
            target_sr=target_sr,
            **kwargs,
        )

    def abs(self) -> "ChannelFrame":
        """絶対値を計算します。"""
        return self.apply_operation("abs")

    def power(self, exponent: float) -> "ChannelFrame":
        """べき乗計算を行います。"""
        return self.apply_operation("power", exponent=exponent)

    def trim(
        self,
        start: float = 0,
        end: Optional[float] = None,
    ) -> "ChannelFrame":
        """
        チャネルをトリミングします。

        Parameters
        ----------
        start : float, optional
            トリミング開始位置（秒）。Noneの場合は先頭から
        end : float, optional
            トリミング終了位置（秒）。Noneの場合はファイル末尾まで

        Returns
        -------
        ChannelFrame
            トリミングされたチャネルフレーム
        """
        if end is None:
            end = self.duration
        if start > end:
            raise ValueError("start must be less than end")
        # トリミング操作を適用
        return self.apply_operation("trim", start=start, end=end)

    def rms_trend(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        dB: bool = False,  # noqa: N803
        Aw: bool = False,  # noqa: N803
    ) -> "ChannelFrame":
        """RMSトレンドを計算します。"""
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
        """合計値を計算します。"""
        return self.apply_operation("sum")

    def mean(self) -> "ChannelFrame":
        """平均値を計算します。"""
        return self.apply_operation("mean")

    def channel_difference(self, other_channel: Union[int, str] = 0) -> "ChannelFrame":
        """チャンネル間の差分を計算します。"""
        if isinstance(other_channel, str):
            return self.apply_operation(
                "channel_difference", other_channel=self.label2index(other_channel)
            )
        return self.apply_operation("channel_difference", other_channel=other_channel)

    def fft(self, n_fft: Optional[int] = None, window: str = "hann") -> "SpectralFrame":
        """時間領域データから周波数領域データへ変換（FFT）"""
        from .spectral_frame import SpectralFrame
        from .time_series_operation import FFT

        params = {"n_fft": n_fft, "window": window}
        operation_name = "fft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from .time_series_operation import create_operation

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("FFT", operation)
        # データに処理を適用
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
        """
        Estimate power spectral density using Welch's method.
        Parameters
        ----------
        n_fft : int, optional
            FFT size. If None, defaults to 2048
        hop_length : int, optional
            Hop length. If None, defaults to n_fft // 4
        win_length : int, default=2048
            Window length. If None, defaults to n_fft
        window : str, default="hann"
            Window function type
        average : str, default="mean"
            Averaging method. Options: "mean", "median"

        Returns
        -------
        SpectralFrame
            A new spectral frame containing the power spectral density
        """
        from .spectral_frame import SpectralFrame
        from .time_series_operation import Welch

        params = dict(
            n_fft=n_fft or win_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            average=average,
        )
        operation_name = "welch"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from .time_series_operation import create_operation

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("Welch", operation)
        # データに処理を適用
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
        """ノクターナルスペクトルを計算します。"""

        from .noct_frame import NOctFrame
        from .time_series_operation import NOctSpectrum

        params = {"fmin": fmin, "fmax": fmax, "n": n, "G": G, "fr": fr}
        operation_name = "noct_spectrum"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from .time_series_operation import create_operation

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("NOctSpectrum", operation)
        # データに処理を適用
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
        """
        短時間フーリエ変換（STFT）を計算し、時間-周波数領域のスペクトログラムを返します。

        Parameters
        ----------
        n_fft : int, default=2048
            FFTのサンプル数
        hop_length : int, optional
            フレーム間のホップ長。指定がない場合は n_fft//4
        win_length : int, optional
            各フレームの窓長。指定がない場合は n_fft
        window : str, default="hann"
            窓関数の種類
        center : bool, default=True
            信号の中心化を行うかどうか

        Returns
        -------
        SpectrogramFrame
            スペクトログラムデータを含むSpectrogramFrameオブジェクト
        """
        from .spectrogram_frame import SpectrogramFrame
        from .time_series_operation import STFT, create_operation

        # ホップ長とウィンドウ長の設定
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

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("STFT", operation)

        # データに処理を適用
        spectrogram_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectrogramFrame with operation {operation_name} added to graph"  # noqa: E501
        )

        # 新しいインスタンスを作成
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

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        ChannelFrame に必要な追加の初期化引数を提供します。
        """
        return {}
