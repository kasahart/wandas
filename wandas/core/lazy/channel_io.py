import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import dask
import dask.array as da
import numpy as np
import soundfile as sf

from wandas.utils.types import NDArrayReal

from .channel_protocol import ChannelProtocol
from .file_readers import get_file_reader

if TYPE_CHECKING:
    from .channel_frame import ChannelFrame

logger = logging.getLogger(__name__)

dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]
da_from_array = da.from_array  # type: ignore [unused-ignore]


class ChannelIOMixin:
    """
    ChannelFrameクラスにファイル入出力機能を提供するためのMixinクラス。
    このクラスは単体では使用せず、ChannelFrameクラスに継承して使用します。
    """

    def save(
        self: ChannelProtocol, path: Union[str, Path], format: Optional[str] = None
    ) -> None:
        """
        音声データをファイルに保存します。

        Parameters
        ----------
        path : str or Path
            保存先のファイルパス
        format : str, optional
            ファイル形式。None の場合は拡張子から判断
        """
        logger.debug(f"Saving audio data to file: {path} (will compute now)")
        data = self.compute()
        data = data.T
        if data.shape[1] == 1:
            data = data.squeeze(axis=1)
        sf.write(str(path), data, int(self.sampling_rate), format=format)
        logger.debug(f"Save complete: {path}")

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
        ファイルから遅延読み込みでチャネルを作成します。
        様々なファイル形式（WAV、CSV など）を自動的に検出してサポートします。

        Parameters
        ----------
        path : str or Path
            読み込むファイルのパス
        channel : int or list of int, optional
            読み込むチャネル番号。None の場合はすべてのチャネル
        start : float, optional
            読み込み開始位置（秒）。None の場合は先頭から
        end : float, optional
            読み込み終了位置（秒）。None の場合はファイル末尾まで
        chunk_size : int, optional
            処理するチャンクサイズ。遅延処理の分割サイズを指定
        ch_labels : list of str, optional
            各チャネルに設定するラベル
        **kwargs :
            追加のファイル固有パラメータ

        Returns
        -------
        ChannelFrame
            読み込んだデータを含む新しいチャネルフレーム（遅延読み込み）
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
                "duration": frames_to_read / sr,
                "channels": channels_to_load,
                "n_channels_original": n_channels,
            },
        )
        if ch_labels is not None:
            for i, label in enumerate(ch_labels):
                cf.channel[i].label = label

        return cf

    @classmethod
    def read_wav(
        cls, filename: str, ch_labels: Optional[list[str]] = None
    ) -> "ChannelFrame":
        """
        WAVファイルを読み込むユーティリティメソッド

        Parameters
        ----------
        filename : str
            WAVファイルのパス
        ch_labels : list of str, optional
            各チャネルに設定するラベル

        Returns
        -------
        ChannelFrame
            読み込んだデータを含む新しいチャネルフレーム（遅延読み込み）
        """
        from .channel_frame import ChannelFrame

        cf = ChannelFrame.from_file(filename, ch_labels=ch_labels)
        return cf

    @classmethod
    def read_csv(
        cls,
        filename: str,
        ch_labels: Optional[list[str]] = None,
        time_column: Union[int, str] = 0,
        delimiter: str = ",",
        header: Optional[int] = 0,
    ) -> "ChannelFrame":
        """
        CSVファイルを読み込むユーティリティメソッド

        Parameters
        ----------
        filename : str
            CSVファイルのパス
        ch_labels : list of str, optional
            各チャネルに設定するラベル
        time_column : int or str, optional
            時間列のインデックスまたは列名
        delimiter : str, optional
            区切り文字
        header : int, optional
            ヘッダー行の番号

        Returns
        -------
        ChannelFrame
            読み込んだデータを含む新しいチャネルフレーム（遅延読み込み）
        """
        from .channel_frame import ChannelFrame

        cf = ChannelFrame.from_file(
            filename,
            ch_labels=ch_labels,
            time_column=time_column,
            delimiter=delimiter,
            header=header,
        )
        return cf

    @classmethod
    def from_numpy(
        cls,
        data: NDArrayReal,
        sampling_rate: float,
        label: Optional[str] = None,
    ) -> "ChannelFrame":
        """
        NumPy配列からチャネルフレームを作成します。

        Parameters
        ----------
        data : numpy.ndarray
            音声データ。形状は (channels, samples) または (samples,)
        sampling_rate : float
            サンプリングレート (Hz)
        label : str, optional
            チャネルのラベル

        Returns
        -------
        ChannelFrame
            データを含む新しいチャネルフレーム
        """
        from .channel_frame import ChannelFrame

        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(
                f"データは1次元または2次元である必要があります。形状: {data.shape}"
            )

        # NumPy配列をdask配列に変換
        dask_data = da_from_array(data)

        return ChannelFrame(
            data=dask_data,
            sampling_rate=sampling_rate,
            label=label or "numpy_data",
        )

    @classmethod
    def zeros(
        cls,
        n_channels: int,
        n_samples: int,
        sampling_rate: float,
        label: Optional[str] = None,
    ) -> "ChannelFrame":
        """
        ゼロ値を持つチャネルフレームを作成します。

        Parameters
        ----------
        n_channels : int
            チャネル数
        n_samples : int
            サンプル数
        sampling_rate : float
            サンプリングレート (Hz)
        label : str, optional
            チャネルのラベル

        Returns
        -------
        ChannelFrame
            ゼロで初期化された新しいチャネルフレーム
        """
        from .channel_frame import ChannelFrame

        data = da.zeros((n_channels, n_samples), dtype=np.float32)
        return ChannelFrame(
            data=data,
            sampling_rate=sampling_rate,
            label=label or "zeros",
        )

    @classmethod
    def ones(
        cls,
        n_channels: int,
        n_samples: int,
        sampling_rate: float,
        label: Optional[str] = None,
    ) -> "ChannelFrame":
        """
        1の値を持つチャネルフレームを作成します。

        Parameters
        ----------
        n_channels : int
            チャネル数
        n_samples : int
            サンプル数
        sampling_rate : float
            サンプリングレート (Hz)
        label : str, optional
            チャネルのラベル

        Returns
        -------
        ChannelFrame
            1で初期化された新しいチャネルフレーム
        """
        from .channel_frame import ChannelFrame

        data = da.ones((n_channels, n_samples), dtype=np.float32)
        return ChannelFrame(
            data=data,
            sampling_rate=sampling_rate,
            label=label or "ones",
        )
