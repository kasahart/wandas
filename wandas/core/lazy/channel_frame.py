import copy
import logging
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# Add missing imports
import soundfile as sf
from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayReal

from .file_readers import get_file_reader

# 新しいモジュールをインポート
from .plotting import get_plot_strategy

logger = logging.getLogger(__name__)

dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]


class ChannelMetadata:
    """単一チャネルのメタデータにアクセスするためのクラス"""

    def __init__(self, owner: "ChannelFrame", channel_idx: int):
        self._owner = owner
        self._channel_idx = channel_idx

    @property
    def label(self) -> str:
        """チャネルのラベルを取得します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        metadata = self._owner._channel_metadata.get(self._channel_idx, {})
        _label: str = metadata.get(
            "label", f"{self._owner.label}_ch{self._channel_idx}"
        )
        return _label

    @label.setter
    def label(self, value: str) -> None:
        """チャネルのラベルを設定します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        if self._channel_idx not in self._owner._channel_metadata:
            self._owner._channel_metadata[self._channel_idx] = {}
        self._owner._channel_metadata[self._channel_idx]["label"] = value

    @property
    def unit(self) -> str:
        """チャネルの単位を取得します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        metadata = self._owner._channel_metadata.get(self._channel_idx, {})
        _unit: str = metadata.get("unit", "")
        return _unit

    @unit.setter
    def unit(self, value: str) -> None:
        """チャネルの単位を設定します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        if self._channel_idx not in self._owner._channel_metadata:
            self._owner._channel_metadata[self._channel_idx] = {}
        self._owner._channel_metadata[self._channel_idx]["unit"] = value

    def __getitem__(self, key: str) -> Any:
        """任意のメタデータ項目を取得します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        metadata = self._owner._channel_metadata.get(self._channel_idx, {})
        return metadata.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """任意のメタデータ項目を設定します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        if self._channel_idx not in self._owner._channel_metadata:
            self._owner._channel_metadata[self._channel_idx] = {}
        self._owner._channel_metadata[self._channel_idx][key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """指定したキーのメタデータを取得し、ない場合はデフォルト値を返します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        metadata = self._owner._channel_metadata.get(self._channel_idx, {})
        return metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """指定したキーのメタデータを設定します。"""
        self.__setitem__(key, value)

    def all(self) -> dict[str, Any]:
        """チャネルのすべてのメタデータを辞書として返します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        return self._owner._channel_metadata.get(self._channel_idx, {}).copy()


class ChannelMetadataCollection:
    """すべてのチャネルのメタデータにアクセスするためのクラス"""

    def __init__(self, owner: "ChannelFrame"):
        self._owner = owner

    def __getitem__(self, channel_idx: int) -> ChannelMetadata:
        """指定したチャネルのメタデータにアクセスするためのオブジェクトを返します。"""
        self._owner._validate_channel_idx(channel_idx)
        return ChannelMetadata(self._owner, channel_idx)

    def get_all(self) -> dict[int, dict[str, Any]]:
        """すべてのチャネルのメタデータを辞書として返します。"""
        return copy.deepcopy(self._owner._channel_metadata)

    def set_all(self, metadata: dict[int, dict[str, Any]]) -> None:
        """すべてのチャネルのメタデータを設定します。"""
        valid_indices = {idx for idx in metadata if 0 <= idx < self._owner.n_channels}
        for idx in valid_indices:
            self._owner._channel_metadata[idx] = copy.deepcopy(metadata[idx])


class ChannelFrame:
    """
    音声チャネルのラッパークラス
    データ形状: (channels, samples) または単一チャネルの場合 (1, samples)
    """

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[dict[int, dict[str, Any]]] = None,
    ):
        """
        AudioChannelの初期化

        Parameters
        ----------
        data : dask.array.Array
            音声データ。形状は (channels, samples) または (1, samples)
        sampling_rate : float
            サンプリングレート (Hz)
        label : str, optional
            チャネルのラベル
        metadata : dict, optional
            メタデータ辞書
        operation_history : list, optional
            適用された操作の履歴
        channel_metadata : dict, optional
            チャネルごとのメタデータ。キーはチャネルインデックス、値はメタデータ辞書
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(
                f"データは1次元または2次元である必要があります。形状: {data.shape}"
            )

        self._data = data
        self.sampling_rate = sampling_rate
        self.label = label or "unnamed_frame"
        self.metadata = metadata or {}
        self.operation_history = operation_history or []

        # チャネルごとのメタデータを初期化
        self._channel_metadata: dict[int, dict[str, Any]] = {}

        # メタデータアクセサーを初期化
        self._metadata_accessor = ChannelMetadataCollection(self)

        # 提供されたチャネルメタデータがあれば使用、
        # なければデフォルト値を各チャネルに設定
        if channel_metadata:
            self._channel_metadata = channel_metadata.copy()

        # 各チャネルにメタデータがあることを確認（なければデフォルト値を設定）
        for i in range(self.n_channels):
            if i not in self._channel_metadata:
                self._channel_metadata[i] = {}

            # ラベルがなければデフォルトを設定
            if "label" not in self._channel_metadata[i]:
                if self.n_channels == 1:
                    self._channel_metadata[i]["label"] = self.label
                else:
                    self._channel_metadata[i]["label"] = f"{self.label}_ch{i}"

            # 単位がなければデフォルトを設定
            if "unit" not in self._channel_metadata[i]:
                self._channel_metadata[i]["unit"] = ""

        try:
            # 新しいdaskバージョンでの情報表示
            logger.debug(f"Dask graph layers: {list(self._data.dask.layers.keys())}")
            logger.debug(
                f"Dask graph dependencies: {len(self._data.dask.dependencies)}"
            )
        except Exception as e:
            logger.debug(f"Dask graph visualization details unavailable: {e}")

    @property
    def time(self) -> NDArrayReal:
        """
        時刻データを返します。
        """
        return np.arange(self.n_samples) / self.sampling_rate

    @property
    def channel(self) -> ChannelMetadataCollection:
        """チャネルのメタデータにアクセスするためのプロパティ。"""
        return self._metadata_accessor

    @property
    def data(self) -> NDArrayReal:
        """
        ユーザーから見ると普通の NumPy 配列ですが、内部では遅延計算されています。
        初めてアクセスしたときに計算が実行されます。
        """
        return self.compute()

    @property
    def n_channels(self) -> int:
        n: int = self._data.shape[-2]
        return n

    @property
    def n_samples(self) -> int:
        n: int = self._data.shape[-1]
        return n

    @property
    def duration(self) -> float:
        return self.n_samples / self.sampling_rate

    @property
    def shape(self) -> tuple[int, int]:
        _shape: tuple[int, int] = self._data.shape
        return _shape

    @property
    def labels(self) -> list[str]:
        """すべてのチャネルのラベルをリストとして取得します。"""
        return [self.channel[i].label for i in range(self.n_channels)]

    def compute(self) -> NDArrayReal:
        logger.debug(
            "COMPUTING DASK ARRAY - This will trigger file reading and all processing"
        )
        result = self._data.compute()

        if not isinstance(result, np.ndarray):
            raise ValueError(f"計算結果がnp.ndarrayではありません: {type(result)}")

        logger.debug(f"Computation complete, result shape: {result.shape}")
        return result

    def persist(self) -> "ChannelFrame":
        logger.debug("Persisting data in memory for reuse.")
        persisted_data = self._data.persist()
        return ChannelFrame(
            data=persisted_data,
            sampling_rate=self.sampling_rate,
            label=self.label,
            metadata=self.metadata.copy(),
            operation_history=self.operation_history.copy(),
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def get_channel(self, channel_idx: int) -> "ChannelFrame":
        if channel_idx < 0 or channel_idx >= self.n_channels:
            range_max = self.n_channels - 1
            raise ValueError(
                f"チャネル指定が範囲外です: {channel_idx} (有効範囲: 0-{range_max})"
            )
        logger.debug(f"Extracting channel index={channel_idx} (lazy operation).")
        channel_data = self._data[channel_idx : channel_idx + 1]

        # チャネル固有のメタデータを取得
        channel_label = self.channel[channel_idx].label
        channel_metadata = {**self.metadata, "channel_idx": channel_idx}

        # チャネル固有のメタデータを新しいChannelFrameに渡す
        new_channel_metadata = {0: self._channel_metadata[channel_idx].copy()}

        return ChannelFrame(
            data=channel_data,
            sampling_rate=self.sampling_rate,
            label=channel_label,
            metadata=channel_metadata,
            operation_history=self.operation_history.copy(),
            channel_metadata=new_channel_metadata,
        )

    def save(self, path: Union[str, Path], format: Optional[str] = None) -> None:
        logger.debug(f"Saving audio data to file: {path} (will compute now)")
        data = self.compute()
        data = data.T
        if data.shape[1] == 1:
            data = data.squeeze(axis=1)
        sf.write(str(path), data, int(self.sampling_rate), format=format)
        logger.debug(f"Save complete: {path}")

    def apply_operation(self, operation_name: str, **params: Any) -> "ChannelFrame":
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from .time_series_operation import create_operation

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)

        # データに処理を適用
        processed_data = operation.process(self._data)

        # メタデータ更新
        operation_metadata = {"operation": operation_name, "params": params}
        new_history = self.operation_history.copy()
        new_history.append(operation_metadata)
        new_metadata = {**self.metadata}
        new_metadata[operation_name] = params

        logger.debug(
            f"Created new ChannelFrame with operation {operation_name} added to graph"
        )
        return ChannelFrame(
            data=processed_data,
            sampling_rate=self.sampling_rate,
            label=self.label,
            metadata=new_metadata,
            operation_history=new_history,
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def _binary_op(
        self,
        other: Union["ChannelFrame", int, float, NDArrayReal, DaArray],
        op: Callable[[DaArray, Any], DaArray],
        symbol: str,
    ) -> "ChannelFrame":
        """
        二項演算の共通実装 - daskの遅延演算を活用

        Parameters
        ----------
        other : LazyAudioChannel, int, float, ndarray, dask.array
            演算の右オペランド
        op : callable
            演算を実行する関数 (例: lambda a, b: a + b)
        symbol : str
            演算のシンボル表現 (例: '+')

        Returns
        -------
        LazyAudioChannel
            演算結果を含む新しいチャネル（遅延実行）
        """
        logger.debug(f"Setting up {symbol} operation (lazy)")

        # LazyAudioChannel同士の演算
        if isinstance(other, ChannelFrame):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    "サンプリングレートが一致していません。演算できません。"
                )

            # dask arrayを直接演算（遅延実行を維持）
            result_data = op(self._data, other._data)

            # チャネルメタデータを結合
            merged_channel_metadata = self._metadata_accessor.get_all()
            # 同じインデックスのチャネルについては、演算結果を示すラベルを作成
            for idx in merged_channel_metadata:
                if idx in other._channel_metadata:
                    self_label = self.channel[idx].label
                    other_label = other.channel[idx].label
                    merged_channel_metadata[idx] = copy.deepcopy(
                        merged_channel_metadata[idx]
                    )
                    merged_channel_metadata[idx]["label"] = (
                        f"({self_label} {symbol} {other_label})"
                    )

            return ChannelFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                label=f"({self.label} {symbol} {other.label})",
                metadata=self.metadata.copy(),
                operation_history=self.operation_history
                + [{"operation": symbol, "with": other.label}],
                channel_metadata=merged_channel_metadata,
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
            elif isinstance(other, DaArray):
                other_str = f"dask.array{other.shape}"
            else:
                other_str = str(type(other).__name__)

            # チャネルメタデータを更新
            updated_channel_metadata = {}
            for idx in self._channel_metadata:
                updated_channel_metadata[idx] = self._channel_metadata[idx].copy()
                updated_channel_metadata[idx]["label"] = (
                    f"({self.channel[idx].label} {symbol} {other_str})"
                )

            return ChannelFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                label=f"({self.label} {symbol} {other_str})",
                metadata=self.metadata.copy(),
                operation_history=self.operation_history
                + [{"operation": symbol, "with": other_str}],
                channel_metadata=updated_channel_metadata,
            )

    # forでループを回すためのメソッド
    def __iter__(self) -> Iterator["ChannelFrame"]:
        for idx in range(self.n_channels):
            yield self.get_channel(idx)

    def __array__(self, dtype: npt.DTypeLike = None) -> NDArrayReal:
        """
        NumPy の ndarray への暗黙的な変換をサポート
        ユーザーが np.array(channel) や他の NumPy 操作を実行すると、
        自動的に compute されます
        """
        result = self.compute()
        if dtype is not None:
            return result.astype(dtype)
        return result

    # NumPy 配列のように振る舞うための追加メソッド
    def __getitem__(self, key: Union[str, int]) -> Union["ChannelFrame", NDArrayReal]:
        """
        インデックス操作をサポート（遅延実行を維持）
        """
        # インデックス操作をサポート（遅延実行を維持）
        if isinstance(key, tuple) and len(key) > 2:
            raise IndexError("インデックスの次元が多すぎます")

        new_data = self._data[key]

        # インデックス操作の結果が単一の要素の場合は値を返す（自動計算）
        if isinstance(new_data, DaArray) and new_data.ndim == 0:
            new_data = new_data.compute()
            if isinstance(new_data, np.ndarray):
                return new_data
            else:
                raise ValueError(
                    f"計算結果がnp.ndarrayではありません: {type(new_data)}"
                )

        # それ以外は新しい LazyAudioChannel を返す
        if isinstance(new_data, DaArray):
            return ChannelFrame(
                data=new_data,
                sampling_rate=self.sampling_rate,
                label=f"{self.label}[{key}]",
                metadata=self.metadata.copy(),
                operation_history=self.operation_history.copy(),
                channel_metadata=self._metadata_accessor.get_all(),
            )

        # スカラー値など、配列でない場合はそのまま返す
        raise ValueError(f"インデックス操作の結果が不明: {type(new_data)}")

    # 二項演算子のシンプルな実装 - _binary_opを使用
    def __add__(
        self, other: Union["ChannelFrame", int, float, NDArrayReal]
    ) -> "ChannelFrame":
        """加算演算子（遅延実行）"""
        return self._binary_op(other, lambda a, b: a + b, "+")

    def __sub__(
        self, other: Union["ChannelFrame", int, float, NDArrayReal]
    ) -> "ChannelFrame":
        """減算演算子（遅延実行）"""
        return self._binary_op(other, lambda a, b: a - b, "-")

    def __mul__(
        self, other: Union["ChannelFrame", int, float, NDArrayReal]
    ) -> "ChannelFrame":
        """乗算演算子（遅延実行）"""
        return self._binary_op(other, lambda a, b: a * b, "*")

    def __truediv__(
        self, other: Union["ChannelFrame", int, float, NDArrayReal]
    ) -> "ChannelFrame":
        """除算演算子（遅延実行）"""
        return self._binary_op(other, lambda a, b: a / b, "/")

    # その他の有用な演算メソッド
    def abs(self) -> "ChannelFrame":
        """絶対値を計算（遅延実行）"""
        logger.debug("Setting up absolute value operation (lazy)")
        result_data = abs(self._data)

        return ChannelFrame(
            data=result_data,
            sampling_rate=self.sampling_rate,
            label=f"abs({self.label})",
            metadata=self.metadata.copy(),
            operation_history=self.operation_history + [{"operation": "abs"}],
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def power(self, exponent: float) -> "ChannelFrame":
        """べき乗計算（遅延実行）"""
        logger.debug(f"Setting up power operation with exponent={exponent} (lazy)")
        result_data = self._data**exponent

        return ChannelFrame(
            data=result_data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label})^{exponent}",
            metadata=self.metadata.copy(),
            operation_history=self.operation_history
            + [{"operation": "power", "exponent": exponent}],
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def sum(self) -> "ChannelFrame":
        """すべてのチャンネルを合計します。"""
        summed_data = self._data.sum(axis=0, keepdims=True)
        return ChannelFrame(
            data=summed_data,
            sampling_rate=self.sampling_rate,
            label=f"sum({self.label})",
            metadata={**self.metadata},
            operation_history=self.operation_history + [{"operation": "sum"}],
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def mean(self) -> "ChannelFrame":
        """すべてのチャンネルの平均を計算します。"""
        mean_data = self._data.mean(axis=0, keepdims=True)
        return ChannelFrame(
            data=mean_data,
            sampling_rate=self.sampling_rate,
            label=f"mean({self.label})",
            metadata={**self.metadata},
            operation_history=self.operation_history + [{"operation": "mean"}],
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def channel_difference(self, other_channel: int = 0) -> "ChannelFrame":
        """チャンネル間の差分を計算します。"""
        if other_channel < 0 or other_channel >= self.n_channels:
            raise ValueError(
                f"チャネル指定が範囲外です: {other_channel} "
                f"(有効範囲: 0-{self.n_channels - 1})"
            )

        # 基準チャネルのデータを取得
        ref_channel_data = self._data[other_channel : other_channel + 1]

        # 全チャネルから基準チャネルを引く
        diff_data = self._data - ref_channel_data

        return ChannelFrame(
            data=diff_data,
            sampling_rate=self.sampling_rate,
            label=f"(ch[*] - ch[{other_channel}])",
            metadata={**self.metadata},
            operation_history=self.operation_history
            + [{"operation": "channel_difference", "reference": other_channel}],
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def highpass_filter(self, cutoff: float, order: int = 4) -> "ChannelFrame":
        logger.debug(
            f"Setting up highpas1s filter: cutoff={cutoff}, order={order} (lazy)"
        )
        return self.apply_operation("highpass_filter", cutoff=cutoff, order=order)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> "ChannelFrame":
        logger.debug(
            f"Setting up lowpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        return self.apply_operation("lowpass_filter", cutoff=cutoff, order=order)

    def normalize(
        self, target_level: float = -20, channel_wise: bool = True
    ) -> "ChannelFrame":
        logger.debug(
            f"Setting up normalize: target_level={target_level}, channel_wise={channel_wise} (lazy)"  # noqa: E501
        )
        return self.apply_operation(
            "normalize", target_level=target_level, channel_wise=channel_wise
        )

    def a_weighting(self) -> "ChannelFrame":
        """A加重フィルタを適用します。"""
        return self.apply_operation("a_weighting")

    def hpss_harmonic(self, **kwargs: Any) -> "ChannelFrame":
        """HPSS（Harmonic-Percussive Source Separation）の調波成分を抽出します。"""
        return self.apply_operation("hpss_harmonic", **kwargs)

    def hpss_percussive(self, **kwargs: Any) -> "ChannelFrame":
        """HPSS（Harmonic-Percussive Source Separation）の打撃音成分を抽出します。"""
        return self.apply_operation("hpss_percussive", **kwargs)

    def visualize_graph(self, filename: Optional[str] = None) -> Optional[str]:
        """計算グラフを可視化してファイルに保存"""
        try:
            # daskのvisualize機能を使用
            filename = filename or f"audio_graph_{uuid.uuid4().hex[:8]}.png"
            logger.debug(f"Visualizing computation graph to file: {filename}")
            self._data.visualize(filename=filename)
            logger.debug(f"Graph visualization saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to visualize graph: {str(e)}")
            return None

    def plot(
        self, plot_type: str = "waveform", ax: Optional["Axes"] = None, **kwargs: Any
    ) -> Union["Axes", Iterator["Axes"]]:
        """
        様々な形式のプロット (Strategyパターンを使用)

        Parameters
        ----------
        plot_type : str
            'waveform', 'spectrogram'などのプロット種類
        ax : matplotlib.axes.Axes, optional
            プロット先の軸。Noneの場合は新しい軸を作成
        **kwargs : dict
            プロット固有のパラメータ
        """

        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # プロット戦略を取得
        plot_strategy = get_plot_strategy(plot_type)

        # プロット実行
        _ax = plot_strategy.plot(self, ax=ax, **kwargs)

        logger.debug("Plot rendering complete")

        return _ax

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
        ファイルから遅延読み込みでチャネルを作成
        Factory パターンを使用して異なるファイルフォーマットをサポート
        """
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

        # Use reader's get_audio_data method with delayed execution
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
            "LazyAudioChannel setup complete - actual file reading will occur on compute()"  # noqa: E501
        )

        cf = cls(
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
        """
        cf = cls.from_file(filename, ch_labels=ch_labels)

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
        WAVファイルを読み込むユーティリティメソッド
        """

        cf = cls.from_file(
            filename,
            ch_labels=ch_labels,
            time_column=time_column,
            delimiter=delimiter,
            header=header,
        )

        return cf

    def debug_info(self) -> None:
        """詳細なデバッグ情報を出力"""
        logger.debug("=== LazyAudioChannel Debug Info ===")
        logger.debug(f"Label: {self.label}")
        logger.debug(f"Shape: {self.shape}")
        logger.debug(f"Sampling rate: {self.sampling_rate} Hz")
        logger.debug(f"Duration: {self.duration:.2f} seconds")
        logger.debug(f"Channels: {self.n_channels}")
        logger.debug(f"Samples: {self.n_samples}")
        logger.debug(f"Dask chunks: {self._data.chunks}")
        logger.debug(f"Operation history: {len(self.operation_history)} operations")
        for i, op in enumerate(self.operation_history):
            logger.debug(f"  Operation {i + 1}: {op}")
        logger.debug(f"Metadata: {self.metadata}")
        logger.debug("=== End Debug Info ===")

    def _validate_channel_idx(self, channel_idx: int) -> None:
        """チャネルインデックスが有効かどうかを確認します。"""
        if not isinstance(channel_idx, int):
            raise TypeError(
                f"チャネルインデックスは整数である必要があります: {type(channel_idx)}"
            )
        if channel_idx < 0 or channel_idx >= self.n_channels:
            raise ValueError(
                f"チャネル指定が範囲外です: {channel_idx} (有効範囲: 0-{self.n_channels - 1})"  # noqa: E501
            )
