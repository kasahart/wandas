import copy
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np

from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from dask.array.core import Array as DaArray

    from .channel_frame import ChannelFrame
    from .metadata import ChannelMetadataCollection

logger = logging.getLogger(__name__)


class ChannelOperationsMixin:
    """
    ChannelFrameクラスに演算機能を提供するためのMixinクラス。
    このクラスは単体では使用せず、ChannelFrameクラスに継承して使用します。
    """

    _data: "DaArray"
    sampling_rate: float
    label: Optional[str]
    metadata: Optional[dict[str, Any]]
    operation_history: Optional[list[dict[str, Any]]]
    channel_metadata: Optional[dict[int, dict[str, Any]]]
    _metadata_accessor: "ChannelMetadataCollection"
    _channel_metadata: dict[int, dict[str, Any]]

    @property
    def n_channels(self) -> int:
        # Missing return statement
        raise NotImplementedError("Subclasses must implement n_channels property")

    @property
    def channel(self) -> "ChannelMetadataCollection":
        # Missing return statement
        raise NotImplementedError("Subclasses must implement channel property")

    def compute(self) -> NDArrayReal:
        # Missing return statement
        raise NotImplementedError("Subclasses must implement compute method")

    def _validate_channel_idx(self, channel_idx: int) -> None:
        # Missing return statement
        raise NotImplementedError(
            "Subclasses must implement _validate_channel_idx method"
        )

    def apply_operation(self, operation_name: str, **params: Any) -> "ChannelFrame":
        raise NotImplementedError("Subclasses must implement apply_operation method")

    def _binary_op(
        self,
        other: Union["ChannelFrame", int, float, NDArrayReal, "DaArray"],
        op: Callable[["DaArray", Any], "DaArray"],
        symbol: str,
    ) -> "ChannelFrame":
        """
        二項演算の共通実装 - daskの遅延演算を活用

        Parameters
        ----------
        other : ChannelFrame, int, float, ndarray, dask.array
            演算の右オペランド
        op : callable
            演算を実行する関数 (例: lambda a, b: a + b)
        symbol : str
            演算のシンボル表現 (例: '+')

        Returns
        -------
        ChannelFrame
            演算結果を含む新しいチャネル（遅延実行）
        """
        from .channel_frame import ChannelFrame

        logger.debug(f"Setting up {symbol} operation (lazy)")

        # Check if other is a ChannelFrame - improved type checking
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

            # Handle potentially None metadata and operation_history
            metadata = {}
            if self.metadata is not None:
                metadata = self.metadata.copy()

            operation_history = []
            if self.operation_history is not None:
                operation_history = self.operation_history.copy()

            operation_history.append({"operation": symbol, "with": other.label})

            return ChannelFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                label=f"({self.label} {symbol} {other.label})",
                metadata=metadata,
                operation_history=operation_history,
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
            elif hasattr(other, "shape"):  # dask.array.Arrayのチェック
                other_str = f"dask.array{other.shape}"
            else:
                other_str = str(type(other).__name__)

            # チャネルメタデータを更新
            updated_channel_metadata = {}
            for idx in self._channel_metadata or {}:
                updated_channel_metadata[idx] = self._channel_metadata[idx].copy()
                updated_channel_metadata[idx]["label"] = (
                    f"({self.channel[idx].label} {symbol} {other_str})"
                )

            # Handle potentially None metadata and operation_history
            metadata = {}
            if self.metadata is not None:
                metadata = self.metadata.copy()

            operation_history = []
            if self.operation_history is not None:
                operation_history = self.operation_history.copy()

            operation_history.append({"operation": symbol, "with": other_str})

            return ChannelFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                label=f"({self.label} {symbol} {other_str})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=updated_channel_metadata,
            )

    def __add__(
        self, other: Union["ChannelFrame", int, float, NDArrayReal, "DaArray"]
    ) -> "ChannelFrame":
        """加算演算子（遅延実行）"""
        return self._binary_op(other, lambda a, b: a + b, "+")

    def __sub__(
        self, other: Union["ChannelFrame", int, float, NDArrayReal, "DaArray"]
    ) -> "ChannelFrame":
        """減算演算子（遅延実行）"""
        return self._binary_op(other, lambda a, b: a - b, "-")

    def __mul__(
        self, other: Union["ChannelFrame", int, float, NDArrayReal, "DaArray"]
    ) -> "ChannelFrame":
        """乗算演算子（遅延実行）"""
        return self._binary_op(other, lambda a, b: a * b, "*")

    def __truediv__(
        self, other: Union["ChannelFrame", int, float, NDArrayReal, "DaArray"]
    ) -> "ChannelFrame":
        """除算演算子（遅延実行）"""
        return self._binary_op(other, lambda a, b: a / b, "/")

    def abs(self) -> "ChannelFrame":
        """絶対値を計算（遅延実行）"""
        from .channel_frame import ChannelFrame

        logger.debug("Setting up absolute value operation (lazy)")
        result_data = abs(self._data)

        # Handle potentially None metadata and operation_history
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        operation_history.append({"operation": "abs"})

        return ChannelFrame(
            data=result_data,
            sampling_rate=self.sampling_rate,
            label=f"abs({self.label})",
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def power(self, exponent: float) -> "ChannelFrame":
        """べき乗計算（遅延実行）"""
        from .channel_frame import ChannelFrame

        logger.debug(f"Setting up power operation with exponent={exponent} (lazy)")
        result_data = self._data**exponent

        # Handle potentially None metadata and operation_history
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        operation_history.append({"operation": "power", "exponent": exponent})

        return ChannelFrame(
            data=result_data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label})^{exponent}",
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def sum(self) -> "ChannelFrame":
        """すべてのチャンネルを合計します。"""
        from .channel_frame import ChannelFrame

        summed_data = self._data.sum(axis=0, keepdims=True)

        # Handle potentially None metadata and operation_history
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        operation_history.append({"operation": "sum"})

        return ChannelFrame(
            data=summed_data,
            sampling_rate=self.sampling_rate,
            label=f"sum({self.label})",
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def mean(self) -> "ChannelFrame":
        """すべてのチャンネルの平均を計算します。"""
        from .channel_frame import ChannelFrame

        mean_data = self._data.mean(axis=0, keepdims=True)

        # Handle potentially None metadata and operation_history
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        operation_history.append({"operation": "mean"})

        return ChannelFrame(
            data=mean_data,
            sampling_rate=self.sampling_rate,
            label=f"mean({self.label})",
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def channel_difference(self, other_channel: int = 0) -> "ChannelFrame":
        """チャンネル間の差分を計算します。"""
        from .channel_frame import ChannelFrame

        if other_channel < 0 or other_channel >= self.n_channels:
            raise ValueError(
                f"チャネル指定が範囲外です: {other_channel} "
                f"(有効範囲: 0-{self.n_channels - 1})"
            )

        # 基準チャネルのデータを取得
        ref_channel_data = self._data[other_channel : other_channel + 1]

        # 全チャネルから基準チャネルを引く
        diff_data = self._data - ref_channel_data

        # Handle potentially None metadata and operation_history
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        operation_history.append(
            {"operation": "channel_difference", "reference": other_channel}
        )

        return ChannelFrame(
            data=diff_data,
            sampling_rate=self.sampling_rate,
            label=f"(ch[*] - ch[{other_channel}])",
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=self._metadata_accessor.get_all(),
        )

    def highpass_filter(self, cutoff: float, order: int = 4) -> "ChannelFrame":
        """ハイパスフィルターを適用します。"""
        logger.debug(
            f"Setting up highpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        return self.apply_operation("highpass_filter", cutoff=cutoff, order=order)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> "ChannelFrame":
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

    def hpss_harmonic(self, **kwargs: Any) -> "ChannelFrame":
        """HPSS（Harmonic-Percussive Source Separation）の調波成分を抽出します。"""
        return self.apply_operation("hpss_harmonic", **kwargs)

    def hpss_percussive(self, **kwargs: Any) -> "ChannelFrame":
        """HPSS（Harmonic-Percussive Source Separation）の打撃音成分を抽出します。"""
        return self.apply_operation("hpss_percussive", **kwargs)
