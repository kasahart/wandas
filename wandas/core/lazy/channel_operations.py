import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

from .channel_metadata import ChannelMetadata

if TYPE_CHECKING:
    from dask.array.core import Array as DaArray

    from .channel_frame import ChannelFrame

logger = logging.getLogger(__name__)


class ChannelOperationsMixin(ABC):
    """
    ChannelFrameクラスに演算機能を提供するためのMixinクラス。
    このクラスは単体では使用せず、ChannelFrameクラスに継承して使用します。
    """

    _data: "DaArray"
    sampling_rate: float
    label: str
    metadata: dict[str, Any]
    operation_history: list[dict[str, Any]]
    _channel_metadata: list[ChannelMetadata]

    @abstractmethod
    def apply_operation(self, operation_name: str, **params: Any) -> "ChannelFrame":
        """操作適用の実装"""
        pass

    @abstractmethod
    def label2index(self, label: str) -> int:
        """ラベルからインデックスへの変換"""
        pass

    # def sum(self) -> "ChannelFrame":
    #     """すべてのチャンネルを合計します。"""
    #     from .channel_frame import ChannelFrame

    #     summed_data = self._data.sum(axis=0, keepdims=True)

    #     # Handle potentially None metadata and operation_history
    #     metadata = {}
    #     if self.metadata is not None:
    #         metadata = self.metadata.copy()

    #     operation_history = []
    #     if self.operation_history is not None:
    #         operation_history = self.operation_history.copy()

    #     operation_history.append({"operation": "sum"})

    #     return ChannelFrame(
    #         data=summed_data,
    #         sampling_rate=self.sampling_rate,
    #         label=f"sum({self.label})",
    #         metadata=metadata,
    #         operation_history=operation_history,
    #         channel_metadata=self._metadata_accessor.get_all(),
    #     )

    # def mean(self) -> "ChannelFrame":
    #     """すべてのチャンネルの平均を計算します。"""
    #     from .channel_frame import ChannelFrame

    #     mean_data = self._data.mean(axis=0, keepdims=True)

    #     # Handle potentially None metadata and operation_history
    #     metadata = {}
    #     if self.metadata is not None:
    #         metadata = self.metadata.copy()

    #     operation_history = []
    #     if self.operation_history is not None:
    #         operation_history = self.operation_history.copy()

    #     operation_history.append({"operation": "mean"})

    #     return ChannelFrame(
    #         data=mean_data,
    #         sampling_rate=self.sampling_rate,
    #         label=f"mean({self.label})",
    #         metadata=metadata,
    #         operation_history=operation_history,
    #         channel_metadata=self._metadata_accessor.get_all(),
    #     )

    # def channel_difference(self, other_channel: int = 0) -> "ChannelFrame":
    #     """チャンネル間の差分を計算します。"""
    #     from .channel_frame import ChannelFrame

    #     if other_channel < 0 or other_channel >= self.n_channels:
    #         raise ValueError(
    #             f"チャネル指定が範囲外です: {other_channel} "
    #             f"(有効範囲: 0-{self.n_channels - 1})"
    #         )

    #     # 基準チャネルのデータを取得
    #     ref_channel_data = self._data[other_channel : other_channel + 1]

    #     # 全チャネルから基準チャネルを引く
    #     diff_data = self._data - ref_channel_data

    #     # Handle potentially None metadata and operation_history
    #     metadata = {}
    #     if self.metadata is not None:
    #         metadata = self.metadata.copy()

    #     operation_history = []
    #     if self.operation_history is not None:
    #         operation_history = self.operation_history.copy()

    #     operation_history.append(
    #         {"operation": "channel_difference", "reference": other_channel}
    #     )

    #     return ChannelFrame(
    #         data=diff_data,
    #         sampling_rate=self.sampling_rate,
    #         label=f"(ch[*] - ch[{other_channel}])",
    #         metadata=metadata,
    #         operation_history=operation_history,
    #         channel_metadata=self._metadata_accessor.get_all(),
    #     )

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

    def abs(self) -> "ChannelFrame":
        """絶対値を計算します。"""
        return self.apply_operation("abs")

    def power(self, exponent: float) -> "ChannelFrame":
        """べき乗計算を行います。"""
        return self.apply_operation("power", exponent=exponent)

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
