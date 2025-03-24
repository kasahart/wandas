from typing import Any, Protocol, runtime_checkable

from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayReal

from .metadata import ChannelMetadataCollection


@runtime_checkable
class ChannelProtocol(Protocol):
    """ChannelFrameクラスとMixinが共有するインターフェース定義"""

    sampling_rate: float
    metadata: dict[str, Any]
    operation_history: list[dict[str, Any]]
    _channel_metadata: dict[int, dict[str, Any]]
    _data: DaArray

    @property
    def n_channels(self) -> int: ...

    @property
    def channel(self) -> "ChannelMetadataCollection": ...

    def compute(self) -> NDArrayReal: ...

    def _validate_channel_idx(self, channel_idx: int) -> None: ...
