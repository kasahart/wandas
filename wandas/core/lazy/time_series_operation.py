import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

import dask.array as da

# import numpy as np
from dask.array.core import Array as DaArray
from scipy import signal

from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)

_da_map_blocks = da.map_blocks  # type: ignore [unused-ignore]


class AudioOperation(ABC):
    """音声処理操作の抽象基底クラス"""

    # クラス変数：操作の名前
    operation_name: ClassVar[str]

    def __init__(self, sampling_rate: float, **params: Any):
        """
        AudioOperationの初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        **params : Any
            操作固有のパラメータ
        """
        self.sampling_rate = sampling_rate
        self.params = params

        # 初期化時にパラメータを検証
        self.validate_params()

        # プロセッサ関数を作成（遅延初期化も可能）
        self._processor_func = self._create_processor()

        logger.debug(
            f"Initialized {self.__class__.__name__} operation with params: {params}"
        )

    def validate_params(self) -> None:
        """パラメータの検証（無効な場合は例外を発生）"""
        pass

    @abstractmethod
    def _create_processor(self) -> Callable[[NDArrayReal], NDArrayReal]:
        """処理関数を作成（サブクラスで実装）"""
        pass

    def process(self, data: DaArray) -> DaArray:
        """
        操作を実行して結果を返す
        data の形状は (channels, samples)
        """
        # 処理を計算グラフに追加
        logger.debug("Adding operation to computation graph (no data load yet)")
        result = _da_map_blocks(self._processor_func, data, dtype=data.dtype)
        return result

    @classmethod
    def create(cls, sampling_rate: float, **params: Any) -> "AudioOperation":
        """ファクトリーメソッド - サブクラスのインスタンスを作成"""
        return cls(sampling_rate, **params)


class HighPassFilter(AudioOperation):
    """ハイパスフィルタ操作"""

    operation_name = "highpass_filter"

    def __init__(self, sampling_rate: float, cutoff: float, order: int = 4):
        """
        ハイパスフィルタの初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        cutoff : float
            カットオフ周波数 (Hz)
        order : int, optional
            フィルタ次数、デフォルトは4
        """
        self.cutoff = cutoff  # 型安全のため明示的に保存
        self.order = order
        super().__init__(sampling_rate, cutoff=cutoff, order=order)

    def validate_params(self) -> None:
        """パラメータ検証"""
        if self.cutoff <= 0 or self.cutoff >= self.sampling_rate / 2:
            limit = self.sampling_rate / 2
            raise ValueError(
                f"カットオフ周波数は0Hzから{limit}Hzの間である必要があります"
            )

    def _create_processor(self) -> Callable[[NDArrayReal], NDArrayReal]:
        """ハイパスフィルタのプロセッサ関数を作成"""
        # フィルタ係数を計算（一度だけ）- インスタンス変数から安全に取得
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff / nyquist

        # フィルタ係数を事前計算して保存
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype="high")  # type: ignore [unused-ignore]

        def _apply_filter(x: NDArrayReal) -> NDArrayReal:
            logger.debug(f"Applying highpass filter to block with shape: {x.shape}")
            result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)
            logger.debug(f"Filter applied, returning result with shape: {result.shape}")
            return result

        return _apply_filter


class LowPassFilter(AudioOperation):
    """ローパスフィルタ操作"""

    operation_name = "lowpass_filter"
    a: NDArrayReal
    b: NDArrayReal

    def __init__(self, sampling_rate: float, cutoff: float, order: int = 4):
        """
        ローパスフィルタの初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        cutoff : float
            カットオフ周波数 (Hz)
        order : int, optional
            フィルタ次数、デフォルトは4
        """
        self.cutoff = cutoff
        self.order = order
        super().__init__(sampling_rate, cutoff=cutoff, order=order)

    def validate_params(self) -> None:
        """パラメータ検証"""
        if self.cutoff <= 0 or self.cutoff >= self.sampling_rate / 2:
            raise ValueError(
                f"カットオフ周波数は0Hzから{self.sampling_rate / 2}Hzの間である必要があります"  # noqa: E501
            )

    def _create_processor(self) -> Callable[[NDArrayReal], NDArrayReal]:
        """ローパスフィルタのプロセッサ関数を作成"""
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff / nyquist

        # フィルタ係数を事前計算して保存
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype="low")  # type: ignore [unused-ignore]

        def _apply_filter(x: NDArrayReal) -> NDArrayReal:
            logger.debug(f"Applying lowpass filter to block with shape: {x.shape}")
            result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)
            logger.debug(f"Filter applied, returning result with shape: {result.shape}")
            return result

        return _apply_filter


# class Normalize(AudioOperation):
#     """音量正規化操作"""

#     operation_name = "normalize"

#     def __init__(
#         self, sampling_rate: float, target_level: float = -20,
#         channel_wise: bool = True
#     ):
#         """
#         正規化処理の初期化

#         Parameters
#         ----------
#         sampling_rate : float
#             サンプリングレート (Hz)
#         target_level : float, optional
#             目標レベル (dB)、デフォルトは-20dB
#         channel_wise : bool, optional
#             チャネル個別に正規化するか、デフォルトはTrue
#         """
#         self.target_level = target_level
#         self.channel_wise = channel_wise
#         self.target_rms = 10 ** (target_level / 20)  # 事前計算
#         super().__init__(
#             sampling_rate, target_level=target_level, channel_wise=channel_wise
#         )

#     def _create_processor(self) -> Callable[[NDArrayReal], NDArrayReal]:
#         """正規化処理のプロセッサ関数を作成"""

#         def _apply_normalize(x: NDArrayReal) -> NDArrayReal:
#             logger.debug(f"Applying normalize to block with shape: {x.shape}")

#             # データが空かチェック
#             if x.size == 0:
#                 return x

#             if x.shape[0] == 1 or not self.channel_wise:
#                 # モノラルまたは全チャネル一括正規化
#                 rms = np.sqrt(np.mean(x**2))
#                 if rms < 1e-10:  # 無音の場合
#                     return x
#                 gain = self.target_rms / rms
#                 return x * gain

#             else:
#                 # マルチチャネル、チャネルごとに独立正規化
#                 rms_per_channel = np.sqrt(np.mean(x**2, axis=-1))
#                 silent_mask = rms_per_channel < 1e-10
#                 gain = np.ones_like(rms_per_channel)
#                 gain[~silent_mask] = self.target_rms / rms_per_channel[~silent_mask]
#                 return x * gain[:, np.newaxis]

#         return _apply_normalize


# 操作タイプと対応するクラスのマッピング
_OPERATION_REGISTRY: dict[str, type[AudioOperation]] = {
    "highpass_filter": HighPassFilter,
    "lowpass_filter": LowPassFilter,
    # "normalize": Normalize,
}


def register_operation(name: str, operation_class: type[AudioOperation]) -> None:
    """新しい操作タイプを登録"""
    if not issubclass(operation_class, AudioOperation):
        raise TypeError("操作クラスは AudioOperation を継承する必要があります")
    _OPERATION_REGISTRY[name] = operation_class


def get_operation(name: str) -> type[AudioOperation]:
    """名前から操作クラスを取得"""
    if name not in _OPERATION_REGISTRY:
        raise ValueError(f"未知の操作タイプです: {name}")
    return _OPERATION_REGISTRY[name]


def create_operation(name: str, sampling_rate: float, **params: Any) -> AudioOperation:
    """操作名とパラメータから操作インスタンスを作成"""
    operation_class = get_operation(name)
    return operation_class(sampling_rate, **params)
