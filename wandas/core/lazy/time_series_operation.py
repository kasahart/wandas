import inspect
import logging
from typing import Any, ClassVar, Optional, Union

import dask
import dask.array as da
import numpy as np

# import numpy as np
from dask.array.core import Array as DaArray
from scipy import signal

from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)

_da_map_blocks = da.map_blocks  # type: ignore [unused-ignore]
_da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]


class AudioOperation:
    """音声処理操作の抽象基底クラス"""

    # クラス変数：操作の名前
    name: ClassVar[str]

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
        self._setup_processor()

        logger.debug(
            f"Initialized {self.__class__.__name__} operation with params: {params}"
        )

    def validate_params(self) -> None:
        """パラメータの検証（無効な場合は例外を発生）"""
        pass

    def _setup_processor(self) -> None:
        """処理関数のセットアップ（サブクラスで実装）"""
        pass

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """@dask.delayed でラップされた処理関数（サブクラスで実装）"""
        # デフォルトは何もしない関数
        logger.debug(f"Default process operation on data with shape: {x.shape}")
        return x

    def process(self, data: DaArray) -> DaArray:
        """
        操作を実行して結果を返す
        data の形状は (channels, samples)
        """
        # 遅延処理としてタスクを追加
        logger.debug("Adding delayed operation to computation graph")
        delayed_result = self._process_array(data)
        # 遅延結果をdask配列に変換して返す
        return _da_from_delayed(delayed_result, shape=data.shape, dtype=data.dtype)

    @classmethod
    def create(cls, sampling_rate: float, **params: Any) -> "AudioOperation":
        """ファクトリーメソッド - サブクラスのインスタンスを作成"""
        return cls(sampling_rate, **params)


class HighPassFilter(AudioOperation):
    """ハイパスフィルタ操作"""

    name = "highpass_filter"

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

    def _setup_processor(self) -> None:
        """ハイパスフィルタのプロセッサをセットアップ"""
        # フィルタ係数を計算（一度だけ）- インスタンス変数から安全に取得
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff / nyquist

        # フィルタ係数を事前計算して保存
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype="high")  # type: ignore [unused-ignore]
        logger.debug(f"Highpass filter coefficients calculated: b={self.b}, a={self.a}")

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """@dask.delayed でラップされたフィルタ処理"""
        logger.debug(f"Applying highpass filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)
        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class LowPassFilter(AudioOperation):
    """ローパスフィルタ操作"""

    name = "lowpass_filter"
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

    def _setup_processor(self) -> None:
        """ローパスフィルタのプロセッサをセットアップ"""
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff / nyquist

        # フィルタ係数を事前計算して保存
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype="low")  # type: ignore [unused-ignore]
        logger.debug(f"Lowpass filter coefficients calculated: b={self.b}, a={self.a}")

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """@dask.delayed でラップされたフィルタ処理"""
        logger.debug(f"Applying lowpass filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)

        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class ABS(AudioOperation):
    """絶対値操作"""

    name = "abs"

    def __init__(self, sampling_rate: float):
        """
        絶対値操作の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        """
        super().__init__(sampling_rate)

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """絶対値操作のプロセッサ関数を作成"""
        logger.debug(f"Applying abs to array with shape: {x.shape}")
        result = np.abs(x)
        logger.debug(f"Abs applied, returning result with shape: {result.shape}")
        return result


class Power(AudioOperation):
    """べき乗計算"""

    name = "power"

    def __init__(self, sampling_rate: float, exponent: float):
        """
        絶対値操作の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        """
        super().__init__(sampling_rate)
        self.exp = exponent

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """べき乗操作のプロセッサ関数を作成"""
        logger.debug(
            f"Applying power with exp {self.exp} to array with shape: {x.shape}"
        )
        result = np.power(x, self.exp)
        logger.debug(f"Power applied, returning result with shape: {result.shape}")
        return result


class Sum(AudioOperation):
    """合計計算"""

    name = "sum"

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return data.sum(axis=0, keepdims=True)


class Mean(AudioOperation):
    """平均計算"""

    name = "mean"

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return data.mean(axis=0, keepdims=True)


class ChannelDifference(AudioOperation):
    """チャネル間の差分計算"""

    name = "channel_difference"
    other_channel: Union[int, str]

    def __init__(self, sampling_rate: float, other_channel: Union[int, str] = 0):
        """
        チャネル間の差分計算の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        other_channel : int or str, optional
            差分を計算するチャネル、デフォルトは0
        """
        self.other_channel = other_channel
        super().__init__(sampling_rate, other_channel=other_channel)

    def process(self, data: DaArray) -> DaArray:
        # チャネル間の差分を計算
        return data - data[self.other_channel]


class FFT(AudioOperation):
    """FFT"""

    name = "fft"
    n_fft: int
    window: str

    def __init__(self, sampling_rate: float, n_fft: int = 2048, window: str = "hann"):
        """
        FFT操作の初期化
        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        n_fft : int, optional
            FFTのサイズ、デフォルトは2048
        window : str, optional
            窓関数の種類、デフォルトは'hann'
        """
        self.n_fft = n_fft
        self.window = window
        super().__init__(sampling_rate, n_fft=n_fft, window=window)

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """FFT操作のプロセッサ関数を作成"""
        from scipy.signal import get_window

        n_samples = x.shape[-1]
        win = get_window(self.window, n_samples)
        x = x * win
        result = np.fft.rfft(x, n=self.n_fft, axis=-1)
        return result


class IFFT(AudioOperation):
    """IFFT"""

    name = "ifft"
    n_fft: int
    window: str

    def __init__(self, sampling_rate: float, n_fft: int = 2048, window: str = "hann"):
        """
        IFFT操作の初期化
        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        n_fft : int, optional
            FFTのサイズ、デフォルトは2048
        window : str, optional
            窓関数の種類、デフォルトは'hann'
        """
        self.n_fft = n_fft
        self.window = window
        super().__init__(sampling_rate, n_fft=n_fft, window=window)

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """FFT操作のプロセッサ関数を作成"""
        from scipy.signal import get_window

        result = np.fft.irfft(x, n=self.n_fft, axis=-1)

        n_samples = result.shape[-1]
        win = get_window(self.window, n_samples)
        result = result / (win + 1e-12)
        return result


class Welch(AudioOperation):
    """Welch"""

    name = "welch"
    n_fft: int
    window: str
    hop_length: int
    win_length: int
    average: str
    detrend: str

    def __init__(
        self,
        sampling_rate: float,
        win_length: int = 2048,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        window: str = "hann",
        average: str = "mean",
        detrend: str = "constant",
    ):
        """
        Welch操作の初期化
        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        n_fft : int, optional
            FFTのサイズ、デフォルトは2048
        window : str, optional
            窓関数の種類、デフォルトは'hann'
        """
        self.win_length = win_length
        self.n_fft = n_fft if n_fft is not None else win_length
        self.window = window
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.average = average
        self.detrend = detrend
        super().__init__(
            sampling_rate,
        )

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Welch操作のプロセッサ関数を作成"""
        from scipy import signal as ss

        _, result = ss.welch(
            x,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            average=self.average,
            detrend=self.detrend,
            scaling="spectrum",
        )

        if not isinstance(x, np.ndarray):
            # Dask配列の場合、計算をトリガー
            raise ValueError(
                "Welch operation requires a Dask array, but received a non-ndarray."
            )
        return np.array(result)


# 操作タイプと対応するクラスのマッピングを自動で収集
_OPERATION_REGISTRY: dict[str, type[AudioOperation]] = {}


def register_operation(operation_class: type) -> None:
    """新しい操作タイプを登録"""

    if not issubclass(operation_class, AudioOperation):
        raise TypeError("Strategy class must inherit from AudioOperation.")
    if inspect.isabstract(operation_class):
        raise TypeError("Cannot register abstract AudioOperation class.")

    _OPERATION_REGISTRY[operation_class.name] = operation_class


for strategy_cls in AudioOperation.__subclasses__():
    if not inspect.isabstract(strategy_cls):
        register_operation(strategy_cls)


def get_operation(name: str) -> type[AudioOperation]:
    """名前から操作クラスを取得"""
    if name not in _OPERATION_REGISTRY:
        raise ValueError(f"未知の操作タイプです: {name}")
    return _OPERATION_REGISTRY[name]


def create_operation(name: str, sampling_rate: float, **params: Any) -> AudioOperation:
    """操作名とパラメータから操作インスタンスを作成"""
    operation_class = get_operation(name)
    return operation_class(sampling_rate, **params)
