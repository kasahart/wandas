import inspect
import logging
from typing import Any, ClassVar, Optional, Union

import dask
import dask.array as da
import librosa
import numpy as np
from dask.array.core import Array as DaArray
from scipy import fft, signal  # Updated import statement
from waveform_analysis import A_weight

from wandas.utils.types import NDArrayComplex, NDArrayReal

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

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """@dask.delayed でラップされた処理関数（サブクラスで実装）"""
        # デフォルトは何もしない関数
        return x

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def process_array(self, x: NDArrayReal) -> NDArrayReal:
        """@dask.delayed でラップされた処理関数（サブクラスで実装）"""
        # デフォルトは何もしない関数
        logger.debug(f"Default process operation on data with shape: {x.shape}")
        return self._process_array(x)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        操作後の出力データの形状を計算します

        Parameters
        ----------
        input_shape : tuple
            入力データの形状

        Returns
        -------
        tuple
            出力データの形状
        """
        # デフォルトでは形状が変わらない
        sample_input = np.ones((1, *input_shape[1:]))
        x = self._process_array(sample_input)
        return (input_shape[0], *x.shape[1:])

    def process(self, data: DaArray) -> DaArray:
        """
        操作を実行して結果を返す
        data の形状は (channels, samples)
        """
        # 遅延処理としてタスクを追加
        logger.debug("Adding delayed operation to computation graph")
        delayed_result = self.process_array(data)
        # 遅延結果をdask配列に変換して返す
        output_shape = self.calculate_output_shape(data.shape)
        return _da_from_delayed(delayed_result, shape=output_shape, dtype=data.dtype)

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

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """@dask.delayed でラップされたフィルタ処理"""
        logger.debug(f"Applying lowpass filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)

        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class AWeighting(AudioOperation):
    """Aウェイトフィルタ操作"""

    name = "a_weighting"

    def __init__(self, sampling_rate: float):
        """
        Aウェイトフィルタの初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        """
        super().__init__(sampling_rate)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Aウェイトフィルタのプロセッサ関数を作成"""
        logger.debug(f"Applying A-weighting to array with shape: {x.shape}")
        result = A_weight(x, self.sampling_rate)
        logger.debug(
            f"A-weighting applied, returning result with shape: {result.shape}"
        )
        return np.array(result)


class HpssHarmonic(AudioOperation):
    """HPSS Harmonic操作"""

    name = "hpss_harmonic"

    def __init__(
        self,
        sampling_rate: float,
        **kwargs: Any,
    ):
        """
        HPSS Harmonicの初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        """
        self.kwargs = kwargs
        super().__init__(sampling_rate, **kwargs)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """HPSS Harmonicのプロセッサ関数を作成"""
        logger.debug(f"Applying HPSS Harmonic to array with shape: {x.shape}")
        result = librosa.effects.harmonic(x, **self.kwargs)
        logger.debug(
            f"HPSS Harmonic applied, returning result with shape: {result.shape}"
        )
        return result


class HpssPercussive(AudioOperation):
    """HPSS Percussive操作"""

    name = "hpss_percussive"

    def __init__(
        self,
        sampling_rate: float,
        **kwargs: Any,
    ):
        """
        HPSS Percussiveの初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        """
        self.kwargs = kwargs
        super().__init__(sampling_rate, **kwargs)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """HPSS Percussiveのプロセッサ関数を作成"""
        logger.debug(f"Applying HPSS Percussive to array with shape: {x.shape}")
        result = librosa.effects.percussive(x, **self.kwargs)
        logger.debug(
            f"HPSS Percussive applied, returning result with shape: {result.shape}"
        )
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

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """べき乗操作のプロセッサ関数を作成"""
        logger.debug(
            f"Applying power with exp {self.exp} to array with shape: {x.shape}"
        )
        result = np.power(x, self.exp)
        logger.debug(f"Power applied, returning result with shape: {result.shape}")
        return result


class Trim(AudioOperation):
    """トリミング操作"""

    name = "trim"

    def __init__(self, sampling_rate: float, start: float, end: float):
        """
        トリミング操作の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        start : float
            トリミング開始位置 (秒)
        end : float
            トリミング終了位置 (秒)
        """
        super().__init__(sampling_rate, start=start, end=end)
        self.start = start
        self.end = end
        self.start_sample = int(start * sampling_rate)
        self.end_sample = int(end * sampling_rate)
        logger.debug(
            f"Initialized Trim operation with start: {self.start}, end: {self.end}"
        )

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """トリミング操作のプロセッサ関数を作成"""
        logger.debug(f"Applying trim to array with shape: {x.shape}")
        # トリミングを適用
        result = x[..., self.start_sample : self.end_sample]
        logger.debug(f"Trim applied, returning result with shape: {result.shape}")
        return result


class RmsTrend(AudioOperation):
    """RMS計算"""

    name = "rms_trend"
    frame_length: int
    hop_length: int
    Aw: bool

    def __init__(
        self,
        sampling_rate: float,
        frame_length: int = 2048,
        hop_length: int = 512,
        Aw: bool = False,  # noqa: N803
    ) -> None:
        """
        RMS計算の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        frame_length : int
            フレームの長さ、デフォルトは2048
        hop_length : int
            ホップの長さ、デフォルトは512
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.Aw = Aw
        super().__init__(
            sampling_rate, frame_length=frame_length, hop_length=hop_length, Aw=Aw
        )

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """RMS計算のプロセッサ関数を作成"""
        logger.debug(f"Applying RMS to array with shape: {x.shape}")

        if self.Aw:
            # A-weightingを適用
            _x = A_weight(x, self.sampling_rate)
            if isinstance(_x, np.ndarray):
                # A_weightがタプルを返す場合、最初の要素を使用
                x = _x
            elif isinstance(_x, tuple):
                # A_weightがtupleを返す場合、最初の要素を使用
                x = _x[0]
            else:
                raise ValueError("A_weighting returned an unexpected type.")

        # RMSを計算
        result = librosa.feature.rms(
            y=x, frame_length=self.frame_length, hop_length=self.hop_length
        )[..., 0, :]
        logger.debug(f"RMS applied, returning result with shape: {result.shape}")
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
    n_fft: Optional[int]
    window: str

    def __init__(
        self, sampling_rate: float, n_fft: Optional[int] = None, window: str = "hann"
    ):
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

    def _process_array(self, x: NDArrayReal) -> NDArrayComplex:
        """FFT操作のプロセッサ関数を作成"""
        from scipy.signal import get_window

        win = get_window(self.window, x.shape[-1])
        x = x * win
        result: NDArrayComplex = fft.rfft(x, n=self.n_fft, axis=-1)
        result[..., 1:-1] *= 2.0
        # 窓関数補正
        scaling_factor = np.sum(win)
        result /= scaling_factor
        return result


class IFFT(AudioOperation):
    """IFFT"""

    name = "ifft"
    n_fft: Optional[int]
    window: str

    def __init__(
        self, sampling_rate: float, n_fft: Optional[int] = None, window: str = "hann"
    ):
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

    def _process_array(self, x: NDArrayComplex) -> NDArrayReal:
        """FFT操作のプロセッサ関数を作成"""
        from scipy.signal import get_window

        _x = np.copy(x)
        _x[..., 1:-1] /= 2.0
        result = fft.irfft(_x, n=self.n_fft, axis=-1)

        # 窓関数補正
        n_samples = result.shape[-1]
        win = get_window(self.window, n_samples)
        scaling_factor = np.sum(win)
        result *= scaling_factor
        result = result / (win + 1e-12)
        return result


class Welch(AudioOperation):
    """Welch"""

    name = "welch"
    n_fft: int
    window: str
    hop_length: Optional[int]
    win_length: Optional[int]
    average: str
    detrend: str

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
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
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.noverlap = (
            self.win_length - self.hop_length if hop_length is not None else None
        )
        self.window = window
        self.average = average
        self.detrend = detrend
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=window,
            average=average,
            detrend=detrend,
        )

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Welch操作のプロセッサ関数を作成"""
        from scipy import signal as ss

        _, result = ss.welch(
            x,
            nperseg=self.win_length,
            noverlap=self.noverlap,
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


class STFT(AudioOperation):
    """Short-Time Fourier Transform 操作"""

    name = "stft"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        boundary: Optional[str] = "zeros",
    ):
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.noverlap = (
            self.win_length - self.hop_length if hop_length is not None else None
        )
        self.window = window
        self.boundary = boundary
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=window,
            boundary=boundary,
        )

    def _process_array(self, x: NDArrayReal) -> NDArrayComplex:
        """複数チャネルを一度にSciPyのSTFT処理"""
        logger.debug(f"Applying SciPy STFT to array with shape: {x.shape}")

        # 入力が1次元の場合は2次元に変換
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Apply STFT to all channels at once
        result: NDArrayComplex
        _, _, result = signal.stft(
            x,
            fs=self.sampling_rate,
            window=self.window,
            nperseg=self.win_length,
            noverlap=self.noverlap,
            nfft=self.n_fft,
            boundary=self.boundary,  # type: ignore [unused-ignore]
            padded=True,
            axis=-1,  # Process along the samples axis
        )
        result[..., 1:-1, :] *= 2.0
        logger.debug(f"SciPy STFT applied, returning result with shape: {result.shape}")
        return result


class ISTFT(AudioOperation):
    """Inverse Short-Time Fourier Transform 操作"""

    name = "istft"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        boundary: Optional[str] = "zeros",
        length: Optional[int] = None,
    ):
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.noverlap = (
            self.win_length - self.hop_length if hop_length is not None else None
        )
        self.window = window
        self.boundary = boundary
        self.length = length
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=window,
            boundary=boundary,
            length=length,
        )

    def _process_array(self, x: NDArrayComplex) -> NDArrayReal:
        """複数チャネルを一度にSciPyのISTFT処理"""
        logger.debug(f"Applying SciPy ISTFT to array with shape: {x.shape}")

        # 入力が2次元の場合は3次元に変換 (単一チャネルとみなす)
        if x.ndim == 2:
            x = x.reshape(1, *x.shape)
        _x = np.copy(x)
        _x[..., 1:-1, :] /= 2.0
        result: NDArrayReal
        _, result = signal.istft(
            _x,
            fs=self.sampling_rate,
            window=self.window,
            nperseg=self.win_length,
            noverlap=self.noverlap,
            nfft=self.n_fft,
            input_onesided=True,
            boundary=False if self.boundary is None else True,
            time_axis=-1,  # Process along the time axis
            freq_axis=-2,  # Process along the frequency axis
        )

        logger.debug(
            f"SciPy ISTFT applied, returning result with shape: {result.shape}"
        )
        return result[..., : self.length] if self.length is not None else result


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
