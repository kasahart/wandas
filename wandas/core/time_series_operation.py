import inspect
import logging
from typing import Any, ClassVar, Generic, Optional, TypeVar, Union

import dask
import dask.array as da
import librosa
import numpy as np
from dask.array.core import Array as DaArray
from mosqito.sound_level_meter import noct_spectrum, noct_synthesis
from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq
from scipy import signal
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import get_window
from waveform_analysis import A_weight

from wandas.core import util
from wandas.utils.types import NDArrayComplex, NDArrayReal

logger = logging.getLogger(__name__)

_da_map_blocks = da.map_blocks  # type: ignore [unused-ignore]
_da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]

# Define TypeVars for input and output array types
InputArrayType = TypeVar("InputArrayType", NDArrayReal, NDArrayComplex)
OutputArrayType = TypeVar("OutputArrayType", NDArrayReal, NDArrayComplex)


class AudioOperation(Generic[InputArrayType, OutputArrayType]):
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

    def _process_array(self, x: InputArrayType) -> OutputArrayType:
        """処理関数（サブクラスで実装）"""
        # デフォルトは何もしない関数
        raise NotImplementedError("Subclasses must implement this method.")

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def process_array(self, x: InputArrayType) -> OutputArrayType:
        """@dask.delayed でラップされた処理関数"""
        # デフォルトは何もしない関数
        logger.debug(f"Default process operation on data with shape: {x.shape}")
        return self._process_array(x)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        操作後の出力データの形状を計算します（サブクラスで実装）

        Parameters
        ----------
        input_shape : tuple
            入力データの形状

        Returns
        -------
        tuple
            出力データの形状
        """
        raise NotImplementedError("Subclasses must implement this method.")

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

    # @classmethod
    # def create(cls, sampling_rate: float, **params: Any) -> "AudioOperation":
    #     """ファクトリーメソッド - サブクラスのインスタンスを作成"""
    #     return cls(sampling_rate, **params)


class HighPassFilter(AudioOperation[NDArrayReal, NDArrayReal]):
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

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """@dask.delayed でラップされたフィルタ処理"""
        logger.debug(f"Applying highpass filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)
        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class LowPassFilter(AudioOperation[NDArrayReal, NDArrayReal]):
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

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """@dask.delayed でラップされたフィルタ処理"""
        logger.debug(f"Applying lowpass filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)

        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class AWeighting(AudioOperation[NDArrayReal, NDArrayReal]):
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

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Aウェイトフィルタのプロセッサ関数を作成"""
        logger.debug(f"Applying A-weighting to array with shape: {x.shape}")
        result = A_weight(x, self.sampling_rate)

        # Handle case where A_weight returns a tuple
        if isinstance(result, tuple):
            # Use the first element of the tuple
            result = result[0]

        logger.debug(
            f"A-weighting applied, returning result with shape: {result.shape}"
        )
        return np.array(result)


class HpssHarmonic(AudioOperation[NDArrayReal, NDArrayReal]):
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

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """HPSS Harmonicのプロセッサ関数を作成"""
        logger.debug(f"Applying HPSS Harmonic to array with shape: {x.shape}")
        result = librosa.effects.harmonic(x, **self.kwargs)
        logger.debug(
            f"HPSS Harmonic applied, returning result with shape: {result.shape}"
        )
        return result


class HpssPercussive(AudioOperation[NDArrayReal, NDArrayReal]):
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

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """HPSS Percussiveのプロセッサ関数を作成"""
        logger.debug(f"Applying HPSS Percussive to array with shape: {x.shape}")
        result = librosa.effects.percussive(x, **self.kwargs)
        logger.debug(
            f"HPSS Percussive applied, returning result with shape: {result.shape}"
        )
        return result


class ReSampling(AudioOperation[NDArrayReal, NDArrayReal]):
    """リサンプリング操作"""

    name = "resampling"

    def __init__(self, sampling_rate: float, target_sr: float):
        """
        リサンプリング操作の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        target_sampling_rate : float
            目標サンプリングレート (Hz)
        """
        super().__init__(sampling_rate, target_sr=target_sr)
        self.target_sr = target_sr

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
        # リサンプリング後の長さを計算
        ratio = float(self.target_sr) / float(self.sampling_rate)
        n_samples = int(np.ceil(input_shape[-1] * ratio))
        return (*input_shape[:-1], n_samples)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """リサンプリング操作のプロセッサ関数を作成"""
        logger.debug(f"Applying resampling to array with shape: {x.shape}")
        result = librosa.resample(
            x, orig_sr=self.sampling_rate, target_sr=self.target_sr
        )
        logger.debug(f"Resampling applied, returning result with shape: {result.shape}")
        return result


class ABS(AudioOperation[NDArrayReal, NDArrayReal]):
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

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return da.abs(data)  # type: ignore [unused-ignore]


class Power(AudioOperation[NDArrayReal, NDArrayReal]):
    """べき乗操作"""

    name = "power"

    def __init__(self, sampling_rate: float, exponent: float):
        """
        べき乗操作の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        """
        super().__init__(sampling_rate)
        self.exp = exponent

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return da.power(data, self.exp)  # type: ignore [unused-ignore]


class Trim(AudioOperation[NDArrayReal, NDArrayReal]):
    """トリミング操作"""

    name = "trim"

    def __init__(
        self,
        sampling_rate: float,
        start: float,
        end: float,
    ):
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
        # リサンプリング後の長さを計算
        # 信号がない部分は除外する
        end_sample = min(self.end_sample, input_shape[-1])
        n_samples = end_sample - self.start_sample
        return (*input_shape[:-1], n_samples)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """トリミング操作のプロセッサ関数を作成"""
        logger.debug(f"Applying trim to array with shape: {x.shape}")
        # トリミングを適用
        result = x[..., self.start_sample : self.end_sample]
        logger.debug(f"Trim applied, returning result with shape: {result.shape}")
        return result


class RmsTrend(AudioOperation[NDArrayReal, NDArrayReal]):
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
        ref: Union[list[float], float] = 1.0,
        dB: bool = False,  # noqa: N803
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
        self.dB = dB
        self.Aw = Aw
        self.ref = np.array(ref if isinstance(ref, list) else [ref])
        super().__init__(
            sampling_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            dB=dB,
            Aw=Aw,
            ref=self.ref,
        )

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        操作後の出力データの形状を計算します

        Parameters
        ----------
        input_shape : tuple
            入力データの形状 (channels, samples)

        Returns
        -------
        tuple
            出力データの形状 (channels, frames)
        """
        n_frames = librosa.feature.rms(
            y=np.ones((1, input_shape[-1])),
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        ).shape[-1]
        return (*input_shape[:-1], n_frames)

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

        if self.dB:
            # dBに変換
            result = 20 * np.log10(
                np.maximum(result / self.ref[..., np.newaxis], 1e-12)
            )
        #
        logger.debug(f"RMS applied, returning result with shape: {result.shape}")
        return result


class Sum(AudioOperation[NDArrayReal, NDArrayReal]):
    """合計計算"""

    name = "sum"

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return data.sum(axis=0, keepdims=True)


class Mean(AudioOperation[NDArrayReal, NDArrayReal]):
    """平均計算"""

    name = "mean"

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return data.mean(axis=0, keepdims=True)


class ChannelDifference(AudioOperation[NDArrayReal, NDArrayReal]):
    """チャネル間の差分計算"""

    name = "channel_difference"
    other_channel: int

    def __init__(self, sampling_rate: float, other_channel: int = 0):
        """
        チャネル間の差分計算の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        other_channel : int
            差分を計算するチャネル、デフォルトは0
        """
        self.other_channel = other_channel
        super().__init__(sampling_rate, other_channel=other_channel)

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        result = data - data[self.other_channel]
        return result


class FFT(AudioOperation[NDArrayReal, NDArrayComplex]):
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

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        操作後の出力データの形状を計算します

        Parameters
        ----------
        input_shape : tuple
            入力データの形状 (channels, samples)

        Returns
        -------
        tuple
            出力データの形状 (channels, freqs)
        """
        n_freqs = self.n_fft // 2 + 1 if self.n_fft else input_shape[-1] // 2 + 1
        return (*input_shape[:-1], n_freqs)

    def _process_array(self, x: NDArrayReal) -> NDArrayComplex:
        """FFT操作のプロセッサ関数を作成"""
        from scipy.signal import get_window

        win = get_window(self.window, x.shape[-1])
        x = x * win
        result: NDArrayComplex = np.fft.rfft(x, n=self.n_fft, axis=-1)
        result[..., 1:-1] *= 2.0
        # 窓関数補正
        scaling_factor = np.sum(win)
        result = result / scaling_factor
        return result


class IFFT(AudioOperation[NDArrayComplex, NDArrayReal]):
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
            IFFTのサイズ、デフォルトはNone（入力サイズに基づいて決定）
        window : str, optional
            窓関数の種類、デフォルトは'hann'
        """
        self.n_fft = n_fft
        self.window = window
        super().__init__(sampling_rate, n_fft=n_fft, window=window)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        操作後の出力データの形状を計算します

        Parameters
        ----------
        input_shape : tuple
            入力データの形状 (channels, freqs)

        Returns
        -------
        tuple
            出力データの形状 (channels, samples)
        """
        n_samples = 2 * (input_shape[-1] - 1) if self.n_fft is None else self.n_fft
        return (*input_shape[:-1], n_samples)

    def _process_array(self, x: NDArrayComplex) -> NDArrayReal:
        """IFFT操作のプロセッサ関数を作成"""
        logger.debug(f"Applying IFFT to array with shape: {x.shape}")

        # 周波数成分のスケールを戻す（FFTで乗算した2.0を除去）
        _x = x.copy()
        _x[..., 1:-1] /= 2.0

        # IFFT実行
        result: NDArrayReal = np.fft.irfft(_x, n=self.n_fft, axis=-1)

        # 窓関数補正（FFTの逆操作）
        from scipy.signal import get_window

        win = get_window(self.window, result.shape[-1])

        # FFTの窓関数スケーリングを補正
        scaling_factor = np.sum(win) / result.shape[-1]
        result = result / scaling_factor

        logger.debug(f"IFFT applied, returning result with shape: {result.shape}")
        return result


class Welch(AudioOperation[NDArrayReal, NDArrayReal]):
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

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        操作後の出力データの形状を計算します

        Parameters
        ----------
        input_shape : tuple
            入力データの形状 (channels, samples)

        Returns
        -------
        tuple
            出力データの形状 (channels, freqs)
        """
        n_freqs = self.n_fft // 2 + 1
        return (*input_shape[:-1], n_freqs)

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


class NOctSpectrum(AudioOperation[NDArrayReal, NDArrayReal]):
    """Nオクターブスペクトル操作"""

    name = "noct_spectrum"

    def __init__(
        self,
        sampling_rate: float,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ):
        """
        ノクターブスペクトルの初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        fmin : float
            最小周波数 (Hz)
        fmax : float
            最大周波数 (Hz)
        n : int, optional
            オクターブの分割数, デフォルトは3
        G : int, optional
            基準レベル, デフォルトは10
        fr : int, optional
            基準周波数, デフォルトは1000
        """
        super().__init__(sampling_rate, fmin=fmin, fmax=fmax, n=n, G=G, fr=fr)
        self.fmin = fmin
        self.fmax = fmax
        self.n = n
        self.G = G
        self.fr = fr

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
        # ノクターブスペクトルの出力形状を計算
        _, fpref = _center_freq(
            fmin=self.fmin, fmax=self.fmax, n=self.n, G=self.G, fr=self.fr
        )
        return (input_shape[0], fpref.shape[0])

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """ノクターブスペクトルのプロセッサ関数を作成"""
        logger.debug(f"Applying NoctSpectrum to array with shape: {x.shape}")
        spec, _ = noct_spectrum(
            sig=x.T,
            fs=self.sampling_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        if spec.ndim == 1:
            # 1次元の場合、チャネル数を追加
            spec = np.expand_dims(spec, axis=0)
        else:
            spec = spec.T
        logger.debug(f"NoctSpectrum applied, returning result with shape: {spec.shape}")
        return np.array(spec)


class NOctSynthesis(AudioOperation[NDArrayReal, NDArrayReal]):
    """ノクターブ合成操作"""

    name = "noct_synthesis"

    def __init__(
        self,
        sampling_rate: float,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ):
        """
        ノクターブ合成の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        fmin : float
            最小周波数 (Hz)
        fmax : float
            最大周波数 (Hz)
        n : int, optional
            オクターブの分割数, デフォルトは3
        G : int, optional
            基準レベル, デフォルトは10
        fr : int, optional
            基準周波数, デフォルトは1000
        """
        super().__init__(sampling_rate, fmin=fmin, fmax=fmax, n=n, G=G, fr=fr)

        self.fmin = fmin
        self.fmax = fmax
        self.n = n
        self.G = G
        self.fr = fr

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
        # ノクターブスペクトルの出力形状を計算
        _, fpref = _center_freq(
            fmin=self.fmin, fmax=self.fmax, n=self.n, G=self.G, fr=self.fr
        )
        return (input_shape[0], fpref.shape[0])

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """ノクターブ合成のプロセッサ関数を作成"""
        logger.debug(f"Applying NoctSynthesis to array with shape: {x.shape}")
        # nをshape[-1]から計算する
        n = x.shape[-1]  # nをshape[-1]から計算する
        if n % 2 == 0:
            n = n * 2 - 1
        else:
            n = (n - 1) * 2
        freqs = np.fft.rfftfreq(n, d=1 / self.sampling_rate)
        result, _ = noct_synthesis(
            spectrum=np.abs(x).T,
            freqs=freqs,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        result = result.T
        logger.debug(
            f"NoctSynthesis applied, returning result with shape: {result.shape}"
        )
        return np.array(result)


class STFT(AudioOperation[NDArrayReal, NDArrayComplex]):
    """Short-Time Fourier Transform 操作"""

    name = "stft"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
    ):
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.noverlap = (
            self.win_length - self.hop_length if hop_length is not None else None
        )
        self.window = window

        self.SFT = ShortTimeFFT(
            win=get_window(window, self.win_length),
            hop=self.hop_length,
            fs=sampling_rate,
            mfft=self.n_fft,
            scale_to="magnitude",
        )
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=window,
        )

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
        n_samples = input_shape[-1]
        n_f = len(self.SFT.f)
        n_t = len(self.SFT.t(n_samples))
        return (input_shape[0], n_f, n_t)

    def _process_array(self, x: NDArrayReal) -> NDArrayComplex:
        """複数チャネルを一度にSciPyのSTFT処理"""
        logger.debug(f"Applying SciPy STFT to array with shape: {x.shape}")

        # 入力が1次元の場合は2次元に変換
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Apply STFT to all channels at once
        result: NDArrayComplex = self.SFT.stft(x)
        result[..., 1:-1, :] *= 2.0
        logger.debug(f"SciPy STFT applied, returning result with shape: {result.shape}")
        return result


class ISTFT(AudioOperation[NDArrayComplex, NDArrayReal]):
    """Inverse Short-Time Fourier Transform 操作"""

    name = "istft"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        length: Optional[int] = None,
    ):
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.window = window
        self.length = length

        # Instantiate ShortTimeFFT for ISTFT calculation
        self.SFT = ShortTimeFFT(
            win=get_window(window, self.win_length),
            hop=self.hop_length,
            fs=sampling_rate,
            mfft=self.n_fft,
            scale_to="magnitude",  # Consistent scaling with STFT
        )

        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=window,
            length=length,
        )

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        操作後の出力データの形状を計算します

        Parameters
        ----------
        input_shape : tuple
            入力データの形状 (channels, freqs, time_frames)

        Returns
        -------
        tuple
            出力データの形状 (channels, samples)
        """
        k0: int = 0
        q_max = input_shape[-1] + self.SFT.p_min
        k_max = (q_max - 1) * self.SFT.hop + self.SFT.m_num - self.SFT.m_num_mid
        k_q0, k_q1 = self.SFT.nearest_k_p(k0), self.SFT.nearest_k_p(k_max, left=False)
        n_pts = k_q1 - k_q0 + self.SFT.m_num - self.SFT.m_num_mid

        return input_shape[:-2] + (n_pts,)

    def _process_array(self, x: NDArrayComplex) -> NDArrayReal:
        """複数チャネルを一度にSciPyのISTFT処理 using ShortTimeFFT"""
        logger.debug(
            f"Applying SciPy ISTFT (ShortTimeFFT) to array with shape: {x.shape}"
        )

        # 入力が2次元の場合は3次元に変換 (単一チャネルとみなす)
        if x.ndim == 2:
            x = x.reshape(1, *x.shape)

        # Adjust scaling back if STFT applied factor of 2
        _x = np.copy(x)
        _x[..., 1:-1, :] /= 2.0

        # Apply ISTFT using the ShortTimeFFT instance
        result: NDArrayReal = self.SFT.istft(_x)

        # Trim to desired length if specified
        if self.length is not None:
            result = result[..., : self.length]

        logger.debug(
            f"ShortTimeFFT applied, returning result with shape: {result.shape}"
        )
        return result


class AddWithSNR(AudioOperation[NDArrayReal, NDArrayReal]):
    """SNRを考慮した加算操作"""

    name = "add_with_snr"

    def __init__(self, sampling_rate: float, other: DaArray, snr: float):
        """
        SNRを考慮した加算操作の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        other : DaArray
            加算するノイズ信号 (チャネルフレーム形式)
        snr : float
            信号対雑音比 (dB)
        """
        super().__init__(sampling_rate, other=other, snr=snr)

        self.other = other
        self.snr = snr
        logger.debug(f"Initialized AddWithSNR operation with SNR: {snr} dB")

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
            出力データの形状（入力と同じ）
        """
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """SNRを考慮した加算処理を実行"""
        logger.debug(f"Applying SNR-based addition with shape: {x.shape}")
        other: NDArrayReal = self.other.compute()

        # 多チャンネル対応版のcalculate_rmsとcalculate_desired_noise_rmsを使用
        clean_rms = util.calculate_rms(x)
        other_rms = util.calculate_rms(other)

        # 指定されたSNRに基づいてノイズのゲインを調整（チャネルごとに適用）
        desired_noise_rms = util.calculate_desired_noise_rms(clean_rms, self.snr)

        # ブロードキャストでチャネルごとのゲインを適用
        gain = desired_noise_rms / other_rms
        # 調整したノイズを信号に加算
        result: NDArrayReal = x + other * gain
        return result


class Coherence(AudioOperation[NDArrayReal, NDArrayReal]):
    """コヒーレンス推定操作"""

    name = "coherence"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        detrend: str,
    ):
        """
        コヒーレンス推定操作の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        n_fft : int
            FFTのサイズ
        hop_length : int
            ホップ長
        win_length : int
            窓の長さ
        window : str
            窓関数
        detrend : str
            デトレンドの種類
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.detrend = detrend
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
        )

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        操作後の出力データの形状を計算します

        Parameters
        ----------
        input_shape : tuple
            入力データの形状 (channels, samples)

        Returns
        -------
        tuple
            出力データの形状 (channels * channels, freqs)
        """
        n_channels = input_shape[0]
        n_freqs = self.n_fft // 2 + 1
        return (n_channels * n_channels, n_freqs)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """コヒーレンス推定操作のプロセッサ関数"""
        logger.debug(f"Applying coherence estimation to array with shape: {x.shape}")
        from scipy import signal as ss

        _, coh = ss.coherence(
            x=x[:, np.newaxis],
            y=x[np.newaxis, :],
            fs=self.sampling_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
        )

        # 結果を (n_channels * n_channels, n_freqs) に再整形
        result: NDArrayReal = coh.reshape(-1, coh.shape[-1])

        logger.debug(f"Coherence estimation applied, result shape: {result.shape}")
        return result


class CSD(AudioOperation[NDArrayReal, NDArrayComplex]):
    """クロススペクトル密度推定操作"""

    name = "csd"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        detrend: str,
        scaling: str,
        average: str,
    ):
        """
        クロススペクトル密度推定操作の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        n_fft : int
            FFTのサイズ
        hop_length : int
            ホップ長
        win_length : int
            窓の長さ
        window : str
            窓関数
        detrend : str
            デトレンドの種類
        scaling : str
            スケーリングの種類
        average : str
            平均化の方法
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.detrend = detrend
        self.scaling = scaling
        self.average = average
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        操作後の出力データの形状を計算します

        Parameters
        ----------
        input_shape : tuple
            入力データの形状 (channels, samples)

        Returns
        -------
        tuple
            出力データの形状 (channels * channels, freqs)
        """
        n_channels = input_shape[0]
        n_freqs = self.n_fft // 2 + 1
        return (n_channels * n_channels, n_freqs)

    def _process_array(self, x: NDArrayReal) -> NDArrayComplex:
        """クロススペクトル密度推定操作のプロセッサ関数"""
        logger.debug(f"Applying CSD estimation to array with shape: {x.shape}")
        from scipy import signal as ss

        # scipyのcsd関数で全ての組み合わせを計算
        _, csd_result = ss.csd(
            x=x[:, np.newaxis],
            y=x[np.newaxis, :],
            fs=self.sampling_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
        )

        # 結果を (n_channels * n_channels, n_freqs) に再整形
        result: NDArrayComplex = csd_result.reshape(-1, csd_result.shape[-1])

        logger.debug(f"CSD estimation applied, result shape: {result.shape}")
        return result


class TransferFunction(AudioOperation[NDArrayReal, NDArrayComplex]):
    """伝達関数推定操作"""

    name = "transfer_function"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        detrend: str,
        scaling: str,
        average: str,
    ):
        """
        伝達関数推定操作の初期化

        Parameters
        ----------
        sampling_rate : float
            サンプリングレート (Hz)
        n_fft : int
            FFTのサイズ
        hop_length : int
            ホップ長
        win_length : int
            窓の長さ
        window : str
            窓関数
        detrend : str
            デトレンドの種類
        scaling : str
            スケーリングの種類
        average : str
            平均化の方法
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.detrend = detrend
        self.scaling = scaling
        self.average = average
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        操作後の出力データの形状を計算します

        Parameters
        ----------
        input_shape : tuple
            入力データの形状 (channels, samples)

        Returns
        -------
        tuple
            出力データの形状 (channels * channels, freqs)
        """
        n_channels = input_shape[0]
        n_freqs = self.n_fft // 2 + 1
        return (n_channels * n_channels, n_freqs)

    def _process_array(self, x: NDArrayReal) -> NDArrayComplex:
        """伝達関数推定操作のプロセッサ関数"""
        logger.debug(
            f"Applying transfer function estimation to array with shape: {x.shape}"
        )
        from scipy import signal as ss

        # 全チャネル間のクロススペクトル密度を計算
        f, p_yx = ss.csd(
            x=x[:, np.newaxis, :],
            y=x[np.newaxis, :, :],
            fs=self.sampling_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
            axis=-1,
        )
        # p_yx shape: (num_channels, num_channels, num_frequencies)

        # 各チャネルのパワースペクトル密度を計算
        f, p_xx = ss.welch(
            x=x,
            fs=self.sampling_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
            axis=-1,
        )
        # p_xx shape: (num_channels, num_frequencies)

        # 伝達関数 H(f) = P_yx / P_xx を計算
        h_f = p_yx / p_xx[np.newaxis, :, :]

        result: NDArrayComplex = h_f.reshape(-1, h_f.shape[-1])

        logger.debug(
            f"Transfer function estimation applied, result shape: {result.shape}"
        )
        return result


# 操作タイプと対応するクラスのマッピングを自動で収集
_OPERATION_REGISTRY: dict[str, type[AudioOperation[Any, Any]]] = {}


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


def get_operation(name: str) -> type[AudioOperation[Any, Any]]:
    """名前から操作クラスを取得"""
    if name not in _OPERATION_REGISTRY:
        raise ValueError(f"未知の操作タイプです: {name}")
    return _OPERATION_REGISTRY[name]


def create_operation(
    name: str, sampling_rate: float, **params: Any
) -> AudioOperation[Any, Any]:
    """操作名とパラメータから操作インスタンスを作成"""
    operation_class = get_operation(name)
    return operation_class(sampling_rate, **params)
