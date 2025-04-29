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

from wandas.utils import util
from wandas.utils.types import NDArrayComplex, NDArrayReal

logger = logging.getLogger(__name__)

_da_map_blocks = da.map_blocks  # type: ignore [unused-ignore]
_da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]

# Define TypeVars for input and output array types
InputArrayType = TypeVar("InputArrayType", NDArrayReal, NDArrayComplex)
OutputArrayType = TypeVar("OutputArrayType", NDArrayReal, NDArrayComplex)


class AudioOperation(Generic[InputArrayType, OutputArrayType]):
    """Abstract base class for audio processing operations."""

    # Class variable: operation name
    name: ClassVar[str]

    def __init__(self, sampling_rate: float, **params: Any):
        """
        Initialize AudioOperation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        **params : Any
            Operation-specific parameters
        """
        self.sampling_rate = sampling_rate
        self.params = params

        # Validate parameters during initialization
        self.validate_params()

        # Create processor function (lazy initialization possible)
        self._setup_processor()

        logger.debug(
            f"Initialized {self.__class__.__name__} operation with params: {params}"
        )

    def validate_params(self) -> None:
        """Validate parameters (raises exception if invalid)"""
        pass

    def _setup_processor(self) -> None:
        """Set up processor function (implemented by subclasses)"""
        pass

    def _process_array(self, x: InputArrayType) -> OutputArrayType:
        """Processing function (implemented by subclasses)"""
        # Default is no-op function
        raise NotImplementedError("Subclasses must implement this method.")

    @dask.delayed  # type: ignore [misc, unused-ignore]
    def process_array(self, x: InputArrayType) -> OutputArrayType:
        """Processing function wrapped with @dask.delayed"""
        # Default is no-op function
        logger.debug(f"Default process operation on data with shape: {x.shape}")
        return self._process_array(x)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation (implemented by subclasses)

        Parameters
        ----------
        input_shape : tuple
            Input data shape

        Returns
        -------
        tuple
            Output data shape
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def process(self, data: DaArray) -> DaArray:
        """
        Execute operation and return result
        data shape is (channels, samples)
        """
        # Add task as delayed processing
        logger.debug("Adding delayed operation to computation graph")
        delayed_result = self.process_array(data)
        # Convert delayed result to dask array and return
        output_shape = self.calculate_output_shape(data.shape)
        return _da_from_delayed(delayed_result, shape=output_shape, dtype=data.dtype)

    # @classmethod
    # def create(cls, sampling_rate: float, **params: Any) -> "AudioOperation":
    #     """Factory method - create instance of subclass"""
    #     return cls(sampling_rate, **params)


class HighPassFilter(AudioOperation[NDArrayReal, NDArrayReal]):
    """High-pass filter operation"""

    name = "highpass_filter"

    def __init__(self, sampling_rate: float, cutoff: float, order: int = 4):
        """
        Initialize high-pass filter

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        cutoff : float
            Cutoff frequency (Hz)
        order : int, optional
            Filter order, default is 4
        """
        self.cutoff = cutoff
        self.order = order
        super().__init__(sampling_rate, cutoff=cutoff, order=order)

    def validate_params(self) -> None:
        """Validate parameters"""
        if self.cutoff <= 0 or self.cutoff >= self.sampling_rate / 2:
            limit = self.sampling_rate / 2
            raise ValueError(f"Cutoff frequency must be between 0 Hz and {limit} Hz")

    def _setup_processor(self) -> None:
        """Set up high-pass filter processor"""
        # Calculate filter coefficients (once) - safely retrieve from instance variables
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff / nyquist

        # フィルタ係数を事前計算して保存
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype="high")  # type: ignore [unused-ignore]
        logger.debug(f"Highpass filter coefficients calculated: b={self.b}, a={self.a}")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Filter processing wrapped with @dask.delayed"""
        logger.debug(f"Applying highpass filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)
        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class LowPassFilter(AudioOperation[NDArrayReal, NDArrayReal]):
    """Low-pass filter operation"""

    name = "lowpass_filter"
    a: NDArrayReal
    b: NDArrayReal

    def __init__(self, sampling_rate: float, cutoff: float, order: int = 4):
        """
        Initialize low-pass filter

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        cutoff : float
            Cutoff frequency (Hz)
        order : int, optional
            Filter order, default is 4
        """
        self.cutoff = cutoff
        self.order = order
        super().__init__(sampling_rate, cutoff=cutoff, order=order)

    def validate_params(self) -> None:
        """Validate parameters"""
        if self.cutoff <= 0 or self.cutoff >= self.sampling_rate / 2:
            raise ValueError(
                f"Cutoff frequency must be between 0 Hz and {self.sampling_rate / 2} Hz"
            )

    def _setup_processor(self) -> None:
        """Set up low-pass filter processor"""
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff / nyquist

        # フィルタ係数を事前計算して保存
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype="low")  # type: ignore [unused-ignore]
        logger.debug(f"Lowpass filter coefficients calculated: b={self.b}, a={self.a}")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Filter processing wrapped with @dask.delayed"""
        logger.debug(f"Applying lowpass filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)

        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class AWeighting(AudioOperation[NDArrayReal, NDArrayReal]):
    """A-weighting filter operation"""

    name = "a_weighting"

    def __init__(self, sampling_rate: float):
        """
        Initialize A-weighting filter

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        super().__init__(sampling_rate)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for A-weighting filter"""
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
    """HPSS Harmonic operation"""

    name = "hpss_harmonic"

    def __init__(
        self,
        sampling_rate: float,
        **kwargs: Any,
    ):
        """
        Initialize HPSS Harmonic

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        self.kwargs = kwargs
        super().__init__(sampling_rate, **kwargs)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for HPSS Harmonic"""
        logger.debug(f"Applying HPSS Harmonic to array with shape: {x.shape}")
        result = librosa.effects.harmonic(x, **self.kwargs)
        logger.debug(
            f"HPSS Harmonic applied, returning result with shape: {result.shape}"
        )
        return result


class HpssPercussive(AudioOperation[NDArrayReal, NDArrayReal]):
    """HPSS Percussive operation"""

    name = "hpss_percussive"

    def __init__(
        self,
        sampling_rate: float,
        **kwargs: Any,
    ):
        """
        Initialize HPSS Percussive

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        self.kwargs = kwargs
        super().__init__(sampling_rate, **kwargs)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for HPSS Percussive"""
        logger.debug(f"Applying HPSS Percussive to array with shape: {x.shape}")
        result = librosa.effects.percussive(x, **self.kwargs)
        logger.debug(
            f"HPSS Percussive applied, returning result with shape: {result.shape}"
        )
        return result


class ReSampling(AudioOperation[NDArrayReal, NDArrayReal]):
    """Resampling operation"""

    name = "resampling"

    def __init__(self, sampling_rate: float, target_sr: float):
        """
        Initialize resampling operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        target_sampling_rate : float
            Target sampling rate (Hz)
        """
        super().__init__(sampling_rate, target_sr=target_sr)
        self.target_sr = target_sr

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape

        Returns
        -------
        tuple
            Output data shape
        """
        # Calculate length after resampling
        ratio = float(self.target_sr) / float(self.sampling_rate)
        n_samples = int(np.ceil(input_shape[-1] * ratio))
        return (*input_shape[:-1], n_samples)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for resampling operation"""
        logger.debug(f"Applying resampling to array with shape: {x.shape}")
        result = librosa.resample(
            x, orig_sr=self.sampling_rate, target_sr=self.target_sr
        )
        logger.debug(f"Resampling applied, returning result with shape: {result.shape}")
        return result


class ABS(AudioOperation[NDArrayReal, NDArrayReal]):
    """Absolute value operation"""

    name = "abs"

    def __init__(self, sampling_rate: float):
        """
        Initialize absolute value operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        super().__init__(sampling_rate)

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return da.abs(data)  # type: ignore [unused-ignore]


class Power(AudioOperation[NDArrayReal, NDArrayReal]):
    """Power operation"""

    name = "power"

    def __init__(self, sampling_rate: float, exponent: float):
        """
        Initialize power operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        exponent : float
            Power exponent
        """
        super().__init__(sampling_rate)
        self.exp = exponent

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return da.power(data, self.exp)  # type: ignore [unused-ignore]


class Trim(AudioOperation[NDArrayReal, NDArrayReal]):
    """Trimming operation"""

    name = "trim"

    def __init__(
        self,
        sampling_rate: float,
        start: float,
        end: float,
    ):
        """
        Initialize trimming operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        start : float
            Start time for trimming (seconds)
        end : float
            End time for trimming (seconds)
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
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape

        Returns
        -------
        tuple
            Output data shape
        """
        # Calculate length after trimming
        # Exclude parts where there is no signal
        end_sample = min(self.end_sample, input_shape[-1])
        n_samples = end_sample - self.start_sample
        return (*input_shape[:-1], n_samples)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for trimming operation"""
        logger.debug(f"Applying trim to array with shape: {x.shape}")
        # Apply trimming
        result = x[..., self.start_sample : self.end_sample]
        logger.debug(f"Trim applied, returning result with shape: {result.shape}")
        return result


class RmsTrend(AudioOperation[NDArrayReal, NDArrayReal]):
    """RMS calculation"""

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
        Initialize RMS calculation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        frame_length : int
            Frame length, default is 2048
        hop_length : int
            Hop length, default is 512
        ref : Union[list[float], float]
            Reference value(s) for dB calculation
        dB : bool
            Whether to convert to decibels
        Aw : bool
            Whether to apply A-weighting before RMS calculation
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
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, samples)

        Returns
        -------
        tuple
            Output data shape (channels, frames)
        """
        n_frames = librosa.feature.rms(
            y=np.ones((1, input_shape[-1])),
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        ).shape[-1]
        return (*input_shape[:-1], n_frames)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for RMS calculation"""
        logger.debug(f"Applying RMS to array with shape: {x.shape}")

        if self.Aw:
            # Apply A-weighting
            _x = A_weight(x, self.sampling_rate)
            if isinstance(_x, np.ndarray):
                # A_weightがタプルを返す場合、最初の要素を使用
                x = _x
            elif isinstance(_x, tuple):
                # Use the first element if A_weight returns a tuple
                x = _x[0]
            else:
                raise ValueError("A_weighting returned an unexpected type.")

        # Calculate RMS
        result = librosa.feature.rms(
            y=x, frame_length=self.frame_length, hop_length=self.hop_length
        )[..., 0, :]

        if self.dB:
            # Convert to dB
            result = 20 * np.log10(
                np.maximum(result / self.ref[..., np.newaxis], 1e-12)
            )
        #
        logger.debug(f"RMS applied, returning result with shape: {result.shape}")
        return result


class Sum(AudioOperation[NDArrayReal, NDArrayReal]):
    """Sum calculation"""

    name = "sum"

    def process(self, data: DaArray) -> DaArray:
        # Use Dask's aggregate function directly without map_blocks
        return data.sum(axis=0, keepdims=True)


class Mean(AudioOperation[NDArrayReal, NDArrayReal]):
    """Mean calculation"""

    name = "mean"

    def process(self, data: DaArray) -> DaArray:
        # Use Dask's aggregate function directly without map_blocks
        return data.mean(axis=0, keepdims=True)


class ChannelDifference(AudioOperation[NDArrayReal, NDArrayReal]):
    """Channel difference calculation operation"""

    name = "channel_difference"
    other_channel: int

    def __init__(self, sampling_rate: float, other_channel: int = 0):
        """
        Initialize channel difference calculation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        other_channel : int
            Channel to calculate difference with, default is 0
        """
        self.other_channel = other_channel
        super().__init__(sampling_rate, other_channel=other_channel)

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        result = data - data[self.other_channel]
        return result


class FFT(AudioOperation[NDArrayReal, NDArrayComplex]):
    """FFT (Fast Fourier Transform) operation"""

    name = "fft"
    n_fft: Optional[int]
    window: str

    def __init__(
        self, sampling_rate: float, n_fft: Optional[int] = None, window: str = "hann"
    ):
        """
        Initialize FFT operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        n_fft : int, optional
            FFT size, default is None (determined by input size)
        window : str, optional
            Window function type, default is 'hann'
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
    """IFFT (Inverse Fast Fourier Transform) operation"""

    name = "ifft"
    n_fft: Optional[int]
    window: str

    def __init__(
        self, sampling_rate: float, n_fft: Optional[int] = None, window: str = "hann"
    ):
        """
        Initialize IFFT operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        n_fft : Optional[int], optional
            IFFT size, default is None (determined based on input size)
        window : str, optional
            Window function type, default is 'hann'
        """
        self.n_fft = n_fft
        self.window = window
        super().__init__(sampling_rate, n_fft=n_fft, window=window)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, freqs)

        Returns
        -------
        tuple
            Output data shape (channels, samples)
        """
        n_samples = 2 * (input_shape[-1] - 1) if self.n_fft is None else self.n_fft
        return (*input_shape[:-1], n_samples)

    def _process_array(self, x: NDArrayComplex) -> NDArrayReal:
        """Create processor function for IFFT operation"""
        logger.debug(f"Applying IFFT to array with shape: {x.shape}")

        # Restore frequency component scaling (remove the 2.0 multiplier applied in FFT)
        _x = x.copy()
        _x[..., 1:-1] /= 2.0

        # Execute IFFT
        result: NDArrayReal = np.fft.irfft(_x, n=self.n_fft, axis=-1)

        # Window function correction (inverse of FFT operation)
        from scipy.signal import get_window

        win = get_window(self.window, result.shape[-1])

        # Correct the FFT window function scaling
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
        Initialize Welch operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        n_fft : int, optional
            FFT size, default is 2048
        window : str, optional
            Window function type, default is 'hann'
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
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, samples)

        Returns
        -------
        tuple
            Output data shape (channels, freqs)
        """
        n_freqs = self.n_fft // 2 + 1
        return (*input_shape[:-1], n_freqs)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for Welch operation"""
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
            # Trigger computation for Dask array
            raise ValueError(
                "Welch operation requires a Dask array, but received a non-ndarray."
            )
        return np.array(result)


class NOctSpectrum(AudioOperation[NDArrayReal, NDArrayReal]):
    """N-octave spectrum operation"""

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
        Initialize N-octave spectrum

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        fmin : float
            Minimum frequency (Hz)
        fmax : float
            Maximum frequency (Hz)
        n : int, optional
            Number of octave divisions, default is 3
        G : int, optional
            Reference level, default is 10
        fr : int, optional
            Reference frequency, default is 1000
        """
        super().__init__(sampling_rate, fmin=fmin, fmax=fmax, n=n, G=G, fr=fr)
        self.fmin = fmin
        self.fmax = fmax
        self.n = n
        self.G = G
        self.fr = fr

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape

        Returns
        -------
        tuple
            Output data shape
        """
        # Calculate output shape for octave spectrum
        _, fpref = _center_freq(
            fmin=self.fmin, fmax=self.fmax, n=self.n, G=self.G, fr=self.fr
        )
        return (input_shape[0], fpref.shape[0])

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for octave spectrum"""
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
            # Add channel dimension for 1D
            spec = np.expand_dims(spec, axis=0)
        else:
            spec = spec.T
        logger.debug(f"NoctSpectrum applied, returning result with shape: {spec.shape}")
        return np.array(spec)


class NOctSynthesis(AudioOperation[NDArrayReal, NDArrayReal]):
    """Octave synthesis operation"""

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
        Initialize octave synthesis

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        fmin : float
            Minimum frequency (Hz)
        fmax : float
            Maximum frequency (Hz)
        n : int, optional
            Number of octave divisions, default is 3
        G : int, optional
            Reference level, default is 10
        fr : int, optional
            Reference frequency, default is 1000
        """
        super().__init__(sampling_rate, fmin=fmin, fmax=fmax, n=n, G=G, fr=fr)

        self.fmin = fmin
        self.fmax = fmax
        self.n = n
        self.G = G
        self.fr = fr

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape

        Returns
        -------
        tuple
            Output data shape
        """
        # Calculate output shape for octave spectrum
        _, fpref = _center_freq(
            fmin=self.fmin, fmax=self.fmax, n=self.n, G=self.G, fr=self.fr
        )
        return (input_shape[0], fpref.shape[0])

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for octave synthesis"""
        logger.debug(f"Applying NoctSynthesis to array with shape: {x.shape}")
        # Calculate n from shape[-1]
        n = x.shape[-1]  # Calculate n from shape[-1]
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
    """Short-Time Fourier Transform operation"""

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
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape

        Returns
        -------
        tuple
            Output data shape
        """
        n_samples = input_shape[-1]
        n_f = len(self.SFT.f)
        n_t = len(self.SFT.t(n_samples))
        return (input_shape[0], n_f, n_t)

    def _process_array(self, x: NDArrayReal) -> NDArrayComplex:
        """Apply SciPy STFT processing to multiple channels at once"""
        logger.debug(f"Applying SciPy STFT to array with shape: {x.shape}")

        # Convert 1D input to 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Apply STFT to all channels at once
        result: NDArrayComplex = self.SFT.stft(x)
        result[..., 1:-1, :] *= 2.0
        logger.debug(f"SciPy STFT applied, returning result with shape: {result.shape}")
        return result


class ISTFT(AudioOperation[NDArrayComplex, NDArrayReal]):
    """Inverse Short-Time Fourier Transform operation"""

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
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, freqs, time_frames)

        Returns
        -------
        tuple
            Output data shape (channels, samples)
        """
        k0: int = 0
        q_max = input_shape[-1] + self.SFT.p_min
        k_max = (q_max - 1) * self.SFT.hop + self.SFT.m_num - self.SFT.m_num_mid
        k_q0, k_q1 = self.SFT.nearest_k_p(k0), self.SFT.nearest_k_p(k_max, left=False)
        n_pts = k_q1 - k_q0 + self.SFT.m_num - self.SFT.m_num_mid

        return input_shape[:-2] + (n_pts,)

    def _process_array(self, x: NDArrayComplex) -> NDArrayReal:
        """
        Apply SciPy ISTFT processing to multiple channels at once using ShortTimeFFT"""
        logger.debug(
            f"Applying SciPy ISTFT (ShortTimeFFT) to array with shape: {x.shape}"
        )

        # Convert 2D input to 3D (assume single channel)
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
    """Addition operation considering SNR"""

    name = "add_with_snr"

    def __init__(self, sampling_rate: float, other: DaArray, snr: float):
        """
        Initialize addition operation considering SNR

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        other : DaArray
            Noise signal to add (channel-frame format)
        snr : float
            Signal-to-noise ratio (dB)
        """
        super().__init__(sampling_rate, other=other, snr=snr)

        self.other = other
        self.snr = snr
        logger.debug(f"Initialized AddWithSNR operation with SNR: {snr} dB")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape

        Returns
        -------
        tuple
            Output data shape (same as input)
        """
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Perform addition processing considering SNR"""
        logger.debug(f"Applying SNR-based addition with shape: {x.shape}")
        other: NDArrayReal = self.other.compute()

        # Use multi-channel versions of calculate_rms and calculate_desired_noise_rms
        clean_rms = util.calculate_rms(x)
        other_rms = util.calculate_rms(other)

        # Adjust noise gain based on specified SNR (apply per channel)
        desired_noise_rms = util.calculate_desired_noise_rms(clean_rms, self.snr)

        # Apply gain per channel using broadcasting
        gain = desired_noise_rms / other_rms
        # Add adjusted noise to signal
        result: NDArrayReal = x + other * gain
        return result


class Coherence(AudioOperation[NDArrayReal, NDArrayReal]):
    """Coherence estimation operation"""

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
        Initialize coherence estimation operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        n_fft : int
            FFT size
        hop_length : int
            Hop length
        win_length : int
            Window length
        window : str
            Window function
        detrend : str
            Type of detrend
        """
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.window = window
        self.detrend = detrend
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            detrend=detrend,
        )

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, samples)

        Returns
        -------
        tuple
            Output data shape (channels * channels, freqs)
        """
        n_channels = input_shape[0]
        n_freqs = self.n_fft // 2 + 1
        return (n_channels * n_channels, n_freqs)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Processor function for coherence estimation operation"""
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

        # Reshape result to (n_channels * n_channels, n_freqs)
        result: NDArrayReal = coh.transpose(1, 0, 2).reshape(-1, coh.shape[-1])

        logger.debug(f"Coherence estimation applied, result shape: {result.shape}")
        return result


class CSD(AudioOperation[NDArrayReal, NDArrayComplex]):
    """Cross-spectral density estimation operation"""

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
        Initialize cross-spectral density estimation operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        n_fft : int
            FFT size
        hop_length : int
            Hop length
        win_length : int
            Window length
        window : str
            Window function
        detrend : str
            Type of detrend
        scaling : str
            Type of scaling
        average : str
            Method of averaging
        """
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.window = window
        self.detrend = detrend
        self.scaling = scaling
        self.average = average
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, samples)

        Returns
        -------
        tuple
            Output data shape (channels * channels, freqs)
        """
        n_channels = input_shape[0]
        n_freqs = self.n_fft // 2 + 1
        return (n_channels * n_channels, n_freqs)

    def _process_array(self, x: NDArrayReal) -> NDArrayComplex:
        """Processor function for cross-spectral density estimation operation"""
        logger.debug(f"Applying CSD estimation to array with shape: {x.shape}")
        from scipy import signal as ss

        # Calculate all combinations using scipy's csd function
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

        # Reshape result to (n_channels * n_channels, n_freqs)
        result: NDArrayComplex = csd_result.transpose(1, 0, 2).reshape(
            -1, csd_result.shape[-1]
        )

        logger.debug(f"CSD estimation applied, result shape: {result.shape}")
        return result


class TransferFunction(AudioOperation[NDArrayReal, NDArrayComplex]):
    """Transfer function estimation operation"""

    name = "transfer_function"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        detrend: str,
        scaling: str = "spectrum",
        average: str = "mean",
    ):
        """
        Initialize transfer function estimation operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        n_fft : int
            FFT size
        hop_length : int
            Hop length
        win_length : int
            Window length
        window : str
            Window function
        detrend : str
            Type of detrend
        scaling : str
            Type of scaling
        average : str
            Method of averaging
        """
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.window = window
        self.detrend = detrend
        self.scaling = scaling
        self.average = average
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, samples)

        Returns
        -------
        tuple
            Output data shape (channels * channels, freqs)
        """
        n_channels = input_shape[0]
        n_freqs = self.n_fft // 2 + 1
        return (n_channels * n_channels, n_freqs)

    def _process_array(self, x: NDArrayReal) -> NDArrayComplex:
        """Processor function for transfer function estimation operation"""
        logger.debug(
            f"Applying transfer function estimation to array with shape: {x.shape}"
        )
        from scipy import signal as ss

        # Calculate cross-spectral density between all channels
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

        # Calculate power spectral density for each channel
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

        # Calculate transfer function H(f) = P_yx / P_xx
        h_f = p_yx / p_xx[np.newaxis, :, :]
        result: NDArrayComplex = h_f.transpose(1, 0, 2).reshape(-1, h_f.shape[-1])

        logger.debug(
            f"Transfer function estimation applied, result shape: {result.shape}"
        )
        return result


# Automatically collect operation types and corresponding classes
_OPERATION_REGISTRY: dict[str, type[AudioOperation[Any, Any]]] = {}


def register_operation(operation_class: type) -> None:
    """Register a new operation type"""

    if not issubclass(operation_class, AudioOperation):
        raise TypeError("Strategy class must inherit from AudioOperation.")
    if inspect.isabstract(operation_class):
        raise TypeError("Cannot register abstract AudioOperation class.")

    _OPERATION_REGISTRY[operation_class.name] = operation_class


for strategy_cls in AudioOperation.__subclasses__():
    if not inspect.isabstract(strategy_cls):
        register_operation(strategy_cls)


def get_operation(name: str) -> type[AudioOperation[Any, Any]]:
    """Get operation class by name"""
    if name not in _OPERATION_REGISTRY:
        raise ValueError(f"Unknown operation type: {name}")
    return _OPERATION_REGISTRY[name]


def create_operation(
    name: str, sampling_rate: float, **params: Any
) -> AudioOperation[Any, Any]:
    """Create operation instance from name and parameters"""
    operation_class = get_operation(name)
    return operation_class(sampling_rate, **params)
