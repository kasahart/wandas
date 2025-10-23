import logging
from typing import Optional

import numpy as np
from mosqito.sound_level_meter import noct_spectrum, noct_synthesis
from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import get_window

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayComplex, NDArrayReal

logger = logging.getLogger(__name__)


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

        if self.n_fft is not None and x.shape[-1] > self.n_fft:
            # If n_fft is specified and input length exceeds it, truncate
            x = x[..., : self.n_fft]

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


# Register all operations
for op_class in [
    FFT,
    IFFT,
    STFT,
    ISTFT,
    Welch,
    NOctSpectrum,
    NOctSynthesis,
    Coherence,
    CSD,
    TransferFunction,
]:
    register_operation(op_class)
