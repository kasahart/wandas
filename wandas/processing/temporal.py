import logging
from typing import Any

import librosa
import numpy as np
from scipy.signal import lfilter

from wandas.processing.base import AudioOperation, register_operation
from wandas.processing.weighting import A_weight, frequency_weight
from wandas.utils import validate_sampling_rate
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


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

        Raises
        ------
        ValueError
            If sampling_rate or target_sr is not positive
        """
        validate_sampling_rate(sampling_rate, "source sampling rate")
        validate_sampling_rate(target_sr, "target sampling rate")
        super().__init__(sampling_rate, target_sr=target_sr)
        self.target_sr = target_sr

    def get_metadata_updates(self) -> dict[str, Any]:
        """
        Update sampling rate to target sampling rate.

        Returns
        -------
        dict
            Metadata updates with new sampling rate

        Notes
        -----
        Resampling always produces output at target_sr, regardless of input
        sampling rate. All necessary parameters are provided at initialization.
        """
        return {"sampling_rate": self.target_sr}

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

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return "rs"

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for resampling operation"""
        logger.debug(f"Applying resampling to array with shape: {x.shape}")
        result: NDArrayReal = librosa.resample(x, orig_sr=self.sampling_rate, target_sr=self.target_sr)
        logger.debug(f"Resampling applied, returning result with shape: {result.shape}")
        return result


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
        logger.debug(f"Initialized Trim operation with start: {self.start}, end: {self.end}")

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

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return "trim"

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for trimming operation"""
        logger.debug(f"Applying trim to array with shape: {x.shape}")
        # Apply trimming
        result = x[..., self.start_sample : self.end_sample]
        logger.debug(f"Trim applied, returning result with shape: {result.shape}")
        return result


class FixLength(AudioOperation[NDArrayReal, NDArrayReal]):
    """信号の長さを指定された長さに調整する操作"""

    name = "fix_length"

    def __init__(
        self,
        sampling_rate: float,
        length: int | None = None,
        duration: float | None = None,
    ):
        """
        Initialize fix length operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        length : Optional[int]
            Target length for fixing
        duration : Optional[float]
            Target length for fixing
        """
        if length is None:
            if duration is None:
                raise ValueError("Either length or duration must be provided.")
            else:
                length = int(duration * sampling_rate)
        self.target_length = length

        super().__init__(sampling_rate, target_length=self.target_length)

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
        return (*input_shape[:-1], self.target_length)

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return "fix"

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for padding operation"""
        logger.debug(f"Applying padding to array with shape: {x.shape}")
        # Apply padding
        pad_width = self.target_length - x.shape[-1]
        if pad_width > 0:
            result = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
        else:
            result = x[..., : self.target_length]
        logger.debug(f"Padding applied, returning result with shape: {result.shape}")
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
        ref: list[float] | float = 1.0,
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

    def get_metadata_updates(self) -> dict[str, Any]:
        """
        Update sampling rate based on hop length.

        Returns
        -------
        dict
            Metadata updates with new sampling rate based on hop length

        Notes
        -----
        The output sampling rate is determined by downsampling the input
        by hop_length. All necessary parameters are provided at initialization.
        """
        new_sr = self.sampling_rate / self.hop_length
        return {"sampling_rate": new_sr}

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

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return "RMS"

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
        result: NDArrayReal = librosa.feature.rms(y=x, frame_length=self.frame_length, hop_length=self.hop_length)[
            ..., 0, :
        ]

        if self.dB:
            # Convert to dB
            result = 20 * np.log10(np.maximum(result / self.ref[..., np.newaxis], 1e-12))
        #
        logger.debug(f"RMS applied, returning result with shape: {result.shape}")
        return result


class SoundLevel(AudioOperation[NDArrayReal, NDArrayReal]):
    """Time-varying sound pressure level with frequency and time weighting."""

    name = "sound_level"

    def __init__(
        self,
        sampling_rate: float,
        ref: list[float] | float = 1.0,
        freq_weighting: str | None = "Z",
        time_weighting: str = "Fast",
    ) -> None:
        validate_sampling_rate(sampling_rate)
        self.ref = np.atleast_1d(np.asarray(ref, dtype=float))
        if np.any(self.ref <= 0):
            raise ValueError(
                "Invalid sound level reference\n"
                f"  Got: {self.ref.tolist()}\n"
                "  Expected: Positive reference values\n"
                "Sound pressure level requires a positive reference pressure."
            )
        self.freq_weighting = self._normalize_freq_weighting(freq_weighting)
        self.time_weighting = self._normalize_time_weighting(time_weighting)
        super().__init__(
            sampling_rate,
            ref=self.ref,
            freq_weighting=self.freq_weighting,
            time_weighting=self.time_weighting,
        )

    @staticmethod
    def _normalize_freq_weighting(freq_weighting: str | None) -> str:
        normalized = "Z" if freq_weighting is None else str(freq_weighting).upper()
        if normalized not in {"A", "C", "Z"}:
            raise ValueError(
                "Invalid frequency weighting\n"
                f"  Got: {freq_weighting!r}\n"
                "  Expected: 'A', 'C', or 'Z'\n"
                "Choose a supported IEC-style weighting curve before calculating sound level."
            )
        return normalized

    @staticmethod
    def _normalize_time_weighting(time_weighting: str) -> str:
        normalized = str(time_weighting).strip().upper()
        if normalized in {"F", "FAST"}:
            return "Fast"
        if normalized in {"S", "SLOW"}:
            return "Slow"
        raise ValueError(
            "Invalid time weighting\n"
            f"  Got: {time_weighting!r}\n"
            "  Expected: 'Fast' or 'Slow'\n"
            "Choose a supported sound level meter time constant before calculating sound level."
        )

    @property
    def time_constant(self) -> float:
        """Return the RC time constant in seconds."""
        return 0.125 if self.time_weighting == "Fast" else 1.0

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Sound level keeps the same channel and sample dimensions."""
        return input_shape

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return f"L{self.freq_weighting}{self.time_weighting[0]}"

    def _reference_squared(self, n_channels: int) -> NDArrayReal:
        """Return squared reference pressure for each channel."""
        if self.ref.size == 1:
            ref = np.repeat(self.ref, n_channels)
        elif self.ref.size == n_channels:
            ref = self.ref
        else:
            raise ValueError(
                "Reference count mismatch\n"
                f"  Got: {self.ref.size} reference values for {n_channels} channels\n"
                "  Expected: One shared reference or one reference per channel\n"
                "Provide ref as a scalar or a list matching the number of channels."
            )
        return np.asarray(np.square(ref), dtype=np.float64)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for sound level calculation."""
        logger.debug(
            "Applying sound level to array with shape %s using %s/%s weighting",
            x.shape,
            self.freq_weighting,
            self.time_weighting,
        )
        weighted = frequency_weight(np.asarray(x, dtype=np.float64), self.sampling_rate, curve=self.freq_weighting)
        squared = np.square(weighted)
        alpha = float(np.exp(-1.0 / (self.sampling_rate * self.time_constant)))
        smoothed = lfilter([1.0 - alpha], [1.0, -alpha], squared, axis=-1)
        ref_squared = self._reference_squared(smoothed.shape[0])[:, np.newaxis]
        result = np.asarray(
            10.0 * np.log10(np.maximum(smoothed / ref_squared, np.finfo(np.float64).tiny)),
            dtype=np.float64,
        )
        logger.debug(f"Sound level applied, returning result with shape: {result.shape}")
        return result


# Register all operations
for op_class in [ReSampling, Trim, RmsTrend, FixLength, SoundLevel]:
    register_operation(op_class)
