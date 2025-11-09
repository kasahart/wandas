import logging
from typing import Any, Optional, Union

import librosa
import numpy as np
from waveform_analysis import A_weight

from wandas.processing.base import AudioOperation, register_operation
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
        result: NDArrayReal = librosa.resample(
            x, orig_sr=self.sampling_rate, target_sr=self.target_sr
        )
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
        length: Optional[int] = None,
        duration: Optional[float] = None,
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
        result: NDArrayReal = librosa.feature.rms(
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


# Register all operations
for op_class in [ReSampling, Trim, RmsTrend, FixLength]:
    register_operation(op_class)
