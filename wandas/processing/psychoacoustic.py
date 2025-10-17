"""
Psychoacoustic metrics processing operations.

This module provides psychoacoustic metrics operations for audio signals,
including loudness calculation using standardized methods.
"""

import logging

import numpy as np
from mosqito.sq_metrics import loudness_zwtv as loudness_zwtv_mosqito

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


class LoudnessZwtv(AudioOperation[NDArrayReal, NDArrayReal]):
    """
    Calculate time-varying loudness using Zwicker method (ISO 532-1:2017).

    This operation computes the loudness of non-stationary signals according to
    the Zwicker method, as specified in ISO 532-1:2017. It uses the MoSQITo library's
    implementation of the standardized loudness calculation.

    The loudness is calculated in sones, a unit of perceived loudness where a doubling
    of sones corresponds to a doubling of perceived loudness.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz. The signal should be sampled at a rate appropriate
        for the analysis (typically 44100 Hz or 48000 Hz for audio).
    field_type : str, default="free"
        Type of sound field. Options:
        - 'free': Free field (sound arriving from a specific direction)
        - 'diffuse': Diffuse field (sound arriving uniformly from all directions)

    Attributes
    ----------
    name : str
        Operation name: "loudness_zwtv"
    field_type : str
        The sound field type used for calculation

    Examples
    --------
    Calculate loudness for a signal:
    >>> import wandas as wd
    >>> signal = wd.read_wav("audio.wav")
    >>> loudness = signal.loudness_zwtv(field_type="free")

    Notes
    -----
    - The output contains time-varying loudness values in sones
    - For mono signals, the loudness is calculated directly
    - For multi-channel signals, loudness is calculated per channel
    - The method follows ISO 532-1:2017 standard for time-varying loudness
    - Typical loudness values: 1 sone ≈ 40 phon (loudness level)

    References
    ----------
    .. [1] ISO 532-1:2017, "Acoustics — Methods for calculating loudness —
           Part 1: Zwicker method"
    .. [2] MoSQITo documentation:
           https://mosqito.readthedocs.io/en/latest/
    """

    name = "loudness_zwtv"

    def __init__(self, sampling_rate: float, field_type: str = "free"):
        """
        Initialize Loudness calculation operation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        field_type : str, default="free"
            Type of sound field ('free' or 'diffuse')
        """
        self.field_type = field_type
        super().__init__(sampling_rate, field_type=field_type)

    def validate_params(self) -> None:
        """
        Validate parameters.

        Raises
        ------
        ValueError
            If field_type is not 'free' or 'diffuse'
        """
        if self.field_type not in ("free", "diffuse"):
            raise ValueError(
                f"field_type must be 'free' or 'diffuse', got '{self.field_type}'"
            )

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation.

        The loudness calculation produces a time-varying output where the time
        resolution depends on the algorithm's internal processing. The exact
        output length is determined dynamically by the loudness_zwtv function.

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, samples)

        Returns
        -------
        tuple
            Output data shape. For loudness, we return a placeholder shape
            since the actual length is determined by the algorithm.
            The shape will be (channels, time_samples) where time_samples
            depends on the input length and algorithm parameters.
        """
        # Return a placeholder shape - the actual shape will be determined
        # after processing since loudness_zwtv determines the time resolution
        # For now, we estimate based on typical behavior (approx 2ms time steps)
        n_channels = input_shape[0] if len(input_shape) > 1 else 1
        # Rough estimate: one loudness value per 2ms (0.002s)
        estimated_time_samples = int(input_shape[-1] / (self.sampling_rate * 0.002))
        return (n_channels, estimated_time_samples)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process array to calculate loudness.

        This method calculates the time-varying loudness for each channel
        of the input signal using the Zwicker method.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (channels, samples) or (samples,)

        Returns
        -------
        NDArrayReal
            Time-varying loudness in sones for each channel.
            Shape: (channels, time_samples)

        Notes
        -----
        The function processes each channel independently and returns
        the loudness values. The time axis information is not returned
        here but can be reconstructed based on the MoSQITo algorithm's
        behavior (typically 2ms time steps).
        """
        logger.debug(
            f"Calculating loudness for signal with shape: {x.shape}, "
            f"field_type: {self.field_type}"
        )

        # Handle 1D input (single channel)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_channels = x.shape[0]
        loudness_results = []

        for ch in range(n_channels):
            channel_data = x[ch, :]

            # Call MoSQITo's loudness_zwtv function
            # Returns: N (loudness), N_spec (specific loudness),
            #          bark_axis, time_axis
            loudness_n, _, _, _ = loudness_zwtv_mosqito(
                channel_data, self.sampling_rate, field_type=self.field_type
            )

            loudness_results.append(loudness_n)

            logger.debug(
                f"Channel {ch}: Calculated loudness with "
                f"{len(loudness_n)} time points, "
                f"max loudness: {np.max(loudness_n):.2f} sones"
            )

        # Stack results
        result: NDArrayReal = np.stack(loudness_results, axis=0)

        logger.debug(f"Loudness calculation complete, output shape: {result.shape}")
        return result


# Register the operation
register_operation(LoudnessZwtv)
