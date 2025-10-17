"""Psychoacoustic metrics operations using MoSQITo.

This module provides psychoacoustic metrics operations for sound quality analysis.
"""

import logging
from typing import Literal, Optional

import numpy as np
from mosqito.sq_metrics import roughness_dw_freq, roughness_dw_time

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


class RoughnessDW(AudioOperation[NDArrayReal, NDArrayReal]):
    """Roughness calculation using Daniel & Weber method.

    This operation computes roughness according to the Daniel and Weber method,
    which is a widely used psychoacoustic metric for assessing the perceived
    roughness of a sound. Roughness is related to rapid amplitude modulations
    in the 15-300 Hz range.

    The operation supports both time-domain and frequency-domain calculation methods.
    """

    name = "roughness_dw"

    def __init__(
        self,
        sampling_rate: float,
        method: Literal["time", "freq"] = "time",
        overlap: float = 0.0,
    ):
        """Initialize Roughness calculation operation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate in Hz.
        method : {'time', 'freq'}, default='time'
            Calculation method:
            - 'time': Time-domain calculation using `roughness_dw_time`
            - 'freq': Frequency-domain calculation using `roughness_dw_freq`
        overlap : float, default=0.0
            Overlap ratio for time-domain method (0.0 to 1.0).
            Only applicable when method='time'.

        Raises
        ------
        ValueError
            If method is not 'time' or 'freq', or if overlap is not in [0, 1].
        """
        if method not in ("time", "freq"):
            raise ValueError(
                f"Invalid method: {method}. Must be 'time' or 'freq'."
            )
        if not 0.0 <= overlap <= 1.0:
            raise ValueError(
                f"Invalid overlap: {overlap}. Must be between 0.0 and 1.0."
            )

        self.method = method
        self.overlap = overlap
        super().__init__(
            sampling_rate,
            method=method,
            overlap=overlap,
        )

    def validate_params(self) -> None:
        """Validate parameters for roughness calculation."""
        # Additional validation if needed
        pass

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output data shape after roughness calculation.

        The roughness operation returns a single scalar value per channel.

        Parameters
        ----------
        input_shape : tuple
            Input data shape (n_channels, n_samples)

        Returns
        -------
        tuple
            Output data shape (n_channels, 1) for scalar roughness per channel
        """
        return (input_shape[0], 1)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Process array to calculate roughness.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (n_channels, n_samples)

        Returns
        -------
        NDArrayReal
            Roughness values with shape (n_channels, 1)
        """
        logger.debug(
            f"Calculating roughness using {self.method} method for "
            f"array with shape: {x.shape}"
        )

        n_channels = x.shape[0]
        roughness_values = np.zeros((n_channels, 1), dtype=np.float64)

        for i in range(n_channels):
            signal = x[i, :]

            if self.method == "time":
                # Time-domain roughness calculation
                r_result = roughness_dw_time(
                    signal,
                    fs=self.sampling_rate,
                    overlap=self.overlap,
                )
                # Extract the overall roughness value
                # roughness_dw_time returns (R, R_specific, bark_axis, time_axis)
                if isinstance(r_result, tuple):
                    roughness_values[i, 0] = float(np.mean(r_result[0]))
                else:
                    roughness_values[i, 0] = float(r_result)

            elif self.method == "freq":
                # Frequency-domain roughness calculation
                # First, compute FFT of the signal
                spectrum = np.fft.rfft(signal)
                freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sampling_rate)

                # Calculate magnitude spectrum
                magnitude = np.abs(spectrum)

                # Compute roughness from frequency spectrum
                r_result = roughness_dw_freq(magnitude, freqs)
                # roughness_dw_freq returns (R, R_specific, bark_axis)
                if isinstance(r_result, tuple):
                    roughness_values[i, 0] = float(r_result[0])
                else:
                    roughness_values[i, 0] = float(r_result)

        logger.debug(
            f"Roughness calculation complete. Result shape: {roughness_values.shape}"
        )
        return roughness_values


# Register the operation
register_operation(RoughnessDW)
