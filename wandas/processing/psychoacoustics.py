"""
Psychoacoustic metrics operations.

This module provides psychoacoustic metrics operations for audio signals,
including loudness calculation based on ISO 532-1:2017 using the MoSQITo library.
"""

import logging
from typing import Any, Literal, Optional, cast

import dask
import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)

_dask_delayed = dask.delayed  # type: ignore [unused-ignore]
_da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]


class LoudnessZwtv(AudioOperation[NDArrayReal, dict[str, Any]]):
    """
    Time-varying loudness calculation using Zwicker method.

    This operation computes the acoustic loudness according to Zwicker method for
    time-varying signals (ISO 532-1:2017) using the MoSQITo library.

    The input signal is expected to be in Pascals (Pa). If your signal is in a
    different unit (e.g., normalized amplitude), you may need to convert it first.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz.
    field_type : {"free", "diffuse"}, default="free"
        Type of sound field:
        - "free": Free-field condition (anechoic environment)
        - "diffuse": Diffuse-field condition (reverberant environment)

    Attributes
    ----------
    name : str
        Operation name: "loudness_zwtv"

    Notes
    -----
    The loudness is calculated using the MoSQITo library's implementation of the
    Zwicker method, which is compliant with ISO 532-1:2017.

    References
    ----------
    .. [1] ISO 532-1:2017 - Acoustics — Methods for calculating loudness —
           Part 1: Zwicker method
    .. [2] MoSQITo documentation:
           https://mosqito.readthedocs.io/en/latest/source/reference/mosqito.sq_metrics.loudness.html

    Examples
    --------
    >>> import wandas as wd
    >>> signal = wd.read_wav("audio.wav")
    >>> # Note: Ensure signal is in Pascals (Pa)
    >>> loudness_op = LoudnessZwtv(sampling_rate=signal.sampling_rate)
    >>> result = loudness_op.process(signal.data)
    >>> N = result["N"]  # Overall loudness [sone]
    >>> N_spec = result["N_spec"]  # Specific loudness [sone/bark]
    >>> bark_axis = result["bark_axis"]  # Bark frequency scale
    >>> time_axis = result["time_axis"]  # Time axis [s]
    """

    name = "loudness_zwtv"

    def __init__(
        self,
        sampling_rate: float,
        field_type: Literal["free", "diffuse"] = "free",
    ) -> None:
        """
        Initialize time-varying loudness calculation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate in Hz.
        field_type : {"free", "diffuse"}, default="free"
            Type of sound field.

        Raises
        ------
        ValueError
            If field_type is not "free" or "diffuse".
        """
        if field_type not in ("free", "diffuse"):
            raise ValueError(
                f"field_type must be 'free' or 'diffuse', got '{field_type}'"
            )

        super().__init__(sampling_rate, field_type=field_type)
        self.field_type = field_type

    def validate_params(self) -> None:
        """Validate parameters."""
        if self.sampling_rate <= 0:
            raise ValueError(
                f"Sampling rate must be positive, got {self.sampling_rate}"
            )

    def _process_array(self, x: NDArrayReal) -> dict[str, Any]:
        """
        Process the input array to calculate time-varying loudness.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (n_channels, n_samples).
            Signal should be in Pascals (Pa).

        Returns
        -------
        dict
            Dictionary containing:
            - "N" : ndarray
                Overall loudness over time [sone], shape (n_time,)
            - "N_spec" : ndarray
                Specific loudness [sone/bark], shape (n_bark, n_time)
            - "bark_axis" : ndarray
                Bark frequency scale, shape (n_bark,)
            - "time_axis" : ndarray
                Time axis [s], shape (n_time,)
            - "n_channels" : int
                Number of channels processed

        Notes
        -----
        For multi-channel signals, each channel is processed separately and
        the results are averaged.
        """
        from mosqito.sq_metrics import loudness_zwtv

        logger.debug(
            f"Computing time-varying loudness: "
            f"shape={x.shape}, field_type={self.field_type}"
        )

        # Handle mono vs stereo
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_channels = x.shape[0]

        # Process each channel
        results_list = []
        for i in range(n_channels):
            channel_data = x[i, :]
            N, N_spec, bark_axis, time_axis = loudness_zwtv(
                channel_data, self.sampling_rate, field_type=self.field_type
            )
            results_list.append(
                {
                    "N": N,
                    "N_spec": N_spec,
                    "bark_axis": bark_axis,
                    "time_axis": time_axis,
                }
            )

        # Average results across channels if multi-channel
        if n_channels == 1:
            result = results_list[0]
        else:
            # Average N and N_spec across channels
            N_avg = np.mean([r["N"] for r in results_list], axis=0)
            N_spec_avg = np.mean([r["N_spec"] for r in results_list], axis=0)
            result = {
                "N": N_avg,
                "N_spec": N_spec_avg,
                "bark_axis": results_list[0]["bark_axis"],
                "time_axis": results_list[0]["time_axis"],
            }

        result["n_channels"] = n_channels

        logger.debug(
            f"Loudness calculation complete: "
            f"N.shape={result['N'].shape}, "
            f"N_spec.shape={result['N_spec'].shape}"
        )

        return result

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output shape after loudness calculation.

        Parameters
        ----------
        input_shape : tuple
            Input data shape (n_channels, n_samples).

        Returns
        -------
        tuple
            Output shape. Returns (1,) as a placeholder since actual output
            is a dictionary with variable-length arrays.
        """
        # Return a placeholder shape since the output is a dictionary
        return (1,)

    def process(self, data: DaArray) -> dict[str, Any]:
        """
        Execute loudness calculation and return result.

        Parameters
        ----------
        data : DaArray
            Input Dask array with shape (n_channels, n_samples).

        Returns
        -------
        dict
            Dictionary containing loudness metrics. This is computed immediately
            (not lazy) since MoSQITo functions are not dask-compatible.
        """
        logger.debug("Processing loudness (eager evaluation)")
        # MoSQITo functions are not dask-compatible, so we compute immediately
        data_np = data.compute()
        result = self._process_array(data_np)
        return result


class LoudnessZwst(AudioOperation[NDArrayReal, dict[str, Any]]):
    """
    Stationary loudness calculation using Zwicker method.

    This operation computes the acoustic loudness according to Zwicker method for
    stationary signals (ISO 532-1:2017) using the MoSQITo library.

    The input signal is expected to be in Pascals (Pa). If your signal is in a
    different unit (e.g., normalized amplitude), you may need to convert it first.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz.
    field_type : {"free", "diffuse"}, default="free"
        Type of sound field:
        - "free": Free-field condition (anechoic environment)
        - "diffuse": Diffuse-field condition (reverberant environment)

    Attributes
    ----------
    name : str
        Operation name: "loudness_zwst"

    Notes
    -----
    The loudness is calculated using the MoSQITo library's implementation of the
    Zwicker method for stationary signals, which is compliant with ISO 532-1:2017.

    References
    ----------
    .. [1] ISO 532-1:2017 - Acoustics — Methods for calculating loudness —
           Part 1: Zwicker method
    .. [2] MoSQITo documentation:
           https://mosqito.readthedocs.io/en/latest/source/reference/mosqito.sq_metrics.loudness.html

    Examples
    --------
    >>> import wandas as wd
    >>> signal = wd.read_wav("audio.wav")
    >>> # Note: Ensure signal is in Pascals (Pa)
    >>> loudness_op = LoudnessZwst(sampling_rate=signal.sampling_rate)
    >>> result = loudness_op.process(signal.data)
    >>> N = result["N"]  # Overall loudness [sone]
    >>> N_spec = result["N_spec"]  # Specific loudness [sone/bark]
    >>> bark_axis = result["bark_axis"]  # Bark frequency scale
    """

    name = "loudness_zwst"

    def __init__(
        self,
        sampling_rate: float,
        field_type: Literal["free", "diffuse"] = "free",
    ) -> None:
        """
        Initialize stationary loudness calculation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate in Hz.
        field_type : {"free", "diffuse"}, default="free"
            Type of sound field.

        Raises
        ------
        ValueError
            If field_type is not "free" or "diffuse".
        """
        if field_type not in ("free", "diffuse"):
            raise ValueError(
                f"field_type must be 'free' or 'diffuse', got '{field_type}'"
            )

        super().__init__(sampling_rate, field_type=field_type)
        self.field_type = field_type

    def validate_params(self) -> None:
        """Validate parameters."""
        if self.sampling_rate <= 0:
            raise ValueError(
                f"Sampling rate must be positive, got {self.sampling_rate}"
            )

    def _process_array(self, x: NDArrayReal) -> dict[str, Any]:
        """
        Process the input array to calculate stationary loudness.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (n_channels, n_samples).
            Signal should be in Pascals (Pa).

        Returns
        -------
        dict
            Dictionary containing:
            - "N" : float
                Overall loudness [sone]
            - "N_spec" : ndarray
                Specific loudness [sone/bark], shape (n_bark,)
            - "bark_axis" : ndarray
                Bark frequency scale, shape (n_bark,)
            - "n_channels" : int
                Number of channels processed

        Notes
        -----
        For multi-channel signals, each channel is processed separately and
        the results are averaged.
        """
        from mosqito.sq_metrics import loudness_zwst

        logger.debug(
            f"Computing stationary loudness: "
            f"shape={x.shape}, field_type={self.field_type}"
        )

        # Handle mono vs stereo
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_channels = x.shape[0]

        # Process each channel
        results_list = []
        for i in range(n_channels):
            channel_data = x[i, :]
            N, N_spec, bark_axis = loudness_zwst(
                channel_data, self.sampling_rate, field_type=self.field_type
            )
            results_list.append(
                {
                    "N": N,
                    "N_spec": N_spec,
                    "bark_axis": bark_axis,
                }
            )

        # Average results across channels if multi-channel
        if n_channels == 1:
            result = results_list[0]
        else:
            # Average N and N_spec across channels
            N_avg = np.mean([r["N"] for r in results_list], axis=0)
            N_spec_avg = np.mean([r["N_spec"] for r in results_list], axis=0)
            result = {
                "N": cast(float, N_avg),
                "N_spec": N_spec_avg,
                "bark_axis": results_list[0]["bark_axis"],
            }

        result["n_channels"] = n_channels

        logger.debug(
            f"Loudness calculation complete: " f"N={result['N']:.2f} sone"
        )

        return result

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output shape after loudness calculation.

        Parameters
        ----------
        input_shape : tuple
            Input data shape (n_channels, n_samples).

        Returns
        -------
        tuple
            Output shape. Returns (1,) as a placeholder since actual output
            is a dictionary with scalar and variable-length arrays.
        """
        # Return a placeholder shape since the output is a dictionary
        return (1,)

    def process(self, data: DaArray) -> dict[str, Any]:
        """
        Execute loudness calculation and return result.

        Parameters
        ----------
        data : DaArray
            Input Dask array with shape (n_channels, n_samples).

        Returns
        -------
        dict
            Dictionary containing loudness metrics. This is computed immediately
            (not lazy) since MoSQITo functions are not dask-compatible.
        """
        logger.debug("Processing stationary loudness (eager evaluation)")
        # MoSQITo functions are not dask-compatible, so we compute immediately
        data_np = data.compute()
        result = self._process_array(data_np)
        return result


# Register operations
register_operation(LoudnessZwtv)
register_operation(LoudnessZwst)
