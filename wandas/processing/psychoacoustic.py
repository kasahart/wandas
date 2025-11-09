"""
Psychoacoustic metrics processing operations.

This module provides psychoacoustic metrics operations for audio signals,
including loudness calculation using standardized methods.
"""

import logging
from typing import Any

import numpy as np
from mosqito.sq_metrics import loudness_zwst as loudness_zwst_mosqito
from mosqito.sq_metrics import loudness_zwtv as loudness_zwtv_mosqito
from mosqito.sq_metrics import roughness_dw as roughness_dw_mosqito

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

    def get_metadata_updates(self) -> dict[str, Any]:
        """
        Update sampling rate based on MoSQITo's time resolution.

        The Zwicker method uses approximately 2ms time steps,
        which corresponds to 500 Hz sampling rate, independent of
        the input sampling rate.

        Returns
        -------
        dict
            Metadata updates with new sampling rate

        Notes
        -----
        All necessary parameters are provided at initialization.
        The output sampling rate is always 500 Hz regardless of input.
        """
        return {"sampling_rate": 500.0}

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

            # Ensure channel_data is a contiguous 1D NumPy array
            channel_data = np.asarray(channel_data).ravel()

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


class LoudnessZwst(AudioOperation[NDArrayReal, NDArrayReal]):
    """
    Calculate steady-state loudness using Zwicker method (ISO 532-1:2017).

    This operation computes the loudness of stationary (steady) signals according to
    the Zwicker method, as specified in ISO 532-1:2017. It uses the MoSQITo library's
    implementation of the standardized loudness calculation for steady signals.

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
        Operation name: "loudness_zwst"
    field_type : str
        The sound field type used for calculation

    Examples
    --------
    Calculate steady-state loudness for a signal:
    >>> import wandas as wd
    >>> signal = wd.read_wav("fan_noise.wav")
    >>> loudness = signal.loudness_zwst(field_type="free")
    >>> print(f"Steady-state loudness: {loudness.data[0]:.2f} sones")

    Notes
    -----
    - The output contains a single loudness value in sones for each channel
    - For mono signals, the loudness is calculated directly
    - For multi-channel signals, loudness is calculated per channel
    - The method follows ISO 532-1:2017 standard for steady-state loudness
    - Typical loudness values: 1 sone ≈ 40 phon (loudness level)
    - This method is suitable for stationary signals such as fan noise,
      constant machinery sounds, or other steady sounds

    References
    ----------
    .. [1] ISO 532-1:2017, "Acoustics — Methods for calculating loudness —
           Part 1: Zwicker method"
    .. [2] MoSQITo documentation:
           https://mosqito.readthedocs.io/en/latest/
    """

    name = "loudness_zwst"

    def __init__(self, sampling_rate: float, field_type: str = "free"):
        """
        Initialize steady-state loudness calculation operation.

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

    def get_metadata_updates(self) -> dict[str, Any]:
        """
        Get metadata updates to apply after processing.

        For steady-state loudness, the output is a single value per channel,
        so no sampling rate update is needed (output is scalar, not time-series).

        Returns
        -------
        dict
            Empty dictionary (no metadata updates needed)

        Notes
        -----
        Unlike time-varying loudness, steady-state loudness produces a single
        value, not a time series, so the sampling rate concept doesn't apply.
        """
        return {}

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation.

        The steady-state loudness calculation produces a single loudness value
        per channel.

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, samples)

        Returns
        -------
        tuple
            Output data shape: (channels, 1) - one loudness value per channel
        """
        n_channels = input_shape[0] if len(input_shape) > 1 else 1
        return (n_channels, 1)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process array to calculate steady-state loudness.

        This method calculates the steady-state loudness for each channel
        of the input signal using the Zwicker method.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (channels, samples) or (samples,)

        Returns
        -------
        NDArrayReal
            Steady-state loudness in sones for each channel.
            Shape: (channels, 1)

        Notes
        -----
        The function processes each channel independently and returns
        a single loudness value per channel.
        """
        logger.debug(
            f"Calculating steady-state loudness for signal with shape: {x.shape}, "
            f"field_type: {self.field_type}"
        )

        # Handle 1D input (single channel)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_channels = x.shape[0]
        loudness_results = []

        for ch in range(n_channels):
            channel_data = x[ch, :]

            # Ensure channel_data is a contiguous 1D NumPy array
            channel_data = np.asarray(channel_data).ravel()

            # Call MoSQITo's loudness_zwst function
            # Returns: N (single loudness value), N_spec (specific loudness),
            #          bark_axis
            loudness_n, _, _ = loudness_zwst_mosqito(
                channel_data, self.sampling_rate, field_type=self.field_type
            )

            loudness_results.append(loudness_n)

            logger.debug(
                f"Channel {ch}: Calculated steady-state loudness: "
                f"{loudness_n:.2f} sones"
            )

        # Stack results and reshape to (channels, 1)
        result: NDArrayReal = np.array(loudness_results).reshape(n_channels, 1)

        logger.debug(
            f"Steady-state loudness calculation complete, output shape: {result.shape}"
        )
        return result


# Register the operation
register_operation(LoudnessZwst)


class RoughnessDw(AudioOperation[NDArrayReal, NDArrayReal]):
    """
    Calculate time-varying roughness using Daniel and Weber method.

    This operation computes the roughness of audio signals according to
    the Daniel and Weber (1997) method. It uses the MoSQITo library's
    implementation of the standardized roughness calculation.

    Roughness is a psychoacoustic metric that quantifies the perceived
    harshness or roughness of a sound. The unit is asper, where higher
    values indicate rougher sounds.

    The calculation follows the standard formula:
    R = 0.25 * sum(R'_i) for i=1 to 47 Bark bands

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz. The signal should be sampled at a rate appropriate
        for the analysis (typically 44100 Hz or 48000 Hz for audio).
    overlap : float, default=0.5
        Overlapping coefficient for the analysis windows (0.0 to 1.0).
        The analysis uses 200ms windows:
        - overlap=0.5: 100ms hop size → ~10 Hz output sampling rate
        - overlap=0.0: 200ms hop size → ~5 Hz output sampling rate

    Attributes
    ----------
    name : str
        Operation name: "roughness_dw"
    overlap : float
        The overlapping coefficient used for calculation

    Examples
    --------
    Calculate roughness for a signal:
    >>> import wandas as wd
    >>> signal = wd.read_wav("motor_noise.wav")
    >>> roughness = signal.roughness_dw(overlap=0.5)
    >>> print(f"Mean roughness: {roughness.data.mean():.2f} asper")

    Notes
    -----
    - The output contains time-varying roughness values in asper
    - For mono signals, the roughness is calculated directly
    - For multi-channel signals, roughness is calculated per channel
    - The method follows Daniel & Weber (1997) standard
    - Typical roughness values: 0-2 asper for most sounds
    - Higher overlap values provide better time resolution but increase
      computational cost

    References
    ----------
    .. [1] Daniel, P., & Weber, R. (1997). "Psychoacoustical roughness:
           Implementation of an optimized model." Acustica, 83, 113-123.
    .. [2] MoSQITo documentation:
           https://mosqito.readthedocs.io/en/latest/
    """

    name = "roughness_dw"

    def __init__(self, sampling_rate: float, overlap: float = 0.5) -> None:
        """
        Initialize Roughness calculation operation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        overlap : float, default=0.5
            Overlapping coefficient (0.0 to 1.0)
        """
        self.overlap = overlap
        super().__init__(sampling_rate, overlap=overlap)

    def validate_params(self) -> None:
        """
        Validate parameters.

        Raises
        ------
        ValueError
            If overlap is not in [0.0, 1.0]
        """
        if not 0.0 <= self.overlap <= 1.0:
            raise ValueError(f"overlap must be in [0.0, 1.0], got {self.overlap}")

    def get_metadata_updates(self) -> dict[str, Any]:
        """
        Update sampling rate based on overlap and window size.

        The Daniel & Weber method uses 200ms windows. The output
        sampling rate depends on the overlap:
        - overlap=0.0: hop=200ms → fs=5 Hz
        - overlap=0.5: hop=100ms → fs=10 Hz
        - overlap=0.75: hop=50ms → fs=20 Hz

        Returns
        -------
        dict
            Metadata updates with new sampling rate

        Notes
        -----
        The output sampling rate is approximately 1 / (0.2 * (1 - overlap)) Hz.
        """
        window_duration = 0.2  # 200ms window
        hop_duration = window_duration * (1 - self.overlap)
        output_sampling_rate = 1.0 / hop_duration if hop_duration > 0 else 5.0
        return {"sampling_rate": output_sampling_rate}

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation.

        The roughness calculation produces a time-varying output where the
        number of time points depends on the signal length and overlap.

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, samples)

        Returns
        -------
        tuple
            Output data shape (channels, time_samples)
        """
        n_channels = input_shape[0] if len(input_shape) > 1 else 1
        n_samples = input_shape[-1]

        # Estimate output length based on window size and overlap
        window_samples = int(0.2 * self.sampling_rate)  # 200ms
        hop_samples = int(window_samples * (1 - self.overlap))

        if hop_samples > 0:
            estimated_time_samples = max(
                1, (n_samples - window_samples) // hop_samples + 1
            )
        else:
            estimated_time_samples = 1

        return (n_channels, estimated_time_samples)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process array to calculate roughness.

        This method calculates the time-varying roughness for each channel
        of the input signal using the Daniel and Weber method.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (channels, samples) or (samples,)

        Returns
        -------
        NDArrayReal
            Time-varying roughness in asper for each channel.
            Shape: (channels, time_samples)

        Notes
        -----
        The function processes each channel independently and returns
        the total roughness values (R). The specific roughness per Bark
        band (R_spec) is not returned by this operation but can be obtained
        using the roughness_dw_spec method.
        """
        logger.debug(
            f"Calculating roughness for signal with shape: {x.shape}, "
            f"overlap: {self.overlap}"
        )

        # Handle 1D input (single channel)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_channels = x.shape[0]
        roughness_results = []

        for ch in range(n_channels):
            channel_data = x[ch, :]

            # Ensure channel_data is a contiguous 1D NumPy array
            channel_data = np.asarray(channel_data).ravel()

            # Call MoSQITo's roughness_dw function
            # Returns: R (total roughness), R_spec (specific roughness),
            #          bark_axis, time_axis
            roughness_r, _, _, _ = roughness_dw_mosqito(
                channel_data, self.sampling_rate, overlap=self.overlap
            )

            roughness_results.append(roughness_r)

            logger.debug(
                f"Channel {ch}: Calculated roughness with "
                f"{len(roughness_r)} time points, "
                f"max roughness: {np.max(roughness_r):.2f} asper"
            )

        # Stack results
        result: NDArrayReal = np.stack(roughness_results, axis=0)

        logger.debug(f"Roughness calculation complete, output shape: {result.shape}")
        return result


# Register the operation
register_operation(RoughnessDw)


class RoughnessDwSpec(AudioOperation[NDArrayReal, NDArrayReal]):
    """Specific roughness (R_spec) operation.

    Computes per-Bark-band specific roughness over time using MoSQITo's
    `roughness_dw` implementation. Output is band-by-time.
    """

    name = "roughness_dw_spec"

    def __init__(self, sampling_rate: float, overlap: float = 0.5) -> None:
        self.overlap = overlap
        # Retrieve bark_axis dynamically from MoSQITo to ensure consistency
        # Use a minimal reference signal to get the bark_axis structure
        reference_signal = np.zeros(int(sampling_rate * 0.2))  # 200ms minimal signal
        _, _, bark_axis_from_mosqito, _ = roughness_dw_mosqito(
            reference_signal, sampling_rate, overlap=overlap
        )
        self._bark_axis: NDArrayReal = bark_axis_from_mosqito
        super().__init__(sampling_rate, overlap=overlap)

    @property
    def bark_axis(self) -> NDArrayReal:
        return self._bark_axis

    def validate_params(self) -> None:
        if not 0.0 <= self.overlap <= 1.0:
            raise ValueError(f"overlap must be in [0.0, 1.0], got {self.overlap}")

    def get_metadata_updates(self) -> dict[str, Any]:
        window_duration = 0.2
        hop_duration = window_duration * (1 - self.overlap)
        output_sampling_rate = 1.0 / hop_duration if hop_duration > 0 else 5.0

        return {"sampling_rate": output_sampling_rate, "bark_axis": self._bark_axis}

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        n_bark_bands = len(self._bark_axis)
        if len(input_shape) == 1:
            n_samples = input_shape[0]
            n_channels = 1
        else:
            n_channels, n_samples = input_shape[:2]

        window_samples = int(0.2 * self.sampling_rate)
        hop_samples = int(window_samples * (1 - self.overlap))

        if hop_samples > 0:
            estimated_time_samples = max(
                1, (n_samples - window_samples) // hop_samples + 1
            )
        else:
            estimated_time_samples = 1

        if n_channels == 1:
            return (n_bark_bands, estimated_time_samples)
        return (n_channels, n_bark_bands, estimated_time_samples)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        logger.debug(
            "Calculating specific roughness for signal with shape: %s, overlap: %s",
            x.shape,
            self.overlap,
        )

        # Ensure (n_channels, n_samples)
        if x.ndim == 1:
            x_proc = x.reshape(1, -1)
        else:
            x_proc = x

        n_channels = x_proc.shape[0]
        r_spec_list: list[NDArrayReal] = []

        for ch in range(n_channels):
            channel_data = np.asarray(x_proc[ch]).ravel()

            # Call MoSQITo's roughness_dw (module-level import)
            _, r_spec, bark_axis, _ = roughness_dw_mosqito(
                channel_data, self.sampling_rate, overlap=self.overlap
            )

            r_spec_list.append(r_spec)
            if self._bark_axis is None:
                self._bark_axis = bark_axis

            logger.debug(
                "Channel %d: calculated specific roughness shape=%s",
                ch,
                r_spec.shape,
            )

        if n_channels == 1:
            result: NDArrayReal = r_spec_list[0]
            return result
        return np.stack(r_spec_list, axis=0)


# Register the operation
register_operation(RoughnessDwSpec)
