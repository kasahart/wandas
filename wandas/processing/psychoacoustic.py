"""
Psychoacoustic metrics processing operations.

This module provides psychoacoustic metrics operations for audio signals,
including loudness calculation using standardized methods.
"""

import logging
from collections.abc import Callable
from typing import Any, ClassVar, overload

import numpy as np
from mosqito.sq_metrics import loudness_zwst as loudness_zwst_mosqito
from mosqito.sq_metrics import loudness_zwtv as loudness_zwtv_mosqito
from mosqito.sq_metrics import roughness_dw as roughness_dw_mosqito
from mosqito.sq_metrics import sharpness_din_st as sharpness_din_st_mosqito
from mosqito.sq_metrics import sharpness_din_tv as sharpness_din_tv_mosqito

from wandas.processing.base import AudioOperation, get_operation, register_operation
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


@overload
def _process_per_channel(
    x: NDArrayReal,
    func: Callable[[NDArrayReal], float],
    *,
    scalar: bool = True,
) -> NDArrayReal: ...


@overload
def _process_per_channel(
    x: NDArrayReal,
    func: Callable[[NDArrayReal], NDArrayReal],
    *,
    scalar: bool = False,
) -> NDArrayReal: ...


def _process_per_channel(
    x: NDArrayReal,
    func: Callable[[NDArrayReal], NDArrayReal | float],
    *,
    scalar: bool = False,
) -> NDArrayReal:
    """Run *func* on each channel of *x* and collect results.

    Parameters
    ----------
    x : NDArrayReal
        Input array, shape ``(channels, samples)`` or ``(samples,)``.
    func : callable
        ``func(channel_1d) -> result``. *result* is a 1-D array for
        time-varying metrics or a scalar for steady-state metrics.
    scalar : bool
        If ``True`` each *func* return is a scalar and results are
        stacked into shape ``(channels, 1)``.  Otherwise results are
        ``np.stack``-ed along axis 0.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_channels = x.shape[0]
    results: list[Any] = []

    for ch in range(n_channels):
        channel_data = np.asarray(x[ch, :]).ravel()
        results.append(func(channel_data))

    if scalar:
        out: NDArrayReal = np.array(results).reshape(n_channels, 1)
        return out
    return np.stack(results, axis=0)


def _validate_field_type(field_type: str) -> None:
    """Raise ``ValueError`` if *field_type* is not 'free' or 'diffuse'."""
    if field_type not in ("free", "diffuse"):
        raise ValueError(f"field_type must be 'free' or 'diffuse', got '{field_type}'")


_VALID_SHARPNESS_WEIGHTINGS = ("din", "aures", "bismarck", "fastl")


def _validate_sharpness_params(weighting: str, field_type: str) -> None:
    """Validate sharpness weighting and field_type."""
    if weighting not in _VALID_SHARPNESS_WEIGHTINGS:
        raise ValueError(
            f"Invalid weighting function\n"
            f"  Got: '{weighting}'\n"
            f"  Expected: one of {', '.join(repr(w) for w in _VALID_SHARPNESS_WEIGHTINGS)}\n"
            f"Use a supported weighting function"
        )
    if field_type not in ("free", "diffuse"):
        raise ValueError(
            f"Invalid field type\n  Got: '{field_type}'\n  Expected: 'free' or 'diffuse'\nUse a supported field type"
        )


def _register_canonical(operation_class: type[AudioOperation[Any, Any]]) -> None:
    register_operation(operation_class)
    globals()[operation_class.__name__] = get_operation(operation_class.name)


class _ZwickerTimeVaryingBase(AudioOperation[NDArrayReal, NDArrayReal]):
    """Shared base for Zwicker time-varying metrics (loudness, sharpness).

    These operations share:
    - output sampling rate of 500 Hz (~2 ms time steps)
    - identical output shape estimation
    """

    def get_metadata_updates(self) -> dict[str, Any]:
        """Return new sampling rate of 500 Hz for ~2 ms time steps."""
        return {"sampling_rate": 500.0}

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) == 0:
            raise ValueError("Input shape must have at least one dimension")
        n_channels = input_shape[0] if len(input_shape) > 1 else 1
        estimated_time_samples = int(input_shape[-1] / (self.sampling_rate * 0.002))
        return (n_channels, estimated_time_samples)


class _SteadyStateBase(AudioOperation[NDArrayReal, NDArrayReal]):
    """Shared base for steady-state psychoacoustic metrics.

    These operations return one scalar per channel (shape ``(n_channels, 1)``)
    and do not change the sampling rate.
    """

    def get_metadata_updates(self) -> dict[str, Any]:
        return {}

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        n_channels = input_shape[0] if len(input_shape) > 1 else 1
        return (n_channels, 1)


class LoudnessZwtv(_ZwickerTimeVaryingBase):
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
        self.field_type = field_type
        super().__init__(sampling_rate, field_type=field_type)

    def validate_params(self) -> None:
        _validate_field_type(self.field_type)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process array to calculate loudness.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (channels, samples) or (samples,)

        Returns
        -------
        NDArrayReal
            Time-varying loudness in sones for each channel.
            Shape: (channels, time_samples)
        """
        logger.debug(f"Calculating loudness for signal with shape: {x.shape}, field_type: {self.field_type}")

        def _compute(ch: NDArrayReal) -> NDArrayReal:
            loudness_n, _, _, _ = loudness_zwtv_mosqito(ch, self.sampling_rate, field_type=self.field_type)
            return loudness_n

        return _process_per_channel(x, _compute)


# Register the operation
_register_canonical(LoudnessZwtv)


class LoudnessZwst(_SteadyStateBase):
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
        self.field_type = field_type
        super().__init__(sampling_rate, field_type=field_type)

    def validate_params(self) -> None:
        _validate_field_type(self.field_type)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process array to calculate steady-state loudness.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (channels, samples) or (samples,)

        Returns
        -------
        NDArrayReal
            Steady-state loudness in sones for each channel.
            Shape: (channels, 1)
        """
        logger.debug(
            f"Calculating steady-state loudness for signal with shape: {x.shape}, field_type: {self.field_type}"
        )

        def _compute(ch: NDArrayReal) -> float:
            loudness_n, _, _ = loudness_zwst_mosqito(ch, self.sampling_rate, field_type=self.field_type)
            return float(loudness_n)

        return _process_per_channel(x, _compute, scalar=True)


# Register the operation
_register_canonical(LoudnessZwst)


class _RoughnessBase(AudioOperation[NDArrayReal, NDArrayReal]):
    """Shared base for Daniel-Weber roughness operations.

    Provides common parameter validation, output estimation helpers,
    and metadata updates for the 200 ms analysis window used by roughness_dw.
    """

    _WINDOW_DURATION = 0.2  # 200 ms analysis window
    overlap: float  # Set by subclass __init__

    def validate_params(self) -> None:
        """Validate overlap is in [0.0, 1.0]."""
        if not 0.0 <= self.overlap <= 1.0:
            raise ValueError(f"overlap must be in [0.0, 1.0], got {self.overlap}")

    def _output_sampling_rate(self) -> float:
        """Compute output sampling rate from window duration and overlap."""
        hop_duration = self._WINDOW_DURATION * (1 - self.overlap)
        return 1.0 / hop_duration if hop_duration > 0 else 5.0

    def _estimated_time_samples(self, n_samples: int) -> int:
        """Estimate output time-axis length for *n_samples* input."""
        window_samples = int(self._WINDOW_DURATION * self.sampling_rate)
        hop_samples = int(window_samples * (1 - self.overlap))
        if hop_samples > 0:
            return max(1, (n_samples - window_samples) // hop_samples + 1)
        return 1

    def get_metadata_updates(self) -> dict[str, Any]:
        return {"sampling_rate": self._output_sampling_rate()}

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        n_channels = input_shape[0] if len(input_shape) > 1 else 1
        return (n_channels, self._estimated_time_samples(input_shape[-1]))


class RoughnessDw(_RoughnessBase):
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

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process array to calculate roughness.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (channels, samples) or (samples,)

        Returns
        -------
        NDArrayReal
            Time-varying roughness in asper for each channel.
            Shape: (channels, time_samples)
        """
        logger.debug(f"Calculating roughness for signal with shape: {x.shape}, overlap: {self.overlap}")

        def _compute(ch: NDArrayReal) -> NDArrayReal:
            roughness_r, _, _, _ = roughness_dw_mosqito(ch, self.sampling_rate, overlap=self.overlap)
            return np.asarray(roughness_r)

        return _process_per_channel(x, _compute)


# Register the operation
_register_canonical(RoughnessDw)


class RoughnessDwSpec(_RoughnessBase):
    """Specific roughness (R_spec) operation.

    Computes per-Bark-band specific roughness over time using MoSQITo's
    `roughness_dw` implementation. Output is band-by-time.

    The bark_axis is retrieved dynamically from MoSQITo during initialization
    to ensure consistency with MoSQITo's implementation. Results are cached
    based on sampling_rate and overlap to avoid redundant computations.
    """

    name = "roughness_dw_spec"
    # Class-level cache: {(sampling_rate, overlap): bark_axis}
    _bark_axis_cache: ClassVar[dict[tuple[float, float], NDArrayReal]] = {}

    def __init__(self, sampling_rate: float, overlap: float = 0.5) -> None:
        self.overlap = overlap
        self.validate_params()
        # Check cache first to avoid redundant MoSQITo calls
        cache_key = (sampling_rate, overlap)
        if cache_key in RoughnessDwSpec._bark_axis_cache:
            logger.debug(f"Using cached bark_axis for sampling_rate={sampling_rate}, overlap={overlap}")
            self._bark_axis: NDArrayReal = RoughnessDwSpec._bark_axis_cache[cache_key]
        else:
            # Retrieve bark_axis dynamically from MoSQITo to ensure consistency
            # Use a minimal reference signal to get the bark_axis structure
            logger.debug(f"Computing bark_axis from MoSQITo for sampling_rate={sampling_rate}, overlap={overlap}")
            reference_signal = np.zeros(int(sampling_rate * 0.2))  # 200ms minimal signal
            try:
                _, _, bark_axis_from_mosqito, _ = roughness_dw_mosqito(reference_signal, sampling_rate, overlap=overlap)
            except Exception as e:
                logger.error(f"Failed to retrieve bark_axis from MoSQITo's roughness_dw: {e}")
                raise RuntimeError(
                    "Could not initialize RoughnessDwSpec: error retrieving bark_axis from MoSQITo."
                ) from e
            if bark_axis_from_mosqito is None or (
                hasattr(bark_axis_from_mosqito, "__len__") and len(bark_axis_from_mosqito) == 0
            ):
                logger.error("MoSQITo's roughness_dw returned an empty or None bark_axis.")
                raise RuntimeError(
                    "Could not initialize RoughnessDwSpec: MoSQITo's roughness_dw returned an empty or None bark_axis."
                )
            self._bark_axis = bark_axis_from_mosqito
            # Cache the result for future use
            RoughnessDwSpec._bark_axis_cache[cache_key] = bark_axis_from_mosqito
        super().__init__(sampling_rate, overlap=overlap)

    @property
    def bark_axis(self) -> NDArrayReal:
        return self._bark_axis

    def get_metadata_updates(self) -> dict[str, Any]:
        return {"sampling_rate": self._output_sampling_rate(), "bark_axis": self._bark_axis}

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        n_bark_bands = len(self._bark_axis)
        if len(input_shape) == 1:
            n_samples = input_shape[0]
            n_channels = 1
        else:
            n_channels, n_samples = input_shape[:2]

        estimated_time_samples = self._estimated_time_samples(n_samples)

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
            x_proc: NDArrayReal = x.reshape(1, -1)
        else:
            x_proc = x

        n_channels = x_proc.shape[0]
        r_spec_list: list[NDArrayReal] = []

        for ch in range(n_channels):
            channel_data = np.asarray(x_proc[ch]).ravel()

            # Call MoSQITo's roughness_dw (module-level import)
            _, r_spec, bark_axis, _ = roughness_dw_mosqito(channel_data, self.sampling_rate, overlap=self.overlap)

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
_register_canonical(RoughnessDwSpec)


class SharpnessDin(_ZwickerTimeVaryingBase):
    """
    Calculate time-varying sharpness using DIN 45692 method.

    This operation computes the sharpness of audio signals according to
    the DIN 45692 standard. It uses the MoSQITo library's implementation
    of the standardized sharpness calculation.

    Sharpness quantifies the perceived sharpness of a sound, with units
    in acum (acum = 1 when the sound has the same sharpness as a
    2 kHz narrow-band noise with a level of 60 dB).

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz. The signal should be sampled at a rate appropriate
        for the analysis (typically 44100 Hz or 48000 Hz for audio).
    weighting : str, default="din"
        Weighting function used for the sharpness computation. Options:
        - 'din': DIN 45692 method
        - 'aures': Aures method
        - 'bismarck': Bismarck method
        - 'fastl': Fastl method
    field_type : str, default="free"
        Type of sound field. Options:
        - 'free': Free field (sound arriving from a specific direction)
        - 'diffuse': Diffuse field (sound arriving uniformly from all directions)

    Attributes
    ----------
    name : str
        Operation name: "sharpness_din"
    weighting : str
        The weighting function used for sharpness calculation
    field_type : str
        The sound field type used for calculation

    Examples
    --------
    Calculate sharpness for a signal:
    >>> import wandas as wd
    >>> signal = wd.read_wav("sharp_sound.wav")
    >>> sharpness = signal.sharpness_din(weighting="din", field_type="free")
    >>> print(f"Mean sharpness: {sharpness.data.mean():.2f} acum")

    Notes
    -----
    - The output contains time-varying sharpness values in acum
    - For mono signals, the sharpness is calculated directly
    - For multi-channel signals, sharpness is calculated per channel
    - The method follows DIN 45692 standard
    - Typical sharpness values: 0-5 acum for most sounds

    References
    ----------
    .. [1] DIN 45692:2009, "Measurement technique for the simulation of the
           auditory sensation of sharpness"
    .. [2] MoSQITo documentation:
           https://mosqito.readthedocs.io/en/latest/
    """

    name = "sharpness_din"

    def __init__(self, sampling_rate: float, weighting: str = "din", field_type: str = "free"):
        self.weighting = weighting
        self.field_type = field_type
        super().__init__(sampling_rate, weighting=weighting, field_type=field_type)

    def validate_params(self) -> None:
        _validate_sharpness_params(self.weighting, self.field_type)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process array to calculate sharpness.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (channels, samples) or (samples,)

        Returns
        -------
        NDArrayReal
            Time-varying sharpness in acum for each channel.
            Shape: (channels, time_samples)
        """
        logger.debug(f"Calculating sharpness for signal with shape: {x.shape}")

        def _compute(ch: NDArrayReal) -> NDArrayReal:
            sharpness_s, _ = sharpness_din_tv_mosqito(
                ch,
                self.sampling_rate,
                weighting=self.weighting,
                field_type=self.field_type,
                skip=0,
            )
            return sharpness_s

        return _process_per_channel(x, _compute)


# Register the operation
_register_canonical(SharpnessDin)


class SharpnessDinSt(_SteadyStateBase):
    """
    Calculate steady-state sharpness using DIN 45692 method.

    This operation computes the sharpness of stationary (steady) audio signals
    according to the DIN 45692 standard. It uses the MoSQITo library's
    implementation of the standardized sharpness calculation for steady signals.

    Sharpness quantifies the perceived sharpness of a sound, with units
    in acum (acum = 1 when the sound has the same sharpness as a
    2 kHz narrow-band noise with a level of 60 dB).

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz. The signal should be sampled at a rate appropriate
        for the analysis (typically 44100 Hz or 48000 Hz for audio).
    weighting : str, default="din"
        Weighting function used for the sharpness computation. Options:
        - 'din': DIN 45692 method
        - 'aures': Aures method
        - 'bismarck': Bismarck method
        - 'fastl': Fastl method
    field_type : str, default="free"
        Type of sound field. Options:
        - 'free': Free field (sound arriving from a specific direction)
        - 'diffuse': Diffuse field (sound arriving uniformly from all directions)

    Attributes
    ----------
    name : str
        Operation name: "sharpness_din_st"
    weighting : str
        The weighting function used for sharpness calculation
    field_type : str
        The sound field type used for calculation

    Examples
    --------
    Calculate steady-state sharpness for a signal:
    >>> import wandas as wd
    >>> signal = wd.read_wav("constant_tone.wav")
    >>> sharpness = signal.sharpness_din_st(weighting="din", field_type="free")
    >>> print(f"Steady-state sharpness: {sharpness.data[0]:.2f} acum")

    Notes
    -----
    - The output contains a single sharpness value in acum for each channel
    - For mono signals, the sharpness is calculated directly
    - For multi-channel signals, sharpness is calculated per channel
    - The method follows DIN 45692 standard for steady-state sharpness
    - Typical sharpness values: 0-5 acum for most sounds
    - This method is suitable for stationary signals such as constant tones,
      steady noise, or other unchanging sounds

    References
    ----------
    .. [1] DIN 45692:2009, "Measurement technique for the simulation of the
           auditory sensation of sharpness"
    .. [2] MoSQITo documentation:
           https://mosqito.readthedocs.io/en/latest/
    """

    name = "sharpness_din_st"

    def __init__(self, sampling_rate: float, weighting: str = "din", field_type: str = "free"):
        self.weighting = weighting
        self.field_type = field_type
        super().__init__(sampling_rate, weighting=weighting, field_type=field_type)

    def validate_params(self) -> None:
        _validate_sharpness_params(self.weighting, self.field_type)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process array to calculate steady-state sharpness.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (channels, samples) or (samples,)

        Returns
        -------
        NDArrayReal
            Steady-state sharpness in acum for each channel.
            Shape: (channels, 1)
        """
        logger.debug(
            f"Calculating steady-state sharpness for signal with shape: {x.shape}, "
            f"weighting: {self.weighting}, field_type: {self.field_type}"
        )

        def _compute(ch: NDArrayReal) -> float:
            return float(
                sharpness_din_st_mosqito(
                    ch,
                    self.sampling_rate,
                    weighting=self.weighting,
                    field_type=self.field_type,
                )
            )

        return _process_per_channel(x, _compute, scalar=True)


# Register the operation
_register_canonical(SharpnessDinSt)
