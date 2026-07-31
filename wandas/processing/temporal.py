import logging
import math
import warnings
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import lfilter, resample, resample_poly

from wandas.processing.base import AudioOperation, ChannelIndependentAudioOperation, register_operation
from wandas.processing.weighting import A_weight, frequency_weight, frequency_weighting
from wandas.utils import validate_sampling_rate
from wandas.utils.types import NDArrayReal
from wandas.utils.util import DB_FLOOR

logger = logging.getLogger(__name__)
MIN_SOUND_LEVEL_POWER_RATIO = 1e-20
_LEVEL_FILTER_SAFE_MIN_EXPONENT = -256
_LEVEL_FILTER_SAFE_MAX_EXPONENT = 256
_LEVEL_FILTER_EQUIVALENCE_ATOL_DB = 1e-9
_LEVEL_FILTER_SAFE_MIN_ABS = math.ldexp(1.0, _LEVEL_FILTER_SAFE_MIN_EXPONENT - 1)
_LEVEL_FILTER_SAFE_MAX_ABS = math.ldexp(1.0, _LEVEL_FILTER_SAFE_MAX_EXPONENT)


def _reference_floor_requires_log_power(
    reference: NDArrayReal,
    amplitude_scale: NDArrayReal | float = 1.0,
) -> bool:
    """Return whether float64 subnormal power can occur above the dB floor.

    The linear fast path has full precision only for normal float64 powers. It is
    safe to quantize or discard smaller powers only when they are already at or
    below the reference-relative output floor. Compare ``ref**2 * floor`` with
    the least normal float64 in an extended-precision logarithmic domain so the
    check never forms a potentially underflowing squared reference. When raw
    samples and a separate calibration scale are supplied, the equivalent raw
    power floor is ``(ref / scale)**2 * floor``.
    """
    reference_extended = np.asarray(reference, dtype=np.longdouble)
    scale_extended = np.asarray(amplitude_scale, dtype=np.longdouble)
    min_normal = np.longdouble(np.finfo(np.float64).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        absolute_floor_log = 2.0 * (np.log(reference_extended) - np.log(scale_extended)) + np.log(
            np.longdouble(MIN_SOUND_LEVEL_POWER_RATIO)
        )
        min_normal_log = np.log(min_normal)
    return bool(np.any(absolute_floor_log <= min_normal_log))


def _validated_calibration_scale(
    calibration_scale: list[float] | float | NDArrayReal,
    *,
    operation_label: str,
) -> tuple[float, ...]:
    """Return an immutable positive finite internal amplitude scale."""
    scale_array = np.atleast_1d(np.array(calibration_scale, dtype=float, copy=True))
    if scale_array.size == 0 or np.any(~np.isfinite(scale_array)) or np.any(scale_array <= 0):
        raise ValueError(
            f"Invalid {operation_label} calibration scale\n"
            f"  Got: {scale_array.tolist()}\n"
            "  Expected: Positive finite scale values\n"
            "Provide one shared scale or one scale per channel."
        )
    return tuple(float(value) for value in scale_array)


def _calibration_scale_values(
    calibration_scale: tuple[float, ...],
    n_channels: int,
) -> NDArrayReal:
    """Return one internal calibration scale for each channel."""
    scale_config = np.asarray(calibration_scale, dtype=np.float64)
    if scale_config.size == 1:
        scale = np.repeat(scale_config, n_channels)
    elif scale_config.size == n_channels:
        scale = scale_config
    else:
        raise ValueError(
            "Calibration scale count mismatch\n"
            f"  Got: {scale_config.size} scale values for {n_channels} channels\n"
            "  Expected: One shared scale or one scale per channel\n"
            "Provide a scalar scale or a list matching the number of channels."
        )
    return np.asarray(scale, dtype=np.float64)


def _level_filter_scaled_channels(x: NDArrayReal) -> NDArrayReal:
    """Return channels that need the causal scaled-state weighting path."""
    source = np.asarray(x, dtype=np.float64)
    scaled = np.zeros(source.shape[0], dtype=bool)
    if source.shape[0] == 0:
        return scaled
    for channel_index, channel in enumerate(source.reshape(source.shape[0], -1)):
        magnitude = np.abs(channel)
        finite = np.isfinite(magnitude)
        peak = np.max(magnitude, where=finite, initial=0.0)
        minimum = np.min(magnitude, where=finite & (magnitude > 0.0), initial=np.inf)
        if peak == 0.0:
            continue
        scaled[channel_index] = bool(minimum < _LEVEL_FILTER_SAFE_MIN_ABS or peak >= _LEVEL_FILTER_SAFE_MAX_ABS)
    return scaled


def _scaled_product(value: tuple[float, int], coefficient: float) -> tuple[float, int]:
    """Multiply a scaled float by one finite filter coefficient."""
    mantissa, exponent = value
    if mantissa == 0.0 or coefficient == 0.0:
        return 0.0, 0
    coefficient_mantissa, coefficient_exponent = math.frexp(coefficient)
    product_mantissa, product_exponent = math.frexp(mantissa * coefficient_mantissa)
    return product_mantissa, exponent + coefficient_exponent + product_exponent


def _scaled_sum(*values: tuple[float, int]) -> tuple[float, int]:
    """Add scaled floats without materializing their absolute magnitudes."""
    nonzero = tuple(value for value in values if value[0] != 0.0)
    if not nonzero:
        return 0.0, 0
    common_exponent = max(exponent for _, exponent in nonzero)
    aligned = (math.ldexp(mantissa, exponent - common_exponent) for mantissa, exponent in nonzero)
    total = math.fsum(aligned)
    if total == 0.0:
        return 0.0, 0
    mantissa, exponent = math.frexp(total)
    return mantissa, common_exponent + exponent


def _scaled_sos_log_amplitude(signal: NDArrayReal, sos: NDArrayReal) -> NDArrayReal:
    """Filter one channel causally and return log absolute amplitude.

    Each direct-form-II transposed state is represented by a signed float64
    mantissa and an integer base-2 exponent. State rescaling is therefore exact,
    never emits an overflowing linear array, and depends only on the processed
    prefix. Downstream level equivalence is contracted to
    ``_LEVEL_FILTER_EQUIVALENCE_ATOL_DB``; the exceptional path is used only for
    channels containing finite samples outside the normal weighting range.
    """
    samples = np.asarray(signal, dtype=np.float64).reshape(-1)
    sections = np.asarray(sos, dtype=np.float64)
    state_mantissas = np.zeros((sections.shape[0], 2), dtype=np.float64)
    state_exponents = np.zeros((sections.shape[0], 2), dtype=np.int64)
    log_amplitude = np.empty(samples.shape, dtype=np.float64)
    poisoned = False
    log_two = math.log(2.0)

    for sample_index, sample in enumerate(samples):
        if poisoned:
            log_amplitude[sample_index] = np.nan
            continue
        if not np.isfinite(sample):
            log_amplitude[sample_index] = np.nan if np.isnan(sample) else np.inf
            poisoned = True
            continue

        current = math.frexp(float(sample))
        for section_index, (b0, b1, b2, a0, a1, a2) in enumerate(sections):
            if a0 != 1.0:
                b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
            state0 = (
                float(state_mantissas[section_index, 0]),
                int(state_exponents[section_index, 0]),
            )
            state1 = (
                float(state_mantissas[section_index, 1]),
                int(state_exponents[section_index, 1]),
            )
            output = _scaled_sum(_scaled_product(current, float(b0)), state0)
            next_state0 = _scaled_sum(
                _scaled_product(current, float(b1)),
                _scaled_product(output, float(-a1)),
                state1,
            )
            next_state1 = _scaled_sum(
                _scaled_product(current, float(b2)),
                _scaled_product(output, float(-a2)),
            )
            state_mantissas[section_index] = (next_state0[0], next_state1[0])
            state_exponents[section_index] = (next_state0[1], next_state1[1])
            current = output

        mantissa, exponent = current
        if mantissa == 0.0:
            log_amplitude[sample_index] = -np.inf
        else:
            log_amplitude[sample_index] = math.log(abs(mantissa)) + exponent * log_two
    return log_amplitude


def _frequency_weight_log_amplitude(
    x: NDArrayReal,
    sampling_rate: float,
    *,
    curve: str,
    scaled_channels: NDArrayReal,
) -> NDArrayReal:
    """Return frequency-weighted log amplitude without a dangerous unscale."""
    source = np.asarray(x, dtype=np.float64)
    result = np.empty(source.shape, dtype=np.float64)
    safe_channels = ~np.asarray(scaled_channels, dtype=bool)
    if np.any(safe_channels):
        weighted = frequency_weight(source[safe_channels], sampling_rate, curve=curve)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[safe_channels] = np.log(np.abs(weighted))
    if np.any(scaled_channels):
        sos = frequency_weighting(sampling_rate, curve=curve, output="sos")
        for channel_index in np.flatnonzero(scaled_channels):
            result[channel_index] = _scaled_sos_log_amplitude(source[channel_index], sos)
    return result


def _bounded_db_from_log_ratio(
    log_ratio: NDArrayReal,
    *,
    scale: float,
    ratio_floor: float,
) -> NDArrayReal:
    """Convert a natural-log ratio to bounded dB without forming the ratio."""
    level = np.array(log_ratio, copy=True)
    np.multiply(level, scale / np.log(10.0), out=level)
    np.maximum(level, scale * np.log10(ratio_floor), out=level)
    return level


def _requires_scaled_square(
    x: NDArrayReal,
    *,
    accumulation_terms: int = 1,
    minimum_power_scale: float = 1.0,
) -> bool:
    """Return whether finite samples need a scaled/logarithmic square path.

    The check is O(channels * samples) and uses transient boolean comparisons,
    released before numerical output allocation. Non-finite values use the
    stable path as well so their state transitions remain explicit.
    ``accumulation_terms`` conservatively reserves headroom for a positive sum
    such as one RMS window. ``minimum_power_scale`` is the smallest multiplier
    applied to one squared sample before it contributes to the output, such as
    ``1 / frame_length`` for an RMS mean or ``1 - alpha`` for the first sample
    of an exponential power recurrence.
    """
    if x.size == 0:
        return False
    dtype = np.dtype(np.float64)
    min_normal = np.finfo(dtype).tiny
    lower = np.nextafter(
        np.sqrt(min_normal / minimum_power_scale),
        dtype.type(np.inf),
    )
    upper = np.nextafter(
        np.sqrt(np.finfo(dtype).max / accumulation_terms),
        dtype.type(0.0),
    )
    if np.any(~np.isfinite(x)):
        return True
    if np.any((x > upper) | (x < -upper)):
        return True
    return bool(np.any(((x > 0) & (x < lower)) | ((x < 0) & (x > -lower))))


MAX_RESAMPLING_FACTOR = 1_000_000


def _centered_frame_count(n_samples: int, frame_length: int, hop_length: int) -> int:
    padded_length = n_samples + 2 * (frame_length // 2)
    if padded_length < frame_length:
        raise ValueError(f"Input is too short (n={padded_length}) for frame_length={frame_length}")
    return 1 + ((padded_length - frame_length) // hop_length)


def _centered_frames(y: NDArrayReal, frame_length: int, hop_length: int) -> NDArrayReal:
    """Return the centered, zero-padded sliding-window view used by RMS."""
    pad = frame_length // 2
    pad_width = [(0, 0)] * (y.ndim - 1) + [(pad, pad)]
    y_padded = np.pad(y, pad_width, mode="constant")
    n_frames = _centered_frame_count(y.shape[-1], frame_length, hop_length)
    return np.lib.stride_tricks.as_strided(
        y_padded,
        shape=y_padded.shape[:-1] + (frame_length, n_frames),
        strides=y_padded.strides[:-1] + (y_padded.strides[-1], y_padded.strides[-1] * hop_length),
    )


def _frame_rms(y: NDArrayReal, frame_length: int, hop_length: int) -> NDArrayReal:
    frames = _centered_frames(y, frame_length, hop_length)
    frames_float = frames.astype(float, copy=False)
    return np.sqrt(np.mean(frames_float**2, axis=-2))


def _frame_log_rms(y: NDArrayReal, frame_length: int, hop_length: int) -> NDArrayReal:
    """Return natural-log RMS without squaring samples at their original scale.

    Normal-range samples use the released framed square after a range check.
    Extreme/non-finite samples use one full framed-size floating buffer and
    normalize each channel/frame before the in-place square. Zero, NaN, and
    infinite frames retain the corresponding ``-inf``, ``nan``, and ``+inf``
    log states.
    """
    if not _requires_scaled_square(
        y,
        accumulation_terms=frame_length,
        minimum_power_scale=1.0 / frame_length,
    ):
        rms = _frame_rms(y, frame_length, hop_length)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(rms)

    frames = _centered_frames(y, frame_length, hop_length)
    frames_float = frames.astype(float, copy=False)
    normalized = np.empty(frames_float.shape, dtype=float)
    np.absolute(frames_float, out=normalized)
    frame_scale = np.max(normalized, axis=-2)
    safe_scale = np.where(np.isfinite(frame_scale) & (frame_scale > 0), frame_scale, 1.0)
    np.divide(frames_float, safe_scale[..., np.newaxis, :], out=normalized)
    np.square(normalized, out=normalized)
    mean_normalized_power = np.mean(normalized, axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(safe_scale) + 0.5 * np.log(mean_normalized_power)


def _frame_log_rms_from_log_amplitude(
    log_amplitude: NDArrayReal,
    frame_length: int,
    hop_length: int,
) -> NDArrayReal:
    """Return framed log RMS from sample-wise log absolute amplitudes."""
    pad = frame_length // 2
    padded = np.pad(
        np.asarray(log_amplitude, dtype=np.float64),
        [(0, 0)] * (log_amplitude.ndim - 1) + [(pad, pad)],
        mode="constant",
        constant_values=-np.inf,
    )
    n_frames = _centered_frame_count(log_amplitude.shape[-1], frame_length, hop_length)
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=padded.shape[:-1] + (frame_length, n_frames),
        strides=padded.strides[:-1] + (padded.strides[-1], padded.strides[-1] * hop_length),
    )
    with np.errstate(invalid="ignore"):
        log_power_sum = np.logaddexp.reduce(2.0 * frames, axis=-2)
    return 0.5 * (log_power_sum - np.log(float(frame_length)))


def _exponential_power_from_log_amplitude(log_amplitude: NDArrayReal, alpha: float) -> NDArrayReal:
    """Return log exponential power from sample-wise log amplitude."""
    log_power = np.array(log_amplitude, dtype=np.float64, copy=True)
    log_power *= 2.0
    if alpha == 0.0 or log_power.shape[-1] == 0:
        return log_power

    log_alpha = np.log(alpha)
    log_power += np.log1p(-alpha)
    decay = np.arange(log_power.shape[-1], dtype=np.float64) * log_alpha
    log_power -= decay
    with np.errstate(invalid="ignore"):
        np.logaddexp.accumulate(log_power, axis=-1, out=log_power)
    log_power += decay
    return log_power


def _exponential_power_log(x: NDArrayReal, alpha: float) -> NDArrayReal:
    """Return natural-log first-order exponential power in linear time.

    For ``p[n] = (1 - alpha) * x[n] ** 2 + alpha * p[n - 1]``, the recurrence
    is evaluated with a cumulative ``logaddexp`` after removing the geometric
    decay. The implementation is O(channels * samples) and owns one full-size
    float64 output buffer plus one time-axis vector; it never forms ``x ** 2``.
    """
    log_amplitude = np.empty(x.shape, dtype=np.float64)
    np.absolute(x, out=log_amplitude)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log(log_amplitude, out=log_amplitude)
    return _exponential_power_from_log_amplitude(log_amplitude, alpha)


def _resampling_fraction(source_sr: float, target_sr: float) -> Fraction:
    return Fraction(str(target_sr)) / Fraction(str(source_sr))


def _ceil_resampled_length(n_samples: int, ratio: Fraction) -> int:
    return (n_samples * ratio.numerator + ratio.denominator - 1) // ratio.denominator


def _resampling_ratio(source_sr: float, target_sr: float) -> tuple[int, int]:
    ratio = _resampling_fraction(source_sr, target_sr).limit_denominator(MAX_RESAMPLING_FACTOR)
    return ratio.numerator, ratio.denominator


class ReSampling(ChannelIndependentAudioOperation[NDArrayReal, NDArrayReal]):
    """Resampling operation"""

    name = "resampling"
    _display = "rs"

    def __init__(self, sampling_rate: float, target_sr: float):
        """
        Initialize a resampling operation.

        Args:
            sampling_rate (float): Source sampling rate in Hz.
            target_sr (float): Target sampling rate in Hz.

        Raises:
            ValueError: If ``sampling_rate`` or ``target_sr`` is not positive.
        """
        validate_sampling_rate(sampling_rate, "source sampling rate")
        validate_sampling_rate(target_sr, "target sampling rate")
        super().__init__(sampling_rate, target_sr=target_sr)

    @property
    def target_sr(self) -> float:
        """Target sampling rate captured at operation construction time."""
        return self._config_value("target_sr")

    def get_metadata_updates(self) -> dict[str, Any]:
        """
        Update sampling rate to target sampling rate.

        Returns:
            dict: Metadata updates with the new sampling rate.

        Note:
            Resampling always produces output at ``target_sr``, regardless of
            the input sampling rate.
        """
        return {"sampling_rate": self.target_sr}

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate the output data shape after the operation.

        Args:
            input_shape (tuple[int, ...]): Input data shape.

        Returns:
            tuple[int, ...]: Output data shape.
        """
        # Calculate length after resampling using exact decimal sampling-rate ratio.
        ratio = _resampling_fraction(self.sampling_rate, self.target_sr)
        n_samples = _ceil_resampled_length(input_shape[-1], ratio)
        return (*input_shape[:-1], n_samples)

    @staticmethod
    def _output_dtype(input_dtype: np.dtype[Any]) -> np.dtype[Any]:
        dtype = np.dtype(input_dtype)
        if dtype.kind == "f":
            return dtype
        return np.dtype(np.float64)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for resampling operation"""
        logger.debug(f"Applying resampling to array with shape: {x.shape}")
        up, down = _resampling_ratio(self.sampling_rate, self.target_sr)
        target_len = self.calculate_output_shape(x.shape)[-1]
        poly_len = _ceil_resampled_length(x.shape[-1], Fraction(up, down))
        if poly_len == target_len:
            result: NDArrayReal = resample_poly(x, up, down, axis=-1)
        else:
            result = resample(x, target_len, axis=-1)
        logger.debug(f"Resampling applied, returning result with shape: {result.shape}")
        return result

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        """Return resampling output dtype metadata."""
        return self._output_dtype(input_dtype)


class Trim(AudioOperation[NDArrayReal, NDArrayReal]):
    """Deprecated array-level trimming operation.

    Use :meth:`wandas.frames.channel.ChannelFrame.trim` for structural
    time-range selection.
    """

    name = "trim"
    _display = "trim"

    def __init__(
        self,
        sampling_rate: float,
        start: float,
        end: float,
    ) -> None:
        warnings.warn(
            "wandas.processing.Trim is deprecated; use Frame.trim() for structural time-range selection",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(sampling_rate, start=start, end=end)

    @property
    def start(self) -> float:
        """Start time captured at operation construction time."""
        return self._config_value("start")

    @property
    def end(self) -> float:
        """End time captured at operation construction time."""
        return self._config_value("end")

    @property
    def start_sample(self) -> int:
        """Start sample index derived from the captured start time."""
        return int(self.start * self.sampling_rate)

    @property
    def end_sample(self) -> int:
        """End sample index derived from the captured end time."""
        return int(self.end * self.sampling_rate)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return the legacy array-slice output shape."""
        start_sample, end_sample, _ = slice(self.start_sample, self.end_sample).indices(input_shape[-1])
        n_samples = max(0, end_sample - start_sample)
        return (*input_shape[:-1], n_samples)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Apply the legacy array-level slice."""
        return x[..., self.start_sample : self.end_sample]


class FixLength(AudioOperation[NDArrayReal, NDArrayReal]):
    """Operation to adjust signal length to a specified length."""

    name = "fix_length"
    _display = "fix"

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
            length = int(duration * sampling_rate)
        super().__init__(sampling_rate, target_length=length)

    @property
    def target_length(self) -> int:
        """Target length captured at operation construction time."""
        return self._config_value("target_length")

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

    def _process(self, x: NDArrayReal) -> NDArrayReal:
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
    """Windowed linear RMS or reference-relative RMS amplitude level.

    ``dB=False`` returns RMS in the input unit. ``dB=True`` returns
    ``20 * log10(max(RMS / ref, 1e-12))``, bounded below by -240 dB.
    Applying ``Aw`` changes the frequency weighting before RMS; it does not
    establish instrument conformance.
    """

    name = "rms_trend"
    _display = "RMS"
    _validate_reference_count = True
    _calibration_scale: tuple[float, ...]

    def __init__(
        self,
        sampling_rate: float,
        frame_length: int = 2048,
        hop_length: int = 512,
        ref: list[float] | float = 1.0,
        dB: bool = False,
        Aw: bool = False,
        *,
        _calibration_scale: list[float] | float | NDArrayReal = 1.0,
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
            Positive amplitude reference value(s) for dB calculation. A Pa
            reference of ``2e-5`` makes pressure results dB SPL.
        dB : bool
            Whether to convert RMS to reference-relative amplitude level.
        Aw : bool
            Whether to apply the implemented A-frequency-weighting filter
            before RMS calculation.
        """
        ref_array = np.array(ref if isinstance(ref, list) else [ref], dtype=float)
        if ref_array.size == 0 or np.any(~np.isfinite(ref_array)) or np.any(ref_array <= 0):
            raise ValueError(
                "Invalid RMS level reference\n"
                f"  Got: {ref_array.tolist()}\n"
                "  Expected: Positive finite reference values\n"
                "Reference-relative dB output requires a positive finite amplitude reference."
            )
        calibration_scale = _validated_calibration_scale(
            _calibration_scale,
            operation_label="RMS level",
        )
        super().__init__(
            sampling_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            dB=dB,
            Aw=Aw,
            ref=ref_array,
        )
        object.__setattr__(self, "_calibration_scale", calibration_scale)

    @property
    def frame_length(self) -> int:
        """Frame length captured at operation construction time."""
        return self._config_value("frame_length")

    @property
    def hop_length(self) -> int:
        """Hop length captured at operation construction time."""
        return self._config_value("hop_length")

    @property
    def dB(self) -> bool:  # noqa: N802
        """Whether output is converted to decibels."""
        return self._config_value("dB")

    @property
    def Aw(self) -> bool:  # noqa: N802
        """Whether A-weighting is applied before RMS calculation."""
        return self._config_value("Aw")

    @property
    def ref(self) -> NDArrayReal:
        """Reference values captured at operation construction time."""
        return self._config_value("ref")

    def _reference_values(self, n_channels: int) -> NDArrayReal:
        """Return one validated reference value for each channel."""
        ref_config = self._config["ref"]
        if ref_config.size == 1:
            ref = np.repeat(ref_config, n_channels)
        elif ref_config.size == n_channels:
            ref = ref_config
        else:
            raise ValueError(
                "Reference count mismatch\n"
                f"  Got: {ref_config.size} reference values for {n_channels} channels\n"
                "  Expected: One shared reference or one reference per channel\n"
                "Provide ref as a scalar or a list matching the number of channels."
            )
        return np.asarray(ref, dtype=np.float64)

    def _calibration_scale_values(self, n_channels: int) -> NDArrayReal:
        """Return one validated internal amplitude scale for each channel."""
        return _calibration_scale_values(self._calibration_scale, n_channels)

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
        if self.dB and self._validate_reference_count:
            self._reference_values(input_shape[0])
            self._calibration_scale_values(input_shape[0])
        n_frames = _centered_frame_count(
            input_shape[-1],
            self.frame_length,
            self.hop_length,
        )
        return (*input_shape[:-1], n_frames)

    @staticmethod
    def _output_dtype() -> np.dtype[Any]:
        return np.dtype(np.float64)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for RMS calculation"""
        logger.debug(f"Applying RMS to array with shape: {x.shape}")

        weighted_log_amplitude: NDArrayReal | None = None
        if self.Aw:
            # Apply A-weighting
            weighting_input = x
            if self.dB:
                weighting_input = np.asarray(x, dtype=np.float64)
                scaled_channels = _level_filter_scaled_channels(weighting_input)
                if np.any(scaled_channels):
                    weighted_log_amplitude = _frequency_weight_log_amplitude(
                        weighting_input,
                        self.sampling_rate,
                        curve="A",
                        scaled_channels=scaled_channels,
                    )
            if weighted_log_amplitude is None:
                _x = A_weight(weighting_input, self.sampling_rate)
                if isinstance(_x, np.ndarray):
                    x = _x
                elif isinstance(_x, tuple):
                    x = _x[0]
                else:
                    raise ValueError("A_weighting returned an unexpected type.")

        if self.dB:
            references = self._reference_values(x.shape[0])
            calibration_scale = self._calibration_scale_values(x.shape[0])
            if weighted_log_amplitude is None:
                log_rms = _frame_log_rms(
                    x,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length,
                )
            else:
                log_rms = _frame_log_rms_from_log_amplitude(
                    weighted_log_amplitude,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length,
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                np.add(
                    log_rms,
                    np.log(calibration_scale)[..., np.newaxis],
                    out=log_rms,
                )
                np.subtract(
                    log_rms,
                    np.log(references)[..., np.newaxis],
                    out=log_rms,
                )
            result = _bounded_db_from_log_ratio(
                log_rms,
                scale=20.0,
                ratio_floor=DB_FLOOR,
            )
        else:
            # Preserve the released linear RMS path exactly. Numerical scaling
            # belongs only to the versioned dB contract.
            result = _frame_rms(
                x,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
            )
        logger.debug(f"RMS applied, returning result with shape: {result.shape}")
        return result

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        """Return RMS trend output dtype metadata."""
        return self._output_dtype()


class _RecipeRmsTrendV1(RmsTrend):
    """Released Recipe v1 RMS numerical contract retained for exact replay."""

    name = "_recipe_rms_trend_v1"
    _display = "RMS"
    _validate_reference_count = False

    def __init__(
        self,
        sampling_rate: float,
        frame_length: int = 2048,
        hop_length: int = 512,
        ref: list[float] | float = 1.0,
        dB: bool = False,
        Aw: bool = False,
    ) -> None:
        """Capture parameters with the validation behavior released for v1."""
        ref_array = np.array(ref if isinstance(ref, list) else [ref])
        AudioOperation.__init__(
            self,
            sampling_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            dB=dB,
            Aw=Aw,
            ref=ref_array,
        )

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Apply the released direct-division RMS dB implementation."""
        if self.Aw:
            weighted = A_weight(x, self.sampling_rate)
            if isinstance(weighted, np.ndarray):
                x = weighted
            elif isinstance(weighted, tuple):
                x = weighted[0]
            else:
                raise ValueError("A_weighting returned an unexpected type.")

        result = _frame_rms(x, frame_length=self.frame_length, hop_length=self.hop_length)
        if self.dB:
            result = 20 * np.log10(np.maximum(result / self._config["ref"][..., np.newaxis], DB_FLOOR))
        return result


class SoundLevel(AudioOperation[NDArrayReal, NDArrayReal]):
    """Frequency- and exponentially time-weighted RMS or level.

    The operation applies A, C, or flat Z frequency weighting, smooths squared
    samples with a 125 ms (Fast) or 1 s (Slow) first-order exponential filter,
    and returns either the square root (linear RMS) or
    ``10 * log10(max(smoothed_power / ref**2, 1e-20))``, bounded below by
    -200 dB. The result is dB SPL only for pressure in Pa with ``ref=2e-5``.
    The implementation is not a claim of complete IEC/JIS sound-level-meter
    conformance.
    """

    name = "sound_level"
    _calibration_scale: tuple[float, ...]

    def __init__(
        self,
        sampling_rate: float,
        ref: list[float] | float | NDArrayReal = 1.0,
        freq_weighting: str | None = "Z",
        time_weighting: str = "Fast",
        dB: bool = False,
        *,
        _calibration_scale: list[float] | float | NDArrayReal = 1.0,
    ) -> None:
        validate_sampling_rate(sampling_rate)
        ref_array = np.atleast_1d(np.array(ref, dtype=float, copy=True))
        if ref_array.size == 0 or np.any(~np.isfinite(ref_array)) or np.any(ref_array <= 0):
            raise ValueError(
                "Invalid sound level reference\n"
                f"  Got: {ref_array.tolist()}\n"
                "  Expected: Positive finite reference values\n"
                "Reference-relative dB output requires a positive finite amplitude reference."
            )
        calibration_scale = _validated_calibration_scale(
            _calibration_scale,
            operation_label="sound level",
        )
        normalized_freq_weighting = self._normalize_freq_weighting(freq_weighting)
        normalized_time_weighting = self._normalize_time_weighting(time_weighting)
        super().__init__(
            sampling_rate,
            ref=ref_array,
            freq_weighting=normalized_freq_weighting,
            time_weighting=normalized_time_weighting,
            dB=dB,
        )
        object.__setattr__(self, "_calibration_scale", calibration_scale)

    @staticmethod
    def _normalize_freq_weighting(freq_weighting: str | None) -> str:
        normalized = "Z" if freq_weighting is None else str(freq_weighting).upper()
        if normalized not in {"A", "C", "Z"}:
            raise ValueError(
                "Invalid frequency weighting\n"
                f"  Got: {freq_weighting!r}\n"
                "  Expected: 'A', 'C', or 'Z'\n"
                "Choose one of the implemented frequency-weighting curves."
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
            "Choose one of the implemented exponential time constants."
        )

    @property
    def ref(self) -> NDArrayReal:
        """Reference values captured at operation construction time."""
        return self._config_value("ref")

    @property
    def freq_weighting(self) -> str:
        """Frequency weighting captured at operation construction time."""
        return self._config_value("freq_weighting")

    @property
    def time_weighting(self) -> str:
        """Time weighting captured at operation construction time."""
        return self._config_value("time_weighting")

    @property
    def dB(self) -> bool:  # noqa: N802
        """Whether output is converted to decibels."""
        return self._config_value("dB")

    @property
    def time_constant(self) -> float:
        """Return the RC time constant in seconds."""
        return 0.125 if self.time_weighting == "Fast" else 1.0

    @staticmethod
    def _output_dtype(
        input_dtype: np.dtype[Any],
    ) -> np.dtype[np.float32] | np.dtype[np.float64]:
        """Return the floating output dtype for the given input dtype."""
        if np.dtype(input_dtype) == np.dtype(np.float32):
            return np.dtype(np.float32)
        return np.dtype(np.float64)

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        freq_weighting = self.freq_weighting
        time_weighting = self.time_weighting
        if self.dB:
            return f"L{freq_weighting}{time_weighting[0]}"
        return f"{freq_weighting}{time_weighting[0]}RMS"

    def _reference_values(self, n_channels: int) -> NDArrayReal:
        """Return one validated reference value for each channel."""
        ref_config = self._config["ref"]
        if ref_config.size == 1:
            ref = np.repeat(ref_config, n_channels)
        elif ref_config.size == n_channels:
            ref = ref_config
        else:
            raise ValueError(
                "Reference count mismatch\n"
                f"  Got: {ref_config.size} reference values for {n_channels} channels\n"
                "  Expected: One shared reference or one reference per channel\n"
                "Provide ref as a scalar or a list matching the number of channels."
            )
        return np.asarray(ref, dtype=np.float64)

    def _calibration_scale_values(self, n_channels: int) -> NDArrayReal:
        """Return one validated internal amplitude scale for each channel."""
        return _calibration_scale_values(self._calibration_scale, n_channels)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Validate channel-wise dB configuration and preserve input shape."""
        if self.dB:
            self._reference_values(input_shape[0])
            self._calibration_scale_values(input_shape[0])
        return input_shape

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for sound level calculation."""
        logger.debug(
            "Applying sound level to array with shape %s using %s/%s weighting",
            x.shape,
            self.freq_weighting,
            self.time_weighting,
        )
        output_dtype = self._output_dtype(x.dtype)
        weighted_input = x if x.dtype == np.float64 else np.asarray(x, dtype=np.float64)
        weighted = weighted_input
        freq_weighting = self.freq_weighting
        weighted_log_amplitude: NDArrayReal | None = None
        if freq_weighting == "Z":
            weighted = weighted_input
        else:
            if self.dB:
                scaled_channels = _level_filter_scaled_channels(weighted_input)
                if np.any(scaled_channels):
                    weighted_log_amplitude = _frequency_weight_log_amplitude(
                        weighted_input,
                        self.sampling_rate,
                        curve=freq_weighting,
                        scaled_channels=scaled_channels,
                    )
                else:
                    weighted = frequency_weight(weighted_input, self.sampling_rate, curve=freq_weighting)
            else:
                weighted = frequency_weight(weighted_input, self.sampling_rate, curve=freq_weighting)
        alpha = np.asarray(np.exp(-1.0 / (self.sampling_rate * self.time_constant)), dtype=np.float64).item()
        if self.dB:
            references = self._reference_values(weighted_input.shape[0])
            calibration_scale = self._calibration_scale_values(weighted_input.shape[0])
            if weighted_log_amplitude is not None:
                log_smoothed_power = _exponential_power_from_log_amplitude(weighted_log_amplitude, alpha)
            elif _reference_floor_requires_log_power(
                references,
                calibration_scale,
            ) or _requires_scaled_square(
                weighted,
                minimum_power_scale=1.0 - alpha,
            ):
                log_smoothed_power = _exponential_power_log(weighted, alpha)
            else:
                squared = np.square(weighted)
                log_smoothed_power = lfilter([1.0 - alpha], [1.0, -alpha], squared, axis=-1)
                del squared
                with np.errstate(divide="ignore", invalid="ignore"):
                    np.log(log_smoothed_power, out=log_smoothed_power)
            with np.errstate(divide="ignore", invalid="ignore"):
                np.add(
                    log_smoothed_power,
                    2.0 * np.log(calibration_scale[:, np.newaxis]),
                    out=log_smoothed_power,
                )
                np.subtract(
                    log_smoothed_power,
                    2.0 * np.log(references[:, np.newaxis]),
                    out=log_smoothed_power,
                )
            result = _bounded_db_from_log_ratio(
                log_smoothed_power,
                scale=10.0,
                ratio_floor=MIN_SOUND_LEVEL_POWER_RATIO,
            )
        else:
            # Preserve the released linear RMS path exactly. Numerical scaling
            # belongs only to the versioned dB contract.
            squared = np.square(weighted)
            smoothed = lfilter([1.0 - alpha], [1.0, -alpha], squared, axis=-1)
            result = np.sqrt(smoothed)
        logger.debug(f"Sound level applied, returning result with shape: {result.shape}")
        return np.asarray(result, dtype=output_dtype)

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        """Return sound level output dtype metadata."""
        return self._output_dtype(input_dtype)


class _RecipeSoundLevelV1(SoundLevel):
    """Released Recipe v1 sound-level contract retained for exact replay."""

    name = "_recipe_sound_level_v1"

    def __init__(
        self,
        sampling_rate: float,
        ref: list[float] | float | NDArrayReal = 1.0,
        freq_weighting: str | None = "Z",
        time_weighting: str = "Fast",
        dB: bool = False,
    ) -> None:
        """Capture parameters with the validation behavior released for v1."""
        validate_sampling_rate(sampling_rate)
        ref_array = np.atleast_1d(np.array(ref, dtype=float, copy=True))
        if np.any(ref_array <= 0):
            raise ValueError(
                "Invalid sound level reference\n"
                f"  Got: {ref_array.tolist()}\n"
                "  Expected: Positive reference values\n"
                "Sound pressure level requires a positive reference pressure."
            )
        AudioOperation.__init__(
            self,
            sampling_rate,
            ref=ref_array,
            freq_weighting=self._normalize_freq_weighting(freq_weighting),
            time_weighting=self._normalize_time_weighting(time_weighting),
            dB=dB,
        )

    def _reference_squared(self, n_channels: int) -> NDArrayReal:
        """Return squared v1 reference pressure for each channel."""
        ref_config = self._config["ref"]
        if ref_config.size == 1:
            ref = np.repeat(ref_config, n_channels)
        elif ref_config.size == n_channels:
            ref = ref_config
        else:
            raise ValueError(
                "Reference count mismatch\n"
                f"  Got: {ref_config.size} reference values for {n_channels} channels\n"
                "  Expected: One shared reference or one reference per channel\n"
                "Provide ref as a scalar or a list matching the number of channels."
            )
        return np.asarray(np.square(ref), dtype=np.float64)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Preserve the released v1 shape contract without v2 validation."""
        return input_shape

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Apply the released square-then-divide sound-level implementation."""
        output_dtype = self._output_dtype(x.dtype)
        weighted_input = x if x.dtype == np.float64 else np.asarray(x, dtype=np.float64)
        weighted = (
            weighted_input
            if self.freq_weighting == "Z"
            else frequency_weight(weighted_input, self.sampling_rate, curve=self.freq_weighting)
        )
        squared = np.square(weighted)
        alpha = np.asarray(np.exp(-1.0 / (self.sampling_rate * self.time_constant)), dtype=np.float64).item()
        smoothed = lfilter([1.0 - alpha], [1.0, -alpha], squared, axis=-1)
        if self.dB:
            ref_squared = self._reference_squared(smoothed.shape[0])[:, np.newaxis]
            result = 10.0 * np.log10(np.maximum(smoothed / ref_squared, MIN_SOUND_LEVEL_POWER_RATIO))
        else:
            result = np.sqrt(smoothed)
        return np.asarray(result, dtype=output_dtype)


# Register all operations
for op_class in [ReSampling, Trim, RmsTrend, FixLength, SoundLevel]:
    register_operation(op_class)
