# wandas/utils/generate_sample.py

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from wandas.frames.channel import ChannelFrame

Frequency: TypeAlias = int | float | np.integer[Any] | np.floating[Any]
Frequencies: TypeAlias = Frequency | list[Any]


def _normalize_frequencies(freqs: Frequencies) -> list[float]:
    if isinstance(freqs, list):
        if not freqs:
            raise ValueError(
                "Invalid freqs\n"
                "  Got: an empty list\n"
                "  Expected: one real frequency or a non-empty list of real frequencies\n"
                "Pass one frequency in Hz for each output channel."
            )
        values = freqs
    elif isinstance(freqs, (int, float, np.integer, np.floating)) and not isinstance(freqs, (bool, np.bool_)):
        values = [freqs]
    else:
        raise TypeError(
            "Invalid freqs\n"
            f"  Got: {type(freqs).__name__} ({freqs!r})\n"
            "  Expected: one real frequency or a non-empty list of real frequencies\n"
            "Pass a numeric scalar in Hz, or one scalar per output channel."
        )

    normalized: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(
                "Invalid frequency\n"
                f"  Got: freqs[{index}]={value!r} ({type(value).__name__})\n"
                "  Expected: a finite real number greater than 0 Hz\n"
                "Replace the invalid element with a positive numeric frequency."
            )

        normalized_value = float(value)
        if not np.isfinite(normalized_value) or normalized_value <= 0:
            raise ValueError(
                "Invalid frequency\n"
                f"  Got: freqs[{index}]={value!r}\n"
                "  Expected: a finite real number greater than 0 Hz\n"
                "Pass a positive finite frequency in Hz."
            )
        normalized.append(normalized_value)

    return normalized


def generate_sin(
    freqs: Frequencies = 1000.0,
    sampling_rate: int = 16000,
    duration: float = 1.0,
    label: str | None = None,
) -> "ChannelFrame":
    """
    Generate sample sine wave signals.

    Args:
        freqs: real number or list of real numbers, default=1000.0. Positive
            finite frequency of each sine wave in Hz. A scalar creates one
            channel; a list creates one channel per element. Python and NumPy integer
            and floating scalars are accepted and normalized to ``float``.
        sampling_rate: int, default=16000. Sampling rate in Hz.
        duration: float, default=1.0. Duration of the signal in seconds.
        label: str, optional. Label for the entire signal.

    Returns:
        ChannelFrame: Dask-backed ChannelFrame containing the sine wave(s).

    Raises:
        TypeError: If ``freqs`` or one of its elements is not a real numeric scalar.
        ValueError: If a frequency list is empty or a frequency is non-finite or not positive.

    Examples:
        >>> import wandas as wd
        >>> signal = wd.generate_sin()
        >>> signal.sampling_rate
        16000
    """
    return generate_sin_lazy(freqs=freqs, sampling_rate=sampling_rate, duration=duration, label=label)


def generate_sin_lazy(
    freqs: Frequencies = 1000.0,
    sampling_rate: int = 16000,
    duration: float = 1.0,
    label: str | None = None,
) -> "ChannelFrame":
    """
    Generate sample sine wave signals using lazy computation.

    Args:
        freqs: real number or list of real numbers, default=1000.0. Positive
            finite frequency of each sine wave in Hz. A scalar creates one
            channel; a list creates one channel per element. Python and NumPy integer
            and floating scalars are accepted and normalized to ``float``.
        sampling_rate: int, default=16000. Sampling rate in Hz.
        duration: float, default=1.0. Duration of the signal in seconds.
        label: str, optional. Label for the entire signal.

    Returns:
        ChannelFrame: Dask-backed ChannelFrame containing the sine wave(s).

    Raises:
        TypeError: If ``freqs`` or one of its elements is not a real numeric scalar.
        ValueError: If a frequency list is empty or a frequency is non-finite or not positive.

    Notes:
        This is the low-level implementation name used by ``generate_sin``. It is not
        exported from the top-level ``wandas`` namespace.
    """
    from wandas.frames.channel import ChannelFrame

    label = label or "Generated Sin"
    normalized_freqs = _normalize_frequencies(freqs)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    channels = []
    labels = []
    for idx, freq in enumerate(normalized_freqs):
        data = np.sin(2 * np.pi * freq * t)
        labels.append(f"Channel {idx + 1}")
        channels.append(data)
    return ChannelFrame.from_numpy(
        data=np.array(channels),
        label=label,
        sampling_rate=sampling_rate,
        ch_labels=labels,
    )
