# wandas/utils/generate_sample.py

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wandas.frames.channel import ChannelFrame


def generate_sin(
    freqs: float | list[float] = 1000,
    sampling_rate: int = 16000,
    duration: float = 1.0,
    label: str | None = None,
) -> "ChannelFrame":
    """
    Generate sample sine wave signals.

    Parameters
    ----------
    freqs : float or list of float, default=1000
        Frequency of the sine wave(s) in Hz.
        If multiple frequencies are specified, multiple channels will be created.
    sampling_rate : int, default=16000
        Sampling rate in Hz.
    duration : float, default=1.0
        Duration of the signal in seconds.
    label : str, optional
        Label for the entire signal.

    Returns
    -------
    ChannelFrame
        ChannelFrame object containing the sine wave(s).
    """
    # 直接、generate_sin_lazy関数を呼び出す
    return generate_sin_lazy(
        freqs=freqs, sampling_rate=sampling_rate, duration=duration, label=label
    )


def generate_sin_lazy(
    freqs: float | list[float] = 1000,
    sampling_rate: int = 16000,
    duration: float = 1.0,
    label: str | None = None,
) -> "ChannelFrame":
    """
    Generate sample sine wave signals using lazy computation.

    Parameters
    ----------
    freqs : float or list of float, default=1000
        Frequency of the sine wave(s) in Hz.
        If multiple frequencies are specified, multiple channels will be created.
    sampling_rate : int, default=16000
        Sampling rate in Hz.
    duration : float, default=1.0
        Duration of the signal in seconds.
    label : str, optional
        Label for the entire signal.

    Returns
    -------
    ChannelFrame
        Lazy ChannelFrame object containing the sine wave(s).
    """
    from wandas.frames.channel import ChannelFrame

    label = label or "Generated Sin"
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    _freqs: list[float]
    if isinstance(freqs, float):
        _freqs = [freqs]
    elif isinstance(freqs, list):
        _freqs = freqs
    else:
        raise ValueError("freqs must be a float or a list of floats.")

    channels = []
    labels = []
    for idx, freq in enumerate(_freqs):
        data = np.sin(2 * np.pi * freq * t)
        labels.append(f"Channel {idx + 1}")
        channels.append(data)
    return ChannelFrame.from_numpy(
        data=np.array(channels),
        label=label,
        sampling_rate=sampling_rate,
        ch_labels=labels,
    )
