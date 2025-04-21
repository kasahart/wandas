# wandas/utils/generate_sample.py

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from ..core.channel_frame import ChannelFrame as CoreChannelFrame
    from ..core.lazy.channel_frame import ChannelFrame as LazyChannelFrame


def generate_sin(
    freqs: Union[float, list[float]] = 1000,
    sampling_rate: int = 16000,
    duration: float = 1.0,
    label: Optional[str] = None,
) -> "CoreChannelFrame":
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
    from ..core.channel import Channel
    from ..core.channel_frame import ChannelFrame

    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    if isinstance(freqs, list):
        # For multiple frequencies, create a channel for each frequency
        channels = []
        for idx, freq in enumerate(freqs):
            data = np.sin(2 * np.pi * freq * t) * 2 * np.sqrt(2)
            channel_label = f"Channel {idx + 1}"
            channel = Channel(
                data=data, sampling_rate=sampling_rate, label=channel_label, unit=None
            )
            channels.append(channel)
    else:
        # For a single frequency, create one channel
        data = np.sin(2 * np.pi * freqs * t) * 2 * np.sqrt(2)
        channel = Channel(
            data=np.squeeze(data),
            sampling_rate=sampling_rate,
            label="Channel 1",
            unit=None,
        )
        channels = [channel]

    return ChannelFrame(channels=channels, label=label)


def generate_sin_lazy(
    freqs: Union[float, list[float]] = 1000,
    sampling_rate: int = 16000,
    duration: float = 1.0,
    label: Optional[str] = None,
) -> "LazyChannelFrame":
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
    from ..core.lazy.channel_frame import ChannelFrame

    label = "Generated Sin"
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    _freqs: list[float]
    if isinstance(freqs, float):
        _freqs = [freqs]
    else:
        _freqs = freqs

    channels = []
    labels = []
    for idx, freq in enumerate(_freqs):
        data = np.sin(2 * np.pi * freq * t) * 2 * np.sqrt(2)
        labels.append(f"Channel {idx + 1}")
        channels.append(data)
    return ChannelFrame.from_numpy(
        data=np.array(channels),
        label=label,
        sampling_rate=sampling_rate,
        ch_labels=labels,
    )
