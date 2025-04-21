# wandas/io/wav_io.py

import os
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from ..core.channel import Channel
    from ..core.channel_frame import ChannelFrame


def read_wav(filename: str, labels: Optional[list[str]] = None) -> "ChannelFrame":
    """
    Read a WAV file and create a ChannelFrame object.

    Parameters
    ----------
    filename : str
        Path to the WAV file.
    labels : list of str, optional
        Labels for each channel.

    Returns
    -------
    ChannelFrame
        ChannelFrame object containing the audio data.
    """
    from wandas.core.channel import Channel
    from wandas.core.channel_frame import ChannelFrame

    sampling_rate, data = wavfile.read(filename, mmap=True)

    # Convert data to 2D array (num_samples, num_channels)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    num_channels = data.shape[1]
    channels = []

    for i in range(num_channels):
        channel_data = data[:, i]
        channel_label = labels[i] if labels and i < len(labels) else f"Channel {i + 1}"
        channels.append(
            Channel(data=channel_data, sampling_rate=sampling_rate, label=channel_label)
        )

    return ChannelFrame(channels=channels, label=filename)


def write_wav(filename: str, target: Union["ChannelFrame", "Channel"]) -> None:
    """
    Write a ChannelFrame or Channel object to a WAV file.

    Parameters
    ----------
    filename : str
        Path to the WAV file.
    target : ChannelFrame or Channel
        ChannelFrame or Channel object containing the data to write.

    Raises
    ------
    ValueError
        If target is neither a ChannelFrame nor a Channel object.
    """
    from ..core.channel import Channel
    from ..core.channel_frame import ChannelFrame

    def scale_data(data: NDArrayReal, norm: Optional[float] = None) -> npt.ArrayLike:
        if norm is None:
            _norm = np.max(np.abs(data))
        else:
            _norm = norm
        max_int16 = np.iinfo(np.int16).max
        return np.int16(data / _norm * max_int16)

    if isinstance(target, Channel):
        data = target.data
        wavfile.write(
            filename=filename,
            rate=target.sampling_rate,
            data=scale_data(data, np.max(np.abs(data))),
        )

    elif isinstance(target, ChannelFrame):
        # Remove extension from filename
        _filename = os.path.splitext(filename)[0]
        # Create folder
        os.makedirs(_filename, exist_ok=True)
        _data = np.column_stack([ch.data for ch in target])
        norm = np.max(np.abs(_data))

        for ch in target:
            wavfile.write(
                filename=os.path.join(_filename, f"{ch.label}.wav"),
                rate=target.sampling_rate,
                data=scale_data(ch.data, norm),
            )
    else:
        raise ValueError("target must be a ChannelFrame or Channel object.")
