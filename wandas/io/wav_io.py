# wandas/io/wav_io.py

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from scipy.io import wavfile

if TYPE_CHECKING:
    from ..core.channel import Channel
    from ..core.channel_frame import ChannelFrame


def read_wav(filename: str, labels: Optional[list[str]] = None) -> "ChannelFrame":
    """
    WAV ファイルを読み込み、ChannelFrame オブジェクトを作成します。

    Parameters:
        filename (str): WAV ファイルのパス。
        labels (list of str, optional): 各チャンネルのラベル。

    Returns:
        ChannelFrame: オーディオデータを含む ChannelFrame オブジェクト。
    """
    from wandas.core.channel import Channel
    from wandas.core.channel_frame import ChannelFrame

    sampling_rate, data = wavfile.read(filename, mmap=True)

    # データ型の正規化
    # if data.dtype != np.float32 and data.dtype != np.float64:
    #     data = data.astype(np.float32) / np.iinfo(data.dtype).max
    # else:
    #     data = data.astype(np.float32)

    # データを2次元配列に変換（num_samples, num_channels）
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
    ChannelFrame オブジェクトを WAV ファイルに書き込みます。

    Parameters:
        filename (str): WAV ファイルのパス。
        data (ChannelFrame): 書き込むデータを含む ChannelFrame オブジェクト。
    """
    from ..core.channel import Channel
    from ..core.channel_frame import ChannelFrame

    if isinstance(target, Channel):
        data = target.data
    elif isinstance(target, ChannelFrame):
        data = np.column_stack([ch.data for ch in target])
    else:
        raise ValueError(
            "target は ChannelFrame または Channel オブジェクトである必要があります。"
        )

    # 16ビット整数にスケーリング
    max_int16 = np.iinfo(np.int16).max
    scaled_data = np.int16(data / np.max(np.abs(data)) * max_int16)
    wavfile.write(filename=filename, rate=target.sampling_rate, data=scaled_data)
