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
    サンプルの正弦波信号を生成します。

    Parameters:
        freqs (float またはリスト): 正弦波の周波数（Hz）。
            複数の周波数を指定すると複数のチャンネルになります。
        sampling_rate (int): サンプリングレート（Hz）。
        duration (float): 信号の持続時間（秒）。
        label (str, optional): Signal 全体のラベル。

    Returns:
        ChannelFrame: 正弦波を含む ChannelFrame オブジェクト。
    """
    from ..core.channel import Channel
    from ..core.channel_frame import ChannelFrame

    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    if isinstance(freqs, list):
        # 複数の周波数の場合、各周波数に対してチャンネルを作成
        channels = []
        for idx, freq in enumerate(freqs):
            data = np.sin(2 * np.pi * freq * t) * 2 * np.sqrt(2)
            channel_label = f"Channel {idx + 1}"
            channel = Channel(
                data=data, sampling_rate=sampling_rate, label=channel_label, unit=None
            )
            channels.append(channel)
    else:
        # 単一の周波数の場合、チャンネルを一つ作成
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
    サンプルの正弦波信号を生成します。

    Parameters:
        freqs (float またはリスト): 正弦波の周波数（Hz）。
            複数の周波数を指定すると複数のチャンネルになります。
        sampling_rate (int): サンプリングレート（Hz）。
        duration (float): 信号の持続時間（秒）。
        label (str, optional): Signal 全体のラベル。

    Returns:
        ChannelFrame: 正弦波を含む ChannelFrame オブジェクト。
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
    labrels = []
    for idx, freq in enumerate(_freqs):
        data = np.sin(2 * np.pi * freq * t) * 2 * np.sqrt(2)
        labrels.append(f"Channel {idx + 1}")
        channels.append(data)
    return ChannelFrame.from_numpy(
        data=np.array(channels),
        label=label,
        sampling_rate=sampling_rate,
        ch_labels=labrels,
    )
