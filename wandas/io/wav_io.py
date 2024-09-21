# wandas/io/wav_io.py

from typing import Optional, List, TYPE_CHECKING
import numpy as np
from scipy.io import wavfile

if TYPE_CHECKING:
    from wandas.core.signal import Signal


def read_wav(filename: str, labels: Optional[List[str]] = None) -> "Signal":
    """
    WAV ファイルを読み込み、Signal オブジェクトを作成します。

    Parameters:
        filename (str): WAV ファイルのパス。
        labels (list of str, optional): 各チャンネルのラベル。

    Returns:
        Signal: オーディオデータを含む Signal オブジェクト。
    """
    from wandas.core.channel import Channel
    from wandas.core.signal import Signal

    sampling_rate, data = wavfile.read(filename)

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
        channel_label = labels[i] if labels and i < len(labels) else f"Channel {i+1}"
        channels.append(
            Channel(data=channel_data, sampling_rate=sampling_rate, label=channel_label)
        )

    return Signal(channels=channels, label=filename)
