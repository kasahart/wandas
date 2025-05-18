# wandas/io/wav_io.py
import io
import logging
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import requests
import soundfile as sf
from scipy.io import wavfile

if TYPE_CHECKING:
    from ..frames.channel import ChannelFrame

logger = logging.getLogger(__name__)


def read_wav(filename: str, labels: Optional[list[str]] = None) -> "ChannelFrame":
    """
    Read a WAV file and create a ChannelFrame object.

    Parameters
    ----------
    filename : str
        Path to the WAV file or URL to the WAV file.
    labels : list of str, optional
        Labels for each channel.

    Returns
    -------
    ChannelFrame
        ChannelFrame object containing the audio data.
    """
    from wandas.frames.channel import ChannelFrame

    # ファイル名がURLかどうかを判断
    if filename.startswith("http://") or filename.startswith("https://"):
        # URLの場合、requestsを使用してダウンロード

        response = requests.get(filename)
        file_obj = io.BytesIO(response.content)
        file_label = os.path.basename(filename)
        # メモリマッピングは使用せずに読み込む
        sampling_rate, data = wavfile.read(file_obj)
    else:
        # ローカルファイルパスの場合
        file_label = os.path.basename(filename)
        # データの読み込み（メモリマッピングを使用）
        sampling_rate, data = wavfile.read(filename, mmap=True)

    # データを(num_channels, num_samples)形状のNumPy配列に変換
    if data.ndim == 1:
        # モノラル：(samples,) -> (1, samples)
        data = np.expand_dims(data, axis=0)
    else:
        # ステレオ：(samples, channels) -> (channels, samples)
        data = data.T

    # NumPy配列からChannelFrameを作成
    channel_frame = ChannelFrame.from_numpy(
        data=data,
        sampling_rate=sampling_rate,
        label=file_label,
        ch_labels=labels,
    )

    return channel_frame


def write_wav(
    filename: str, target: "ChannelFrame", format: Optional[str] = None
) -> None:
    """
    Write a ChannelFrame object to a WAV file.

    Parameters
    ----------
    filename : str
        Path to the WAV file.
    target : ChannelFrame
        ChannelFrame object containing the data to write.
    format : str, optional
        File format. If None, determined from file extension.

    Raises
    ------
    ValueError
        If target is not a ChannelFrame object.
    """
    from wandas.frames.channel import ChannelFrame

    if not isinstance(target, ChannelFrame):
        raise ValueError("target must be a ChannelFrame object.")

    logger.debug(f"Saving audio data to file: {filename} (will compute now)")
    data = target.compute()
    data = data.T
    if data.shape[1] == 1:
        data = data.squeeze(axis=1)
    if data.dtype == float and max([np.abs(data.max()), np.abs(data.min())]) < 1:
        sf.write(
            str(filename),
            data,
            int(target.sampling_rate),
            subtype="FLOAT",
            format=format,
        )
    else:
        sf.write(str(filename), data, int(target.sampling_rate), format=format)
    logger.debug(f"Save complete: {filename}")
