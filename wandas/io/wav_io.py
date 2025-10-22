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

    Raises
    ------
    FileNotFoundError
        If the WAV file is not found at the specified path.
    ValueError
        If the file is not a valid WAV file or is corrupted.
    requests.exceptions.RequestException
        If the URL cannot be accessed (for URL inputs).
    """
    from wandas.frames.channel import ChannelFrame

    # ファイル名がURLかどうかを判断
    if filename.startswith("http://") or filename.startswith("https://"):
        # URLの場合、requestsを使用してダウンロード
        try:
            response = requests.get(filename, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Failed to download WAV file from URL:\n"
                f"  URL: {filename}\n"
                f"  Error: {e}\n"
                f"\n"
                f"Solution:\n"
                f"  - Check if the URL is correct and accessible\n"
                f"  - Verify your internet connection\n"
                f"  - Try downloading the file manually first\n"
            ) from e
        
        file_obj = io.BytesIO(response.content)
        file_label = os.path.basename(filename)
        try:
            # メモリマッピングは使用せずに読み込む
            sampling_rate, data = wavfile.read(file_obj)
        except Exception as e:
            raise ValueError(
                f"Failed to read WAV file from URL:\n"
                f"  URL: {filename}\n"
                f"  Error: {e}\n"
                f"\n"
                f"Solution:\n"
                f"  - Verify the file is a valid WAV format\n"
                f"  - Check if the file is corrupted\n"
                f"  - Try opening the file with other audio software\n"
            ) from e
    else:
        # ローカルファイルパスの場合
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"WAV file not found:\n"
                f"  Path: {filename}\n"
                f"  Absolute path: {os.path.abspath(filename)}\n"
                f"\n"
                f"Solution:\n"
                f"  - Check if the file path is correct\n"
                f"  - Ensure the file exists in the specified location\n"
                f"  - Use absolute path if relative path is not working\n"
                f"  - Check file permissions\n"
            )
        
        file_label = os.path.basename(filename)
        try:
            # データの読み込み（メモリマッピングを使用）
            sampling_rate, data = wavfile.read(filename, mmap=True)
        except Exception as e:
            raise ValueError(
                f"Failed to read WAV file:\n"
                f"  Path: {filename}\n"
                f"  Error: {e}\n"
                f"\n"
                f"Solution:\n"
                f"  - Verify the file is a valid WAV format (not MP3, AAC, etc.)\n"
                f"  - Check if the file is corrupted\n"
                f"  - Try opening the file with other audio software\n"
                f"  - Convert the file to WAV format if needed\n"
                f"\n"
                f"Background:\n"
                f"  This function only supports WAV format.\n"
                f"  For other formats, convert to WAV first."
            ) from e

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
