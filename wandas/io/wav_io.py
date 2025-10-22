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
        If the local file does not exist.
    ValueError
        If the file format is invalid or corrupted.
    requests.RequestException
        If URL download fails.
    """
    from pathlib import Path

    from wandas.frames.channel import ChannelFrame

    # ファイル名がURLかどうかを判断
    if filename.startswith("http://") or filename.startswith("https://"):
        # URLの場合、requestsを使用してダウンロード
        try:
            response = requests.get(filename)
            response.raise_for_status()
            file_obj = io.BytesIO(response.content)
            file_label = os.path.basename(filename)
            # メモリマッピングは使用せずに読み込む
            try:
                sampling_rate, data = wavfile.read(file_obj)
            except Exception as e:
                raise ValueError(
                    f"Failed to read WAV data from URL:\n"
                    f"  URL: {filename}\n"
                    f"  Error: {str(e)}\n"
                    f"\n"
                    f"Solution:\n"
                    f"  - Verify the URL points to a valid WAV file\n"
                    f"  - Check the file is not corrupted\n"
                    f"  - Ensure the file format is standard WAV\n"
                    f"\n"
                    f"Background:\n"
                    f"  WAV files must follow the RIFF WAVE format specification.\n"
                    f"  Corrupted or non-standard files cannot be read."
                ) from e
        except requests.RequestException as e:
            raise ValueError(
                f"Failed to download WAV file from URL:\n"
                f"  URL: {filename}\n"
                f"  Error: {str(e)}\n"
                f"\n"
                f"Solution:\n"
                f"  - Verify the URL is accessible\n"
                f"  - Check your internet connection\n"
                f"  - Ensure the URL is correct and points to a WAV file\n"
                f"\n"
                f"Background:\n"
                f"  Network errors can occur due to connectivity issues,\n"
                f"  invalid URLs, or server problems."
            ) from e
    else:
        # ローカルファイルパスの場合
        file_path = Path(filename)
        if not file_path.exists():
            # Provide helpful suggestions based on the file path
            suggestions = [f"  - Verify the file path: {file_path}"]
            if file_path.is_absolute():
                suggestions.append("  - Use absolute path if file is outside current directory")
            else:
                suggestions.append(f"  - Current directory: {Path.cwd()}")
                suggestions.append("  - Use absolute path if needed")

            raise FileNotFoundError(
                f"WAV file not found:\n"
                f"  Given path: {filename}\n"
                f"  Absolute path: {file_path.absolute()}\n"
                f"\n"
                f"Solution:\n"
                + "\n".join(suggestions)
                + "\n"
                f"  - Check file name spelling\n"
                f"  - Ensure the file has not been moved or deleted\n"
            )

        file_label = os.path.basename(filename)
        # データの読み込み（メモリマッピングを使用）
        try:
            sampling_rate, data = wavfile.read(filename, mmap=True)
        except Exception as e:
            raise ValueError(
                f"Failed to read WAV file:\n"
                f"  File: {filename}\n"
                f"  Error: {str(e)}\n"
                f"\n"
                f"Solution:\n"
                f"  - Verify the file is a valid WAV file\n"
                f"  - Check the file is not corrupted\n"
                f"  - Ensure the file format is standard WAV (not compressed)\n"
                f"  - Try opening the file in an audio player to verify\n"
                f"\n"
                f"Background:\n"
                f"  WAV files must follow the RIFF WAVE format specification.\n"
                f"  Corrupted or non-standard files (e.g., compressed WAV) may fail."
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
