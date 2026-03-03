# wandas/io/wav_io.py
import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol

import numpy as np
import requests
import soundfile as sf
from scipy.io import wavfile

if TYPE_CHECKING:
    from ..frames.channel import ChannelFrame

from ..core.metadata import FrameMetadata

logger = logging.getLogger(__name__)


class ReadableBinary(Protocol):
    def read(self, n: int = -1) -> bytes: ...


def read_wav(
    filename: str | Path | bytes | bytearray | memoryview | ReadableBinary,
    labels: list[str] | None = None,
    normalize: bool = False,
) -> "ChannelFrame":
    """
    Read a WAV file and create a ChannelFrame object.

    Parameters
    ----------
    filename : str | Path | bytes | bytearray | memoryview | ReadableBinary
        Path to the WAV file, URL to the WAV file, or in-memory bytes/stream.
    labels : list of str, optional
        Labels for each channel.
    normalize : bool, optional
        When False (default), return raw integer samples as produced by
        scipy.io.wavfile.read (e.g. int16 for 16-bit PCM). When True,
        normalize to float32 in [-1.0, 1.0] via soundfile.

    Returns
    -------
    ChannelFrame
        ChannelFrame object containing the audio data.
    """
    from wandas.frames.channel import ChannelFrame

    file_obj: BinaryIO | ReadableBinary
    source_file: str | None = None

    # ファイル名がURLかどうかを判断
    if isinstance(filename, str) and (filename.startswith("http://") or filename.startswith("https://")):
        # URLの場合、requestsを使用してダウンロード
        response = requests.get(filename)
        file_obj = io.BytesIO(response.content)
        file_label = os.path.basename(filename)
        source_file = filename
        if normalize:
            data, sampling_rate = sf.read(file_obj, dtype="float32", always_2d=True)
            data = data.T
        else:
            sampling_rate, data = wavfile.read(file_obj)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            else:
                data = data.T
    elif isinstance(filename, (bytes, bytearray, memoryview)) or (
        hasattr(filename, "read") and not isinstance(filename, (str, Path))
    ):
        # in-memory bytes or stream
        if isinstance(filename, (bytes, bytearray, memoryview)):
            file_obj = io.BytesIO(bytes(filename))
            file_label = "in_memory"
        else:
            raw_name = getattr(filename, "name", None)
            if isinstance(raw_name, (str, os.PathLike)):
                raw_name_str = os.fspath(raw_name)
                base_name = os.path.basename(raw_name_str)
                file_label = base_name or "in_memory"
                if base_name:
                    source_file = raw_name_str
            else:
                file_label = "in_memory"
            seekable = False
            if hasattr(filename, "seek"):
                try:
                    filename.seek(0)
                    seekable = True
                except Exception as exc:
                    logger.debug("Failed to seek to start of file-like object: %s", exc)
            file_obj = filename if seekable else io.BytesIO(filename.read())
        if normalize:
            data, sampling_rate = sf.read(file_obj, dtype="float32", always_2d=True)
            data = data.T
        else:
            sampling_rate, data = wavfile.read(file_obj)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            else:
                data = data.T
    else:
        # ローカルファイルパスの場合
        file_path = str(filename)
        file_label = os.path.basename(file_path)
        source_file = file_path
        if normalize:
            data, sampling_rate = sf.read(file_path, dtype="float32", always_2d=True)
            data = data.T
        else:
            sampling_rate, data = wavfile.read(file_path)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            else:
                data = data.T

    # NumPy配列からChannelFrameを作成
    channel_frame = ChannelFrame.from_numpy(
        data=data,
        sampling_rate=sampling_rate,
        label=file_label,
        metadata=FrameMetadata(source_file=source_file),
        ch_labels=labels,
    )

    return channel_frame


def write_wav(filename: str, target: "ChannelFrame", format: str | None = None) -> None:
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
