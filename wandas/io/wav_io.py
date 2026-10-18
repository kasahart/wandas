# wandas/io/wav_io.py
import io
import logging
import os
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from ..frames.channel import ChannelFrame

logger = logging.getLogger(__name__)


def write_wav(filename: str | os.PathLike[str] | BinaryIO, target: "ChannelFrame", format: str | None = None) -> None:
    """
    Write a ChannelFrame object to a WAV file.

    Floating-point samples use IEEE FLOAT regardless of amplitude, preserving
    values outside [-1, 1] without normalization or PCM clipping. Values are
    stored at float32 precision. Integer input retains the format's default
    subtype. Writing computes the calibrated samples synchronously.

    Args:
        filename: Path or writable binary stream. Caller-owned streams remain open.
            Stream output is encoded in memory and written once at the current
            position, so non-seekable streams and streams with existing content
            are supported.
        target: ChannelFrame. ChannelFrame object containing the data to write.
        format: Required for streams. Paths infer the format from their extension.

    Raises:
        ValueError: If target is not a ChannelFrame or a stream has no format.
    """
    from wandas.frames.channel import ChannelFrame

    if not isinstance(target, ChannelFrame):
        raise ValueError("target must be a ChannelFrame object.")

    stream: BinaryIO | None = None
    buffer: io.BytesIO | None = None
    if isinstance(filename, (str, os.PathLike)):
        sink: str | io.BytesIO = os.fspath(filename)
    else:
        if format is None:
            raise ValueError("format is required when writing to a binary stream")
        # libsndfile seeks back to patch the header, which fails silently on
        # non-seekable streams and assumes the stream starts at offset 0. Encode
        # into a private buffer and hand the caller's stream finished bytes.
        stream = filename
        sink = buffer = io.BytesIO()

    logger.debug(f"Saving audio data to file: {filename} (will compute now)")
    data = target._compute()
    data = data.T
    if data.shape[1] == 1:
        data = data.squeeze(axis=1)
    if np.issubdtype(data.dtype, np.floating):
        sf.write(
            sink,
            data,
            int(target.sampling_rate),
            subtype="FLOAT",
            format=format,
        )
    else:
        sf.write(sink, data, int(target.sampling_rate), format=format)
    if stream is not None and buffer is not None:
        _write_all(stream, buffer.getbuffer())
    logger.debug(f"Save complete: {filename}")


def _write_all(stream: BinaryIO, payload: memoryview) -> None:
    # Raw and nonblocking streams may accept only part of a write.
    offset = 0
    while offset < len(payload):
        written = stream.write(payload[offset:])
        if not written:
            raise OSError(
                f"Stream accepted {offset} of {len(payload)} WAV bytes; wrap nonblocking streams in a blocking writer"
            )
        offset += written
