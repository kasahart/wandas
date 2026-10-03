"""Synchronous audio-header preflight without a Frame or sample graph."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO, cast

from .read import _infer_in_memory_file_type, _is_file_like
from .readers import SoundFileReader, get_file_reader


def inspect(
    source: str | Path | bytes | bytearray | memoryview | BinaryIO,
    *,
    file_type: str | None = None,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Inspect an audio header before constructing a Frame or decoding samples.

    This synchronous preflight supports the built-in SoundFile audio reader
    (WAV, FLAC, OGG, AIFF/AIF and SND). It neither constructs a Dask graph nor
    decodes PCM. CSV, WDF, URL downloads and custom readers are deliberately
    unsupported: CSV metadata requires parsing the sampled table, while custom
    readers cannot guarantee header-only inspection. Use ``read`` or ``load``
    for those workflows.

    In-memory format inference matches ``read``: explicit ``file_type``, then
    a stream's ``.name`` suffix, then ``source_name``, then anonymous WAV.
    Local paths use their suffix unless ``file_type`` is explicit. A seekable
    binary stream is inspected from position zero and its original position is
    restored on success or failure. A borrowed stream is never closed. Local
    handles opened internally are closed by the reader. This is a snapshot;
    reading the source later inspects it again and does not reuse this result.

    Args:
        source: Local path, bytes-like value, or seekable readable binary stream.
        file_type: Optional audio extension, case-insensitive with or without a dot.
        source_name: Optional in-memory format hint; never another resource to open.

    Returns:
        A dictionary with ``samplerate`` (Hz), ``channels``, ``frames``,
        ``duration`` (seconds), ``format``, ``subtype`` and ``unit``. Values
        describe the complete original recording, without channel or time slicing.

    Raises:
        TypeError: The source type is unsupported or the stream is text.
        ValueError: The format is unsupported, a URL is supplied, or a stream
            cannot be rewound and restored without consuming input.
        FileNotFoundError: A local path does not exist.
        OSError: A filesystem operation fails for an inaccessible local source.
        RuntimeError: SoundFile raises ``LibsndfileError`` for corrupt or
            unsupported audio; the exception propagates without wrapping.

    Examples:
        >>> import wandas as wd
        >>> info = wd.inspect("recording.wav")
        >>> if info["channels"] <= 8 and info["duration"] <= 180:
        ...     recording = wd.read("recording.wav")
    """
    if isinstance(source, io.TextIOBase):
        raise TypeError("inspect requires a binary stream, not a text stream")
    if isinstance(source, (str, Path)):
        if "://" in str(source):
            raise ValueError("inspect accepts local audio sources, not URL downloads")
        if not Path(source).exists():
            raise FileNotFoundError(source)
    elif not isinstance(source, (bytes, bytearray, memoryview)) and not _is_file_like(source):
        raise TypeError("inspect requires a local path, bytes or a seekable binary stream")
    selected_type = _infer_in_memory_file_type(source, file_type, source_name)
    reader = get_file_reader(source, file_type=selected_type)
    if type(reader) is not SoundFileReader:
        raise ValueError("inspect supports built-in audio headers only; use read() for CSV or custom readers")
    if not _is_file_like(source):
        return reader.get_file_info(source)
    stream = cast(BinaryIO, source)
    try:
        if not stream.seekable():
            raise ValueError("stream is not seekable")
        position = stream.tell()
        stream.seek(position)
    except (AttributeError, OSError, ValueError) as error:
        raise ValueError("inspect requires a seekable binary stream with a restorable position") from error
    try:
        try:
            stream.seek(0)
        except (OSError, ValueError) as error:
            raise ValueError("inspect requires a seekable binary stream that can rewind to position zero") from error
        return reader.get_file_info(stream)
    finally:
        stream.seek(position)
