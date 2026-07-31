from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO
from urllib.parse import urlparse

if TYPE_CHECKING:
    from wandas.frames.channel import ChannelFrame


def _is_file_like(obj: object) -> bool:
    return hasattr(obj, "read") and not isinstance(obj, (str, Path))


def _is_wdf_request(
    path: str | Path | bytes | bytearray | memoryview | BinaryIO,
    file_type: str | None,
) -> bool:
    if file_type is not None:
        normalized = file_type.lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        return normalized == ".wdf"
    if not isinstance(path, (str, Path)):
        return False
    path_value = str(path)
    if path_value.lower().startswith(("http://", "https://")):
        path_value = urlparse(path_value).path
    return Path(path_value).suffix.lower() == ".wdf"


def _is_in_memory_source(path: object) -> bool:
    return isinstance(path, (bytes, bytearray, memoryview)) or _is_file_like(path)


def _named_in_memory_source(path: object) -> str | None:
    if not _is_file_like(path):
        return None
    name = getattr(path, "name", None)
    if name is None:
        return None
    try:
        source_name = str(name)
    except Exception:
        return None
    return source_name or None


def _source_name_suffix(source_name: str) -> str:
    """Return a filesystem or HTTP(S) URL path suffix."""
    source_path = source_name
    if source_name.lower().startswith(("http://", "https://")):
        source_path = urlparse(source_name).path
    return Path(source_path).suffix


def _infer_in_memory_file_type(
    path: str | Path | bytes | bytearray | memoryview | BinaryIO,
    file_type: str | None,
    source_name: str | None,
) -> str | None:
    """Apply the stable in-memory format-inference precedence."""
    if file_type is not None or not _is_in_memory_source(path):
        return file_type
    named_source_name = _named_in_memory_source(path)
    for candidate_source_name in (named_source_name, source_name):
        if candidate_source_name is None:
            continue
        suffix = _source_name_suffix(candidate_source_name)
        if suffix:
            return suffix
    return ".wav"


def _infer_in_memory_source_name(
    path: str | Path | bytes | bytearray | memoryview | BinaryIO,
    source_name: str | None,
) -> str | None:
    if source_name is not None or not _is_in_memory_source(path):
        return source_name
    return _named_in_memory_source(path)


def _raise_read_wdf_error(path: object) -> None:
    raise ValueError(
        f"WDF files are loaded with wd.load(), not wd.read()\n  Path: {path}\n  Use: wd.load({str(path)!r})"
    )


def read(
    path: str | Path | bytes | bytearray | memoryview | BinaryIO,
    channel: int | list[int] | None = None,
    start: float | None = None,
    end: float | None = None,
    ch_labels: list[str] | None = None,
    time_column: int | str = 0,
    delimiter: str = ",",
    header: int | None = 0,
    file_type: str | None = None,
    source_name: str | None = None,
    timeout: float = 10.0,
) -> "ChannelFrame":
    """Read external source data into a ChannelFrame.

    Use this entry point for WAV, CSV, other registered audio formats, local
    paths, HTTP/HTTPS URLs, bytes, and binary file-like objects. The returned
    Frame is channel-first and Dask-backed. Source metadata is inspected
    synchronously. Audio sample decoding is deferred until the Dask data is
    computed. CSV metadata inspection parses the complete table synchronously
    to determine shape and sampling rate, and the table is parsed again when
    the Dask sample data is computed.

    For bytes and file-like sources, format selection uses the first available
    value in this order:

    1. explicit ``file_type``;
    2. the suffix of a file-like object's ``.name``;
    3. the suffix of ``source_name``;
    4. ``".wav"`` for an otherwise anonymous in-memory source.

    Filesystem paths use their path suffix. URL paths, including URL values used
    as ``source_name``, use the suffix before any query or fragment unless
    ``file_type`` is supplied. ``file_type`` is case-insensitive and accepts
    values with or without a leading dot. A filename hint that appears only in
    a URL query or fragment is not inferred; pass ``file_type`` explicitly.
    Use :func:`wandas.load` rather than this function for Wandas native WDF
    files.

    Args:
        path: Local path, HTTP/HTTPS URL, bytes-like value, or readable binary
            file-like object.
        channel: Zero-based channel index or indices to load. ``None`` loads
            every channel.
        start: Optional start time in seconds.
        end: Optional end time in seconds.
        ch_labels: Optional replacement labels for the selected channels.
        time_column: CSV time-column index or unique column name.
        delimiter: CSV field delimiter.
        header: CSV header row, or ``None`` for a headerless file.
        file_type: Explicit registered extension for in-memory data or a URL,
            for example ``".wav"`` or ``"csv"``. It takes precedence over
            in-memory name inference.
        source_name: Optional logical source name for in-memory data. Its suffix
            participates in format inference after a file-like ``.name``. It is
            also used for the Frame label and ``_source_file`` metadata; it does
            not read or download another resource.
        timeout: HTTP/HTTPS download timeout in seconds. It has no effect for
            local or in-memory sources.

    Returns:
        A Dask-backed :class:`~wandas.frames.channel.ChannelFrame` with
        channel-first ``float64`` data. Audio decoding is deferred; CSV has the
        synchronous metadata pass described above.

    Raises:
        FileNotFoundError: If a local source path does not exist.
        OSError: If an HTTP/HTTPS download fails or exceeds its size limit.
        ValueError: If no registered reader matches the selected format, a WDF
            source is passed, or CSV options or channel indices are invalid.

    Examples:
        Read a local file using its suffix:

        >>> import wandas as wd
        >>> frame = wd.read("recording.wav", channel=0, start=0.25, end=1.25)

        Infer CSV from a logical name attached to bytes:

        >>> csv_bytes = b"time,left\\n0.0,1.0\\n0.1,2.0\\n"
        >>> frame = wd.read(csv_bytes, source_name="sensor.csv")

        Anonymous bytes retain the compatibility default of WAV; pass
        ``file_type`` when the bytes contain another format:

        >>> frame = wd.read(csv_bytes, file_type="csv")
    """
    from wandas.frames.channel import ChannelFrame

    source_name = _infer_in_memory_source_name(path, source_name)
    file_type = _infer_in_memory_file_type(path, file_type, source_name)
    if _is_wdf_request(path, file_type):
        _raise_read_wdf_error(path)
    return ChannelFrame.from_file(
        path,
        channel=channel,
        start=start,
        end=end,
        ch_labels=ch_labels,
        time_column=time_column,
        delimiter=delimiter,
        header=header,
        file_type=file_type,
        source_name=source_name,
        timeout=timeout,
    )
