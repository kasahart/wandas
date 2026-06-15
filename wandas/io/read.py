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


def _infer_in_memory_file_type(
    path: str | Path | bytes | bytearray | memoryview | BinaryIO,
    file_type: str | None,
) -> str | None:
    if file_type is not None or not _is_in_memory_source(path):
        return file_type
    source_name = _named_in_memory_source(path)
    if source_name is not None:
        suffix = Path(source_name).suffix
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
    normalize: bool = False,
    timeout: float = 10.0,
) -> "ChannelFrame":
    """Read external source data into a ChannelFrame.

    Use this for WAV, CSV, supported audio files, URLs, bytes, and file-like
    objects. Use ``wd.load()`` for Wandas native WDF files.
    """
    from wandas.frames.channel import ChannelFrame

    file_type = _infer_in_memory_file_type(path, file_type)
    source_name = _infer_in_memory_source_name(path, source_name)
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
        normalize=normalize,
        timeout=timeout,
    )
