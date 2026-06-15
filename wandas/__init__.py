# wandas/__init__.py
import logging
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO
from urllib.parse import urlparse

from .frames.channel import ChannelFrame
from .frames.noct import NOctFrame
from .frames.spectral import SpectralFrame
from .frames.spectrogram import SpectrogramFrame
from .io.wdf_io import load
from .utils import generate_sample

if TYPE_CHECKING:
    from .utils.frame_dataset import ChannelFrameDataset

__version__ = version(__package__ or "wandas")

read_wav = ChannelFrame.read_wav
read_csv = ChannelFrame.read_csv
from_numpy = ChannelFrame.from_numpy
from_ndarray = ChannelFrame.from_ndarray

generate_sin = generate_sample.generate_sin_lazy
__all__ = [
    "ChannelFrame",
    "SpectralFrame",
    "SpectrogramFrame",
    "NOctFrame",
    "ChannelFrameDataset",
    "read",
    "load",
    "from_numpy",
    "from_folder",
]


def supported_formats() -> list[str]:
    """Return file extensions supported by the registered readers."""
    from .io.readers import supported_formats as _supported_formats

    return _supported_formats()


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
    return isinstance(path, (bytes, bytearray, memoryview)) or hasattr(path, "read")


def _infer_in_memory_file_type(path: object, file_type: str | None) -> str | None:
    if file_type is not None or not _is_in_memory_source(path):
        return file_type
    source_name = getattr(path, "name", None)
    if isinstance(source_name, (str, Path)):
        suffix = Path(source_name).suffix
        if suffix:
            return suffix
    return ".wav"


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
) -> ChannelFrame:
    """Read external source data into a ChannelFrame.

    Use this for WAV, CSV, supported audio files, URLs, bytes, and file-like
    objects. Use ``wd.load()`` for Wandas native WDF files.
    """
    effective_file_type = _infer_in_memory_file_type(path, file_type)
    if _is_wdf_request(path, effective_file_type):
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
        file_type=effective_file_type,
        source_name=source_name,
        normalize=normalize,
        timeout=timeout,
    )


def from_folder(
    folder_path: str,
    sampling_rate: int | None = None,
    file_extensions: list[str] | None = None,
    recursive: bool = False,
    lazy_loading: bool = True,
) -> "ChannelFrameDataset":
    """Create a ChannelFrameDataset from a folder."""
    from .utils.frame_dataset import ChannelFrameDataset

    return ChannelFrameDataset.from_folder(
        folder_path,
        sampling_rate=sampling_rate,
        file_extensions=file_extensions,
        recursive=recursive,
        lazy_loading=lazy_loading,
    )


def __getattr__(name: str) -> Any:
    if name == "ChannelFrameDataset":
        from .utils.frame_dataset import ChannelFrameDataset

        return ChannelFrameDataset
    raise AttributeError(f"module 'wandas' has no attribute {name!r}")


def setup_wandas_logging(level: str | int = "INFO", add_handler: bool = True) -> logging.Logger:
    """
    Utility function to set up logging for the wandas library.

    Parameters
    ----------
    level : str or int
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    add_handler : bool
        If True, adds a console handler for output

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if isinstance(level, str):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level_map.get(level.upper(), logging.INFO)

    logger = logging.getLogger("wandas")
    logger.setLevel(level)

    # Optionally add a handler
    if add_handler and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

    return logger
