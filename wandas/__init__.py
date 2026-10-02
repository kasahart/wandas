# wandas/__init__.py
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .core.metadata import ChannelCalibration, LevelReference
from .frames.cepstral import CepstralFrame
from .frames.cepstrogram import CepstrogramFrame
from .frames.channel import ChannelFrame
from .frames.noct import NOctFrame
from .frames.pairwise import CoherenceFrame, CrossSpectralFrame, TransferFunctionFrame
from .frames.spectral import SpectralFrame
from .frames.spectrogram import SpectrogramFrame
from .io.read import read
from .io.wdf_io import load
from .utils import generate_sample

if TYPE_CHECKING:
    import pandas as pd

    from .utils.frame_dataset import ChannelFrameDataset

__version__ = version(__package__ or "wandas")

read_wav = ChannelFrame.read_wav
read_csv = ChannelFrame.read_csv
from_numpy = ChannelFrame.from_numpy
from_ndarray = ChannelFrame.from_ndarray

generate_sin = generate_sample.generate_sin
__all__ = [
    "ChannelFrame",
    "ChannelCalibration",
    "LevelReference",
    "CepstralFrame",
    "CepstrogramFrame",
    "SpectralFrame",
    "SpectrogramFrame",
    "CoherenceFrame",
    "CrossSpectralFrame",
    "TransferFunctionFrame",
    "NOctFrame",
    "ChannelFrameDataset",
    "read",
    "load",
    "from_numpy",
    "from_folder",
    "from_files",
    "from_table",
    "supported_formats",
    "generate_sin",
]


def supported_formats() -> list[str]:
    """Return file extensions supported by the registered readers."""
    from .io.readers import supported_formats as _supported_formats

    return _supported_formats()


def from_folder(
    folder_path: str,
    sampling_rate: int | None = None,
    file_extensions: list[str] | None = None,
    recursive: bool = False,
    lazy_loading: bool = True,
    metadata_resolver: Callable[[Path], Mapping[str, object]] | None = None,
    path_metadata: bool = False,
) -> "ChannelFrameDataset":
    """Create a ChannelFrameDataset from a folder.

    Set ``path_metadata=True`` to infer AWS Glue-style partition metadata from
    relative parent directories. It cannot be combined with ``metadata_resolver``.
    """
    from .utils.frame_dataset import ChannelFrameDataset

    return ChannelFrameDataset.from_folder(
        folder_path,
        sampling_rate=sampling_rate,
        file_extensions=file_extensions,
        recursive=recursive,
        lazy_loading=lazy_loading,
        metadata_resolver=metadata_resolver,
        path_metadata=path_metadata,
    )


def from_files(paths: Iterable[str | Path], *, base_dir: str | Path | None = None) -> "ChannelFrameDataset":
    """Create a lazy recording collection from ordered local file paths.

    No folder scan, header read, or sample decode occurs during construction.
    Duplicates remain separate items. Sources must remain available through
    deferred computation. See ``ChannelFrameDataset.from_files`` for the contract.

    Args:
        paths: Finite iterable of local strings or Paths.
        base_dir: Relative path base; defaults to the construction-time directory.

    Returns:
        ChannelFrameDataset preserving input order and duplicate paths.

    Raises:
        TypeError: Input is not a collection of local paths.
        ValueError: Empty, null-containing, or URL paths are supplied.
    """
    from .utils.frame_dataset import ChannelFrameDataset

    return ChannelFrameDataset.from_files(paths, base_dir=base_dir)


def from_table(
    table: "pd.DataFrame | str | Path",
    *,
    path_column: str,
    base_dir: str | Path | None = None,
    metadata_columns: Sequence[str] | None = None,
) -> "ChannelFrameDataset":
    """Create a lazy recording collection from a DataFrame or CSV catalog.

    Each row remains an independent observation. CSV metadata remains strings;
    DataFrame basic scalar types are retained and missing values become ``None``.
    This reads a recording catalog, not the sampled-signal CSV used by ``read``.
    See ``ChannelFrameDataset.from_table`` for validation and source-lifetime rules.

    Args:
        table: pandas DataFrame or local UTF-8/BOM comma-separated CSV catalog.
        path_column: Explicit audio-location column; never inferred.
        base_dir: Audio path base; defaults to CSV parent or DataFrame call-time cwd.
        metadata_columns: Included metadata columns; defaults to all except path_column.

    Returns:
        ChannelFrameDataset preserving row order, duplicates, and metadata snapshots.

    Raises:
        TypeError: Unsupported table, source cell, or selected metadata type.
        ValueError: Invalid schema, paths, or selected metadata values.
        OSError: The CSV catalog cannot be read.
        UnicodeError: The CSV catalog is not valid UTF-8.
    """
    from .utils.frame_dataset import ChannelFrameDataset

    return ChannelFrameDataset.from_table(
        table, path_column=path_column, base_dir=base_dir, metadata_columns=metadata_columns
    )


def __getattr__(name: str) -> Any:
    if name == "ChannelFrameDataset":
        from .utils.frame_dataset import ChannelFrameDataset

        return ChannelFrameDataset
    raise AttributeError(f"module 'wandas' has no attribute {name!r}")


def setup_wandas_logging(level: str | int = "INFO", add_handler: bool = True) -> logging.Logger:
    """
    Utility function to set up logging for the wandas library.

    Args:
        level: str or int. Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        add_handler: bool. If True, adds a console handler for output

    Returns:
        logging.Logger: Configured logger instance
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
