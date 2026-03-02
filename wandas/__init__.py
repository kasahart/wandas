# wandas/__init__.py
import logging
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

# coreからのインポートをframesからのインポートに変更
from .frames.channel import ChannelFrame
from .io import wav_io
from .utils import generate_sample

if TYPE_CHECKING:
    from .utils.frame_dataset import ChannelFrameDataset

__version__ = version(__package__ or "wandas")
read_wav = wav_io.read_wav

read_csv = ChannelFrame.read_csv
from_numpy = ChannelFrame.from_numpy
from_ndarray = from_numpy

generate_sin = generate_sample.generate_sin_lazy
__all__ = ["read_wav", "read_csv", "from_ndarray", "generate_sin", "from_folder", "ChannelFrameDataset"]


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
