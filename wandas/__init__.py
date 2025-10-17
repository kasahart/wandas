# wandas/__init__.py
import logging
from importlib.metadata import version
from typing import Union

# coreからのインポートをframesからのインポートに変更
from .frames.channel import ChannelFrame
from .io import wav_io
from .utils import generate_sample

try:
    __version__ = version(__package__ or "wandas")
except Exception:
    # Fallback for development when package is not installed
    __version__ = "0.1.7-dev"
read_wav = wav_io.read_wav

read_csv = ChannelFrame.read_csv
from_numpy = ChannelFrame.from_numpy
from_ndarray = from_numpy

generate_sin = generate_sample.generate_sin_lazy
__all__ = ["read_wav", "read_csv", "from_ndarray", "generate_sin"]


def setup_wandas_logging(
    level: Union[str, int] = "INFO", add_handler: bool = True
) -> logging.Logger:
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
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    return logger
