# wandas/__init__.py
import logging
from importlib.metadata import version
from typing import Union

from .core import ChannelFrame
from .utils import generate_sample

__version__ = version(__package__ or "wandas")
read_wav = ChannelFrame.read_wav
read_csv = ChannelFrame.read_csv
from_ndarray = ChannelFrame.from_ndarray
generate_sin = generate_sample.generate_sin
__all__ = ["read_wav", "read_csv", "from_ndarray", "generate_sin"]


def setup_wandas_logging(
    level: Union[str, int] = "INFO", add_handler: bool = True
) -> logging.Logger:
    """
    wandasライブラリのログレベルを設定する便利関数

    Parameters
    ----------
    level : str or int
        ログレベル ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    add_handler : bool
        Trueの場合、コンソール出力用のハンドラを追加
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

    # オプションでハンドラを追加
    if add_handler and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    return logger
