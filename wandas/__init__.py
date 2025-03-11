# wandas/__init__.py
from importlib.metadata import version

from .core import ChannelFrame
from .utils import generate_sample

__version__ = version(__package__ or "wandas")
read_wav = ChannelFrame.read_wav
read_csv = ChannelFrame.read_csv
from_ndarray = ChannelFrame.from_ndarray
generate_sin = generate_sample.generate_sin
__all__ = ["read_wav", "read_csv", "from_ndarray", "generate_sin"]
