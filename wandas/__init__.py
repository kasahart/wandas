# wandas/__init__.py

from typing import Optional
from .core.signal import Signal
from .core.spectrum import Spectrum
from .utils.generate_sample import generate_sample

def read_wav(filename: str, label: Optional[str] = None) -> Signal:
    return Signal.read_wav(filename, label=label)
