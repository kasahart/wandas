"""Static check: concrete frames satisfy the mixin protocols.

The mixin methods annotate ``self`` with ``ProcessingFrameProtocol`` or
``TransformFrameProtocol``. If a concrete frame stops satisfying them, a type
checker reports every call such as ``frame.fft()`` in user code as an error.
``ty check`` runs on this file, so a protocol member that a frame cannot
satisfy (for example a writable attribute where the frame has a read-only
property) fails here instead of in users' code.
"""

from __future__ import annotations

import numpy as np

import wandas as wd
from wandas.frames.mixins.protocols import (
    BaseFrameProtocol,
    ProcessingFrameProtocol,
    TransformFrameProtocol,
)


def _as_base(frame: wd.ChannelFrame) -> BaseFrameProtocol:
    return frame


def _as_processing(frame: wd.ChannelFrame) -> ProcessingFrameProtocol:
    return frame


def _as_transform(frame: wd.ChannelFrame) -> TransformFrameProtocol:
    return frame


def test_channel_frame_satisfies_mixin_protocols() -> None:
    frame = wd.from_numpy(np.zeros((2, 256), dtype=np.float32), sampling_rate=1000)

    assert _as_base(frame) is frame
    assert _as_processing(frame) is frame
    assert _as_transform(frame) is frame
    assert frame.fft().n_channels == 2
    assert frame.transfer_function().n_channels >= 1
