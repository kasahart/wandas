"""Shared public-input builders for Recipe integration tests."""

import dask.array as da
import numpy as np

from wandas.frames.channel import ChannelFrame

RECIPE_SAMPLE_RATE = 8_000


def make_recipe_source(sample_count: int, *, offset: float = 0.0) -> ChannelFrame:
    """Create a deterministic source with metadata and source-time context."""
    time = np.arange(sample_count, dtype=float) / RECIPE_SAMPLE_RATE
    data = (np.sin(2 * np.pi * 500 * time) + 0.25 * np.cos(2 * np.pi * 1_000 * time))[None, :]
    return ChannelFrame(
        data=da.from_array(data, chunks=(1, -1)),
        sampling_rate=RECIPE_SAMPLE_RATE,
        label="recipe-source",
        metadata={"recording": "speech"},
        source_time_offset=offset,
    )
