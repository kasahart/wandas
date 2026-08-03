"""Shared deterministic fixtures for dedicated pairwise Frame tests."""

from __future__ import annotations

from collections.abc import Sequence

import dask.array as da
import numpy as np

from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame


def make_pairwise_source(
    *,
    n_channels: int = 2,
    sampling_rate: float = 256.0,
    n_samples: int = 256,
    units: Sequence[str] | None = None,
    references: Sequence[float] | None = None,
    calibration_factors: Sequence[float] | None = None,
) -> ChannelFrame:
    """Build a lazy multi-channel source with stable IDs and rich metadata."""
    if n_channels < 1:
        raise ValueError("n_channels must be positive")
    time = np.arange(n_samples, dtype=float) / sampling_rate
    signals = np.stack(
        [
            np.sin(2.0 * np.pi * 16.0 * time),
            1.5 * np.sin(2.0 * np.pi * 16.0 * time + 0.31),
            0.65 * np.cos(2.0 * np.pi * 40.0 * time - 0.22),
        ][:n_channels]
    )
    labels = tuple(f"source-{index}" for index in range(n_channels))
    source_ids = tuple(f"source-id-{index}" for index in range(n_channels))
    normalized_units = tuple(units or ("V", "Pa", "V")[:n_channels])
    normalized_references = tuple(references or (2.0, 5.0, 3.0)[:n_channels])
    normalized_factors = tuple(calibration_factors or (1.0,) * n_channels)
    if not (len(normalized_units) == len(normalized_references) == len(normalized_factors) == n_channels):
        raise ValueError("units, references, and calibration_factors must match n_channels")

    metadata = [
        ChannelMetadata(
            label=label,
            calibration=ChannelCalibration(
                factor=factor,
                unit=unit,
                ref=reference,
            ),
            extra={"sensor": f"S{index}"},
        )
        for index, (label, unit, reference, factor) in enumerate(
            zip(labels, normalized_units, normalized_references, normalized_factors, strict=True)
        )
    ]
    return ChannelFrame(
        data=da.from_array(signals, chunks=(1, -1)),
        sampling_rate=sampling_rate,
        label="pairwise-source",
        metadata={"recording": {"take": "pairwise"}},
        channel_metadata=metadata,
        channel_ids=list(source_ids),
        source_time_offset=np.arange(n_channels, dtype=float) + 0.25,
    )


def expected_pair_indices(n_channels: int) -> tuple[tuple[int, int], ...]:
    """Return canonical output-major/input-minor pair positions."""
    return tuple((output_index, input_index) for output_index in range(n_channels) for input_index in range(n_channels))
