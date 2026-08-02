"""Mixin providing common spectral properties (magnitude, phase, power, dB, dBA).

These properties are shared between SpectralFrame (2D) and SpectrogramFrame (3D).
Broadcasting follows the materialized public ``data`` shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from wandas.processing.weighting import a_weighting_db
from wandas.utils.types import NDArrayReal
from wandas.utils.util import DB_FLOOR


def _ref_weighted_db_public_shape(
    data: NDArrayReal,
    channel_metadata: Sequence[Any],
) -> NDArrayReal:
    """Convert amplitudes to dB using the materialized public shape.

    Spectral properties follow ``Frame.data``: a single channel has no leading
    channel axis, while multiple channels retain it.  The older utility
    ``ref_weighted_dB`` is intentionally left unchanged because ``NOctFrame``
    uses its internal channel-first rank contract.
    """
    channel_count = len(channel_metadata)
    if channel_count == 0:
        raise ValueError("Spectral level conversion requires channel metadata")

    if channel_count == 1:
        reference: float | NDArrayReal = float(channel_metadata[0].ref)
    else:
        if data.ndim == 0 or data.shape[0] != channel_count:
            raise ValueError(
                "Spectral level data shape does not match channel metadata\n"
                f"  Data shape: {data.shape}\n"
                f"  Channels: {channel_count}"
            )
        reference = np.asarray([channel.ref for channel in channel_metadata], dtype=float).reshape(
            (channel_count,) + (1,) * (data.ndim - 1)
        )

    result: NDArrayReal = 20 * np.log10(np.maximum(data / reference, DB_FLOOR))
    return result


class SpectralPropertiesMixin:
    """Shared magnitude, phase, squared-magnitude, and level properties.

    Host classes must provide ``data`` (computed array),
    ``_data`` (Dask array), ``_channel_metadata``, and ``freqs``.
    The operation that created the host defines the stored quantity and unit.

    NumPy properties use the same channel-axis convention as ``data``: a
    single-channel ``SpectralFrame`` returns ``(frequency,)`` and a
    single-channel ``SpectrogramFrame`` returns ``(frequency, time)``; multiple
    channels retain a leading channel axis. Plotting restores that axis only at
    its boundary when it needs channel-first input.
    """

    # -- read-only properties reused by SpectralFrame & SpectrogramFrame --

    @property
    def magnitude(self: Any) -> NDArrayReal:
        """Absolute magnitude of the stored spectral quantity."""
        result: NDArrayReal = np.abs(self.data)
        return result

    @property
    def phase(self: Any) -> NDArrayReal:
        """Phase angles in radians."""
        result: NDArrayReal = np.angle(self.data)
        return result

    @property
    def power(self: Any) -> NDArrayReal:
        """Squared magnitude, a compatibility property that is not a PSD."""
        mag: NDArrayReal = np.abs(self.data)
        result: NDArrayReal = mag**2
        return result

    @property
    def dB(self: Any) -> NDArrayReal:  # noqa: N802
        """Magnitude level: ``20 * log10(magnitude / channel_ref)``.

        For the canonical FFT, STFT, and Welch amplitude quantities, this is
        an amplitude level.
        """
        mag: NDArrayReal = np.abs(self.data)
        return _ref_weighted_db_public_shape(mag, self._channel_metadata)

    @property
    def dBA(self: Any) -> NDArrayReal:  # noqa: N802
        """A-weighted magnitude level relative to each channel reference.

        For the canonical FFT, STFT, and Welch amplitude quantities, this is
        an A-weighted amplitude level.
        """
        level: NDArrayReal = self.dB
        weighted: NDArrayReal = a_weighting_db(frequencies=self.freqs, min_db=None)
        frequency_axis = level.ndim - 2 if self._xarray_dim_suffix[-1:] == ("time",) else level.ndim - 1
        if level.shape[frequency_axis] != weighted.shape[0]:
            raise ValueError(
                "A-weighting frequency axis does not match spectral data\n"
                f"  Data shape: {level.shape}\n"
                f"  Frequencies: {weighted.shape[0]}"
            )
        weight_shape = [1] * level.ndim
        weight_shape[frequency_axis] = weighted.shape[0]
        result: NDArrayReal = level + weighted.reshape(weight_shape)
        return result
