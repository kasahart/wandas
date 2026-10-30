"""Frame for normalized cross-correlation over a lag axis.

The numerical operation remains in :mod:`wandas.processing.temporal`.  This
module owns the lag axis, the typed pair rows shared with the pairwise
spectral Frames, the peak-lag projections, and plotting.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata
from wandas.frames.pairwise import PairwiseFrame, SpectralPairState
from wandas.processing.spectral_contracts import DerivedSpectralDomain, derive_coherence_domain
from wandas.utils.optional_imports import require_pandas
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


CorrelationFrameT = TypeVar("CorrelationFrameT", bound="CrossCorrelationFrame")


def _normalize_lag_start(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("lag_start must be an integer number of samples")
    return int(value)


def _normalize_correlation_data(data: DaArray | np.ndarray[Any, Any]) -> DaArray:
    """Normalize rank and dtype without materializing a Dask graph."""
    if not isinstance(data, DaArray):
        data = da.asarray(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim != 2:
        raise ValueError(
            "Invalid data rank for CrossCorrelationFrame\n"
            f"  Got: {data.ndim}D with shape {data.shape}\n"
            "  Expected: one or two dimensions with (pair, lag) semantics."
        )
    if int(data.shape[-1]) == 0:
        raise ValueError("CrossCorrelationFrame requires at least one lag")
    dtype = np.dtype(data.dtype)
    if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)):
        raise TypeError(
            "CrossCorrelationFrame requires a real numeric dtype\n"
            f"  Got: {dtype}\n"
            "  Pass real correlation coefficients."
        )
    return data


class CrossCorrelationFrame(PairwiseFrame):
    """Normalized cross-correlation of output/input channel pairs over lag.

    ``data`` is real, rank one or two, with flattened ``(pair, lag)`` storage
    internally; a single pair has the normal single-channel public shape.  Row
    ``output * n_channels + input`` holds
    ``correlate(x_output, x_input) / sqrt(E_output * E_input)``, a
    dimensionless coefficient in [-1, 1].  The required constructor state is
    ``sampling_rate``, ``pair_state`` and ``lag_start``: lag ``j`` is
    ``(lag_start + j) / sampling_rate`` seconds, positive when the output
    lags the input.  ``pair_state`` supplies the output/input roles, source
    identity and row order; labels and lineage do not define pair meaning.

    Construction and selection stay lazy for Dask input.  Pair selection,
    contiguous lag slicing and metadata changes preserve the concrete type;
    arithmetic is rejected.  ``delays`` and ``peak_coefficients`` compute the
    peak of each row.
    """

    _xarray_dim_suffix = ("channel", "lag")

    def __init__(
        self,
        data: DaArray | np.ndarray[Any, Any],
        sampling_rate: float,
        pair_state: Sequence[SpectralPairState],
        *,
        lag_start: int,
        source_channel_ids: Sequence[str] | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]] | None = None,
        channel_ids: list[str] | None = None,
        previous: BaseFrame[Any] | None = None,
        source_time_offset: float | Sequence[float] | NDArrayReal = 0.0,
        lineage: Any | None = None,
        operation_history_prefix: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        normalized_lag_start = _normalize_lag_start(lag_start)
        normalized_data = _normalize_correlation_data(data)
        records, source_ids, channel_metadata, channel_ids = self._prepare_pair_rows(
            pair_state,
            data_rows=int(normalized_data.shape[0]),
            source_channel_ids=source_channel_ids,
            channel_metadata=channel_metadata,
            channel_ids=channel_ids,
        )
        self._lag_start = normalized_lag_start
        self._pair_state = records
        self._source_channel_ids = source_ids
        super().__init__(
            data=normalized_data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            channel_metadata=channel_metadata,
            channel_ids=channel_ids,
            source_time_offset=source_time_offset,
            lineage=lineage,
            operation_history_prefix=operation_history_prefix,
            previous=previous,
        )

    def _expected_record_domain(self, record: SpectralPairState) -> DerivedSpectralDomain:
        return derive_coherence_domain()

    @property
    def lag_start(self) -> int:
        """Return the first represented lag in samples."""
        return self._lag_start

    @property
    def n_lags(self) -> int:
        """Return the number of represented lags."""
        return int(self._data.shape[-1])

    @property
    def lags(self) -> NDArrayReal:
        """Return the represented lags in seconds (positive: output after input)."""
        return (self._lag_start + np.arange(self.n_lags)) / self.sampling_rate

    @property
    def coefficients(self) -> NDArrayReal:
        """Return the raw correlation coefficients in the public shape."""
        return np.asarray(self.data)

    def _peak_indices(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.argmax(np.nan_to_num(np.abs(values), nan=-np.inf), axis=-1)

    @property
    def delays(self) -> NDArrayReal:
        """Return the lag of the largest ``abs(coefficient)`` of every pair, in seconds.

        The result has shape ``(n_pairs,)``.  The peak magnitude is used so an
        inverted channel still gives its delay; :attr:`peak_coefficients`
        holds the signed value there.  A pair whose row is all NaN (a channel
        with zero energy) gives NaN.  Accessing it computes the data.
        """
        values = self._row_values(np.asarray(self.data))
        delays = self.lags[self._peak_indices(values)]
        return np.where(np.isnan(values).all(axis=-1), np.nan, delays)

    @property
    def peak_coefficients(self) -> NDArrayReal:
        """Return the signed coefficient at each pair's :attr:`delays` lag, shape ``(n_pairs,)``."""
        values = self._row_values(np.asarray(self.data))
        indices = self._peak_indices(values)
        return values[np.arange(values.shape[0]), indices]

    def _handle_multidim_indexing(self: CorrelationFrameT, key: tuple[Any, ...]) -> CorrelationFrameT:
        """Select pair rows and a contiguous lag range together."""
        if len(key) > self._data.ndim:
            raise ValueError(f"Invalid key length: {len(key)} for shape {self.shape}")
        indices = self._channel_indices(key[0])
        axis_selectors = key[1:]
        selected_data = self._data[indices]
        lag_start = self._lag_start
        if axis_selectors:
            selector = axis_selectors[0]
            if not isinstance(selector, slice):
                raise ValueError("Only slice selectors on the lag axis are supported for CrossCorrelationFrame")
            start, stop, step = selector.indices(self.n_lags)
            if step != 1:
                raise ValueError("Lag slices must be contiguous (step 1) to keep an evenly spaced lag axis")
            lag_start += start
            selected_data = selected_data[:, start:stop]
        return self._create_new_instance(
            data=selected_data,
            channel_metadata=self._borrowed_channel_metadata_descriptors(indices),
            channel_ids=self._channel_ids_for_selection(indices),
            source_time_offset=self.source_time_offset[indices],
            lineage=self._required_semantic_lineage(),
            lag_start=lag_start,
            **self._selection_constructor_kwargs(indices),
        )

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """Preserve lag and source-identity state for Frame copies."""
        return {"lag_start": self._lag_start, "source_channel_ids": self._source_channel_ids}

    def _get_dataframe_index(self) -> pd.Index[Any]:
        """Return lag rows (seconds) for DataFrame export."""
        pd = require_pandas("CrossCorrelationFrame.to_dataframe")
        return pd.Index(self.lags, name="lag")

    def plot(
        self,
        plot_type: str = "lag",
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Plot correlation coefficient over lag, one line per selected pair.

        The x axis is lag in seconds (positive when the output lags the
        input) and the y axis the dimensionless coefficient.  Plotting
        materializes the values; Frame construction stays lazy.
        """
        from wandas.visualization.plotting import create_operation

        return create_operation(plot_type).plot(self, ax=ax, title=title, overlay=overlay, **kwargs)


__all__ = ["CrossCorrelationFrame"]
