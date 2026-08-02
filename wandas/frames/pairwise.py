"""Dedicated Frame types for flattened pairwise spectral quantities.

The numerical operations remain in :mod:`wandas.processing.spectral`.  This
module owns the domain boundary: immutable ordered pair state, quantity-specific
properties, typed plotting projections, and propagation of pair state through
the common Frame reconstruction paths.
"""

from __future__ import annotations

import copy
import numbers
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, cast

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.processing.spectral_contracts import (
    DerivedSpectralDomain,
    OrderedSpectralPair,
    SpectralChannelRole,
    SpectralScaling,
    TransferDenominator,
    csd_level,
    derive_coherence_domain,
    derive_csd_domain,
    derive_transfer_domain,
    reject_pairwise_a_weighting,
    transfer_level,
)
from wandas.utils.optional_imports import require_pandas
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


PairwiseFrameT = TypeVar("PairwiseFrameT", bound="PairwiseSpectralFrame")
PairQuantity = Literal["coherence", "csd", "transfer"]


def _normalize_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _normalize_nonblank(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-blank string")
    return value.strip()


def _normalize_source_channel_id(value: object, *, name: str) -> str:
    """Validate an opaque source ID without changing its caller-provided text."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-blank string")
    return value


def _normalize_pairwise_data(
    data: DaArray | np.ndarray[Any, Any],
    *,
    n_fft: int,
    frequency_count: int,
    frame_type: str,
    domain: Literal["real", "complex"],
) -> DaArray:
    """Normalize rank and dtype without materializing a Dask graph."""
    if not isinstance(data, DaArray):
        data = da.asarray(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim != 2:
        raise ValueError(
            f"Invalid data rank for {frame_type}\n"
            f"  Got: {data.ndim}D with shape {data.shape}\n"
            "  Expected: one or two dimensions with (pair, frequency) semantics."
        )
    expected_bins = frequency_count
    if int(data.shape[-1]) != expected_bins:
        raise ValueError(
            f"Invalid frequency bin count for {frame_type}\n"
            f"  Got: {data.shape[-1]} bins\n"
            f"  Expected: {expected_bins} represented bins for n_fft={n_fft}."
        )
    dtype = np.dtype(data.dtype)
    is_complex = np.issubdtype(dtype, np.complexfloating)
    is_real = np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)
    if domain == "real" and not is_real:
        raise TypeError(
            f"{frame_type} requires a real numeric dtype\n"
            f"  Got: {dtype}\n"
            "  Pass magnitude-squared coherence values without a complex phase."
        )
    if domain == "complex" and not is_complex:
        raise TypeError(
            f"{frame_type} requires a complex numeric dtype\n"
            f"  Got: {dtype}\n"
            "  Pass the stored complex pairwise quantity."
        )
    return data


def _validate_coherence_block(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Validate one lazy coherence block at its materialization boundary."""
    if np.isinf(values).any():
        raise ValueError("Coherence values must not contain infinity")
    finite = values[np.isfinite(values)]
    tolerance = 1e-12
    if np.issubdtype(values.dtype, np.floating):
        tolerance = max(tolerance, 4.0 * np.finfo(values.dtype).eps)
    if finite.size and ((finite < -tolerance).any() or (finite > 1.0 + tolerance).any()):
        raise ValueError("Coherence finite values must lie between 0 and 1")
    return values


@dataclass(frozen=True, slots=True)
class SpectralPairState:
    """Immutable row state for one flattened output/input pair.

    ``pair`` is the numerical role truth. ``display_label`` is retained only as
    the current display contract, including the released Recipe v1 label order;
    it is never used to recover pair meaning.
    """

    pair: OrderedSpectralPair
    domain: DerivedSpectralDomain
    row_id: str
    display_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.pair, OrderedSpectralPair):
            raise TypeError("SpectralPairState.pair must be an OrderedSpectralPair")
        if not isinstance(self.domain, DerivedSpectralDomain):
            raise TypeError("SpectralPairState.domain must be a DerivedSpectralDomain")
        object.__setattr__(self, "row_id", _normalize_nonblank(self.row_id, name="Pair row id"))
        object.__setattr__(self, "display_label", _normalize_nonblank(self.display_label, name="Pair display label"))


def build_pair_state(
    channel_metadata: Sequence[ChannelMetadata],
    channel_ids: Sequence[str],
    *,
    quantity: PairQuantity,
    scaling: SpectralScaling | None = None,
    denominator_role: TransferDenominator = "input",
    label_template: str,
) -> tuple[SpectralPairState, ...]:
    """Build output-major/input-minor state from source Frame metadata."""
    if len(channel_metadata) != len(channel_ids):
        raise ValueError("Pairwise source metadata and channel IDs must have equal lengths")
    source_ids = tuple(_normalize_source_channel_id(value, name="Source channel id") for value in channel_ids)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Pairwise source channel IDs must be unique")
    roles = tuple(
        SpectralChannelRole(
            index=index,
            label=channel.label,
            unit=channel.unit,
            reference=channel.ref,
            channel_id=source_ids[index],
        )
        for index, channel in enumerate(channel_metadata)
    )
    source_count = len(roles)
    records: list[SpectralPairState] = []
    for output in roles:
        for input_ in roles:
            pair = OrderedSpectralPair(output=output, input=input_, n_channels=source_count)
            if quantity == "coherence":
                domain = derive_coherence_domain()
            elif quantity == "csd":
                if scaling is None:
                    raise ValueError("CSD pair state requires a scaling mode")
                domain = derive_csd_domain(pair, scaling)
            else:
                domain = derive_transfer_domain(pair, denominator_role)
            display_label = label_template.format(
                out_label=output.label,
                in_label=input_.label,
            )
            row_id = f"pair:{pair.pair_index}"
            records.append(SpectralPairState(pair, domain, row_id, display_label))
    return tuple(records)


def _metadata_for_pair_state(
    records: Sequence[SpectralPairState],
    source_channel_metadata: Sequence[ChannelMetadata] | None = None,
) -> list[ChannelMetadata]:
    """Derive pair metadata, preserving source extras without making them semantic."""
    source_extras: dict[int, dict[str, Any]] = {}
    if source_channel_metadata is not None:
        source_extras = {index: copy.deepcopy(channel.extra) for index, channel in enumerate(source_channel_metadata)}
        if any(
            index not in source_extras
            for record in records
            for index in (record.pair.output.index, record.pair.input.index)
        ):
            raise ValueError("Source channel metadata does not cover every pair role")

    return [
        ChannelMetadata(
            label=record.display_label,
            calibration=ChannelCalibration(
                factor=1.0,
                unit=record.domain.unit,
                ref=record.domain.reference,
            ),
            extra=(
                {
                    "output": copy.deepcopy(source_extras[record.pair.output.index]),
                    "input": copy.deepcopy(source_extras[record.pair.input.index]),
                }
                if source_channel_metadata is not None
                else {}
            ),
        )
        for record in records
    ]


def _normalize_pair_state(
    value: Sequence[SpectralPairState],
    *,
    data_rows: int,
    source_channel_ids: Sequence[str] | None,
) -> tuple[tuple[SpectralPairState, ...], tuple[str, ...]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("pair_state must be an ordered sequence of SpectralPairState values")
    records = tuple(value)
    if len(records) != data_rows:
        raise ValueError(
            f"Pair state row count must match data rows\n  Rows: {data_rows}\n  Pair state: {len(records)}"
        )
    if any(not isinstance(record, SpectralPairState) for record in records):
        raise TypeError("pair_state must contain only SpectralPairState values")
    if len({record.pair.pair_index for record in records}) != len(records):
        raise ValueError("Pair state contains duplicate output/input pairs")
    if len({record.row_id for record in records}) != len(records):
        raise ValueError("Pair state contains duplicate row IDs")
    if records:
        source_count = records[0].pair.n_channels
        if any(record.pair.n_channels != source_count for record in records):
            raise ValueError("Pair state records must use one source channel count")
    else:
        source_count = 0
    canonical_roles: list[SpectralChannelRole | None] = [None] * source_count
    for row, record in enumerate(records):
        for role_name, role in (("output", record.pair.output), ("input", record.pair.input)):
            canonical = canonical_roles[role.index]
            if canonical is None:
                canonical_roles[role.index] = role
            elif role != canonical:
                raise ValueError(
                    "Pair state contains inconsistent source channel roles"
                    f" at source index {role.index} in row {row} ({role_name});"
                    f" expected {canonical!r}, got {role!r}"
                )
    if source_channel_ids is None:
        ids_by_index = ["" for _ in range(source_count)]
        for record in records:
            ids_by_index[record.pair.output.index] = record.pair.output.channel_id
            ids_by_index[record.pair.input.index] = record.pair.input.channel_id
        normalized_ids = tuple(ids_by_index)
        if any(not value for value in normalized_ids):
            raise ValueError(
                "Pair state does not identify every source channel; "
                "pass source_channel_ids explicitly for a selected pair subset"
            )
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("Pair state source channel IDs must be unique")
    else:
        normalized_ids = tuple(
            _normalize_source_channel_id(item, name="Source channel id") for item in source_channel_ids
        )
        if len(normalized_ids) != source_count:
            raise ValueError("Source channel ID count must match the pair state source channel count")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("Source channel IDs must be unique")
    if any(
        record.pair.output.channel_id != normalized_ids[record.pair.output.index]
        or record.pair.input.channel_id != normalized_ids[record.pair.input.index]
        for record in records
    ):
        raise ValueError("Pair role channel IDs do not match source channel IDs")
    return records, normalized_ids


def _expected_pair_domain(
    record: SpectralPairState,
    *,
    quantity: PairQuantity,
    scaling: SpectralScaling | None,
    denominator_role: TransferDenominator,
) -> DerivedSpectralDomain:
    """Derive the domain that a concrete pairwise Frame is allowed to carry."""
    if quantity == "coherence":
        return derive_coherence_domain()
    if quantity == "csd":
        if scaling is None:
            raise ValueError("CSD pair state requires a scaling mode")
        return derive_csd_domain(record.pair, scaling)
    return derive_transfer_domain(record.pair, denominator_role)


class PairwiseSpectralFrame(BaseFrame[Any]):
    """Private flattened pairwise frequency-domain Frame base.

    The array contract is ``(pair, frequency)`` internally and public
    ``data`` follows the normal Frame convention by suppressing the pair axis
    for a single pair.  Pair meaning is carried by immutable
    :class:`SpectralPairState` records, never by labels or lineage.
    """

    _xarray_dim_suffix = ("channel", "frequency")
    _pair_quantity: ClassVar[PairQuantity]

    def __init__(
        self,
        data: DaArray | np.ndarray[Any, Any],
        sampling_rate: float,
        n_fft: int,
        window: str,
        pair_state: Sequence[SpectralPairState],
        frequency_indices: Sequence[int] | None = None,
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
        normalized_n_fft = _normalize_positive_integer(n_fft, name="n_fft")
        normalized_window = _normalize_nonblank(window, name="window")
        complete_frequency_count = normalized_n_fft // 2 + 1
        if frequency_indices is None:
            normalized_frequency_indices = tuple(range(complete_frequency_count))
        else:
            if isinstance(frequency_indices, (str, bytes)):
                raise TypeError("frequency_indices must be an ordered sequence of integers")
            normalized_frequency_indices = tuple(frequency_indices)
            if not normalized_frequency_indices:
                raise ValueError("frequency_indices must contain at least one frequency bin")
            if any(
                isinstance(index, bool) or not isinstance(index, numbers.Integral)
                for index in normalized_frequency_indices
            ):
                raise TypeError("frequency_indices must contain only integer bin indices")
            normalized_frequency_indices = tuple(int(index) for index in normalized_frequency_indices)
            if any(index < 0 or index >= complete_frequency_count for index in normalized_frequency_indices):
                raise ValueError("frequency_indices contains a bin outside the n_fft frequency axis")
            if len(set(normalized_frequency_indices)) != len(normalized_frequency_indices):
                raise ValueError("frequency_indices must not contain duplicate bins")
        normalized_data = _normalize_pairwise_data(
            data,
            n_fft=normalized_n_fft,
            frequency_count=len(normalized_frequency_indices),
            frame_type=type(self).__name__,
            domain=self._data_domain,
        )
        records, normalized_source_ids = _normalize_pair_state(
            pair_state,
            data_rows=int(normalized_data.shape[0]),
            source_channel_ids=source_channel_ids,
        )
        scaling = cast(SpectralScaling | None, getattr(self, "scaling", None))
        denominator_role = cast(TransferDenominator, getattr(self, "denominator_role", "input"))
        for record in records:
            expected_domain = _expected_pair_domain(
                record,
                quantity=self._pair_quantity,
                scaling=scaling,
                denominator_role=denominator_role,
            )
            if record.domain != expected_domain:
                raise ValueError(
                    f"{type(self).__name__} pair domain does not match its typed pair and definition: "
                    f"expected {expected_domain!r}, got {record.domain!r}"
                )
        if channel_metadata is None:
            channel_metadata = _metadata_for_pair_state(records)
        else:
            channel_metadata = self._normalize_channel_metadata_for_count(channel_metadata, len(records))
        if any(float(channel.calibration.factor) != 1.0 for channel in channel_metadata):
            raise ValueError(
                f"{type(self).__name__} output channel calibration must be 1.0; "
                "input calibration is consumed before the pairwise operation and must not be reapplied."
            )
        for row, (channel, record) in enumerate(zip(channel_metadata, records, strict=True)):
            if channel.unit != record.domain.unit or float(channel.ref) != float(record.domain.reference):
                raise ValueError(
                    f"{type(self).__name__} channel metadata at row {row} must match its typed pair domain."
                )
        expected_channel_ids = [record.row_id for record in records]
        if channel_ids is None:
            channel_ids = expected_channel_ids
        if len(channel_ids) != len(records):
            raise ValueError("Pair channel ID count must match pair data rows")
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("Pair channel IDs must be unique")
        if channel_ids != expected_channel_ids:
            raise ValueError(
                "Pair channel IDs must match typed pair row IDs; "
                "channel row identity is carried by pair_state, not by an arbitrary label."
            )
        self.n_fft = normalized_n_fft
        self.window = normalized_window
        self._frequency_indices = normalized_frequency_indices
        self._pair_state = records
        self._source_channel_ids = normalized_source_ids
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

    @property
    def _data_domain(self) -> Literal["real", "complex"]:
        return "real"

    @property
    def n_pairs(self) -> int:
        """Return the number of flattened pair rows currently selected."""
        return len(self._pair_state)

    @property
    def n_source_channels(self) -> int:
        """Return the source channel count used by canonical pair indices."""
        return len(self._source_channel_ids)

    @property
    def source_channel_ids(self) -> tuple[str, ...]:
        """Return immutable opaque source-channel IDs in source index order."""
        return self._source_channel_ids

    @property
    def pairs(self) -> tuple[OrderedSpectralPair, ...]:
        """Return ordered pair roles in current row order."""
        return tuple(record.pair for record in self._pair_state)

    @property
    def ordered_pairs(self) -> tuple[OrderedSpectralPair, ...]:
        """Alias for :attr:`pairs` emphasizing row ordering."""
        return self.pairs

    @property
    def pair_state(self) -> tuple[SpectralPairState, ...]:
        """Return immutable typed state for every current flattened row."""
        return self._pair_state

    @property
    def pair_domains(self) -> tuple[DerivedSpectralDomain, ...]:
        """Return immutable derived unit/reference state for every row."""
        return tuple(record.domain for record in self._pair_state)

    @property
    def freqs(self) -> NDArrayReal:
        """Return the represented one-sided frequency bins in Hz."""
        complete = np.fft.rfftfreq(self.n_fft, 1.0 / self.sampling_rate)
        return complete[list(self._frequency_indices)].copy()

    def pair_at(self, row_index: int) -> OrderedSpectralPair:
        """Return the typed pair at a current flattened row index."""
        if isinstance(row_index, bool) or not isinstance(row_index, numbers.Integral):
            raise TypeError("Pair row index must be an integer")
        index = int(row_index)
        if index < 0:
            index += self.n_pairs
        if not 0 <= index < self.n_pairs:
            raise IndexError(f"Pair row index out of range: {row_index}")
        return self._pair_state[index].pair

    def get_pair(self, row_index: int) -> OrderedSpectralPair:
        """Return the typed pair at a current flattened row index."""
        return self.pair_at(row_index)

    def pair_row_index(self, output: int | str, input: int | str) -> int:
        """Return the current row for an output/input source-ID or index pair.

        String selectors are opaque source IDs. Display labels are intentionally
        not accepted as identity selectors.
        """
        output_index = self._source_index(output, role="output")
        input_index = self._source_index(input, role="input")
        for row, record in enumerate(self._pair_state):
            if record.pair.output.index == output_index and record.pair.input.index == input_index:
                return row
        raise KeyError(
            f"Pair ({output!r}, {input!r}) is not selected. Select only an output/input pair present in this Frame."
        )

    def select_pair(self: PairwiseFrameT, output: int | str, input: int | str) -> PairwiseFrameT:
        """Return one selected pair while preserving its concrete Frame type."""
        return cast(PairwiseFrameT, self.get_channel(self.pair_row_index(output, input)))

    def select_pairs(self: PairwiseFrameT, rows: Sequence[int]) -> PairwiseFrameT:
        """Return selected flattened rows with their exact typed pair state."""
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError("Pair rows must be an ordered sequence of integer row indices")
        normalized: list[int] = []
        for value in rows:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError("Pair rows must contain integer indices")
            index = int(value)
            if index < 0:
                index += self.n_pairs
            if not 0 <= index < self.n_pairs:
                raise IndexError(f"Pair row index out of range: {value}")
            normalized.append(index)
        if not normalized:
            raise ValueError("Cannot select an empty pair set")
        return cast(PairwiseFrameT, self.get_channel(normalized))

    def _source_index(self, value: int | str, *, role: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (numbers.Integral, str)):
            raise TypeError(f"{role} selector must be a source-channel integer index or opaque ID")
        if isinstance(value, str):
            if value not in self._source_channel_ids:
                raise KeyError(f"Unknown {role} source channel ID {value!r}; display labels are not pair identity")
            return self._source_channel_ids.index(value)
        index = int(value)
        if index < 0:
            index += self.n_source_channels
        if not 0 <= index < self.n_source_channels:
            raise IndexError(f"{role} source channel index out of range: {value}")
        return index

    def _selection_constructor_kwargs(self, indices: Sequence[int]) -> dict[str, Any]:
        """Select typed rows alongside BaseFrame's channel-row selection."""
        return {"pair_state": tuple(self._pair_state[index] for index in indices)}

    def _frequency_indices_for_slice(self, selector: slice) -> tuple[int, ...]:
        """Map a public frequency slice onto canonical one-sided bin IDs."""
        start, stop, step = selector.indices(len(self._frequency_indices))
        return self._frequency_indices[slice(start, stop, step)]

    def _handle_multidim_indexing(self: PairwiseFrameT, key: tuple[Any, ...]) -> PairwiseFrameT:
        """Select pair rows and represented frequency bins together."""
        if len(key) > self._data.ndim:
            raise ValueError(f"Invalid key length: {len(key)} for shape {self.shape}")
        indices = self._channel_indices(key[0])
        axis_selectors = key[1:]
        if axis_selectors and not all(isinstance(selector, slice) for selector in axis_selectors):
            raise ValueError("Only slice selectors on the frequency axis are supported for pairwise Frames")
        selected_data = self._data[indices]
        frequency_indices = self._frequency_indices
        if axis_selectors:
            if len(axis_selectors) != 1:
                raise ValueError("Pairwise Frames expose only one non-channel frequency axis")
            selector = cast(slice, axis_selectors[0])
            frequency_indices = self._frequency_indices_for_slice(selector)
            selected_data = selected_data[(slice(None), selector)]
        return self._create_new_instance(
            data=selected_data,
            channel_metadata=self._borrowed_channel_metadata_descriptors(indices),
            channel_ids=self._channel_ids_for_selection(indices),
            source_time_offset=self.source_time_offset[indices],
            lineage=self._required_semantic_lineage(),
            frequency_indices=frequency_indices,
            **self._selection_constructor_kwargs(indices),
        )

    def _create_new_instance(self: PairwiseFrameT, data: DaArray, **kwargs: Any) -> PairwiseFrameT:
        """Preserve or subset typed state during generic Frame reconstruction."""
        if "pair_state" not in kwargs:
            channel_ids = kwargs.get("channel_ids")
            if channel_ids is None or tuple(channel_ids) == tuple(self._channel_ids):
                kwargs["pair_state"] = self._pair_state
            else:
                by_id = dict(zip(self._channel_ids, self._pair_state, strict=True))
                try:
                    kwargs["pair_state"] = tuple(by_id[channel_id] for channel_id in channel_ids)
                except KeyError as exc:
                    raise ValueError(
                        "Pair state cannot be reconstructed from unknown row IDs; "
                        "use the pair selection hook or preserve existing pair IDs."
                    ) from exc
        return super()._create_new_instance(data, **kwargs)

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """Preserve frequency and source-identity state for Frame copies."""
        return {
            "n_fft": self.n_fft,
            "window": self.window,
            "source_channel_ids": self._source_channel_ids,
            "frequency_indices": self._frequency_indices,
        }

    def _get_dataframe_index(self) -> pd.Index[Any]:
        """Return frequency rows for DataFrame export."""
        pd = require_pandas(f"{type(self).__name__}.to_dataframe")
        return pd.Index(self.freqs, name="frequency")

    def _reject_arithmetic(self) -> None:
        raise TypeError(
            f"Arithmetic is undefined for {type(self).__name__}; "
            "select a quantity-specific pair or use its documented property."
        )

    def _binary_op(self, *args: Any, **kwargs: Any) -> Any:
        self._reject_arithmetic()

    def _binary_operand_op(self, *args: Any, **kwargs: Any) -> Any:
        self._reject_arithmetic()

    def __array_ufunc__(self, *args: Any, **kwargs: Any) -> Any:
        """Reject NumPy arithmetic that would otherwise bypass the quantity contract."""
        self._reject_arithmetic()

    def _supports_base_reverse_scalar_op(self) -> bool:
        return True

    def _public_pair_values(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return values if self.n_pairs != 1 else values[0]

    def _row_values(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.n_pairs == 1:
            return values.reshape(1, -1)
        return values

    def _unit_summary(self, prefix: str) -> str:
        units = {record.domain.unit for record in self._pair_state}
        unit = next(iter(units)) if len(units) == 1 else "pair-dependent units"
        return f"{prefix} [{unit}]"

    def _matrix_plot_entries(
        self,
        view: str | None,
        Aw: bool,  # noqa: N803
    ) -> tuple[tuple[int, int, np.ndarray[Any, Any], str], ...]:
        """Return typed matrix positions and plotted values for the strategy."""
        values, _ = self._plot_frequency_values(view=view, Aw=Aw)
        rows = self._row_values(values)
        return tuple(
            (
                record.pair.output.index,
                record.pair.input.index,
                rows[row],
                self.channels[row].label,
            )
            for row, record in enumerate(self._pair_state)
        )

    def _plot_frequency_values(
        self,
        *,
        view: str | None,
        Aw: bool,  # noqa: N803
    ) -> tuple[np.ndarray[Any, Any], str]:
        raise NotImplementedError

    def _plot_ylabel(self, *, view: str | None, Aw: bool) -> str:  # noqa: N803
        """Return a quantity-specific ylabel without materializing Frame data."""
        raise NotImplementedError

    def plot(
        self,
        plot_type: str = "frequency",
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        alpha: float = 1.0,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        Aw: bool = False,  # noqa: N803
        view: str | None = None,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Plot a quantity-specific frequency or typed pair matrix view.

        ``view`` is interpreted by the concrete Frame (for example ``phase`` or
        ``level``); no operation history or display label is inspected. Plotting
        materializes the requested values, while Frame construction remains lazy.
        """
        from wandas.visualization.plotting import create_operation

        strategy = create_operation(plot_type)
        plot_kwargs: dict[str, Any] = {
            "title": title,
            "overlay": overlay,
            "Aw": Aw,
            "view": view,
            **kwargs,
        }
        if xlabel is not None:
            plot_kwargs["xlabel"] = xlabel
        if ylabel is not None:
            plot_kwargs["ylabel"] = ylabel
        if alpha != 1.0:
            plot_kwargs["alpha"] = alpha
        if xlim is not None:
            plot_kwargs["xlim"] = xlim
        if ylim is not None:
            plot_kwargs["ylim"] = ylim
        return strategy.plot(self, ax=ax, **plot_kwargs)

    def plot_matrix(
        self,
        plot_type: str = "matrix",
        *,
        view: str | None = None,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Plot selected pairs at their typed output-row/input-column cells."""
        from wandas.visualization.plotting import create_operation

        return create_operation(plot_type).plot(self, view=view, **kwargs)


class CoherenceFrame(PairwiseSpectralFrame):
    """Magnitude-squared coherence values in the dimensionless 0-to-1 domain."""

    _pair_quantity = "coherence"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raw_data = args[0] if args else kwargs.get("data")
        if raw_data is not None and not isinstance(raw_data, DaArray):
            _validate_coherence_block(np.asarray(raw_data))
        super().__init__(*args, **kwargs)
        # Keep Dask operation construction lazy; validation runs when a block is
        # materialized and therefore cannot trigger an eager compute here.
        self._xr = self._xr.copy(data=da.map_blocks(_validate_coherence_block, self._data, dtype=self._data.dtype))

    @property
    def coherence(self) -> NDArrayReal:
        """Return raw magnitude-squared coherence values in the public shape."""
        return cast(NDArrayReal, self.data)

    def _plot_frequency_values(
        self,
        *,
        view: str | None,
        Aw: bool,  # noqa: N803
    ) -> tuple[np.ndarray[Any, Any], str]:
        return np.asarray(self.coherence), self._plot_ylabel(view=view, Aw=Aw)

    def _plot_ylabel(self, *, view: str | None, Aw: bool) -> str:  # noqa: N803
        reject_pairwise_a_weighting(Aw)
        if view not in {None, "coherence", "raw"}:
            raise ValueError("CoherenceFrame supports only the 'coherence' plot view")
        return "Coherence"


class CrossSpectralFrame(PairwiseSpectralFrame):
    """Complex cross-spectral values ``P_out_in`` with typed pair domains."""

    _pair_quantity = "csd"

    def __init__(self, scaling: SpectralScaling, *args: Any, **kwargs: Any) -> None:
        if scaling not in {"spectrum", "density"}:
            raise ValueError("CrossSpectralFrame scaling must be 'spectrum' or 'density'")
        self._scaling = scaling
        super().__init__(*args, **kwargs)

    @property
    def scaling(self) -> SpectralScaling:
        """Return the immutable CSD scaling contract."""
        return self._scaling

    @property
    def _data_domain(self) -> Literal["real", "complex"]:
        return "complex"

    @property
    def magnitude(self) -> NDArrayReal:
        """Return ``abs(P_out_in)`` in the public shape."""
        return cast(NDArrayReal, np.abs(self.data))

    @property
    def phase(self) -> NDArrayReal:
        """Return ``angle(P_out_in)`` in radians in the public shape."""
        return cast(NDArrayReal, np.angle(self.data))

    @property
    def level_db(self) -> NDArrayReal:
        """Return CSD level ``10 * log10(abs(P_out_in) / pair_reference)``."""
        magnitude = self._row_values(np.asarray(self.magnitude))
        levels = np.stack(
            [
                csd_level(row.astype(complex), record.domain.reference)
                for row, record in zip(magnitude, self._pair_state)
            ]
        )
        return cast(NDArrayReal, self._public_pair_values(levels))

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        result = super()._get_additional_init_kwargs()
        result["scaling"] = self.scaling
        return result

    def _plot_frequency_values(
        self,
        *,
        view: str | None,
        Aw: bool,  # noqa: N803
    ) -> tuple[np.ndarray[Any, Any], str]:
        ylabel = self._plot_ylabel(view=view, Aw=Aw)
        selected = "magnitude" if view is None else view
        if selected == "magnitude":
            return np.asarray(self.magnitude), ylabel
        if selected == "phase":
            return np.asarray(self.phase), ylabel
        if selected == "level":
            return np.asarray(self.level_db), ylabel
        raise AssertionError("CrossSpectralFrame._plot_ylabel must validate the view")

    def _plot_ylabel(self, *, view: str | None, Aw: bool) -> str:  # noqa: N803
        reject_pairwise_a_weighting(Aw)
        selected = "magnitude" if view is None else view
        if selected == "magnitude":
            return self._unit_summary("CSD magnitude")
        if selected == "phase":
            return "CSD phase [rad]"
        if selected == "level":
            return "CSD level [dB]"
        raise ValueError("CrossSpectralFrame view must be 'magnitude', 'phase', or 'level'")


class TransferFunctionFrame(PairwiseSpectralFrame):
    """Complex transfer values with truthful denominator and reference state."""

    _pair_quantity = "transfer"

    def __init__(
        self,
        scaling: SpectralScaling,
        denominator_role: TransferDenominator = "input",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if scaling not in {"spectrum", "density"}:
            raise ValueError("TransferFunctionFrame scaling must be 'spectrum' or 'density'")
        if denominator_role not in {"input", "output"}:
            raise ValueError("TransferFunctionFrame denominator_role must be 'input' or 'output'")
        self._scaling = scaling
        self._denominator_role = denominator_role
        super().__init__(*args, **kwargs)

    @property
    def scaling(self) -> SpectralScaling:
        """Return the immutable transfer scaling contract."""
        return self._scaling

    @property
    def denominator_role(self) -> TransferDenominator:
        """Return the immutable transfer denominator definition."""
        return self._denominator_role

    @property
    def _data_domain(self) -> Literal["real", "complex"]:
        return "complex"

    @property
    def definition(self) -> str:
        """Return the persisted numerical-definition identifier."""
        return "canonical_input_denominator" if self.denominator_role == "input" else "legacy_output_denominator"

    @property
    def gain(self) -> NDArrayReal:
        """Return linear transfer gain ``abs(H_out_in)``."""
        return cast(NDArrayReal, np.abs(self.data))

    @property
    def phase(self) -> NDArrayReal:
        """Return transfer phase ``angle(H_out_in)`` in radians."""
        return cast(NDArrayReal, np.angle(self.data))

    @property
    def gain_db(self) -> NDArrayReal:
        """Return ``20 * log10(abs(H))`` for dimensionless selected pairs only."""
        if any(record.domain.unit != "1" for record in self._pair_state):
            raise ValueError(
                "gain_db is defined only for dimensionless transfer pairs. "
                "Select a same-unit pair first or use transfer_level_db for explicit unit references."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 20.0 * np.log10(self._row_values(np.asarray(self.gain)))
        return cast(NDArrayReal, self._public_pair_values(result))

    @property
    def transfer_level_db(self) -> NDArrayReal:
        """Return transfer level relative to each typed output/input reference ratio."""
        gains = self._row_values(np.asarray(self.gain))
        levels = np.stack(
            [
                transfer_level(row.astype(complex), record.domain.reference)
                for row, record in zip(gains, self._pair_state)
            ]
        )
        return cast(NDArrayReal, self._public_pair_values(levels))

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        result = super()._get_additional_init_kwargs()
        result.update({"scaling": self.scaling, "denominator_role": self.denominator_role})
        return result

    def _plot_frequency_values(
        self,
        *,
        view: str | None,
        Aw: bool,  # noqa: N803
    ) -> tuple[np.ndarray[Any, Any], str]:
        ylabel = self._plot_ylabel(view=view, Aw=Aw)
        selected = "gain" if view is None else view
        if selected == "gain":
            return np.asarray(self.gain), ylabel
        if selected == "phase":
            return np.asarray(self.phase), ylabel
        if selected == "gain_db":
            return np.asarray(self.gain_db), ylabel
        if selected == "transfer_level_db":
            return np.asarray(self.transfer_level_db), ylabel
        raise AssertionError("TransferFunctionFrame._plot_ylabel must validate the view")

    def _plot_ylabel(self, *, view: str | None, Aw: bool) -> str:  # noqa: N803
        reject_pairwise_a_weighting(Aw)
        selected = "gain" if view is None else view
        if selected == "gain":
            return self._unit_summary("Transfer gain")
        if selected == "phase":
            return "Transfer phase [rad]"
        if selected == "gain_db":
            return "Transfer gain [dB]"
        if selected == "transfer_level_db":
            return "Transfer level [dB]"
        raise ValueError("TransferFunctionFrame view must be 'gain', 'phase', 'gain_db', or 'transfer_level_db'")


__all__ = [
    "CoherenceFrame",
    "CrossSpectralFrame",
    "PairwiseSpectralFrame",
    "SpectralPairState",
    "TransferFunctionFrame",
    "build_pair_state",
]
