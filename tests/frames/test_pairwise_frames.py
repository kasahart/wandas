"""Public contracts for dedicated flattened pairwise spectral Frames."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy import signal as ss

import wandas as wd
from tests.frame_helpers import channel_first_values
from tests.pairwise_test_helpers import expected_pair_indices, make_pairwise_source
from tests.processing.test_pairwise_spectral_contracts import (
    _flatten_pair_matrix,
    _make_fixture,
    _scipy_csd_matrix,
    _scipy_transfer_matrix,
)
from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.pairwise import (
    CoherenceFrame,
    CrossSpectralFrame,
    PairwiseSpectralFrame,
    SpectralPairState,
    TransferFunctionFrame,
    _expected_pair_domain,
    _metadata_for_pair_state,
    _normalize_pair_state,
    build_pair_state,
)
from wandas.frames.spectral import SpectralFrame
from wandas.processing.spectral_contracts import DerivedSpectralDomain, csd_level, transfer_level

_N_FFT = 32
_WINDOW = "hann"
_HOP_LENGTH = 16
_WIN_LENGTH = 32


def _pairwise_frames() -> tuple[CoherenceFrame, CrossSpectralFrame, TransferFunctionFrame]:
    source = make_pairwise_source()
    return (
        source.coherence(n_fft=_N_FFT, win_length=_WIN_LENGTH, hop_length=_HOP_LENGTH, window=_WINDOW),
        source.csd(
            n_fft=_N_FFT,
            win_length=_WIN_LENGTH,
            hop_length=_HOP_LENGTH,
            window=_WINDOW,
            scaling="density",
        ),
        source.transfer_function(
            n_fft=_N_FFT,
            win_length=_WIN_LENGTH,
            hop_length=_HOP_LENGTH,
            window=_WINDOW,
            scaling="spectrum",
        ),
    )


def test_public_pairwise_operations_return_exact_dedicated_types_and_stay_lazy() -> None:
    source = make_pairwise_source(n_channels=3)
    source_values = channel_first_values(source).copy()

    with patch.object(DaArray, "compute", autospec=True) as compute:
        coherence = source.coherence(n_fft=_N_FFT, win_length=_WIN_LENGTH, hop_length=_HOP_LENGTH)
        csd = source.csd(
            n_fft=_N_FFT,
            win_length=_WIN_LENGTH,
            hop_length=_HOP_LENGTH,
            scaling="density",
        )
        transfer = source.transfer_function(
            n_fft=_N_FFT,
            win_length=_WIN_LENGTH,
            hop_length=_HOP_LENGTH,
            scaling="spectrum",
        )
        compute.assert_not_called()

    assert type(coherence) is CoherenceFrame
    assert type(csd) is CrossSpectralFrame
    assert type(transfer) is TransferFunctionFrame
    assert all(isinstance(frame, BaseFrame) for frame in (coherence, csd, transfer))
    assert all(not isinstance(frame, SpectralFrame) for frame in (coherence, csd, transfer))
    assert all(isinstance(frame._data, DaArray) for frame in (coherence, csd, transfer))
    assert all(frame._data.shape == (9, _N_FFT // 2 + 1) for frame in (coherence, csd, transfer))
    assert coherence.sampling_rate == csd.sampling_rate == transfer.sampling_rate == source.sampling_rate
    assert coherence.n_fft == csd.n_fft == transfer.n_fft == _N_FFT
    assert coherence.window == csd.window == transfer.window == _WINDOW
    assert csd.scaling == "density"
    assert transfer.scaling == "spectrum"
    assert transfer.denominator_role == "input"
    assert transfer.definition == "canonical_input_denominator"

    expected_offsets = np.tile(source.source_time_offset, source.n_channels)
    for frame in (coherence, csd, transfer):
        assert frame.previous is source
        assert frame.metadata == source.metadata
        np.testing.assert_array_equal(frame.source_time_offset, expected_offsets)
        assert frame.operation_history[-1]["version"] == 2
        assert frame.lineage.operation is not None
        assert frame.lineage.operation.operation_id.startswith("wandas.audio.")
    np.testing.assert_array_equal(channel_first_values(source), source_values)


@pytest.mark.parametrize("operation_name", ["fft", "welch"])
def test_ordinary_fft_and_welch_spectral_frame_amplitude_contract_regresses(
    operation_name: str,
) -> None:
    source = make_pairwise_source(n_channels=1)
    result = getattr(source, operation_name)(n_fft=_N_FFT, window="boxcar")

    assert type(result) is SpectralFrame
    assert all(hasattr(result, name) for name in ("magnitude", "power", "dB", "dBA", "ifft"))
    assert np.asarray(result.dB).shape == np.asarray(result.data).shape


def test_pair_state_is_output_major_input_minor_and_has_opaque_source_identity() -> None:
    source = make_pairwise_source(n_channels=3)
    frame = source.csd(n_fft=_N_FFT, win_length=_WIN_LENGTH, hop_length=_HOP_LENGTH, scaling="density")

    assert frame.n_pairs == 9
    assert frame.n_source_channels == 3
    assert frame.source_channel_ids == tuple(source._channel_ids)
    assert tuple((pair.output.index, pair.input.index) for pair in frame.ordered_pairs) == expected_pair_indices(3)
    assert [pair.pair_index for pair in frame.pairs] == list(range(9))
    assert all(record.pair is pair for record, pair in zip(frame.pair_state, frame.pairs, strict=True))
    assert all(
        record.row_id == channel_id for record, channel_id in zip(frame.pair_state, frame._channel_ids, strict=True)
    )
    assert frame.pair_row_index(2, 1) == 7
    assert frame.pair_row_index("source-id-2", "source-id-1") == 7
    with pytest.raises(KeyError, match="display labels are not pair identity"):
        frame.pair_row_index("source-2", "source-1")

    assert frame.labels == [f"csd(source-{output}, source-{input_})" for output, input_ in expected_pair_indices(3)]
    assert frame.channels[0].extra == {
        "output": {"sensor": "S0"},
        "input": {"sensor": "S0"},
    }
    assert frame.channels[1].extra == {
        "output": {"sensor": "S0"},
        "input": {"sensor": "S1"},
    }
    assert [record.domain.unit for record in frame.pair_state] == [
        "V*V/Hz",
        "Pa*V/Hz",
        "V*V/Hz",
        "V*Pa/Hz",
        "Pa*Pa/Hz",
        "V*Pa/Hz",
        "V*V/Hz",
        "Pa*V/Hz",
        "V*V/Hz",
    ]


@pytest.mark.parametrize("frame", _pairwise_frames(), ids=lambda value: type(value).__name__)
def test_dedicated_frames_do_not_expose_generic_amplitude_apis(frame: BaseFrame[Any]) -> None:
    forbidden = {"dB", "dBA", "power", "ifft", "noct_synthesis"}
    for name in forbidden:
        assert not hasattr(frame, name), f"{type(frame).__name__} unexpectedly exposes {name}"

    if type(frame) is CoherenceFrame:
        assert hasattr(frame, "coherence")
        assert not hasattr(frame, "magnitude")
    elif type(frame) is CrossSpectralFrame:
        assert all(hasattr(frame, name) for name in ("magnitude", "phase", "level_db"))
        assert not hasattr(frame, "gain")
    else:
        assert all(
            isinstance(getattr(type(frame), name), property)
            for name in ("gain", "phase", "gain_db", "transfer_level_db")
        )
        assert not hasattr(frame, "magnitude")


def test_properties_preserve_public_shape_and_typed_state() -> None:
    coherence, csd, transfer = _pairwise_frames()
    assert np.asarray(coherence.coherence).shape == np.asarray(coherence.data).shape
    assert np.asarray(csd.magnitude).shape == np.asarray(csd.data).shape
    assert np.asarray(csd.phase).shape == np.asarray(csd.data).shape
    assert np.asarray(csd.level_db).shape == np.asarray(csd.data).shape
    assert np.asarray(transfer.gain).shape == np.asarray(transfer.data).shape
    assert np.asarray(transfer.phase).shape == np.asarray(transfer.data).shape
    assert np.asarray(transfer.transfer_level_db).shape == np.asarray(transfer.data).shape
    assert all(
        record.pair.output.index == pair.output.index for record, pair in zip(csd.pair_state, csd.pairs, strict=True)
    )


def test_pair_selection_frequency_slice_and_annotation_copies_keep_concrete_type_and_state() -> None:
    source = make_pairwise_source(n_channels=3)
    frame = source.transfer_function(n_fft=_N_FFT, win_length=_WIN_LENGTH, hop_length=_HOP_LENGTH)

    selected = frame.select_pair("source-id-2", "source-id-0")
    assert type(selected) is TransferFunctionFrame
    assert selected.n_pairs == 1
    assert selected.pairs[0].pair_index == 6
    assert selected.pair_state[0] == frame.pair_state[6]
    assert selected.pair_row_index(2, 0) == 0
    np.testing.assert_array_equal(selected.source_time_offset, np.array([source.source_time_offset[0]]))
    assert np.asarray(selected.data).shape == (_N_FFT // 2 + 1,)

    subset = frame.select_pairs([8, 0, 4])
    assert type(subset) is TransferFunctionFrame
    assert subset.pairs == tuple(frame.pairs[index] for index in (8, 0, 4))
    assert subset.pair_state == tuple(frame.pair_state[index] for index in (8, 0, 4))
    assert subset._channel_ids == [frame._channel_ids[index] for index in (8, 0, 4)]

    sliced = frame[:, 2:10]
    assert type(sliced) is TransferFunctionFrame
    assert sliced.pair_state == frame.pair_state
    assert sliced.source_channel_ids == frame.source_channel_ids
    assert sliced._data.shape == (frame.n_pairs, 8)

    reversed_frequency = frame[:, ::-1]
    assert type(reversed_frequency) is TransferFunctionFrame
    assert reversed_frequency.pair_state == frame.pair_state
    assert reversed_frequency._frequency_indices == tuple(reversed(frame._frequency_indices))
    np.testing.assert_array_equal(reversed_frequency.freqs, frame.freqs[::-1])

    annotated = selected.with_label("selected").with_metadata({"review": "pairwise"})
    annotated = annotated.with_channel_extra(0, {"display": "kept"})
    renamed = annotated.rename_channels({0: "renamed pair"})
    assert type(renamed) is TransferFunctionFrame
    assert renamed.label == "selected"
    assert renamed.metadata["review"] == "pairwise"
    assert renamed.channels[0].extra["display"] == "kept"
    assert renamed.labels == ["renamed pair"]
    assert renamed.pair_state == selected.pair_state
    assert renamed.source_channel_ids == selected.source_channel_ids
    assert list(renamed)[0].pair_state == renamed.pair_state[:1]


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (np.full((1, 5), 1.1), ValueError, "between 0 and 1"),
        (np.full((1, 5), -0.1), ValueError, "between 0 and 1"),
        (np.full((1, 5), np.inf), ValueError, "infinity"),
        (np.ones((1, 5, 1)), ValueError, "data rank"),
        (np.ones((1, 4)), ValueError, "frequency bin count"),
        (np.ones((1, 5), dtype=np.complex128), TypeError, "real numeric dtype"),
    ],
)
def test_coherence_direct_constructor_rejects_invalid_domain_values(
    values: np.ndarray[Any, Any], error: type[Exception], message: str
) -> None:
    source = make_pairwise_source()
    valid = source.coherence(n_fft=8, win_length=8, hop_length=4, window="boxcar").pair_state[:1]
    with pytest.raises(error, match=message):
        CoherenceFrame(
            values,
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            pair_state=valid,
            source_channel_ids=source._channel_ids,
        )


def test_coherence_direct_constructor_preserves_undefined_nan_bins() -> None:
    source = make_pairwise_source()
    valid = source.coherence(n_fft=8, win_length=8, hop_length=4, window="boxcar").pair_state[:1]
    frame = CoherenceFrame(
        np.array([[0.0, np.nan, 1.0, 0.5, 0.25]]),
        source.sampling_rate,
        n_fft=8,
        window="boxcar",
        pair_state=valid,
        source_channel_ids=source._channel_ids,
    )
    values = np.asarray(frame.coherence)
    assert values.shape == (5,)
    assert np.isnan(values[1])
    np.testing.assert_allclose(values[[0, 2, 3, 4]], [0.0, 1.0, 0.5, 0.25])


def test_coherence_float32_validation_allows_only_dtype_scale_rounding() -> None:
    source = make_pairwise_source()
    valid = source.coherence(n_fft=8, win_length=8, hop_length=4, window="boxcar").pair_state[:1]
    rounded = CoherenceFrame(
        np.full((1, 5), np.float32(1.0000001), dtype=np.float32),
        source.sampling_rate,
        n_fft=8,
        window="boxcar",
        pair_state=valid,
        source_channel_ids=source._channel_ids,
    )
    np.testing.assert_allclose(np.asarray(rounded.coherence), np.float32(1.0000001))

    with pytest.raises(ValueError, match="between 0 and 1"):
        CoherenceFrame(
            np.full((1, 5), np.float32(1.00001), dtype=np.float32),
            source.sampling_rate,
            n_fft=8,
            window="boxcar",
            pair_state=valid,
            source_channel_ids=source._channel_ids,
        )


def test_csd_properties_use_complex_magnitude_phase_and_ten_log_level() -> None:
    frame = make_pairwise_source().csd(
        n_fft=_N_FFT,
        win_length=_WIN_LENGTH,
        hop_length=_HOP_LENGTH,
        scaling="density",
    )
    raw = channel_first_values(frame)
    expected_level = np.stack(
        [csd_level(row, record.domain.reference) for row, record in zip(raw, frame.pair_state, strict=True)]
    )
    np.testing.assert_allclose(frame.magnitude, np.abs(raw))
    np.testing.assert_allclose(frame.phase, np.angle(raw))
    np.testing.assert_allclose(frame.level_db, expected_level, equal_nan=True)
    assert all(record.domain.unit.endswith("/Hz") for record in frame.pair_state)


def test_transfer_properties_use_twenty_log_gain_and_references_without_flooring() -> None:
    unlike = make_pairwise_source().transfer_function(
        n_fft=_N_FFT,
        win_length=_WIN_LENGTH,
        hop_length=_HOP_LENGTH,
        scaling="spectrum",
    )
    raw = channel_first_values(unlike)
    with pytest.raises(ValueError, match="dimensionless|Select a same-unit pair"):
        _ = unlike.gain_db
    expected_level = np.stack(
        [transfer_level(row, record.domain.reference) for row, record in zip(raw, unlike.pair_state, strict=True)]
    )
    np.testing.assert_allclose(unlike.gain, np.abs(raw))
    np.testing.assert_allclose(unlike.phase, np.angle(raw))
    np.testing.assert_allclose(unlike.transfer_level_db, expected_level, equal_nan=True)

    same_unit = make_pairwise_source(units=("V", "V"), references=(2.0, 5.0)).transfer_function(
        n_fft=_N_FFT,
        win_length=_WIN_LENGTH,
        hop_length=_HOP_LENGTH,
        scaling="spectrum",
    )
    same_raw = channel_first_values(same_unit)
    np.testing.assert_allclose(
        same_unit.gain_db,
        20.0 * np.log10(np.abs(same_raw)),
        equal_nan=True,
    )

    direct = TransferFunctionFrame(
        data=np.array([[0.0 + 0.0j, 1.0e-12 + 0.0j, np.nan + 0.0j]]),
        sampling_rate=8.0,
        n_fft=4,
        window="boxcar",
        scaling="spectrum",
        pair_state=same_unit.pair_state[:1],
        source_channel_ids=same_unit.source_channel_ids,
    )
    assert np.isneginf(direct.gain_db[0])
    assert direct.gain_db[1] < -200.0
    assert np.isnan(direct.gain_db[2])
    with pytest.raises(AttributeError):
        direct.denominator_role = "output"  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        direct.scaling = "density"  # ty: ignore[invalid-assignment]


def test_pairwise_direct_constructor_rejects_reapplied_calibration_and_validates_lists() -> None:
    source = make_pairwise_source(n_channels=2)
    valid = source.coherence(n_fft=8, win_length=8, hop_length=4, window="boxcar")
    with pytest.raises(ValueError, match="calibration.*1.0|must not be reapplied"):
        CoherenceFrame(
            cast(Any, [[0.5] * 5] * 4),
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            pair_state=valid.pair_state,
            source_channel_ids=source._channel_ids,
            channel_metadata=[
                ChannelMetadata(
                    label=record.display_label,
                    calibration=ChannelCalibration(factor=2.0, unit="1", ref=1.0),
                )
                for record in valid.pair_state
            ],
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        CoherenceFrame(
            cast(Any, [[1.1] * 5] * 4),
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            pair_state=valid.pair_state,
            source_channel_ids=source._channel_ids,
        )

    with pytest.raises(ValueError, match="at least one selected pair"):
        CoherenceFrame(
            np.empty((0, 5)),
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            pair_state=(),
            source_channel_ids=(),
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"n_fft": True}, TypeError, "positive integer"),
        ({"n_fft": 0}, ValueError, "positive integer"),
        ({"window": ""}, TypeError, "non-blank string"),
        ({"frequency_indices": "bins"}, TypeError, "ordered sequence"),
        ({"frequency_indices": []}, ValueError, "at least one"),
        ({"frequency_indices": [0.5]}, TypeError, "integer bin"),
        ({"frequency_indices": [5]}, ValueError, "outside"),
        ({"frequency_indices": [0, 0]}, ValueError, "duplicate"),
    ],
)
def test_pairwise_direct_constructor_rejects_invalid_scalar_and_frequency_state(
    kwargs: dict[str, Any], error: type[Exception], message: str
) -> None:
    source = make_pairwise_source(n_channels=2)
    valid = source.coherence(n_fft=8, win_length=8, hop_length=4, window="boxcar")
    constructor = {
        "data": np.ones((4, 5)),
        "sampling_rate": source.sampling_rate,
        "n_fft": 8,
        "window": "boxcar",
        "pair_state": valid.pair_state,
        "source_channel_ids": source._channel_ids,
    }
    constructor.update(kwargs)

    with pytest.raises(error, match=message):
        CoherenceFrame(**cast(Any, constructor))


def test_pairwise_direct_constructor_accepts_one_dimensional_rows_and_rejects_complex_coherence() -> None:
    source = make_pairwise_source(n_channels=2)
    valid = source.coherence(n_fft=8, win_length=8, hop_length=4, window="boxcar")
    row = CoherenceFrame(
        np.ones(5),
        source.sampling_rate,
        n_fft=8,
        window="boxcar",
        pair_state=valid.pair_state[:1],
        source_channel_ids=source._channel_ids,
    )
    assert row._data.shape == (1, 5)
    assert row.data.shape == (5,)

    with pytest.raises(TypeError, match="real numeric dtype"):
        CoherenceFrame(
            da.from_array(np.ones((1, 5), dtype=np.complex128), chunks=(1, 5)),
            source.sampling_rate,
            n_fft=8,
            window="boxcar",
            pair_state=valid.pair_state[:1],
            source_channel_ids=source._channel_ids,
        )

    with pytest.raises(TypeError, match="complex numeric dtype"):
        CrossSpectralFrame(
            data=np.ones((1, 5)),
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            scaling="density",
            pair_state=valid.pair_state[:1],
            source_channel_ids=source._channel_ids,
        )


def test_pairwise_state_helpers_reject_malformed_identity_and_domains() -> None:
    source = make_pairwise_source(n_channels=2)
    frame = source.csd(n_fft=8, win_length=8, hop_length=4, window="boxcar", scaling="density")
    records = frame.pair_state

    with pytest.raises(TypeError, match="SpectralPairState"):
        SpectralPairState(
            pair=cast(Any, object()),
            domain=records[0].domain,
            row_id="row",
            display_label="label",
        )
    with pytest.raises(TypeError, match="DerivedSpectralDomain"):
        SpectralPairState(
            pair=records[0].pair,
            domain=cast(Any, object()),
            row_id="row",
            display_label="label",
        )

    with pytest.raises(ValueError, match="equal lengths"):
        build_pair_state(source.channels.to_list(), source._channel_ids[:1], quantity="coherence", label_template="x")
    with pytest.raises(ValueError, match="unique"):
        build_pair_state(
            source.channels.to_list(),
            [source._channel_ids[0], source._channel_ids[0]],
            quantity="coherence",
            label_template="x",
        )
    with pytest.raises(ValueError, match="scaling mode"):
        build_pair_state(source.channels.to_list(), source._channel_ids, quantity="csd", label_template="x")
    with pytest.raises(ValueError, match="cover every pair role"):
        _metadata_for_pair_state((records[1],), source.channels.to_list()[:1])

    with pytest.raises(TypeError, match="ordered sequence"):
        _normalize_pair_state(cast(Any, "not-state"), data_rows=4, source_channel_ids=source._channel_ids)
    with pytest.raises(ValueError, match="row count"):
        _normalize_pair_state(records[:1], data_rows=4, source_channel_ids=source._channel_ids)
    with pytest.raises(TypeError, match="only SpectralPairState"):
        _normalize_pair_state((cast(Any, object()),) * 4, data_rows=4, source_channel_ids=source._channel_ids)
    with pytest.raises(ValueError, match="duplicate output/input"):
        _normalize_pair_state(
            (records[0], records[0], records[2], records[3]),
            data_rows=4,
            source_channel_ids=source._channel_ids,
        )
    duplicate_row_id = replace(records[1], row_id=records[0].row_id)
    with pytest.raises(ValueError, match="duplicate row IDs"):
        _normalize_pair_state(
            (records[0], duplicate_row_id, records[2], records[3]),
            data_rows=4,
            source_channel_ids=source._channel_ids,
        )

    three_channel = make_pairwise_source(n_channels=3).csd(n_fft=8, win_length=8, hop_length=4, window="boxcar")
    with pytest.raises(ValueError, match="one source channel count"):
        _normalize_pair_state((records[0], three_channel.pair_state[1]), data_rows=2, source_channel_ids=None)
    assert _normalize_pair_state((), data_rows=0, source_channel_ids=None) == ((), ())
    with pytest.raises(ValueError, match="every source channel"):
        _normalize_pair_state((records[0],), data_rows=1, source_channel_ids=None)

    altered_role = replace(records[3].pair.output, channel_id=records[0].pair.output.channel_id)
    altered_pair = replace(records[3].pair, output=altered_role, input=altered_role)
    altered_record = replace(records[3], pair=altered_pair, row_id="pair:duplicate-source:duplicate-source")
    with pytest.raises(ValueError, match="source channel IDs must be unique"):
        _normalize_pair_state((records[0], altered_record), data_rows=2, source_channel_ids=None)
    inconsistent_role = replace(records[1].pair.input, unit="A", reference=999.0)
    inconsistent_pair = replace(records[1].pair, input=inconsistent_role)
    inconsistent_record = replace(records[1], pair=inconsistent_pair)
    with pytest.raises(ValueError, match="inconsistent source channel roles"):
        _normalize_pair_state(
            (records[0], inconsistent_record, records[2]),
            data_rows=3,
            source_channel_ids=None,
        )
    with pytest.raises(ValueError, match="count must match"):
        _normalize_pair_state(records, data_rows=4, source_channel_ids=["only"])
    with pytest.raises(ValueError, match="Source channel IDs must be unique"):
        _normalize_pair_state(records, data_rows=4, source_channel_ids=["same", "same"])
    with pytest.raises(TypeError, match="non-blank"):
        _normalize_pair_state(records, data_rows=4, source_channel_ids=[" ", source._channel_ids[1]])
    with pytest.raises(ValueError, match="role channel IDs"):
        _normalize_pair_state(records, data_rows=4, source_channel_ids=["other", source._channel_ids[1]])
    with pytest.raises(ValueError, match="CSD pair state"):
        _expected_pair_domain(records[0], quantity="csd", scaling=None, denominator_role="input")


def test_pairwise_constructor_rejects_typed_domain_and_channel_id_mismatches() -> None:
    source = make_pairwise_source(n_channels=2)
    frame = source.csd(n_fft=8, win_length=8, hop_length=4, window="boxcar", scaling="density")
    with pytest.raises(ValueError, match="scaling"):
        CrossSpectralFrame(
            data=frame._data,
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            scaling=cast(Any, "invalid"),
            pair_state=frame.pair_state,
            source_channel_ids=source._channel_ids,
        )
    with pytest.raises(ValueError, match="scaling"):
        TransferFunctionFrame(
            data=frame._data,
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            scaling=cast(Any, "invalid"),
            denominator_role="input",
            pair_state=frame.pair_state,
            source_channel_ids=source._channel_ids,
        )
    with pytest.raises(ValueError, match="denominator_role"):
        TransferFunctionFrame(
            data=frame._data,
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            scaling="spectrum",
            denominator_role=cast(Any, "invalid"),
            pair_state=frame.pair_state,
            source_channel_ids=source._channel_ids,
        )
    bad_record = replace(frame.pair_state[0], domain=DerivedSpectralDomain(unit="wrong", reference=1.0))
    records = (bad_record, *frame.pair_state[1:])
    with pytest.raises(ValueError, match="pair domain"):
        CrossSpectralFrame(
            data=frame._data,
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            scaling="density",
            pair_state=records,
            source_channel_ids=source._channel_ids,
        )
    with pytest.raises(ValueError, match="count must match"):
        CrossSpectralFrame(
            data=frame._data,
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            scaling="density",
            pair_state=frame.pair_state,
            source_channel_ids=source._channel_ids,
            channel_ids=["one"],
        )
    with pytest.raises(ValueError, match="channel IDs must be unique"):
        CrossSpectralFrame(
            data=frame._data,
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            scaling="density",
            pair_state=frame.pair_state,
            source_channel_ids=source._channel_ids,
            channel_ids=["same"] * frame.n_pairs,
        )
    with pytest.raises(ValueError, match="typed pair row IDs"):
        CrossSpectralFrame(
            data=frame._data,
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            scaling="density",
            pair_state=frame.pair_state,
            source_channel_ids=source._channel_ids,
            channel_ids=[f"arbitrary-{index}" for index in range(frame.n_pairs)],
        )
    mismatched_metadata = [
        ChannelMetadata(
            label=record.display_label,
            calibration=ChannelCalibration(factor=1.0, unit="wrong", ref=1.0),
        )
        for record in frame.pair_state
    ]
    with pytest.raises(ValueError, match="typed pair domain"):
        CrossSpectralFrame(
            data=frame._data,
            sampling_rate=source.sampling_rate,
            n_fft=8,
            window="boxcar",
            scaling="density",
            pair_state=frame.pair_state,
            source_channel_ids=source._channel_ids,
            channel_metadata=mismatched_metadata,
        )


def test_pairwise_selection_and_reconstruction_errors_are_actionable() -> None:
    frame = make_pairwise_source(n_channels=2).transfer_function(
        n_fft=8, win_length=8, hop_length=4, window="boxcar", scaling="spectrum"
    )
    assert frame.pair_domains == tuple(record.domain for record in frame.pair_state)
    assert frame.get_pair(-1) == frame.pair_at(-1)
    for invalid in (True, 1.5):
        with pytest.raises(TypeError, match="row index"):
            frame.pair_at(cast(Any, invalid))
    with pytest.raises(IndexError, match="out of range"):
        frame.pair_at(99)

    selected = frame.select_pair(0, 0)
    with pytest.raises(KeyError, match="not selected"):
        selected.pair_row_index(0, 1)
    with pytest.raises(TypeError, match="selector"):
        frame.pair_row_index(cast(Any, 1.5), 0)
    assert frame.pair_row_index(-1, 0) == 2
    with pytest.raises(IndexError, match="source channel index"):
        frame.pair_row_index(99, 0)
    with pytest.raises(KeyError, match="Unknown"):
        frame.pair_row_index("missing", 0)

    with pytest.raises(TypeError, match="ordered sequence"):
        frame.select_pairs(cast(Any, "rows"))
    with pytest.raises(TypeError, match="integer indices"):
        frame.select_pairs([True])
    with pytest.raises(IndexError, match="out of range"):
        frame.select_pairs([99])
    with pytest.raises(ValueError, match="empty pair"):
        frame.select_pairs([])
    assert frame.select_pairs([-1]).pair_at(0) == frame.pair_at(-1)

    with pytest.raises(ValueError, match="Invalid key length"):
        frame[(slice(None), slice(None), slice(None))]
    with pytest.raises(ValueError, match="Only slice selectors"):
        frame[(slice(None), [0])]
    with pytest.raises(ValueError, match="unknown row IDs"):
        frame._create_new_instance(frame._data, channel_ids=["unknown"])
    assert frame.to_dataframe().index.name == "frequency"
    assert frame._supports_base_reverse_scalar_op()
    with pytest.raises(TypeError, match="Arithmetic is undefined"):
        _ = 1.0 + frame
    with pytest.raises(TypeError, match="Arithmetic is undefined"):
        np.add(frame, 1.0)


@pytest.mark.parametrize("frame_type", [CoherenceFrame, CrossSpectralFrame, TransferFunctionFrame])
def test_public_pairwise_constructors_expose_complete_typed_signatures(frame_type: type[BaseFrame[Any]]) -> None:
    parameters = inspect.signature(frame_type).parameters
    assert not any(
        parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for parameter in parameters.values()
    )
    assert {"data", "sampling_rate", "n_fft", "window", "pair_state"} <= set(parameters)
    if frame_type is not CoherenceFrame:
        assert "scaling" in parameters
    if frame_type is TransferFunctionFrame:
        assert "denominator_role" in parameters


def test_private_pairwise_base_is_not_a_public_module_export() -> None:
    import wandas.frames.pairwise as pairwise_module

    assert "PairwiseSpectralFrame" not in pairwise_module.__all__
    assert inspect.isabstract(PairwiseSpectralFrame)
    frame = _pairwise_frames()[0]
    with pytest.raises(NotImplementedError):
        PairwiseSpectralFrame._plot_frequency_values(frame, view=None, Aw=False)
    with pytest.raises(NotImplementedError):
        PairwiseSpectralFrame._plot_ylabel(frame, view=None, Aw=False)


@pytest.mark.parametrize("frame", _pairwise_frames(), ids=lambda value: type(value).__name__)
def test_pairwise_fft_state_is_immutable(frame: PairwiseSpectralFrame) -> None:
    original_n_fft = frame.n_fft
    original_window = frame.window

    with pytest.raises(AttributeError):
        frame.n_fft = original_n_fft + 1  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        frame.window = "boxcar"  # ty: ignore[invalid-assignment]

    assert frame.n_fft == original_n_fft
    assert frame.window == original_window


@pytest.mark.parametrize("frame", _pairwise_frames(), ids=lambda value: type(value).__name__)
def test_pairwise_arithmetic_and_a_weighting_are_explicitly_rejected(frame: BaseFrame[Any]) -> None:
    with pytest.raises(TypeError, match="Arithmetic is undefined"):
        _ = frame + 1.0
    with pytest.raises(ValueError, match="A-weighting"):
        frame.plot(Aw=True)


def test_public_pairwise_values_match_reused_independent_scipy_oracles() -> None:
    fixture = _make_fixture(3)
    source = wd.ChannelFrame.from_numpy(
        fixture.signals,
        256.0,
        ch_labels=list(fixture.labels),
    )
    params: dict[str, Any] = {"n_fft": 64, "win_length": 32, "hop_length": 16, "window": "hann"}
    expected_csd = _flatten_pair_matrix(_scipy_csd_matrix(fixture.signals, scaling="density", **params))
    expected_transfer = _flatten_pair_matrix(_scipy_transfer_matrix(fixture.signals, scaling="spectrum", **params))
    actual_csd = source.csd(**params, scaling="density")
    actual_transfer = source.transfer_function(**params, scaling="spectrum")
    np.testing.assert_allclose(channel_first_values(actual_csd), expected_csd, rtol=1e-9, atol=1e-11, equal_nan=True)
    np.testing.assert_allclose(
        channel_first_values(actual_transfer),
        expected_transfer,
        rtol=1e-9,
        atol=1e-11,
        equal_nan=True,
    )

    coherence_expected = np.stack(
        [
            ss.coherence(
                fixture.signals[input_index],
                fixture.signals[output_index],
                fs=256.0,
                nperseg=32,
                noverlap=16,
                nfft=64,
                window="hann",
                detrend="constant",
            )[1]
            for output_index, input_index in expected_pair_indices(3)
        ]
    )
    actual_coherence = source.coherence(**params)
    np.testing.assert_allclose(
        channel_first_values(actual_coherence),
        coherence_expected,
        rtol=1e-9,
        atol=1e-11,
        equal_nan=True,
    )
