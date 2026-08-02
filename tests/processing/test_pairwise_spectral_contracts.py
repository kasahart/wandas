"""Independent numerical and semantic contracts for pairwise spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy import signal as ss

from tests.frame_helpers import channel_first_values
from tests.processing_helpers import run_operation_eager
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame
from wandas.processing.spectral import CSD, TransferFunction
from wandas.processing.spectral_contracts import (
    DerivedSpectralDomain,
    OrderedSpectralPair,
    SpectralChannelRole,
    as_output_input_pairs,
    csd_level,
    derive_coherence_domain,
    derive_csd_domain,
    derive_transfer_domain,
    flatten_output_input_pairs,
    flatten_pair_index,
    reject_pairwise_a_weighting,
    transfer_function_ratio,
    transfer_level,
)

_SAMPLING_RATE = 256.0
_N_SAMPLES = 512
_WINDOW = "hann"
_DETREND = "constant"
_AVERAGE = "mean"
_SCALINGS = ("spectrum", "density")
_WINDOWS = ("boxcar", "hann")
_FFT_CONFIGS = ((32, 32, 16), (64, 32, 16))

SpectralScaling = Literal["spectrum", "density"]


@dataclass(frozen=True)
class _OracleFixture:
    """Physical signals and channel domains owned entirely by this test."""

    signals: np.ndarray
    labels: tuple[str, ...]
    units: tuple[str, ...]
    references: tuple[float, ...]

    @property
    def n_channels(self) -> int:
        return self.signals.shape[0]


def _make_fixture(n_channels: int) -> _OracleFixture:
    """Build distinct delayed/gained channels without using Wandas helpers."""
    if n_channels not in (2, 3):
        raise ValueError("The independent fixture supports two or three channels")

    time = np.arange(_N_SAMPLES, dtype=float) / _SAMPLING_RATE
    rng = np.random.default_rng(402_110)
    reference = (
        0.15
        + 1.1 * np.sin(2.0 * np.pi * 16.0 * time + 0.37)
        + 0.45 * np.cos(2.0 * np.pi * 40.0 * time - 0.19)
        + 0.02 * rng.standard_normal(_N_SAMPLES)
    )
    positive_delay = 2.25 * np.roll(reference, 3)
    negative_delay = 0.65 * np.roll(reference, -2)

    all_signals = np.stack([reference, positive_delay, negative_delay]).astype(np.float64)
    return _OracleFixture(
        signals=all_signals[:n_channels].copy(),
        labels=("reference", "positive-delay", "negative-delay")[:n_channels],
        units=("V", "Pa", "V")[:n_channels],
        references=(2.0, 5.0, 3.0)[:n_channels],
    )


def _scipy_csd_matrix(
    signals: np.ndarray,
    *,
    scaling: SpectralScaling,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str,
) -> np.ndarray:
    """Compute each output/input CSD pair independently with SciPy."""
    return np.asarray(
        [
            [
                ss.csd(
                    x=signals[input_index],
                    y=signals[output_index],
                    fs=_SAMPLING_RATE,
                    nperseg=win_length,
                    noverlap=win_length - hop_length,
                    nfft=n_fft,
                    window=window,
                    detrend=_DETREND,
                    scaling=scaling,
                    average=_AVERAGE,
                )[1]
                for input_index in range(signals.shape[0])
            ]
            for output_index in range(signals.shape[0])
        ]
    )


def _scipy_power_matrix(
    signals: np.ndarray,
    *,
    scaling: SpectralScaling,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str,
) -> np.ndarray:
    """Compute the input auto-spectrum independently for every channel."""
    return np.asarray(
        [
            ss.welch(
                x=signals[input_index],
                fs=_SAMPLING_RATE,
                nperseg=win_length,
                noverlap=win_length - hop_length,
                nfft=n_fft,
                window=window,
                detrend=_DETREND,
                scaling=scaling,
                average=_AVERAGE,
            )[1]
            for input_index in range(signals.shape[0])
        ]
    )


def _scipy_transfer_matrix(
    signals: np.ndarray,
    *,
    scaling: SpectralScaling,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str,
) -> np.ndarray:
    """Compute H[output, input] from independent SciPy pair calls."""
    return np.asarray(
        [
            [
                ss.csd(
                    x=signals[input_index],
                    y=signals[output_index],
                    fs=_SAMPLING_RATE,
                    nperseg=win_length,
                    noverlap=win_length - hop_length,
                    nfft=n_fft,
                    window=window,
                    detrend=_DETREND,
                    scaling=scaling,
                    average=_AVERAGE,
                )[1]
                / ss.welch(
                    x=signals[input_index],
                    fs=_SAMPLING_RATE,
                    nperseg=win_length,
                    noverlap=win_length - hop_length,
                    nfft=n_fft,
                    window=window,
                    detrend=_DETREND,
                    scaling=scaling,
                    average=_AVERAGE,
                )[1]
                for input_index in range(signals.shape[0])
            ]
            for output_index in range(signals.shape[0])
        ]
    )


def _flatten_pair_matrix(matrix: np.ndarray) -> np.ndarray:
    """Flatten an oracle matrix by explicit output-major/input-minor iteration."""
    return np.stack(
        [
            matrix[output_index][input_index]
            for output_index in range(matrix.shape[0])
            for input_index in range(matrix.shape[1])
        ]
    )


def _make_operation(
    operation_name: str,
    *,
    scaling: SpectralScaling,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str,
) -> CSD | TransferFunction:
    operation_type: type[CSD] | type[TransferFunction]
    if operation_name == "csd":
        operation_type = CSD
    elif operation_name == "transfer_function":
        operation_type = TransferFunction
    else:
        raise AssertionError(f"Unexpected pairwise operation: {operation_name}")
    return operation_type(
        _SAMPLING_RATE,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        detrend=_DETREND,
        scaling=scaling,
        average=_AVERAGE,
    )


def _roles(fixture: _OracleFixture) -> tuple[SpectralChannelRole, ...]:
    return tuple(
        SpectralChannelRole(
            index=index,
            label=fixture.labels[index],
            unit=fixture.units[index],
            reference=fixture.references[index],
        )
        for index in range(fixture.n_channels)
    )


def _pair(fixture: _OracleFixture, output_index: int, input_index: int) -> OrderedSpectralPair:
    roles = _roles(fixture)
    return OrderedSpectralPair(
        output=roles[output_index],
        input=roles[input_index],
        n_channels=fixture.n_channels,
    )


@pytest.mark.parametrize("n_channels", [2, 3], ids=lambda value: f"{value}ch")
@pytest.mark.parametrize("output_index,input_index", [(0, 1), (1, 0)], ids=["out0-in1", "out1-in0"])
def test_ordered_pair_contract_uses_output_major_indices(n_channels: int, output_index: int, input_index: int) -> None:
    fixture = _make_fixture(n_channels)
    pair = _pair(fixture, output_index, input_index)

    assert pair.output.index == output_index
    assert pair.input.index == input_index
    assert pair.pair_index == output_index * n_channels + input_index
    assert flatten_pair_index(output_index, input_index, n_channels) == pair.pair_index


def test_spectral_roles_and_pairs_are_immutable_and_carry_physical_domains() -> None:
    fixture = _make_fixture(3)
    role = _roles(fixture)[1]
    pair = _pair(fixture, 1, 0)

    assert role.label == "positive-delay"
    assert role.unit == "Pa"
    assert role.reference == 5.0
    assert pair.output == role
    assert pair.input.index == 0

    with pytest.raises((AttributeError, TypeError)):
        role.unit = "mV"  # ty: ignore[invalid-assignment]
    with pytest.raises((AttributeError, TypeError)):
        pair.input = role  # ty: ignore[invalid-assignment]


def test_pairwise_contract_validation_edges_are_explicit() -> None:
    """Pure contract values reject malformed roles, domains, and pair arrays."""
    role = SpectralChannelRole(index=0, label="source", unit="Pa", reference=1.0)

    for invalid_index in (True, 1.5, "0"):
        with pytest.raises(TypeError, match="integer index"):
            SpectralChannelRole(
                index=cast(Any, invalid_index),
                label="source",
                unit="Pa",
                reference=1.0,
            )
    with pytest.raises(TypeError, match="Channel label"):
        SpectralChannelRole(index=0, label=cast(Any, 1), unit="Pa", reference=1.0)
    with pytest.raises(TypeError, match="Channel unit"):
        SpectralChannelRole(index=0, label="source", unit=cast(Any, 1), reference=1.0)
    for invalid_reference in (True, "1"):
        with pytest.raises(TypeError, match="positive finite"):
            SpectralChannelRole(index=0, label="source", unit="Pa", reference=cast(Any, invalid_reference))
    for invalid_reference in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="positive finite"):
            SpectralChannelRole(index=0, label="source", unit="Pa", reference=invalid_reference)

    with pytest.raises(ValueError, match="Channel count"):
        OrderedSpectralPair(output=role, input=role, n_channels=0)
    with pytest.raises(TypeError, match="requires SpectralChannelRole"):
        OrderedSpectralPair(output=cast(Any, object()), input=role, n_channels=1)
    out_of_range = SpectralChannelRole(index=1, label="output", unit="Pa", reference=1.0)
    with pytest.raises(ValueError, match="Output channel index"):
        OrderedSpectralPair(output=out_of_range, input=role, n_channels=1)
    with pytest.raises(ValueError, match="Input channel index"):
        OrderedSpectralPair(output=role, input=out_of_range, n_channels=1)
    with pytest.raises(TypeError, match="Derived spectral unit"):
        DerivedSpectralDomain(unit=cast(Any, 1), reference=1.0)

    with pytest.raises(ValueError, match="Channel count"):
        flatten_pair_index(0, 0, 0)
    with pytest.raises(ValueError, match="Output channel index"):
        flatten_pair_index(1, 0, 1)
    with pytest.raises(ValueError, match="Input channel index"):
        flatten_pair_index(0, 1, 1)

    fixture = _make_fixture(3)
    pair = _pair(fixture, output_index=1, input_index=0)
    with pytest.raises(ValueError, match="scaling"):
        derive_csd_domain(pair, "invalid")
    assert derive_coherence_domain() == DerivedSpectralDomain(unit="1", reference=1.0)

    output_only = OrderedSpectralPair(
        output=SpectralChannelRole(index=0, label="output", unit="V", reference=2.0),
        input=SpectralChannelRole(index=1, label="input", unit="", reference=4.0),
        n_channels=2,
    )
    input_only = OrderedSpectralPair(
        output=SpectralChannelRole(index=0, label="output", unit="", reference=2.0),
        input=SpectralChannelRole(index=1, label="input", unit="V", reference=4.0),
        n_channels=2,
    )
    assert derive_transfer_domain(output_only).unit == "V"
    assert derive_transfer_domain(input_only).unit == "1/V"

    with pytest.raises(ValueError, match="output and input axes"):
        transfer_function_ratio(np.ones(2, dtype=np.complex128), np.ones(1))
    with pytest.raises(ValueError, match="omit only the output axis"):
        transfer_function_ratio(np.ones((2, 2, 3), dtype=np.complex128), np.ones((2, 3, 1)))
    with pytest.raises(ValueError, match="one value per input channel"):
        transfer_function_ratio(np.ones((2, 2, 3), dtype=np.complex128), np.ones((1, 3)))
    with pytest.raises(ValueError, match="equal input and output axes"):
        as_output_input_pairs(np.ones((2, 3, 4)))
    with pytest.raises(ValueError, match="equal output and input axes"):
        flatten_output_input_pairs(np.ones((2, 3, 4)))


def test_derived_csd_domain_distinguishes_same_units_and_scaling() -> None:
    fixture = _make_fixture(3)
    different_units = _pair(fixture, output_index=1, input_index=0)
    same_units = _pair(fixture, output_index=2, input_index=0)

    spectrum = derive_csd_domain(different_units, "spectrum")
    density = derive_csd_domain(different_units, "density")
    same_unit_spectrum = derive_csd_domain(same_units, "spectrum")

    assert isinstance(spectrum, DerivedSpectralDomain)
    assert spectrum.unit == "V*Pa"
    assert spectrum.reference == 10.0
    assert density.unit == "V*Pa/Hz"
    assert density.reference == spectrum.reference
    assert same_unit_spectrum.unit == "V*V"
    assert same_unit_spectrum.reference == 6.0


def test_derived_transfer_domain_is_output_over_input() -> None:
    fixture = _make_fixture(3)
    different_units = derive_transfer_domain(_pair(fixture, output_index=1, input_index=0))
    same_units = derive_transfer_domain(_pair(fixture, output_index=2, input_index=0))

    assert different_units.unit == "Pa/V"
    assert different_units.reference == 2.5
    assert same_units.unit == "1"
    assert same_units.reference == 1.5


def test_pairwise_levels_use_quantity_appropriate_logarithms() -> None:
    values = np.array([1.0 + 0.0j, 2.0 + 2.0j, 10.0 - 5.0j])
    reference = 2.0

    np.testing.assert_allclose(csd_level(values, reference), 10.0 * np.log10(np.abs(values) / reference))
    np.testing.assert_allclose(transfer_level(values, reference), 20.0 * np.log10(np.abs(values) / reference))


def test_pairwise_a_weighting_is_explicitly_rejected() -> None:
    assert reject_pairwise_a_weighting(False) is None
    with pytest.raises(ValueError, match="A-weighting"):
        reject_pairwise_a_weighting(True)


def test_transfer_ratio_handles_zero_and_near_zero_denominators_without_flooring() -> None:
    cross = np.array(
        [
            [[2.0 + 1.0j, 3.0 - 4.0j], [5.0 + 2.0j, 7.0 - 1.0j]],
            [[11.0 - 3.0j, 13.0 + 2.0j], [17.0 + 1.0j, 19.0 - 5.0j]],
        ]
    )
    input_power = np.array([[2.0, 1.0e-12], [4.0, 0.0]])
    cross_before = cross.copy()
    power_before = input_power.copy()

    result = transfer_function_ratio(cross, input_power)

    np.testing.assert_allclose(result[:, 0, 0], cross[:, 0, 0] / 2.0)
    np.testing.assert_allclose(result[:, 1, 0], cross[:, 1, 0] / 4.0)
    assert np.isfinite(result[:, 0, 1]).all()
    assert np.isnan(result[:, 1, 1].real).all()
    assert np.isnan(result[:, 1, 1].imag).all()
    nonfinite_denominator = transfer_function_ratio(
        np.ones((1, 1, 1), dtype=np.complex128),
        np.array([[np.inf]], dtype=np.float64),
    )
    assert np.isnan(nonfinite_denominator.real).all()
    assert np.isnan(nonfinite_denominator.imag).all()
    np.testing.assert_array_equal(cross, cross_before)
    np.testing.assert_array_equal(input_power, power_before)


@pytest.mark.parametrize("scaling", _SCALINGS)
def test_scipy_transfer_ratio_cancels_common_spectrum_density_scaling(scaling: SpectralScaling) -> None:
    fixture = _make_fixture(3)
    cross = _scipy_csd_matrix(
        fixture.signals,
        scaling=scaling,
        n_fft=64,
        win_length=32,
        hop_length=16,
        window=_WINDOW,
    )
    power = _scipy_power_matrix(
        fixture.signals,
        scaling=scaling,
        n_fft=64,
        win_length=32,
        hop_length=16,
        window=_WINDOW,
    )
    expected = _scipy_transfer_matrix(
        fixture.signals,
        scaling=scaling,
        n_fft=64,
        win_length=32,
        hop_length=16,
        window=_WINDOW,
    )

    actual = transfer_function_ratio(cross, power)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("n_channels", [2, 3], ids=lambda value: f"{value}ch")
@pytest.mark.parametrize("operation_name", ["csd", "transfer_function"])
@pytest.mark.parametrize("scaling", _SCALINGS)
@pytest.mark.parametrize("window", _WINDOWS)
@pytest.mark.parametrize(
    "n_fft,win_length,hop_length",
    _FFT_CONFIGS,
    ids=["full-window-half-hop", "zero-padded-half-window"],
)
def test_pairwise_operation_matches_independent_scipy_pairs(
    n_channels: int,
    operation_name: str,
    scaling: SpectralScaling,
    window: str,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> None:
    fixture = _make_fixture(n_channels)
    if operation_name == "csd":
        expected_matrix = _scipy_csd_matrix(
            fixture.signals,
            scaling=scaling,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
        )
    else:
        expected_matrix = _scipy_transfer_matrix(
            fixture.signals,
            scaling=scaling,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
        )
    expected = _flatten_pair_matrix(expected_matrix)

    operation = _make_operation(
        operation_name,
        scaling=scaling,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
    )
    input_data = da.from_array(fixture.signals, chunks=(1, -1))
    input_snapshot = fixture.signals.copy()
    with patch.object(DaArray, "compute") as compute:
        result = operation.process(input_data)
        compute.assert_not_called()

    expected_shape = (n_channels * n_channels, n_fft // 2 + 1)
    assert result.shape == expected_shape
    assert operation.calculate_output_shape(input_data.shape) == expected_shape
    assert np.dtype(result.dtype) == np.dtype(np.complex128)
    assert np.dtype(operation.calculate_output_dtype(input_data.dtype)) == np.dtype(np.complex128)

    actual = result.compute()
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11, equal_nan=True)
    np.testing.assert_array_equal(input_data.compute(), input_snapshot)


def test_transfer_operation_marks_exact_zero_input_psd_as_complex_nan() -> None:
    """The operation applies the explicit zero-denominator policy per input pair."""
    time = np.arange(128, dtype=float) / _SAMPLING_RATE
    signals = np.stack([np.zeros_like(time), np.sin(2.0 * np.pi * 16.0 * time)])

    actual = run_operation_eager(
        _make_operation(
            "transfer_function",
            scaling="spectrum",
            n_fft=32,
            win_length=32,
            hop_length=16,
            window="boxcar",
        ),
        signals,
    )

    # Pair indices 0 and 2 have input_index=0 and therefore an exact zero PSD.
    assert np.isnan(actual[[0, 2]].real).all()
    assert np.isnan(actual[[0, 2]].imag).all()
    assert np.isfinite(actual[[1, 3]]).all()


def test_csd_oracle_preserves_complex_conjugate_relation_for_ordered_pairs() -> None:
    fixture = _make_fixture(3)
    matrix = _scipy_csd_matrix(
        fixture.signals,
        scaling="spectrum",
        n_fft=64,
        win_length=32,
        hop_length=16,
        window="boxcar",
    )
    actual = run_operation_eager(
        CSD(
            _SAMPLING_RATE,
            n_fft=64,
            win_length=32,
            hop_length=16,
            window="boxcar",
            scaling="spectrum",
            average=_AVERAGE,
        ),
        fixture.signals,
    )

    for output_index in range(fixture.n_channels):
        for input_index in range(output_index):
            np.testing.assert_allclose(matrix[output_index][input_index], np.conj(matrix[input_index][output_index]))
    np.testing.assert_allclose(actual, _flatten_pair_matrix(matrix), rtol=1e-9, atol=1e-11)


def test_transfer_oracle_covers_known_gain_signed_delay_and_inverse_pairs() -> None:
    fixture = _make_fixture(3)
    matrix = _scipy_transfer_matrix(
        fixture.signals,
        scaling="spectrum",
        n_fft=64,
        win_length=32,
        hop_length=16,
        window="boxcar",
    )
    frequencies = np.fft.rfftfreq(64, 1.0 / _SAMPLING_RATE)
    tone_bins = [int(np.flatnonzero(np.isclose(frequencies, frequency))[0]) for frequency in (16.0, 40.0)]

    for output_index, gain, delay in ((1, 2.25, 3), (2, 0.65, -2)):
        expected_phase = gain * np.exp(-2j * np.pi * frequencies * delay / _SAMPLING_RATE)
        np.testing.assert_allclose(matrix[output_index][0][tone_bins], expected_phase[tone_bins], rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(
            matrix[output_index][0][tone_bins] * matrix[0][output_index][tone_bins],
            1.0,
            rtol=1e-4,
            atol=1e-4,
        )

    flattened = _flatten_pair_matrix(matrix)
    assert not np.allclose(flattened[1], flattened[2])
    assert not np.allclose(flattened[3], flattened[6])


@pytest.mark.parametrize(
    ("operation_name", "scaling"),
    [("csd", "density"), ("transfer_function", "spectrum")],
)
def test_public_channel_pairwise_route_matches_independent_pairs(
    operation_name: str,
    scaling: SpectralScaling,
) -> None:
    """The current public route preserves the contract before #406 typed Frames."""
    fixture = _make_fixture(3)
    frame = ChannelFrame.from_numpy(fixture.signals, _SAMPLING_RATE, ch_labels=list(fixture.labels))
    frame = frame.with_source_time_offset([0.25, 0.5, 0.75])

    if operation_name == "csd":
        expected_matrix = _scipy_csd_matrix(
            fixture.signals,
            scaling=scaling,
            n_fft=64,
            win_length=32,
            hop_length=16,
            window="hann",
        )
        expected_operation_version = 2
    else:
        expected_matrix = _scipy_transfer_matrix(
            fixture.signals,
            scaling=scaling,
            n_fft=64,
            win_length=32,
            hop_length=16,
            window="hann",
        )
        expected_operation_version = 2

    with patch.object(DaArray, "compute") as compute:
        result = getattr(frame, operation_name)(
            n_fft=64,
            win_length=32,
            hop_length=16,
            window="hann",
            scaling=scaling,
        )
        compute.assert_not_called()

    assert isinstance(result, SpectralFrame)
    assert isinstance(result._data, DaArray)
    assert result.previous is frame
    assert result.sampling_rate == _SAMPLING_RATE
    assert result.operation_history[-1]["version"] == expected_operation_version
    np.testing.assert_array_equal(result.source_time_offset, np.array([0.25, 0.5, 0.75] * 3))
    np.testing.assert_allclose(
        channel_first_values(result),
        _flatten_pair_matrix(expected_matrix),
        rtol=1e-9,
        atol=1e-11,
        equal_nan=True,
    )
    expected_labels = [
        (
            f"csd({fixture.labels[output_index]}, {fixture.labels[input_index]})"
            if operation_name == "csd"
            else f"$H_{{{fixture.labels[output_index]}, {fixture.labels[input_index]}}}$"
        )
        for output_index in range(fixture.n_channels)
        for input_index in range(fixture.n_channels)
    ]
    assert result.labels == expected_labels
    np.testing.assert_array_equal(channel_first_values(frame), fixture.signals)
