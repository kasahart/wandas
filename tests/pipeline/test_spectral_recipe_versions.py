"""Recipe v1 replay and v2 publication contracts for spectral operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy.signal import csd as scipy_csd
from scipy.signal import get_window
from scipy.signal import welch as scipy_welch

import wandas as wd
from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.cepstral import CepstralFrame
from wandas.frames.channel import ChannelFrame
from wandas.frames.pairwise import CoherenceFrame, CrossSpectralFrame, TransferFunctionFrame
from wandas.frames.spectral import SpectralFrame
from wandas.pipeline import RecipePlan, default_recipe_registry
from wandas.pipeline.errors import RecipeExecutionError
from wandas.processing import get_operation
from wandas.processing.cepstral import Cepstrum, _RecipeCepstrumV1
from wandas.processing.spectral import (
    FFT,
    TransferFunction,
    Welch,
    _RecipeFFTV1,
    _RecipeIFFTV1,
    _RecipeNOctSynthesisV1,
    _RecipeTransferFunctionV1,
    _RecipeWelchV1,
)
from wandas.processing.spectral_contracts import derive_transfer_domain
from wandas.processing.temporal import _RecipeRmsTrendV1, _RecipeSoundLevelV1

_SAMPLING_RATE = 8_000
_FLOOR = 1e-6


def _source(
    values: np.ndarray,
    *,
    offset: float = 0.25,
    sampling_rate: float = _SAMPLING_RATE,
    channel_ids: list[str] | None = None,
    source_time_offset: np.ndarray | None = None,
) -> ChannelFrame:
    """Build a lazy source with stable channel and recording metadata."""
    channel_count = values.shape[0]
    return ChannelFrame(
        data=da.from_array(values, chunks=(1, -1)),
        sampling_rate=sampling_rate,
        label="recipe-source",
        metadata={"recording": {"take": "A"}},
        channel_metadata=[
            ChannelMetadata(
                label=f"channel-{index}",
                calibration=ChannelCalibration(factor=1.0, unit="Pa", ref=2e-5),
                extra={"sensor": f"S{index}"},
            )
            for index in range(channel_count)
        ],
        channel_ids=(
            channel_ids if channel_ids is not None else [f"channel-id-{index}" for index in range(channel_count)]
        ),
        source_time_offset=(
            source_time_offset if source_time_offset is not None else np.arange(channel_count, dtype=float) + offset
        ),
    )


def _rfft_amplitude_oracle(
    values: np.ndarray,
    *,
    n_fft: int,
    window: str,
    pad_before_window: bool,
) -> np.ndarray:
    """Calculate FFT amplitude values independently of Wandas operations."""
    analysis = values[..., :n_fft]
    if pad_before_window and analysis.shape[-1] < n_fft:
        analysis = np.pad(analysis, [(0, 0)] * (analysis.ndim - 1) + [(0, n_fft - analysis.shape[-1])])
    window_values = get_window(window, n_fft if pad_before_window else analysis.shape[-1])
    spectrum = np.fft.rfft(analysis * window_values, n=n_fft, axis=-1)
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    spectrum[..., 1 : -1 if n_fft % 2 == 0 else None] *= 2.0
    return spectrum / np.sum(window_values)


def _cepstrum_oracle(
    values: np.ndarray,
    *,
    n_fft: int,
    window: str,
    floor: float,
    pad_before_window: bool,
) -> np.ndarray:
    """Calculate the normalized real cepstrum from the independent FFT oracle."""
    magnitude = np.abs(
        _rfft_amplitude_oracle(
            values,
            n_fft=n_fft,
            window=window,
            pad_before_window=pad_before_window,
        )
    )
    return np.asarray(np.fft.irfft(np.log(np.maximum(magnitude, floor)), n=n_fft, axis=-1), dtype=np.float64)


def _welch_oracle(
    values: np.ndarray,
    *,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str,
    detrend: str | None,
    legacy_scaling: bool,
) -> np.ndarray:
    """Calculate Welch peak amplitudes with an explicit independent bin rule."""
    _, power = scipy_welch(
        values,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        window=window,
        average="mean",
        detrend=detrend,
        scaling="spectrum",
        axis=-1,
    )
    amplitude = np.sqrt(power)
    if legacy_scaling:
        amplitude[..., 1:-1] *= np.sqrt(2.0)
    else:
        amplitude[..., 1 : -1 if n_fft % 2 == 0 else None] *= np.sqrt(2.0)
    return amplitude


def _assert_frame_contract(result: ChannelFrame | SpectralFrame | CepstralFrame, source: ChannelFrame) -> None:
    """Check metadata and process-local provenance shared by replay results."""
    assert result.sampling_rate == source.sampling_rate
    assert result.metadata == source.metadata
    assert result._channel_ids == source._channel_ids
    assert [channel.label for channel in result.channels] == [channel.label for channel in source.channels]
    assert [channel.calibration for channel in result.channels] == [channel.calibration for channel in source.channels]
    assert [channel.extra for channel in result.channels] == [channel.extra for channel in source.channels]
    np.testing.assert_array_equal(result.source_time_offset, source.source_time_offset)
    assert result.previous is source
    assert result.lineage is not None
    assert result.lineage.operation is not None


def _assert_source_unchanged(source: ChannelFrame, values: np.ndarray) -> None:
    """Check that direct execution and replay did not mutate the input Frame."""
    np.testing.assert_array_equal(channel_first_values(source), values)
    assert source.operation_history == []
    assert source.lineage is not None
    assert source.lineage.operation is None


def test_released_fft_v1_payload_replays_legacy_window_before_padding() -> None:
    """A literal schema-2 payload preserves the released short-input FFT values."""
    values = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64)
    source = _source(values)
    direct_v2 = source.fft(n_fft=8, window="hamming")
    direct_v2_values = channel_first_values(direct_v2)

    released_payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": "wandas.audio.fft",
                "version": 1,
                "inputs": ["input-0"],
                "params": {
                    "$type": "map",
                    "entries": [["n_fft", 8], ["window", "hamming"]],
                },
            }
        ],
        "output": "node-0",
    }

    plan = RecipePlan.from_dict(released_payload)
    with (
        patch.object(DaArray, "compute", autospec=True) as compute,
        patch.object(FFT, "_process", side_effect=AssertionError("Recipe v1 invoked current FFT")),
    ):
        replayed = plan.apply({"signal": source})
        compute.assert_not_called()

    expected_v1 = _rfft_amplitude_oracle(values, n_fft=8, window="hamming", pad_before_window=False)
    expected_v2 = _rfft_amplitude_oracle(values, n_fft=8, window="hamming", pad_before_window=True)
    np.testing.assert_allclose(channel_first_values(replayed), expected_v1)
    np.testing.assert_allclose(direct_v2_values, expected_v2)
    assert not np.allclose(expected_v1, expected_v2)
    assert isinstance(replayed, SpectralFrame)
    assert isinstance(replayed._data, DaArray)
    assert replayed._data.shape == (2, 5)
    assert replayed._data.dtype == np.dtype(np.complex128)
    assert replayed.n_fft == 8
    assert replayed.window == "hamming"
    assert replayed.operation_history == [
        {"operation": "wandas.audio.fft", "version": 1, "params": {"n_fft": 8, "window": "hamming"}}
    ]
    _assert_frame_contract(replayed, source)
    assert replayed.lineage.operation.version == 1
    assert direct_v2.operation_history[-1]["version"] == 2
    _assert_source_unchanged(source, values)


def test_spectral_recipe_v1_defensive_and_truncation_branches() -> None:
    """Cover v1 validation guards and the released FFT truncation path."""
    complex_operation = _RecipeCepstrumV1(_SAMPLING_RATE, n_fft=8, window="hamming", floor=_FLOOR)
    with pytest.raises(TypeError, match="real-valued input"):
        complex_operation._process(cast(Any, np.array([[1.0 + 0.0j]], dtype=np.complex128)))

    empty_operation = _RecipeCepstrumV1(_SAMPLING_RATE, n_fft=1, window="boxcar", floor=_FLOOR)
    with pytest.raises(ValueError, match="Invalid window gain"):
        empty_operation._process(np.empty((1, 0), dtype=np.float64))

    values = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    fft_operation = _RecipeFFTV1(_SAMPLING_RATE, n_fft=3, window="hamming")
    np.testing.assert_allclose(
        fft_operation._process(values),
        _rfft_amplitude_oracle(values, n_fft=3, window="hamming", pad_before_window=False),
    )

    welch_operation = _RecipeWelchV1(
        _SAMPLING_RATE,
        n_fft=5,
        hop_length=5,
        win_length=5,
        window="boxcar",
        average="mean",
    )
    with pytest.raises(ValueError, match="requires a numpy ndarray"):
        welch_operation._process(cast(Any, da.from_array(np.ones((1, 5)), chunks=(1, 5))))

    source = _source(np.ones((1, 5), dtype=np.float64))
    fallback_payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": "wandas.audio.welch",
                "version": 1,
                "inputs": ["input-0"],
                "params": {
                    "$type": "map",
                    "entries": [
                        ["average", "mean"],
                        ["hop_length", 5],
                        ["n_fft", 0],
                        ["win_length", 5],
                        ["window", "boxcar"],
                    ],
                },
            }
        ],
        "output": "node-0",
    }
    fallback_replayed = RecipePlan.from_dict(fallback_payload).apply({"signal": source})
    expected_fallback = _welch_oracle(
        np.ones((1, 5), dtype=np.float64),
        n_fft=5,
        win_length=5,
        hop_length=5,
        window="boxcar",
        detrend="constant",
        legacy_scaling=True,
    )
    np.testing.assert_allclose(channel_first_values(fallback_replayed), expected_fallback)
    assert fallback_replayed.n_fft == 5
    assert fallback_replayed.operation_history[-1]["params"]["n_fft"] == 0


def test_released_welch_v1_payload_preserves_odd_final_positive_bin() -> None:
    """A literal schema-2 payload preserves v1's odd-size Welch endpoint."""
    time = np.arange(5, dtype=float)
    values = (1.25 + 2.0 * np.cos(2 * np.pi * time / 5) + 3.0 * np.cos(4 * np.pi * time / 5))[None, :]
    source = _source(values, offset=1.5)
    direct_v2 = source.welch(n_fft=5, hop_length=5, win_length=5, window="boxcar", average="mean")
    direct_v2_values = channel_first_values(direct_v2)

    released_payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": "wandas.audio.welch",
                "version": 1,
                "inputs": ["input-0"],
                "params": {
                    "$type": "map",
                    "entries": [
                        ["average", "mean"],
                        ["hop_length", 5],
                        ["n_fft", 5],
                        ["win_length", 5],
                        ["window", "boxcar"],
                    ],
                },
            }
        ],
        "output": "node-0",
    }

    plan = RecipePlan.from_dict(released_payload)
    with (
        patch.object(DaArray, "compute", autospec=True) as compute,
        patch.object(Welch, "_process", side_effect=AssertionError("Recipe v1 invoked current Welch")),
    ):
        replayed = plan.apply({"signal": source})
        compute.assert_not_called()

    expected_v1 = _welch_oracle(
        values,
        n_fft=5,
        win_length=5,
        hop_length=5,
        window="boxcar",
        detrend="constant",
        legacy_scaling=True,
    )
    expected_v2 = _welch_oracle(
        values,
        n_fft=5,
        win_length=5,
        hop_length=5,
        window="boxcar",
        detrend="constant",
        legacy_scaling=False,
    )
    np.testing.assert_allclose(channel_first_values(replayed), expected_v1)
    np.testing.assert_allclose(direct_v2_values, expected_v2)
    assert np.isclose(expected_v1[0, 0], 0.0)
    assert np.isclose(expected_v1[0, 1], expected_v2[0, 1])
    assert not np.isclose(expected_v1[0, -1], expected_v2[0, -1])
    assert isinstance(replayed, SpectralFrame)
    assert isinstance(replayed._data, DaArray)
    assert replayed._data.shape == (1, 3)
    assert replayed._data.dtype == np.dtype(np.float64)
    assert replayed.n_fft == 5
    assert replayed.window == "boxcar"
    assert replayed.operation_history == [
        {
            "operation": "wandas.audio.welch",
            "version": 1,
            "params": {"n_fft": 5, "hop_length": 5, "win_length": 5, "window": "boxcar", "average": "mean"},
        }
    ]
    _assert_frame_contract(replayed, source)
    assert replayed.lineage.operation.version == 1
    assert direct_v2.operation_history[-1]["version"] == 2
    _assert_source_unchanged(source, values)


def test_welch_oracle_checks_dc_internal_and_even_nyquist_bins() -> None:
    """The corrected Welch rule preserves DC/Nyquist and scales interior bins."""
    time = np.arange(6, dtype=float)
    values = (
        1.25 + 2.0 * np.cos(2 * np.pi * time / 6) + 1.5 * np.cos(4 * np.pi * time / 6) + 3.0 * np.cos(np.pi * time)
    )[None, :]
    operation = Welch(
        _SAMPLING_RATE,
        n_fft=6,
        hop_length=6,
        win_length=6,
        window="boxcar",
        detrend=None,  # ty: ignore[invalid-argument-type]
    )
    result = operation._process(values)
    expected = _welch_oracle(
        values,
        n_fft=6,
        win_length=6,
        hop_length=6,
        window="boxcar",
        detrend=None,
        legacy_scaling=False,
    )
    np.testing.assert_allclose(result, expected)
    assert np.isclose(result[0, 0], 1.25)
    assert np.isclose(result[0, 1], 2.0)
    assert np.isclose(result[0, 2], 1.5)
    assert np.isclose(result[0, -1], 3.0)


def test_transfer_function_recipe_v1_preserves_legacy_denominator_and_v2_corrects_it(tmp_path: Path) -> None:
    """A literal Recipe v1 payload keeps the released output-axis denominator."""
    sampling_rate = 256.0
    time = np.arange(256, dtype=float) / sampling_rate
    values = np.stack(
        [
            np.sin(2 * np.pi * 16 * time) + 0.2 * np.cos(2 * np.pi * 40 * time),
            2.75 * np.sin(2 * np.pi * 16 * time + 0.4) + 0.1 * np.cos(2 * np.pi * 40 * time),
            0.35 * np.cos(2 * np.pi * 40 * time - 0.7) + 0.05 * np.sin(2 * np.pi * 16 * time),
        ]
    )
    source = _source(values, offset=0.75, sampling_rate=sampling_rate)
    params = {
        "n_fft": 32,
        "hop_length": 16,
        "win_length": 32,
        "window": "boxcar",
        "detrend": "constant",
        "scaling": "spectrum",
        "average": "mean",
    }
    direct_v2 = source.transfer_function(**params)

    released_payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": "wandas.audio.transfer_function",
                "version": 1,
                "inputs": ["input-0"],
                "params": {
                    "$type": "map",
                    "entries": [
                        ["average", "mean"],
                        ["detrend", "constant"],
                        ["hop_length", 16],
                        ["n_fft", 32],
                        ["scaling", "spectrum"],
                        ["win_length", 32],
                        ["window", "boxcar"],
                    ],
                },
            }
        ],
        "output": "node-0",
    }

    with (
        patch.object(DaArray, "compute", autospec=True) as compute,
        patch.object(TransferFunction, "_process", side_effect=AssertionError("Recipe v1 invoked current TF")),
    ):
        replayed = RecipePlan.from_dict(released_payload).apply({"signal": source})
        compute.assert_not_called()

    assert type(replayed) is TransferFunctionFrame
    assert type(direct_v2) is TransferFunctionFrame

    def scipy_transfer(denominator_role: str) -> np.ndarray:
        rows = []
        for output_index in range(values.shape[0]):
            for input_index in range(values.shape[0]):
                _, cross = scipy_csd(
                    values[input_index],
                    values[output_index],
                    fs=sampling_rate,
                    nperseg=32,
                    noverlap=16,
                    nfft=32,
                    window="boxcar",
                    detrend="constant",
                    scaling="spectrum",
                    average="mean",
                )
                denominator_index = output_index if denominator_role == "output" else input_index
                _, power = scipy_welch(
                    values[denominator_index],
                    fs=sampling_rate,
                    nperseg=32,
                    noverlap=16,
                    nfft=32,
                    window="boxcar",
                    detrend="constant",
                    scaling="spectrum",
                    average="mean",
                )
                rows.append(cross / power)
        return np.stack(rows)

    expected_v1 = scipy_transfer("output")
    expected_v2 = scipy_transfer("input")
    np.testing.assert_allclose(channel_first_values(replayed), expected_v1, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(channel_first_values(direct_v2), expected_v2, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert not np.allclose(expected_v1, expected_v2, equal_nan=True)
    assert replayed.operation_history[-1]["version"] == 1
    assert direct_v2.operation_history[-1]["version"] == 2
    assert replayed.denominator_role == "output"
    assert replayed.definition == "legacy_output_denominator"
    assert direct_v2.denominator_role == "input"
    assert direct_v2.definition == "canonical_input_denominator"
    assert all(record.domain == derive_transfer_domain(record.pair, "output") for record in replayed.pair_state)
    assert all(record.domain == derive_transfer_domain(record.pair, "input") for record in direct_v2.pair_state)
    assert replayed.labels == [
        f"$H_{{{values_input}, {values_output}}}$"
        for values_output in ("channel-0", "channel-1", "channel-2")
        for values_input in ("channel-0", "channel-1", "channel-2")
    ]
    assert direct_v2.labels == [
        f"$H_{{{values_output}, {values_input}}}$"
        for values_output in ("channel-0", "channel-1", "channel-2")
        for values_input in ("channel-0", "channel-1", "channel-2")
    ]
    assert isinstance(replayed._data, DaArray)

    path = tmp_path / "transfer-v1.wdf"
    replayed.save(path)
    loaded = wd.load(path)
    assert type(loaded) is TransferFunctionFrame
    assert loaded.denominator_role == "output"
    assert loaded.definition == "legacy_output_denominator"
    assert loaded.pair_state == replayed.pair_state
    assert loaded.labels == replayed.labels
    np.testing.assert_allclose(channel_first_values(loaded), expected_v1, equal_nan=True)
    _assert_source_unchanged(source, values)


def test_released_cepstrum_v1_payload_replays_legacy_window_before_padding() -> None:
    """A literal schema-2 payload preserves the released short-input cepstrum."""
    values = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64)
    source = _source(values, offset=2.5)
    direct_v2 = source.cepstrum(n_fft=8, window="hamming", floor=_FLOOR)
    direct_v2_values = channel_first_values(direct_v2)

    released_payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": "wandas.audio.cepstrum",
                "version": 1,
                "inputs": ["input-0"],
                "params": {
                    "$type": "map",
                    "entries": [
                        ["floor", {"$type": "number", "data": "3eb0c6f7a0b5ed8d", "kind": "python-float"}],
                        ["n_fft", 8],
                        ["window", "hamming"],
                    ],
                },
            }
        ],
        "output": "node-0",
    }

    plan = RecipePlan.from_dict(released_payload)
    complex_source = _source(values.astype(np.complex128), offset=2.5)
    with pytest.raises(RecipeExecutionError, match="Recipe operation failed") as error:
        plan.apply({"signal": complex_source})
    assert isinstance(error.value.__cause__, TypeError)
    assert "real-valued input" in str(error.value)

    with (
        patch.object(DaArray, "compute", autospec=True) as compute,
        patch.object(Cepstrum, "_process", side_effect=AssertionError("Recipe v1 invoked current Cepstrum")),
    ):
        replayed = plan.apply({"signal": source})
        compute.assert_not_called()

    expected_v1 = _cepstrum_oracle(
        values,
        n_fft=8,
        window="hamming",
        floor=_FLOOR,
        pad_before_window=False,
    )
    expected_v2 = _cepstrum_oracle(
        values,
        n_fft=8,
        window="hamming",
        floor=_FLOOR,
        pad_before_window=True,
    )
    np.testing.assert_allclose(channel_first_values(replayed), expected_v1)
    np.testing.assert_allclose(direct_v2_values, expected_v2)
    assert not np.allclose(expected_v1, expected_v2)
    assert isinstance(replayed, CepstralFrame)
    assert isinstance(replayed._data, DaArray)
    assert replayed._data.shape == (2, 8)
    assert replayed._data.dtype == np.dtype(np.float64)
    assert replayed.n_fft == 8
    assert replayed.window == "hamming"
    assert replayed.operation_history == [
        {
            "operation": "wandas.audio.cepstrum",
            "version": 1,
            "params": {"n_fft": 8, "window": "hamming", "floor": _FLOOR},
        }
    ]
    _assert_frame_contract(replayed, source)
    assert replayed.lineage.operation.version == 1
    assert direct_v2.operation_history[-1]["version"] == 2


def test_public_fft_v2_recipe_roundtrip_is_lazy_and_preserves_frame_contract() -> None:
    """Public FFT extraction serializes version 2 and replays it end to end."""
    source = _source(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64))
    with patch.object(DaArray, "compute", autospec=True) as compute:
        direct = source.fft(n_fft=8, window="hamming")
        plan = RecipePlan.from_frame(direct, input_names=("signal",))
        payload = plan.to_dict()
        loaded = RecipePlan.from_dict(json.loads(json.dumps(payload, allow_nan=False)))
        replayed = loaded.apply({"signal": source})
        compute.assert_not_called()

    assert payload["nodes"][0]["version"] == 2
    assert loaded.to_dict() == payload
    assert [(node.operation, node.version) for node in loaded.nodes] == [("wandas.audio.fft", 2)]
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(direct))
    _assert_frame_contract(direct, source)
    _assert_frame_contract(replayed, source)
    assert direct.lineage is not None and direct.lineage.operation is not None
    assert replayed.lineage is not None and replayed.lineage.operation is not None
    assert direct.lineage.operation.version == replayed.lineage.operation.version == 2
    _assert_source_unchanged(source, np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64))


def test_public_welch_v2_recipe_roundtrip_is_lazy_and_preserves_frame_contract() -> None:
    """Public Welch extraction serializes version 2 and replays it end to end."""
    time = np.arange(5, dtype=float)
    values = (1.25 + 2.0 * np.cos(2 * np.pi * time / 5) + 3.0 * np.cos(4 * np.pi * time / 5))[None, :]
    source = _source(values, offset=1.5)
    with patch.object(DaArray, "compute", autospec=True) as compute:
        direct = source.welch(n_fft=5, hop_length=5, win_length=5, window="boxcar", average="mean")
        plan = RecipePlan.from_frame(direct, input_names=("signal",))
        payload = plan.to_dict()
        loaded = RecipePlan.from_dict(json.loads(json.dumps(payload, allow_nan=False)))
        replayed = loaded.apply({"signal": source})
        compute.assert_not_called()

    assert payload["nodes"][0]["version"] == 2
    assert loaded.to_dict() == payload
    assert [(node.operation, node.version) for node in loaded.nodes] == [("wandas.audio.welch", 2)]
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(direct))
    _assert_frame_contract(direct, source)
    _assert_frame_contract(replayed, source)
    assert direct.lineage is not None and direct.lineage.operation is not None
    assert replayed.lineage is not None and replayed.lineage.operation is not None
    assert direct.lineage.operation.version == replayed.lineage.operation.version == 2
    _assert_source_unchanged(source, values)


def test_public_cepstrum_v2_recipe_roundtrip_is_lazy_and_preserves_frame_contract() -> None:
    """Public Cepstrum extraction serializes version 2 and replays it end to end."""
    source = _source(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64), offset=2.5)
    with patch.object(DaArray, "compute", autospec=True) as compute:
        direct = source.cepstrum(n_fft=8, window="hamming", floor=_FLOOR)
        plan = RecipePlan.from_frame(direct, input_names=("signal",))
        payload = plan.to_dict()
        loaded = RecipePlan.from_dict(json.loads(json.dumps(payload, allow_nan=False)))
        replayed = loaded.apply({"signal": source})
        compute.assert_not_called()

    assert payload["nodes"][0]["version"] == 2
    assert loaded.to_dict() == payload
    assert [(node.operation, node.version) for node in loaded.nodes] == [("wandas.audio.cepstrum", 2)]
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(direct))
    assert direct.operation_history[-1]["params"]["floor"] == _FLOOR
    assert replayed.operation_history[-1]["params"]["floor"] == _FLOOR
    _assert_frame_contract(direct, source)
    _assert_frame_contract(replayed, source)
    assert direct.lineage is not None and direct.lineage.operation is not None
    assert replayed.lineage is not None and replayed.lineage.operation is not None
    assert direct.lineage.operation.version == replayed.lineage.operation.version == 2
    _assert_source_unchanged(source, np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64))


@pytest.mark.parametrize(
    ("operation_id", "method_name", "v1_labels", "v2_labels", "params"),
    [
        (
            "wandas.audio.coherence",
            "coherence",
            [
                r"$\gamma_{channel-0, channel-0}$",
                r"$\gamma_{channel-1, channel-0}$",
                r"$\gamma_{channel-0, channel-1}$",
                r"$\gamma_{channel-1, channel-1}$",
            ],
            [
                r"$\gamma_{channel-0, channel-0}$",
                r"$\gamma_{channel-0, channel-1}$",
                r"$\gamma_{channel-1, channel-0}$",
                r"$\gamma_{channel-1, channel-1}$",
            ],
            {"n_fft": 32, "hop_length": 16, "win_length": 32, "window": "boxcar"},
        ),
        (
            "wandas.audio.csd",
            "csd",
            [
                "csd(channel-0, channel-0)",
                "csd(channel-1, channel-0)",
                "csd(channel-0, channel-1)",
                "csd(channel-1, channel-1)",
            ],
            [
                "csd(channel-0, channel-0)",
                "csd(channel-0, channel-1)",
                "csd(channel-1, channel-0)",
                "csd(channel-1, channel-1)",
            ],
            {
                "n_fft": 32,
                "hop_length": 16,
                "win_length": 32,
                "window": "boxcar",
                "scaling": "spectrum",
            },
        ),
    ],
)
def test_pairwise_recipe_v1_preserves_released_label_order(
    operation_id: str,
    method_name: str,
    v1_labels: list[str],
    v2_labels: list[str],
    params: dict[str, Any],
) -> None:
    """Schema-2 v1 replay keeps historical labels while v2 is canonical."""
    sample_index = np.arange(256)
    values = np.array(
        [
            np.sin(2 * np.pi * sample_index / 16),
            1.5 * np.sin(2 * np.pi * sample_index / 16 + 0.3),
        ],
        dtype=float,
    )
    source = _source(values, sampling_rate=256.0)
    direct_v2 = getattr(source, method_name)(**params)
    released_payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": operation_id,
                "version": 1,
                "inputs": ["input-0"],
                "params": {
                    "$type": "map",
                    "entries": [[name, value] for name, value in sorted(params.items())],
                },
            }
        ],
        "output": "node-0",
    }

    replayed = RecipePlan.from_dict(released_payload).apply({"signal": source})

    expected_type = {
        "wandas.audio.coherence": CoherenceFrame,
        "wandas.audio.csd": CrossSpectralFrame,
    }[operation_id]
    assert type(replayed) is expected_type
    assert type(direct_v2) is expected_type

    np.testing.assert_allclose(
        channel_first_values(replayed),
        channel_first_values(direct_v2),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    assert replayed.labels == v1_labels
    assert direct_v2.labels == v2_labels
    assert replayed.operation_history[-1]["version"] == 1
    assert direct_v2.operation_history[-1]["version"] == 2
    assert [(pair.output.index, pair.input.index) for pair in replayed.pairs] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [(record.pair, record.domain, record.row_id) for record in replayed.pair_state] == [
        (record.pair, record.domain, record.row_id) for record in direct_v2.pair_state
    ]
    _assert_source_unchanged(source, values)


@pytest.mark.parametrize(
    ("operation_name", "params", "expected_type"),
    [
        ("coherence", {"n_fft": 32, "hop_length": 16, "win_length": 32, "window": "boxcar"}, CoherenceFrame),
        (
            "csd",
            {
                "n_fft": 32,
                "hop_length": 16,
                "win_length": 32,
                "window": "boxcar",
                "scaling": "density",
            },
            CrossSpectralFrame,
        ),
        (
            "transfer_function",
            {
                "n_fft": 32,
                "hop_length": 16,
                "win_length": 32,
                "window": "boxcar",
                "scaling": "spectrum",
            },
            TransferFunctionFrame,
        ),
    ],
)
def test_public_pairwise_v2_recipe_roundtrip_rebuilds_exact_type_and_state(
    operation_name: str,
    params: dict[str, Any],
    expected_type: type[CoherenceFrame] | type[CrossSpectralFrame] | type[TransferFunctionFrame],
) -> None:
    values = np.stack(
        [
            np.sin(2.0 * np.pi * np.arange(256) / 16.0),
            1.5 * np.sin(2.0 * np.pi * np.arange(256) / 16.0 + 0.3),
        ]
    )
    source = _source(values, sampling_rate=256.0)
    direct = getattr(source, operation_name)(**params)
    payload = RecipePlan.from_frame(direct, input_names=("signal",)).to_dict()
    serialized = json.loads(json.dumps(payload, allow_nan=False))
    replayed = RecipePlan.from_dict(serialized).apply({"signal": source})

    assert serialized == payload
    assert type(direct) is expected_type
    assert type(replayed) is expected_type
    np.testing.assert_allclose(
        channel_first_values(replayed),
        channel_first_values(direct),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    assert replayed.pair_state == direct.pair_state
    assert replayed.source_channel_ids == direct.source_channel_ids
    assert replayed.n_pairs == direct.n_pairs
    assert replayed.operation_history[-1]["version"] == 2


def test_pair_selection_recipe_replays_opaque_source_selectors_after_reordering() -> None:
    """A selected pair Recipe resolves source IDs on the replayed input Frame."""
    values = np.stack(
        [
            np.sin(2.0 * np.pi * np.arange(256) / 16.0),
            1.5 * np.sin(2.0 * np.pi * np.arange(256) / 16.0 + 0.3),
            0.75 * np.cos(2.0 * np.pi * np.arange(256) / 32.0),
        ]
    )
    source = _source(values, sampling_rate=256.0)
    selected = source.csd(n_fft=32, hop_length=16, win_length=32, window="boxcar").select_pair(
        "channel-id-1", "channel-id-0"
    )

    payload = RecipePlan.from_frame(selected, input_names=("signal",)).to_dict()
    assert [node["operation"] for node in payload["nodes"]] == [
        "wandas.audio.csd",
        "wandas.frame.select_pair",
    ]
    assert payload["nodes"][-1]["params"] == {
        "$type": "map",
        "entries": [["input", "channel-id-0"], ["output", "channel-id-1"]],
    }

    reordered = _source(
        values[[2, 0, 1]],
        sampling_rate=256.0,
        channel_ids=["channel-id-2", "channel-id-0", "channel-id-1"],
        source_time_offset=np.array([2.25, 0.25, 1.25]),
    )
    replayed = RecipePlan.from_dict(json.loads(json.dumps(payload))).apply({"signal": reordered})

    assert type(replayed) is CrossSpectralFrame
    np.testing.assert_allclose(
        channel_first_values(replayed),
        channel_first_values(selected),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    assert replayed.pairs[0].output.channel_id == "channel-id-1"
    assert replayed.pairs[0].input.channel_id == "channel-id-0"
    assert replayed.pairs[0].output.index == 2
    assert replayed.pairs[0].input.index == 1
    assert replayed.pairs[0].pair_index == 7
    np.testing.assert_array_equal(replayed.source_time_offset, np.array([0.25]))
    assert replayed.operation_history[-1]["operation"] == "wandas.frame.select_pair"


def test_default_registry_contains_v1_and_v2_for_all_spectral_recipe_operations() -> None:
    """The immutable default registry exposes both persisted meanings."""
    registry = default_recipe_registry()
    for operation_id in (
        "wandas.audio.fft",
        "wandas.audio.welch",
        "wandas.audio.cepstrum",
        "wandas.audio.coherence",
        "wandas.audio.csd",
        "wandas.audio.transfer_function",
    ):
        assert registry.require(operation_id, 1).version == 1
        assert registry.require(operation_id, 2).version == 2
    assert registry.require("wandas.frame.select_pair", 1).version == 1

    for private_name, operation_class in (
        ("_recipe_fft_v1", _RecipeFFTV1),
        ("_recipe_welch_v1", _RecipeWelchV1),
        ("_recipe_cepstrum_v1", _RecipeCepstrumV1),
        ("_recipe_ifft_v1", _RecipeIFFTV1),
        ("_recipe_noct_synthesis_v1", _RecipeNOctSynthesisV1),
        ("_recipe_transfer_function_v1", _RecipeTransferFunctionV1),
        ("_recipe_rms_trend_v1", _RecipeRmsTrendV1),
        ("_recipe_sound_level_v1", _RecipeSoundLevelV1),
    ):
        assert get_operation(private_name) is operation_class

    for private_name in (
        "_coherence_recipe_v1",
        "_csd_recipe_v1",
    ):
        with pytest.raises(ValueError, match="Unknown operation type"):
            get_operation(private_name)
