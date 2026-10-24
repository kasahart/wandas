"""Recipe replay and publication contracts for versioned spectral operations."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy.signal import get_window
from scipy.signal import welch as scipy_welch

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.cepstral import CepstralFrame
from wandas.frames.channel import ChannelFrame
from wandas.frames.pairwise import CoherenceFrame, CrossSpectralFrame, TransferFunctionFrame
from wandas.frames.spectral import SpectralFrame
from wandas.pipeline import RecipePlan, default_recipe_registry
from wandas.processing import get_operation
from wandas.processing.cepstral import Cepstrum, _RecipeCepstrumV2
from wandas.processing.spectral import (
    FFT,
    Welch,
    _RecipeFFTV2,
    _RecipeIFFTV2,
)

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
) -> np.ndarray:
    """Calculate released v2 (pad-then-window) FFT amplitudes independently of Wandas."""
    analysis = values[..., :n_fft]
    if analysis.shape[-1] < n_fft:
        analysis = np.pad(analysis, [(0, 0)] * (analysis.ndim - 1) + [(0, n_fft - analysis.shape[-1])])
    window_values = get_window(window, n_fft)
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
) -> np.ndarray:
    """Calculate the normalized real cepstrum from the independent FFT oracle."""
    magnitude = np.abs(
        _rfft_amplitude_oracle(
            values,
            n_fft=n_fft,
            window=window,
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


def test_released_welch_v2_payload_keeps_win_length_fallback_for_zero_n_fft() -> None:
    """Version 2 replays n_fft=0 with win_length; version 3 rejects it like other spectral calls."""
    source = _source(np.ones((1, 5), dtype=np.float64))
    payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": "wandas.audio.welch",
                "version": 2,
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

    replayed = RecipePlan.from_dict(payload).apply({"signal": source})

    expected = _welch_oracle(
        np.ones((1, 5), dtype=np.float64),
        n_fft=5,
        win_length=5,
        hop_length=5,
        window="boxcar",
        detrend="constant",
    )
    np.testing.assert_allclose(channel_first_values(replayed), expected)
    assert replayed.n_fft == 5
    assert replayed.operation_history[-1]["version"] == 2
    with pytest.raises(ValueError, match="Invalid FFT size"):
        source.welch(n_fft=0, hop_length=5, win_length=5, window="boxcar")


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
    )
    np.testing.assert_allclose(result, expected)
    assert np.isclose(result[0, 0], 1.25)
    assert np.isclose(result[0, 1], 2.0)
    assert np.isclose(result[0, 2], 1.5)
    assert np.isclose(result[0, -1], 3.0)


def _single_node_payload(operation: str, version: int, entries: list[list[Any]]) -> dict[str, Any]:
    return {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": operation,
                "version": version,
                "inputs": ["input-0"],
                "params": {"$type": "map", "entries": entries},
            }
        ],
        "output": "node-0",
    }


def test_released_fft_v2_payload_replays_padding_before_window() -> None:
    """Version 2 keeps the released n_fft-long window over zero-padded input."""
    values = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64)
    source = _source(values)
    plan = RecipePlan.from_dict(_single_node_payload("wandas.audio.fft", 2, [["n_fft", 8], ["window", "hamming"]]))

    with patch.object(FFT, "_process", side_effect=AssertionError("Recipe v2 invoked current FFT")):
        replayed = plan.apply({"signal": source})
        replayed_values = channel_first_values(replayed)

    expected_v2 = _rfft_amplitude_oracle(values, n_fft=8, window="hamming")
    np.testing.assert_allclose(replayed_values, expected_v2)
    assert not np.allclose(expected_v2, channel_first_values(source.fft(n_fft=8, window="hamming")))
    assert replayed.win_length == 8
    assert replayed.operation_history == [
        {"operation": "wandas.audio.fft", "version": 2, "params": {"n_fft": 8, "window": "hamming"}}
    ]
    # The stored n_fft-long window keeps the released v2 IFFT round trip.
    np.testing.assert_allclose(
        channel_first_values(replayed.ifft()),
        np.pad(values, ((0, 0), (0, 5))) * get_window("hamming", 8),
        atol=1e-12,
    )


def test_released_cepstrum_v2_payload_replays_padding_before_window() -> None:
    """Version 2 keeps the released n_fft-long window over zero-padded input."""
    values = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64)
    source = _source(values, offset=2.5)
    plan = RecipePlan.from_dict(_single_node_payload("wandas.audio.cepstrum", 2, [["n_fft", 8], ["window", "hamming"]]))

    with patch.object(Cepstrum, "_process", side_effect=AssertionError("Recipe v2 invoked current Cepstrum")):
        replayed = plan.apply({"signal": source})
        replayed_values = channel_first_values(replayed)

    expected_v2 = _cepstrum_oracle(values, n_fft=8, window="hamming", floor=1e-12)
    np.testing.assert_allclose(replayed_values, expected_v2)
    assert not np.allclose(expected_v2, channel_first_values(source.cepstrum(n_fft=8, window="hamming")))
    assert replayed.operation_history[-1]["version"] == 2


def test_public_fft_v3_recipe_roundtrip_is_lazy_and_preserves_frame_contract() -> None:
    """Public FFT extraction serializes version 3 and replays it end to end."""
    source = _source(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64))
    with patch.object(DaArray, "compute", autospec=True) as compute:
        direct = source.fft(n_fft=8, window="hamming")
        plan = RecipePlan.from_frame(direct, input_names=("signal",))
        payload = plan.to_dict()
        loaded = RecipePlan.from_dict(json.loads(json.dumps(payload, allow_nan=False)))
        replayed = loaded.apply({"signal": source})
        compute.assert_not_called()

    assert payload["nodes"][0]["version"] == 3
    assert loaded.to_dict() == payload
    assert [(node.operation, node.version) for node in loaded.nodes] == [("wandas.audio.fft", 3)]
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(direct))
    _assert_frame_contract(direct, source)
    _assert_frame_contract(replayed, source)
    assert direct.lineage is not None and direct.lineage.operation is not None
    assert replayed.lineage is not None and replayed.lineage.operation is not None
    assert direct.lineage.operation.version == replayed.lineage.operation.version == 3
    _assert_source_unchanged(source, np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64))


def test_public_welch_v3_recipe_roundtrip_is_lazy_and_preserves_frame_contract() -> None:
    """Public Welch extraction serializes version 3 and replays it end to end."""
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

    assert payload["nodes"][0]["version"] == 3
    assert loaded.to_dict() == payload
    assert [(node.operation, node.version) for node in loaded.nodes] == [("wandas.audio.welch", 3)]
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(direct))
    _assert_frame_contract(direct, source)
    _assert_frame_contract(replayed, source)
    assert direct.lineage is not None and direct.lineage.operation is not None
    assert replayed.lineage is not None and replayed.lineage.operation is not None
    assert direct.lineage.operation.version == replayed.lineage.operation.version == 3
    _assert_source_unchanged(source, values)


def test_public_cepstrum_v3_recipe_roundtrip_is_lazy_and_preserves_frame_contract() -> None:
    """Public Cepstrum extraction serializes version 3 and replays it end to end."""
    source = _source(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64), offset=2.5)
    with patch.object(DaArray, "compute", autospec=True) as compute:
        direct = source.cepstrum(n_fft=8, window="hamming", floor=_FLOOR)
        plan = RecipePlan.from_frame(direct, input_names=("signal",))
        payload = plan.to_dict()
        loaded = RecipePlan.from_dict(json.loads(json.dumps(payload, allow_nan=False)))
        replayed = loaded.apply({"signal": source})
        compute.assert_not_called()

    assert payload["nodes"][0]["version"] == 3
    assert loaded.to_dict() == payload
    assert [(node.operation, node.version) for node in loaded.nodes] == [("wandas.audio.cepstrum", 3)]
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(direct))
    assert direct.operation_history[-1]["params"]["floor"] == _FLOOR
    assert replayed.operation_history[-1]["params"]["floor"] == _FLOOR
    _assert_frame_contract(direct, source)
    _assert_frame_contract(replayed, source)
    assert direct.lineage is not None and direct.lineage.operation is not None
    assert replayed.lineage is not None and replayed.lineage.operation is not None
    assert direct.lineage.operation.version == replayed.lineage.operation.version == 3
    _assert_source_unchanged(source, np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64))


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


def test_default_registry_replays_the_versions_written_by_the_previous_release() -> None:
    """Versions written by 0.9.0 replay; versions written only before 0.9.0 are gone."""
    registry = default_recipe_registry()
    for operation_id in (
        "wandas.audio.fft",
        "wandas.audio.welch",
        "wandas.audio.cepstrum",
        "wandas.audio.coherence",
        "wandas.audio.csd",
        "wandas.audio.transfer_function",
    ):
        assert registry.require(operation_id, 2).version == 2
        with pytest.raises(KeyError, match="not registered"):
            registry.require(operation_id, 1)
    for operation_id in ("wandas.audio.fft", "wandas.audio.cepstrum"):
        assert registry.require(operation_id, 3).version == 3
    assert registry.require("wandas.frame.select_pair", 1).version == 1

    for private_name, operation_class in (
        ("_recipe_fft_v2", _RecipeFFTV2),
        ("_recipe_cepstrum_v2", _RecipeCepstrumV2),
        ("_recipe_ifft_v2", _RecipeIFFTV2),
    ):
        assert get_operation(private_name) is operation_class

    for private_name in (
        "_recipe_fft_v1",
        "_recipe_welch_v1",
        "_recipe_cepstrum_v1",
        "_recipe_ifft_v1",
        "_recipe_noct_synthesis_v1",
        "_recipe_transfer_function_v1",
        "_recipe_rms_trend_v1",
        "_recipe_sound_level_v1",
    ):
        with pytest.raises(ValueError, match="Unknown operation type"):
            get_operation(private_name)


@pytest.mark.parametrize(
    ("method_name", "operation_id", "frame_type"),
    [
        ("transfer_function", "wandas.cross_spectral.transfer_function", TransferFunctionFrame),
        ("coherence", "wandas.cross_spectral.coherence", CoherenceFrame),
    ],
)
def test_csd_derivation_recipe_replays_after_source_reordering(
    method_name: str,
    operation_id: str,
    frame_type: type[Any],
) -> None:
    """A CSD-derived quantity replays from the complete CSD of the new source."""
    values = np.stack(
        [
            np.sin(2.0 * np.pi * np.arange(256) / 16.0),
            1.5 * np.sin(2.0 * np.pi * np.arange(256) / 16.0 + 0.3),
            0.75 * np.cos(2.0 * np.pi * np.arange(256) / 32.0),
        ]
    )
    source = _source(values, sampling_rate=256.0)
    csd = source.csd(n_fft=32, hop_length=16, win_length=32, window="hann")
    selected = getattr(csd, method_name)().select_pair("channel-id-1", "channel-id-0")

    payload = RecipePlan.from_frame(selected, input_names=("signal",)).to_dict()
    assert [node["operation"] for node in payload["nodes"]] == [
        "wandas.audio.csd",
        operation_id,
        "wandas.frame.select_pair",
    ]

    reordered = _source(
        values[[2, 0, 1]],
        sampling_rate=256.0,
        channel_ids=["channel-id-2", "channel-id-0", "channel-id-1"],
    )
    replayed = RecipePlan.from_dict(json.loads(json.dumps(payload))).apply({"signal": reordered})

    assert type(replayed) is frame_type
    np.testing.assert_allclose(
        channel_first_values(replayed),
        channel_first_values(selected),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
    assert replayed.pairs[0].output.channel_id == "channel-id-1"
    assert replayed.pairs[0].input.channel_id == "channel-id-0"
