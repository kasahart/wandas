"""Recipe compatibility for versioned N-octave synthesis semantics."""

from __future__ import annotations

import json
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

import wandas.processing.spectral as spectral
from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.noct import NOctFrame
from wandas.frames.spectral import SpectralFrame
from wandas.pipeline import RecipePlan, default_recipe_registry
from wandas.pipeline.errors import RecipeExecutionError
from wandas.processing.spectral import NOctSynthesis


def _spectral_source() -> SpectralFrame:
    """Return a lazy odd-length spectrum with complete channel provenance."""
    return SpectralFrame(
        da.from_array(
            np.array(
                [
                    [1.0 + 0.5j, 2.0 + 0.25j, 3.0 + 0.1j, 4.0 + 0.05j, 5.0],
                    [0.5 + 0.25j, 1.5 + 0.1j, 2.5 + 0.05j, 3.5, 4.5],
                ],
                dtype=np.complex128,
            ),
            chunks=(1, -1),
        ),
        sampling_rate=48000,
        n_fft=9,
        window="hann",
        label="spectral-source",
        metadata={"recording": {"take": "A"}},
        channel_metadata=[
            ChannelMetadata(
                label="left",
                calibration=ChannelCalibration(factor=2.0, unit="Pa", ref=2e-5),
                extra={"sensor": "L"},
            ),
            ChannelMetadata(
                label="right",
                calibration=ChannelCalibration(factor=3.0, unit="Pa", ref=2e-5),
                extra={"sensor": "R"},
            ),
        ],
        channel_ids=["mic-left", "mic-right"],
        source_time_offset=[1.25, 2.5],
    )


def _patch_noct_backend(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Patch MoSQITo with a lazy-test backend that records only frequency axes."""
    captured_freqs: list[np.ndarray] = []
    monkeypatch.setattr(spectral, "require_mosqito_center_freq", lambda _feature: None)
    monkeypatch.setattr(
        spectral,
        "_center_freq",
        lambda **_kwargs: (np.arange(2), np.array([100.0, 200.0])),
    )

    def fake_noct_synthesis(
        *, spectrum: np.ndarray, freqs: np.ndarray, **_kwargs: object
    ) -> tuple[np.ndarray, np.ndarray]:
        captured_freqs.append(np.array(freqs, copy=True))
        channels = 1 if spectrum.ndim == 1 else spectrum.shape[-1]
        values = np.arange(2 * channels, dtype=np.float64).reshape(2, channels)
        return values, np.array([100.0, 200.0])

    monkeypatch.setattr(spectral, "noct_synthesis", fake_noct_synthesis)
    return captured_freqs


def test_released_v1_payload_replays_legacy_frequency_axis_without_current_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A literal schema-2 v1 node retains the released five-bin inference."""
    captured_freqs = _patch_noct_backend(monkeypatch)

    def fail_current_kernel(_self: object, _data: object) -> object:
        raise AssertionError("Recipe v1 invoked the current NOctSynthesis kernel")

    monkeypatch.setattr(NOctSynthesis, "_process", fail_current_kernel)
    released_payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": "wandas.spectral.noct_synthesis",
                "version": 1,
                "inputs": ["input-0"],
                "params": {
                    "$type": "map",
                    "entries": [["fmax", 1000], ["fmin", 100]],
                },
            }
        ],
        "output": "node-0",
    }

    source = _spectral_source()
    plan = RecipePlan.from_dict(released_payload)
    replayed = plan.apply({"signal": source})

    assert isinstance(replayed, NOctFrame)
    assert isinstance(replayed._data, DaArray)
    assert captured_freqs == []
    values = channel_first_values(replayed)

    np.testing.assert_allclose(captured_freqs[0], np.array([0.0, 6000.0, 12000.0, 18000.0, 24000.0]))
    assert values.shape == (2, 2)
    assert replayed.operation_history == [
        {
            "operation": "wandas.spectral.noct_synthesis",
            "version": 1,
            "params": {"fmin": 100, "fmax": 1000},
        }
    ]
    assert replayed.previous is source


def test_released_v1_payload_rejects_non_48000_sampling_rate() -> None:
    """Recipe v1 preserves the public synthesis sampling-rate contract."""
    released_payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": "wandas.spectral.noct_synthesis",
                "version": 1,
                "inputs": ["input-0"],
                "params": {
                    "$type": "map",
                    "entries": [["fmax", 1000], ["fmin", 100]],
                },
            }
        ],
        "output": "node-0",
    }
    source = SpectralFrame(
        da.from_array(np.ones((1, 5), dtype=np.complex128), chunks=(1, -1)),
        sampling_rate=44100,
        n_fft=9,
    )

    plan = RecipePlan.from_dict(released_payload)

    with pytest.raises(RecipeExecutionError, match="48000 Hz"):
        plan.apply({"signal": source})


def test_public_v2_recipe_roundtrip_uses_spectral_frame_n_fft_and_preserves_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public v2 extraction/load/apply stays lazy and does not serialize n_fft."""
    captured_freqs = _patch_noct_backend(monkeypatch)
    source = _spectral_source()

    with mock.patch.object(DaArray, "compute") as mock_compute:
        direct = source.noct_synthesis(fmin=100, fmax=1000)
        plan = RecipePlan.from_frame(direct, input_names=("signal",))
        payload = plan.to_dict()
        serialized = json.dumps(payload)
        loaded = RecipePlan.from_dict(json.loads(serialized))
        replayed = loaded.apply({"signal": source})
        mock_compute.assert_not_called()

    assert isinstance(direct, NOctFrame)
    assert isinstance(replayed, NOctFrame)
    assert direct._data.dtype == np.dtype(np.float64)
    assert replayed._data.dtype == np.dtype(np.float64)
    assert [(node.operation, node.version) for node in plan.nodes] == [("wandas.spectral.noct_synthesis", 2)]
    assert dict(plan.nodes[0].params.entries) == {"fmin": 100, "fmax": 1000}
    serialized_param_names = [entry[0] for entry in payload["nodes"][0]["params"]["entries"]]
    assert serialized_param_names == ["fmax", "fmin"]
    assert "n_fft" not in serialized_param_names
    assert [(node.operation, node.version) for node in loaded.nodes] == [("wandas.spectral.noct_synthesis", 2)]
    assert captured_freqs == []

    direct_values = channel_first_values(direct)
    replayed_values = channel_first_values(replayed)
    np.testing.assert_allclose(replayed_values, direct_values)
    np.testing.assert_allclose(captured_freqs[0], np.fft.rfftfreq(9, d=1 / 48000))
    np.testing.assert_allclose(captured_freqs[1], np.fft.rfftfreq(9, d=1 / 48000))
    assert direct.operation_history == [
        {
            "operation": "wandas.spectral.noct_synthesis",
            "version": 2,
            "params": {"fmin": 100, "fmax": 1000},
        }
    ]
    assert replayed.operation_history == direct.operation_history
    assert replayed.metadata == direct.metadata == source.metadata
    assert replayed._channel_ids == direct._channel_ids == source._channel_ids
    assert [channel.label for channel in replayed.channels] == ["left", "right"]
    assert [channel.calibration for channel in replayed.channels] == [
        channel.calibration for channel in source.channels
    ]
    assert [channel.extra for channel in replayed.channels] == [
        {"sensor": "L"},
        {"sensor": "R"},
    ]
    np.testing.assert_array_equal(replayed.source_time_offset, source.source_time_offset)
    assert replayed.previous is source
    assert replayed.lineage is not None
    assert replayed.lineage.operation is not None
    assert replayed.lineage.operation.version == 2


def test_noct_synthesis_recipe_registry_contains_legacy_and_current_versions() -> None:
    """The default immutable registry exposes both persisted operation meanings."""
    registry = default_recipe_registry()

    assert registry.require("wandas.spectral.noct_synthesis", 1).version == 1
    assert registry.require("wandas.spectral.noct_synthesis", 2).version == 2


def test_noct_synthesis_recipe_validator_rejects_invalid_g_on_load() -> None:
    """A malformed G is rejected by the operation declaration before apply."""
    payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": "wandas.spectral.noct_synthesis",
                "version": 2,
                "inputs": ["input-0"],
                "params": {
                    "$type": "map",
                    "entries": [["G", 20], ["fmax", 1000], ["fmin", 100]],
                },
            }
        ],
        "output": "node-0",
    }

    with pytest.raises(ValueError, match="Recipe"):
        RecipePlan.from_dict(payload)
