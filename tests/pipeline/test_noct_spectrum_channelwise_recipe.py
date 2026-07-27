"""Recipe round-trip contracts for channel-wise N-octave spectrum execution."""

import json
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from mosqito.sound_level_meter import noct_spectrum

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame, ChannelMetadata
from wandas.frames.noct import NOctFrame
from wandas.pipeline import RecipePlan
from wandas.processing.semantic import thaw_params

_SAMPLING_RATE = 8_192
_SAMPLES = 2_048
_FMIN = 250.0
_FMAX = 2_000.0
_G = 10
_FR = 1_000


def _source() -> tuple[ChannelFrame, np.ndarray]:
    time = np.arange(_SAMPLES, dtype=np.float64) / _SAMPLING_RATE
    values = np.stack(
        [
            0.6 * np.sin(2 * np.pi * 375 * time) + 0.2 * np.cos(2 * np.pi * 750 * time),
            0.5 * np.sin(2 * np.pi * 625 * time) + 0.15 * np.cos(2 * np.pi * 1_250 * time),
        ]
    )
    frame = ChannelFrame(
        da.from_array(values, chunks=(1, 512)),
        sampling_rate=_SAMPLING_RATE,
        label="recipe-source",
        metadata={"workflow": {"name": "noct-spectrum"}},
        channel_metadata=[
            ChannelMetadata(
                label="first",
                calibration=ChannelCalibration(factor=2.0, unit="Pa", ref=2e-5),
                extra={"position": 1},
            ),
            ChannelMetadata(
                label="second",
                calibration=ChannelCalibration(factor=0.25, unit="V", ref=1.0),
                extra={"position": 2},
            ),
        ],
        channel_ids=["first-id", "second-id"],
        source_time_offset=[0.125, 0.375],
    )
    return frame, values


@pytest.mark.parametrize("n", [1, 3])
def test_noct_spectrum_recipe_extract_serialize_deserialize_and_replay_preserves_contract(
    n: int,
) -> None:
    source, caller_values = _source()
    caller_values_before = caller_values.copy()
    source_values = channel_first_values(source).copy()
    source_metadata = source.metadata
    source_channels = source.channels.to_list()
    source_offsets = source.source_time_offset.copy()
    source_lineage = source.lineage

    processed = source.noct_spectrum(
        fmin=_FMIN,
        fmax=_FMAX,
        n=n,
        G=_G,
        fr=_FR,
    )
    plan = RecipePlan.from_frame(processed, input_names=("signal",))
    payload = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
    loaded = RecipePlan.from_dict(payload)
    replayed = loaded.apply({"signal": source})

    assert payload["schema"] == "wandas.recipe"
    assert payload["version"] == 2
    assert payload["inputs"] == [{"id": "input-0", "name": "signal", "kind": "frame"}]
    assert len(payload["nodes"]) == 1
    assert payload["nodes"][0]["operation"] == "wandas.audio.noct_spectrum"
    assert payload["nodes"][0]["version"] == 1
    assert thaw_params(loaded.nodes[0].params) == {
        "G": _G,
        "fmax": _FMAX,
        "fmin": _FMIN,
        "fr": _FR,
        "n": n,
    }
    assert loaded.to_dict() == plan.to_dict()
    assert [node.operation for node in loaded.nodes] == ["wandas.audio.noct_spectrum"]

    assert isinstance(replayed, NOctFrame)
    assert isinstance(replayed._data, DaArray)
    assert replayed.previous is source
    assert replayed.shape == processed.shape
    assert replayed._data.dtype == np.dtype(np.float64)
    assert replayed._data.dtype == processed._data.dtype
    assert replayed._data.chunks == processed._data.chunks
    assert replayed._xr.dims == ("channel", "band")
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.fmin == processed.fmin == _FMIN
    assert replayed.fmax == processed.fmax == _FMAX
    assert replayed.n == processed.n == n
    assert replayed.G == processed.G == _G
    assert replayed.fr == processed.fr == _FR
    np.testing.assert_array_equal(replayed.freqs, processed.freqs)
    assert replayed.label == processed.label == f"1/{n}Oct of recipe-source"
    assert replayed.metadata == processed.metadata == source.metadata
    assert [channel.id for channel in replayed.channels] == ["first-id", "second-id"]
    assert replayed.labels == processed.labels == ["first", "second"]
    assert [channel.unit for channel in replayed.channels] == ["Pa", "V"]
    assert [channel.ref for channel in replayed.channels] == [2e-5, 1.0]
    assert [channel.calibration.factor for channel in replayed.channels] == [1.0, 1.0]
    assert [channel.extra for channel in replayed.channels] == [
        {"position": 1},
        {"position": 2},
    ]
    np.testing.assert_array_equal(replayed.source_time_offset, processed.source_time_offset)
    assert replayed.operation_history == processed.operation_history
    assert replayed.operation_history[-1] == {
        "operation": "wandas.audio.noct_spectrum",
        "version": 1,
        "params": {
            "fmin": _FMIN,
            "fmax": _FMAX,
            "n": n,
            "G": _G,
            "fr": _FR,
        },
    }
    assert replayed.lineage.operation is not None
    assert replayed.lineage.operation.operation_id == "wandas.audio.noct_spectrum"
    assert replayed.lineage.inputs == (source.lineage,)
    np.testing.assert_array_equal(
        channel_first_values(replayed),
        channel_first_values(processed),
    )

    np.testing.assert_array_equal(channel_first_values(source), source_values)
    np.testing.assert_array_equal(caller_values, caller_values_before)
    assert source.metadata == source_metadata
    assert source.channels.to_list() == source_channels
    np.testing.assert_array_equal(source.source_time_offset, source_offsets)
    assert source.operation_history == []
    assert source.lineage is source_lineage


def test_noct_spectrum_single_band_recipe_replay_preserves_shape_axes_dtype_laziness_and_values() -> None:
    source, caller_values = _source()

    with mock.patch.object(DaArray, "compute") as compute:
        processed = source.noct_spectrum(
            fmin=1_000.0,
            fmax=1_000.0,
            n=3,
            G=_G,
            fr=_FR,
        )
        plan = RecipePlan.from_frame(processed, input_names=("signal",))
        loaded = RecipePlan.from_dict(json.loads(json.dumps(plan.to_dict(), allow_nan=False)))
        replayed = loaded.apply({"signal": source})
        compute.assert_not_called()

    calibrated = caller_values * np.array([[2.0], [0.25]])
    authority, frequencies = noct_spectrum(
        sig=calibrated.T,
        fs=_SAMPLING_RATE,
        fmin=1_000.0,
        fmax=1_000.0,
        n=3,
        G=_G,
        fr=_FR,
    )
    expected = np.asarray(authority).reshape(-1, calibrated.shape[0]).T

    for result in (processed, replayed):
        assert isinstance(result, NOctFrame)
        assert isinstance(result._data, DaArray)
        assert result.previous is source
        assert result.shape == expected.shape == (2, 1)
        assert result._data.dtype == expected.dtype == np.dtype(np.float64)
        assert result._data.chunks == ((1, 1), (1,))
        assert result._xr.dims == ("channel", "band")
        np.testing.assert_array_equal(result.freqs, frequencies)
        np.testing.assert_array_equal(channel_first_values(result), expected)
