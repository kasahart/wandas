"""Recipe compatibility for the versioned Butterworth filter contracts."""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy import signal

from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan
from wandas.processing.filters import BandPassFilter, HighPassFilter, LowPassFilter

_SAMPLING_RATE = 8_000.0
_FILTER_CASES = [
    pytest.param(
        "wandas.audio.highpass_filter",
        {"cutoff": 500, "order": 4},
        ([500.0 / 4_000.0], "high"),
        HighPassFilter,
        id="highpass",
    ),
    pytest.param(
        "wandas.audio.lowpass_filter",
        {"cutoff": 500, "order": 4},
        ([500.0 / 4_000.0], "low"),
        LowPassFilter,
        id="lowpass",
    ),
    pytest.param(
        "wandas.audio.bandpass_filter",
        {"low_cutoff": 300, "high_cutoff": 900, "order": 3},
        ([300.0 / 4_000.0, 900.0 / 4_000.0], "band"),
        BandPassFilter,
        id="bandpass",
    ),
]


def _source() -> ChannelFrame:
    values = np.random.default_rng(3).standard_normal((2, 256))
    return ChannelFrame(da.from_array(values, chunks=(1, -1)), sampling_rate=_SAMPLING_RATE)


def _payload(operation: str, version: int, params: dict[str, Any]) -> dict[str, Any]:
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
                "params": {"$type": "map", "entries": [[key, value] for key, value in sorted(params.items())]},
            }
        ],
        "output": "node-0",
    }


@pytest.mark.parametrize(("operation", "params", "design", "current_class"), _FILTER_CASES)
def test_released_v1_payload_replays_transfer_function_filtfilt(
    operation: str,
    params: dict[str, Any],
    design: tuple[list[float], str],
    current_class: type[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v1 payload keeps the released (b, a) + filtfilt numerics, never the SOS kernel."""

    def fail_v2(_self: Any, _data: Any) -> Any:
        raise AssertionError("released v1 payload invoked the v2 numerical kernel")

    monkeypatch.setattr(current_class, "_process", fail_v2)
    source = _source()

    replayed = RecipePlan.from_dict(_payload(operation, 1, params)).apply({"signal": source})

    cutoff, btype = design
    b, a = signal.butter(params["order"], cutoff if len(cutoff) > 1 else cutoff[0], btype=btype)
    expected = signal.filtfilt(b, a, channel_first_values(source), axis=1)
    assert isinstance(replayed._data, DaArray)
    np.testing.assert_array_equal(channel_first_values(replayed), expected)
    assert replayed.operation_history == [{"operation": operation, "version": 1, "params": params}]


@pytest.mark.parametrize(("operation", "params", "design", "current_class"), _FILTER_CASES)
def test_current_calls_record_v2_and_replay_second_order_sections(
    operation: str,
    params: dict[str, Any],
    design: tuple[list[float], str],
    current_class: type[Any],
) -> None:
    source = _source()

    replayed = RecipePlan.from_dict(_payload(operation, 2, params)).apply({"signal": source})

    cutoff, btype = design
    sos = signal.butter(params["order"], cutoff if len(cutoff) > 1 else cutoff[0], btype=btype, output="sos")
    expected = signal.sosfiltfilt(sos, channel_first_values(source), axis=1)
    np.testing.assert_allclose(channel_first_values(replayed), expected, rtol=1e-12, atol=1e-12)
    assert replayed.operation_history == [{"operation": operation, "version": 2, "params": params}]
