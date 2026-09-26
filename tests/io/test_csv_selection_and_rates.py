"""Public CSV selection and fractional-rate regression contracts."""

from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest

import wandas as wd
from wandas.io.readers import CSVFileReader

_CSV = b"time,left,right\n0,10,20\n0.1,11,21\n0.2,12,22\n"


@pytest.mark.parametrize(
    "channels",
    [pytest.param([1], id="single-channel"), pytest.param([1, 0], id="reordered-channels")],
)
@pytest.mark.parametrize(
    "explicit_labels",
    [pytest.param(False, id="inferred-labels"), pytest.param(True, id="explicit-labels")],
)
def test_csv_selection_aligns_values_and_labels(channels: list[int], explicit_labels: bool) -> None:
    labels = [f"selected_{i}" for i in channels] if explicit_labels else None
    with patch.object(CSVFileReader, "get_data", wraps=CSVFileReader.get_data) as read_data:
        frame = wd.read(_CSV, file_type="csv", channel=channels, ch_labels=labels)
        read_data.assert_not_called()
        assert isinstance(frame._data, da.Array)
        assert frame.labels == (labels if labels is not None else [["left", "right"][i] for i in channels])
        np.testing.assert_array_equal(frame._compute(), np.array([[10, 11, 12], [20, 21, 22]])[channels])
        read_data.assert_called_once()


@pytest.mark.parametrize("step", [0.4, 2.0, 0.01, 1 / 48000])
def test_csv_retains_fractional_and_integer_sampling_rates(step: float) -> None:
    csv = ("time,value\n" + "\n".join(f"{i * step:.17g},{i}" for i in range(6))).encode()
    frame = wd.read(csv, file_type="csv")
    assert frame.sampling_rate == pytest.approx(1 / step)
    np.testing.assert_allclose(frame.time, np.arange(6) * step)
    np.testing.assert_array_equal(frame.data, np.arange(6))


def test_csv_fractional_rate_partial_read_retains_source_time() -> None:
    csv = b"time,value\n10,1\n10.4,2\n10.8,3\n11.2,4\n11.6,5\n12,6\n"
    frame = wd.read(csv, file_type="csv", start=0.8, end=1.6)
    assert frame.sampling_rate == pytest.approx(2.5)
    np.testing.assert_array_equal(frame.data, [3, 4])
    np.testing.assert_allclose(frame.source_time, [[10.8, 11.2]])


@pytest.mark.parametrize("rows", ["0,1", "0,1\n0,2", "1,1\n0,2", "nan,1\n1,2", "a,1\nb,2"])
def test_csv_invalid_sampling_rate_fails_before_indexing(rows: str) -> None:
    with pytest.raises(ValueError, match="sampling_rate"):
        wd.read(f"time,value\n{rows}\n".encode(), file_type="csv")
