import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import h5py
import numpy as np
import pytest

from wandas.core.base_channel import BaseChannel
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class DummyChannel(BaseChannel):
    def plot(
        self, ax: Optional["Axes"] = None, title: Optional[str] = None
    ) -> tuple["Axes", NDArrayReal]:
        # Minimal implementation for abstract method
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        return (ax, self.data)


def test_init_success(tmp_path: Path) -> None:
    data: NDArrayReal = np.array([1, 2, 3, 4, 5])
    sampling_rate: int = 1000
    label: str = "Test Channel"
    unit: str = "V"
    metadata: dict[str, Any] = {"info": "test"}

    channel: DummyChannel = DummyChannel(
        data=data,
        sampling_rate=sampling_rate,
        label=label,
        unit=unit,
        metadata=metadata,
    )

    # Verify data computed correctly
    np.testing.assert_array_equal(channel.data, data)
    assert channel.sampling_rate == sampling_rate
    assert channel.label == label
    assert channel.unit == unit
    assert channel.metadata == metadata
    # previous should be None by default
    assert channel.get_previous() is None
    # For non-memmap data, _data_path should be set and
    # the file exists during initialization
    assert channel._data_path is not None
    assert os.path.exists(channel._data_path)
    # Cleanup
    channel.close()
    # After closing, file should be removed
    if channel._data_path:
        assert not os.path.exists(channel._data_path)


def test_init_with_memmap(tmp_path: Path) -> None:
    # Create a temporary file with numpy memmap
    temp_file: Path = tmp_path / "temp.dat"
    data: NDArrayReal = np.array([10, 20, 30, 40])
    temp_file.write_bytes(np.zeros(data.nbytes, dtype=np.uint8).tobytes())
    mem = np.memmap(str(temp_file), dtype=data.dtype, mode="r+", shape=data.shape)
    mem[:] = data[:]

    sampling_rate: int = 500
    channel: DummyChannel = DummyChannel(data=mem, sampling_rate=sampling_rate)

    np.testing.assert_array_equal(channel.data, data)
    # For memmap input, _data_path should be None
    assert channel._data_path is None
    channel.close()
    mem._mmap.close()  # type: ignore[unused-ignore]


def test_os_unlink_actual_deletion(monkeypatch: pytest.MonkeyPatch) -> None:
    data = np.array([10, 20, 30, 40])

    # 一時ファイルの名前を記録するため、tempfile.NamedTemporaryFile をパッチ
    captured = {}

    original_named_tempfile = tempfile.NamedTemporaryFile

    def fake_named_tempfile(*args: Any, **kwargs: Any) -> Any:
        temp = original_named_tempfile(*args, **kwargs)
        captured["name"] = temp.name
        return temp

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fake_named_tempfile)

    # h5py.File をパッチして、create_dataset 呼び出し時に例外を強制的に発生させる
    class FakeH5pyFile:
        def __init__(self, filename: str, mode: str) -> None:
            self.filename = filename

        def create_dataset(self, *args: Any, **kwargs: Any) -> None:
            raise Exception("Forced Error in create_dataset")

        def close(self) -> None:
            pass

    def create_fake_h5py_file(filename: str, mode: str) -> FakeH5pyFile:
        return FakeH5pyFile(filename, mode)

    monkeypatch.setattr(h5py, "File", create_fake_h5py_file)
    with pytest.raises(RuntimeError, match="HDF5 file creation failed"):
        _ = DummyChannel(data=data, sampling_rate=100)
    # BaseChannel の生成時に例外が発生することを検証
    with pytest.raises(RuntimeError, match="HDF5 file creation failed"):
        _ = DummyChannel(data=data, sampling_rate=100)

    # 例外後、一時ファイルが削除されていることを確認
    file_name = captured.get("name")
    assert file_name is not None, "Temporary file name was not captured"
    assert not os.path.exists(file_name), f"Temporary file {file_name} still exists"
