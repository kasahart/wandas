import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame


class TestChannelFrameCollection:
    def test_add_remove_channel(self):
        arr = np.arange(16).reshape(2, 8)
        cf = ChannelFrame.from_numpy(arr, sampling_rate=1000, ch_labels=["L", "R"])
        new_ch = np.ones(8)
        cf2 = cf.add_channel(new_ch, label="mono")
        assert cf2.n_channels == 3
        assert [ch.label for ch in cf2._channel_metadata] == ["L", "R", "mono"]
        cf3 = cf2.remove_channel("R")
        assert cf3.n_channels == 2
        assert [ch.label for ch in cf3._channel_metadata] == ["L", "mono"]
        cf2.add_channel(np.zeros(8), label="zero", inplace=True)
        assert cf2.n_channels == 4
        cf2.remove_channel(0, inplace=True)
        assert [ch.label for ch in cf2._channel_metadata][0] == "R"

    def test_add_channel_align(self):
        arr = np.arange(8)
        cf = ChannelFrame.from_numpy(arr, sampling_rate=1000, ch_labels=["A"])
        with pytest.raises(ValueError):
            cf.add_channel(np.arange(6), label="B", align="strict")
        cf2 = cf.add_channel(np.arange(6), label="B", align="pad")
        assert cf2._data.shape == (2, 8)
        cf3 = cf.add_channel(np.arange(10), label="C", align="truncate")
        assert cf3._data.shape == (2, 8)
        with pytest.raises(ValueError):
            cf.add_channel(np.arange(8), label="A")
        cf4 = cf.add_channel(np.arange(8), label="A", suffix_on_dup="_dup")
        assert cf4._channel_metadata[1].label == "A_dup"

    def test_add_channel_dask(self):
        import dask.array as da

        arr = np.arange(8)
        cf = ChannelFrame.from_numpy(arr, sampling_rate=1000, ch_labels=["A"])
        dask_ch = da.ones(8, chunks=8)
        cf2 = cf.add_channel(dask_ch, label="B")
        assert cf2.n_channels == 2
        assert [ch.label for ch in cf2._channel_metadata] == ["A", "B"]

    def test_add_channel_frame(self):
        arr1 = np.arange(8)
        arr2 = np.arange(8, 16)
        cf1 = ChannelFrame.from_numpy(arr1, sampling_rate=1000, ch_labels=["A"])
        cf2 = ChannelFrame.from_numpy(arr2, sampling_rate=1000, ch_labels=["B"])
        cf3 = cf1.add_channel(cf2)
        assert cf3.n_channels == 2
        assert [ch.label for ch in cf3._channel_metadata] == ["A", "B"]

    def test_add_channel_frame_label_dup(self):
        arr1 = np.arange(8)
        arr2 = np.arange(8, 16)
        cf1 = ChannelFrame.from_numpy(arr1, sampling_rate=1000, ch_labels=["A"])
        cf2 = ChannelFrame.from_numpy(arr2, sampling_rate=1000, ch_labels=["A"])
        with pytest.raises(ValueError):
            cf1.add_channel(cf2)
        cf3 = cf1.add_channel(cf2, suffix_on_dup="_dup")
        assert cf3._channel_metadata[1].label == "A_dup"

    def test_add_channel_frame_length_mismatch(self):
        arr1 = np.arange(8)
        arr2 = np.arange(6)
        cf1 = ChannelFrame.from_numpy(arr1, sampling_rate=1000, ch_labels=["A"])
        cf2 = ChannelFrame.from_numpy(arr2, sampling_rate=1000, ch_labels=["B"])
        with pytest.raises(ValueError):
            cf1.add_channel(cf2, align="strict")
        cf3 = cf1.add_channel(cf2, align="pad")
        assert cf3._data.shape == (2, 8)
        cf4 = cf1.add_channel(cf2, align="truncate")
        assert cf4._data.shape == (2, 8)

    def test_add_channel_sampling_rate_mismatch(self):
        arr1 = np.arange(8)
        arr2 = np.arange(8, 16)
        cf1 = ChannelFrame.from_numpy(arr1, sampling_rate=1000, ch_labels=["A"])
        cf2 = ChannelFrame.from_numpy(arr2, sampling_rate=2000, ch_labels=["B"])
        with pytest.raises(ValueError):
            cf1.add_channel(cf2)

    def test_add_channel_type_error(self):
        arr = np.arange(8)
        cf = ChannelFrame.from_numpy(arr, sampling_rate=1000, ch_labels=["A"])
        with pytest.raises(TypeError):
            cf.add_channel("not_array")

    def test_remove_channel_index(self):
        arr = np.arange(16).reshape(2, 8)
        cf = ChannelFrame.from_numpy(arr, sampling_rate=1000, ch_labels=["L", "R"])
        cf2 = cf.remove_channel(0)
        assert cf2.n_channels == 1
        assert cf2._channel_metadata[0].label == "R"

    def test_remove_channel_label(self):
        arr = np.arange(16).reshape(2, 8)
        cf = ChannelFrame.from_numpy(arr, sampling_rate=1000, ch_labels=["L", "R"])
        cf2 = cf.remove_channel("L")
        assert cf2.n_channels == 1
        assert cf2._channel_metadata[0].label == "R"

    def test_remove_channel_inplace(self):
        arr = np.arange(16).reshape(2, 8)
        cf = ChannelFrame.from_numpy(arr, sampling_rate=1000, ch_labels=["L", "R"])
        cf.remove_channel(1, inplace=True)
        assert cf.n_channels == 1
        assert cf._channel_metadata[0].label == "L"

    def test_remove_channel_keyerror(self):
        arr = np.arange(8)
        cf = ChannelFrame.from_numpy(arr, sampling_rate=1000, ch_labels=["A"])
        with pytest.raises(KeyError):
            cf.remove_channel("notfound")

    def test_remove_channel_indexerror(self):
        arr = np.arange(8)
        cf = ChannelFrame.from_numpy(arr, sampling_rate=1000, ch_labels=["A"])
        with pytest.raises(IndexError):
            cf.remove_channel(2)

    def test_add_channel_inplace(self):
        arr = np.arange(8)
        cf = ChannelFrame.from_numpy(arr, sampling_rate=1000, ch_labels=["A"])
        cf.add_channel(np.ones(8), label="B", inplace=True)
        assert cf.n_channels == 2
        assert [ch.label for ch in cf._channel_metadata] == ["A", "B"]
