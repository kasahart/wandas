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

    def test_add_channel_numpy_preserves_all_metadata(self):
        """Test comprehensive metadata preservation when adding numpy arrays"""
        arr = np.arange(8)
        original_metadata = {
            "source": "recording",
            "format": "wav",
            "bitrate": 192,
        }

        cf = ChannelFrame.from_numpy(
            arr,
            sampling_rate=1000,
            ch_labels=["original"],
            ch_units="Pa",
            metadata=original_metadata,
        )
        cf._channel_metadata[0].ref = 20e-6
        cf._channel_metadata[0].extra["mic_type"] = "condenser"
        cf._channel_metadata[0].extra["polar_pattern"] = "cardioid"

        # numpy配列を3回追加
        cf2 = cf.add_channel(np.ones(8), label="added1")
        cf3 = cf2.add_channel(np.ones(8) * 2, label="added2")
        cf4 = cf3.add_channel(np.ones(8) * 3, label="added3")

        # 元のフレームのmetadataが保持されていることを確認
        assert cf4.metadata == original_metadata
        assert cf4.metadata["source"] == "recording"
        assert cf4.metadata["format"] == "wav"
        assert cf4.metadata["bitrate"] == 192

        # 元のチャンネルのChannelMetadataがすべて保持されていることを確認
        assert cf4._channel_metadata[0].label == "original"
        assert cf4._channel_metadata[0].unit == "Pa"
        assert cf4._channel_metadata[0].ref == 20e-6
        assert cf4._channel_metadata[0].extra["mic_type"] == "condenser"
        assert cf4._channel_metadata[0].extra["polar_pattern"] == "cardioid"

        # 追加したチャンネルは新しいChannelMetadataを持つ
        assert cf4.n_channels == 4
        assert cf4._channel_metadata[1].label == "added1"
        assert cf4._channel_metadata[2].label == "added2"
        assert cf4._channel_metadata[3].label == "added3"

    def test_add_channel_frame_metadata_independence(self):
        """Test metadata independence when adding ChannelFrame instances"""
        arr1 = np.arange(8)
        arr2 = np.arange(8, 16)

        # 元のフレームのmetadata
        metadata1 = {
            "experiment": "test1",
            "date": "2025-01-01",
            "operator": "Alice",
        }

        # 追加するフレームのmetadata
        metadata2 = {
            "experiment": "test2",
            "date": "2025-01-02",
            "operator": "Bob",
        }

        cf1 = ChannelFrame.from_numpy(
            arr1, sampling_rate=1000, ch_labels=["A"], metadata=metadata1
        )
        cf2 = ChannelFrame.from_numpy(
            arr2, sampling_rate=1000, ch_labels=["B"], metadata=metadata2
        )

        # add_channelでChannelFrameを追加
        cf3 = cf1.add_channel(cf2)

        # 元のフレーム(cf1)のmetadataのみが保持されることを確認
        assert cf3.metadata == metadata1
        assert cf3.metadata["experiment"] == "test1"
        assert cf3.metadata["date"] == "2025-01-01"
        assert cf3.metadata["operator"] == "Alice"

        # cf2のmetadataは引き継がれていないことを確認
        assert cf3.metadata["experiment"] != "test2"
        assert cf3.metadata["date"] != "2025-01-02"
        assert cf3.metadata["operator"] != "Bob"
        assert "other" not in cf3.metadata  # 存在しないキーの確認

    def test_add_channel_frame_comprehensive_channel_metadata(self):
        """Test comprehensive preservation of all ChannelMetadata properties"""
        arr1 = np.arange(8)
        arr2 = np.arange(8, 16)
        arr3 = np.arange(16, 24)

        # 元のフレーム: すべてのChannelMetadataプロパティを設定
        cf1 = ChannelFrame.from_numpy(
            arr1, sampling_rate=1000, ch_labels=["original"], ch_units="Pa"
        )
        cf1._channel_metadata[0].ref = 20e-6  # 手動でref設定
        cf1._channel_metadata[0].extra["sensor_id"] = "SN001"
        cf1._channel_metadata[0].extra["location"] = "front"
        cf1._channel_metadata[0].extra["calibration_factor"] = 1.02

        # 追加するフレーム1: 異なる設定
        cf2 = ChannelFrame.from_numpy(
            arr2, sampling_rate=1000, ch_labels=["added1"], ch_units="V"
        )
        cf2._channel_metadata[0].ref = 1.0  # V基準
        cf2._channel_metadata[0].extra["sensor_id"] = "SN002"
        cf2._channel_metadata[0].extra["location"] = "back"
        cf2._channel_metadata[0].extra["gain"] = 2.5

        # 追加するフレーム2: さらに異なる設定
        cf3 = ChannelFrame.from_numpy(
            arr3, sampling_rate=1000, ch_labels=["added2"], ch_units="m/s^2"
        )
        cf3._channel_metadata[0].ref = 1e-6  # 加速度基準
        cf3._channel_metadata[0].extra["sensor_id"] = "SN003"
        cf3._channel_metadata[0].extra["location"] = "top"
        cf3._channel_metadata[0].extra["sensitivity"] = 100.0

        # 2つのChannelFrameを順次追加
        cf_combined = cf1.add_channel(cf2).add_channel(cf3)

        # すべてのチャンネルが存在することを確認
        assert cf_combined.n_channels == 3

        # チャンネル0 (元のフレーム) のすべてのプロパティを確認
        assert cf_combined._channel_metadata[0].label == "original"
        assert cf_combined._channel_metadata[0].unit == "Pa"
        assert cf_combined._channel_metadata[0].ref == 20e-6
        assert cf_combined._channel_metadata[0].extra["sensor_id"] == "SN001"
        assert cf_combined._channel_metadata[0].extra["location"] == "front"
        assert cf_combined._channel_metadata[0].extra["calibration_factor"] == 1.02

        # チャンネル1 (追加したフレーム1) のすべてのプロパティを確認
        assert cf_combined._channel_metadata[1].label == "added1"
        assert cf_combined._channel_metadata[1].unit == "V"
        assert cf_combined._channel_metadata[1].ref == 1.0
        assert cf_combined._channel_metadata[1].extra["sensor_id"] == "SN002"
        assert cf_combined._channel_metadata[1].extra["location"] == "back"
        assert cf_combined._channel_metadata[1].extra["gain"] == 2.5

        # チャンネル2 (追加したフレーム2) のすべてのプロパティを確認
        assert cf_combined._channel_metadata[2].label == "added2"
        assert cf_combined._channel_metadata[2].unit == "m/s^2"
        assert cf_combined._channel_metadata[2].ref == 1e-6
        assert cf_combined._channel_metadata[2].extra["sensor_id"] == "SN003"
        assert cf_combined._channel_metadata[2].extra["location"] == "top"
        assert cf_combined._channel_metadata[2].extra["sensitivity"] == 100.0

    def test_add_channel_empty_extra_dict_preserved(self):
        """Test that empty extra dict is properly preserved"""
        arr1 = np.arange(8)
        arr2 = np.arange(8, 16)

        cf1 = ChannelFrame.from_numpy(
            arr1, sampling_rate=1000, ch_labels=["A"], ch_units="Pa"
        )
        cf2 = ChannelFrame.from_numpy(
            arr2, sampling_rate=1000, ch_labels=["B"], ch_units="V"
        )

        cf3 = cf1.add_channel(cf2)

        # 両方のチャンネルのextraが空のdictであることを確認
        assert cf3._channel_metadata[0].extra == {}
        assert cf3._channel_metadata[1].extra == {}
        assert isinstance(cf3._channel_metadata[0].extra, dict)
        assert isinstance(cf3._channel_metadata[1].extra, dict)

    def test_add_channel_inplace_preserves_all_metadata(self):
        """Test comprehensive metadata preservation with inplace operations"""
        arr = np.arange(8)
        arr2 = np.arange(8, 16)
        original_metadata = {"key1": "value1", "key2": "value2"}

        # numpy配列のinplace追加
        cf_numpy = ChannelFrame.from_numpy(
            arr,
            sampling_rate=1000,
            ch_labels=["ch0"],
            ch_units="Pa",
            metadata=original_metadata.copy(),
        )
        cf_numpy._channel_metadata[0].ref = 20e-6
        cf_numpy._channel_metadata[0].extra["custom"] = "data"

        cf_numpy.add_channel(np.ones(8), label="ch1", inplace=True)

        # metadataが保持されていることを確認
        assert cf_numpy.metadata == original_metadata
        assert cf_numpy._channel_metadata[0].label == "ch0"
        assert cf_numpy._channel_metadata[0].unit == "Pa"
        assert cf_numpy._channel_metadata[0].ref == 20e-6
        assert cf_numpy._channel_metadata[0].extra["custom"] == "data"
        assert cf_numpy.n_channels == 2
        assert cf_numpy._channel_metadata[1].label == "ch1"

        # ChannelFrameのinplace追加
        cf1 = ChannelFrame.from_numpy(
            arr,
            sampling_rate=1000,
            ch_labels=["original"],
            ch_units="Pa",
            metadata={"test": "data"},
        )
        cf1._channel_metadata[0].ref = 20e-6
        cf1._channel_metadata[0].extra["info"] = "test"

        cf2 = ChannelFrame.from_numpy(
            arr2, sampling_rate=1000, ch_labels=["added"], ch_units="V"
        )
        cf2._channel_metadata[0].ref = 1.0
        cf2._channel_metadata[0].extra["other"] = "info"

        cf1.add_channel(cf2, inplace=True)

        # 両方のチャンネルのChannelMetadataが保持されていることを確認
        assert cf1.metadata == {"test": "data"}
        assert cf1.n_channels == 2
        assert cf1._channel_metadata[0].label == "original"
        assert cf1._channel_metadata[0].unit == "Pa"
        assert cf1._channel_metadata[0].ref == 20e-6
        assert cf1._channel_metadata[0].extra["info"] == "test"
        assert cf1._channel_metadata[1].label == "added"
        assert cf1._channel_metadata[1].unit == "V"
        assert cf1._channel_metadata[1].ref == 1.0
        assert cf1._channel_metadata[1].extra["other"] == "info"
