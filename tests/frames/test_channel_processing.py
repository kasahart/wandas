from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame, ChannelMetadata

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestChannelProcessing:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        # Create a simple dask array for testing
        self.sample_rate: float = 16000
        self.data: np.ndarray = np.random.random((2, 16000))  # 2 channels, 1 second
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, 4000))
        self.channel_frame: ChannelFrame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_high_pass_filter(self) -> None:
        """Test high_pass_filter operation."""
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op: mock.MagicMock = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op

            # Apply filter operations
            result: ChannelFrame = self.channel_frame.high_pass_filter(cutoff=100)
            mock_create_op.assert_called_with(
                "highpass_filter", self.sample_rate, cutoff=100, order=4
            )

            # No compute should have happened
            assert isinstance(result, ChannelFrame)
            assert isinstance(result._data, DaArray)

    def test_low_pass_filter(self) -> None:
        """Test low_pass_filter operation."""
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op: mock.MagicMock = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op

            # Apply filter operations
            result: ChannelFrame = self.channel_frame.low_pass_filter(cutoff=5000)
            mock_create_op.assert_called_with(
                "lowpass_filter", self.sample_rate, cutoff=5000, order=4
            )

            # No compute should have happened
            assert isinstance(result, ChannelFrame)
            assert isinstance(result._data, DaArray)

    def test_band_pass_filter(self) -> None:
        """Test band_pass_filter operation."""
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op: mock.MagicMock = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op

            # Apply band-pass filter operation
            result: ChannelFrame = self.channel_frame.band_pass_filter(
                low_cutoff=200, high_cutoff=5000
            )
            mock_create_op.assert_called_with(
                "bandpass_filter",
                self.sample_rate,
                low_cutoff=200,
                high_cutoff=5000,
                order=4,
            )

            # Test with custom order
            result = self.channel_frame.band_pass_filter(
                low_cutoff=300, high_cutoff=3000, order=6
            )
            mock_create_op.assert_called_with(
                "bandpass_filter",
                self.sample_rate,
                low_cutoff=300,
                high_cutoff=3000,
                order=6,
            )

            # No compute should have happened
            assert isinstance(result, ChannelFrame)
            assert isinstance(result._data, DaArray)

    def test_a_weighting(self) -> None:
        """Test a_weighting operation."""
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op

            # Test a_weighting
            result = self.channel_frame.a_weighting()
            mock_create_op.assert_called_with("a_weighting", self.sample_rate)
            assert isinstance(result, ChannelFrame)

    def test_abs(self) -> None:
        """Test abs method."""
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op

            result = self.channel_frame.abs()
            mock_create_op.assert_called_with("abs", self.sample_rate)
            assert isinstance(result, ChannelFrame)

    def test_power(self) -> None:
        """Test power method."""
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op

            result = self.channel_frame.power(exponent=2.0)
            mock_create_op.assert_called_with("power", self.sample_rate, exponent=2.0)
            assert isinstance(result, ChannelFrame)

    def test_sum_methods(self) -> None:
        """Test sum() methods."""
        # Test that sum method is lazy
        with mock.patch.object(DaArray, "compute") as mock_compute:
            # Call sum() - this should be lazy and not trigger computation
            sum_cf = self.channel_frame.sum()

            # Check no computation happened yet
            mock_compute.assert_not_called()

            # Verify result is the expected type
            assert isinstance(sum_cf, ChannelFrame)
            assert sum_cf.n_channels == 1

        # Test correctness of computation result
        sum_cf = self.channel_frame.sum()
        sum_data = sum_cf.compute()
        expected_sum = self.data.sum(axis=-2, keepdims=True)
        np.testing.assert_array_almost_equal(sum_data, expected_sum)

    def test_mean_methods(self) -> None:
        """Test mean() methods."""
        # Test mean method
        with mock.patch.object(DaArray, "compute") as mock_compute:
            # Call mean() - this should be lazy and not trigger computation
            mean_cf = self.channel_frame.mean()

            # Check no computation happened yet
            mock_compute.assert_not_called()

            # Verify result is the expected type
            assert isinstance(mean_cf, ChannelFrame)
            assert mean_cf.n_channels == 1

        # Compute and check results
        mean_cf = self.channel_frame.mean()
        mean_data = mean_cf.compute()
        expected_mean = self.data.mean(axis=-2, keepdims=True)
        np.testing.assert_array_almost_equal(mean_data, expected_mean)

    def test_channel_difference(self) -> None:
        """Test channel_difference method."""
        # Test that channel_difference is lazy
        with mock.patch.object(DaArray, "compute") as mock_compute:
            # Call channel_difference - this should be lazy and not trigger computation
            diff_cf = self.channel_frame.channel_difference(other_channel=0)

            # Check no computation happened yet
            mock_compute.assert_not_called()

            # Verify result is the expected type
            assert isinstance(diff_cf, ChannelFrame)
            assert diff_cf.n_channels == self.channel_frame.n_channels

        # Test correctness of computation result
        diff_cf = self.channel_frame.channel_difference(other_channel=0)
        computed = diff_cf.compute()
        expected = self.data - self.data[0:1]
        np.testing.assert_array_almost_equal(computed, expected)

        # Test that channel_difference with other_channel=0 works correctly
        diff_cf = self.channel_frame.channel_difference(other_channel="ch0")
        computed = diff_cf.compute()
        expected = self.data - self.data[0:1]
        np.testing.assert_array_almost_equal(computed, expected)

        # Test invalid channel index
        with pytest.raises(IndexError):
            self.channel_frame.channel_difference(other_channel=10)

    def test_trim(self) -> None:
        """Test the trim method."""
        # Test trimming with start and end times
        trimmed_frame = self.channel_frame.trim(start=0.1, end=0.5)
        assert isinstance(trimmed_frame, ChannelFrame)
        assert trimmed_frame.n_samples == int(0.4 * self.sample_rate)
        assert trimmed_frame.n_channels == self.channel_frame.n_channels

        # Test trimming with only start time
        trimmed_frame = self.channel_frame.trim(start=0.2)
        assert isinstance(trimmed_frame, ChannelFrame)
        assert trimmed_frame.n_samples == int(0.8 * self.sample_rate)

        # Test trimming with only end time
        trimmed_frame = self.channel_frame.trim(end=0.3)
        assert isinstance(trimmed_frame, ChannelFrame)
        assert trimmed_frame.n_samples == int(0.3 * self.sample_rate)

        # Test trimming with no start or end (should return the same frame)
        trimmed_frame = self.channel_frame.trim()
        assert isinstance(trimmed_frame, ChannelFrame)
        assert trimmed_frame.n_samples == self.channel_frame.n_samples

        # Test trimming with invalid start and end times
        with pytest.raises(ValueError):
            self.channel_frame.trim(start=0.5, end=0.1)

    def test_hpss_operations(self) -> None:
        """Test HPSS (Harmonic-Percussive Source Separation) methods."""
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op

            # Test HPSS methods
            result = self.channel_frame.hpss_harmonic(kernel_size=31)
            mock_create_op.assert_called_with(
                "hpss_harmonic",
                self.sample_rate,
                kernel_size=31,
                power=2,
                margin=1,
                n_fft=2048,
                hop_length=None,
                win_length=None,
                window="hann",
                center=True,
                pad_mode="constant",
            )
            assert isinstance(result, ChannelFrame)

            result = self.channel_frame.hpss_percussive(kernel_size=31)
            mock_create_op.assert_called_with(
                "hpss_percussive",
                self.sample_rate,
                kernel_size=31,
                power=2,
                margin=1,
                n_fft=2048,
                hop_length=None,
                win_length=None,
                window="hann",
                center=True,
                pad_mode="constant",
            )
            assert isinstance(result, ChannelFrame)

    def test_add_with_snr(self) -> None:
        """Test add method with SNR parameter."""
        # 別のChannelFrameを作成
        signal_data = np.random.random((2, 16000))
        signal_dask_data = _da_from_array(signal_data, chunks=-1)
        signal_cf = ChannelFrame(signal_dask_data, self.sample_rate, label="signal")

        # ノイズデータを作成
        noise_data = np.random.random((2, 16000)) * 0.1  # 小さいノイズ
        noise_dask_data = _da_from_array(noise_data, chunks=-1)
        noise_cf = ChannelFrame(noise_dask_data, self.sample_rate, label="noise")

        # SNRを指定して加算
        snr_value = 10.0  # 10dBのSNR
        result = signal_cf.add(noise_cf, snr=snr_value)

        # 基本的なプロパティをチェック
        assert isinstance(result, ChannelFrame)
        assert result.sampling_rate == self.sample_rate
        assert result.n_channels == 2
        assert result.n_samples == 16000

        # 演算履歴の確認 - 実装に合わせて調整
        # この部分はapply_addの実装によって異なる可能性があるため、
        # 一般的な作成チェックのみを行う
        assert len(result.operation_history) > len(signal_cf.operation_history)

        # 実際の計算をトリガー
        computed = result.compute()

        # SNRを考慮した加算の結果を確認
        # 実際の結果はSNRの具体的な実装によって異なりますが、型と形状は確認可能
        assert isinstance(computed, np.ndarray)
        assert computed.shape == (2, 16000)

        # 負のSNR値もテスト
        # 値が適用されることを確認する
        neg_result = signal_cf.add(noise_cf, snr=-10.0)
        neg_computed = neg_result.compute()
        assert isinstance(neg_computed, np.ndarray)
        assert neg_computed.shape == (2, 16000)

    def test_add_with_different_lengths(self) -> None:
        """異なる長さの信号を加算するテスト。"""
        # 標準の長さのフレーム（self.channel_frame）
        # 長さが標準フレームよりも短いフレーム（切り詰め必要）
        short_data = np.random.random((2, 8000))  # 半分の長さ
        short_dask_data = _da_from_array(short_data, chunks=(1, 2000))
        short_cf = ChannelFrame(short_dask_data, self.sample_rate, label="short_audio")

        # 長さが標準フレームよりも長いフレーム（パディング必要）
        long_data = np.random.random((2, 24000))  # 1.5倍の長さ
        long_dask_data = _da_from_array(long_data, chunks=(1, 6000))
        long_cf = ChannelFrame(long_dask_data, self.sample_rate, label="long_audio")

        # 短いフレームを標準フレームに加算（パディングが必要）
        result_short = self.channel_frame.add(short_cf)
        computed_short = result_short.compute()

        # 結果の形状が元のフレームと同じであることを確認
        assert computed_short.shape == self.data.shape

        # 短いフレーム部分は加算され、残りは元のフレームのままであることを確認
        expected_short = self.data.copy()
        expected_short[:, : short_data.shape[1]] = (
            expected_short[:, : short_data.shape[1]] + short_data
        )
        np.testing.assert_array_almost_equal(computed_short, expected_short)

        # 長いフレームを標準フレームに加算（切り詰めが必要）
        result_long = self.channel_frame.add(long_cf)
        computed_long = result_long.compute()

        # 結果の形状が元のフレームと同じであることを確認
        assert computed_long.shape == self.data.shape

        # 元のフレームと同じ長さだけ長いフレームを切り詰めて加算されることを確認
        expected_long = self.data + long_data[:, : self.data.shape[1]]
        np.testing.assert_array_almost_equal(computed_long, expected_long)

    def test_rms_trend(self) -> None:
        """Test rms_trend operation."""
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op

            # 通常呼び出し（1行が長くならないように分割）
            result = self.channel_frame.rms_trend(
                frame_length=1024, hop_length=256, dB=True, Aw=True
            )
            mock_create_op.assert_called_with(
                "rms_trend",
                self.sample_rate,
                frame_length=1024,
                hop_length=256,
                ref=[1, 1],
                dB=True,
                Aw=True,
            )
            assert isinstance(result, ChannelFrame)

            # _channel_metadata から ref を取得するケース
            frame = self.channel_frame
            frame._channel_metadata = [
                ChannelMetadata(label="ch0", unit="", ref=0.5, extra={}),
                ChannelMetadata(label="ch1", unit="", ref=1.0, extra={}),
            ]
            result2 = frame.rms_trend()
            mock_create_op.assert_called_with(
                "rms_trend",
                self.sample_rate,
                frame_length=2048,
                hop_length=512,
                ref=[0.5, 1.0],
                dB=False,
                Aw=False,
            )
            assert isinstance(result2, ChannelFrame)

    def test_rms_trend_channel_frame_attributes(self) -> None:
        """rms_trend後のChannelFrame属性を確認するテスト"""
        # 事前に属性をセット
        self.channel_frame.label = "test_label"
        self.channel_frame.metadata = {"foo": "bar"}
        self.channel_frame._channel_metadata = [
            ChannelMetadata(label="test_ch0", unit="", ref=1.0, extra={"foo": 123}),
            ChannelMetadata(label="test_ch1", unit="Pa", extra={"bar": "baz"}),
        ]

        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            # Mock get_metadata_updates to return updated sampling rate
            hop_length = 256
            mock_op.get_metadata_updates.return_value = {
                "sampling_rate": self.sample_rate / hop_length
            }
            mock_create_op.return_value = mock_op

            result = self.channel_frame.rms_trend(frame_length=1024, hop_length=256)
            self._check_channel_frame_attrs(
                result, self.channel_frame, hop_length=256, op_key="rms_trend"
            )

    def _check_channel_frame_attrs(self, result, base, hop_length=None, op_key=None):
        expected_sr = (
            base.sampling_rate / hop_length if hop_length else base.sampling_rate
        )
        assert result.sampling_rate == expected_sr
        assert result.label == base.label
        # metadata: baseの内容が含まれていること、新しい操作分のキーが追加されていること
        for k, v in base.metadata.items():
            assert k in result.metadata
            assert result.metadata[k] == v
        if op_key is not None:
            assert op_key in result.metadata
        # _channel_metadata: unit="Pa"のrefが期待値通りか確認
        if hasattr(result, "_channel_metadata") and hasattr(base, "_channel_metadata"):
            for res_meta, base_meta in zip(
                result._channel_metadata, base._channel_metadata
            ):
                if res_meta.unit == "Pa":
                    assert res_meta.ref == 2e-5, (
                        f"unit='Pa'のrefが一致しません: "
                        f"{res_meta.ref} != {base_meta.ref}"
                    )
                # Check that labels are updated (not the same as base)
                # Labels should now contain the operation name
                if op_key:
                    assert base_meta.label in res_meta.label, (
                        f"Expected base label '{base_meta.label}' to be in "
                        f"result label '{res_meta.label}'"
                    )
        # Note: We no longer check for exact equality of _channel_metadata
        # because labels are now updated to reflect operations
        assert len(result.operation_history) == len(base.operation_history) + 1
        assert result.previous is base

    def test_high_pass_filter_channel_frame_attributes(self) -> None:
        self.channel_frame.label = "test_label"
        self.channel_frame.metadata = {"foo": "bar"}
        self.channel_frame._channel_metadata = [
            ChannelMetadata(label="test_ch0", unit="", ref=1.0, extra={"foo": 123}),
            ChannelMetadata(label="test_ch1", unit="Pa", extra={"bar": "baz"}),
        ]
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op
            result = self.channel_frame.high_pass_filter(cutoff=100)
            self._check_channel_frame_attrs(
                result, self.channel_frame, op_key="highpass_filter"
            )

    def test_low_pass_filter_channel_frame_attributes(self) -> None:
        self.channel_frame.label = "test_label"
        self.channel_frame.metadata = {"foo": "bar"}
        self.channel_frame._channel_metadata = [
            ChannelMetadata(label="test_ch0", unit="", ref=1.0, extra={"foo": 123}),
            ChannelMetadata(label="test_ch1", unit="Pa", extra={"bar": "baz"}),
        ]
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op
            result = self.channel_frame.low_pass_filter(cutoff=5000)
            self._check_channel_frame_attrs(
                result, self.channel_frame, op_key="lowpass_filter"
            )

    def test_band_pass_filter_channel_frame_attributes(self) -> None:
        self.channel_frame.label = "test_label"
        self.channel_frame.metadata = {"foo": "bar"}
        self.channel_frame._channel_metadata = [
            ChannelMetadata(label="test_ch0", unit="", ref=1.0, extra={"foo": 123}),
            ChannelMetadata(label="test_ch1", unit="Pa", extra={"bar": "baz"}),
        ]
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op
            result = self.channel_frame.band_pass_filter(
                low_cutoff=200, high_cutoff=5000
            )
            self._check_channel_frame_attrs(
                result, self.channel_frame, op_key="bandpass_filter"
            )

    def test_a_weighting_channel_frame_attributes(self) -> None:
        self.channel_frame.label = "test_label"
        self.channel_frame.metadata = {"foo": "bar"}
        self.channel_frame._channel_metadata = [
            ChannelMetadata(label="test_ch0", unit="", ref=1.0, extra={"foo": 123}),
            ChannelMetadata(label="test_ch1", unit="Pa", extra={"bar": "baz"}),
        ]
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op
            result = self.channel_frame.a_weighting()
            self._check_channel_frame_attrs(
                result, self.channel_frame, op_key="a_weighting"
            )

    def test_abs_channel_frame_attributes(self) -> None:
        self.channel_frame.label = "test_label"
        self.channel_frame.metadata = {"foo": "bar"}
        self.channel_frame._channel_metadata = [
            ChannelMetadata(label="test_ch0", unit="", ref=1.0, extra={"foo": 123}),
            ChannelMetadata(label="test_ch1", unit="Pa", extra={"bar": "baz"}),
        ]
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op
            result = self.channel_frame.abs()
            self._check_channel_frame_attrs(result, self.channel_frame, op_key="abs")

    def test_power_channel_frame_attributes(self) -> None:
        self.channel_frame.label = "test_label"
        self.channel_frame.metadata = {"foo": "bar"}
        self.channel_frame._channel_metadata = [
            ChannelMetadata(label="test_ch0", unit="", ref=1.0, extra={"foo": 123}),
            ChannelMetadata(label="test_ch1", unit="Pa", extra={"bar": "baz"}),
        ]
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op
            result = self.channel_frame.power(exponent=2.0)
            self._check_channel_frame_attrs(result, self.channel_frame, op_key="power")

    def test_trim_channel_frame_attributes(self) -> None:
        self.channel_frame.label = "test_label"
        self.channel_frame.metadata = {"foo": "bar"}
        self.channel_frame._channel_metadata = [
            ChannelMetadata(label="test_ch0", unit="", ref=1.0, extra={"foo": 123}),
            ChannelMetadata(label="test_ch1", unit="Pa", extra={"bar": "baz"}),
        ]
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op
            result = self.channel_frame.trim(start=0.1, end=0.5)
            self._check_channel_frame_attrs(result, self.channel_frame, op_key="trim")

    def test_fix_length_channel_frame_attributes(self) -> None:
        self.channel_frame.label = "test_label"
        self.channel_frame.metadata = {"foo": "bar"}
        self.channel_frame._channel_metadata = [
            ChannelMetadata(label="test_ch0", unit="", ref=1.0, extra={"foo": 123}),
            ChannelMetadata(label="test_ch1", unit="Pa", extra={"bar": "baz"}),
        ]
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op
            result = self.channel_frame.fix_length(length=10000)
            self._check_channel_frame_attrs(
                result, self.channel_frame, op_key="fix_length"
            )

    def test_resampling_channel_frame_attributes(self) -> None:
        self.channel_frame.label = "test_label"
        self.channel_frame.metadata = {"foo": "bar"}
        self.channel_frame._channel_metadata = [
            ChannelMetadata(label="test_ch0", unit="", ref=1.0, extra={"foo": 123}),
            ChannelMetadata(label="test_ch1", unit="Pa", extra={"bar": "baz"}),
        ]
        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            # Mock get_metadata_updates to return target sampling rate
            target_sr = 8000
            mock_op.get_metadata_updates.return_value = {"sampling_rate": target_sr}
            mock_create_op.return_value = mock_op
            result = self.channel_frame.resampling(target_sr=target_sr)
            self._check_channel_frame_attrs(
                result, self.channel_frame, hop_length=2, op_key="resampling"
            )


class TestSamplingRateUpdates:
    """Integration tests for sampling rate updates via metadata."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 44100
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        signal = np.array([0.1 * np.sin(2 * np.pi * 440 * t)])
        self.dask_data = _da_from_array(signal, chunks=-1)
        self.frame = ChannelFrame(data=self.dask_data, sampling_rate=self.sample_rate)

    def test_loudness_zwtv_updates_sampling_rate(self) -> None:
        """Test that loudness_zwtv correctly updates sampling rate."""
        loudness = self.frame.loudness_zwtv(field_type="free")

        # Sampling rate should be updated to 500 Hz (2ms time steps)
        assert loudness.sampling_rate == 500.0
        assert loudness.sampling_rate != self.sample_rate

    def test_rms_trend_updates_sampling_rate(self) -> None:
        """Test that rms_trend correctly updates sampling rate."""
        hop_length = 512
        rms = self.frame.rms_trend(hop_length=hop_length)

        # Sampling rate should be updated based on hop_length
        expected_sr = self.sample_rate / hop_length
        assert np.isclose(rms.sampling_rate, expected_sr)
        assert rms.sampling_rate != self.sample_rate

    def test_resampling_updates_sampling_rate(self) -> None:
        """Test that resampling correctly updates sampling rate."""
        target_sr = 16000
        resampled = self.frame.resampling(target_sr=target_sr)

        # Sampling rate should be updated to target_sr
        assert resampled.sampling_rate == target_sr
        assert resampled.sampling_rate != self.sample_rate

    def test_operations_without_metadata_updates_preserve_sampling_rate(
        self,
    ) -> None:
        """Test that operations without metadata updates preserve sampling rate."""
        # Operations that don't change sampling rate
        filtered = self.frame.low_pass_filter(cutoff=1000)
        a_weighted = self.frame.a_weighting()
        power_op = self.frame.power(exponent=2.0)

        # Sampling rate should remain unchanged
        assert filtered.sampling_rate == self.sample_rate
        assert a_weighted.sampling_rate == self.sample_rate
        assert power_op.sampling_rate == self.sample_rate

    def test_chained_operations_with_sampling_rate_updates(self) -> None:
        """Test chained operations that update sampling rate."""
        # Chain operations: filter -> a_weighting -> rms_trend
        hop_length = 512
        result = (
            self.frame.low_pass_filter(cutoff=5000)
            .a_weighting()
            .rms_trend(hop_length=hop_length)
        )

        # Final sampling rate should reflect rms_trend's update
        expected_sr = self.sample_rate / hop_length
        assert np.isclose(result.sampling_rate, expected_sr)


class TestRoughnessOperations:
    """Test roughness calculation operations."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        # Create a test signal (1 second at 44100 Hz)
        self.sample_rate: float = 44100.0
        duration: float = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # Create a signal with modulated amplitude (roughness stimuli)
        carrier_freq = 1000.0  # 1 kHz carrier
        mod_freq = 70.0  # 70 Hz modulation (creates roughness)
        signal = np.sin(2 * np.pi * carrier_freq * t) * (
            1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        )

        self.data: np.ndarray = signal.reshape(1, -1)  # 1 channel
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, 4410))
        self.frame: ChannelFrame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            label="roughness_test",
        )

    def test_roughness_dw_basic(self) -> None:
        """Test basic roughness_dw calculation."""
        result = self.frame.roughness_dw(overlap=0.5)

        # Check that result is a ChannelFrame
        assert isinstance(result, ChannelFrame)

        # Check that data shape is reduced (time-varying roughness)
        # Note: mono signals are squeezed by ChannelFrame.data property to 1D
        assert result.data.ndim in (1, 2)
        n_time_points = (
            result.data.shape[0] if result.data.ndim == 1 else result.data.shape[1]
        )
        original_samples = (
            self.frame.data.shape[0]
            if self.frame.data.ndim == 1
            else self.frame.data.shape[1]
        )
        assert n_time_points < original_samples  # Reduced time points

        # Check that sampling rate is updated
        # For overlap=0.5 with 200ms windows, sampling rate should be ~10 Hz
        assert result.sampling_rate < self.sample_rate
        # Check operation history (accept both 'name' and 'operation' keys)
        assert len(result.operation_history) == 1
        first_op = result.operation_history[0]
        op_name = first_op.get("name") or first_op.get("operation")
        assert op_name == "roughness_dw"
        assert first_op["params"]["overlap"] == 0.5

    def test_roughness_dw_different_overlap(self) -> None:
        """Test roughness_dw with different overlap values."""
        result_overlap_0 = self.frame.roughness_dw(overlap=0.0)
        result_overlap_05 = self.frame.roughness_dw(overlap=0.5)

        # Higher overlap should result in more time points
        n_time_0 = (
            result_overlap_0.data.shape[0]
            if result_overlap_0.data.ndim == 1
            else result_overlap_0.data.shape[1]
        )
        n_time_05 = (
            result_overlap_05.data.shape[0]
            if result_overlap_05.data.ndim == 1
            else result_overlap_05.data.shape[1]
        )
        assert n_time_05 > n_time_0

        # Sampling rates should be different
        assert result_overlap_05.sampling_rate > result_overlap_0.sampling_rate

    def test_roughness_dw_validates_overlap(self) -> None:
        """Test that roughness_dw validates overlap parameter."""
        with pytest.raises(ValueError, match="overlap must be in"):
            self.frame.roughness_dw(overlap=1.5)

        with pytest.raises(ValueError, match="overlap must be in"):
            self.frame.roughness_dw(overlap=-0.1)

    def test_roughness_dw_spec_basic(self) -> None:
        """Test basic roughness_dw_spec calculation."""
        from mosqito.sq_metrics import roughness_dw as roughness_dw_mosqito

        from wandas.frames.roughness import RoughnessFrame

        result = self.frame.roughness_dw_spec(overlap=0.5)

        # Check that result is a RoughnessFrame
        assert isinstance(result, RoughnessFrame)

        # Check dimensions
        # Mono signal: (47, n_time), Multi-channel: (n_channels, 47, n_time)
        assert result.data.ndim == 2  # (n_bark_bands, n_time) for mono
        assert result.data.shape[0] == 47  # 47 Bark bands
        assert result.data.shape[1] > 0  # Time points

        # Check bark_axis
        assert len(result.bark_axis) == 47
        assert result.bark_axis[0] == pytest.approx(0.5, abs=0.1)
        assert result.bark_axis[-1] == pytest.approx(23.5, abs=0.1)

        # Check properties
        assert result.n_bark_bands == 47
        assert result.overlap == 0.5
        assert len(result.time) == result.n_time_points
        # Check operation history (accept both 'name' and 'operation' keys)
        assert len(result.operation_history) == 1
        first_op = result.operation_history[0]
        op_name = first_op.get("name") or first_op.get("operation")
        assert op_name == "roughness_dw_spec"

        # Compare with MoSQITo direct calculation
        computed_data = (
            result.data.compute() if hasattr(result.data, "compute") else result.data
        )
        _, r_spec_direct, _, _ = roughness_dw_mosqito(
            self.data[0], self.sample_rate, overlap=0.5
        )
        np.testing.assert_array_equal(
            computed_data,
            r_spec_direct,
            err_msg="Specific roughness values differ from MoSQITo calculation",
        )

    def test_roughness_dw_spec_plot(self) -> None:
        """Test that roughness_dw_spec plot method works."""
        import matplotlib.pyplot as plt

        result = self.frame.roughness_dw_spec(overlap=0.5)

        # Should not raise an error
        ax = result.plot()
        assert ax is not None

        plt.close("all")

    def test_roughness_consistency(self) -> None:
        """Test that roughness_dw and roughness_dw_spec are consistent."""
        roughness = self.frame.roughness_dw(overlap=0.5)
        roughness_spec = self.frame.roughness_dw_spec(overlap=0.5)

        # Time points should match
        n_time_roughness = (
            roughness.data.shape[0]
            if roughness.data.ndim == 1
            else roughness.data.shape[1]
        )
        assert n_time_roughness == roughness_spec.n_time_points

        # Total roughness should approximately equal 0.25 * sum(R_spec)
        # R = 0.25 * sum(R_spec) according to Daniel & Weber
        # roughness_spec.data: (47, n_time) for mono
        total_from_spec = 0.25 * roughness_spec.data.sum(axis=0)
        total_direct = roughness.data

        # They should be close (allowing for numerical differences)
        np.testing.assert_allclose(
            total_direct.flatten(), total_from_spec.flatten(), rtol=0.1, atol=0.01
        )

    def test_roughness_multi_channel(self) -> None:
        """Test roughness calculation with multi-channel signal."""
        # Create 2-channel signal
        data_2ch = np.vstack([self.data, self.data * 0.8])
        dask_data_2ch: DaArray = _da_from_array(data_2ch, chunks=(1, 4410))
        frame_2ch: ChannelFrame = ChannelFrame(
            data=dask_data_2ch,
            sampling_rate=self.sample_rate,
            label="roughness_2ch",
        )

        # Test roughness_dw
        roughness = frame_2ch.roughness_dw(overlap=0.5)
        # Multi-channel data is NOT squeezed
        assert roughness.data.ndim == 2
        assert roughness.data.shape[0] == 2  # 2 channels

        # Test roughness_dw_spec
        roughness_spec = frame_2ch.roughness_dw_spec(overlap=0.5)
        assert roughness_spec.data.ndim == 3
        assert roughness_spec.data.shape[0] == 2  # 2 channels
        assert roughness_spec.data.shape[1] == 47  # 47 Bark bands
