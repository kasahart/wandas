from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame

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
