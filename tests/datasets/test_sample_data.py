# filepath: /workspaces/wandas/tests/datasets/test_sample_data.py
import numpy as np

from wandas.datasets.sample_data import load_sample_signal


class TestSampleData:
    def test_load_sample_signal_shape(self) -> None:
        """サンプル信号の形状が期待通りであることを確認するテスト"""
        # 標準のパラメータでテスト
        frequency = 5.0
        sampling_rate = 100
        duration = 1.0

        signal = load_sample_signal(
            frequency=frequency, sampling_rate=sampling_rate, duration=duration
        )

        # 期待されるサンプル数 = サンプリングレート × 持続時間
        expected_samples = int(sampling_rate * duration)

        # 形状の確認
        assert signal.shape == (expected_samples,)

        # 異なる持続時間での確認
        duration_2 = 2.5
        signal_2 = load_sample_signal(
            frequency=frequency, sampling_rate=sampling_rate, duration=duration_2
        )
        expected_samples_2 = int(sampling_rate * duration_2)
        assert signal_2.shape == (expected_samples_2,)

        # 異なるサンプリングレートでの確認
        sampling_rate_2 = 44100
        signal_3 = load_sample_signal(
            frequency=frequency, sampling_rate=sampling_rate_2, duration=duration
        )
        expected_samples_3 = int(sampling_rate_2 * duration)
        assert signal_3.shape == (expected_samples_3,)

    def test_load_sample_signal_frequency(self) -> None:
        """生成された信号が正しい周波数を持つことを確認するテスト"""
        # 基本的なテスト設定
        frequency = 10.0  # 10Hz
        sampling_rate = 1000  # 1kHz (十分な解像度のために高めに設定)
        duration = 1.0

        signal = load_sample_signal(
            frequency=frequency, sampling_rate=sampling_rate, duration=duration
        )

        # FFTを使用して周波数スペクトルを計算
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1 / sampling_rate)

        # 最大振幅を持つ周波数を特定
        peak_freq_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_freq_idx]

        # 指定した周波数に十分近いことを確認 (小さな誤差は許容)
        assert np.isclose(peak_freq, frequency, rtol=1e-1)

        # 異なる周波数でのテスト
        frequency_2 = 20.0
        signal_2 = load_sample_signal(
            frequency=frequency_2, sampling_rate=sampling_rate, duration=duration
        )

        spectrum_2 = np.abs(np.fft.rfft(signal_2))
        peak_freq_idx_2 = np.argmax(spectrum_2)
        peak_freq_2 = freqs[peak_freq_idx_2]

        assert np.isclose(peak_freq_2, frequency_2, rtol=1e-1)

    def test_load_sample_signal_dtype(self) -> None:
        """生成された信号が期待されるデータ型を持つことを確認するテスト"""
        signal = load_sample_signal()

        # 関数の戻り値の型確認
        assert isinstance(signal, np.ndarray)

        # データ型がfloat64であることを確認
        assert signal.dtype == np.float64

    def test_load_sample_signal_amplitude(self) -> None:
        """生成された信号の振幅が正しいことを確認するテスト"""
        signal = load_sample_signal()

        # サイン波の振幅は1.0であるべき
        assert np.isclose(np.max(signal), 1.0)
        assert np.isclose(np.min(signal), -1.0)

    def test_load_sample_signal_defaults(self) -> None:
        """デフォルトパラメータが正しく機能することを確認するテスト"""
        # デフォルトパラメータを使用して信号を生成
        default_signal = load_sample_signal()

        # 明示的にデフォルト値を指定して生成した信号と比較
        explicit_signal = load_sample_signal(
            frequency=5.0, sampling_rate=100, duration=1.0
        )

        # 両者が同じであることを確認
        np.testing.assert_array_equal(default_signal, explicit_signal)

        # デフォルト値に基づくサンプル数を確認
        assert len(default_signal) == 100  # サンプリングレート100 × 持続時間1秒
