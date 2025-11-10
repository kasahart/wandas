from unittest import mock

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.frames.noct import NOctFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestChannelTransform:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        # Create a simple dask array for testing
        self.sample_rate: float = 16000
        self.data: np.ndarray = np.random.random((2, 16000))  # 2 channels, 1 second
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, 4000))
        self.channel_frame: ChannelFrame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_fft_transform(self) -> None:
        """Test fft method for lazy transformation to frequency domain."""
        from wandas.processing import FFT

        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            # モックFFTオペレーションの設定
            mock_fft = mock.MagicMock(spec=FFT)
            mock_fft.n_fft = 4096
            mock_fft.window = "hann"
            mock_data = mock.MagicMock(spec=DaArray)
            mock_data.ndim = 2  # Set ndim property to pass dimension check
            mock_data.shape = (2, 2049)  # Set appropriate shape for a 2D array
            mock_fft.process.return_value = mock_data
            mock_create_op.return_value = mock_fft

            # fftを遅延実行
            result = self.channel_frame.fft(n_fft=4096, window="hamming")

            # オペレーションが正しく作成されたか確認
            mock_create_op.assert_called_with(
                "fft", self.sample_rate, n_fft=4096, window="hamming"
            )

            # processメソッドが呼び出されたか確認
            mock_fft.process.assert_called_once_with(self.channel_frame._data)

            # 結果が正しい型か確認
            assert isinstance(result, SpectralFrame)
            assert result.n_fft == 4096
            assert result.window == "hann"
            assert result.previous is self.channel_frame

    def test_welch_transform(self) -> None:
        """
        Test welch method for lazy transformation to frequency domain
        using Welch's method.
        """
        from wandas.processing import Welch

        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            # モックWelchオペレーションの設定
            mock_welch = mock.MagicMock(spec=Welch)
            mock_welch.n_fft = 2048
            mock_welch.hop_length = 256
            mock_welch.win_length = 1024
            mock_welch.window = "blackman"
            mock_welch.average = "mean"
            mock_data = mock.MagicMock(spec=DaArray)
            mock_data.ndim = 2  # Set ndim property to pass dimension check
            mock_data.shape = (2, 1025)  # Set appropriate shape for a 2D array
            mock_welch.process.return_value = mock_data
            mock_create_op.return_value = mock_welch

            # welchを遅延実行
            result = self.channel_frame.welch(
                n_fft=2048,
                hop_length=256,
                win_length=1024,
                window="blackman",
                average="mean",
            )

            # オペレーションが正しく作成されたか確認
            mock_create_op.assert_called_with(
                "welch",
                self.sample_rate,
                n_fft=2048,
                hop_length=256,
                win_length=1024,
                window="blackman",
                average="mean",
            )

            # processメソッドが呼び出されたか確認
            mock_welch.process.assert_called_once_with(self.channel_frame._data)

            # 結果が正しい型か確認
            assert isinstance(result, SpectralFrame)
            assert result.n_fft == 2048
            assert result.window == "blackman"
            assert result.previous is self.channel_frame

    def test_stft_transform(self) -> None:
        """Test stft method for lazy short-time Fourier transform."""
        from wandas.processing import STFT

        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            # モックSTFTオペレーションの設定
            mock_stft = mock.MagicMock(spec=STFT)
            mock_data = mock.MagicMock(spec=DaArray)
            mock_data.ndim = 3  # Set ndim property to pass dimension check
            mock_data.shape = (2, 1025, 10)  # Set appropriate shape for a 3D array
            mock_stft.process.return_value = mock_data
            mock_create_op.return_value = mock_stft

            # stftを遅延実行（デフォルト引数）
            result = self.channel_frame.stft()

            # デフォルトパラメータの確認
            mock_create_op.assert_called_with(
                "stft",
                self.sample_rate,
                n_fft=2048,
                hop_length=512,  # n_fft//4
                win_length=2048,
                window="hann",
            )

            # processメソッドが呼び出されたか確認
            mock_stft.process.assert_called_once_with(self.channel_frame._data)

            # 結果が正しい型か確認
            assert isinstance(result, SpectrogramFrame)
            assert result.n_fft == 2048
            assert result.hop_length == 512
            assert result.win_length == 2048
            assert result.window == "hann"

            # カスタムパラメータでテスト
            mock_create_op.reset_mock()
            mock_stft.process.reset_mock()

            # Update mock data shape for n_fft=1024
            mock_data.shape = (
                2,
                513,
                10,
            )  # For n_fft=1024, freq_bins = 1024 // 2 + 1 = 513

            result = self.channel_frame.stft(
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                window="hamming",
            )

            mock_create_op.assert_called_with(
                "stft",
                self.sample_rate,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                window="hamming",
            )

            assert result.n_fft == 1024
            assert result.hop_length == 256
            assert result.win_length == 1024
            assert result.window == "hamming"

    def test_noct_spectrum_transform(self) -> None:
        """Test noct_spectrum method for calculating N-octave spectrum analysis."""
        from wandas.processing import NOctSpectrum

        with mock.patch("wandas.processing.create_operation") as mock_create_op:
            # モックNOctSpectrumオペレーションの設定
            mock_noct = mock.MagicMock(spec=NOctSpectrum)
            mock_data = mock.MagicMock(spec=DaArray)
            mock_data.ndim = 2
            mock_data.shape = (2, 10)  # バンド数に応じた適切な形状
            mock_noct.process.return_value = mock_data
            mock_create_op.return_value = mock_noct

            # noct_spectrumを呼び出す
            fmin, fmax, n = 20, 20000, 3
            G, fr = 10, 1000  # noqa: N806
            result = self.channel_frame.noct_spectrum(
                fmin=fmin, fmax=fmax, n=n, G=G, fr=fr
            )

            # オペレーションが正しく作成されたか確認
            mock_create_op.assert_called_with(
                "noct_spectrum",
                self.sample_rate,
                fmin=fmin,
                fmax=fmax,
                n=n,
                G=G,
                fr=fr,
            )

            # processメソッドが呼び出されたか確認
            mock_noct.process.assert_called_once_with(self.channel_frame._data)

            # 結果が正しい型か確認
            assert isinstance(result, NOctFrame)
            assert result.fmin == fmin
            assert result.fmax == fmax
            assert result.n == n
            assert result.G == G
            assert result.fr == fr
            assert result.previous is self.channel_frame

    def test_csd(self) -> None:
        """クロススペクトル密度（CSD）メソッドのテスト"""
        # テスト用信号を作成
        sr = 1000
        t = np.linspace(0, 1, sr, endpoint=False)

        # 既知の周波数成分を持つ信号を作成（100Hzと200Hz）
        sig1 = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)
        sig2 = 0.8 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)

        # 2チャンネル信号を作成
        sigs = np.array([sig1, sig2])

        # ChannelFrameインスタンスを作成
        cf = ChannelFrame.from_numpy(sigs, sr)

        # CSD計算のパラメータ
        n_fft = 512
        win_length = 256
        hop_length = 128

        # ChannelFrameメソッドを使用してCSDを計算
        csd_frame = cf.csd(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, window="hamming"
        )

        # 実際のデータを取得するために計算
        csd_data = csd_frame.compute()

        # 形状を確認
        assert csd_data.shape == (4, n_fft // 2 + 1)

        # チャンネルラベルが正しいことを確認
        ch_pairs = [ch.label for ch in csd_frame._channel_metadata]

        # 正しいラベル名を取得
        ch0_label = cf.channels[0].label
        ch1_label = cf.channels[1].label

        expected_pairs = [
            f"csd({ch0_label}, {ch0_label})",
            f"csd({ch0_label}, {ch1_label})",
            f"csd({ch1_label}, {ch0_label})",
            f"csd({ch1_label}, {ch1_label})",
        ]

        for pair in expected_pairs:
            assert pair in ch_pairs

        # 主要な周波数成分の存在を確認
        freq_bins = np.fft.rfftfreq(n_fft, d=1 / sr)
        idx_100hz = np.argmin(np.abs(freq_bins - 100))
        idx_200hz = np.argmin(np.abs(freq_bins - 200))
        idx_300hz = np.argmin(np.abs(freq_bins - 300))

        # 自己クロススペクトル密度では対応する周波数でピークが見られるはず
        # ch0 (sig1) の自己CSDは100Hzと200Hzでピークを持つ
        ch0_auto = csd_data[0]  # ch0 -> ch0
        assert abs(ch0_auto[idx_100hz]) > abs(ch0_auto[idx_300hz])
        assert abs(ch0_auto[idx_200hz]) > abs(ch0_auto[idx_300hz])

        # ch1 (sig2) の自己CSDは100Hzと300Hzでピークを持つ
        ch1_auto = csd_data[3]  # ch1 -> ch1
        assert abs(ch1_auto[idx_100hz]) > abs(ch1_auto[idx_200hz])
        assert abs(ch1_auto[idx_300hz]) > abs(ch1_auto[idx_200hz])

    def test_transfer_function(self) -> None:
        """伝達関数メソッドのテスト"""
        # 単純なシステム用のテスト信号を作成
        sr = 1000
        t = np.linspace(0, 1, sr, endpoint=False)

        # 入力信号
        input_sig = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

        # 出力信号（100Hzでゲイン2.0、200Hzでゲイン1.5を持つ）
        # より現実的にするためにノイズを追加
        output_sig = (
            2 * np.sin(2 * np.pi * 100 * t)
            + 0.75 * np.sin(2 * np.pi * 200 * t)
            + 0.05 * np.random.randn(len(t))
        )

        # 2チャンネル信号を作成
        sigs = np.array([input_sig, output_sig])

        # ChannelFrameインスタンスを作成
        cf = ChannelFrame.from_numpy(sigs, sr)

        # 伝達関数計算のパラメータ
        n_fft = 512
        win_length = 256
        hop_length = 128

        # ChannelFrameメソッドを使用して伝達関数を計算
        tf_frame = cf.transfer_function(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, window="hamming"
        )

        # 実際のデータを取得するために計算
        tf_data = tf_frame.compute()

        # 形状を確認
        assert tf_data.shape == (4, n_fft // 2 + 1)

        # テスト周波数のインデックスを探す
        freq_bins = np.fft.rfftfreq(n_fft, 1 / sr)
        idx_100hz = np.argmin(np.abs(freq_bins - 100))
        idx_200hz = np.argmin(np.abs(freq_bins - 200))

        # 入力から出力への伝達関数（インデックス1は入力->出力）
        # 主要周波数でのゲインがほぼ正確かチェック
        h_in_to_out = tf_data[1]
        assert np.isclose(np.abs(h_in_to_out[idx_100hz]), 2.0, rtol=0.2)
        assert np.isclose(np.abs(h_in_to_out[idx_200hz]), 1.5, rtol=0.2)

        # 自己伝達関数は約1.0になるはず
        assert np.isclose(np.abs(tf_data[0, idx_100hz]), 1.0, rtol=0.2)  # 入力->入力
        assert np.isclose(np.abs(tf_data[3, idx_100hz]), 1.0, rtol=0.2)  # 出力->出力

        # チャンネルラベルが正しいことを確認
        ch_pairs = [ch.label for ch in tf_frame._channel_metadata]

        # 正しいラベル名を取得
        ch0_label = cf.channels[0].label
        ch1_label = cf.channels[1].label

        expected_pairs = [
            f"$H_{{{ch0_label}, {ch0_label}}}$",
            f"$H_{{{ch0_label}, {ch1_label}}}$",
            f"$H_{{{ch1_label}, {ch0_label}}}$",
            f"$H_{{{ch1_label}, {ch1_label}}}$",
        ]

        for pair in expected_pairs:
            assert pair in ch_pairs

    def test_coherence(self) -> None:
        """コヒーレンスメソッドのテスト"""
        # テスト用信号を作成
        sr = 1000
        t = np.linspace(0, 1, sr, endpoint=False)

        # 関連性のある2つの信号を作成
        sig1 = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)
        sig2 = (
            0.7 * np.sin(2 * np.pi * 100 * t)
            + 0.4 * np.sin(2 * np.pi * 200 * t)
            + 0.3 * np.sin(2 * np.pi * 300 * t)
        )

        # 2チャンネル信号を作成
        sigs = np.array([sig1, sig2])

        # ChannelFrameインスタンスを作成
        cf = ChannelFrame.from_numpy(sigs, sr)

        # コヒーレンス計算のパラメータ
        n_fft = 512
        win_length = 256
        hop_length = 128

        # ChannelFrameメソッドを使用してコヒーレンスを計算
        coherence_frame = cf.coherence(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, window="hamming"
        )

        # 実際のデータを取得するために計算
        coherence_data = coherence_frame.compute()

        # 形状を確認
        assert coherence_data.shape == (4, n_fft // 2 + 1)

        # 主要な周波数bins
        freq_bins = np.fft.rfftfreq(n_fft, d=1 / sr)
        idx_100hz = np.argmin(np.abs(freq_bins - 100))
        idx_300hz = np.argmin(np.abs(freq_bins - 300))

        # コヒーレンスの範囲チェック（0～1）
        # 数値計算の誤差を考慮して、わずかに1を超える値も許容する
        assert np.all(coherence_data >= 0)
        assert np.all(coherence_data <= 1.001)  # 許容誤差を追加

        # 自己コヒーレンスは1.0に近いはず
        assert np.isclose(
            coherence_data[0, idx_100hz], 1.0, rtol=1e-5
        )  # チャンネル0自己コヒーレンス
        assert np.isclose(
            coherence_data[3, idx_100hz], 1.0, rtol=1e-5
        )  # チャンネル1自己コヒーレンス

        # チャンネル間のコヒーレンスを100Hzと300Hzで確認
        # 100Hzでは両方の信号に成分があるので高いコヒーレンス
        # 300Hzではチャンネル2にのみ成分があるので低いコヒーレンス
        assert coherence_data[1, idx_100hz] > 0.8  # 100Hzでの高いコヒーレンス
        assert coherence_data[1, idx_300hz] < 0.5  # 300Hzでの低いコヒーレンス

        # チャンネルラベルが正しいことを確認
        ch_pairs = [ch.label for ch in coherence_frame._channel_metadata]

        # 正しいラベル名を取得
        ch0_label = cf.channels[0].label
        ch1_label = cf.channels[1].label

        expected_pairs = [
            f"$\\gamma_{{{ch0_label}, {ch0_label}}}$",
            f"$\\gamma_{{{ch0_label}, {ch1_label}}}$",
            f"$\\gamma_{{{ch1_label}, {ch0_label}}}$",
            f"$\\gamma_{{{ch1_label}, {ch1_label}}}$",
        ]

        for pair in expected_pairs:
            assert pair in ch_pairs

    def test_istft_calculate_output_shape_accuracy(self) -> None:
        """Test ISTFT.calculate_output_shape accuracy using ChannelFrame.

        This test verifies that:
        1. ISTFT.calculate_output_shape() predicts the correct output size
        2. The predicted n_samples matches actual reconstructed data shape
        3. The test runs across multiple parameter combinations
        """
        from wandas.processing import ISTFT

        # Test parameters: different n_fft, hop_length combinations
        test_configs = [
            {"n_fft": 512, "hop_length": 128, "win_length": 512},
            {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
            {"n_fft": 2048, "hop_length": 512, "win_length": 2048},
            {"n_fft": 4096, "hop_length": 1024, "win_length": 4096},
        ]

        sr = 16000

        for config in test_configs:
            # Create a test signal
            duration = 2.0  # 2 seconds
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            sig = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
            sigs = np.array([sig, sig])

            # Create ChannelFrame
            original = ChannelFrame.from_numpy(sigs, sr)

            # Apply STFT
            spec = original.stft(
                n_fft=config["n_fft"],
                hop_length=config["hop_length"],
                win_length=config["win_length"],
                window="hann",
            )

            # Create ISTFT operation
            istft_op = ISTFT(
                sampling_rate=sr,
                n_fft=config["n_fft"],
                hop_length=config["hop_length"],
                win_length=config["win_length"],
                window="hann",
            )

            # Get predicted shape from calculate_output_shape
            input_shape = spec.data.shape  # (channels, freqs, frames)
            predicted_shape = istft_op.calculate_output_shape(input_shape)

            # Apply iSTFT to get actual reconstructed ChannelFrame
            reconstructed = spec.istft()

            # Get actual shape from reconstructed ChannelFrame
            actual_shape = reconstructed.data.shape
            actual_n_samples = reconstructed.n_samples

            # Assertions
            assert predicted_shape == actual_shape, (
                f"Config {config}: Predicted {predicted_shape} != actual {actual_shape}"
            )

            assert predicted_shape[-1] == actual_n_samples, (
                f"Config {config}: "
                f"Predicted samples {predicted_shape[-1]} != "
                f"n_samples {actual_n_samples}"
            )

            assert actual_shape[-1] == actual_n_samples, (
                f"Config {config}: "
                f"actual_data.shape[-1] ({actual_shape[-1]}) != "
                f"n_samples ({actual_n_samples})"
            )

            print(
                f"✓ Config {config}: "
                f"predicted={predicted_shape}, actual={actual_shape}, "
                f"n_samples={actual_n_samples}"
            )
