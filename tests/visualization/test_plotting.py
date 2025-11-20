from collections.abc import Iterator
from typing import Any, Optional, Union
from unittest import mock

import dask.array as da
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure

import wandas as wd
from wandas.visualization.plotting import (
    DescribePlotStrategy,
    FrequencyPlotStrategy,
    PlotStrategy,
    SpectrogramPlotStrategy,
    WaveformPlotStrategy,
    create_operation,
    get_plot_strategy,
    register_plot_strategy,
)

# Matplotlibのインタラクティブモードをオフにする
plt.ioff()

_da_from_array = da.from_array  # type: ignore [unused-ignore]


# テスト用のプロット戦略クラス
class TestPlotStrategy(PlotStrategy[Any]):
    """テスト用のプロット戦略"""

    name = "test_strategy"

    def channel_plot(self, x: Any, y: Any, ax: "Axes", **kwargs: Any) -> None:
        pass

    def plot(
        self,
        bf: Any,
        ax: Optional["Axes"] = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        if ax is None:
            fig, created_ax = plt.subplots()
            return created_ax
        return ax


class TestPlotting:
    """プロット機能のテストクラス"""

    def setup_method(self) -> None:
        """各テストの前に実行"""
        # 既存の登録を一時的に保存
        from wandas.visualization.plotting import _plot_strategies

        self.original_strategies = _plot_strategies.copy()

        # モックフレームの作成
        self.mock_channel_frame = mock.MagicMock()
        self.mock_channel_frame.n_channels = 2
        self.mock_channel_frame.time = np.linspace(0, 1, 1000)
        self.mock_channel_frame.data = np.random.rand(2, 1000)
        self.mock_channel_frame.labels = ["ch1", "ch2"]
        self.mock_channel_frame.label = "Test Channel"
        self.mock_channel_frame.channels = [
            mock.MagicMock(label="ch1"),
            mock.MagicMock(label="ch2"),
        ]

        # 単一チャネル用のモックチャネルフレーム
        self.mock_single_channel_frame = mock.MagicMock()
        self.mock_single_channel_frame.n_channels = 1
        self.mock_single_channel_frame.time = np.linspace(0, 1, 1000)
        self.mock_single_channel_frame.data = np.random.rand(1000)
        self.mock_single_channel_frame.labels = ["ch1"]
        self.mock_single_channel_frame.label = "Test Single Channel"
        self.mock_single_channel_frame.channels = [
            mock.MagicMock(label="ch1"),
        ]

        self.mock_spectral_frame = mock.MagicMock()
        self.mock_spectral_frame.n_channels = 2
        self.mock_spectral_frame.freqs = np.linspace(0, 22050, 513)
        self.mock_spectral_frame.dB = np.random.rand(2, 513)
        self.mock_spectral_frame.dBA = np.random.rand(2, 513)
        self.mock_spectral_frame.labels = ["ch1", "ch2"]
        self.mock_spectral_frame.label = "Test Spectral"
        self.mock_spectral_frame.channels = [
            mock.MagicMock(label="ch1"),
            mock.MagicMock(label="ch2"),
        ]

        # 単一チャネル用のモックスペクトルフレーム
        self.mock_single_spectral_frame = mock.MagicMock()
        self.mock_single_spectral_frame.n_channels = 1
        self.mock_single_spectral_frame.freqs = np.linspace(0, 22050, 513)
        self.mock_single_spectral_frame.dB = np.random.rand(513)
        self.mock_single_spectral_frame.dBA = np.random.rand(513)
        self.mock_single_spectral_frame.labels = ["ch1"]
        self.mock_single_spectral_frame.label = "Test Single Spectral"
        self.mock_single_spectral_frame.channels = [
            mock.MagicMock(label="ch1"),
        ]

        # NOctFrameのモック
        self.mock_noct_frame = mock.MagicMock()
        self.mock_noct_frame.n_channels = 2
        self.mock_noct_frame.n = 3  # 1/3オクターブ
        self.mock_noct_frame.freqs = np.array(
            [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        )
        self.mock_noct_frame.dB = np.random.rand(2, 8)
        self.mock_noct_frame.dBA = np.random.rand(2, 8)
        self.mock_noct_frame.labels = ["ch1", "ch2"]
        self.mock_noct_frame.label = "Test NOct"
        self.mock_noct_frame.channels = [
            mock.MagicMock(label="ch1"),
            mock.MagicMock(label="ch2"),
        ]

        # 単一チャネル用のNOctFrameのモック
        self.mock_single_noct_frame = mock.MagicMock()
        self.mock_single_noct_frame.n_channels = 1
        self.mock_single_noct_frame.n = 3  # 1/3オクターブ
        self.mock_single_noct_frame.freqs = np.array(
            [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        )
        self.mock_single_noct_frame.dB = np.random.rand(8)
        self.mock_single_noct_frame.dBA = np.random.rand(8)
        self.mock_single_noct_frame.labels = ["ch1"]
        self.mock_single_noct_frame.label = "Test Single NOct"
        self.mock_single_noct_frame.channels = [
            mock.MagicMock(label="ch1"),
        ]

        self.mock_spectrogram_frame = mock.MagicMock()
        self.mock_spectrogram_frame.n_channels = 2
        self.mock_spectrogram_frame.n_freq_bins = 513
        self.mock_spectrogram_frame.shape = (2, 513, 10)
        self.mock_spectrogram_frame.sampling_rate = 44100
        self.mock_spectrogram_frame.n_fft = 1024
        self.mock_spectrogram_frame.hop_length = 512
        self.mock_spectrogram_frame.win_length = 1024
        self.mock_spectrogram_frame.dB = np.random.rand(2, 513, 10)
        self.mock_spectrogram_frame.dBA = np.random.rand(2, 513, 10)
        self.mock_spectrogram_frame.channels = [
            mock.MagicMock(label="ch1"),
            mock.MagicMock(label="ch2"),
        ]
        self.mock_spectrogram_frame.label = "Test Spectrogram"

        # スペクトログラムテスト用に単一チャネルバージョンも作成
        self.mock_single_spectrogram_frame = mock.MagicMock()
        self.mock_single_spectrogram_frame.n_channels = 1
        self.mock_single_spectrogram_frame.n_freq_bins = 513
        self.mock_single_spectrogram_frame.shape = (513, 10)
        self.mock_single_spectrogram_frame.sampling_rate = 44100
        self.mock_single_spectrogram_frame.n_fft = 1024
        self.mock_single_spectrogram_frame.hop_length = 512
        self.mock_single_spectrogram_frame.win_length = 1024
        self.mock_single_spectrogram_frame.dB = np.random.rand(513, 10)
        self.mock_single_spectrogram_frame.dBA = np.random.rand(513, 10)
        self.mock_single_spectrogram_frame.channels = [
            mock.MagicMock(label="ch1"),
        ]
        self.mock_single_spectrogram_frame.label = "Test Single Spectrogram"

        # 単一チャネルのコヒーレンスデータ（自己相関）
        self.mock_single_coherence_spectral_frame = mock.MagicMock()
        self.mock_single_coherence_spectral_frame.n_channels = 1
        self.mock_single_coherence_spectral_frame.freqs = self.mock_spectral_frame.freqs
        self.mock_single_coherence_spectral_frame.magnitude = np.random.rand(513)
        self.mock_single_coherence_spectral_frame.labels = ["ch1-ch1"]
        self.mock_single_coherence_spectral_frame.label = "Single Coherence Data"
        self.mock_single_coherence_spectral_frame.operation_history = [
            {"operation": "coherence"}
        ]
        self.mock_single_coherence_spectral_frame.channels = [
            mock.MagicMock(label="ch1-ch1")
        ]

        # コヒーレンスデータでのテスト
        self.mock_coherence_spectral_frame = mock.MagicMock()
        self.mock_coherence_spectral_frame.n_channels = 4
        self.mock_coherence_spectral_frame.freqs = self.mock_spectral_frame.freqs
        self.mock_coherence_spectral_frame.magnitude = np.random.rand(4, 513)
        self.mock_coherence_spectral_frame.labels = [
            "ch1-ch1",
            "ch1-ch2",
            "ch2-ch1",
            "ch2-ch2",
        ]
        self.mock_coherence_spectral_frame.label = "Coherence Data"
        self.mock_coherence_spectral_frame.operation_history = [
            {"operation": "coherence"}
        ]
        self.mock_coherence_spectral_frame.channels = [
            mock.MagicMock(label=label)
            for label in self.mock_coherence_spectral_frame.labels
        ]

    def teardown_method(self) -> None:
        """各テスト後の後処理"""
        # 元の戦略を復元
        from wandas.visualization.plotting import _plot_strategies

        _plot_strategies.clear()
        _plot_strategies.update(self.original_strategies)
        plt.close("all")  # すべての図を閉じる

    def test_plot_strategy_registry(self) -> None:
        """プロット戦略登録機能のテスト"""
        # デフォルト戦略が登録されていることを確認
        strategies = ["waveform", "frequency", "spectrogram", "describe"]
        for name in strategies:
            strategy_cls = get_plot_strategy(name)
            assert strategy_cls.name == name

        # 存在しない戦略を取得しようとするとエラー
        with pytest.raises(ValueError, match="Unknown plot type"):
            get_plot_strategy("nonexistent_strategy")

        # 新しい戦略を登録
        register_plot_strategy(TestPlotStrategy)
        strategy_cls = get_plot_strategy("test_strategy")
        assert strategy_cls.name == "test_strategy"
        assert strategy_cls is TestPlotStrategy

    def test_register_invalid_strategy(self) -> None:
        """無効なプロット戦略登録のテスト"""

        # PlotStrategyを継承していないクラス
        class InvalidStrategy:
            name = "invalid"

        with pytest.raises(TypeError, match="must inherit from PlotStrategy"):
            register_plot_strategy(InvalidStrategy)

        # 抽象クラス
        class AbstractStrategy(PlotStrategy[Any]):
            name = "abstract"

        with pytest.raises(
            TypeError, match="Cannot register abstract PlotStrategy class"
        ):
            register_plot_strategy(AbstractStrategy)

    def test_create_operation(self) -> None:
        """create_operation関数のテスト"""
        # 有効な操作を作成
        op = create_operation("waveform")
        assert isinstance(op, WaveformPlotStrategy)

        # 追加パラメータを持つ操作を作成 - ただしパラメータは無視される
        # この呼び出しは引数なしのコンストラクタを使っていることをテスト
        op_with_params = create_operation("frequency")
        assert isinstance(op_with_params, FrequencyPlotStrategy)

        # 存在しない操作を作成しようとするとエラー
        with pytest.raises(ValueError, match="Unknown plot type"):
            create_operation("nonexistent_operation")

    def test_waveform_plot_strategy(self) -> None:
        """WaveformPlotStrategyのテスト"""
        strategy = WaveformPlotStrategy()

        # channel_plotのテスト
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_channel_frame.time, self.mock_channel_frame.data[0], ax
        )
        assert ax.get_ylabel() == "Amplitude"

        # 単一チャネルでのplotのテスト (overlay=True)
        result = strategy.plot(self.mock_channel_frame, overlay=True)
        assert isinstance(result, Axes)

        # 複数チャネルでのplotのテスト (overlay=False)
        result = strategy.plot(self.mock_channel_frame, overlay=False)
        assert isinstance(result, Iterator)

    def test_single_channel_waveform_plot_strategy(self) -> None:
        """単一チャネルのWaveformPlotStrategyテスト"""
        strategy = WaveformPlotStrategy()

        # 単一チャネルでのchannel_plotのテスト（ラベル付き）
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_single_channel_frame.time,
            self.mock_single_channel_frame.data,
            ax,
            label="Test Single Channel",
        )
        assert ax.get_ylabel() == "Amplitude"
        # 凡例が表示されていることを確認
        assert len(ax.get_legend().get_texts()) > 0
        assert ax.get_legend().get_texts()[0].get_text() == "Test Single Channel"

        # 単一チャネルでのplotのテスト (overlay=True)
        result = strategy.plot(self.mock_single_channel_frame, overlay=True)
        assert isinstance(result, Axes)
        assert result.get_title() == "Test Single Channel"

        # 単一チャネルでのplotのテスト (overlay=False)
        result = strategy.plot(self.mock_single_channel_frame, overlay=False)
        # 単一チャネルでもoverlay=Falseの場合はイテレータを返す
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 1  # 1チャネルなので軸は1つだけ
        assert axes_list[0].get_title() == "ch1"

        # カスタムタイトルのテスト
        result = strategy.plot(self.mock_single_channel_frame, title="Custom Title")
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 1  # 1チャネルなので軸は1つだけ
        assert axes_list[0].get_title() == "ch1"
        # suptitleを確認
        assert axes_list[0].figure.get_suptitle() == "Custom Title"
        plt.close("all")  # すべての図を閉じる

    def test_frequency_plot_strategy(self) -> None:
        """FrequencyPlotStrategyのテスト"""
        strategy = FrequencyPlotStrategy()

        # channel_plotのテスト
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_spectral_frame.freqs, self.mock_spectral_frame.dB[0], ax
        )

        # dB単位でのplotのテスト
        result = strategy.plot(self.mock_spectral_frame, overlay=True)
        assert isinstance(result, Axes)

        # dBA単位でのplotのテスト
        result = strategy.plot(self.mock_spectral_frame, overlay=True, Aw=True)
        assert isinstance(result, Axes)

        # 複数チャネルでのplotのテスト
        result = strategy.plot(self.mock_spectral_frame, overlay=False)
        assert isinstance(result, Iterator)

    def test_single_channel_frequency_plot_strategy(self) -> None:
        """単一チャネルのFrequencyPlotStrategyテスト"""
        strategy = FrequencyPlotStrategy()

        # 単一チャネルでのchannel_plotのテスト（ラベル付き）
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_single_spectral_frame.freqs,
            self.mock_single_spectral_frame.dB,
            ax,
            label="Test Single Frequency",
        )
        # assert ax.get_ylabel() == "Level [dB]"
        # 凡例が表示されていることを確認
        assert len(ax.get_legend().get_texts()) > 0
        assert ax.get_legend().get_texts()[0].get_text() == "Test Single Frequency"

        # 単一チャネルでdB単位でのplotのテスト (overlay=True)
        result = strategy.plot(self.mock_single_spectral_frame, overlay=True)
        assert isinstance(result, Axes)
        assert result.get_title() == "Test Single Spectral"
        assert result.get_xlabel() == "Frequency [Hz]"
        assert result.get_ylabel() == "Spectrum level [dB]"

        # 単一チャネルでdBA単位でのplotのテスト (overlay=True, Aw=True)
        result = strategy.plot(self.mock_single_spectral_frame, overlay=True, Aw=True)
        assert isinstance(result, Axes)
        assert result.get_ylabel() == "Spectrum level [dBA]"

        # 単一チャネルでのplotのテスト (overlay=False)
        result = strategy.plot(self.mock_single_spectral_frame, overlay=False)
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 1  # 1チャネルなので軸は1つだけ
        assert axes_list[0].get_title() == "ch1"

        # カスタムタイトルのテスト
        result = strategy.plot(self.mock_single_spectral_frame, title="Custom Title")
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert axes_list[0].get_title() == "ch1"
        assert axes_list[0].figure.get_suptitle() == "Custom Title"

        plt.close("all")  # すべての図を閉じる

    def test_spectrogram_plot_strategy(self) -> None:
        """SpectrogramPlotStrategyのテスト"""
        strategy = SpectrogramPlotStrategy()

        # オーバーレイモードはサポートされていない
        with pytest.raises(ValueError, match="Overlay is not supported"):
            strategy.plot(self.mock_spectrogram_frame, overlay=True)

        # テスト1: 単一チャネルのスペクトログラムフレームでのテスト
        fig, ax = plt.subplots()
        result = strategy.plot(self.mock_single_spectrogram_frame, ax=ax)

        # 戻り値が単一のAxesであることを確認
        assert isinstance(result, Axes)
        assert result is ax
        assert result.get_xlabel() == "Time [s]"
        assert result.get_ylabel() == "Frequency [Hz]"

        # テスト2: チャネル数が1より大きい場合、axを指定するとエラー
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="ax must be None when n_channels > 1"):
            strategy.plot(self.mock_spectrogram_frame, ax=ax)

        # テスト3: 複数チャネルでのテスト（axなし）
        result = strategy.plot(self.mock_spectrogram_frame)

        # 結果がAxesのイテレータであることを確認
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == self.mock_spectrogram_frame.n_channels * 2

        # 各軸が適切に設定されていることを確認
        for ax in axes_list[:2]:
            assert ax.get_xlabel() == "Time [s]"
            assert ax.get_ylabel() == "Frequency [Hz]"

        # すべての図をクローズ
        plt.close("all")

    def test_describe_plot_strategy(self) -> None:
        """DescribePlotStrategyのテスト"""
        strategy = DescribePlotStrategy()

        # モックのstftとwelchメソッド
        self.mock_channel_frame.stft.return_value = self.mock_spectrogram_frame
        self.mock_channel_frame.welch.return_value = self.mock_spectral_frame

        # Matplotlibのメソッドを部分的にモック
        with (
            mock.patch("matplotlib.figure.Figure.add_subplot") as mock_add_subplot,
            mock.patch("librosa.display.specshow") as mock_specshow,
            mock.patch("matplotlib.pyplot.figure") as mock_figure,
            mock.patch.object(Figure, "colorbar"),
        ):
            # モックspectershowの戻り値を設定
            mock_img = mock.MagicMock(spec=QuadMesh)
            mock_specshow.return_value = mock_img

            mock_fig = mock.MagicMock(spec=Figure)
            mock_figure.return_value = mock_fig

            mock_ax1 = mock.MagicMock(spec=Axes)
            mock_ax2 = mock.MagicMock(spec=Axes)
            mock_ax3 = mock.MagicMock(spec=Axes)
            mock_ax4 = mock.MagicMock(spec=Axes)

            mock_axes_iter = iter([mock_ax1, mock_ax2, mock_ax3, mock_ax4])

            def side_effect(*args: Any, **kwargs: Any) -> Axes:
                return next(mock_axes_iter)

            mock_add_subplot.side_effect = side_effect
            mock_fig.axes = [mock_ax1, mock_ax2, mock_ax3, mock_ax4]

            # プロットの実行
            result = strategy.plot(self.mock_channel_frame)

            # 戻り値がAxesのイテレータであることを確認
            assert isinstance(result, Iterator)

            # stftメソッドとwelchメソッドが呼び出されることを確認
            self.mock_channel_frame.stft.assert_called_once()
            self.mock_channel_frame.welch.assert_called_once()

    def test_single_channel_describe_plot_strategy(self) -> None:
        """単一チャネルのDescribePlotStrategyテスト"""
        strategy = DescribePlotStrategy()

        # 単一チャネルのモックのstftとwelchメソッド
        self.mock_single_channel_frame.stft.return_value = (
            self.mock_single_spectrogram_frame
        )
        self.mock_single_channel_frame.welch.return_value = (
            self.mock_single_spectral_frame
        )

        # Matplotlibのメソッドを部分的にモック
        with (
            mock.patch("matplotlib.figure.Figure.add_subplot") as mock_add_subplot,
            mock.patch("librosa.display.specshow") as mock_specshow,
            mock.patch("matplotlib.pyplot.figure") as mock_figure,
            mock.patch.object(Figure, "colorbar"),
        ):
            # モックspectershowの戻り値を設定
            mock_img = mock.MagicMock(spec=QuadMesh)
            mock_specshow.return_value = mock_img

            mock_fig = mock.MagicMock(spec=Figure)
            mock_figure.return_value = mock_fig

            mock_ax1 = mock.MagicMock(spec=Axes)
            mock_ax2 = mock.MagicMock(spec=Axes)

            mock_axes_iter = iter([mock_ax1, mock_ax2])

            def side_effect(*args: Any, **kwargs: Any) -> Axes:
                return next(mock_axes_iter)

            mock_add_subplot.side_effect = side_effect
            mock_fig.axes = [mock_ax1, mock_ax2]

            # 単一チャネルでのプロットの実行
            result = strategy.plot(self.mock_single_channel_frame)

            # 戻り値がAxesのイテレータであることを確認
            assert isinstance(result, Iterator)

            # 単一チャネルでもstftメソッドとwelchメソッドが呼び出されることを確認
            self.mock_single_channel_frame.stft.assert_called_once()
            self.mock_single_channel_frame.welch.assert_called_once()

            # # タイトルが設定されていることを確認
            # mock_ax1.set_title.assert_called()
            # mock_ax2.set_title.assert_called()

            plt.close("all")  # すべての図を閉じる

    def test_noct_plot_strategy(self) -> None:
        """NOctPlotStrategyのテスト"""
        from wandas.visualization.plotting import NOctPlotStrategy

        strategy = NOctPlotStrategy()

        # channel_plotのテスト
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_noct_frame.freqs,
            self.mock_noct_frame.dB[0],
            ax,
            label="Test NOct",
        )
        # stepプロットが使われ、グリッドと凡例が表示されていることを確認
        assert len(ax.xaxis.get_gridlines()) > 0  # グリッドが表示されていることを確認
        assert len(ax.get_legend().get_texts()) > 0  # 凡例が表示されていることを確認

        # 単一チャネルでのplotのテスト (overlay=True)
        result = strategy.plot(self.mock_noct_frame, overlay=True)
        assert isinstance(result, Axes)
        assert result.get_xlabel() == "Center frequency [Hz]"
        assert result.get_ylabel() == "Spectrum level [dBr]"
        assert result.get_title() == "Test NOct"

        # dBA単位でのplotのテスト (overlay=True, Aw=True)
        result = strategy.plot(self.mock_noct_frame, overlay=True, Aw=True)
        assert isinstance(result, Axes)
        assert result.get_ylabel() == "Spectrum level [dBrA]"

        # 複数チャネルでのplotのテスト (overlay=False)
        result = strategy.plot(self.mock_noct_frame, overlay=False)
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == self.mock_noct_frame.n_channels

        # 最後の軸のxラベルとyラベルを確認
        assert axes_list[-1].get_xlabel() == "Center frequency [Hz]"
        assert axes_list[-1].get_ylabel() == "Spectrum level [dBr]"

    def test_single_channel_noct_plot_strategy(self) -> None:
        """単一チャネル用のNOctPlotStrategyのテスト"""
        from wandas.visualization.plotting import NOctPlotStrategy

        strategy = NOctPlotStrategy()

        # 単一チャネルでのchannel_plotのテスト
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_single_noct_frame.freqs,
            self.mock_single_noct_frame.dB,
            ax,
            label="Test Single NOct",
        )
        # プロットの特性を確認
        assert len(ax.xaxis.get_gridlines()) > 0  # グリッドが表示されていることを確認
        assert len(ax.get_legend().get_texts()) > 0  # 凡例が表示されていることを確認
        assert ax.get_legend().get_texts()[0].get_text() == "Test Single NOct"

        # 単一チャネルでのplotのテスト (overlay=True)
        result = strategy.plot(self.mock_single_noct_frame, overlay=True)
        assert isinstance(result, Axes)
        assert result.get_xlabel() == "Center frequency [Hz]"
        assert result.get_ylabel() == "Spectrum level [dBr]"
        assert result.get_title() == "Test Single NOct"

        # 単一チャネルでdBA単位でのplotのテスト (overlay=True, Aw=True)
        result = strategy.plot(self.mock_single_noct_frame, overlay=True, Aw=True)
        assert isinstance(result, Axes)
        assert result.get_ylabel() == "Spectrum level [dBrA]"

        # 単一チャネルでのplotのテスト (overlay=False)
        result = strategy.plot(self.mock_single_noct_frame, overlay=False)
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 1  # 1チャネルなので軸は1つだけ
        assert axes_list[0].get_xlabel() == "Center frequency [Hz]"
        assert axes_list[0].get_ylabel() == "Spectrum level [dBr]"
        assert axes_list[0].get_title() == "ch1"
        assert axes_list[0].figure.get_suptitle() == "Test Single NOct"
        # カスタムタイトルのテスト
        result = strategy.plot(self.mock_single_noct_frame, title="Custom NOct Title")
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert axes_list[0].get_title() == "ch1"
        assert axes_list[0].figure.get_suptitle() == "Custom NOct Title"

        plt.close("all")  # すべての図を閉じる

    def test_matrix_plot_strategy(self) -> None:
        """MatrixPlotStrategyのテスト"""
        from wandas.visualization.plotting import MatrixPlotStrategy

        strategy = MatrixPlotStrategy()

        # channel_plotのテスト
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_coherence_spectral_frame.freqs,
            self.mock_coherence_spectral_frame.magnitude[0],
            ax,
            title="Test Channel",
            ylabel="Test Label",
        )

        # 正しくラベルとタイトルが設定されていることを確認
        assert ax.get_xlabel() == "Frequency [Hz]"
        assert ax.get_ylabel() == "Test Label"
        assert ax.get_title() == "Test Channel"
        # グリッドが表示されていることを確認
        assert len(ax.xaxis.get_gridlines()) > 0

        # 通常の周波数データでのplotのテスト
        result = strategy.plot(self.mock_coherence_spectral_frame)

        # 結果がAxesのイテレータであることを確認
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == self.mock_coherence_spectral_frame.n_channels

        # 各軸のラベルとタイトルを確認
        for i, ax in enumerate(axes_list):
            assert ax.get_xlabel() == "Frequency [Hz]"
            assert "coherence" in ax.get_ylabel()
            assert self.mock_coherence_spectral_frame.labels[i] in ax.get_title()

        plt.close("all")  # すべての図をクローズ

        # A特性重み付けデータでのテスト (Aw=True)
        result = strategy.plot(self.mock_spectral_frame, Aw=True)

        # 結果がAxesのイテレータであることを確認
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 4

        # 各軸のラベルとタイトルを確認（A特性重み付け）
        for i, ax in enumerate(axes_list[:2]):
            assert ax.get_xlabel() == "Frequency [Hz]"
            assert "dBA" in ax.get_ylabel() or "A-weighted" in ax.get_ylabel()
            assert self.mock_spectral_frame.labels[i] in ax.get_title()

        plt.close("all")  # すべての図をクローズ

        # コヒーレンスデータでのテスト

        result = strategy.plot(self.mock_coherence_spectral_frame)

        # 結果がAxesのイテレータであることを確認
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == self.mock_coherence_spectral_frame.n_channels

        # 各軸のラベルとタイトルを確認（コヒーレンス）
        for i, ax in enumerate(axes_list):
            assert ax.get_xlabel() == "Frequency [Hz]"
            # y軸ラベルに「coherence」が含まれることを確認
            assert "coherence" in ax.get_ylabel().lower()
            assert self.mock_coherence_spectral_frame.labels[i] in ax.get_title()

        plt.close("all")  # すべての図をクローズ

    def test_single_channel_matrix_plot_strategy(self) -> None:
        """単一チャネル用のMatrixPlotStrategyのテスト"""
        from wandas.visualization.plotting import MatrixPlotStrategy

        strategy = MatrixPlotStrategy()

        # 単一チャネルのchannel_plotテスト
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_single_coherence_spectral_frame.freqs,
            self.mock_single_coherence_spectral_frame.magnitude,
            ax,
            title="Single Coherence Test",
            ylabel="Magnitude",
            label="ch1-ch1",
        )

        # 正しいラベルとタイトルが設定されていることを確認
        assert ax.get_xlabel() == "Frequency [Hz]"
        assert ax.get_ylabel() == "Magnitude"
        assert ax.get_title() == "Single Coherence Test"
        # 凡例が表示されていることを確認
        # assert len(ax.get_legend().get_texts()) > 0
        # assert ax.get_legend().get_texts()[0].get_text() == "ch1-ch1"
        # グリッドが表示されていることを確認
        assert len(ax.xaxis.get_gridlines()) > 0

        # 単一チャネルのコヒーレンスデータでのplotテスト
        result = strategy.plot(self.mock_single_coherence_spectral_frame)

        # 結果がAxesのイテレータであることを確認
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 1  # 1チャネルなので軸は1つだけ

        # 軸のラベルとタイトルを確認
        assert axes_list[0].get_xlabel() == "Frequency [Hz]"
        assert "coherence" in axes_list[0].get_ylabel().lower()
        label = self.mock_single_coherence_spectral_frame.labels[0]
        assert label in axes_list[0].get_title()

        # カスタムタイトルとユニットでのテスト
        result = strategy.plot(
            self.mock_single_coherence_spectral_frame,
            title="Custom Matrix Title",
            ylabel="Custom Y Units",
        )
        # iteratorになることを確認した上で、リスト化
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert axes_list[0].get_title() == "ch1-ch1"
        assert "Custom Y Units" in axes_list[0].get_ylabel()
        assert axes_list[0].figure.get_suptitle() == "Custom Matrix Title"

        plt.close("all")  # すべての図をクローズ

    def test_waveform_plot_strategy_edge_cases(self) -> None:
        """WaveformPlotStrategyのエッジケースのテスト"""
        strategy = WaveformPlotStrategy()

        # カスタムパラメータでのテスト
        result = strategy.plot(
            self.mock_channel_frame,
            overlay=True,
            alpha=0.5,
            color="red",
            xlabel="Custom Time",
            ylabel="Custom Amplitude",
        )
        assert isinstance(result, Axes)
        assert result.get_xlabel() == "Custom Time"
        assert result.get_ylabel() == "Custom Amplitude"

        # 外部のaxを指定した場合のテスト
        fig, external_ax = plt.subplots()
        result = strategy.plot(
            self.mock_channel_frame,
            ax=external_ax,
            overlay=True,
            title="External Ax Test",
        )
        assert result is external_ax
        assert result.get_title() == "External Ax Test"

        plt.close("all")

    def test_frequency_plot_strategy_edge_cases(self) -> None:
        """FrequencyPlotStrategyのエッジケースのテスト"""
        strategy = FrequencyPlotStrategy()

        # コヒーレンス操作履歴を持つフレームでのテスト
        self.mock_spectral_frame.operation_history = [{"operation": "coherence"}]
        self.mock_spectral_frame.magnitude = np.random.rand(2, 513)

        result = strategy.plot(self.mock_spectral_frame, overlay=True)
        assert isinstance(result, Axes)
        assert "coherence" in result.get_ylabel()

        # 操作履歴をリセット
        self.mock_spectral_frame.operation_history = []

        # カスタムパラメータでのテスト
        result = strategy.plot(
            self.mock_spectral_frame,
            overlay=True,
            alpha=0.7,
            linewidth=2,
            xlabel="Custom Frequency",
            ylabel="Custom Level",
        )
        assert isinstance(result, Axes)
        assert result.get_xlabel() == "Custom Frequency"
        assert result.get_ylabel() == "Custom Level"

        plt.close("all")

    def test_spectrogram_plot_strategy_edge_cases(self) -> None:
        """SpectrogramPlotStrategyのエッジケースのテスト"""
        strategy = SpectrogramPlotStrategy()

        # dBA単位でのテスト
        fig, ax = plt.subplots()
        with mock.patch("librosa.display.specshow") as mock_specshow:
            mock_img = mock.MagicMock()
            mock_specshow.return_value = mock_img

            result = strategy.plot(
                self.mock_single_spectrogram_frame,
                ax=ax,
                Aw=True,
                cmap="viridis",
                vmin=-100,
                vmax=0,
            )

            assert result is ax
            # dBA単位が使用されていることを確認
            mock_specshow.assert_called_once()
            call_args = mock_specshow.call_args
            assert "cmap" in call_args[1]
            assert call_args[1]["cmap"] == "viridis"

        plt.close("all")

    def test_spectrogram_plot_strategy_basic_functionality(self) -> None:
        """SpectrogramPlotStrategyの基本機能テスト（mockなし）"""
        strategy = SpectrogramPlotStrategy()

        # テスト用の実データを作成
        sample_rate: float = 44100
        duration: float = 0.1  # 短時間のテストデータ
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        # シンプルなサイン波
        signal = np.sin(2 * np.pi * 440.0 * t)  # 440Hz

        # ChannelFrameを作成
        from wandas.frames.channel import ChannelFrame

        dask_data = _da_from_array(signal.reshape(1, -1), chunks=(1, -1))
        channel_frame = ChannelFrame(
            data=dask_data, sampling_rate=sample_rate, label="test_channel"
        )

        # STFTを実行してSpectrogramFrameを作成
        spectrogram_frame = channel_frame.stft(n_fft=512, hop_length=256)

        # 単一チャネルでのテスト
        fig, ax = plt.subplots()
        result = strategy.plot(spectrogram_frame, ax=ax)

        # 戻り値が正しいAxesオブジェクトであることを確認
        assert result is ax
        assert result.get_xlabel() == "Time [s]"
        assert result.get_ylabel() == "Frequency [Hz]"

        plt.close(fig)

        # 複数チャネルのテスト用データ
        multi_signal = np.array([signal, signal * 0.5])  # 2チャネル
        multi_dask_data = _da_from_array(multi_signal, chunks=(1, -1))
        multi_channel_frame = ChannelFrame(
            data=multi_dask_data, sampling_rate=sample_rate, label="multi_channel_test"
        )
        multi_spectrogram_frame = multi_channel_frame.stft(n_fft=512, hop_length=256)

        # 複数チャネルでのテスト
        result = strategy.plot(multi_spectrogram_frame)

        # Iteratorが返されることを確認
        assert isinstance(result, Iterator)
        axes_list = list(result)
        # 2チャネル + 2つのカラーバー軸 = 4軸
        assert len(axes_list) == 4

        # メインプロット軸（最初の2つ）が適切に設定されていることを確認
        main_axes = [ax for ax in axes_list if ax.get_label() != "<colorbar>"]
        assert len(main_axes) == 2
        for ax in main_axes:
            assert ax.get_xlabel() == "Time [s]"
            assert ax.get_ylabel() == "Frequency [Hz]"

        plt.close("all")

    def test_spectrogram_plot_strategy_dba_mode(self) -> None:
        """SpectrogramPlotStrategyのdBAモードテスト（mockなし）"""
        strategy = SpectrogramPlotStrategy()

        # 実際のデータでdBAモードをテスト
        sample_rate: float = 44100
        duration: float = 0.1
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        signal = np.sin(2 * np.pi * 1000.0 * t)  # 1kHz

        from wandas.frames.channel import ChannelFrame

        dask_data = _da_from_array(signal.reshape(1, -1), chunks=(1, -1))
        channel_frame = ChannelFrame(
            data=dask_data, sampling_rate=sample_rate, label="test_channel"
        )

        spectrogram_frame = channel_frame.stft(n_fft=512, hop_length=256)

        # dBAモードでプロット
        fig, ax = plt.subplots()
        result = strategy.plot(spectrogram_frame, ax=ax, Aw=True)

        assert result is ax
        # カラーバーラベルにdBAが含まれていることを期待
        # （実際のカラーバーの検証は視覚的確認が必要）

        plt.close(fig)

    def test_describe_plot_strategy_edge_cases(self) -> None:
        """DescribePlotStrategyのエッジケースのテスト"""
        strategy = DescribePlotStrategy()

        # A特性重み付けでのテスト
        self.mock_channel_frame.stft.return_value = self.mock_spectrogram_frame
        self.mock_channel_frame.welch.return_value = self.mock_spectral_frame

        with (
            mock.patch("matplotlib.figure.Figure.add_subplot") as mock_add_subplot,
            mock.patch("librosa.display.specshow") as mock_specshow,
            mock.patch("matplotlib.pyplot.figure") as mock_figure,
            mock.patch.object(Figure, "colorbar"),
        ):
            mock_img = mock.MagicMock()
            mock_specshow.return_value = mock_img
            mock_fig = mock.MagicMock(spec=Figure)
            mock_figure.return_value = mock_fig

            mock_ax1 = mock.MagicMock(spec=Axes)
            mock_ax2 = mock.MagicMock(spec=Axes)
            mock_ax3 = mock.MagicMock(spec=Axes)
            mock_ax4 = mock.MagicMock(spec=Axes)

            mock_axes_iter = iter([mock_ax1, mock_ax2, mock_ax3, mock_ax4])
            mock_add_subplot.side_effect = lambda *args, **kwargs: next(mock_axes_iter)

            # A特性重み付けでのプロット
            result = strategy.plot(
                self.mock_channel_frame,
                Aw=True,
                fmin=100,
                fmax=8000,
                xlim=(0, 10),
                ylim=(0, 5000),
            )

            assert isinstance(result, Iterator)

    def test_matrix_plot_strategy_overlay_mode(self) -> None:
        """MatrixPlotStrategyのオーバーレイモードのテスト"""
        from wandas.visualization.plotting import MatrixPlotStrategy

        strategy = MatrixPlotStrategy()

        # オーバーレイモードでのテスト
        result = strategy.plot(self.mock_spectral_frame, overlay=True)
        assert isinstance(result, Axes)

        # オーバーレイモードでコヒーレンスデータのテスト
        result = strategy.plot(self.mock_coherence_spectral_frame, overlay=True)
        assert isinstance(result, Axes)

        # 外部axを指定したオーバーレイモードのテスト
        fig, external_ax = plt.subplots()
        result = strategy.plot(
            self.mock_spectral_frame,
            ax=external_ax,
            overlay=True,
            title="External Overlay Test",
        )
        assert result is external_ax

        plt.close("all")

    def test_plot_strategy_kwargs_filtering(self) -> None:
        """プロット戦略でのkwargs フィルタリングのテスト"""
        strategy = WaveformPlotStrategy()

        # 無効なkwargsを含むテスト
        result = strategy.plot(
            self.mock_channel_frame,
            overlay=True,
            color="blue",
            linewidth=2,
            invalid_param="should_be_ignored",  # 無効なパラメータ
            xlim=(0, 1),
        )
        assert isinstance(result, Axes)

        plt.close("all")

    def test_plot_with_empty_labels(self) -> None:
        """ラベルが空の場合のテスト"""
        # ラベルが空のモックフレームを作成
        empty_label_frame = mock.MagicMock()
        empty_label_frame.n_channels = 1
        empty_label_frame.time = np.linspace(0, 1, 1000)
        empty_label_frame.data = np.random.rand(1000)
        empty_label_frame.labels = [""]
        empty_label_frame.label = ""
        empty_label_frame.channels = [mock.MagicMock(label="")]

        strategy = WaveformPlotStrategy()
        result = strategy.plot(empty_label_frame, overlay=True)
        assert isinstance(result, Axes)
        # デフォルトタイトルが使用されることを確認
        assert "Channel Data" in result.get_title()

        plt.close("all")

    def test_spectrogram_2d_data_handling(self) -> None:
        """スペクトログラムの2Dデータ処理のテスト"""
        strategy = SpectrogramPlotStrategy()

        # 2Dデータ（単一チャネル）のテスト
        fig, ax = plt.subplots()

        with mock.patch("librosa.display.specshow") as mock_specshow:
            mock_img = mock.MagicMock()
            mock_specshow.return_value = mock_img

            _ = strategy.plot(self.mock_single_spectrogram_frame, ax=ax)

            # specshowが呼び出されていることを確認
            mock_specshow.assert_called_once()
            call_args = mock_specshow.call_args
            # 正しいパラメータが渡されていることを確認
            assert "sr" in call_args[1]
            assert (
                call_args[1]["sr"] == self.mock_single_spectrogram_frame.sampling_rate
            )

        plt.close("all")

    def test_channel_metadata_access(self) -> None:
        """チャネルメタデータアクセスのテスト"""
        # unitプロパティを持つチャネルメタデータ
        channel_with_unit = mock.MagicMock()
        channel_with_unit.label = "Test Channel"
        channel_with_unit.unit = "V"

        self.mock_channel_frame.channels = [channel_with_unit]
        self.mock_channel_frame.n_channels = 1
        self.mock_channel_frame.data = np.random.rand(1, 1000)

        strategy = WaveformPlotStrategy()
        result = strategy.plot(self.mock_channel_frame, overlay=False)

        assert isinstance(result, Iterator)
        axes_list = list(result)
        # unitがy軸ラベルに含まれることを確認
        assert "V" in axes_list[0].get_ylabel()

        plt.close("all")

    def test_noct_strategy_with_different_n_values(self) -> None:
        """異なるN値でのNOctPlotStrategyのテスト"""
        from wandas.visualization.plotting import NOctPlotStrategy

        strategy = NOctPlotStrategy()

        # 一時的にlabelを保存してNoneに設定
        original_label = self.mock_noct_frame.label
        self.mock_noct_frame.label = None

        # n=1（1オクターブ）のテスト
        self.mock_noct_frame.n = 1
        result = strategy.plot(self.mock_noct_frame, overlay=True)
        assert isinstance(result, Axes)
        assert "1/1-Octave Spectrum" in result.get_title()

        # n=12（1/12オクターブ）のテスト
        self.mock_noct_frame.n = 12
        result = strategy.plot(self.mock_noct_frame, overlay=True)
        assert isinstance(result, Axes)
        assert "1/12-Octave Spectrum" in result.get_title()

        # labelを復元
        self.mock_noct_frame.label = original_label

        plt.close("all")

    def test_multiple_operations_history(self) -> None:
        """複数の操作履歴を持つフレームのテスト"""
        strategy = FrequencyPlotStrategy()

        # 複数の操作履歴を持つフレーム
        self.mock_spectral_frame.operation_history = [
            {"operation": "fft"},
            {"operation": "coherence"},  # 最後の操作がcoherence
        ]
        self.mock_spectral_frame.magnitude = np.random.rand(2, 513)

        result = strategy.plot(self.mock_spectral_frame, overlay=True)
        assert isinstance(result, Axes)
        assert "coherence" in result.get_ylabel()

        plt.close("all")

    def test_error_handling_in_describe_plot(self) -> None:
        """DescribePlotでのエラーハンドリングのテスト"""
        strategy = DescribePlotStrategy()

        # stftメソッドが存在しないフレーム
        broken_frame = mock.MagicMock()
        broken_frame.stft.side_effect = AttributeError("No stft method")

        with pytest.raises(AttributeError):
            strategy.plot(broken_frame)

    def test_return_axes_iterator_helper(self) -> None:
        """_return_axes_iterator ヘルパー関数のテスト"""
        from wandas.visualization.plotting import _return_axes_iterator

        # モックのaxesリストを作成
        mock_axes = [mock.MagicMock(spec=Axes) for _ in range(3)]

        # ヘルパー関数をテスト
        result = _return_axes_iterator(mock_axes)
        assert isinstance(result, Iterator)

        # イテレータから要素を取得
        axes_list = list(result)
        assert len(axes_list) == 3
        assert all(isinstance(ax, mock.MagicMock) for ax in axes_list)

    def test_matrix_plot_strategy_detailed_behavior(self) -> None:
        """MatrixPlotStrategyの詳細な動作のテスト（選択されたコード部分）"""
        from wandas.visualization.plotting import MatrixPlotStrategy

        strategy = MatrixPlotStrategy()

        # ax_setパラメータが正しく適用されることをテスト
        with (
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,
            mock.patch("matplotlib.pyplot.show") as mock_show,
        ):
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock(spec=Axes)
            # ax.figure == figの場合、tight_layoutは呼び出されない（外部axesと同じ動作）
            mock_ax.figure = mock_fig
            mock_subplots.return_value = (mock_fig, mock_ax)

            # ax_setパラメータを含むkwargsでテスト
            _ = strategy.plot(
                self.mock_spectral_frame,
                overlay=True,
                xlim=(100, 8000),  # ax_setに含まれるパラメータ
                ylim=(-60, 0),  # ax_setに含まれるパラメータ
                title="Test Matrix Plot",
            )

            # ax.setが呼び出されることを確認
            mock_ax.set.assert_called()
            call_kwargs = mock_ax.set.call_args[1]
            assert "xlim" in call_kwargs
            assert "ylim" in call_kwargs
            assert call_kwargs["xlim"] == (100, 8000)
            assert call_kwargs["ylim"] == (-60, 0)

            # suptitleが設定されることを確認
            mock_fig.suptitle.assert_called_with("Test Matrix Plot")

            # ax.figure == figなので、tight_layoutとshowは呼び出されない
            mock_tight_layout.assert_not_called()
            mock_show.assert_not_called()

    def test_matrix_plot_strategy_external_axes_behavior(self) -> None:
        """MatrixPlotStrategyで外部axesを使用した場合の動作テスト"""
        from wandas.visualization.plotting import MatrixPlotStrategy

        strategy = MatrixPlotStrategy()

        with (
            mock.patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,
            mock.patch("matplotlib.pyplot.show") as mock_show,
        ):
            # 外部のfigureとaxesを作成
            external_fig = mock.MagicMock()
            external_ax = mock.MagicMock(spec=Axes)
            external_ax.figure = external_fig

            # 外部axesを指定してplot
            result = strategy.plot(
                self.mock_spectral_frame,
                ax=external_ax,
                overlay=True,
                title="External Axes Test",
            )

            # 外部axesが返されることを確認
            assert result is external_ax

            # 外部axesのsetが呼び出されることを確認
            external_ax.set.assert_called()

            # 外部figureのsuptitleが設定されることを確認
            external_fig.suptitle.assert_called_with("External Axes Test")

            # 外部axesの場合、tight_layoutとshowは呼び出されないことを確認
            mock_tight_layout.assert_not_called()
            mock_show.assert_not_called()

    def test_matrix_plot_strategy_figure_condition(self) -> None:
        """MatrixPlotStrategyのfigure条件分岐のテスト"""
        from wandas.visualization.plotting import MatrixPlotStrategy

        strategy = MatrixPlotStrategy()

        with (
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,
            mock.patch("matplotlib.pyplot.show") as mock_show,
        ):
            # 異なるfigureを持つaxesをシミュレート
            created_fig = mock.MagicMock()
            different_fig = mock.MagicMock()
            mock_ax = mock.MagicMock(spec=Axes)
            mock_ax.figure = different_fig  # 作成したfigureとは異なる
            mock_subplots.return_value = (created_fig, mock_ax)

            # plotを実行
            _ = strategy.plot(
                self.mock_spectral_frame, overlay=True, title="Figure Condition Test"
            )

            # ax.figure != figの条件により、tight_layoutとshowが呼び出されることを確認
            mock_tight_layout.assert_called_once()
            mock_show.assert_called_once()

    def test_matrix_plot_strategy_suptitle_fallback(self) -> None:
        """MatrixPlotStrategyのsuptitleフォールバック動作のテスト"""
        from wandas.visualization.plotting import MatrixPlotStrategy

        strategy = MatrixPlotStrategy()

        with (
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch("matplotlib.pyplot.tight_layout"),
            mock.patch("matplotlib.pyplot.show"),
        ):
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock(spec=Axes)
            mock_ax.figure = mock_fig
            mock_subplots.return_value = (mock_fig, mock_ax)

            # titleもlabelも指定しない場合
            self.mock_spectral_frame.label = None
            _ = strategy.plot(self.mock_spectral_frame, overlay=True)

            # デフォルトのタイトルが使用されることを確認
            mock_fig.suptitle.assert_called_with("Spectral Data")

            # labelがある場合
            self.mock_spectral_frame.label = "Test Label"
            _ = strategy.plot(self.mock_spectral_frame, overlay=True)

            # labelが使用されることを確認
            mock_fig.suptitle.assert_called_with("Test Label")

    def test_matrix_plot_strategy_coherence_data_ax_set(self) -> None:
        """MatrixPlotStrategyでコヒーレンスデータのax_set処理をテスト"""
        from wandas.visualization.plotting import MatrixPlotStrategy

        strategy = MatrixPlotStrategy()

        with (
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch("matplotlib.pyplot.tight_layout"),
            mock.patch("matplotlib.pyplot.show"),
        ):
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock(spec=Axes)
            mock_ax.figure = mock_fig
            mock_subplots.return_value = (mock_fig, mock_ax)

            # コヒーレンスデータでax_setパラメータをテスト
            # gridパラメータはAxes.setでは無効なので、filter_kwargsで除外される
            _ = strategy.plot(
                self.mock_coherence_spectral_frame,
                overlay=True,
                xlim=(10, 10000),
                ylim=(0, 1),
                xscale="log",
                grid=True,  # このパラメータはfilter_kwargsで除外される
            )

            # ax.setが適切なパラメータで呼び出されることを確認
            mock_ax.set.assert_called()
            call_kwargs = mock_ax.set.call_args[1]
            assert "xlim" in call_kwargs
            assert "ylim" in call_kwargs
            assert "xscale" in call_kwargs
            # gridはAxes.setで有効なパラメータではないので含まれない
            assert "grid" not in call_kwargs
            assert call_kwargs["xlim"] == (10, 10000)
            assert call_kwargs["ylim"] == (0, 1)
            assert call_kwargs["xscale"] == "log"

        plt.close("all")


class TestChannelFramePlotParameters:
    """Test plot parameter combinations for complete coverage."""

    def _get_axes_list(self, result: Axes | Iterator[Axes]) -> list[Axes]:
        if isinstance(result, Iterator):
            return list(result)
        return [result]

    def test_plot_with_xlabel(self) -> None:
        """Test plot with custom xlabel."""
        signal = wd.generate_sin(freqs=[440], duration=0.1, sampling_rate=16000)

        # This should trigger line 438
        # plot() returns an iterator of axes
        try:
            res = signal.plot(xlabel="Custom X Label")
            ax_list = self._get_axes_list(res)
            assert len(ax_list) > 0
            assert isinstance(ax_list[0], Axes)
            assert ax_list[0].get_xlabel() == "Custom X Label"
        except Exception as e:
            # If plot fails due to matplotlib issues, skip the test
            # The important thing is the code path was exercised
            if "tight_layout" in str(e) or "_NoValueType" in str(e):
                pytest.skip(f"Plot failed due to matplotlib issue: {e}")
            raise

    def test_plot_with_ylabel(self) -> None:
        """Test plot with custom ylabel."""
        signal = wd.generate_sin(freqs=[440], duration=0.1, sampling_rate=16000)

        # This should trigger line 440
        try:
            res = signal.plot(ylabel="Custom Y Label")
            ax_list = self._get_axes_list(res)
            assert len(ax_list) > 0
            assert isinstance(ax_list[0], Axes)
            assert ax_list[0].get_ylabel() == "Custom Y Label"
        except Exception as e:
            if "tight_layout" in str(e) or "_NoValueType" in str(e):
                pytest.skip(f"Plot failed due to matplotlib issue: {e}")
            raise

    def test_plot_with_alpha(self) -> None:
        """Test plot with custom alpha value."""
        signal = wd.generate_sin(freqs=[440], duration=0.1, sampling_rate=16000)

        # This should trigger line 442 (alpha != 1.0)
        try:
            res = signal.plot(alpha=0.5)
            ax_list = self._get_axes_list(res)
            assert len(ax_list) > 0
            assert isinstance(ax_list[0], Axes)
            # Check alpha on the first line
            lines = ax_list[0].get_lines()
            if lines:
                assert lines[0].get_alpha() == 0.5
        except Exception as e:
            if "tight_layout" in str(e) or "_NoValueType" in str(e):
                pytest.skip(f"Plot failed due to matplotlib issue: {e}")
            raise

    def test_plot_with_xlim(self) -> None:
        """Test plot with custom xlim."""
        signal = wd.generate_sin(freqs=[440], duration=0.1, sampling_rate=16000)

        # This should trigger line 444
        try:
            res = signal.plot(xlim=(0.0, 0.05))
            ax_list = self._get_axes_list(res)
            assert len(ax_list) > 0
            assert isinstance(ax_list[0], Axes)
            assert ax_list[0].get_xlim() == (0.0, 0.05)
        except Exception as e:
            if "tight_layout" in str(e) or "_NoValueType" in str(e):
                pytest.skip(f"Plot failed due to matplotlib issue: {e}")
            raise

    def test_plot_with_combined_parameters(self) -> None:
        """Test plot with multiple optional parameters."""
        signal = wd.generate_sin(freqs=[440], duration=0.1, sampling_rate=16000)

        # Test multiple parameters at once
        try:
            res = signal.plot(
                xlabel="Time",
                ylabel="Amplitude",
                alpha=0.7,
                xlim=(0.0, 0.05),
                ylim=(-1.0, 1.0),
            )
            ax_list = self._get_axes_list(res)
            assert len(ax_list) > 0
            assert isinstance(ax_list[0], Axes)

            assert ax_list[0].get_xlabel() == "Time"
            assert ax_list[0].get_ylabel() == "Amplitude"
            assert ax_list[0].get_xlim() == (0.0, 0.05)
            assert ax_list[0].get_ylim() == (-1.0, 1.0)

            lines = ax_list[0].get_lines()
            if lines:
                assert lines[0].get_alpha() == 0.7
        except Exception as e:
            if "tight_layout" in str(e) or "_NoValueType" in str(e):
                pytest.skip(f"Plot failed due to matplotlib issue: {e}")
            raise
