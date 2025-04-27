from collections.abc import Iterator
from typing import Any, Optional, Union
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

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
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        if ax is None:
            fig, ax = plt.subplots()
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
        self.mock_single_spectrogram_frame.shape = (1, 513, 10)
        self.mock_single_spectrogram_frame.sampling_rate = 44100
        self.mock_single_spectrogram_frame.n_fft = 1024
        self.mock_single_spectrogram_frame.hop_length = 512
        self.mock_single_spectrogram_frame.win_length = 1024
        self.mock_single_spectrogram_frame.dB = np.random.rand(1, 513, 10)
        self.mock_single_spectrogram_frame.dBA = np.random.rand(1, 513, 10)
        self.mock_single_spectrogram_frame.channels = [
            mock.MagicMock(label="ch1"),
        ]
        self.mock_single_spectrogram_frame.label = "Test Single Spectrogram"

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

    def test_spectrogram_plot_strategy(self) -> None:
        """SpectrogramPlotStrategyのテスト"""
        strategy = SpectrogramPlotStrategy()

        # オーバーレイモードはサポートされていない
        with pytest.raises(ValueError, match="Overlay is not supported"):
            strategy.plot(self.mock_spectrogram_frame, overlay=True)

        # テスト1: 単一チャネルのスペクトログラムフレームでのテスト
        with (
            mock.patch("librosa.display.specshow") as mock_specshow,
            mock.patch("matplotlib.figure.Figure.colorbar") as mock_colorbar,
        ):
            # モックの戻り値を適切に設定
            mock_img = mock.MagicMock(spec=QuadMesh)
            mock_specshow.return_value = mock_img

            mock_cbar = mock.MagicMock(spec=Colorbar)
            mock_colorbar.return_value = mock_cbar

            fig, ax = plt.subplots()

            # Python 3.9のmatplotlibでは、
            # axの図形プロパティへのアクセス方法が異なるため、
            # モックを使わずに実際の図と軸を使用するように修正
            result = strategy.plot(self.mock_single_spectrogram_frame, ax=ax)

            # 戻り値が単一のAxesであることを確認
            assert result is ax

            # specshowが呼び出されたことを確認
            mock_specshow.assert_called_once()

            # カラーバーの作成が呼び出されたことを確認
            mock_colorbar.assert_called_once()

        # テスト2: チャネル数が1より大きい場合、axを指定するとエラー
        with mock.patch("librosa.display.specshow") as mock_specshow:
            fig, ax = plt.subplots()
            with pytest.raises(ValueError, match="ax must be None when n_channels > 1"):
                strategy.plot(self.mock_spectrogram_frame, ax=ax)

        # テスト3: 複数チャネルでのテスト（axなし）
        with (
            mock.patch("librosa.display.specshow") as mock_specshow,
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch("matplotlib.figure.Figure.colorbar") as mock_colorbar,
        ):
            # モックの戻り値を設定
            mock_img = mock.MagicMock(spec=QuadMesh)
            mock_specshow.return_value = mock_img

            mock_cbar = mock.MagicMock(spec=Colorbar)
            mock_colorbar.return_value = mock_cbar

            # 図と軸のモックを設定
            mock_fig = mock.MagicMock(spec=Figure)
            mock_axs = []
            for i in range(self.mock_spectrogram_frame.n_channels):
                mock_ax = mock.MagicMock(spec=Axes)
                # Python 3.9対応: figureプロパティのモックを設定せず、
                # get_figure()メソッドを使う
                mock_ax.get_figure.return_value = mock_fig
                mock_ax.figure = mock_fig
                mock_axs.append(mock_ax)

            if len(mock_axs) == 1:
                mock_subplots.return_value = (mock_fig, mock_axs[0])
            else:
                mock_subplots.return_value = (mock_fig, mock_axs)

            # 複数チャネルでのプロット実行
            result = strategy.plot(self.mock_spectrogram_frame)

            # 結果がAxesのイテレータであることを確認
            assert isinstance(result, Iterator)
            axes_list = list(result)
            assert len(axes_list) == self.mock_spectrogram_frame.n_channels

            # 各チャネルに対してspecshowが呼び出されることを確認
            assert mock_specshow.call_count == self.mock_spectrogram_frame.n_channels

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
