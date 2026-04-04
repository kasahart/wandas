import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure

import wandas as wd
from wandas.core.metadata import ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.visualization.plotting import (
    DescribePlotStrategy,
    FrequencyPlotStrategy,
    MatrixPlotStrategy,
    NOctPlotStrategy,
    PlotStrategy,
    SpectrogramPlotStrategy,
    WaveformPlotStrategy,
    _plot_strategies,
    _reshape_spectrogram_data,
    _reshape_to_2d,
    _resolve_channel_label,
    _return_axes_iterator,
    create_operation,
    get_plot_strategy,
    register_plot_strategy,
)

_da_from_array = da.from_array

# ---------------------------------------------------------------------------
# Module-level constants — eliminate magic numbers
# ---------------------------------------------------------------------------
_N_SAMPLES = 1000  # common sample count for mock data
_N_FREQ_BINS = 513  # n_fft=1024 → N/2+1
_N_SPEC_TIME = 10  # spectrogram time frames
_SR_CD = 44100  # CD-quality sampling rate
_N_FFT = 1024
_HOP_LENGTH = 512
_WIN_LENGTH = 1024


class TestPlotStrategy(PlotStrategy[Any]):
    """Stub PlotStrategy for registry tests."""

    name = "test_strategy"

    def channel_plot(self, x: Any, y: Any, ax: "Axes", label: str | None = None, alpha: float = 1.0) -> None:
        pass

    def plot(
        self,
        bf: Any,
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        if ax is None:
            fig, created_ax = plt.subplots()
            return created_ax
        return ax


class TestPlotting:
    """Test class for plotting functionality."""

    def setup_method(self) -> None:
        """Set up mock frames before each test."""
        # Save existing strategy registry for restoration after each test
        self.original_strategies = _plot_strategies.copy()

        # Create mock frames with deterministic data (no np.random)
        _t = np.linspace(0, 1, _N_SAMPLES, endpoint=False)
        _ch0 = np.sin(2 * np.pi * 100 * _t)  # 100 Hz pure sine
        _ch1 = np.cos(2 * np.pi * 200 * _t)  # 200 Hz pure cosine

        self.mock_channel_frame = mock.MagicMock()
        self.mock_channel_frame.n_channels = 2
        self.mock_channel_frame.time = _t
        self.mock_channel_frame.data = np.stack([_ch0, _ch1], axis=0)
        self.mock_channel_frame.labels = ["ch1", "ch2"]
        self.mock_channel_frame.label = "Test Channel"
        self.mock_channel_frame.channels = [
            mock.MagicMock(label="ch1"),
            mock.MagicMock(label="ch2"),
        ]

        # Single-channel mock channel frame
        self.mock_single_channel_frame = mock.MagicMock()
        self.mock_single_channel_frame.n_channels = 1
        self.mock_single_channel_frame.time = _t
        self.mock_single_channel_frame.data = _ch0
        self.mock_single_channel_frame.labels = ["ch1"]
        self.mock_single_channel_frame.label = "Test Single Channel"
        self.mock_single_channel_frame.channels = [
            mock.MagicMock(label="ch1"),
        ]

        # Spectral frame mock -- deterministic data
        _freqs = np.linspace(0, 22050, _N_FREQ_BINS)
        _spec_ch0 = np.sin(np.linspace(0, np.pi, _N_FREQ_BINS))
        _spec_ch1 = np.cos(np.linspace(0, np.pi, _N_FREQ_BINS))

        self.mock_spectral_frame = mock.MagicMock()
        self.mock_spectral_frame.n_channels = 2
        self.mock_spectral_frame.freqs = _freqs
        self.mock_spectral_frame.dB = np.stack([_spec_ch0, _spec_ch1], axis=0)
        self.mock_spectral_frame.dBA = np.stack([_spec_ch0 * 0.8, _spec_ch1 * 0.8], axis=0)
        self.mock_spectral_frame.labels = ["ch1", "ch2"]
        self.mock_spectral_frame.label = "Test Spectral"
        self.mock_spectral_frame.channels = [
            mock.MagicMock(label="ch1"),
            mock.MagicMock(label="ch2"),
        ]

        # Single-channel mock spectral frame
        self.mock_single_spectral_frame = mock.MagicMock()
        self.mock_single_spectral_frame.n_channels = 1
        self.mock_single_spectral_frame.freqs = _freqs
        self.mock_single_spectral_frame.dB = _spec_ch0
        self.mock_single_spectral_frame.dBA = _spec_ch0 * 0.8
        self.mock_single_spectral_frame.labels = ["ch1"]
        self.mock_single_spectral_frame.label = "Test Single Spectral"
        self.mock_single_spectral_frame.channels = [
            mock.MagicMock(label="ch1"),
        ]

        # NOctFrame mock -- deterministic data
        _noct_freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
        _n_noct_bins = len(_noct_freqs)
        _noct_ch0 = np.linspace(40, 80, _n_noct_bins)  # monotonic dB values
        _noct_ch1 = np.linspace(35, 75, _n_noct_bins)

        self.mock_noct_frame = mock.MagicMock()
        self.mock_noct_frame.n_channels = 2
        self.mock_noct_frame.n = 3  # 1/3-octave
        self.mock_noct_frame.freqs = _noct_freqs
        self.mock_noct_frame.dB = np.stack([_noct_ch0, _noct_ch1], axis=0)
        self.mock_noct_frame.dBA = np.stack([_noct_ch0 * 0.9, _noct_ch1 * 0.9], axis=0)
        self.mock_noct_frame.labels = ["ch1", "ch2"]
        self.mock_noct_frame.label = "Test NOct"
        self.mock_noct_frame.channels = [
            mock.MagicMock(label="ch1"),
            mock.MagicMock(label="ch2"),
        ]

        # Single-channel NOctFrame mock
        self.mock_single_noct_frame = mock.MagicMock()
        self.mock_single_noct_frame.n_channels = 1
        self.mock_single_noct_frame.n = 3  # 1/3-octave
        self.mock_single_noct_frame.freqs = _noct_freqs
        self.mock_single_noct_frame.dB = _noct_ch0
        self.mock_single_noct_frame.dBA = _noct_ch0 * 0.9
        self.mock_single_noct_frame.labels = ["ch1"]
        self.mock_single_noct_frame.label = "Test Single NOct"
        self.mock_single_noct_frame.channels = [
            mock.MagicMock(label="ch1"),
        ]

        # Spectrogram frame mock -- deterministic data
        _spec_grid = np.outer(
            np.sin(np.linspace(0, np.pi, _N_FREQ_BINS)),
            np.linspace(0.5, 1.0, _N_SPEC_TIME),
        )  # (_N_FREQ_BINS, _N_SPEC_TIME) deterministic

        self.mock_spectrogram_frame = mock.MagicMock()
        self.mock_spectrogram_frame.n_channels = 2
        self.mock_spectrogram_frame.n_freq_bins = _N_FREQ_BINS
        self.mock_spectrogram_frame.shape = (2, _N_FREQ_BINS, _N_SPEC_TIME)
        self.mock_spectrogram_frame.sampling_rate = _SR_CD
        self.mock_spectrogram_frame.n_fft = _N_FFT
        self.mock_spectrogram_frame.hop_length = _HOP_LENGTH
        self.mock_spectrogram_frame.win_length = _WIN_LENGTH
        self.mock_spectrogram_frame.dB = np.stack([_spec_grid, _spec_grid * 0.8], axis=0)
        self.mock_spectrogram_frame.dBA = np.stack([_spec_grid * 0.9, _spec_grid * 0.7], axis=0)
        self.mock_spectrogram_frame.channels = [
            mock.MagicMock(label="ch1"),
            mock.MagicMock(label="ch2"),
        ]
        self.mock_spectrogram_frame.label = "Test Spectrogram"

        # Single-channel spectrogram frame mock
        self.mock_single_spectrogram_frame = mock.MagicMock()
        self.mock_single_spectrogram_frame.n_channels = 1
        self.mock_single_spectrogram_frame.n_freq_bins = _N_FREQ_BINS
        self.mock_single_spectrogram_frame.shape = (_N_FREQ_BINS, _N_SPEC_TIME)
        self.mock_single_spectrogram_frame.sampling_rate = _SR_CD
        self.mock_single_spectrogram_frame.n_fft = _N_FFT
        self.mock_single_spectrogram_frame.hop_length = _HOP_LENGTH
        self.mock_single_spectrogram_frame.win_length = _WIN_LENGTH
        self.mock_single_spectrogram_frame.dB = _spec_grid
        self.mock_single_spectrogram_frame.dBA = _spec_grid * 0.9
        self.mock_single_spectrogram_frame.channels = [
            mock.MagicMock(label="ch1"),
        ]
        self.mock_single_spectrogram_frame.label = "Test Single Spectrogram"

        # Coherence data -- deterministic sine pattern in [0, 1]
        _coh_single = 0.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, _N_FREQ_BINS))

        # Single-channel coherence data (auto-correlation)
        self.mock_single_coherence_spectral_frame = mock.MagicMock()
        self.mock_single_coherence_spectral_frame.n_channels = 1
        self.mock_single_coherence_spectral_frame.freqs = _freqs
        self.mock_single_coherence_spectral_frame.magnitude = _coh_single
        self.mock_single_coherence_spectral_frame.labels = ["ch1-ch1"]
        self.mock_single_coherence_spectral_frame.label = "Single Coherence Data"
        self.mock_single_coherence_spectral_frame.operation_history = [{"operation": "coherence"}]
        self.mock_single_coherence_spectral_frame.channels = [mock.MagicMock(label="ch1-ch1")]

        # 4-channel coherence data
        _coh_4ch = np.stack(
            [
                _coh_single,
                np.roll(_coh_single, 128),
                np.roll(_coh_single, 256),
                np.roll(_coh_single, 384),
            ],
            axis=0,
        )

        self.mock_coherence_spectral_frame = mock.MagicMock()
        self.mock_coherence_spectral_frame.n_channels = 4
        self.mock_coherence_spectral_frame.freqs = _freqs
        self.mock_coherence_spectral_frame.magnitude = _coh_4ch
        self.mock_coherence_spectral_frame.labels = [
            "ch1-ch1",
            "ch1-ch2",
            "ch2-ch1",
            "ch2-ch2",
        ]
        self.mock_coherence_spectral_frame.label = "Coherence Data"
        self.mock_coherence_spectral_frame.operation_history = [{"operation": "coherence"}]
        self.mock_coherence_spectral_frame.channels = [
            mock.MagicMock(label=label) for label in self.mock_coherence_spectral_frame.labels
        ]

    def teardown_method(self) -> None:
        """Restore strategy registry after each test.
        Figure cleanup is handled by the conftest.py cleanup_plots fixture."""
        _plot_strategies.clear()
        _plot_strategies.update(self.original_strategies)

    def test_plot_strategy_registry(self) -> None:
        """Test plot strategy registration and retrieval."""
        # Verify default strategies are registered
        strategies = ["waveform", "frequency", "spectrogram", "describe"]
        for name in strategies:
            strategy_cls = get_plot_strategy(name)
            assert strategy_cls.name == name

        # Attempting to get a nonexistent strategy raises an error
        with pytest.raises(ValueError, match="Unknown plot type"):
            get_plot_strategy("nonexistent_strategy")

        # Register a new strategy
        register_plot_strategy(TestPlotStrategy)
        strategy_cls = get_plot_strategy("test_strategy")
        assert strategy_cls.name == "test_strategy"
        assert strategy_cls is TestPlotStrategy

    def test_register_invalid_strategy(self) -> None:
        """Test registration of invalid plot strategies."""

        # Class that does not inherit from PlotStrategy
        class InvalidStrategy:
            name = "invalid"

        with pytest.raises(TypeError, match="must inherit from PlotStrategy"):
            register_plot_strategy(InvalidStrategy)

        # Abstract class
        class AbstractStrategy(PlotStrategy[Any]):
            name = "abstract"

        with pytest.raises(TypeError, match="Cannot register abstract PlotStrategy class"):
            register_plot_strategy(AbstractStrategy)

    def test_create_operation(self) -> None:
        """Test create_operation function."""
        # Create a valid operation
        op = create_operation("waveform")
        assert isinstance(op, WaveformPlotStrategy)

        # Create operation with additional parameters (ignored by constructor)
        # Verifies that the no-arg constructor is used
        op_with_params = create_operation("frequency")
        assert isinstance(op_with_params, FrequencyPlotStrategy)

        # Attempting to create a nonexistent operation raises an error
        with pytest.raises(ValueError, match="Unknown plot type"):
            create_operation("nonexistent_operation")

    def test_waveform_plot_strategy(self) -> None:
        """Test WaveformPlotStrategy."""
        strategy = WaveformPlotStrategy()

        # Test channel_plot
        fig, ax = plt.subplots()
        strategy.channel_plot(self.mock_channel_frame.time, self.mock_channel_frame.data[0], ax)
        assert ax.get_ylabel() == "Amplitude"

        # Test plot with single channel (overlay=True)
        result = strategy.plot(self.mock_channel_frame, overlay=True)
        assert isinstance(result, Axes)

        # Test plot with multiple channels (overlay=False)
        result = strategy.plot(self.mock_channel_frame, overlay=False)
        assert isinstance(result, Iterator)

    def test_single_channel_waveform_plot_strategy(self) -> None:
        """Test single-channel WaveformPlotStrategy."""
        strategy = WaveformPlotStrategy()

        # Test single-channel channel_plot with label
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_single_channel_frame.time,
            self.mock_single_channel_frame.data,
            ax,
            label="Test Single Channel",
        )
        assert ax.get_ylabel() == "Amplitude"
        # Verify legend is displayed
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) > 0
        assert legend.get_texts()[0].get_text() == "Test Single Channel"

        # Test single-channel plot (overlay=True)
        result = strategy.plot(self.mock_single_channel_frame, overlay=True)
        assert isinstance(result, Axes)
        assert result.get_title() == "Test Single Channel"

        # Test single-channel plot (overlay=False)
        result = strategy.plot(self.mock_single_channel_frame, overlay=False)
        # Single channel with overlay=False still returns an Iterator
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 1  # single channel produces exactly 1 axis
        assert axes_list[0].get_title() == "ch1"

        # Test custom title
        result = strategy.plot(self.mock_single_channel_frame, title="Custom Title")
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 1  # single channel produces exactly 1 axis
        assert axes_list[0].get_title() == "ch1"
        # Verify suptitle
        assert axes_list[0].figure.get_suptitle() == "Custom Title"

    def test_frequency_plot_strategy(self) -> None:
        """Test FrequencyPlotStrategy."""
        strategy = FrequencyPlotStrategy()

        # Test channel_plot
        fig, ax = plt.subplots()
        strategy.channel_plot(self.mock_spectral_frame.freqs, self.mock_spectral_frame.dB[0], ax)

        # Test plot in dB units
        result = strategy.plot(self.mock_spectral_frame, overlay=True)
        assert isinstance(result, Axes)

        # Test plot in dBA units
        result = strategy.plot(self.mock_spectral_frame, overlay=True, Aw=True)
        assert isinstance(result, Axes)

        # Test plot with multiple channels
        result = strategy.plot(self.mock_spectral_frame, overlay=False)
        assert isinstance(result, Iterator)

    def test_single_channel_frequency_plot_strategy(self) -> None:
        """Test single-channel FrequencyPlotStrategy."""
        strategy = FrequencyPlotStrategy()

        # Test channel_plot with label
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_single_spectral_frame.freqs,
            self.mock_single_spectral_frame.dB,
            ax,
            label="Test Single Frequency",
        )
        # Verify legend is displayed
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) > 0
        assert legend.get_texts()[0].get_text() == "Test Single Frequency"

        # Test single-channel plot in dB (overlay=True)
        result = strategy.plot(self.mock_single_spectral_frame, overlay=True)
        assert isinstance(result, Axes)
        assert result.get_title() == "Test Single Spectral"
        assert result.get_xlabel() == "Frequency [Hz]"
        assert result.get_ylabel() == "Spectrum level [dB]"

        # Test single-channel plot in dBA (overlay=True, Aw=True)
        result = strategy.plot(self.mock_single_spectral_frame, overlay=True, Aw=True)
        assert isinstance(result, Axes)
        assert result.get_ylabel() == "Spectrum level [dBA]"

        # Test single-channel plot (overlay=False)
        result = strategy.plot(self.mock_single_spectral_frame, overlay=False)
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 1  # single channel produces exactly 1 axis
        assert axes_list[0].get_title() == "ch1"

        # Test custom title
        result = strategy.plot(self.mock_single_spectral_frame, title="Custom Title")
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert axes_list[0].get_title() == "ch1"
        assert axes_list[0].figure.get_suptitle() == "Custom Title"

    def test_spectrogram_plot_strategy(self) -> None:
        """Test SpectrogramPlotStrategy."""
        strategy = SpectrogramPlotStrategy()

        # Overlay mode is not supported
        with pytest.raises(ValueError, match="Overlay is not supported"):
            strategy.plot(self.mock_spectrogram_frame, overlay=True)

        # Test 1: single-channel spectrogram frame
        fig, ax = plt.subplots()
        result = strategy.plot(self.mock_single_spectrogram_frame, ax=ax)

        # Verify return value is a single Axes
        assert isinstance(result, Axes)
        assert result is ax
        assert result.get_xlabel() == "Time [s]"
        assert result.get_ylabel() == "Frequency [Hz]"

        # Test 2: providing ax when n_channels > 1 raises an error
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="ax must be None when n_channels > 1"):
            strategy.plot(self.mock_spectrogram_frame, ax=ax)

        # Test 3: multi-channel without ax
        result = strategy.plot(self.mock_spectrogram_frame)

        # Verify result is an Iterator of Axes
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == self.mock_spectrogram_frame.n_channels * 2

        # Verify each axis is properly configured
        for ax in axes_list[:2]:
            assert ax.get_xlabel() == "Time [s]"
            assert ax.get_ylabel() == "Frequency [Hz]"

        # Close all figures

    def test_describe_plot_strategy(self) -> None:
        """Test DescribePlotStrategy."""
        strategy = DescribePlotStrategy()

        # Set up mock stft and welch methods
        self.mock_channel_frame.stft.return_value = self.mock_spectrogram_frame
        self.mock_channel_frame.welch.return_value = self.mock_spectral_frame

        # Partially patch Matplotlib methods
        with (
            mock.patch("matplotlib.figure.Figure.add_subplot") as mock_add_subplot,
            mock.patch("librosa.display.specshow") as mock_specshow,
            mock.patch("matplotlib.pyplot.figure") as mock_figure,
            mock.patch.object(Figure, "colorbar"),
        ):
            # Set return value for mock specshow
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

            # Execute plot
            result = strategy.plot(self.mock_channel_frame)

            # Verify return value is an Iterator of Axes
            assert isinstance(result, Iterator)

            # Verify stft and welch methods were called
            self.mock_channel_frame.stft.assert_called_once()
            self.mock_channel_frame.welch.assert_called_once()

    def test_single_channel_describe_plot_strategy(self) -> None:
        """Test single-channel DescribePlotStrategy."""
        strategy = DescribePlotStrategy()

        # Set up single-channel mock stft and welch methods
        self.mock_single_channel_frame.stft.return_value = self.mock_single_spectrogram_frame
        self.mock_single_channel_frame.welch.return_value = self.mock_single_spectral_frame

        # Partially patch Matplotlib methods
        with (
            mock.patch("matplotlib.figure.Figure.add_subplot") as mock_add_subplot,
            mock.patch("librosa.display.specshow") as mock_specshow,
            mock.patch("matplotlib.pyplot.figure") as mock_figure,
            mock.patch.object(Figure, "colorbar"),
        ):
            # Set return value for mock specshow
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

            # Execute plot for single channel
            result = strategy.plot(self.mock_single_channel_frame)

            # Verify return value is an Iterator of Axes
            assert isinstance(result, Iterator)

            # Verify stft and welch are called even for single channel
            self.mock_single_channel_frame.stft.assert_called_once()
            self.mock_single_channel_frame.welch.assert_called_once()

    def test_noct_plot_strategy(self) -> None:
        """Test NOctPlotStrategy."""

        strategy = NOctPlotStrategy()

        # Test channel_plot
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_noct_frame.freqs,
            self.mock_noct_frame.dB[0],
            ax,
            label="Test NOct",
        )
        # Verify step plot is used with grid and legend displayed
        assert len(ax.xaxis.get_gridlines()) > 0  # Verify grid is displayed
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) > 0  # Verify legend is displayed

        # Test single-channel plot (overlay=True)
        result = strategy.plot(self.mock_noct_frame, overlay=True)
        assert isinstance(result, Axes)
        assert result.get_xlabel() == "Center frequency [Hz]"
        assert result.get_ylabel() == "Spectrum level [dBr]"
        assert result.get_title() == "Test NOct"

        # Test plot in dBA (overlay=True, Aw=True)
        result = strategy.plot(self.mock_noct_frame, overlay=True, Aw=True)
        assert isinstance(result, Axes)
        assert result.get_ylabel() == "Spectrum level [dBrA]"

        # Test plot with multiple channels (overlay=False)
        result = strategy.plot(self.mock_noct_frame, overlay=False)
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == self.mock_noct_frame.n_channels

        # Verify xlabel and ylabel on the last axis
        assert axes_list[-1].get_xlabel() == "Center frequency [Hz]"
        assert axes_list[-1].get_ylabel() == "Spectrum level [dBr]"

    def test_single_channel_noct_plot_strategy(self) -> None:
        """Test single-channel NOctPlotStrategy."""

        strategy = NOctPlotStrategy()

        # Test single-channel channel_plot
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_single_noct_frame.freqs,
            self.mock_single_noct_frame.dB,
            ax,
            label="Test Single NOct",
        )
        # Verify plot characteristics
        assert len(ax.xaxis.get_gridlines()) > 0  # Verify grid is displayed
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) > 0  # Verify legend is displayed
        assert legend.get_texts()[0].get_text() == "Test Single NOct"

        # Test single-channel plot (overlay=True)
        result = strategy.plot(self.mock_single_noct_frame, overlay=True)
        assert isinstance(result, Axes)
        assert result.get_xlabel() == "Center frequency [Hz]"
        assert result.get_ylabel() == "Spectrum level [dBr]"
        assert result.get_title() == "Test Single NOct"

        # Test single-channel plot in dBA (overlay=True, Aw=True)
        result = strategy.plot(self.mock_single_noct_frame, overlay=True, Aw=True)
        assert isinstance(result, Axes)
        assert result.get_ylabel() == "Spectrum level [dBrA]"

        # Test single-channel plot (overlay=False)
        result = strategy.plot(self.mock_single_noct_frame, overlay=False)
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 1  # single channel produces exactly 1 axis
        assert axes_list[0].get_xlabel() == "Center frequency [Hz]"
        assert axes_list[0].get_ylabel() == "Spectrum level [dBr]"
        assert axes_list[0].get_title() == "ch1"
        assert axes_list[0].figure.get_suptitle() == "Test Single NOct"
        # Test custom title
        result = strategy.plot(self.mock_single_noct_frame, title="Custom NOct Title")
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert axes_list[0].get_title() == "ch1"
        assert axes_list[0].figure.get_suptitle() == "Custom NOct Title"

    @pytest.mark.parametrize(
        "strategy_cls,frame_attr",
        [
            (NOctPlotStrategy, "mock_noct_frame"),
            (WaveformPlotStrategy, "mock_channel_frame"),
            (FrequencyPlotStrategy, "mock_spectral_frame"),
        ],
        ids=["noct", "waveform", "frequency"],
    )
    def test_custom_label_overlay_and_per_channel(self, strategy_cls: type, frame_attr: str) -> None:
        """Verify user-specified labels appear in overlay and per-channel legends."""
        frame = getattr(self, frame_attr)
        strategy = strategy_cls()

        # overlay=True: single Axes with custom label in legend
        result = strategy.plot(frame, overlay=True, label="Overlay Label")
        assert isinstance(result, Axes)
        legend = result.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "Overlay Label" in legend_texts

        # overlay=False: per-channel Axes each with custom label in legend
        result = strategy.plot(frame, overlay=False, label="Per-Ch Label")
        assert isinstance(result, Iterator)
        axes_list = list(result)
        for ax_i in axes_list:
            assert isinstance(ax_i, Axes)
            legend = ax_i.get_legend()
            assert legend is not None
            legend_texts = [t.get_text() for t in legend.get_texts()]
            assert "Per-Ch Label" in legend_texts

    def test_matrix_plot_strategy_channel_plot(self) -> None:
        """Verify MatrixPlotStrategy.channel_plot sets xlabel, ylabel, title, and grid."""
        strategy = MatrixPlotStrategy()
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_coherence_spectral_frame.freqs,
            self.mock_coherence_spectral_frame.magnitude[0],
            ax,
            title="Test Channel",
            ylabel="Test Label",
        )

        assert ax.get_xlabel() == "Frequency [Hz]"
        assert ax.get_ylabel() == "Test Label"
        assert ax.get_title() == "Test Channel"
        assert len(ax.xaxis.get_gridlines()) > 0

    def test_matrix_plot_strategy_coherence(self) -> None:
        """Verify MatrixPlotStrategy.plot returns per-channel axes with coherence labels."""
        strategy = MatrixPlotStrategy()
        result = strategy.plot(self.mock_coherence_spectral_frame)

        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == self.mock_coherence_spectral_frame.n_channels

        for i, ax in enumerate(axes_list):
            assert ax.get_xlabel() == "Frequency [Hz]"
            assert "coherence" in ax.get_ylabel().lower()
            assert self.mock_coherence_spectral_frame.labels[i] in ax.get_title()

    def test_matrix_plot_strategy_aw(self) -> None:
        """Verify MatrixPlotStrategy.plot with Aw=True produces A-weighted labels."""
        strategy = MatrixPlotStrategy()
        result = strategy.plot(self.mock_spectral_frame, Aw=True)

        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 4

        for i, ax in enumerate(axes_list[:2]):
            assert ax.get_xlabel() == "Frequency [Hz]"
            assert "dBA" in ax.get_ylabel() or "A-weighted" in ax.get_ylabel()
            assert self.mock_spectral_frame.labels[i] in ax.get_title()

    def test_single_channel_matrix_plot_strategy(self) -> None:
        """Test single-channel MatrixPlotStrategy."""

        strategy = MatrixPlotStrategy()

        # Test single-channel channel_plot
        fig, ax = plt.subplots()
        strategy.channel_plot(
            self.mock_single_coherence_spectral_frame.freqs,
            self.mock_single_coherence_spectral_frame.magnitude,
            ax,
            title="Single Coherence Test",
            ylabel="Magnitude",
            label="ch1-ch1",
        )

        # Verify correct labels and title are set
        assert ax.get_xlabel() == "Frequency [Hz]"
        assert ax.get_ylabel() == "Magnitude"
        assert ax.get_title() == "Single Coherence Test"
        # Verify grid is displayed
        assert len(ax.xaxis.get_gridlines()) > 0

        # Test single-channel coherence data plot
        result = strategy.plot(self.mock_single_coherence_spectral_frame)

        # Verify result is an Iterator of Axes
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert len(axes_list) == 1  # single channel produces exactly 1 axis

        # Verify axis labels and title
        assert axes_list[0].get_xlabel() == "Frequency [Hz]"
        assert "coherence" in axes_list[0].get_ylabel().lower()
        label = self.mock_single_coherence_spectral_frame.labels[0]
        assert label in axes_list[0].get_title()

        # Test with custom title and unit
        result = strategy.plot(
            self.mock_single_coherence_spectral_frame,
            title="Custom Matrix Title",
            ylabel="Custom Y Units",
        )
        # Verify result is an iterator, then convert to list
        assert isinstance(result, Iterator)
        axes_list = list(result)
        assert axes_list[0].get_title() == "ch1-ch1"
        assert "Custom Y Units" in axes_list[0].get_ylabel()
        assert axes_list[0].figure.get_suptitle() == "Custom Matrix Title"

    def test_waveform_plot_strategy_edge_cases(self) -> None:
        """Test WaveformPlotStrategy edge cases."""
        strategy = WaveformPlotStrategy()

        # Test with custom parameters
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

        # Test with externally provided ax
        fig, external_ax = plt.subplots()
        result = strategy.plot(
            self.mock_channel_frame,
            ax=external_ax,
            overlay=True,
            title="External Ax Test",
        )
        assert result is external_ax
        assert result.get_title() == "External Ax Test"

    def test_frequency_plot_strategy_edge_cases(self) -> None:
        """Test FrequencyPlotStrategy edge cases."""
        strategy = FrequencyPlotStrategy()

        # Test with coherence operation history frame
        self.mock_spectral_frame.operation_history = [{"operation": "coherence"}]
        self.mock_spectral_frame.magnitude = np.abs(self.mock_spectral_frame.dB)

        result = strategy.plot(self.mock_spectral_frame, overlay=True)
        assert isinstance(result, Axes)
        assert "coherence" in result.get_ylabel()

        # Reset operation history
        self.mock_spectral_frame.operation_history = []

        # Test with custom parameters
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

    def test_spectrogram_plot_strategy_edge_cases(self) -> None:
        """Test SpectrogramPlotStrategy edge cases."""
        strategy = SpectrogramPlotStrategy()

        # Test in dBA units
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
            # Verify dBA units are used
            mock_specshow.assert_called_once()
            call_args = mock_specshow.call_args
            assert "cmap" in call_args[1]
            assert call_args[1]["cmap"] == "viridis"

    @staticmethod
    def _make_spectrogram(
        freq_hz: float = 440.0,
        sample_rate: int = 44100,
        duration: float = 0.1,
        n_fft: int = 512,
        hop_length: int = 256,
        label: str = "test_channel",
    ):
        """Create a real SpectrogramFrame from a deterministic sine.

        Used by integration-level tests that need actual STFT data.
        """

        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        signal = np.sin(2 * np.pi * freq_hz * t)
        dask_data = _da_from_array(signal.reshape(1, -1), chunks=(1, -1))
        cf = ChannelFrame(data=dask_data, sampling_rate=sample_rate, label=label)
        return cf, cf.stft(n_fft=n_fft, hop_length=hop_length)

    def test_spectrogram_plot_strategy_basic_functionality(self) -> None:
        """SpectrogramPlotStrategy: real data, single- and multi-channel axes labels."""
        strategy = SpectrogramPlotStrategy()

        # Deterministic 440 Hz sine — analytically predictable
        sample_rate = 44100
        n_fft = 512  # -> N/2+1 = 257 freq bins
        hop_length = 256
        _, spectrogram_frame = self._make_spectrogram(
            freq_hz=440.0, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length
        )

        # --- Single channel with explicit ax ---
        fig, ax = plt.subplots()
        result = strategy.plot(spectrogram_frame, ax=ax)

        # Verify return value is a correct Axes object
        assert result is ax
        assert result.get_xlabel() == "Time [s]"
        assert result.get_ylabel() == "Frequency [Hz]"

        plt.close(fig)

        # --- Multi-channel: 2 channels ---

        n_samples = int(sample_rate * 0.1)
        t = np.linspace(0, 0.1, n_samples, endpoint=False)
        signal = np.sin(2 * np.pi * 440.0 * t)
        multi_signal = np.array([signal, signal * 0.5])
        multi_dask_data = _da_from_array(multi_signal, chunks=(1, -1))
        multi_channel_frame = ChannelFrame(data=multi_dask_data, sampling_rate=sample_rate, label="multi_channel_test")
        multi_spectrogram_frame = multi_channel_frame.stft(n_fft=n_fft, hop_length=hop_length)

        # Test with multiple channels
        result = strategy.plot(multi_spectrogram_frame)

        # Verify Iterator is returned
        assert isinstance(result, Iterator)
        axes_list = list(result)
        # 2 channels + 2 colorbar axes = 4 axes total
        assert len(axes_list) == 4

        # Verify the main plot axes (first 2) are properly configured
        main_axes = [ax for ax in axes_list if ax.get_label() != "<colorbar>"]
        assert len(main_axes) == 2
        for ax in main_axes:
            assert ax.get_xlabel() == "Time [s]"
            assert ax.get_ylabel() == "Frequency [Hz]"

    def test_spectrogram_plot_strategy_dba_mode(self) -> None:
        """SpectrogramPlotStrategy: dBA mode with real data returns same Axes."""
        strategy = SpectrogramPlotStrategy()

        # Deterministic 1 kHz sine — analytically predictable
        _, spectrogram_frame = self._make_spectrogram(freq_hz=1000.0)

        fig, ax = plt.subplots()
        result = strategy.plot(spectrogram_frame, ax=ax, Aw=True)
        assert result is ax

    def test_describe_plot_strategy_edge_cases(self) -> None:
        """Test DescribePlotStrategy edge cases."""
        strategy = DescribePlotStrategy()

        # Test with A-weighting
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

            # Plot with A-weighting
            result = strategy.plot(
                self.mock_channel_frame,
                Aw=True,
                fmin=100,
                fmax=8000,
                xlim=(0, 10),
                ylim=(0, 5000),
            )

            assert isinstance(result, Iterator)

            # Verify stft and welch were invoked
            self.mock_channel_frame.stft.assert_called_once()
            self.mock_channel_frame.welch.assert_called_once()

    def test_matrix_plot_strategy_overlay_mode(self) -> None:
        """Test MatrixPlotStrategy overlay mode."""

        strategy = MatrixPlotStrategy()

        # Test overlay mode
        result = strategy.plot(self.mock_spectral_frame, overlay=True)
        assert isinstance(result, Axes)

        # Test overlay mode with coherence data
        result = strategy.plot(self.mock_coherence_spectral_frame, overlay=True)
        assert isinstance(result, Axes)

        # Test overlay mode with externally provided ax
        fig, external_ax = plt.subplots()
        result = strategy.plot(
            self.mock_spectral_frame,
            ax=external_ax,
            overlay=True,
            title="External Overlay Test",
        )
        assert result is external_ax

    def test_plot_strategy_kwargs_filtering(self) -> None:
        """Test kwargs filtering in plot strategies — valid params applied, invalid ignored."""
        strategy = WaveformPlotStrategy()

        result = strategy.plot(
            self.mock_channel_frame,
            overlay=True,
            color="blue",
            linewidth=2,
            invalid_param="should_be_ignored",  # Invalid parameter filtered out
            xlim=(0, 1),
        )
        assert isinstance(result, Axes)
        # Valid xlim must be applied
        assert result.get_xlim() == pytest.approx((0, 1))

    def test_plot_with_empty_labels(self) -> None:
        """Test behavior when label is empty."""
        # Create a mock frame with an empty label
        empty_label_frame = mock.MagicMock()
        empty_label_frame.n_channels = 1
        empty_label_frame.time = np.linspace(0, 1, _N_SAMPLES)
        empty_label_frame.data = np.sin(np.linspace(0, 2 * np.pi, _N_SAMPLES))
        empty_label_frame.labels = [""]
        empty_label_frame.label = ""
        empty_label_frame.channels = [mock.MagicMock(label="")]

        strategy = WaveformPlotStrategy()
        result = strategy.plot(empty_label_frame, overlay=True)
        assert isinstance(result, Axes)
        # Verify default title is used
        assert "Channel Data" in result.get_title()

    def test_spectrogram_2d_data_handling(self) -> None:
        """Test spectrogram 2D data handling."""
        strategy = SpectrogramPlotStrategy()

        # Test with 2D data (single channel)
        fig, ax = plt.subplots()

        with mock.patch("librosa.display.specshow") as mock_specshow:
            mock_img = mock.MagicMock()
            mock_specshow.return_value = mock_img

            _ = strategy.plot(self.mock_single_spectrogram_frame, ax=ax)

            # Verify specshow was called
            mock_specshow.assert_called_once()
            call_args = mock_specshow.call_args
            # Verify correct parameters were passed
            assert "sr" in call_args[1]
            assert call_args[1]["sr"] == self.mock_single_spectrogram_frame.sampling_rate

    def test_channel_metadata_access(self) -> None:
        """Test channel metadata access."""
        # Channel metadata with unit property
        channel_with_unit = mock.MagicMock()
        channel_with_unit.label = "Test Channel"
        channel_with_unit.unit = "V"

        self.mock_channel_frame.channels = [channel_with_unit]
        self.mock_channel_frame.n_channels = 1
        self.mock_channel_frame.data = np.sin(np.linspace(0, 2 * np.pi, 1000)).reshape(1, -1)

        strategy = WaveformPlotStrategy()
        result = strategy.plot(self.mock_channel_frame, overlay=False)

        assert isinstance(result, Iterator)
        axes_list = list(result)
        # Verify unit is included in y-axis label
        assert "V" in axes_list[0].get_ylabel()

    def test_noct_strategy_with_different_n_values(self) -> None:
        """Test NOctPlotStrategy with different N values."""
        strategy = NOctPlotStrategy()

        # Use a frame with label=None to verify auto-generated title
        self.mock_noct_frame.label = None

        # Test with n=1 (1-octave)
        self.mock_noct_frame.n = 1
        result = strategy.plot(self.mock_noct_frame, overlay=True)
        assert isinstance(result, Axes)
        assert "1/1-Octave Spectrum" in result.get_title()

        # Test with n=12 (1/12-octave)
        self.mock_noct_frame.n = 12
        result = strategy.plot(self.mock_noct_frame, overlay=True)
        assert isinstance(result, Axes)
        assert "1/12-Octave Spectrum" in result.get_title()

    def test_multiple_operations_history(self) -> None:
        """Test frame with multiple operation history entries."""
        strategy = FrequencyPlotStrategy()

        # Frame with multiple operation history entries
        self.mock_spectral_frame.operation_history = [
            {"operation": "fft"},
            {"operation": "coherence"},  # last operation is coherence
        ]
        self.mock_spectral_frame.magnitude = np.abs(self.mock_spectral_frame.dB)

        result = strategy.plot(self.mock_spectral_frame, overlay=True)
        assert isinstance(result, Axes)
        assert "coherence" in result.get_ylabel()

    def test_error_handling_in_describe_plot(self) -> None:
        """Test error handling in DescribePlot."""
        strategy = DescribePlotStrategy()

        # Frame without stft method
        broken_frame = mock.MagicMock()
        broken_frame.stft.side_effect = AttributeError("No stft method")

        with pytest.raises(AttributeError):
            strategy.plot(broken_frame)

    def test_return_axes_iterator_helper(self) -> None:
        """Test _return_axes_iterator helper function."""

        # Create mock axes list
        mock_axes = [mock.MagicMock(spec=Axes) for _ in range(3)]

        # Test the helper function
        result = _return_axes_iterator(mock_axes)
        assert isinstance(result, Iterator)

        # Retrieve elements from iterator
        axes_list = list(result)
        assert len(axes_list) == 3
        assert all(isinstance(ax, mock.MagicMock) for ax in axes_list)

    def test_matrix_plot_strategy_detailed_behavior(self) -> None:
        """Test detailed MatrixPlotStrategy behavior."""
        strategy = MatrixPlotStrategy()

        # Test that ax_set parameters are correctly applied
        with (
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,
            mock.patch("matplotlib.pyplot.show") as mock_show,
        ):
            mock_fig = mock.MagicMock()
            mock_ax = mock.MagicMock(spec=Axes)
            # When ax.figure == fig, tight_layout is not called (same behavior as external axes)
            mock_ax.figure = mock_fig
            mock_subplots.return_value = (mock_fig, mock_ax)

            # Test with kwargs containing ax_set parameters
            _ = strategy.plot(
                self.mock_spectral_frame,
                overlay=True,
                xlim=(100, 8000),  # ax_set parameter
                ylim=(-60, 0),  # ax_set parameter
                title="Test Matrix Plot",
            )

            # Verify ax.set was called
            mock_ax.set.assert_called()
            call_kwargs = mock_ax.set.call_args[1]
            assert "xlim" in call_kwargs
            assert "ylim" in call_kwargs
            assert call_kwargs["xlim"] == (100, 8000)
            assert call_kwargs["ylim"] == (-60, 0)

            # Verify suptitle is set
            mock_fig.suptitle.assert_called_with("Test Matrix Plot")

            # Since ax.figure == fig, tight_layout and show are not called
            mock_tight_layout.assert_not_called()
            mock_show.assert_not_called()

    def test_matrix_plot_strategy_external_axes_behavior(self) -> None:
        """Test MatrixPlotStrategy behavior with external axes."""

        strategy = MatrixPlotStrategy()

        with (
            mock.patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,
            mock.patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Create external figure and axes
            external_fig = mock.MagicMock()
            external_ax = mock.MagicMock(spec=Axes)
            external_ax.figure = external_fig

            # Plot with external axes
            result = strategy.plot(
                self.mock_spectral_frame,
                ax=external_ax,
                overlay=True,
                title="External Axes Test",
            )

            # Verify external axes are returned
            assert result is external_ax

            # Verify set was called on external axes
            external_ax.set.assert_called()

            # Verify suptitle is set on external figure
            external_fig.suptitle.assert_called_with("External Axes Test")

            # Verify tight_layout and show are not called for external axes
            mock_tight_layout.assert_not_called()
            mock_show.assert_not_called()

    def test_matrix_plot_strategy_figure_condition(self) -> None:
        """Test MatrixPlotStrategy figure conditional branching."""

        strategy = MatrixPlotStrategy()

        with (
            mock.patch("matplotlib.pyplot.subplots") as mock_subplots,
            mock.patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,
            mock.patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Simulate axes belonging to a different figure
            created_fig = mock.MagicMock()
            different_fig = mock.MagicMock()
            mock_ax = mock.MagicMock(spec=Axes)
            mock_ax.figure = different_fig  # Different from the created figure
            mock_subplots.return_value = (created_fig, mock_ax)

            # Execute plot
            _ = strategy.plot(self.mock_spectral_frame, overlay=True, title="Figure Condition Test")

            # Since ax.figure != fig, tight_layout and show should be called
            mock_tight_layout.assert_called_once()
            mock_show.assert_called_once()

    def test_matrix_plot_strategy_suptitle_fallback(self) -> None:
        """Test MatrixPlotStrategy suptitle fallback behavior."""

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

            # When neither title nor label is specified
            self.mock_spectral_frame.label = None
            _ = strategy.plot(self.mock_spectral_frame, overlay=True)

            # Verify default title is used
            mock_fig.suptitle.assert_called_with("Spectral Data")

            # When label is present
            self.mock_spectral_frame.label = "Test Label"
            _ = strategy.plot(self.mock_spectral_frame, overlay=True)

            # Verify label is used
            mock_fig.suptitle.assert_called_with("Test Label")

    def test_matrix_plot_strategy_coherence_data_ax_set(self) -> None:
        """Test MatrixPlotStrategy ax_set handling with coherence data."""

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

            # Test ax_set parameters with coherence data
            # grid parameter is invalid for Axes.set, filtered out by filter_kwargs
            _ = strategy.plot(
                self.mock_coherence_spectral_frame,
                overlay=True,
                xlim=(10, 10000),
                ylim=(0, 1),
                xscale="log",
                grid=True,  # This parameter is filtered out by filter_kwargs
            )

            # Verify ax.set is called with appropriate parameters
            mock_ax.set.assert_called()
            call_kwargs = mock_ax.set.call_args[1]
            assert "xlim" in call_kwargs
            assert "ylim" in call_kwargs
            assert "xscale" in call_kwargs
            # grid is not a valid parameter for Axes.set, so it should not be included
            assert "grid" not in call_kwargs
            assert call_kwargs["xlim"] == (10, 10000)
            assert call_kwargs["ylim"] == (0, 1)
            assert call_kwargs["xscale"] == "log"

    def test_plotting_helper_functions_and_noop_methods(self) -> None:
        """Helper utilities and explicit no-op methods should be covered directly."""
        channel_meta = ChannelMetadata(label="default")
        unlabeled_channel_meta = ChannelMetadata()

        assert _resolve_channel_label(None, channel_meta, 0, 2) == "default"
        assert _resolve_channel_label(None, unlabeled_channel_meta, 0, 2) == ""
        assert _resolve_channel_label("shared", channel_meta, 0, 2) == "shared"
        assert _resolve_channel_label(["left", "right"], channel_meta, 1, 2) == "right"
        assert _resolve_channel_label(123, channel_meta, 0, 2) == "123"  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="Channel label count mismatch"):
            _resolve_channel_label(["only-one"], channel_meta, 0, 2)

        one_dimensional = np.arange(4)
        two_dimensional = np.arange(6).reshape(2, 3)
        spectrogram_2d = np.arange(6).reshape(2, 3)
        spectrogram_3d = np.arange(12).reshape(1, 3, 4)

        assert _reshape_to_2d(one_dimensional).shape == (1, 4)
        assert _reshape_to_2d(two_dimensional).shape == (2, 3)
        assert _reshape_spectrogram_data(one_dimensional).shape == (1, 4, 1)
        assert _reshape_spectrogram_data(spectrogram_2d).shape == (1, 2, 3)
        assert _reshape_spectrogram_data(spectrogram_3d).shape == (1, 3, 4)

        dummy_strategy = mock.MagicMock()
        assert PlotStrategy.channel_plot(dummy_strategy, None, None, mock.MagicMock()) is None
        assert PlotStrategy.plot(dummy_strategy, mock.MagicMock()) is None
        assert SpectrogramPlotStrategy().channel_plot(None, None, mock.MagicMock()) is None
        assert DescribePlotStrategy().channel_plot(None, None, mock.MagicMock()) is None

    def test_plotting_module_fallback_import_path(self) -> None:
        """Fallback import should use librosa.display when direct import fails."""
        import wandas.visualization.plotting as plotting_module

        isolated_module = types.ModuleType("wandas.visualization.plotting_fallback_test")
        isolated_module.__file__ = plotting_module.__file__
        isolated_module.__package__ = "wandas.visualization"
        plotting_source = Path(plotting_module.__file__).read_text(encoding="utf-8")
        real_import = __import__

        def import_side_effect(
            name: str,
            globals_: dict[str, Any] | None = None,
            locals_: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] | None = (),
            level: int = 0,
        ) -> Any:
            if name == "librosa" and fromlist and "display" in fromlist:
                raise ImportError("forced display import failure")
            return real_import(name, globals_, locals_, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=import_side_effect):
            exec(compile(plotting_source, plotting_module.__file__, "exec"), isolated_module.__dict__)

        assert isolated_module.display is isolated_module.librosa.display

    def test_spectrogram_plot_strategy_colorbar_error_paths(self) -> None:
        """Spectrogram plotting should swallow colorbar creation errors for both paths."""
        strategy = SpectrogramPlotStrategy()

        fig_single, ax_single = plt.subplots()
        with (
            mock.patch("librosa.display.specshow", return_value=mock.MagicMock()),
            mock.patch.object(
                fig_single,
                "colorbar",
                side_effect=ValueError("bad colorbar"),
            ),
            mock.patch("wandas.visualization.plotting.logger.warning") as mock_warning,
        ):
            result = strategy.plot(self.mock_single_spectrogram_frame, ax=ax_single)
            assert result is ax_single
            mock_warning.assert_called_once()

        plt.close(fig_single)

        fig_multi, axs_multi = plt.subplots(2, 1)
        with (
            mock.patch("matplotlib.pyplot.subplots", return_value=(fig_multi, axs_multi)),
            mock.patch("librosa.display.specshow", return_value=mock.MagicMock()),
            mock.patch.object(
                fig_multi,
                "colorbar",
                side_effect=AttributeError("missing colorbar"),
            ),
            mock.patch("wandas.visualization.plotting.logger.warning") as mock_warning,
            mock.patch("matplotlib.pyplot.show"),
        ):
            result = strategy.plot(self.mock_spectrogram_frame)
            assert isinstance(result, Iterator)
            assert mock_warning.call_count == 2

        plt.close(fig_multi)

    def test_spectrogram_plot_strategy_invalid_figure_type_raises(self) -> None:
        """Spectrogram plotting should reject patched subplots that do not return a Figure."""
        strategy = SpectrogramPlotStrategy()
        mock_ax = mock.MagicMock(spec=Axes)

        with mock.patch(
            "matplotlib.pyplot.subplots",
            return_value=("not-a-figure", np.array([mock_ax])),
        ):
            with pytest.raises(ValueError, match="fig must be a matplotlib Figure object"):
                strategy.plot(self.mock_single_spectrogram_frame)

    def test_matrix_plot_strategy_handles_list_axes_container(self) -> None:
        """Matrix plotting should flatten list-based axes containers."""
        strategy = MatrixPlotStrategy()
        fig, axs = plt.subplots(2, 2)

        with (
            mock.patch("matplotlib.pyplot.subplots", return_value=(fig, axs.tolist())),
            mock.patch("matplotlib.pyplot.tight_layout"),
            mock.patch("matplotlib.pyplot.show"),
        ):
            result = strategy.plot(self.mock_coherence_spectral_frame)
            if isinstance(result, Axes):
                axes_list = [result]
            else:
                axes_list = list(result)

        assert len(axes_list) == 4
        plt.close(fig)


class TestChannelFramePlotParameters:
    """Test plot parameter forwarding for ChannelFrame.plot().

    Visualization policy: verify that xlabel, ylabel, xlim, ylim, alpha
    are correctly reflected in the returned Axes.
    """

    # Deterministic signal constants
    _FREQ_HZ = 440
    _DURATION = 0.1  # 100 ms — short but sufficient for parameter tests
    _SR = 16_000

    def _make_signal(self) -> wd.ChannelFrame:
        return wd.generate_sin(
            freqs=[self._FREQ_HZ],
            duration=self._DURATION,
            sampling_rate=self._SR,
        )

    @staticmethod
    def _get_axes_list(result: Axes | Iterator[Axes]) -> list[Axes]:
        if not isinstance(result, Axes):
            return list(result)
        return [result]

    def test_plot_forwards_xlabel(self) -> None:
        """Custom xlabel must appear on the returned Axes."""
        res = self._make_signal().plot(xlabel="Custom X Label")
        ax = self._get_axes_list(res)[0]
        assert ax.get_xlabel() == "Custom X Label"

    def test_plot_forwards_ylabel(self) -> None:
        """Custom ylabel must appear on the returned Axes."""
        res = self._make_signal().plot(ylabel="Custom Y Label")
        ax = self._get_axes_list(res)[0]
        assert ax.get_ylabel() == "Custom Y Label"

    def test_plot_forwards_alpha(self) -> None:
        """Custom alpha must be applied to drawn Line2D objects."""
        res = self._make_signal().plot(alpha=0.5)
        ax = self._get_axes_list(res)[0]
        lines = ax.get_lines()
        assert len(lines) >= 1, "At least one line should be drawn"
        assert lines[0].get_alpha() == 0.5

    def test_plot_forwards_xlim(self) -> None:
        """Custom xlim must be set on the returned Axes."""
        res = self._make_signal().plot(xlim=(0.0, 0.05))
        ax = self._get_axes_list(res)[0]
        assert ax.get_xlim() == pytest.approx((0.0, 0.05))

    def test_plot_forwards_combined_parameters(self) -> None:
        """Multiple parameters applied together must all be reflected."""
        res = self._make_signal().plot(
            xlabel="Time",
            ylabel="Amplitude",
            alpha=0.7,
            xlim=(0.0, 0.05),
            ylim=(-1.0, 1.0),
        )
        ax = self._get_axes_list(res)[0]

        assert ax.get_xlabel() == "Time"
        assert ax.get_ylabel() == "Amplitude"
        assert ax.get_xlim() == pytest.approx((0.0, 0.05))
        assert ax.get_ylim() == pytest.approx((-1.0, 1.0))

        lines = ax.get_lines()
        if lines:
            assert lines[0].get_alpha() == 0.7


def test_spectrogram_plot_single_channel_scalar_axes_converted() -> None:
    """SpectrogramPlotStrategy: single-channel with ax=None triggers scalar→array conversion."""
    sr = 16_000
    n_fft = 512
    hop = n_fft // 4  # 128
    freq_hz = 440  # A4 — deterministic sine
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * freq_hz * t).reshape(1, -1)
    cf = wd.ChannelFrame.from_numpy(data, sampling_rate=sr)
    spec = cf.stft(n_fft=n_fft, hop_length=hop)

    strategy = SpectrogramPlotStrategy()
    result = strategy.plot(spec)
    # Single-channel without ax= returns an iterator (scalar Axes is wrapped internally)
    assert isinstance(result, Iterator)
    axes_list = list(result)
    assert len(axes_list) >= 1  # at least 1 axis for single channel
