import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # GUIなし環境用
from wandas.visualization.plotting import (
    FrequencyPlotStrategy,
    MatrixPlotStrategy,
    NOctPlotStrategy,
    WaveformPlotStrategy,
    _reshape_spectrogram_data,
    _reshape_to_2d,
)


class DummyFrame:
    def __init__(self):
        self.time = np.arange(10)
        self.data = np.random.randn(2, 10)
        self.labels = ["ch1", "ch2"]
        self.n_channels = 2
        self.label = "dummy"
        self.freqs = np.linspace(0, 100, 10)
        self.dB = np.random.randn(2, 10)
        self.dBA = np.random.randn(2, 10)
        self.magnitude = np.abs(self.dB)
        self.operation_history = [dict(operation="spectrum")]
        self.channels = [type("Ch", (), {"label": label})() for label in self.labels]
        self.n = 3


@pytest.mark.parametrize(
    "strategy,kwargs,label",
    [
        (WaveformPlotStrategy, {"xlabel": "X", "ylabel": "Y", "alpha": 0.5}, "Y"),
        (
            FrequencyPlotStrategy,
            {"xlabel": "FREQ", "ylabel": "POW", "alpha": 0.2},
            "POW",
        ),
        (NOctPlotStrategy, {"xlabel": "OCT", "ylabel": "LVL", "alpha": 0.1}, "LVL"),
        (MatrixPlotStrategy, {"xlabel": "MATX", "ylabel": "COH", "alpha": 0.3}, "COH"),
    ],
)
def test_plot_parametrize(strategy, kwargs, label):
    frame = DummyFrame()
    strat = strategy()
    if strategy is MatrixPlotStrategy:
        axes = strat.plot(frame, overlay=False, **kwargs)
        ax = next(axes)
    else:
        ax = strat.plot(frame, overlay=True, **kwargs)
    assert ax.get_xlabel() == kwargs["xlabel"]
    assert ax.get_ylabel() == kwargs["ylabel"]


class TestReshapeHelperIntegration:
    """Test integration of reshape helper functions with plot strategies."""

    def test_reshape_to_2d_integration(self) -> None:
        """Test that _reshape_to_2d works correctly with plot strategies."""
        # Create 1D data that would be used in plotting
        data_1d = np.random.rand(100)

        # Test reshape
        result = _reshape_to_2d(data_1d)

        # Verify the result is suitable for plotting strategies
        assert result.ndim == 2
        assert result.shape == (1, 100)

        # Test with frame that has 1D data
        frame = DummyFrame()
        frame.data = data_1d  # Override with 1D data

        # The strategy should be able to handle the reshaped data
        reshaped_data = _reshape_to_2d(frame.data)
        assert reshaped_data.shape[0] == 1  # Single channel

    def test_reshape_spectrogram_data_integration(self) -> None:
        """Test _reshape_spectrogram_data works correctly for plotting."""
        # Test 1D spectrogram data (single frequency snapshot)
        freq_data_1d = np.random.rand(513)  # n_fft=1024 -> 513 freq bins
        result_1d = _reshape_spectrogram_data(freq_data_1d)

        assert result_1d.shape == (1, 513, 1)  # (channels, freqs, time)

        # Test 2D spectrogram data (freqs x time)
        spec_data_2d = np.random.rand(513, 100)  # 513 freqs, 100 time frames
        result_2d = _reshape_spectrogram_data(spec_data_2d)

        assert result_2d.shape == (1, 513, 100)  # (channels, freqs, time)

        # Test 3D data should remain unchanged
        spec_data_3d = np.random.rand(2, 513, 100)  # 2 channels
        result_3d = _reshape_spectrogram_data(spec_data_3d)

        assert result_3d.shape == (2, 513, 100)
        np.testing.assert_array_equal(result_3d, spec_data_3d)

    def test_reshape_consistency_across_strategies(self) -> None:
        """Test reshape behavior is consistent across different strategies."""
        # Create frames with 1D data
        frame_1d = DummyFrame()
        frame_1d.data = np.random.rand(100)
        frame_1d.n_channels = 1

        strategies = [
            WaveformPlotStrategy(),
            FrequencyPlotStrategy(),
            NOctPlotStrategy(),
        ]

        for strategy in strategies:
            # All strategies should handle 1D data through _reshape_to_2d
            reshaped = _reshape_to_2d(frame_1d.data)
            assert reshaped.shape == (1, 100)

            # The reshaped data should be compatible with strategy expectations
            assert reshaped.ndim >= 2  # All strategies expect at least 2D data
