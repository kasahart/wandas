import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # GUIなし環境用
from wandas.visualization.plotting import (
    FrequencyPlotStrategy,
    MatrixPlotStrategy,
    NOctPlotStrategy,
    WaveformPlotStrategy,
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
