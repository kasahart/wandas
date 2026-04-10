import numpy as np
import pytest

from wandas.visualization.plotting import (
    FrequencyPlotStrategy,
    MatrixPlotStrategy,
    NOctPlotStrategy,
    WaveformPlotStrategy,
)

_N_SAMPLES = 10  # small deterministic sample count
_N_CHANNELS = 2


class DummyFrame:
    """Lightweight frame stub with deterministic data for parameter tests."""

    def __init__(self) -> None:
        self.time = np.arange(_N_SAMPLES)
        # Deterministic sinusoidal data — no randomness
        t = np.linspace(0, 1, _N_SAMPLES, endpoint=False)
        self.data = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], axis=0)
        self.labels = ["ch1", "ch2"]
        self.n_channels = _N_CHANNELS
        self.label = "dummy"
        self.freqs = np.linspace(0, 100, _N_SAMPLES)
        self.dB = self.data  # reuse deterministic data
        self.dBA = self.data * 0.5
        self.magnitude = np.abs(self.dB)
        self.operation_history = [dict(operation="spectrum")]
        self.channels = [type("Ch", (), {"label": label, "unit": ""})() for label in self.labels]
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
    """Verify xlabel, ylabel, and alpha forwarding across all plot strategies."""
    frame = DummyFrame()
    strat = strategy()
    if strategy is MatrixPlotStrategy:
        axes = strat.plot(frame, overlay=False, **kwargs)
        ax = next(axes)
    else:
        ax = strat.plot(frame, overlay=True, **kwargs)
    assert ax.get_xlabel() == kwargs["xlabel"]
    assert ax.get_ylabel() == kwargs["ylabel"]


@pytest.mark.parametrize("strategy_cls", [WaveformPlotStrategy, FrequencyPlotStrategy, NOctPlotStrategy])
def test_non_overlay_label_sequence_uses_per_channel_labels(strategy_cls):
    """Non-overlay mode applies per-channel labels from a sequence."""
    frame = DummyFrame()
    axes = list(strategy_cls().plot(frame, overlay=False, label=["left", "right"]))

    legend_labels = [ax.get_legend().get_texts()[0].get_text() for ax in axes]
    assert legend_labels == ["left", "right"]


@pytest.mark.parametrize("strategy_cls", [WaveformPlotStrategy, FrequencyPlotStrategy, NOctPlotStrategy])
def test_non_overlay_label_sequence_length_mismatch_raises_clear_error(strategy_cls):
    """Label sequence length != channel count raises ValueError with structured message."""
    frame = DummyFrame()

    with pytest.raises(ValueError) as exc_info:
        strategy_cls().plot(frame, overlay=False, label=["only_one"])

    error_message = str(exc_info.value)
    assert error_message.splitlines()[0] == "Channel label count mismatch"
    assert "Got: 1 labels for 2 channels" in error_message
    assert "Expected: One label per channel in non-overlay mode" in error_message
    assert (
        "Provide label as a single string for all channels or a sequence matching the channel count." in error_message
    )
