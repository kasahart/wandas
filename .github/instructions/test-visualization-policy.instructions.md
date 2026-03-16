---
description: "Visualization test patterns: plot strategy dispatch, axes return types, parameter forwarding, and memory leak prevention"
applyTo: "tests/visualization/**"
---
# Wandas Test Policy: Visualization (`tests/visualization/`)

Visualization テストは「プロット生成の正しさ」と「フレームメソッドとの整合性」を検証する層です。
数値計算の正確性は Processing テストで担保されるため、Visualization テストでは
**「正しいデータが正しい形式でプロットに渡されているか」** に集中します。

**前提**: このファイルは [test-grand-policy.instructions.md](test-grand-policy.instructions.md) と同時に適用されます。

---

## Common Fixtures for Visualization Tests

```python
import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pytest
from wandas.frames.channel import ChannelFrame


@pytest.fixture
def channel_frame():
    """Standard mono frame for plot testing (440 Hz sine)."""
    sr = 16000
    t = np.arange(sr) / sr
    data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return ChannelFrame.from_numpy(data, sampling_rate=sr, label="test_signal")


@pytest.fixture
def stereo_frame():
    """2-channel frame for multi-channel plot testing."""
    sr = 16000
    t = np.arange(sr) / sr
    ch0 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    ch1 = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    return ChannelFrame.from_numpy(
        np.stack([ch0, ch1]), sampling_rate=sr,
        ch_labels=["440Hz", "880Hz"],
    )


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Ensure all matplotlib figures are properly cleaned up after each test."""
    yield
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.clf()
    plt.close("all")
```

---

## Visualization Test Strategy

### What to Test (and What NOT to Test)

**Test (Visualization の責任):**
- PlotStrategy の dispatch が正しいこと（`create_operation("waveform")` が正しい Strategy を返す）
- Axes オブジェクトが返されること
- 正しい数のサブプロットが生成されること（チャンネル数に対応）
- 軸ラベル、タイトル、凡例が設定されていること
- overlay=True/False で挙動が変わること

**Do NOT Test (Processing/Frame の責任):**
- プロットされるデータの数値的正確性（これは Processing テストで担保）
- FFT や STFT の計算結果（spectral テストで担保）
- メタデータの伝播（frame テストで担保）

---

## PlotStrategy Dispatch Tests

```python
from wandas.visualization.plotting import create_operation

def test_waveform_strategy_dispatch():
    """'waveform' must dispatch to WaveformPlotStrategy."""
    strategy = create_operation("waveform")
    assert strategy is not None
    assert hasattr(strategy, "plot")

def test_unknown_strategy_raises():
    """Unknown plot type must raise a clear error."""
    with pytest.raises((ValueError, KeyError)):
        create_operation("nonexistent_plot_type")

def test_all_registered_strategies():
    """All documented plot types must be dispatchable."""
    for plot_type in ["waveform", "frequency", "spectrogram", "describe"]:
        strategy = create_operation(plot_type)
        assert strategy is not None
```

---

## Axes Return Type Tests

```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from collections.abc import Iterator
from matplotlib.axes import Axes

def test_plot_returns_axes(channel_frame):
    """plot() must return an Axes object."""
    result = channel_frame.plot()
    assert isinstance(result, (Axes, Iterator))

def test_plot_overlay_single_axes(stereo_frame):
    """overlay=True must produce a single Axes with all channels."""
    ax = stereo_frame.plot(overlay=True)
    assert isinstance(ax, Axes)
    # Multiple lines on the same axes
    assert len(ax.get_lines()) >= stereo_frame.n_channels

def test_plot_separate_subplots(stereo_frame):
    """overlay=False must produce separate subplots per channel."""
    result = stereo_frame.plot(overlay=False)
    # Result should be iterable for multi-channel
    if hasattr(result, "__iter__"):
        axes_list = list(result)
        assert len(axes_list) == stereo_frame.n_channels
```

---

## Describe Method Tests

`describe()` は複合プロットメソッドであり、特別な検証が必要。

```python
def test_describe_returns_figures(channel_frame):
    """describe(is_close=False) must return list of Figure objects."""
    figures = channel_frame.describe(is_close=False)
    assert figures is not None
    assert len(figures) == channel_frame.n_channels
    for fig in figures:
        assert isinstance(fig, plt.Figure)

def test_describe_closes_figures_by_default(channel_frame):
    """describe(is_close=True) must return None and close figures."""
    result = channel_frame.describe(is_close=True)
    assert result is None

def test_describe_image_save(channel_frame, tmp_path):
    """describe(image_save=path) must save the figure to disk."""
    path = tmp_path / "test.png"
    channel_frame.describe(image_save=str(path))
    assert path.exists()

def test_describe_multichannel_image_save(stereo_frame, tmp_path):
    """Multi-channel describe must save with channel index suffix."""
    path = tmp_path / "test.png"
    stereo_frame.describe(image_save=str(path))
    # Should create test_0.png, test_1.png
    for i in range(stereo_frame.n_channels):
        expected = tmp_path / f"test_{i}.png"
        assert expected.exists()
```

---

## Plot Parameter Forwarding Tests

```python
def test_plot_custom_title(channel_frame):
    """Custom title must be set on the axes."""
    ax = channel_frame.plot(overlay=True, title="Custom Title")
    assert ax.get_title() == "Custom Title"

def test_plot_custom_labels(channel_frame):
    """Custom xlabel/ylabel must be set on the axes."""
    ax = channel_frame.plot(overlay=True, xlabel="Time [s]", ylabel="Amplitude")
    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Amplitude"

def test_plot_xlim_ylim(channel_frame):
    """xlim/ylim must be forwarded to the axes."""
    ax = channel_frame.plot(overlay=True, xlim=(0, 0.5), ylim=(-1, 1))
    assert ax.get_xlim() == pytest.approx((0, 0.5), abs=0.01)
    assert ax.get_ylim() == pytest.approx((-1, 1), abs=0.01)
```

---

## Memory Leak Prevention

Visualization テストでは `fig.clf()` で Figure 内部状態をクリアした後、
`plt.close()` でウィンドウを閉じること。`plt.close("all")` のみでは不十分。

```python
# GOOD: fig.clf() + plt.close() の組み合わせ
def test_plot_cleanup(channel_frame):
    ax = channel_frame.plot()
    # ... assertions ...
    fig = ax.get_figure()
    fig.clf()
    plt.close(fig)

# BEST: autouse fixture で全テストに自動適用
@pytest.fixture(autouse=True)
def cleanup_plots():
    """Ensure all matplotlib figures are properly cleaned up after each test."""
    yield
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.clf()
    plt.close("all")
```

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [frames-design.prompt.md](frames-design.prompt.md) — Frame method behavior that plots depend on
