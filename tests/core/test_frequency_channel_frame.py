# tests/core/test_frequency_channel_frame.py

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from wandas.core.frequency_channel import FrequencyChannel
from wandas.core.frequency_channel_frame import FrequencyChannelFrame
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def test_spectrum_initialization() -> None:
    data1 = np.array([10, 9, 8, 7, 6])
    data2 = np.array([5, 4, 3, 2, 1])
    sampling_rate = 1000
    n_fft = 1024
    window = np.hanning(5)
    label = "Test Spectrum"
    unit = "V"
    metadata = {"note": "Test metadata"}

    freq_channel1 = FrequencyChannel(
        data=data1,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label=label,
        unit=unit,
        metadata=metadata,
    )
    freq_channel2 = FrequencyChannel(
        data=data2,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label=label,
        unit=unit,
        metadata=metadata,
    )

    spectrum = FrequencyChannelFrame(
        channels=[freq_channel1, freq_channel2], label="Test Spectrum"
    )

    assert spectrum.label == "Test Spectrum"
    assert len(spectrum.channels) == 2
    assert spectrum.channels[0] == freq_channel1
    assert spectrum.channels[1] == freq_channel2


def test_plot_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Define a dummy frequency channel with a recording plot method.
    class DummyFrequencyChannel(FrequencyChannel):
        def __init__(self, label: str) -> None:
            self._label = label
            self.plot_called = False

        def plot(
            self,
            ax: Optional["Axes"] = None,
            title: Optional[str] = None,
            Aw: bool = False,  # noqa: N803
        ) -> tuple["Axes", NDArrayReal]:
            self.plot_called = True
            _, ax = plt.subplots(figsize=(10, 4))
            return ax, np.array([0, 1, 2, 3, 4])

    dummy1 = DummyFrequencyChannel("Channel 1")
    dummy2 = DummyFrequencyChannel("Channel 2")
    spectrum = FrequencyChannelFrame(channels=[dummy1, dummy2], label="Test Spectrum")

    # Monkeypatch plt.show to record its call.
    show_called = False

    def fake_show() -> None:
        nonlocal show_called
        show_called = True

    monkeypatch.setattr(plt, "show", fake_show)

    # Call plot without providing an axis.
    spectrum.plot(ax=None, title="My Spectrum")
    assert dummy1.plot_called
    assert dummy2.plot_called
    assert show_called


def test_plot_with_ax(monkeypatch: pytest.MonkeyPatch) -> None:
    # Define a dummy frequency channel with a recording plot method.
    class DummyFrequencyChannel(FrequencyChannel):
        def __init__(self, label: str) -> None:
            self._label: str = label
            self.plot_called: bool = False

        def plot(
            self,
            ax: Optional["Axes"] = None,
            title: Optional[str] = None,
            Aw: bool = False,  # noqa: N803
        ) -> tuple["Axes", NDArrayReal]:
            self.plot_called = True
            _, ax = plt.subplots(figsize=(10, 4))
            return ax, np.array([0, 1, 2, 3, 4])

    dummy1: DummyFrequencyChannel = DummyFrequencyChannel("Channel 1")
    dummy2: DummyFrequencyChannel = DummyFrequencyChannel("Channel 2")
    spectrum = FrequencyChannelFrame(channels=[dummy1, dummy2], label="Test Spectrum")

    fig, ax = plt.subplots()
    # Monkeypatch plt.show to record its call.
    show_called: bool = False

    def fake_show() -> None:
        nonlocal show_called
        show_called = True

    monkeypatch.setattr(plt, "show", fake_show)

    # Call plot with a provided axis.
    spectrum.plot(ax=ax, title="Provided Axis")
    assert dummy1.plot_called
    assert dummy2.plot_called
    # Since an axis was provided, plt.show should not be called.
    assert not show_called
    plt.close(fig)


def test_plot_matrix() -> None:
    # Define a dummy frequency channel with a recording plot method.
    class DummyFrequencyChannel(FrequencyChannel):
        def __init__(self, label: str) -> None:
            self._label: str = label
            self.plot_called: bool = False

        def plot(
            self,
            ax: Optional["Axes"] = None,
            title: Optional[str] = None,
            Aw: bool = False,  # noqa: N803
        ) -> tuple["Axes", NDArrayReal]:
            self.plot_called = True
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 4))
            if title:
                ax.set_title(title)
            return ax, np.array([0, 1, 2, 3, 4])

    dummy1: DummyFrequencyChannel = DummyFrequencyChannel("Channel 1")
    dummy2: DummyFrequencyChannel = DummyFrequencyChannel("Channel 2")
    spectrum = FrequencyChannelFrame(channels=[dummy1, dummy2], label="Test Spectrum")

    # Call plot_matrix with a title.
    fig, axes = spectrum.plot_matrix(title="Matrix Plot", Aw=True)
    # Verify that figure and axes were returned correctly.
    assert isinstance(fig, Figure)
    axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    assert len(axes_list) >= 2

    # Verify that the dummy channels' plot methods were invoked.
    assert dummy1.plot_called
    assert dummy2.plot_called

    # Check if the figure suptitle is set correctly.
    suptitle: str = fig.get_suptitle()
    assert suptitle == "Matrix Plot"
    plt.close(fig)
