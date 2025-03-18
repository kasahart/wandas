# tests/core/test_frequency_channel_frame.py

from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from wandas.core.frequency_channel import FrequencyChannel
from wandas.core.frequency_channel_frame import FrequencyChannelFrame
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def test_getitem_by_index_and_label() -> None:
    # Test __getitem__ both for index and label.
    data1 = np.array([0, 1, 2])
    data2 = np.array([3, 4, 5])
    sampling_rate = 1000
    n_fft = 1024
    window = np.hanning(5)

    unit = "V"
    metadata = {"note": "Test metadata"}
    ch1 = FrequencyChannel(
        data=data1,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label="First",
        unit=unit,
        metadata=metadata,
    )
    ch2 = FrequencyChannel(
        data=data2,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label="Second",
        unit=unit,
        metadata=metadata,
    )
    cf = FrequencyChannelFrame(channels=[ch1, ch2], label="GetItemTest")

    # Access by index.
    assert cf[0] == ch1
    assert cf[1] == ch2
    with pytest.raises(IndexError):
        _ = cf[2]

    # Access by label.
    assert cf["First"] == ch1
    assert cf["Second"] == ch2
    with pytest.raises(KeyError):
        _ = cf["NonExistent"]


def test_setitem_by_index_and_label() -> None:
    # Test __setitem__ both for index and label.
    data1 = np.array([0, 1, 2])
    data2 = np.array([3, 4, 5])
    sampling_rate = 1000
    n_fft = 1024
    window = np.hanning(5)
    unit = "V"
    metadata = {"note": "Test metadata"}
    ch1 = FrequencyChannel(
        data=data1,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label="First",
        unit=unit,
        metadata=metadata,
    )
    ch2 = FrequencyChannel(
        data=data2,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label="Second",
        unit=unit,
        metadata=metadata,
    )
    cf = FrequencyChannelFrame(channels=[ch1, ch2], label="SetItemTest")

    # Set by index.
    data3 = np.array([6, 7, 8])
    ch3 = FrequencyChannel(
        data=data3,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label="Third",
        unit=unit,
        metadata=metadata,
    )
    cf[1] = ch3
    assert cf[1] == ch3

    # Set by label.
    data4 = np.array([9, 10, 11])
    ch4 = FrequencyChannel(
        data=data4,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label="Fourth",
        unit=unit,
        metadata=metadata,
    )
    cf["First"] = ch4
    assert cf["First"] == ch4


def test_iter_and_len() -> None:
    # Test __iter__ and __len__
    data = np.array([0, 1, 2])
    sampling_rate = 1000
    n_fft = 1024
    window = np.hanning(5)
    unit = "V"
    metadata = {"note": "Test metadata"}
    ch_list = [
        FrequencyChannel(
            data=data,
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            window=window,
            label=f"ch{i}",
            unit=unit,
            metadata=metadata,
        )
        for i in range(3)
    ]
    cf = FrequencyChannelFrame(channels=ch_list, label="IterTest")
    # Check length.
    assert len(cf) == 3
    # Check iteration produces all channels.
    iterated = [ch for ch in cf]
    assert iterated == ch_list


def test_spectrum_initialization() -> None:
    data1 = np.array([10, 9, 8, 7, 6])
    data2 = np.array([5, 4, 3, 2, 1])
    sampling_rate = 1000
    n_fft = 1024
    window = np.hanning(5)
    unit = "V"
    metadata = {"note": "Test metadata"}

    freq_channel1 = FrequencyChannel(
        data=data1,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label="Channel 1",
        unit=unit,
        metadata=metadata,
    )
    freq_channel2 = FrequencyChannel(
        data=data2,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label="Channel 2",
        unit=unit,
        metadata=metadata,
    )

    spectrum = FrequencyChannelFrame(
        channels=[freq_channel1, freq_channel2], label="Test Spectrum"
    )

    assert spectrum.label == "Test Spectrum"
    assert len(spectrum._channels) == 2
    assert spectrum._channels[0] == freq_channel1
    assert spectrum._channels[1] == freq_channel2


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
            plot_kwargs: Optional[dict[str, Any]] = None,
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
            plot_kwargs: Optional[dict[str, Any]] = None,
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
            plot_kwargs: Optional[dict[str, Any]] = None,
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
