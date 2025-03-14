# tests/core/channel_frame.py

import csv
from pathlib import Path
from typing import Any, Optional

import ipywidgets as widgets
import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from wandas.core.channel import Channel
from wandas.core.channel_frame import ChannelFrame


@pytest.fixture  # type: ignore [misc, unused-ignore]
def generate_signals() -> tuple[ChannelFrame, ChannelFrame]:
    # サンプルの直流データを生成
    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data1_signal1 = np.full_like(t, 2)  # Signal 1の振幅2の直流信号
    data2_signal1 = np.full_like(t, 3)  # Signal 1の振幅3の直流信号
    data1_signal2 = np.full_like(t, 4)  # Signal 2の振幅4の直流信号
    data2_signal2 = np.full_like(t, 5)  # Signal 2の振幅5の直流信号

    ch1_signal1 = Channel(
        data=data1_signal1, sampling_rate=sampling_rate, label="Channel 1"
    )
    ch2_signal1 = Channel(
        data=data2_signal1, sampling_rate=sampling_rate, label="Channel 2"
    )
    ch1_signal2 = Channel(
        data=data1_signal2, sampling_rate=sampling_rate, label="Channel 1"
    )
    ch2_signal2 = Channel(
        data=data2_signal2, sampling_rate=sampling_rate, label="Channel 2"
    )

    signal1 = ChannelFrame(channels=[ch1_signal1, ch2_signal1], label="Signal 1")
    signal2 = ChannelFrame(channels=[ch1_signal2, ch2_signal2], label="Signal 2")

    return signal1, signal2


def test_signal_initialization() -> None:
    data1 = np.array([0, 1, 2, 3, 4])
    data2 = np.array([5, 6, 7, 8, 9])
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")

    signal = ChannelFrame(channels=[channel1, channel2], label="Test Signal")

    assert signal.label == "Test Signal"
    assert len(signal.channels) == 2
    assert signal.channels[0] == channel1
    assert signal.channels[1] == channel2
    assert signal.sampling_rate == sampling_rate


def test_signal_sampling_rate_mismatch() -> None:
    data1 = np.array([0, 1, 2, 3, 4])
    data2 = np.array([5, 6, 7, 8, 9])
    channel1 = Channel(data=data1, sampling_rate=1000)
    channel2 = Channel(data=data2, sampling_rate=2000)

    with pytest.raises(ValueError):
        ChannelFrame(channels=[channel1, channel2])


def test_signal_high_pass_filter() -> None:
    t = np.linspace(0, 1, 1000)
    data1 = np.sin(2 * np.pi * 50 * t)
    data2 = np.sin(2 * np.pi * 100 * t)
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")
    signal = ChannelFrame(channels=[channel1, channel2])

    filtered_signal = signal.high_pass_filter(cutoff=30)

    # 各チャンネルがフィルタリングされていることを確認
    for original_ch, filtered_ch in zip(signal.channels, filtered_signal.channels):
        assert not np.array_equal(original_ch.data, filtered_ch.data)


def test_signal_low_pass_filter() -> None:
    t = np.linspace(0, 1, 1000)
    data1 = np.sin(2 * np.pi * 50 * t)
    data2 = np.sin(2 * np.pi * 100 * t)
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")
    signal = ChannelFrame(channels=[channel1, channel2])

    filtered_signal = signal.low_pass_filter(cutoff=30)

    # 各チャンネルがフィルタリングされていることを確認
    for original_ch, filtered_ch in zip(signal.channels, filtered_signal.channels):
        assert not np.array_equal(original_ch.data, filtered_ch.data)


def test_signal_fft() -> None:
    signal_length = 1000
    t = np.linspace(0, 1, signal_length)
    data1 = np.sin(2 * np.pi * 50 * t)
    data2 = np.sin(2 * np.pi * 100 * t)
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")
    signal = ChannelFrame(channels=[channel1, channel2])

    spectrum = signal.fft(n_fft=1024, window="hann")

    assert len(spectrum.channels) == 2
    for freq_ch, label, expected_freq in zip(
        spectrum.channels, ["Channel 1", "Channel 2"], [50, 100]
    ):
        assert freq_ch.label == label
        assert freq_ch.n_fft == 1024
        assert not np.array_equal(freq_ch.window, np.hanning(signal_length))

        # Find the frequency bin with the maximum amplitude
        freqs = np.fft.fftfreq(1024, 1 / sampling_rate)
        fft_data = np.abs(freq_ch.data)
        peak_freq = freqs[np.argmax(fft_data)]

        # Check if the peak frequency matches the expected frequency
        assert np.isclose(peak_freq, expected_freq, atol=1)


def test_signal_welch() -> None:
    n_fft = 1024
    win_length = n_fft
    signal_length = n_fft * 5
    sampling_rate = 1000

    t = np.linspace(0, 1, sampling_rate)
    data1 = np.sin(2 * np.pi * 125 * t)
    data2 = np.sin(2 * np.pi * 250 * t)
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")
    signal = ChannelFrame(channels=[channel1, channel2])

    spectrum = signal.welch(n_fft=n_fft, win_length=win_length, window="hann")

    assert len(spectrum.channels) == 2
    for freq_ch, label, expected_freq in zip(
        spectrum.channels, ["Channel 1", "Channel 2"], [125, 250]
    ):
        assert freq_ch.label == label
        assert freq_ch.n_fft == n_fft
        assert not np.array_equal(freq_ch.window, np.hanning(signal_length))

        # Find the frequency bin with the maximum amplitude
        freqs = np.fft.rfftfreq(n_fft, 1 / sampling_rate)
        fft_data = np.abs(freq_ch.data)
        peak_freq = freqs[np.argmax(fft_data)]

        # Check if the peak frequency matches the expected frequency
        assert np.isclose(peak_freq, expected_freq, atol=1)


def test_signal_addition(generate_signals: tuple[ChannelFrame, ChannelFrame]) -> None:
    signal1, signal2 = generate_signals
    result_signal = signal1 + signal2

    # 各チャンネルの加算結果を確認
    for i in range(len(signal1.channels)):
        expected_data = signal1.channels[i].data + signal2.channels[i].data
        assert np.array_equal(result_signal.channels[i].data, expected_data), (
            f"Signal addition failed for channel {i + 1}."
        )


def test_signal_subtraction(
    generate_signals: tuple[ChannelFrame, ChannelFrame],
) -> None:
    signal1, signal2 = generate_signals
    result_signal = signal1 - signal2

    # 各チャンネルの減算結果を確認
    for i in range(len(signal1.channels)):
        expected_data = signal1.channels[i].data - signal2.channels[i].data
        assert np.array_equal(result_signal.channels[i].data, expected_data), (
            f"Signal subtraction failed for channel {i + 1}."
        )


def test_signal_multiplication(
    generate_signals: tuple[ChannelFrame, ChannelFrame],
) -> None:
    signal1, signal2 = generate_signals
    result_signal = signal1 * signal2

    # 各チャンネルの乗算結果を確認
    for i in range(len(signal1.channels)):
        expected_data = signal1.channels[i].data * signal2.channels[i].data
        assert np.array_equal(result_signal.channels[i].data, expected_data), (
            f"Signal multiplication failed for channel {i + 1}."
        )


def test_signal_division(generate_signals: tuple[ChannelFrame, ChannelFrame]) -> None:
    signal1, signal2 = generate_signals
    result_signal = signal1 / signal2

    # 各チャンネルの除算結果を確認
    for i in range(len(signal1.channels)):
        expected_data = signal1.channels[i].data / signal2.channels[i].data
        assert np.allclose(result_signal.channels[i].data, expected_data, atol=1e-6), (
            f"Signal division failed for channel {i + 1}."
        )


def test_channel_frame_from_ndarray() -> None:
    array = np.array([[0, 1, 2], [3, 4, 5]])
    sampling_rate = 1000
    labels = ["Channel 1", "Channel 2"]
    channel_frame = ChannelFrame.from_ndarray(array, sampling_rate, labels)

    assert len(channel_frame.channels) == 2
    assert channel_frame.channels[0].label == "Channel 1"
    assert channel_frame.channels[1].label == "Channel 2"
    assert np.array_equal(channel_frame.channels[0].data, array[0])
    assert np.array_equal(channel_frame.channels[1].data, array[1])
    assert channel_frame.sampling_rate == sampling_rate


def test_channel_frame_read_wav(tmp_path: Path) -> None:
    filename = tmp_path / "test.wav"
    sampling_rate = 1000
    data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int16)
    wavfile.write(filename, sampling_rate, data.T)

    channel_frame = ChannelFrame.read_wav(str(filename))

    assert len(channel_frame.channels) == 2
    assert np.array_equal(channel_frame.channels[0].data, data[0])
    assert np.array_equal(channel_frame.channels[1].data, data[1])
    assert channel_frame.sampling_rate == sampling_rate


def test_channel_frame_to_wav(tmp_path: Path) -> None:
    filename = tmp_path / "test.wav"
    data1 = np.array([0, 1, 2], dtype=np.float32)
    data2 = np.array([3, 4, 5], dtype=np.float32)
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")
    channel_frame = ChannelFrame(channels=[channel1, channel2])

    channel_frame.to_wav(str(filename))

    sr, data = wavfile.read(filename)
    assert sr == sampling_rate


# Test sampling rate mismatch in __init__
def test_sampling_rate_mismatch_init() -> None:
    channel1 = Channel(data=np.array([0, 1, 2]), sampling_rate=1000, label="Ch1")
    channel2 = Channel(data=np.array([0, 1, 2]), sampling_rate=1100, label="Ch2")
    with pytest.raises(ValueError):
        ChannelFrame(channels=[channel1, channel2])


# Test duplicate channel labels in __init__
def test_duplicate_channel_labels() -> None:
    channel1 = Channel(data=np.array([0, 1, 2]), sampling_rate=1000, label="Same")
    channel2 = Channel(data=np.array([3, 4, 5]), sampling_rate=1000, label="Same")
    with pytest.raises(ValueError):
        ChannelFrame(channels=[channel1, channel2])


# Test from_ndarray
def test_from_ndarray() -> None:
    data = np.array([[0, 1, 2], [3, 4, 5]])
    sampling_rate = 1000
    labels = ["Channel 1", "Channel 2"]
    cf = ChannelFrame.from_ndarray(data, sampling_rate, labels)
    assert len(cf.channels) == 2
    np.testing.assert_array_equal(cf.channels[0].data, data[0])
    np.testing.assert_array_equal(cf.channels[1].data, data[1])
    assert cf.sampling_rate == sampling_rate
    assert cf.channels[0].label == "Channel 1"


# Test read_csv with valid CSV file
def test_read_csv_valid(tmp_path: Path) -> None:
    # Create CSV data with time column and two channels
    filename = tmp_path / "test.csv"
    header = ["time", "A", "B"]
    rows = [
        [0.0, 10, 20],
        [0.1, 11, 21],
        [0.2, 12, 22],
        [0.3, 13, 23],
    ]
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(filename, index=False)

    # Sampling rate calculation: time diff=0.1 so sampling_rate should be int(1/0.1)=10
    cf = ChannelFrame.read_csv(str(filename), time_column="time")
    # After dropping time column, columns are ['A', 'B']
    assert cf.sampling_rate == 10
    # Check the first channel
    expected_a = np.array([10, 11, 12, 13])
    expected_b = np.array([20, 21, 22, 23])
    np.testing.assert_array_equal(cf.channels[0].data, expected_a)
    np.testing.assert_array_equal(cf.channels[1].data, expected_b)
    # If header is present, labels should be the remaining column names.
    assert cf.channels[0].label == "A"
    assert cf.channels[1].label == "B"


# Test read_csv with missing time column
def test_read_csv_missing_time(tmp_path: Path) -> None:
    filename = tmp_path / "test_missing.csv"
    header = ["timestamp", "A", "B"]
    rows = [
        [0.0, 10, 20],
        [0.1, 11, 21],
        [0.2, 12, 22],
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    with pytest.raises(KeyError):
        ChannelFrame.read_csv(str(filename), time_column="time")


# Test read_csv with insufficient time points
def test_read_csv_insufficient_time(tmp_path: Path) -> None:
    filename = tmp_path / "test_insufficient.csv"
    header = ["time", "A"]
    rows = [[0.0, 10]]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    with pytest.raises(ValueError):
        ChannelFrame.read_csv(str(filename), time_column="time")


# Test to_audio method
@pytest.fixture  # type: ignore [misc, unused-ignore]
def generate_simple_signal() -> ChannelFrame:
    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = np.sin(2 * np.pi * 10 * t)
    channel1 = Channel(data=data, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data, sampling_rate=sampling_rate, label="Channel 2")
    return ChannelFrame(channels=[channel1, channel2], label="Simple Signal")


def test_to_audio() -> None:
    # Create a simple signal,
    # convert to audio and verify it returns a VBox with children.
    sampling_rate = 1000
    channel1 = Channel(
        data=np.array([0, 1, 2]), sampling_rate=sampling_rate, label="Ch1"
    )
    channel2 = Channel(
        data=np.array([3, 4, 5]), sampling_rate=sampling_rate, label="Ch2"
    )
    cf = ChannelFrame(channels=[channel1, channel2], label="Test")
    audio_widget = cf.to_audio()
    assert isinstance(audio_widget, widgets.VBox)
    # Check that the number of children equals number of channels.
    assert len(audio_widget.children) == len(cf.channels)


def test_describe_returns_vbox() -> None:
    # Check that the describe method returns a VBox with HTML content.
    sampling_rate = 1000
    channel = Channel(
        data=np.ones(sampling_rate * 5), sampling_rate=sampling_rate, label="TestCh"
    )
    cf = ChannelFrame(channels=[channel], label="Description Test")
    desc = cf.describe()
    assert isinstance(desc, widgets.VBox)
    # Expect at least one child widget (the header plus channel description).
    assert len(desc.children) >= 1
    # Check that the first child is an HTML widget containing the signal label.
    html_widget = desc.children[0]
    assert isinstance(html_widget, widgets.HTML)
    assert "Description Test" in html_widget.value


def test_getitem_by_index_and_label() -> None:
    # Test __getitem__ both for index and label.
    data1 = np.array([0, 1, 2])
    data2 = np.array([3, 4, 5])
    sampling_rate = 1000
    ch1 = Channel(data=data1, sampling_rate=sampling_rate, label="First")
    ch2 = Channel(data=data2, sampling_rate=sampling_rate, label="Second")
    cf = ChannelFrame(channels=[ch1, ch2], label="GetItemTest")

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


def test_iter_and_len() -> None:
    # Test __iter__ and __len__
    data = np.array([0, 1, 2])
    sampling_rate = 1000
    ch_list = [
        Channel(data=data, sampling_rate=sampling_rate, label=f"Ch{i}")
        for i in range(3)
    ]
    cf = ChannelFrame(channels=ch_list, label="IterTest")
    # Check length.
    assert len(cf) == 3
    # Check iteration produces all channels.
    iterated = [ch for ch in cf]
    assert iterated == ch_list


def test_sum() -> None:
    # Test that sum() combines channels correctly.
    data1 = np.array([0, 1, 2])
    data2 = np.array([3, 4, 5])
    sampling_rate = 1000
    ch1 = Channel(data=data1, sampling_rate=sampling_rate, label="Ch1")
    ch2 = Channel(data=data2, sampling_rate=sampling_rate, label="Ch2")
    cf = ChannelFrame(channels=[ch1, ch2], label="SumTest")
    summed = cf.sum()
    expected = data1 + data2
    np.testing.assert_array_equal(summed.data, expected)


def test_mean() -> None:
    # Test that mean() computes the average of channel data.
    data1 = np.array([0, 2, 4])
    data2 = np.array([1, 3, 5])
    sampling_rate = 1000
    ch1 = Channel(data=data1, sampling_rate=sampling_rate, label="Ch1")
    ch2 = Channel(data=data2, sampling_rate=sampling_rate, label="Ch2")
    cf = ChannelFrame(channels=[ch1, ch2], label="MeanTest")
    mean_ch = cf.mean()
    expected = (data1 + data2) / 2
    np.testing.assert_array_equal(mean_ch.data, expected)


def test_channel_difference() -> None:
    # Test that channel_difference subtracts a chosen channel from all channels.
    data1 = np.array([5, 6, 7])
    data2 = np.array([2, 3, 4])
    data3 = np.array([1, 1, 1])
    sampling_rate = 1000
    ch1 = Channel(data=data1, sampling_rate=sampling_rate, label="Ch1")
    ch2 = Channel(data=data2, sampling_rate=sampling_rate, label="Ch2")
    ch3 = Channel(data=data3, sampling_rate=sampling_rate, label="Ch3")
    cf = ChannelFrame(channels=[ch1, ch2, ch3], label="DiffTest")
    # Subtract channel 1 from all channels.
    diff_cf = cf.channel_difference(other_channel=0)
    # For channel 1, result should be zero.
    np.testing.assert_array_equal(diff_cf.channels[0].data, data1 - data1)
    np.testing.assert_array_equal(diff_cf.channels[1].data, data2 - data1)
    np.testing.assert_array_equal(diff_cf.channels[2].data, data3 - data1)


def test_plot_overlay_with_ax() -> None:
    import matplotlib.pyplot as plt

    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate)
    # Use a simple sine wave for channel data
    data = np.sin(2 * np.pi * 10 * t)
    channel1 = Channel(data=data, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data, sampling_rate=sampling_rate, label="Channel 2")
    cf = ChannelFrame(channels=[channel1, channel2], label="Test Signal")

    fig, ax = plt.subplots()
    # Call plot with overlay mode using provided axis; should not invoke plt.show
    cf.plot(ax=ax, overlay=True)

    # Check that the axis has been updated (e.g., grid and legend have been set)
    # The presence of a legend object (even if empty) indicates channel.plot was called.
    assert ax.get_legend() is not None, "Legend was not set on the provided axis."


def test_plot_separate_calls_show(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.pyplot as plt

    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate)
    data = np.sin(2 * np.pi * 10 * t)
    channel1 = Channel(data=data, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data, sampling_rate=sampling_rate, label="Channel 2")
    cf = ChannelFrame(channels=[channel1, channel2], label="Test Signal")

    show_called = False

    def dummy_show() -> None:
        nonlocal show_called
        show_called = True

    monkeypatch.setattr(plt, "show", dummy_show)

    # Call plot in non-overlay mode; this branch creates subplots with suptitle.
    cf.plot(overlay=False, title="Test Title")

    assert show_called, "plt.show was not called in non-overlay mode."

    # Retrieve the current figure and check that the suptitle was set correctly.
    fig = plt.gcf()
    # Access the suptitle text to verify title
    suptitle_text = fig.get_suptitle()
    assert suptitle_text == "Test Title", (
        f"Expected title 'Test Title', got '{suptitle_text}'."
    )


def test_rms_plot_overlay_with_ax(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    overlay モードで rms_plot を呼び出したときに、渡された Axes に対して各チャネルの
    rms_plot が実行され、タイトル・グリッド・legend が設定されることを検証するテスト。
    """
    import matplotlib.pyplot as plt

    sampling_rate = 1000
    # ダミーデータ（一定値）のチャンネルを生成
    data = np.full(sampling_rate, 3.0)

    # チャンネル毎の rms_plot 呼び出しを記録するためのリスト
    call_list = []

    # インラインでダミーの rms_plot 関数を定義
    def dummy_rms_plot(
        ax: Optional[Any] = None, title: Optional[str] = None
    ) -> Channel:
        call_list.append("called")
        # ダミーとして、ax に適当な Line2D を追加
        if ax is not None:
            ax.plot([0, 1], [0, 1], label="dummy")
        return Channel(data=data, sampling_rate=sampling_rate, label="dummy")

    # チャンネルの生成と、rms_plot を上書き
    channel1 = Channel(data=data, sampling_rate=sampling_rate, label="Test Channel 1")
    channel2 = Channel(data=data, sampling_rate=sampling_rate, label="Test Channel 2")
    monkeypatch.setattr(channel1, "rms_plot", dummy_rms_plot)
    monkeypatch.setattr(channel2, "rms_plot", dummy_rms_plot)

    # 2 つのチャネルで ChannelFrame を生成
    cf = ChannelFrame(channels=[channel1, channel2], label="RMS Test")

    # overlay モード用に既存の Axes を作成して渡す
    fig, ax = plt.subplots(figsize=(10, 4))
    cf.rms_plot(ax=ax, overlay=True, title="Test RMS Overlay")

    # 各チャネルの dummy_rms_plot が呼ばれた数を検証（2 チャネルの場合、2 回の呼び出し）
    assert len(call_list) == 2, (
        f"期待する呼び出し回数は 2 回ですが、実際は {len(call_list)} 回です。"
    )
    # Axes のタイトルが設定されていること
    assert ax.get_title() == "Test RMS Overlay"
    # Axes にプロットされた Line2D オブジェクトがあることを確認（dummy で追加した線）
    assert len(ax.get_lines()) > 0, "Axes にプロットが追加されていません。"
    # legend が生成されていることを確認
    assert ax.get_legend() is not None, "legend が作成されていません。"

    plt.close(fig)


def test_rms_plot_non_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    non overlay モードで rms_plot を呼び出したときに、新規 Figure が作成され、
    suptitle, x軸ラベルが正しく設定され、plt.show() が呼ばれることを検証するテスト。
    """
    import matplotlib.pyplot as plt

    sampling_rate = 1000
    # サイン波データの生成
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t)

    # 呼び出し回数を記録するリスト
    call_list = []

    def dummy_rms_plot(
        ax: Optional[Any] = None, title: Optional[str] = None
    ) -> Channel:
        call_list.append("called")
        # ダミーとして、ax に適当な Line2D を追加
        if ax is not None:
            ax.plot([0, 1], [0, 1], label="dummy")
        return Channel(data=data, sampling_rate=sampling_rate, label="dummy")

    # チャンネルの生成と、rms_plot を上書き
    channel1 = Channel(data=data, sampling_rate=sampling_rate, label="Test Channel 1")
    channel2 = Channel(data=data, sampling_rate=sampling_rate, label="Test Channel 2")
    monkeypatch.setattr(channel1, "rms_plot", dummy_rms_plot)
    monkeypatch.setattr(channel2, "rms_plot", dummy_rms_plot)

    # 2 つのチャネルで ChannelFrame を生成
    cf = ChannelFrame(channels=[channel1, channel2], label="RMS Test")

    show_called = False

    def dummy_show() -> None:
        nonlocal show_called
        show_called = True

    monkeypatch.setattr(plt, "show", dummy_show)

    # non overlay モードで呼び出し
    cf.rms_plot(overlay=False, title="Non Overlay RMS")

    # plt.show() が呼ばれていることを確認
    assert show_called, "non overlay モードで plt.show() が呼ばれていません。"

    # 現在の Figure の suptitle が指定したタイトルと一致することを検証
    fig = plt.gcf()
    suptitle_text = fig.get_suptitle()
    assert suptitle_text == "Non Overlay RMS", (
        f"期待する suptitle は 'Non Overlay RMS' ですが、'{suptitle_text}' です。"
    )

    # 最下部の Axes の x軸ラベルが "Time (s)" に設定されていることを検証
    axs = fig.get_axes()
    assert axs[-1].get_xlabel() == "Time [s]", (
        f"x軸ラベルが期待 'Time (s)' ではなく、'{axs[-1].get_xlabel()}' です。"
    )

    # 各チャネル毎に dummy_rms_plot が呼ばれている数（2 チャネルの場合、2 回）を確認
    assert len(call_list) == 2, (
        f"期待する呼び出し回数は 2 回ですが、実際は {len(call_list)} 回です。"
    )

    plt.close(fig)
