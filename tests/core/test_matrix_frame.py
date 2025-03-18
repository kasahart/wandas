import numpy as np
import pytest
from scipy import signal

from wandas.core.channel import Channel
from wandas.core.channel_frame import ChannelFrame
from wandas.core.matrix_frame import MatrixFrame


def test_matrix_frame_initialization() -> None:
    data = np.random.rand(3, 100)
    sampling_rate = 1000
    channel_units = ["mV", "mV", "mV"]
    channel_labels = ["Ch1", "Ch2", "Ch3"]
    channel_metadata = [{"type": "EEG"}, {"type": "EEG"}, {"type": "EEG"}]
    label = "Test Frame"

    mf = MatrixFrame(
        data, sampling_rate, channel_units, channel_labels, channel_metadata, label
    )

    assert mf.data.shape == (3, 100)
    assert mf.sampling_rate == 1000
    assert mf.label == "Test Frame"
    assert len(mf._channels) == 3
    assert mf._channels[0].unit == "mV"
    assert mf._channels[0].label == "Ch1"
    assert mf._channels[0].metadata == {"type": "EEG"}


def test_matrix_frame_invalid_initialization() -> None:
    data = np.random.rand(3, 100)
    sampling_rate = 1000

    with pytest.raises(ValueError):
        MatrixFrame(
            data,
            sampling_rate,
            channel_units=["mV"],
            channel_labels=["Ch1"],
            channel_metadata=[{}],
        )


def test_matrix_frame_len() -> None:
    data = np.random.rand(3, 100)
    mf = MatrixFrame(data, 1000)
    assert len(mf) == 3


def test_matrix_frame_iter() -> None:
    data = np.random.rand(3, 100)
    mf = MatrixFrame(data, 1000)
    channels = list(mf)
    assert len(channels) == 3
    assert isinstance(channels[0], Channel)


def test_matrix_frame_getitem() -> None:
    data = np.random.rand(3, 100)
    mf = MatrixFrame(data, 1000, channel_labels=["Ch1", "Ch2", "Ch3"])

    ch1 = mf[0]
    assert ch1.label == "Ch1"

    ch2 = mf["Ch2"]
    assert ch2.label == "Ch2"

    with pytest.raises(IndexError):
        mf[10]

    with pytest.raises(KeyError):
        mf["Ch10"]


def test_matrix_frame_to_channel_frame() -> None:
    data = np.random.rand(3, 100)
    mf = MatrixFrame(data, 1000)
    cf = mf.to_channel_frame()
    assert isinstance(cf, ChannelFrame)
    assert len(cf) == 3


def test_matrix_frame_from_channel_frame() -> None:
    data = np.random.rand(3, 100)
    mf = MatrixFrame(data, 1000)
    cf = mf.to_channel_frame()
    mf2 = MatrixFrame.from_channel_frame(cf)
    assert isinstance(mf2, MatrixFrame)
    assert mf2.data.shape == (3, 100)


def test_matrix_frame_coherence() -> None:
    # ランダムシードの設定
    rng = np.random.default_rng()

    # サンプリング周波数とサンプル数
    fs = 10e3
    N = int(1e5)  # noqa: N806

    # 信号のパラメータ
    amp = 20
    freq = 1234.0
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs

    # 信号の生成
    b, a = signal.butter(2, 0.25, "low")  # type: ignore [type-arg, unused-ignore]
    x = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    y = signal.lfilter(b, a, x)
    x += amp * np.sin(2 * np.pi * freq * time)
    y += rng.normal(scale=0.1 * np.sqrt(noise_power), size=time.shape)
    z = np.random.randn(N)

    # 信号をスタックして多次元配列に
    xy = np.stack([x, y, z], axis=0)  # 形状: (3, N)

    # コヒーレンスの計算
    f, Cxy = signal.coherence(  # noqa: N806
        xy[:, None, :],  # 形状: (3, 1, N)
        xy[None, :, :],  # 形状: (1, 3, N)
        fs=fs,
        nperseg=1024,
    )  # Cxy の形状: (3, 3, 周波数数)

    # コヒーレンスデータを2次元にリシェイプ
    Cxy_reshaped = Cxy.reshape(-1, Cxy.shape[-1])  # noqa: N806

    matrix_frame = MatrixFrame(
        data=xy,
        sampling_rate=int(fs),
    )

    spectrums = matrix_frame.coherence(win_length=1024)
    for i in range(Cxy_reshaped.shape[0]):
        assert np.allclose(spectrums._channels[i].data, Cxy_reshaped[i], atol=1e-5), (
            f"Expected {Cxy_reshaped[i]}, but got {(spectrums._channels[i].data,)}"
        )


def test_matrix_frame_csd() -> None:
    # ランダムシードの設定
    rng = np.random.default_rng()

    # サンプリング周波数とサンプル数
    fs = 10e3
    N = int(1e5)  # noqa: N806

    # 信号のパラメータ
    amp = 20
    freq = 1234.0
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs

    # 信号の生成
    b, a = signal.butter(2, 0.25, "low")  # type: ignore [type-arg, unused-ignore]
    x = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    y = signal.lfilter(b, a, x)
    x += amp * np.sin(2 * np.pi * freq * time)
    y += rng.normal(scale=0.1 * np.sqrt(noise_power), size=time.shape)
    z = np.random.randn(N)

    # 信号をスタックして多次元配列に
    xy = np.stack([x, y, z], axis=0)  # 形状: (3, N)

    # コヒーレンスの計算
    f, Cxy = signal.csd(  # noqa: N806
        xy[:, None, :],  # 形状: (3, 1, N)
        xy[None, :, :],  # 形状: (1, 3, N)
        fs=fs,
        nperseg=1024,
        scaling="spectrum",
    )  # Cxy の形状: (3, 3, 周波数数)

    # コヒーレンスデータを2次元にリシェイプ
    Cxy_reshaped = np.sqrt(Cxy.reshape(-1, Cxy.shape[-1]))  # noqa: N806

    matrix_frame = MatrixFrame(
        data=xy,
        sampling_rate=int(fs),
    )

    spectrums = matrix_frame.csd(win_length=1024)
    for i in range(Cxy_reshaped.shape[0]):
        assert np.allclose(spectrums._channels[i].data, Cxy_reshaped[i], atol=1e-5), (
            f"Expected {Cxy_reshaped[i]}, but got {(spectrums._channels[i].data,)}"
        )


# def test_matrix_frame_plot(mocker):
#     data = np.random.rand(3, 100)
#     mf = MatrixFrame(data, 1000)
#     mocker.patch("wandas.core.signal.ChannelFrame.plot")
#     mf.plot()
#     wandas.core.signal.ChannelFrame.plot.assert_called_once()
