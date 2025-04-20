from typing import TYPE_CHECKING, Any, Optional, Union

import librosa
import numpy as np
from scipy.signal import butter, filtfilt
from waveform_analysis import A_weight

from wandas.core import util
from wandas.utils.types import NDArrayComplex, NDArrayReal

from .frequency_channel import FrequencyChannel, NOctChannel
from .time_frequency_channel import TimeFrequencyChannel

if TYPE_CHECKING:
    from .channel import Channel


def apply_add(ch1: "Channel", ch2: "Channel", snr: float) -> "Channel":
    if ch1.sampling_rate != ch2.sampling_rate:
        raise ValueError("Sampling rates of the two channels are different.")

    if ch1.data.shape != ch2.data.shape:
        raise ValueError("Data shapes of the two channels are different.")

    other_rms = util.calculate_rms(ch2.data)
    if other_rms == 0:
        raise ValueError("RMS of the noise channel is zero.")

    clean_rms = util.calculate_rms(ch1.data)
    desired_noise_rms = util.calculate_desired_noise_rms(clean_rms, snr)
    gain = desired_noise_rms / other_rms

    result: Channel = ch1 + ch2 * gain
    return result


def apply_filter(
    ch: "Channel",
    cutoff: Union[float, int],
    filter_type: str,
    order: int = 5,
) -> dict[str, Any]:
    nyq = 0.5 * ch.sampling_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)  # type: ignore[unused-ignore]
    filtered = filtfilt(b, a, ch.data)
    if isinstance(filtered, np.ndarray):
        return dict(
            data=filtered,
        )
    else:
        raise ValueError("Filtered data is not a ndarray.")


def apply_hpss_harmonic(
    ch: "Channel",
    **kwargs: Any,
) -> dict[str, NDArrayReal]:
    harmonic = librosa.effects.harmonic(ch.data, **kwargs)
    result = dict(
        data=harmonic,
    )
    return result


def apply_hpss_percussive(
    ch: "Channel",
    **kwargs: Any,
) -> dict[str, NDArrayReal]:
    percussive = librosa.effects.percussive(ch.data, **kwargs)
    result = dict(
        data=percussive,
    )
    return result


def compute_fft(
    ch: "Channel", n_fft: Optional[int] = None, window: Optional[str] = None
) -> dict[str, NDArrayComplex]:
    result = FrequencyChannel.fft(data=ch.data, n_fft=n_fft, window=window)

    return result


def compute_welch(
    ch: "Channel",
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: int = 2048,
    window: str = "hann",
    average: str = "mean",
    # pad_mode: str = "constant"
) -> dict[str, NDArrayReal]:
    result = FrequencyChannel.welch(
        data=ch.data,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        average=average,
    )
    return result


def compute_octave(
    ch: "Channel",
    n_octaves: int = 3,
    fmin: float = 20,
    fmax: float = 20000,
    G: int = 10,  # noqa: N803
    fr: int = 1000,
) -> dict[str, NDArrayReal]:
    result = NOctChannel.noct_spectrum(
        data=ch.data,
        sampling_rate=ch.sampling_rate,
        fmin=fmin,
        fmax=fmax,
        n=n_octaves,
        G=G,
        fr=fr,
    )
    return result


def compute_stft(
    ch: "Channel",
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    # pad_mode: str = "constant",
) -> dict[str, Union[NDArrayReal, NDArrayComplex]]:
    result = TimeFrequencyChannel.stft(
        data=ch.data,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        # center=center,
        # pad_mode=pad_mode,
    )
    return result


def compute_rms_trend(
    ch: "Channel",
    frame_length: int = 2048,
    hop_length: int = 512,
    Aw: bool = False,  # noqa: N803
) -> dict[str, Any]:
    data: NDArrayReal = ch.data
    if Aw:
        data = np.array(A_weight(data, ch.sampling_rate))

    rms_data = librosa.feature.rms(
        y=data, frame_length=frame_length, hop_length=hop_length
    )
    result = dict(
        data=rms_data.squeeze(),
        sampling_rate=int(ch.sampling_rate / hop_length),
    )
    return result


def apply_normalize(
    ch: "Channel",
    target_level: float = -20,
    channel_wise: bool = True,
) -> dict[str, NDArrayReal]:
    """
    チャンネル信号を正規化します。

    Parameters:
        ch (Channel): 正規化するチャンネル。
        target_level (float): 目標信号レベル（dB）。
        channel_wise (bool): チャンネルごとに正規化するかどうか。

    Returns:
        dict[str, NDArrayReal]: 正規化されたデータを含む辞書。
    """
    data = ch.data

    # RMSを計算
    current_rms = util.calculate_rms(data)

    # 現在のdBレベルを計算（基準は1.0）
    current_db = 20 * np.log10(current_rms) if current_rms > 0 else -100

    # 必要なゲインを計算
    gain = 10 ** ((target_level - current_db) / 20)

    # 正規化された信号を作成
    normalized_data = data * gain

    return dict(
        data=normalized_data,
    )
