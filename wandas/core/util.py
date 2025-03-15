from typing import TYPE_CHECKING, Any, TypeVar

import librosa
import numpy as np
from scipy.signal.windows import tukey

if TYPE_CHECKING:
    from wandas.core.base_channel import BaseChannel
    from wandas.utils.types import NDArrayReal
T = TypeVar("T", bound="BaseChannel")


def transform_channel(org: "BaseChannel", target_class: type[T], **kwargs: Any) -> T:
    # データ変換を実行
    return target_class(
        data=kwargs.pop("data"),
        sampling_rate=kwargs.pop("sampling_rate", org.sampling_rate),
        label=kwargs.pop("label", org.label),
        unit=kwargs.pop("unit", org.unit),
        metadata=kwargs.pop("metadata", org.metadata.copy()),
        **kwargs,  # target_classに必要な追加の引数
    )


def unit_to_ref(unit: str) -> float:
    """
    単位を参照値に変換します。
    """
    if unit == "Pa":
        return 2e-5

    else:
        return 1.0


def calculate_rms(wave: "NDArrayReal") -> float:
    """
    Calculate the root mean square of the wave.
    """
    return float(np.sqrt(np.mean(np.square(wave))))


def calculate_desired_noise_rms(clean_rms: float, snr: float) -> float:
    a = snr / 20
    noise_rms = clean_rms / 10**a
    return noise_rms


def amplitude_to_db(amplitude: "NDArrayReal", ref: float) -> "NDArrayReal":
    """
    Convert amplitude to decibel.
    """
    db: NDArrayReal = librosa.amplitude_to_db(
        np.abs(amplitude), ref=ref, amin=1e-15, top_db=300
    )
    return db


def level_trigger(
    data: "NDArrayReal", level: float, offset: int = 0, hold: int = 1
) -> list[int]:
    """
    Level trigger
    """
    trig_point: list[int] = []

    sig_len = len(data)
    diff = np.diff(np.sign(data - level))
    level_point = np.where(diff > 0)[0]
    level_point = level_point[(level_point + hold) < sig_len]

    if len(level_point) == 0:
        return list()

    last_point = level_point[0]
    trig_point.append(last_point + offset)
    for i in level_point:
        if (last_point + hold) < i:
            trig_point.append(i + offset)
            last_point = i

    return trig_point


def cut_sig(
    data: "NDArrayReal",
    point_list: list[int],
    cut_len: int,
    taper_rate: float = 0,
    dc_cut: bool = False,
) -> "NDArrayReal":
    length = len(data)
    point_list_ = [p for p in point_list if p >= 0 and p + cut_len <= length]
    trial = np.zeros((len(point_list_), cut_len))

    for i, v in enumerate(point_list_):
        trial[i] = data[v : v + cut_len]
        if dc_cut:
            trial[i] = trial[i] - trial[i].mean()

    trial = trial * tukey(cut_len, taper_rate)
    return trial
