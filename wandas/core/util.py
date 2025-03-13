from typing import TYPE_CHECKING, Any, TypeVar

import librosa
import numpy as np

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
