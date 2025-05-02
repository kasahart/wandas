from typing import TYPE_CHECKING

import librosa
import numpy as np
from scipy.signal.windows import tukey

if TYPE_CHECKING:
    from wandas.utils.types import NDArrayReal


def unit_to_ref(unit: str) -> float:
    """
    Convert unit to reference value.

    Parameters
    ----------
    unit : str
        Unit string.

    Returns
    -------
    float
        Reference value for the unit. For 'Pa', returns 2e-5 (20 μPa).
        For other units, returns 1.0.
    """
    if unit == "Pa":
        return 2e-5

    else:
        return 1.0


def calculate_rms(wave: "NDArrayReal") -> "NDArrayReal":
    """
    Calculate the root mean square of the wave.

    Parameters
    ----------
    wave : NDArrayReal
        Input waveform data. Can be multi-channel (shape: [channels, samples])
        or single channel (shape: [samples]).

    Returns
    -------
    Union[float, NDArray[np.float64]]
        RMS value(s). For multi-channel input, returns an array of RMS values,
        one per channel. For single-channel input, returns a single RMS value.
    """
    # Calculate RMS considering axis (over the last dimension)
    axis_to_use = -1 if wave.ndim > 1 else None
    rms_values: NDArrayReal = np.sqrt(
        np.mean(np.square(wave), axis=axis_to_use, keepdims=True)
    )
    return rms_values


def calculate_desired_noise_rms(clean_rms: "NDArrayReal", snr: float) -> "NDArrayReal":
    """
    Calculate the desired noise RMS based on clean signal RMS and target SNR.

    Parameters
    ----------
    clean_rms : "NDArrayReal"
        RMS value(s) of the clean signal.
        Can be a single value or an array for multi-channel.
    snr : float
        Target Signal-to-Noise Ratio in dB.

    Returns
    -------
    "NDArrayReal"
        Desired noise RMS value(s) to achieve the target SNR.
    """
    a = snr / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms


def amplitude_to_db(amplitude: "NDArrayReal", ref: float) -> "NDArrayReal":
    """
    Convert amplitude to decibel.

    Parameters
    ----------
    amplitude : NDArrayReal
        Input amplitude data.
    ref : float
        Reference value for conversion.

    Returns
    -------
    NDArrayReal
        Amplitude data converted to decibels.
    """
    db: NDArrayReal = librosa.amplitude_to_db(
        np.abs(amplitude), ref=ref, amin=1e-15, top_db=None
    )
    return db


def level_trigger(
    data: "NDArrayReal", level: float, offset: int = 0, hold: int = 1
) -> list[int]:
    """
    Find points where the signal crosses the specified level from below.

    Parameters
    ----------
    data : NDArrayReal
        Input signal data.
    level : float
        Threshold level for triggering.
    offset : int, default=0
        Offset to add to trigger points.
    hold : int, default=1
        Minimum number of samples between successive trigger points.

    Returns
    -------
    list of int
        List of sample indices where the signal crosses the level.
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
    """
    Cut segments from signal at specified points.

    Parameters
    ----------
    data : NDArrayReal
        Input signal data.
    point_list : list of int
        List of starting points for cutting.
    cut_len : int
        Length of each segment to cut.
    taper_rate : float, default=0
        Taper rate for Tukey window applied to segments.
        A value of 0 means no tapering, 1 means full tapering.
    dc_cut : bool, default=False
        Whether to remove DC component (mean) from segments.

    Returns
    -------
    NDArrayReal
        Array containing cut segments with shape (n_segments, cut_len).
    """
    length = len(data)
    point_list_ = [p for p in point_list if p >= 0 and p + cut_len <= length]
    trial = np.zeros((len(point_list_), cut_len))

    for i, v in enumerate(point_list_):
        trial[i] = data[v : v + cut_len]
        if dc_cut:
            trial[i] = trial[i] - trial[i].mean()

    trial = trial * tukey(cut_len, taper_rate)
    return trial
