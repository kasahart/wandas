# wandas/core/channel.py

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import ipywidgets as widgets
import numpy as np
from IPython.display import Audio, display
from waveform_analysis import A_weight

from wandas.core import channel_processing, util
from wandas.core.arithmetic import ArithmeticMixin
from wandas.io import wav_io
from wandas.utils.types import NDArrayReal

from .base_channel import BaseChannel
from .channel_plotter import ChannelPlotter
from .frequency_channel import FrequencyChannel, NOctChannel
from .time_frequency_channel import TimeFrequencyChannel, TimeMelFrequencyChannel

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class Channel(BaseChannel, ArithmeticMixin):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        previous: Optional["Channel"] = None,
    ):
        """
        Initialize a Channel object.

        Parameters
        ----------
        data : numpy.ndarray
            Time series data.
        sampling_rate : int
            Sampling rate (Hz).
        label : str, optional
            Channel label.
        unit : str, optional
            Unit of measurement.
        metadata : dict, optional
            Additional metadata.
        previous : Channel, optional
            Reference to the original channel before transformation.
        """
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            unit=unit,
            metadata=metadata,
            previous=previous,
        )

    @property
    def time(self) -> NDArrayReal:
        """
        Returns the time data.

        Returns
        -------
        NDArrayReal
            Array containing time values in seconds.
        """
        num_samples = len(self._data)
        return np.arange(num_samples) / self.sampling_rate

    def trim(self, start: float, end: float) -> "Channel":
        """
        Extract data within the specified time range.

        Parameters
        ----------
        start : float
            Start time (seconds).
        end : float
            End time (seconds).

        Returns
        -------
        Channel
            New Channel object containing the extracted data.
        """
        start_idx = int(start * self.sampling_rate)
        end_idx = int(end * self.sampling_rate)
        data = self.data[start_idx:end_idx]

        return Channel.from_channel(self, data=data)

    def trigger(
        self,
        threshold: float,
        offset: int = 0,
        hold: int = 1,
        trigger_type: str = "level",
    ) -> list[int]:
        """
        Detect triggers in the signal.

        Parameters
        ----------
        threshold : float
            Trigger threshold.
        offset : int, default=0
            Offset for trigger detection position.
        hold : int, default=1
            Trigger hold.
        trigger_type : str, default="level"
            Trigger type:
            - "level": Level trigger

        Returns
        -------
        list[int]
            List of trigger positions.
        """
        if trigger_type == "level":
            return util.level_trigger(self.data, threshold, offset=offset, hold=hold)
        else:
            raise ValueError(f"Unsupported trigger type: {trigger_type}")

    def cut(
        self,
        point_list: Union[list[int], list[float]],
        cut_len: Union[int, float],
        taper_rate: float = 0,
        dc_cut: bool = False,
    ) -> list["Channel"]:
        """
        Cut channel data at specified points.

        Parameters
        ----------
        point_list : list[int] or list[float]
            List of cutting points. If floats, treated as time in seconds.
        cut_len : int or float
            Length of data to cut. If float, treated as time in seconds.
        taper_rate : float, default=0
            Taper rate.
        dc_cut : bool, default=False
            DC cut.

        Returns
        -------
        list[Channel]
            List of Channel objects containing the cut data.
        """
        # Convert float points to integer sample indices
        _point_list: list[int] = [
            int(p * self.sampling_rate) if isinstance(p, float) else p
            for p in point_list
        ]
        # Convert float cut_len to integer samples
        _cut_len = (
            int(cut_len * self.sampling_rate) if isinstance(cut_len, float) else cut_len
        )
        data = util.cut_sig(self.data, _point_list, _cut_len, taper_rate, dc_cut)
        return [Channel.from_channel(self, data=d) for d in data]

    def high_pass_filter(self, cutoff: float, order: int = 5) -> "Channel":
        """
        Apply a high-pass filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency (Hz).
        order : int, default=5
            Filter order.

        Returns
        -------
        Channel
            New Channel object containing the filtered data.
        """
        result = channel_processing.apply_filter(
            ch=self,
            cutoff=cutoff,
            order=order,
            filter_type="highpass",
        )
        return Channel.from_channel(self, **result)

    def low_pass_filter(self, cutoff: float, order: int = 5) -> "Channel":
        """
        Apply a low-pass filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency (Hz).
        order : int, default=5
            Filter order.

        Returns
        -------
        Channel
            New Channel object containing the filtered data.
        """
        result = channel_processing.apply_filter(
            ch=self,
            cutoff=cutoff,
            order=order,
            filter_type="lowpass",
        )
        return Channel.from_channel(self, **result)

    def a_weighting(self) -> "Channel":
        """
        Apply A-weighting filter.

        Returns
        -------
        Channel
            New Channel object containing the A-weighted data.
        """
        data: NDArrayReal = np.array(A_weight(signal=self.data, fs=self.sampling_rate))

        return Channel.from_channel(self, data=data, unit="dB(A)")

    def hpss_harmonic(
        self,
        kernel_size: Union[int, tuple[int, int], list[int]] = 31,
        power: float = 2.0,
        mask: bool = False,
        margin: Union[float, tuple[float, float], list[float]] = 1.0,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Union[str, NDArrayReal] = "hann",
        center: bool = True,
        pad_mode: Union[
            Literal["constant", "edge", "linear_ramp", "reflect", "symmetric", "empty"],
            Callable[..., Any],
        ] = "constant",
    ) -> "Channel":
        """
        Extract the harmonic component using HPSS
        (Harmonic-Percussive Source Separation).

        Parameters
        ----------
        kernel_size : int or tuple[int, int] or list[int], default=31
            Size of the median filter kernel.
        power : float, default=2.0
            Exponent for the Wiener filter when constructing soft mask matrices.
        mask : bool, default=False
            Return the mask matrices instead of components.
        margin : float or tuple[float, float] or list[float], default=1.0
            Margin size for the masks.
        n_fft : int, default=2048
            FFT window size.
        hop_length : int, optional
            Number of samples between successive frames.
        win_length : int, optional
            Window size. If None, defaults to n_fft.
        window : str or NDArrayReal, default="hann"
            Window function.
        center : bool, default=True
            If True, the input is padded on both sides so that frames are centered.
        pad_mode : str or Callable, default="constant"
            Padding mode for centered frames.

        Returns
        -------
        Channel
            New Channel object containing the harmonic component.
        """
        result = channel_processing.apply_hpss_harmonic(
            ch=self,
            kernel_size=kernel_size,
            power=power,
            mask=mask,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return Channel.from_channel(self, **result)

    def hpss_percussive(
        self,
        kernel_size: Union[int, tuple[int, int], list[int]] = 31,
        power: float = 2.0,
        mask: bool = False,
        margin: Union[float, tuple[float, float], list[float]] = 1.0,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Union[str, NDArrayReal] = "hann",
        center: bool = True,
        pad_mode: Union[
            Literal["constant", "edge", "linear_ramp", "reflect", "symmetric", "empty"],
            Callable[..., Any],
        ] = "constant",
    ) -> "Channel":
        """
        Extract the percussive component using HPSS
        (Harmonic-Percussive Source Separation).

        Parameters
        ----------
        kernel_size : int or tuple[int, int] or list[int], default=31
            Size of the median filter kernel.
        power : float, default=2.0
            Exponent for the Wiener filter when constructing soft mask matrices.
        mask : bool, default=False
            Return the mask matrices instead of components.
        margin : float or tuple[float, float] or list[float], default=1.0
            Margin size for the masks.
        n_fft : int, default=2048
            FFT window size.
        hop_length : int, optional
            Number of samples between successive frames.
        win_length : int, optional
            Window size. If None, defaults to n_fft.
        window : str or NDArrayReal, default="hann"
            Window function.
        center : bool, default=True
            If True, the input is padded on both sides so that frames are centered.
        pad_mode : str or Callable, default="constant"
            Padding mode for centered frames.

        Returns
        -------
        Channel
            New Channel object containing the percussive component.
        """
        result = channel_processing.apply_hpss_percussive(
            ch=self,
            kernel_size=kernel_size,
            power=power,
            mask=mask,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return Channel.from_channel(self, **result)

    def fft(
        self,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
    ) -> "FrequencyChannel":
        """
        Perform Fourier transform.

        Parameters
        ----------
        n_fft : int, optional
            Number of FFT samples.
        window : str, optional
            Type of window function.

        Returns
        -------
        FrequencyChannel
            Object containing the spectrum data.
        """
        result = channel_processing.compute_fft(
            ch=self,
            n_fft=n_fft,
            window=window,
        )

        return FrequencyChannel.from_channel(self, **result)

    def welch(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        average: str = "mean",
        # pad_mode: str = "constant"
    ) -> "FrequencyChannel":
        """
        Perform power spectral density estimation using Welch's method.

        Parameters
        ----------
        n_fft : int, optional
            FFT size.
        hop_length : int, optional
            Number of samples between successive frames.
        win_length : int, default=2048
            Size of each segment.
        window : str, default="hann"
            Window function.
        average : str, default="mean"
            Method for averaging the segments.

        Returns
        -------
        FrequencyChannel
            Object containing the spectrum data.
        """
        result = channel_processing.compute_welch(
            ch=self,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            average=average,
        )
        return FrequencyChannel.from_channel(self, **result)

    def noct_spectrum(
        self,
        n_octaves: int = 3,
        fmin: float = 20,
        fmax: float = 20000,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> "NOctChannel":
        """
        Calculate octave band spectrum.

        Parameters
        ----------
        n_octaves : int, default=3
            Number of octaves.
        fmin : float, default=20
            Minimum frequency (Hz).
        fmax : float, default=20000
            Maximum frequency (Hz).
        G : int, default=10
            Band number of the reference frequency band.
        fr : int, default=1000
            Reference frequency (Hz).

        Returns
        -------
        NOctChannel
            Object containing the octave band spectrum data.
        """

        result = channel_processing.compute_octave(
            ch=self,
            n_octaves=n_octaves,
            fmin=fmin,
            fmax=fmax,
            G=G,
            fr=fr,
        )
        return NOctChannel.from_channel(self, **result)

    def stft(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        # pad_mode: str = "constant",
    ) -> "TimeFrequencyChannel":
        """
        Perform Short-Time Fourier Transform (STFT).

        Parameters
        ----------
        n_fft : int, default=2048
            FFT size.
        hop_length : int, optional
            Hop size (number of samples between successive frames).
        win_length : int, optional
            Window length. Defaults to n_fft if None.
        window : str, default="hann"
            Window function.
        center : bool, default=True
            If True, the signal is padded so that frames are centered.

        Returns
        -------
        TimeFrequencyChannel
            Object containing the STFT results.
        """

        result = channel_processing.compute_stft(
            ch=self,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            # pad_mode=pad_mode
        )
        return TimeFrequencyChannel.from_channel(self, **result)

    def melspectrogram(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        window: str = "hann",
        center: bool = True,
        # pad_mode: str = "constant",
    ) -> "TimeMelFrequencyChannel":
        """
        Compute mel spectrogram.

        Parameters
        ----------
        n_mels : int, default=128
            Number of mel bands.
        n_fft : int, default=2048
            FFT size.
        hop_length : int, default=512
            Hop size (number of samples between successive frames).
        win_length : int, default=2048
            Window length.
        window : str, default="hann"
            Window function.
        center : bool, default=True
            If True, the signal is padded so that frames are centered.

        Returns
        -------
        TimeMelFrequencyChannel
            Object containing the mel spectrogram.
        """
        tf_ch = self.stft(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            # center=center,
            # pad_mode=pad_mode,
        )

        return tf_ch.melspectrogram(n_mels=n_mels)

    def rms_trend(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        Aw: bool = False,  # noqa: N803
    ) -> "Channel":
        """
        Calculate RMS energy trend.

        Parameters
        ----------
        frame_length : int, default=2048
            Window size for the RMS calculation.
        hop_length : int, default=512
            Number of samples between successive frames.
        Aw : bool, default=False
            Apply A-weighting before RMS calculation.

        Returns
        -------
        Channel
            Channel object containing the RMS energy trend.
        """
        result = channel_processing.compute_rms_trend(
            ch=self,
            frame_length=frame_length,
            hop_length=hop_length,
            Aw=Aw,  # noqa: N803
        )
        return Channel.from_channel(self, **result)

    def normalize(
        self, target_level: float = -20, channel_wise: bool = True
    ) -> "Channel":
        """
        Normalize signal level.

        Parameters
        ----------
        target_level : float, default=-20
            Target signal level (dB).
        channel_wise : bool, default=True
            Whether to normalize each channel separately.

        Returns
        -------
        Channel
            Normalized Channel object.
        """
        result = channel_processing.apply_normalize(
            ch=self,
            target_level=target_level,
            channel_wise=channel_wise,
        )
        return Channel.from_channel(self, **result)

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> "Axes":
        """
        Plot time-series data.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new one is created.
        title : str, optional
            Plot title.
        plot_kwargs : dict, optional
            Additional keyword arguments for the plot function.

        Returns
        -------
        Axes
            Matplotlib axes containing the plot.
        """
        plotter = ChannelPlotter(self)

        return plotter.plot_time(ax=ax, title=title, plot_kwargs=plot_kwargs)

    def rms_plot(
        self,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        Aw: bool = False,  # noqa: N803
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> "Axes":
        """
        Plot RMS data.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new one is created.
        title : str, optional
            Plot title.
        Aw : bool, default=False
            Apply A-weighting before RMS calculation.
        plot_kwargs : dict, optional
            Additional keyword arguments for the plot function.

        Returns
        -------
        Axes
            Matplotlib axes containing the plot.
        """
        plotter = ChannelPlotter(self)

        return plotter.rms_plot(ax=ax, title=title, Aw=Aw, plot_kwargs=plot_kwargs)

    def __len__(self) -> int:
        """
        Return the length of the channel data.

        Returns
        -------
        int
            Number of samples in the channel.
        """
        return int(self._data.shape[-1])

    def add(
        self, other: Union["Channel", NDArrayReal], snr: Optional[float] = None
    ) -> "Channel":
        """
        Add another channel or array to this channel, optionally with specified SNR.

        Parameters
        ----------
        other : Channel or NDArrayReal
            Channel or array to add.
        snr : float, optional
            Signal-to-noise ratio (dB). If provided, the other signal is scaled
            to achieve this SNR before adding.

        Returns
        -------
        Channel
            Result of the addition.
        """
        if isinstance(other, np.ndarray):
            other = Channel.from_channel(self, data=other, label="ndarray")

        if snr is None:
            return self + other

        return channel_processing.apply_add(self, other, snr)

    def to_wav(self, filename: str) -> None:
        """
        Export the Channel object to a WAV file.

        Parameters
        ----------
        filename : str
            Path to the output WAV file.
        """
        wav_io.write_wav(filename, self)

    def to_audio(self, normalize: bool = True, label: bool = True) -> widgets.VBox:
        """
        Create an audio widget for playback in Jupyter notebooks.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to normalize the audio.
        label : bool, default=True
            Whether to show the channel label.

        Returns
        -------
        widgets.VBox
            Widget containing the audio player.
        """
        output = widgets.Output()
        with output:
            display(Audio(self.data, rate=self.sampling_rate, normalize=normalize))  # type: ignore [unused-ignore, no-untyped-call]

        if label:
            vbov = widgets.VBox([widgets.Label(self.label) if label else None, output])
        else:
            vbov = widgets.VBox([output])
        return vbov

    def describe(
        self,
        axis_config: Optional[dict[str, dict[str, Any]]] = None,
        cbar_config: Optional[dict[str, Any]] = None,
    ) -> widgets.VBox:
        """
        Display channel statistics and visualizations.

        Parameters
        ----------
        axis_config : dict, optional
            Dictionary containing axis settings for each subplot.
            Example:
            {
                "time_plot": {"xlim": (0, 1)},
                "freq_plot": {"ylim": (0, 20000)}
            }
        cbar_config : dict, optional
            Dictionary containing colorbar settings.
            Example: {"vmin": -80, "vmax": 0}

        Returns
        -------
        widgets.VBox
            Widget containing the visualizations.
        """
        plotter = ChannelPlotter(self)
        return plotter.describe(axis_config=axis_config, cbar_config=cbar_config)
