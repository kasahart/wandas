# wandas/core/signal.py
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import ipywidgets as widgets
import numpy as np
import pandas as pd

from wandas.core.channel import Channel
from wandas.core.channel_access_mixin import ChannelAccessMixin
from wandas.io import wav_io
from wandas.utils.types import NDArrayReal

from . import channel_frame_processing as cfp
from .channel_frame_plotter import ChannelFramePlotter

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.core.frequency_channel_frame import FrequencyChannelFrame
    from wandas.core.matrix_frame import MatrixFrame


class ChannelFrame(ChannelAccessMixin["Channel"]):
    def __init__(self, channels: list["Channel"], label: Optional[str] = None):
        """
        Initialize a ChannelFrame object.

        Parameters
        ----------
        channels : list of Channel
            List of Channel objects.
        label : str, optional
            Label for the signal.
        """
        self._channels = channels
        self.label = label

        # Check consistency of sampling rates
        sampling_rates = set(ch.sampling_rate for ch in channels)
        if len(sampling_rates) > 1:
            raise ValueError("All channels must have the same sampling_rate.")

        self.sampling_rate = channels[0].sampling_rate
        self._channel_dict = {ch.label: ch for ch in self.channels}
        if len(self._channel_dict) != len(self):
            raise ValueError("Channel labels must be unique.")

    @classmethod
    def from_ndarray(
        cls,
        array: NDArrayReal,
        sampling_rate: int,
        labels: Optional[list[str]] = None,
        unit: Optional[str] = None,
    ) -> "ChannelFrame":
        """
        Create a ChannelFrame instance from a numpy ndarray.

        Parameters
        ----------
        array : NDArrayReal
            Signal data. Each row corresponds to a channel.
        sampling_rate : int
            Sampling rate (Hz).
        labels : list[str], optional
            Labels for each channel.
        unit : str, optional
            Unit of the signal.

        Returns
        -------
        ChannelFrame
            ChannelFrame object generated from the ndarray.
        """
        channels = []
        num_channels = array.shape[0]

        if labels is None:
            labels = [f"Channel {i + 1}" for i in range(num_channels)]

        for i in range(num_channels):
            channel = Channel(
                data=array[i], sampling_rate=sampling_rate, label=labels[i], unit=unit
            )
            channels.append(channel)

        return cls(channels=channels)

    @classmethod
    def read_wav(
        cls, filename: str, labels: Optional[list[str]] = None
    ) -> "ChannelFrame":
        """
        Read a WAV file and create a ChannelFrame object.

        Parameters
        ----------
        filename : str
            Path to the WAV file.
        labels : list of str, optional
            Labels for each channel.

        Returns
        -------
        ChannelFrame
            ChannelFrame object containing the audio data.
        """
        return wav_io.read_wav(filename, labels)

    def to_wav(self, filename: str) -> None:
        """
        Write the ChannelFrame object to a WAV file.

        Parameters
        ----------
        filename : str
            Path to the output WAV file.
        """
        wav_io.write_wav(filename, self)

    @classmethod
    def read_csv(
        cls,
        filename: str,
        time_column: Union[int, str] = 0,
        labels: Optional[list[str]] = None,
        delimiter: str = ",",
        header: Optional[int] = 0,
    ) -> "ChannelFrame":
        """
        Read a CSV file and create a ChannelFrame object.

        Parameters
        ----------
        filename : str
            Path to the CSV file.
        time_column : int or str, default=0
            Index or name of the time column. Default is the first column.
        labels : list of str, optional
            Labels for each channel.
        delimiter : str, default=","
            Delimiter character.
        header : int or None, default=0
            Row position for the header. None means no header.

        Returns
        -------
        ChannelFrame
            ChannelFrame object containing the data.
        """
        # Load CSV file using pandas
        df = pd.read_csv(filename, delimiter=delimiter, header=header)

        # Calculate sampling rate
        try:
            time_values = (
                df[time_column].values
                if isinstance(time_column, str)
                else df.iloc[:, time_column].values
            )
        except KeyError:
            raise KeyError(f"Time column '{time_column}' not found in the CSV file.")
        except IndexError:
            raise IndexError(f"Time column index {time_column} is out of range.")
        if len(time_values) < 2:
            raise ValueError("Not enough time points to calculate sampling rate.")
        time_values = np.array(time_values)
        sampling_rate: int = int(1 / np.mean(np.diff(time_values)))

        # Remove time column
        df = df.drop(
            columns=[time_column]
            if isinstance(time_column, str)
            else df.columns[time_column]
        )

        # Convert data to NumPy array
        data = df.values  # shape: (num_samples, num_channels)

        # Transpose to have channels as the first dimension
        data = data.T  # shape: (num_channels, num_samples)

        num_channels = data.shape[0]

        # Process labels
        if labels is not None:
            if len(labels) != num_channels:
                raise ValueError("Length of labels must match number of channels.")
        elif header is not None:
            labels = df.columns.tolist()
        else:
            labels = [f"Ch{i}" for i in range(num_channels)]

        # Create Channel objects for each channel
        channels = []
        for i in range(num_channels):
            ch_data = data[i]
            ch_label = labels[i]
            channel = Channel(
                data=ch_data,
                sampling_rate=sampling_rate,
                label=ch_label,
            )
            channels.append(channel)

        return cls(channels=channels)

    def to_audio(self, normalize: bool = True) -> widgets.VBox:
        return widgets.VBox([ch.to_audio(normalize) for ch in self._channels])

    def describe(
        self,
        axis_config: Optional[dict[str, dict[str, Any]]] = None,
        cbar_config: Optional[dict[str, Any]] = None,
    ) -> widgets.VBox:
        """
        Display information about the channels.

        Parameters
        ----------
        axis_config : dict, optional
            Dictionary containing axis settings for each subplot.
            Example: {"time_plot": {"xlim": (0, 1)}, "freq_plot": {"ylim": (0, 20000)}}.
        cbar_config : dict, optional
            Dictionary containing color bar settings.
            Example: {"vmin": -80, "vmax": 0}.
        """
        content = [
            widgets.HTML(
                f"<span style='font-size:20px; font-weight:normal;'>"
                f"{self.label}, {self.sampling_rate} Hz</span>"
            )
        ]
        content += [
            ch.describe(axis_config=axis_config, cbar_config=cbar_config)
            for ch in self._channels
        ]
        # Set layout for center alignment
        layout = widgets.Layout(
            display="flex", justify_content="center", align_items="center"
        )
        return widgets.VBox(content, layout=layout)

    def trim(self, start: float, end: float) -> "ChannelFrame":
        """
        Trim the channels within the specified time range.

        Parameters
        ----------
        start : float
            Start time for trimming (seconds).
        end : float
            End time for trimming (seconds).

        Returns
        -------
        ChannelFrame
            New ChannelFrame object with trimmed channels.
        """
        return cfp.trim_channel_frame(self, start, end)

    def cut(
        self,
        point_list: Union[list[int], list[float]],
        cut_len: Union[int, float],
        taper_rate: float = 0,
        dc_cut: bool = False,
    ) -> list["MatrixFrame"]:
        """
        Cut the channels at specified time points.

        Parameters
        ----------
        point_list : list[int] or list[float]
            List of cut points.
        cut_len : int or float
            Length of data to cut.
        taper_rate : float, optional
            Taper rate.
        dc_cut : bool, optional
            DC cut.

        Returns
        -------
        list of MatrixFrame
            List of new MatrixFrame objects with cut data.
        """
        return cfp.cut_channel_frame(
            cf=self,
            point_list=point_list,
            cut_len=cut_len,
            taper_rate=taper_rate,
            dc_cut=dc_cut,
        )

    def to_matrix_frame(self) -> "MatrixFrame":
        """
        Convert the ChannelFrame object to a MatrixFrame object.

        Returns
        -------
        MatrixFrame
            MatrixFrame object containing channel data.
        """
        from wandas.core.matrix_frame import MatrixFrame

        return MatrixFrame.from_channel_frame(self)

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = True,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union["Axes", Iterable["Axes"]]:
        """
        Plot all channels.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib Axes object.
        title : str, optional
            Title for the plot.
        overlay : bool, optional
            If True, all channels are plotted on the same plot.
            If False, each channel is plotted separately.
        plot_kwargs : dict, optional
            Additional keyword arguments for the plot.

        Returns
        -------
        Axes or Iterable of Axes
            Matplotlib Axes object(s).
        """
        plotter = ChannelFramePlotter(self)

        return plotter.plot_time(
            ax=ax, title=title, overlay=overlay, plot_kwargs=plot_kwargs
        )

    def rms_plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = True,
        Aw: bool = False,  # noqa: N803
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union["Axes", Iterable["Axes"]]:
        """
        Plot RMS data for all channels.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib Axes object.
        title : str, optional
            Title for the plot.
        overlay : bool, optional
            If True, all channels are plotted on the same plot.
            If False, each channel is plotted separately.
        Aw : bool, optional
            Apply A-weighting.
        plot_kwargs : dict, optional
            Additional keyword arguments for the plot.

        Returns
        -------
        Axes or Iterable of Axes
            Matplotlib Axes object(s).
        """
        plotter = ChannelFramePlotter(self)

        return plotter.rms_plot(
            ax=ax, title=title, overlay=overlay, Aw=Aw, plot_kwargs=plot_kwargs
        )

    def high_pass_filter(self, cutoff: float, order: int = 5) -> "ChannelFrame":
        """
        Apply a high-pass filter to all channels.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency (Hz).
        order : int, optional
            Filter order.

        Returns
        -------
        ChannelFrame
            New ChannelFrame object with filtered channels.
        """
        filtered_channels = [ch.high_pass_filter(cutoff, order) for ch in self]
        return ChannelFrame(filtered_channels, label=self.label)

    def low_pass_filter(self, cutoff: float, order: int = 5) -> "ChannelFrame":
        """
        Apply a low-pass filter to all channels.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency (Hz).
        order : int, optional
            Filter order.

        Returns
        -------
        ChannelFrame
            New ChannelFrame object with filtered channels.
        """
        filtered_channels = [ch.low_pass_filter(cutoff, order) for ch in self]
        return ChannelFrame(filtered_channels, label=self.label)

    def a_weighting(self) -> "ChannelFrame":
        """
        Apply A-weighting to all channels.

        Returns
        -------
        ChannelFrame
            New ChannelFrame object with A-weighted channels.
        """
        weighted_channels = [ch.a_weighting() for ch in self]
        return ChannelFrame(weighted_channels, label=self.label)

    def hpss_harmonic(self, **kwargs: Any) -> "ChannelFrame":
        """
        Extract harmonic components using HPSS
        (Harmonic-Percussive Source Separation).

        Returns
        -------
        ChannelFrame
            New ChannelFrame object with harmonic components.
        """
        harmonic_channels = [ch.hpss_harmonic(**kwargs) for ch in self]
        return ChannelFrame(harmonic_channels, label=self.label)

    def hpss_percussive(self, **kwargs: Any) -> "ChannelFrame":
        """
        Extract percussive components using HPSS
        (Harmonic-Percussive Source Separation).

        Returns
        -------
        ChannelFrame
            New ChannelFrame object with percussive components.
        """
        percussive_channels = [ch.hpss_percussive(**kwargs) for ch in self]
        return ChannelFrame(percussive_channels, label=self.label)

    def fft(
        self,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
    ) -> "FrequencyChannelFrame":
        """
        Apply Fourier Transform to all channels.

        Parameters
        ----------
        n_fft : int, optional
            Number of FFT points.
        window : str, optional
            Window function.

        Returns
        -------
        FrequencyChannelFrame
            FrequencyChannelFrame object containing frequency and amplitude data.
        """
        from wandas.core.frequency_channel_frame import FrequencyChannelFrame

        chs = [ch.fft(n_fft=n_fft, window=window) for ch in self]

        return FrequencyChannelFrame(
            channels=chs,
            label=self.label,
        )

    def welch(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        average: str = "mean",
    ) -> "FrequencyChannelFrame":
        """
        Estimate power spectral density using Welch's method.

        Parameters
        ----------
        n_fft : int, optional
            Number of FFT points.
        hop_length : int, optional
            Hop length.
        win_length : int, default=2048
            Window length.
        window : str, default="hann"
            Window function.
        average : str, default="mean"
            Averaging method.

        Returns
        -------
        FrequencyChannelFrame
            FrequencyChannelFrame object containing frequency and amplitude data.
        """
        from wandas.core.frequency_channel_frame import FrequencyChannelFrame

        chs = [
            ch.welch(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                average=average,
            )
            for ch in self
        ]

        return FrequencyChannelFrame(
            channels=chs,
            label=self.label,
        )

    def normalize(
        self, target_level: float = -20, channel_wise: bool = True
    ) -> "ChannelFrame":
        """
        Normalize signal levels.

        Parameters
        ----------
        target_level : float, default=-20
            Target signal level (dB).
        channel_wise : bool, default=True
            Whether to normalize each channel individually.

        Returns
        -------
        ChannelFrame
            New ChannelFrame object with normalized channels.
        """
        normalized_channels = [ch.normalize(target_level, channel_wise) for ch in self]
        return ChannelFrame(normalized_channels, label=self.label)

    def _op(
        self,
        other: "ChannelFrame",
        op: Callable[["Channel", "Channel"], "Channel"],
        symbol: str,
    ) -> "ChannelFrame":
        """
        Perform an operation between two ChannelFrame objects.

        Parameters
        ----------
        other : ChannelFrame
            Another ChannelFrame object.
        op : Callable
            Operation to perform.
        symbol : str
            Symbol representing the operation.

        Returns
        -------
        ChannelFrame
            New ChannelFrame object resulting from the operation.
        """
        assert len(self) == len(other), (
            "ChannelFrame must have the same number of channels."
        )

        channels: list[Channel] = [op(self[i], other[i]) for i in range(len(self))]

        return ChannelFrame(
            channels=channels, label=f"({self.label} {symbol} {other.label})"
        )

    # Operator overloading
    def __add__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        Addition between signals.

        Parameters
        ----------
        other : ChannelFrame
            Another ChannelFrame object.

        Returns
        -------
        ChannelFrame
            New ChannelFrame object resulting from addition.
        """
        return self._op(other, lambda a, b: a + b, "+")

    def __sub__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        Subtraction between signals.

        Parameters
        ----------
        other : ChannelFrame
            Another ChannelFrame object.

        Returns
        -------
        ChannelFrame
            New ChannelFrame object resulting from subtraction.
        """
        return self._op(other, lambda a, b: a - b, "-")

    def __mul__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        Multiplication between signals.

        Parameters
        ----------
        other : ChannelFrame
            Another ChannelFrame object.

        Returns
        -------
        ChannelFrame
            New ChannelFrame object resulting from multiplication.
        """
        return self._op(other, lambda a, b: a * b, "*")

    def __truediv__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        Division between signals.

        Parameters
        ----------
        other : ChannelFrame
            Another ChannelFrame object.

        Returns
        -------
        ChannelFrame
            New ChannelFrame object resulting from division.
        """
        return self._op(other, lambda a, b: a / b, "/")

    def sum(self) -> "Channel":
        """
        Sum all channels.

        Returns
        -------
        Channel
            Channel object resulting from summation.
        """
        data = np.stack([ch.data for ch in self._channels]).sum(axis=0)
        return Channel.from_channel(self._channels[0], data=data.squeeze())

    def mean(self) -> "Channel":
        """
        Calculate the mean of all channels.

        Returns
        -------
        Channel
            Channel object resulting from averaging.
        """
        data = np.stack([ch.data for ch in self._channels]).mean(axis=0)
        return Channel.from_channel(self._channels[0], data=data.squeeze())

    def channel_difference(self, other_channel: int = 0) -> "ChannelFrame":
        """
        Calculate the difference between channels.

        Parameters
        ----------
        other_channel : int, default=0
            Index of the channel to subtract from.

        Returns
        -------
        ChannelFrame
            New ChannelFrame object with channel differences.
        """
        channels = [ch - self._channels[other_channel] for ch in self._channels]
        return ChannelFrame(channels=channels, label=f"(ch[*] - ch[{other_channel}])")
