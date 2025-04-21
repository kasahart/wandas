# wandas/core/matrix_frame.py

from collections.abc import Iterator
from typing import Any, Optional, Union

import numpy as np
import scipy.signal as ss

from wandas.core.channel import Channel
from wandas.core.channel_frame import ChannelFrame
from wandas.core.frequency_channel_frame import FrequencyChannelFrame
from wandas.utils.types import NDArrayReal

from .frequency_channel import FrequencyChannel


class MatrixFrame:
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        channel_units: Optional[list[str]] = None,
        channel_labels: Optional[list[str]] = None,
        channel_metadata: Optional[list[dict[str, Any]]] = None,
        label: Optional[str] = None,
    ):
        """
        Initialize a MatrixFrame object.

        Parameters
        ----------
        data : NDArrayReal
            A multi-dimensional array with shape (num_channels, num_samples).
        sampling_rate : int
            Sampling rate (Hz).
        channel_units : list of str, optional
            Units for each channel.
        channel_labels : list of str, optional
            Labels for each channel.
        channel_metadata : list of dict, optional
            Metadata for each channel.
        label : str, optional
            Label for the MatrixFrame.
        """
        if data.ndim != 2:
            raise ValueError(
                "Data must be a 2D NumPy array with shape (num_channels, num_samples)."
            )

        self.data = data  # shape: (num_channels, num_samples)
        self.sampling_rate = sampling_rate
        self.label = label

        num_channels = data.shape[0]

        # Process units
        if channel_units is not None:
            if len(channel_units) != num_channels:
                raise ValueError(
                    "Length of channel_units must match number of channels."
                )
        else:
            channel_units = ["" for i in range(num_channels)]

        # Process labels
        if channel_labels is not None:
            if len(channel_labels) != num_channels:
                raise ValueError(
                    "Length of channel_labels must match number of channels."
                )
        else:
            channel_labels = [f"Ch{i}" for i in range(num_channels)]

        # Process metadata
        if channel_metadata is not None:
            if len(channel_metadata) != num_channels:
                raise ValueError(
                    "Length of channel_metadata must match number of channels."
                )
        else:
            channel_metadata = [{} for _ in range(num_channels)]

        # Create list of BaseChannel objects
        self._channels = [
            Channel(
                data=np.array([]),
                sampling_rate=sampling_rate,
                unit=unit,
                label=label,
                metadata=metadata,
            )
            for unit, label, metadata in zip(
                channel_units, channel_labels, channel_metadata
            )
        ]

        # Create mapping from labels to indices
        self.label_to_index = {ch.label: idx for idx, ch in enumerate(self._channels)}

    def __len__(self) -> int:
        """
        Return the number of channels.

        Returns
        -------
        int
            Number of channels.
        """
        return int(self.data.shape[0])

    def __iter__(self) -> Iterator["Channel"]:
        """
        Iterate through the channels.

        Returns
        -------
        Iterator[Channel]
            Iterator of Channel objects.
        """
        for idx in range(self.data.shape[0]):
            yield self[idx]

    def __getitem__(self, key: Union[int, str]) -> "Channel":
        """
        Get a channel by index or label.

        Parameters
        ----------
        key : int or str
            Index or label of the channel.

        Returns
        -------
        Channel
            Corresponding Channel object.

        Raises
        ------
        IndexError
            If the index is out of range.
        KeyError
            If the label is not found.
        TypeError
            If the key is neither an integer nor a string.
        """
        if isinstance(key, int):
            # Access by index
            if key < 0 or key >= self.data.shape[0]:
                raise IndexError("Channel index out of range.")
            idx = key
        elif isinstance(key, str):
            # Access by label
            if key not in self.label_to_index:
                raise KeyError(f"Channel label '{key}' not found.")
            idx = self.label_to_index[key]
        else:
            raise TypeError("Key must be an integer index or a string label.")

        # Get channel data and metadata
        ch = self._channels[idx]

        # Create and return Channel object
        return Channel.from_channel(ch, data=self.data[idx].copy())

    def to_channel_frame(self) -> "ChannelFrame":
        """
        Convert to a ChannelFrame object.

        Returns
        -------
        ChannelFrame
            Converted ChannelFrame object.
        """
        return ChannelFrame(
            channels=[ch for ch in self],
            label=self.label,
        )

    @classmethod
    def from_channel_frame(cls, cf: "ChannelFrame") -> "MatrixFrame":
        """
        Convert a ChannelFrame object to a MatrixFrame object.

        Parameters
        ----------
        cf : ChannelFrame
            Source ChannelFrame object.

        Returns
        -------
        MatrixFrame
            Converted MatrixFrame object.

        Raises
        ------
        ValueError
            If all channels don't have the same length.
        """
        # Check if all channel data have the same length
        length = len(cf[0].data)
        if not all([len(ch.data) == length for ch in cf]):
            raise ValueError("All channels must have the same length.")

        return MatrixFrame(
            data=np.array([ch.data for ch in cf]),
            sampling_rate=cf.sampling_rate,
            channel_units=[ch.unit for ch in cf],
            channel_labels=[ch.label for ch in cf],
            channel_metadata=[ch.metadata for ch in cf],
            label=cf.label,
        )

    def coherence(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        detrend: str = "constant",
    ) -> "FrequencyChannelFrame":
        """
        Perform coherence estimation.

        Parameters
        ----------
        n_fft : int, optional
            Number of FFT points. If None, defaults to win_length.
        hop_length : int, optional
            Number of samples between successive frames.
            If None, defaults to win_length//2.
        win_length : int, default=2048
            Window size.
        window : str, default="hann"
            Window function.
        detrend : str, default="constant"
            Type of detrending.

        Returns
        -------
        FrequencyChannelFrame
            Object containing coherence data.
        """
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        f, coh = ss.coherence(
            x=self.data[:, np.newaxis],
            y=self.data[np.newaxis],
            fs=self.sampling_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window=window,
            detrend=detrend,
        )
        coh = coh.reshape(-1, coh.shape[-1])
        channel_labels = [f"{ich.label} & {jch.label}" for ich in self for jch in self]
        label = "Coherence"

        freq_channels = [
            FrequencyChannel(
                data=data,
                sampling_rate=self.sampling_rate,
                window=window,
                label=label,
                n_fft=n_fft,
            )
            for data, label in zip(coh, channel_labels)
        ]

        return FrequencyChannelFrame(freq_channels, label=label)

    def csd(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "FrequencyChannelFrame":
        """
        Perform cross-spectral density estimation.

        Parameters
        ----------
        n_fft : int, optional
            Number of FFT points. If None, defaults to win_length.
        hop_length : int, optional
            Number of samples between successive frames.
            If None, defaults to win_length//2.
        win_length : int, default=2048
            Window size.
        window : str, default="hann"
            Window function.
        detrend : str, default="constant"
            Type of detrending.
        scaling : str, default="spectrum"
            Scaling type.
        average : str, default="mean"
            Averaging method.

        Returns
        -------
        FrequencyChannelFrame
            Object containing cross-spectral density data.
        """
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        f, csd = ss.csd(
            x=self.data[:, np.newaxis],
            y=self.data[np.newaxis],
            fs=self.sampling_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )
        coh = np.sqrt(csd.reshape(-1, csd.shape[-1]))
        channel_labels = [f"{ich.label} & {jch.label}" for ich in self for jch in self]
        channel_units = [f"{ich.unit}*{jch.unit}" for ich in self for jch in self]
        label = "Cross power spectral"

        freq_channels = [
            FrequencyChannel(
                data=data,
                sampling_rate=self.sampling_rate,
                window=window,
                label=label,
                n_fft=n_fft,
                unit=unit,
            )
            for data, label, unit in zip(coh, channel_labels, channel_units)
        ]

        return FrequencyChannelFrame(freq_channels, label=label)

    def transfer_function(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "FrequencyChannelFrame":
        """
        Estimate transfer function.

        Parameters
        ----------
        n_fft : int, optional
            Number of FFT points. If None, defaults to win_length.
        hop_length : int, optional
            Number of samples between successive frames.
            If None, defaults to win_length//2.
        win_length : int, default=2048
            Window size.
        window : str, default="hann"
            Window function.
        detrend : str, default="constant"
            Type of detrending.
        scaling : str, default="spectrum"
            Scaling type.
        average : str, default="mean"
            Averaging method.

        Returns
        -------
        FrequencyChannelFrame
            Object containing transfer function data.
        """
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        num_channels = self.data.shape[0]

        # Calculate cross-spectral density (between all channels)
        f, p_yx = ss.csd(
            x=self.data[:, np.newaxis, :],  # shape: (num_channels, 1, num_samples)
            y=self.data[np.newaxis, :, :],  # shape: (1, num_channels, num_samples)
            fs=self.sampling_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
            axis=-1,
        )
        # P_yx shape: (num_channels, num_channels, num_frequencies)

        # Calculate power spectral density (for each channel)
        f, p_xx = ss.welch(
            x=self.data,
            fs=self.sampling_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
            axis=-1,
        )
        # P_xx shape: (num_channels, num_frequencies)

        # Calculate transfer function H(f) = P_yx / P_xx (broadcast P_xx)
        h_f = (
            p_yx / p_xx[np.newaxis, :, :]
        )  # Expand P_xx to shape (1, num_channels, num_frequencies)

        # Generate labels and units
        channel_labels = np.array(
            [
                [
                    f"{self._channels[i].label} / {self._channels[j].label}"
                    for j in range(num_channels)
                ]
                for i in range(num_channels)
            ]
        )
        channel_units = np.array(
            [
                [
                    f"{self._channels[i].unit} / {self._channels[j].unit}"
                    for j in range(num_channels)
                ]
                for i in range(num_channels)
            ]
        )

        # Reshape H_f, channel_labels, channel_units to 1D arrays
        h_f_flat = h_f.reshape(
            -1, h_f.shape[-1]
        )  # shape: (num_channels * num_channels, num_frequencies)
        channel_labels_flat = channel_labels.flatten()
        channel_units_flat = channel_units.flatten()

        # Create list of FrequencyChannel objects
        freq_channels = [
            FrequencyChannel(
                data=h_f_flat[k],
                sampling_rate=self.sampling_rate,
                window=window,
                label=channel_labels_flat[k],
                n_fft=n_fft,
                unit=channel_units_flat[k],
            )
            for k in range(h_f_flat.shape[0])
        ]

        return FrequencyChannelFrame(freq_channels, label="Transfer Function")

    def plot(
        self,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        overlay: bool = True,
    ) -> None:
        """
        Plot all channels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        title : str, optional
            Plot title.
        overlay : bool, default=True
            If True, plot all channels on the same axes.
        """
        cf = self.to_channel_frame()
        cf.plot(ax=ax, title=title, overlay=overlay)
