from typing import TYPE_CHECKING, Any, Optional

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy import signal as ss

from wandas.utils.types import NDArrayComplex, NDArrayReal

from .base_channel import BaseChannel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh


class TimeFrequencyChannel(BaseChannel):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        # center: bool = None,
        # pad_mode: str,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        previous: Optional["BaseChannel"] = None,
    ):
        """
        Initialize a TimeFrequencyChannel object.

        Parameters
        ----------
        data : NDArrayReal
            Spectral data (time-frequency components).
        sampling_rate : int
            Sampling rate (Hz).
        n_fft : int
            FFT size.
        hop_length : int
            Number of samples between successive frames.
        win_length : int
            Window length.
        window : str
            Window function type.
        label : str, optional
            Channel label.
        unit : str, optional
            Unit of measurement.
        metadata : dict, optional
            Additional metadata.
        previous : BaseChannel, optional
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
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        # self.center = center
        # self.pad_mode = pad_mode

    @classmethod
    def stft(
        cls,
        data: NDArrayReal,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        # pad_mode: str = "constant",
    ) -> dict[str, Any]:
        """
        Perform Short-Time Fourier Transform (STFT).

        Parameters
        ----------
        data : NDArrayReal
            Input signal.
        n_fft : int, optional
            FFT size. If None, defaults to win_length.
        hop_length : int, optional
            Hop size (number of samples between successive frames).
            If None, defaults to win_length//2.
        win_length : int, optional
            Window length. If None, defaults to 2048.
        window : str, default="hann"
            Window function type.

        Returns
        -------
        dict
            Dictionary containing the STFT results and parameters.
        """
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        spec_data: NDArrayComplex
        _, _, spec_data = ss.stft(
            data,
            nfft=n_fft,
            noverlap=win_length - hop_length,
            nperseg=win_length,
            window=window,
            detrend="constant",  # type: ignore[unused-ignore]
            # pad_mode=pad_mode,
        )
        spec_data[..., 1:-1, :] *= 2.0
        return dict(
            data=spec_data,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

    def melspectrogram(
        self,
        n_mels: int = 128,
        # pad_mode: str = "constant",
    ) -> "TimeMelFrequencyChannel":
        """
        Convert STFT to mel spectrogram.

        Parameters
        ----------
        n_mels : int, default=128
            Number of mel bands.

        Returns
        -------
        TimeMelFrequencyChannel
            Object containing the mel spectrogram.
        """
        result = TimeMelFrequencyChannel.spec2melspec(
            sampling_rate=self.sampling_rate,
            data=np.abs(self.data),
            n_fft=self.n_fft,
            n_mels=n_mels,
        )
        result.update(
            dict(
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
            )
        )
        return TimeMelFrequencyChannel.from_channel(self, **result)

    def data_Aw(self, to_dB: bool = False) -> NDArrayReal:  # noqa: N802, N803
        """
        Return amplitude data with A-weighting applied.

        Parameters
        ----------
        to_dB : bool, default=False
            If True, return the result in decibels.

        Returns
        -------
        NDArrayReal
            A-weighted amplitude data.
        """
        freqs = fft.rfftfreq(self.n_fft, 1 / self.sampling_rate)
        weighted: NDArrayReal = librosa.perceptual_weighting(
            np.abs(self._data) ** 2, freqs, kind="A", ref=self.ref**2
        )

        if to_dB:
            return weighted.astype(np.float64)

        return np.asarray(librosa.db_to_amplitude(weighted), dtype=np.float64)

    def hpss_harmonic(self, **kwargs: Any) -> "TimeFrequencyChannel":
        """
        Extract harmonic component.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to librosa.decompose.hpss.

        Returns
        -------
        TimeFrequencyChannel
            Channel containing the harmonic component.
        """
        harmonic, _ = librosa.decompose.hpss(self.data, **kwargs)
        result = dict(
            data=harmonic,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        return TimeFrequencyChannel.from_channel(self, **result)

    def hpss_percussive(self, **kwargs: Any) -> "TimeFrequencyChannel":
        """
        Extract percussive component.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to librosa.decompose.hpss.

        Returns
        -------
        TimeFrequencyChannel
            Channel containing the percussive component.
        """
        _, percussive = librosa.decompose.hpss(self.data, **kwargs)
        result = dict(
            data=percussive,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        return TimeFrequencyChannel.from_channel(self, **result)

    def _plot(
        self,
        ax: "Axes",
        title: Optional[str] = None,
        db_scale: bool = True,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        Aw: bool = False,  # noqa: N803
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> tuple["QuadMesh", NDArrayReal]:
        """
        Plot time-frequency data (internal method).

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to plot on.
        title : str, optional
            Plot title.
        db_scale : bool, default=True
            If True, plot in dB scale.
        fmin : float, optional
            Minimum frequency to display.
        fmax : float, optional
            Maximum frequency to display.
        Aw : bool, default=False
            If True, apply A-weighting before plotting.
        vmin : float, optional
            Minimum value for color scaling.
        vmax : float, optional
            Maximum value for color scaling.

        Returns
        -------
        tuple
            Tuple containing (image, plotted_data).
        """

        if Aw:
            data_to_plot = self.data_Aw(to_dB=True)
        elif db_scale:
            data_to_plot = self._to_db()
        else:
            data_to_plot = np.abs(self.data)

        # Plot time-frequency data
        img = librosa.display.specshow(
            data=data_to_plot,
            sr=self.sampling_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            win_length=self.win_length,
            x_axis="time",
            y_axis="linear",
            ax=ax,
            fmin=fmin,
            fmax=fmax,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )

        # Set labels and title
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(title or self.label or "Time-Frequency Representation")

        return img, data_to_plot

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        db_scale: bool = True,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        Aw: bool = False,  # noqa: N803
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> tuple["Axes", NDArrayReal]:
        """
        Plot time-frequency data.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new one is created.
        title : str, optional
            Plot title.
        db_scale : bool, default=True
            If True, plot in dB scale.
        fmin : float, optional
            Minimum frequency to display.
        fmax : float, optional
            Maximum frequency to display.
        Aw : bool, default=False
            If True, apply A-weighting before plotting.
        vmin : float, optional
            Minimum value for color scaling.
        vmax : float, optional
            Maximum value for color scaling.

        Returns
        -------
        tuple
            Tuple containing (axes, plotted_data).
        """
        _ax = ax
        if _ax is None:
            _, _ax = plt.subplots(figsize=(10, 6))

        img, data_to_plot = self._plot(
            ax=_ax,
            title=title,
            db_scale=db_scale,
            fmin=fmin,
            fmax=fmax,
            Aw=Aw,
            vmin=vmin,
            vmax=vmax,
        )

        if _ax.figure is not None:
            if db_scale:
                cbar = _ax.figure.colorbar(img, ax=ax, format="%+2.0f")
                if Aw:
                    unit = "dBA"
                else:
                    unit = "dB"
                cbar.set_label(f"Spectrum level [{unit}]")
            else:
                cbar = _ax.figure.colorbar(img, ax=ax)
                cbar.set_label(f"Amplitude [{self.unit}]")

        if ax is None:
            plt.tight_layout()
            plt.show()

        return _ax, data_to_plot

    def _to_db(self) -> NDArrayReal:
        """
        Convert spectral data to dB scale.

        Returns
        -------
        NDArrayReal
            Data converted to dB scale.
        """
        return np.asarray(
            librosa.amplitude_to_db(np.abs(self.data), ref=self.ref, amin=1e-12),
            np.float64,
        )


class TimeMelFrequencyChannel(TimeFrequencyChannel):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        # center: bool = None,
        # pad_mode: str,
        n_mels: int = 128,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        previous: Optional["BaseChannel"] = None,
    ):
        """
        Initialize a TimeMelFrequencyChannel object.

        Parameters
        ----------
        data : NDArrayReal
            Spectral data (time-frequency components in mel scale).
        sampling_rate : int
            Sampling rate (Hz).
        n_fft : int
            FFT size.
        hop_length : int
            Number of samples between successive frames.
        win_length : int
            Window length.
        window : str
            Window function type.
        n_mels : int, default=128
            Number of mel bands.
        label : str, optional
            Channel label.
        unit : str, optional
            Unit of measurement.
        metadata : dict, optional
            Additional metadata.
        previous : BaseChannel, optional
            Reference to the original channel before transformation.
        """
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            unit=unit,
            metadata=metadata,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            previous=previous,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        # self.center = center
        # self.pad_mode = pad_mode

    @classmethod
    def spec2melspec(
        cls,
        sampling_rate: int,
        data: NDArrayReal,
        n_fft: int = 2048,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        n_mels: int = 128,
        # pad_mode: str = "constant",
    ) -> dict[str, Any]:
        """
        Convert linear-frequency spectrogram to mel-frequency spectrogram.

        Parameters
        ----------
        sampling_rate : int
            Sampling rate (Hz).
        data : NDArrayReal
            Linear-frequency spectrogram.
        n_fft : int, default=2048
            FFT size.
        fmin : float, default=0.0
            Minimum frequency (Hz).
        fmax : float, optional
            Maximum frequency (Hz). If None, defaults to sampling_rate/2.
        n_mels : int, default=128
            Number of mel bands.

        Returns
        -------
        dict
            Dictionary containing the mel spectrogram and parameters.
        """
        melfb = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            norm=None,
        )

        melspec: NDArrayReal = np.einsum("...ft,mf->...mt", data, melfb, optimize=True)  # type: ignore[arg-type, unused-ignore]

        return dict(
            sampling_rate=sampling_rate,
            data=melspec,
            n_fft=n_fft,
            n_mels=n_mels,
        )

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        db_scale: bool = True,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        Aw: bool = False,  # noqa: N803
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> tuple["Axes", NDArrayReal]:
        """
        Plot mel spectrogram.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new one is created.
        title : str, optional
            Plot title.
        db_scale : bool, default=True
            If True, plot in dB scale.
        fmin : float, optional
            Minimum frequency to display.
        fmax : float, optional
            Maximum frequency to display.
        Aw : bool, default=False
            If True, apply A-weighting before plotting.
        vmin : float, optional
            Minimum value for color scaling.
        vmax : float, optional
            Maximum value for color scaling.

        Returns
        -------
        tuple
            Tuple containing (axes, plotted_data).
        """
        _ax = ax
        if _ax is None:
            _, _ax = plt.subplots(figsize=(10, 6))

        if Aw:
            data_to_plot = self.data_Aw(to_dB=True)
        elif db_scale:
            data_to_plot = self._to_db()
        else:
            data_to_plot = np.abs(self.data)

        # Plot mel spectrogram
        img = librosa.display.specshow(
            data=data_to_plot,
            sr=self.sampling_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            win_length=self.win_length,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            fmin=fmin,
            fmax=fmax,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )

        # Set labels and title
        _ax.set_xlabel("Time [s]")
        _ax.set_ylabel("Frequency [Hz]")
        _ax.set_title(title or self.label or "Time-Frequency Representation")

        if _ax.figure is not None:
            if db_scale:
                cbar = _ax.figure.colorbar(img, ax=ax, format="%+2.0f")
                if Aw:
                    unit = "dBA"
                else:
                    unit = "dB"
                cbar.set_label(f"Spectrum level [{unit}]")
            else:
                cbar = _ax.figure.colorbar(img, ax=ax)
                cbar.set_label(f"Amplitude [{self.unit}]")

        if ax is None:
            plt.tight_layout()
            plt.show()

        return _ax, data_to_plot
