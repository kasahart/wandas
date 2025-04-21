# wandas/core/frequency_channel.py

from typing import TYPE_CHECKING, Any, Optional, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
from mosqito.sound_level_meter import noct_spectrum, noct_synthesis
from scipy import fft
from scipy import signal as ss

from wandas.core import util
from wandas.utils.types import NDArrayReal

from .base_channel import BaseChannel

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class NOctChannel(BaseChannel):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        fpref: NDArrayReal,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        previous: Optional["BaseChannel"] = None,
    ):
        """
        Initialize a NOctChannel object for N-octave band spectrum analysis.

        Parameters
        ----------
        data : NDArrayReal
            Amplitude data for each band.
        sampling_rate : int
            Sampling rate (Hz).
        fpref : NDArrayReal
            Center frequencies of the bands.
        n : int, default=3
            Fraction denominator for octave bands (e.g., 3 for 1/3-octave).
        G : int, default=10
            Band number of the reference frequency band.
        fr : int, default=1000
            Reference frequency (Hz).
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

        self.n = n
        self.G = G
        self.fr = fr
        self.fpref = fpref

    @classmethod
    def noct_spectrum(
        cls,
        data: NDArrayReal,
        sampling_rate: int,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> dict[str, Any]:
        """
        Calculate N-Octave Spectrum.

        Parameters
        ----------
        data : NDArrayReal
            Input signal.
        sampling_rate : int
            Sampling rate (Hz).
        fmin : float
            Minimum frequency (Hz).
        fmax : float
            Maximum frequency (Hz).
        n : int, default=3
            Fraction denominator for octave bands (e.g., 3 for 1/3-octave).
        G : int, default=10
            Band number of the reference frequency band.
        fr : int, default=1000
            Reference frequency (Hz).

        Returns
        -------
        dict
            Dictionary containing the N-octave spectrum data and parameters.
        """

        spec, fpref = noct_spectrum(
            sig=data, fs=sampling_rate, fmin=fmin, fmax=fmax, n=n, G=G, fr=fr
        )

        return dict(data=np.squeeze(spec), fpref=fpref, n=n, G=G, fr=fr)

    @classmethod
    def noct_synthesis(
        cls,
        data: NDArrayReal,
        freqs: NDArrayReal,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> dict[str, Any]:
        """
        Synthesize N-Octave Spectrum from frequency domain data.

        Parameters
        ----------
        data : NDArrayReal
            Spectral data.
        freqs : NDArrayReal
            Frequency array.
        fmin : float
            Minimum frequency (Hz).
        fmax : float
            Maximum frequency (Hz).
        n : int, default=3
            Fraction denominator for octave bands (e.g., 3 for 1/3-octave).
        G : int, default=10
            Band number of the reference frequency band.
        fr : int, default=1000
            Reference frequency (Hz).

        Returns
        -------
        dict
            Dictionary containing the synthesized N-octave spectrum and parameters.

        Raises
        ------
        ValueError
            If sampling rate is not 48000 Hz.
        """

        fs = freqs.max() * 2
        if round(fs) == 48000:
            spec, fpref = noct_synthesis(
                spectrum=data, freqs=freqs, fmin=fmin, fmax=fmax, n=n, G=G, fr=fr
            )
        # elif n == 3:
        #     if data.ndim == 1:
        #         data = data[..., None]
        #     spec, fpref = freq_band_synthesis(
        #         spectrum=np.abs(data),
        #         freqs=freqs,
        #         fmin=np.array([fmin]),
        #         fmax=np.array([fmax]),
        #     )
        #     spec = np.squeeze(spec)

        else:
            raise ValueError("fs must be 48000")
        return dict(data=spec, fpref=fpref, n=n, G=G, fr=fr)

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

        weighted: NDArrayReal = librosa.perceptual_weighting(
            self.data[..., None] ** 2, self.fpref, kind="A", ref=self.ref**2
        ).squeeze()

        if to_dB:
            return weighted.astype(np.float64)

        return np.asarray(
            librosa.db_to_amplitude(weighted, ref=self.ref), dtype=np.float64
        )

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        Aw: Optional[bool] = False,  # noqa: N803
    ) -> "Axes":
        """
        Plot spectrum data.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new one is created.
        title : str, optional
            Plot title.
        Aw : bool, default=False
            If True, apply A-weighting before plotting.

        Returns
        -------
        Axes
            Matplotlib axes containing the plot.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        if Aw:
            unit = "dBrA"
            data = self.data_Aw(to_dB=True)
        else:
            unit = "dBr"
            data = util.amplitude_to_db(self.data, ref=self.ref)

        ax.step(
            self.fpref,
            data,
            label=self.label or "Spectrum",
        )

        ax.set_xlabel("Center frequency [Hz]")
        ylabel = f"Spectrum level [{unit}]"
        ax.set_ylabel(ylabel)
        ax.set_title(title or self.label or f"1/{str(self.n)}-Octave Spectrum")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax


class FrequencyChannel(BaseChannel):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        n_fft: int,
        window: Union[NDArrayReal, str],
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        previous: Optional["BaseChannel"] = None,
    ):
        """
        Initialize a FrequencyChannel object.

        Parameters
        ----------
        data : NDArrayReal
            Frequency domain data.
        sampling_rate : int
            Sampling rate (Hz).
        n_fft : int
            FFT size.
        window : NDArrayReal or str
            Window function used for the FFT.
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
        self.window = window

    @classmethod
    def fft(
        cls,
        data: NDArrayReal,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Perform FFT on time-domain data.

        Parameters
        ----------
        data : NDArrayReal
            Input time-domain signal.
        n_fft : int, optional
            FFT size. If None, uses the length of the input data.
        window : str, optional
            Name of the window function to apply.

        Returns
        -------
        dict
            Dictionary containing the FFT results and parameters.

        Raises
        ------
        ValueError
            If n_fft is less than the length of the input data.
        """
        length = data.shape[-1]
        if n_fft is None:
            n_fft = length

        if n_fft < length:
            raise ValueError(
                "n_fft must be greater than or equal to the length of the input data."
            )

        if window:
            window_values = ss.get_window(window, length)
        else:
            window_values = np.ones(length)

        data = data * window_values

        out = np.asarray(fft.rfft(data, n=n_fft))
        out[1:-1] *= 2.0
        # Window function correction
        scaling_factor = np.sum(window_values)
        out /= scaling_factor

        return dict(data=out.squeeze(), window=window_values, n_fft=n_fft)

    @classmethod
    def welch(
        cls,
        data: NDArrayReal,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = 2048,
        window: str = "hann",
        average: str = "mean",
        detrend: str = "constant",
    ) -> dict[str, Any]:
        """
        Compute power spectral density using Welch's method.

        Parameters
        ----------
        data : NDArrayReal
            Input time-domain signal.
        n_fft : int, optional
            FFT size. If None, defaults to win_length.
        hop_length : int, optional
            Number of samples between successive segments.
            If None, defaults to win_length//2.
        win_length : int, default=2048
            Window size.
        window : str, default="hann"
            Window function.
        average : str, default="mean"
            Method for averaging the segments.
        detrend : str, default="constant"
            Type of detrending.

        Returns
        -------
        dict
            Dictionary containing the Welch PSD results and parameters.
        """
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        _, out = ss.welch(
            data,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window=window,
            average=average,
            detrend=detrend,
            scaling="spectrum",
        )

        return dict(
            data=out,
            n_fft=n_fft,
            # hop_length=hop_length,
            # win_length=win_length,
            window=window,
            # average=average,
            # detrend=detrend,
        )

    @property
    def freqs(self) -> NDArrayReal:
        """
        Returns the frequency array after Fourier transform.

        Returns
        -------
        NDArrayReal
            Array of frequency values in Hz.
        """
        return np.asarray(fft.rfftfreq(self.n_fft, 1 / self.sampling_rate))

    def noct_synthesis(
        self,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> NOctChannel:
        """
        Calculate N-Octave Spectrum from frequency domain data.

        Parameters
        ----------
        fmin : float
            Minimum frequency (Hz).
        fmax : float
            Maximum frequency (Hz).
        n : int, default=3
            Fraction denominator for octave bands (e.g., 3 for 1/3-octave).
        G : int, default=10
            Band number of the reference frequency band.
        fr : int, default=1000
            Reference frequency (Hz).

        Returns
        -------
        NOctChannel
            Object containing the N-octave band spectrum.
        """
        result = NOctChannel.noct_synthesis(
            data=self.data / np.sqrt(2),
            freqs=self.freqs,
            fmin=fmin,
            fmax=fmax,
            n=n,
            G=G,
            fr=fr,
        )

        return NOctChannel.from_channel(self, **result)

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
        freqs = self.freqs
        weighted: NDArrayReal = librosa.perceptual_weighting(
            np.abs(self.data[..., None]) ** 2, freqs, kind="A", ref=self.ref**2
        ).squeeze()

        if to_dB:
            return weighted.astype(np.float64)

        return np.asarray(
            librosa.db_to_amplitude(weighted, ref=self.ref), dtype=np.float64
        )

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        Aw: bool = False,  # noqa: N803
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple["Axes", NDArrayReal]:
        """
        Plot spectrum data.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new one is created.
        title : str, optional
            Plot title.
        Aw : bool, default=False
            If True, apply A-weighting before plotting.
        plot_kwargs : dict, optional
            Additional keyword arguments for the plot function.

        Returns
        -------
        tuple
            Tuple containing (axes, plotted_data).
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        if Aw:
            unit = "dBA"
            data = self.data_Aw(to_dB=True)
        else:
            unit = "dB"
            data = util.amplitude_to_db(np.abs(self.data), ref=self.ref)

        plot_kwargs = plot_kwargs or {}
        ax.plot(self.freqs, data, label=self.label or "Spectrum", **plot_kwargs)

        ax.set_xlabel("Frequency [Hz]")
        ylabel = f"Spectrum level [{unit}]"
        ax.set_ylabel(ylabel)
        ax.set_title(title or self.label or "Spectrum")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax, data
