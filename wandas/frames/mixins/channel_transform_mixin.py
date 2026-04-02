"""Module providing mixins related to frequency transformations and transform
operations."""

import logging
from typing import TYPE_CHECKING, Any, cast

from ...core.base_frame import BaseFrame
from .protocols import TransformFrameProtocol

if TYPE_CHECKING:
    from wandas.frames.noct import NOctFrame
    from wandas.frames.spectral import SpectralFrame
    from wandas.frames.spectrogram import SpectrogramFrame


logger = logging.getLogger(__name__)


def _build_cross_channel_metadata(
    channel_metadata: list[Any],
    operation_name: str,
    label_template: str,
) -> list[Any]:
    """Build channel metadata for cross-channel operations (coherence, csd, tf).

    Parameters
    ----------
    channel_metadata : list
        Input channel metadata list.
    operation_name : str
        Operation name for the metadata dict key.
    label_template : str
        Format string with ``{in_label}`` and ``{out_label}`` placeholders.
    """
    from wandas.core.metadata import ChannelMetadata

    result = []
    for in_ch in channel_metadata:
        for out_ch in channel_metadata:
            meta = ChannelMetadata()
            meta.label = label_template.format(in_label=in_ch.label, out_label=out_ch.label)
            meta.unit = ""
            meta.ref = 1
            meta["metadata"] = {"in_ch": in_ch["metadata"], "out_ch": out_ch["metadata"]}
            result.append(meta)
    return result


class ChannelTransformMixin:
    """Mixin providing methods related to frequency transformations.

    This mixin provides operations related to frequency analysis and
    transformations such as FFT, STFT, and Welch method.
    """

    @property
    def _as_base_frame(self: TransformFrameProtocol) -> "BaseFrame[Any]":
        """Cast self to BaseFrame for use as ``previous`` in new frames."""
        return cast(BaseFrame[Any], self)

    def _cross_channel_spectral_transform(
        self: TransformFrameProtocol,
        operation_name: str,
        label_prefix: str,
        label_template: str,
        **params: Any,
    ) -> "SpectralFrame":
        """Shared implementation for cross-channel spectral transforms.

        Used by ``coherence``, ``csd``, and ``transfer_function``.
        """
        from wandas.processing import create_operation

        from ..spectral import SpectralFrame

        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        operation = create_operation(operation_name, self.sampling_rate, **params)
        result_data = operation.process(self._data)

        logger.debug(f"Created new SpectralFrame with operation {operation_name} added to graph")

        channel_metadata = _build_cross_channel_metadata(
            self._channel_metadata,
            operation_name,
            label_template,
        )

        n_fft = operation.n_fft
        if n_fft is not None and not isinstance(n_fft, int):
            raise TypeError(
                f"Operation '{operation_name}' must provide an integer n_fft "
                f"or None to create a SpectralFrame, but got {type(n_fft).__name__}."
            )

        return SpectralFrame(
            data=result_data,
            sampling_rate=self.sampling_rate,
            n_fft=n_fft,
            window=operation.window,
            label=f"{label_prefix} {self.label}",
            metadata=self.metadata.merged(**params),
            operation_history=[
                *self.operation_history,
                {"operation": operation_name, "params": params},
            ],
            channel_metadata=channel_metadata,
            previous=self._as_base_frame,
        )

    def fft(self: TransformFrameProtocol, n_fft: int | None = None, window: str = "hann") -> "SpectralFrame":
        """Calculate Fast Fourier Transform (FFT).

        Args:
            n_fft: Number of FFT points. Default is the next power of 2 of the data
                length.
            window: Window type. Default is "hann".

        Returns:
            SpectralFrame containing FFT results
        """
        from wandas.frames.spectral import SpectralFrame
        from wandas.processing import FFT, create_operation

        params = {"n_fft": n_fft, "window": window}
        operation_name = "fft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("FFT", operation)
        # Apply processing to data
        spectrum_data = operation.process(self._data)

        logger.debug(f"Created new SpectralFrame with operation {operation_name} added to graph")

        if n_fft is None:
            is_even = spectrum_data.shape[-1] % 2 == 0
            _n_fft = spectrum_data.shape[-1] * 2 - 2 if is_even else spectrum_data.shape[-1] * 2 - 1
        else:
            _n_fft = n_fft

        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=_n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata=self.metadata.merged(window=window, n_fft=_n_fft),
            operation_history=[
                *self.operation_history,
                {"operation": "fft", "params": {"n_fft": _n_fft, "window": window}},
            ],
            channel_metadata=self._channel_metadata,
            previous=self._as_base_frame,
        )

    def welch(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        average: str = "mean",
    ) -> "SpectralFrame":
        """Calculate power spectral density using Welch's method.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between frames.
                Default is n_fft//4.
            win_length: Window length. Default is n_fft.
            window: Window type. Default is "hann".
            average: Method for averaging segments. Default is "mean".

        Returns:
            SpectralFrame containing power spectral density
        """
        from wandas.frames.spectral import SpectralFrame
        from wandas.processing import Welch, create_operation

        params = {
            "n_fft": n_fft or win_length,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": window,
            "average": average,
        }
        operation_name = "welch"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("Welch", operation)
        # Apply processing to data
        spectrum_data = operation.process(self._data)

        logger.debug(f"Created new SpectralFrame with operation {operation_name} added to graph")

        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata=self.metadata.merged(**params),
            operation_history=[
                *self.operation_history,
                {"operation": "welch", "params": params},
            ],
            channel_metadata=self._channel_metadata,
            previous=self._as_base_frame,
        )

    def noct_spectrum(
        self: TransformFrameProtocol,
        fmin: float = 25,
        fmax: float = 20000,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> "NOctFrame":
        """Calculate N-octave band spectrum.

        Args:
            fmin: Minimum center frequency (Hz). Default is 25 Hz.
            fmax: Maximum center frequency (Hz). Default is 20000 Hz.
            n: Band division (1: octave, 3: 1/3 octave). Default is 3.
            G: Reference gain (dB). Default is 10 dB.
            fr: Reference frequency (Hz). Default is 1000 Hz.

        Returns:
            NOctFrame containing N-octave band spectrum
        """
        from wandas.processing import NOctSpectrum, create_operation

        from ..noct import NOctFrame

        params = {"fmin": fmin, "fmax": fmax, "n": n, "G": G, "fr": fr}
        operation_name = "noct_spectrum"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("NOctSpectrum", operation)
        # Apply processing to data
        spectrum_data = operation.process(self._data)

        logger.debug(f"Created new SpectralFrame with operation {operation_name} added to graph")

        return NOctFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            fmin=fmin,
            fmax=fmax,
            n=n,
            G=G,
            fr=fr,
            label=f"1/{n}Oct of {self.label}",
            metadata=self.metadata.merged(**params),
            operation_history=[
                *self.operation_history,
                {
                    "operation": "noct_spectrum",
                    "params": params,
                },
            ],
            channel_metadata=self._channel_metadata,
            previous=self._as_base_frame,
        )

    def stft(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
    ) -> "SpectrogramFrame":
        """Calculate Short-Time Fourier Transform.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between frames.
                Default is n_fft//4.
            win_length: Window length. Default is n_fft.
            window: Window type. Default is "hann".

        Returns:
            SpectrogramFrame containing STFT results
        """
        from wandas.processing import STFT, create_operation

        from ..spectrogram import SpectrogramFrame

        # Set hop length and window length
        _hop_length = hop_length if hop_length is not None else n_fft // 4
        _win_length = win_length if win_length is not None else n_fft

        params = {
            "n_fft": n_fft,
            "hop_length": _hop_length,
            "win_length": _win_length,
            "window": window,
        }
        operation_name = "stft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("STFT", operation)

        # Apply processing to data
        spectrogram_data = operation.process(self._data)

        logger.debug(f"Created new SpectrogramFrame with operation {operation_name} added to graph")

        # Create new instance
        return SpectrogramFrame(
            data=spectrogram_data,
            sampling_rate=self.sampling_rate,
            n_fft=n_fft,
            hop_length=_hop_length,
            win_length=_win_length,
            window=window,
            label=f"stft({self.label})",
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=self._channel_metadata,
            previous=self._as_base_frame,
        )

    def coherence(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
    ) -> "SpectralFrame":
        """Calculate magnitude squared coherence.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between frames.
                Default is n_fft//4.
            win_length: Window length. Default is n_fft.
            window: Window type. Default is "hann".
            detrend: Detrend method. Options: "constant", "linear", None.

        Returns:
            SpectralFrame containing magnitude squared coherence
        """
        return self._cross_channel_spectral_transform(
            "coherence",
            "Coherence of",
            "$\\gamma_{{{in_label}, {out_label}}}$",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
        )

    def csd(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "SpectralFrame":
        """Calculate cross-spectral density matrix.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between frames.
                Default is n_fft//4.
            win_length: Window length. Default is n_fft.
            window: Window type. Default is "hann".
            detrend: Detrend method. Options: "constant", "linear", None.
            scaling: Scaling method. Options: "spectrum", "density".
            average: Method for averaging segments. Default is "mean".

        Returns:
            SpectralFrame containing cross-spectral density matrix
        """
        return self._cross_channel_spectral_transform(
            "csd",
            "CSD of",
            "csd({in_label}, {out_label})",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )

    def transfer_function(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "SpectralFrame":
        """Calculate transfer function matrix.

        The transfer function represents the signal transfer characteristics between
        channels in the frequency domain and represents the input-output relationship
        of the system.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between frames.
                Default is n_fft//4.
            win_length: Window length. Default is n_fft.
            window: Window type. Default is "hann".
            detrend: Detrend method. Options: "constant", "linear", None.
            scaling: Scaling method. Options: "spectrum", "density".
            average: Method for averaging segments. Default is "mean".

        Returns:
            SpectralFrame containing transfer function matrix
        """
        return self._cross_channel_spectral_transform(
            "transfer_function",
            "Transfer function of",
            "$H_{{{in_label}, {out_label}}}$",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )
