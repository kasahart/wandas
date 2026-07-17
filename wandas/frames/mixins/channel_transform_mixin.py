"""Module providing mixins related to frequency transformations and transform
operations."""

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from wandas.pipeline.decorators import recipe_operation

from ...core.base_frame import BaseFrame
from .protocols import TransformFrameProtocol

if TYPE_CHECKING:
    from wandas.frames.cepstral import CepstralFrame
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
    for out_ch in channel_metadata:
        for in_ch in channel_metadata:
            meta = ChannelMetadata()
            meta.label = label_template.format(in_label=in_ch.label, out_label=out_ch.label)
            meta.unit = ""
            meta.ref = 1
            meta["metadata"] = {"in_ch": in_ch["metadata"], "out_ch": out_ch["metadata"]}
            result.append(meta)
    return result


def _build_cross_channel_source_time_offsets(source_time_offset: Any) -> Any:
    """Build pairwise source offsets for cross-channel spectral outputs."""
    offsets = np.asarray(source_time_offset, dtype=float)
    result: list[float] = []
    for _out_offset in offsets:
        for in_offset in offsets:
            result.append(float(in_offset))
    return np.asarray(result, dtype=float)


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
        result_data = operation.process(self._effective_data)

        logger.debug(f"Created new SpectralFrame with operation {operation_name} added to graph")

        channel_metadata = _build_cross_channel_metadata(
            self._channel_metadata,
            operation_name,
            label_template,
        )

        operation_params = operation.to_params()
        n_fft = operation_params["n_fft"]
        if isinstance(n_fft, bool) or not isinstance(n_fft, int):
            raise TypeError(
                f"Operation '{operation_name}' must provide a positive integer n_fft "
                f"to create a SpectralFrame, but got {type(n_fft).__name__}."
            )
        if n_fft <= 0:
            raise ValueError(
                f"Operation '{operation_name}' must provide a positive integer n_fft "
                f"to create a SpectralFrame, but got {n_fft}."
            )
        lineage = cast(Any, self)._required_semantic_lineage()
        return SpectralFrame(
            data=result_data,
            sampling_rate=self.sampling_rate,
            n_fft=n_fft,
            window=operation_params["window"],
            label=f"{label_prefix} {self.label}",
            metadata=self.metadata,
            channel_metadata=channel_metadata,
            source_time_offset=_build_cross_channel_source_time_offsets(cast(Any, self).source_time_offset),
            lineage=lineage,
            previous=self._as_base_frame,
        )

    @recipe_operation("wandas.audio.cepstrum")
    def cepstrum(
        self: TransformFrameProtocol,
        n_fft: int | None = None,
        window: str = "hann",
        floor: float = 1e-12,
    ) -> "CepstralFrame":
        """Calculate the normalized real cepstrum of each channel.

        Parameters
        ----------
        n_fft : int, optional
            FFT size. ``None`` uses the current sample count. Smaller values
            truncate and larger values zero-pad the analysis input.
        window : str, default="hann"
            SciPy window name applied before the FFT.
        floor : float, default=1e-12
            Positive finite floor applied to normalized magnitude before ``log``.

        Returns
        -------
        CepstralFrame
            New lazy real coefficients with dimensions
            ``(channel, quefrency)``. Channel metadata, IDs, user metadata,
            sampling rate, and source-time offsets are preserved.

        Raises
        ------
        TypeError
            If the input is complex or a parameter has the wrong type.
        ValueError
            If ``n_fft`` or ``floor`` is invalid.

        Notes
        -----
        The method only builds a Dask graph. Accessing ``data``, calling
        ``compute()``, or plotting materializes the coefficients.

        Examples
        --------
        >>> cepstrum = frame.cepstrum(n_fft=2048, window="hann")
        >>> envelope = cepstrum.lifter(0.002).to_spectral_envelope()
        """
        from wandas.frames.cepstral import CepstralFrame
        from wandas.processing import Cepstrum, create_operation

        if np.issubdtype(self._effective_data.dtype, np.complexfloating):
            raise TypeError(
                "Cepstrum analysis requires real-valued input\n"
                f"  Got: {self._effective_data.dtype}\n"
                "  Expected: real time-domain samples\n"
                "Apply cepstrum() to a real ChannelFrame."
            )
        operation = cast(
            "Cepstrum",
            create_operation(
                "cepstrum",
                self.sampling_rate,
                n_fft=n_fft,
                window=window,
                floor=floor,
            ),
        )
        cepstrum_data = operation.process(self._effective_data)
        resolved_n_fft = int(cepstrum_data.shape[-1])
        return CepstralFrame(
            data=cepstrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=resolved_n_fft,
            window=operation.window,
            label=f"Cepstrum of {self.label}",
            metadata=self.metadata,
            channel_metadata=cast(Any, self)._metadata_after_analysis(),
            channel_ids=cast(Any, self)._channel_ids,
            previous=self._as_base_frame,
            source_time_offset=cast(Any, self).source_time_offset,
            lineage=cast(Any, self)._required_semantic_lineage(),
        )

    @recipe_operation("wandas.audio.fft")
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

        _n_fft = int(self._effective_data.shape[-1]) if n_fft is None else n_fft
        params = {"n_fft": _n_fft, "window": window}
        operation_name = "fft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("FFT", operation)
        # Apply processing to data
        spectrum_data = operation.process(self._effective_data)

        logger.debug(f"Created new SpectralFrame with operation {operation_name} added to graph")

        lineage = cast(Any, self)._required_semantic_lineage()
        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=_n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata=self.metadata,
            channel_metadata=cast(Any, self)._metadata_after_analysis(),
            channel_ids=cast(Any, self)._channel_ids,
            source_time_offset=cast(Any, self).source_time_offset,
            lineage=lineage,
            previous=self._as_base_frame,
        )

    @recipe_operation("wandas.audio.welch")
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
        spectrum_data = operation.process(self._effective_data)

        logger.debug(f"Created new SpectralFrame with operation {operation_name} added to graph")

        lineage = cast(Any, self)._required_semantic_lineage()
        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata=self.metadata,
            channel_metadata=cast(Any, self)._metadata_after_analysis(),
            channel_ids=cast(Any, self)._channel_ids,
            source_time_offset=cast(Any, self).source_time_offset,
            lineage=lineage,
            previous=self._as_base_frame,
        )

    @recipe_operation("wandas.audio.noct_spectrum")
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
        spectrum_data = operation.process(self._effective_data)

        logger.debug(f"Created new SpectralFrame with operation {operation_name} added to graph")

        lineage = cast(Any, self)._required_semantic_lineage()
        return NOctFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            fmin=fmin,
            fmax=fmax,
            n=n,
            G=G,
            fr=fr,
            label=f"1/{n}Oct of {self.label}",
            metadata=self.metadata,
            channel_metadata=cast(Any, self)._metadata_after_analysis(),
            channel_ids=cast(Any, self)._channel_ids,
            source_time_offset=cast(Any, self).source_time_offset,
            lineage=lineage,
            previous=self._as_base_frame,
        )

    @recipe_operation("wandas.audio.stft")
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
        spectrogram_data = operation.process(self._effective_data)

        logger.debug(f"Created new SpectrogramFrame with operation {operation_name} added to graph")

        # Create new instance
        lineage = cast(Any, self)._required_semantic_lineage()
        return SpectrogramFrame(
            data=spectrogram_data,
            sampling_rate=self.sampling_rate,
            n_fft=n_fft,
            hop_length=_hop_length,
            win_length=_win_length,
            window=window,
            label=f"stft({self.label})",
            metadata=self.metadata,
            channel_metadata=cast(Any, self)._metadata_after_analysis(),
            channel_ids=cast(Any, self)._channel_ids,
            source_time_offset=cast(Any, self).source_time_offset,
            lineage=lineage,
            previous=self._as_base_frame,
        )

    @recipe_operation("wandas.audio.coherence")
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

    @recipe_operation("wandas.audio.csd")
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

    @recipe_operation("wandas.audio.transfer_function")
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
