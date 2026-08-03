"""Module providing mixins related to frequency transformations and transform
operations."""

import logging
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast, runtime_checkable

import numpy as np
from dask.array.core import Array as DaArray

from wandas.pipeline.decorators import recipe_operation
from wandas.processing.spectral import validate_noct_recipe_params

from ...core.base_frame import BaseFrame
from ..pairwise import CoherenceFrame, CrossSpectralFrame, PairwiseSpectralFrame, TransferFunctionFrame
from .protocols import TransformFrameProtocol

if TYPE_CHECKING:
    from wandas.frames.cepstral import CepstralFrame
    from wandas.frames.noct import NOctFrame
    from wandas.frames.spectral import SpectralFrame
    from wandas.frames.spectrogram import SpectrogramFrame


logger = logging.getLogger(__name__)
PairwiseFrameT = TypeVar("PairwiseFrameT", bound=PairwiseSpectralFrame)


@runtime_checkable
class _PairwiseSpectralOperationProtocol(Protocol):
    """Minimum typed operation surface needed by the pairwise Frame builder."""

    @property
    def n_fft(self) -> int: ...

    @property
    def window(self) -> str: ...

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray: ...


@runtime_checkable
class _ScaledPairwiseSpectralOperationProtocol(_PairwiseSpectralOperationProtocol, Protocol):
    """Pairwise operation surface for CSD and transfer scaling state."""

    @property
    def scaling(self) -> str: ...


def _build_cross_channel_source_time_offsets(source_time_offset: Any) -> Any:
    """Build pairwise source offsets for cross-channel spectral outputs."""
    offsets = np.asarray(source_time_offset, dtype=float)
    result: list[float] = []
    for _out_offset in offsets:
        for in_offset in offsets:
            result.append(float(in_offset))
    return np.asarray(result, dtype=float)


def _validate_real_cepstrum_input(data: Any) -> None:
    """Reject complex channel data before constructing a cepstrum operation."""
    if np.issubdtype(data.dtype, np.complexfloating):
        raise TypeError(
            "Cepstrum analysis requires real-valued input\n"
            f"  Got: {data.dtype}\n"
            "  Expected: real time-domain samples\n"
            "Apply cepstrum() to a real ChannelFrame."
        )


def _cross_channel_spectral_transform(
    source: TransformFrameProtocol,
    operation_name: str,
    label_prefix: str,
    label_template: str,
    output_frame_class: type[PairwiseFrameT],
    quantity: Literal["coherence", "csd", "transfer"],
    denominator_role: Literal["input", "output"] = "input",
    operation_override: _PairwiseSpectralOperationProtocol | None = None,
    **params: Any,
) -> PairwiseFrameT:
    """Build one typed flattened pairwise spectral Frame lazily.

    The generic output class is the sole source of the concrete return type.  This
    helper owns orchestration only; numerical settings and processing remain on the
    supplied Operation, while pair metadata, lineage, and Frame construction remain
    in the Frame layer.
    """
    from wandas.processing import create_operation

    from ..pairwise import _metadata_for_pair_state, build_pair_state

    logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

    operation_candidate = (
        operation_override
        if operation_override is not None
        else create_operation(operation_name, source.sampling_rate, **params)
    )
    if not isinstance(operation_candidate, _PairwiseSpectralOperationProtocol):
        raise TypeError(
            f"Operation '{operation_name}' does not expose the pairwise spectral contract (process, n_fft, and window)."
        )
    operation = operation_candidate
    result_data = operation.process(source._effective_data)

    n_fft = operation.n_fft
    if isinstance(n_fft, bool) or not isinstance(n_fft, int):
        raise TypeError(
            f"Operation '{operation_name}' must provide a positive integer n_fft "
            f"to create a dedicated pairwise Frame, but got {type(n_fft).__name__}."
        )
    if n_fft <= 0:
        raise ValueError(
            f"Operation '{operation_name}' must provide a positive integer n_fft "
            f"to create a dedicated pairwise Frame, but got {n_fft}."
        )

    scaling: Literal["spectrum", "density"] | None = None
    if quantity != "coherence":
        if not isinstance(operation, _ScaledPairwiseSpectralOperationProtocol):
            raise TypeError(
                f"Operation '{operation_name}' does not expose the scaled pairwise spectral contract; "
                "CSD and transfer operations must provide a scaling property."
            )
        operation_scaling = operation.scaling
        if operation_scaling == "spectrum":
            scaling = "spectrum"
        elif operation_scaling == "density":
            scaling = "density"
        else:
            raise ValueError(f"Operation '{operation_name}' must provide a valid scaling mode")

    source_channel_metadata = source._channel_metadata
    pair_state = build_pair_state(
        source_channel_metadata,
        source._channel_ids,
        quantity=quantity,
        scaling=scaling,
        denominator_role=denominator_role,
        label_template=label_template,
    )
    channel_metadata = _metadata_for_pair_state(pair_state, source_channel_metadata)
    logger.debug(f"Created {output_frame_class.__name__} with operation {operation_name} added to graph")

    constructor_kwargs: dict[str, Any] = {
        "data": result_data,
        "sampling_rate": source.sampling_rate,
        "n_fft": n_fft,
        "window": operation.window,
        "pair_state": pair_state,
        "source_channel_ids": tuple(source._channel_ids),
        "label": f"{label_prefix} {source.label}",
        "metadata": source.metadata,
        "channel_metadata": channel_metadata,
        "channel_ids": [record.row_id for record in pair_state],
        "source_time_offset": _build_cross_channel_source_time_offsets(source.source_time_offset),
        "lineage": source._required_semantic_lineage(),
        "previous": source._as_base_frame,
    }
    if quantity != "coherence":
        constructor_kwargs["scaling"] = scaling
        if quantity == "transfer":
            constructor_kwargs["denominator_role"] = denominator_role
    return output_frame_class(**constructor_kwargs)


class ChannelTransformMixin:
    """Mixin providing methods related to frequency transformations.

    This mixin provides operations related to frequency analysis and
    transformations such as FFT, STFT, and Welch method.
    """

    @property
    def _as_base_frame(self: TransformFrameProtocol) -> "BaseFrame[Any]":
        """Cast self to BaseFrame for use as ``previous`` in new frames."""
        return cast(BaseFrame[Any], self)

    @recipe_operation("wandas.audio.cepstrum", version=2)
    def cepstrum(
        self: TransformFrameProtocol,
        n_fft: int | None = None,
        window: str = "hann",
        floor: float = 1e-12,
    ) -> "CepstralFrame":
        """Calculate the normalized real cepstrum of each channel.

        Args:
            n_fft: int, optional. FFT size. ``None`` uses the current sample count. Smaller values
                truncate and larger values zero-pad the analysis input.
            window: str, default="hann". SciPy window name applied before the FFT.
            floor: float, default=1e-12. Positive finite floor applied to normalized magnitude before ``log``.

        Returns:
            CepstralFrame: New lazy real coefficients with dimensions
                ``(channel, quefrency)``. Channel metadata, IDs, user metadata,
                sampling rate, and source-time offsets are preserved.

        Raises:
            TypeError: If the input is complex or a parameter has the wrong type.
            ValueError: If ``n_fft`` or ``floor`` is invalid.

        Notes:
            The method only builds a Dask graph. Accessing ``data``, calling
            ``compute()``, or plotting materializes the coefficients.

        Examples:
            >>> cepstrum = frame.cepstrum(n_fft=2048, window="hann")
            >>> envelope = cepstrum.lifter(0.002).to_spectral_envelope()
        """
        from wandas.processing import Cepstrum, create_operation

        _validate_real_cepstrum_input(self._effective_data)
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
        return cast(Any, self)._cepstrum_with_operation(operation)

    @recipe_operation("wandas.audio.cepstrum", version=1)
    def _cepstrum_recipe_v1(
        self: TransformFrameProtocol,
        n_fft: int | None = None,
        window: str = "hann",
        floor: float = 1e-12,
    ) -> "CepstralFrame":
        """Replay the released Recipe v1 cepstrum preparation contract."""
        _validate_real_cepstrum_input(self._effective_data)
        from wandas.processing.cepstral import _RecipeCepstrumV1

        operation = _RecipeCepstrumV1(
            self.sampling_rate,
            n_fft=n_fft,
            window=window,
            floor=floor,
        )
        return cast(Any, self)._cepstrum_with_operation(operation)

    def _cepstrum_with_operation(self: TransformFrameProtocol, operation: Any) -> "CepstralFrame":
        """Build a CepstralFrame for the public or released Recipe operation."""
        from wandas.frames.cepstral import CepstralFrame

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

    @recipe_operation("wandas.audio.fft", version=2)
    def fft(self: TransformFrameProtocol, n_fft: int | None = None, window: str = "hann") -> "SpectralFrame":
        """Calculate a one-sided peak-amplitude FFT spectrum.

        The signal is truncated or zero-padded to ``n_fft``, windowed, and
        normalized by the window's coherent gain. Values retain each channel's
        physical unit. Positive-frequency bins other than Nyquist are doubled,
        so an on-bin sinusoid's magnitude equals its peak amplitude. Graph
        construction remains lazy.

        Args:
            n_fft: Number of FFT points. By default, use the current sample
                count exactly.
            window: Window type. Default is "hann".

        Returns:
            A lazy SpectralFrame containing complex peak-amplitude values.
        """
        from wandas.processing import FFT, create_operation

        _n_fft = int(self._effective_data.shape[-1]) if n_fft is None else n_fft
        params = {"n_fft": _n_fft, "window": window}
        operation_name = "fft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("FFT", operation)
        return cast(Any, self)._fft_with_operation(operation, n_fft=_n_fft)

    @recipe_operation("wandas.audio.fft", version=1)
    def _fft_recipe_v1(
        self: TransformFrameProtocol,
        n_fft: int | None = None,
        window: str = "hann",
    ) -> "SpectralFrame":
        """Replay the released Recipe v1 FFT preparation contract."""
        from wandas.processing.spectral import _RecipeFFTV1

        _n_fft = int(self._effective_data.shape[-1]) if n_fft is None else n_fft
        operation = _RecipeFFTV1(self.sampling_rate, n_fft=_n_fft, window=window)
        return cast(Any, self)._fft_with_operation(operation, n_fft=_n_fft)

    def _fft_with_operation(
        self: TransformFrameProtocol,
        operation: Any,
        *,
        n_fft: int,
    ) -> "SpectralFrame":
        """Build a SpectralFrame for the public or released Recipe operation."""
        from wandas.frames.spectral import SpectralFrame

        spectrum_data = operation.process(self._effective_data)
        logger.debug("Created new SpectralFrame with FFT operation added to graph")
        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata=self.metadata,
            channel_metadata=cast(Any, self)._metadata_after_analysis(),
            channel_ids=cast(Any, self)._channel_ids,
            source_time_offset=cast(Any, self).source_time_offset,
            lineage=cast(Any, self)._required_semantic_lineage(),
            previous=self._as_base_frame,
        )

    @recipe_operation("wandas.audio.welch", version=2)
    def welch(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        average: str = "mean",
    ) -> "SpectralFrame":
        """Calculate a Welch-averaged one-sided peak-amplitude spectrum.

        Segment power spectra are averaged and converted to peak amplitude.
        Values retain each channel's physical unit and are not power spectral
        density or expressed per hertz. ``SpectralFrame.dB`` therefore uses the
        amplitude rule ``20 * log10(amplitude / channel_ref)``. Graph
        construction remains lazy.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between frames.
                Default is ``win_length // 4``.
            win_length: Window length. Default is n_fft.
            window: Window type. Default is "hann".
            average: Method for averaging segments. Default is "mean".

        Returns:
            A lazy SpectralFrame containing real peak-amplitude values.
        """
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
        return cast(Any, self)._welch_with_operation(operation)

    @recipe_operation("wandas.audio.welch", version=1)
    def _welch_recipe_v1(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        average: str = "mean",
    ) -> "SpectralFrame":
        """Replay the released Recipe v1 Welch scaling contract."""
        from wandas.processing.spectral import _RecipeWelchV1

        # Released v1 captured raw n_fft but executed the public truthy fallback.
        resolved_n_fft = cast(int, n_fft or win_length)
        operation = _RecipeWelchV1(
            self.sampling_rate,
            n_fft=resolved_n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            average=average,
        )
        return cast(Any, self)._welch_with_operation(operation)

    def _welch_with_operation(self: TransformFrameProtocol, operation: Any) -> "SpectralFrame":
        """Build a SpectralFrame for the public or released Recipe operation."""
        from wandas.frames.spectral import SpectralFrame

        spectrum_data = operation.process(self._effective_data)
        logger.debug("Created new SpectralFrame with Welch operation added to graph")
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
            lineage=cast(Any, self)._required_semantic_lineage(),
            previous=self._as_base_frame,
        )

    @recipe_operation(
        "wandas.audio.noct_spectrum",
        validate_params=validate_noct_recipe_params,
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

        Each output value is the RMS amplitude in one fractional-octave band
        and retains the input channel's physical unit. ``NOctFrame.dB`` applies
        ``20 * log10(band_rms / channel_ref)``.

        Args:
            fmin: Minimum center frequency (Hz). Default is 25 Hz.
            fmax: Maximum center frequency (Hz). Default is 20000 Hz.
            n: Band division (1: octave, 3: 1/3 octave). Default is 3.
            G: Exact center-frequency ratio convention. Use 10 for base
                ``10**(3/10)`` or 2 for base 2. Default is 10.
            fr: Reference frequency (Hz). Default is 1000 Hz.

        Returns:
            A lazy NOctFrame containing per-band RMS amplitudes.
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
        """Calculate a one-sided peak-amplitude Short-Time Fourier Transform.

        Each time frame is normalized by its window's coherent gain. Complex
        values retain the input physical unit; an on-bin sinusoid's magnitude
        is its peak amplitude.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between frames.
                Default is ``n_fft // 4``.
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

    @recipe_operation("wandas.audio.coherence", version=2)
    def coherence(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
    ) -> "CoherenceFrame":
        """Calculate typed magnitude-squared coherence for every channel pair.

        The result is a :class:`CoherenceFrame` with flattened ``(pair,
        frequency)`` storage and output-major/input-minor pair order.  Its real
        raw values are dimensionless and lie in ``[0, 1]``; ``NaN`` is retained
        for undefined zero-energy bins.  Pair roles, source identity, domains,
        and row order are carried by immutable typed state, not labels or
        operation history.  See the spectral numerical contracts for the
        canonical mathematical definition.

        Sampling rate and user metadata are preserved.  Each pair's
        ``source_time_offset`` is derived from its input-role source offset, and
        input calibration is consumed before the pairwise operation.  Constructing
        the result remains Dask-lazy; accessing data or plotting is the
        materialization boundary.  Invalid spectral parameters, input shape, or
        coherence-domain values raise an actionable ``TypeError`` or
        ``ValueError`` instead of being silently coerced.

        Args:
            n_fft: Number of FFT points. Default is 2048.
            hop_length: Number of samples between frames.
                Default is n_fft//4.
            win_length: Window length. Default is n_fft.
            window: Window type. Default is "hann".
            detrend: Detrend method. Options: "constant", "linear", None.

        Returns:
            CoherenceFrame whose public single-pair shape is ``(frequency,)`` and
            whose multi-pair shape is ``(pair, frequency)``.  Use ``.coherence``
            for the quantity-specific raw values.

        Example:
            ``coherence = frame.coherence(n_fft=1024, window="hann")``
        """
        from ..pairwise import CoherenceFrame

        return _cross_channel_spectral_transform(
            self,
            "coherence",
            "Coherence of",
            "$\\gamma_{{{out_label}, {in_label}}}$",
            CoherenceFrame,
            "coherence",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
        )

    @recipe_operation("wandas.audio.coherence", version=1)
    def _coherence_recipe_v1(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
    ) -> "CoherenceFrame":
        """Replay the released coherence pair-label order."""
        from ..pairwise import CoherenceFrame

        return _cross_channel_spectral_transform(
            self,
            "coherence",
            "Coherence of",
            "$\\gamma_{{{in_label}, {out_label}}}$",
            CoherenceFrame,
            "coherence",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
        )

    @recipe_operation("wandas.audio.csd", version=2)
    def csd(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "CrossSpectralFrame":
        """Calculate a typed cross-spectral density matrix.

        The result is a :class:`CrossSpectralFrame` with flattened
        ``(pair, frequency)`` storage and output-major/input-minor pair order.
        Each raw complex row stores ``P_out_in = conj(X_input) * X_output``;
        pair domains provide the unit and reference, with ``/Hz`` included for
        ``scaling="density"``.  Pair roles and domains are immutable typed state;
        labels and operation history are display/provenance views only.  See the
        spectral numerical contracts for the canonical definition and scaling.

        Sampling rate and user metadata are preserved.  Pair
        ``source_time_offset`` uses the input-role source offset, and input
        calibration is consumed before constructing output metadata.  The result
        stays Dask-lazy until data, a property, or a plot is materialized.
        Invalid spectral parameters or domain/shape violations raise an actionable
        ``TypeError`` or ``ValueError``.  Use the quantity-specific ``magnitude``,
        ``phase``, and ``level_db`` properties; pairwise A-weighting is rejected.

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
            CrossSpectralFrame whose public single-pair shape is ``(frequency,)``
            and whose multi-pair shape is ``(pair, frequency)``.

        Example:
            ``spectrum = frame.csd(n_fft=1024, scaling="density")``
        """
        from ..pairwise import CrossSpectralFrame

        return _cross_channel_spectral_transform(
            self,
            "csd",
            "CSD of",
            "csd({out_label}, {in_label})",
            CrossSpectralFrame,
            "csd",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )

    @recipe_operation("wandas.audio.csd", version=1)
    def _csd_recipe_v1(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "CrossSpectralFrame":
        """Replay the released CSD pair-label order."""
        from ..pairwise import CrossSpectralFrame

        return _cross_channel_spectral_transform(
            self,
            "csd",
            "CSD of",
            "csd({in_label}, {out_label})",
            CrossSpectralFrame,
            "csd",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )

    @recipe_operation("wandas.audio.transfer_function", version=2)
    def transfer_function(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "TransferFunctionFrame":
        """Calculate the canonical typed output/input transfer-function matrix.

        The v2 result is a :class:`TransferFunctionFrame` with flattened
        ``(pair, frequency)`` storage and output-major/input-minor pair order.  It
        stores ``H_out_in = P_out_in / P_in_in`` and carries the denominator
        definition, pair roles, unit/reference domain, and row order as immutable
        typed state.  Labels and operation history do not define its meaning; the
        released v1 denominator contract is replayed separately by the v1 Recipe
        handler.  See the spectral numerical contracts for the canonical formulas.

        Sampling rate and user metadata are preserved.  Pair
        ``source_time_offset`` uses the input-role source offset, and input
        calibration is consumed before output metadata is derived.  Construction
        remains Dask-lazy; accessing data, a property, or a plot materializes the
        requested values.  Invalid spectral parameters or shape/domain violations
        raise an actionable ``TypeError`` or ``ValueError``.  ``gain_db`` is
        available only after selecting dimensionless pairs; ``transfer_level_db``
        uses each pair's explicit reference ratio.  Pairwise A-weighting is
        rejected.

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
            TransferFunctionFrame whose public single-pair shape is ``(frequency,)``
            and whose multi-pair shape is ``(pair, frequency)``.

        Example:
            ``transfer = frame.transfer_function(n_fft=1024, scaling="spectrum")``
        """
        from ..pairwise import TransferFunctionFrame

        return _cross_channel_spectral_transform(
            self,
            "transfer_function",
            "Transfer function of",
            "$H_{{{out_label}, {in_label}}}$",
            TransferFunctionFrame,
            "transfer",
            "input",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )

    @recipe_operation("wandas.audio.transfer_function", version=1)
    def _transfer_function_recipe_v1(
        self: TransformFrameProtocol,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "TransferFunctionFrame":
        """Replay the released transfer-function denominator contract."""
        from wandas.processing.spectral import _RecipeTransferFunctionV1

        from ..pairwise import TransferFunctionFrame

        operation = _RecipeTransferFunctionV1(
            self.sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )
        return _cross_channel_spectral_transform(
            self,
            "transfer_function",
            "Transfer function of",
            "$H_{{{in_label}, {out_label}}}$",
            TransferFunctionFrame,
            "transfer",
            "output",
            operation_override=operation,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )
