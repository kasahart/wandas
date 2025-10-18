"""Module providing mixins related to signal processing."""

import logging
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from wandas.core.metadata import ChannelMetadata

from .protocols import ProcessingFrameProtocol, T_Processing

if TYPE_CHECKING:
    from librosa._typing import (
        _FloatLike_co,
        _IntLike_co,
        _PadModeSTFT,
        _WindowSpec,
    )
logger = logging.getLogger(__name__)


class ChannelProcessingMixin:
    """Mixin that provides methods related to signal processing.

    This mixin provides processing methods applied to audio signals and
    other time-series data, such as signal processing filters and
    transformation operations.
    """

    def high_pass_filter(
        self: T_Processing, cutoff: float, order: int = 4
    ) -> T_Processing:
        """Apply a high-pass filter to the signal.

        Args:
            cutoff: Filter cutoff frequency (Hz)
            order: Filter order. Default is 4.

        Returns:
            New ChannelFrame after filter application
        """
        logger.debug(
            f"Setting up highpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        result = self.apply_operation("highpass_filter", cutoff=cutoff, order=order)
        return cast(T_Processing, result)

    def low_pass_filter(
        self: T_Processing, cutoff: float, order: int = 4
    ) -> T_Processing:
        """Apply a low-pass filter to the signal.

        Args:
            cutoff: Filter cutoff frequency (Hz)
            order: Filter order. Default is 4.

        Returns:
            New ChannelFrame after filter application
        """
        logger.debug(
            f"Setting up lowpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        result = self.apply_operation("lowpass_filter", cutoff=cutoff, order=order)
        return cast(T_Processing, result)

    def band_pass_filter(
        self: T_Processing, low_cutoff: float, high_cutoff: float, order: int = 4
    ) -> T_Processing:
        """Apply a band-pass filter to the signal.

        Args:
            low_cutoff: Lower cutoff frequency (Hz)
            high_cutoff: Higher cutoff frequency (Hz)
            order: Filter order. Default is 4.

        Returns:
            New ChannelFrame after filter application
        """
        logger.debug(
            f"Setting up bandpass filter: low_cutoff={low_cutoff}, "
            f"high_cutoff={high_cutoff}, order={order} (lazy)"
        )
        result = self.apply_operation(
            "bandpass_filter",
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            order=order,
        )
        return cast(T_Processing, result)

    def normalize(
        self: T_Processing,
        norm: Union[float, None] = float("inf"),
        axis: Union[int, None] = -1,
        threshold: Union[float, None] = None,
        fill: Union[bool, None] = None,
    ) -> T_Processing:
        """Normalize signal levels using librosa.util.normalize.

        This method normalizes the signal amplitude according to the specified norm.

        Args:
            norm: Norm type. Default is np.inf (maximum absolute value normalization).
                Supported values:
                - np.inf: Maximum absolute value normalization
                - -np.inf: Minimum absolute value normalization
                - 0: Peak normalization
                - float: Lp norm
                - None: No normalization
            axis: Axis along which to normalize. Default is -1 (time axis).
                - -1: Normalize along time axis (each channel independently)
                - None: Global normalization across all axes
                - int: Normalize along specified axis
            threshold: Threshold below which values are considered zero.
                If None, no threshold is applied.
            fill: Value to fill when the norm is zero.
                If None, the zero vector remains zero.

        Returns:
            New ChannelFrame containing the normalized signal

        Examples:
            >>> import wandas as wd
            >>> signal = wd.read_wav("audio.wav")
            >>> # Normalize to maximum absolute value of 1.0 (per channel)
            >>> normalized = signal.normalize()
            >>> # Global normalization across all channels
            >>> normalized_global = signal.normalize(axis=None)
            >>> # L2 normalization
            >>> normalized_l2 = signal.normalize(norm=2)
        """
        logger.debug(
            f"Setting up normalize: norm={norm}, axis={axis}, "
            f"threshold={threshold}, fill={fill} (lazy)"
        )
        result = self.apply_operation(
            "normalize", norm=norm, axis=axis, threshold=threshold, fill=fill
        )
        return cast(T_Processing, result)

    def a_weighting(self: T_Processing) -> T_Processing:
        """Apply A-weighting filter to the signal.

        A-weighting adjusts the frequency response to approximate human
        auditory perception, according to the IEC 61672-1:2013 standard.

        Returns:
            New ChannelFrame containing the A-weighted signal
        """
        result = self.apply_operation("a_weighting")
        return cast(T_Processing, result)

    def abs(self: T_Processing) -> T_Processing:
        """Compute the absolute value of the signal.

        Returns:
            New ChannelFrame containing the absolute values
        """
        result = self.apply_operation("abs")
        return cast(T_Processing, result)

    def power(self: T_Processing, exponent: float = 2.0) -> T_Processing:
        """Compute the power of the signal.

        Args:
            exponent: Exponent to raise the signal to. Default is 2.0.

        Returns:
            New ChannelFrame containing the powered signal
        """
        result = self.apply_operation("power", exponent=exponent)
        return cast(T_Processing, result)

    def _reduce_channels(self: T_Processing, op: str) -> T_Processing:
        """Helper to reduce all channels with the given operation ('sum' or 'mean')."""
        if op == "sum":
            reduced_data = self._data.sum(axis=0, keepdims=True)
            label = "sum"
        elif op == "mean":
            reduced_data = self._data.mean(axis=0, keepdims=True)
            label = "mean"
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")

        units = [ch.unit for ch in self._channel_metadata]
        if all(u == units[0] for u in units):
            reduced_unit = units[0]
        else:
            reduced_unit = ""

        reduced_extra = {"source_extras": [ch.extra for ch in self._channel_metadata]}
        new_channel_metadata = [
            ChannelMetadata(
                label=label,
                unit=reduced_unit,
                extra=reduced_extra,
            )
        ]
        new_history = (
            self.operation_history.copy() if hasattr(self, "operation_history") else []
        )
        new_history.append({"operation": op})
        new_metadata = self.metadata.copy() if hasattr(self, "metadata") else {}
        result = self._create_new_instance(
            data=reduced_data,
            metadata=new_metadata,
            operation_history=new_history,
            channel_metadata=new_channel_metadata,
        )
        return result

    def sum(self: T_Processing) -> T_Processing:
        """Sum all channels.

        Returns:
            A new ChannelFrame with summed signal.
        """
        return cast(T_Processing, cast(Any, self)._reduce_channels("sum"))

    def mean(self: T_Processing) -> T_Processing:
        """Average all channels.

        Returns:
            A new ChannelFrame with averaged signal.
        """
        return cast(T_Processing, cast(Any, self)._reduce_channels("mean"))

    def trim(
        self: T_Processing,
        start: float = 0,
        end: Optional[float] = None,
    ) -> T_Processing:
        """Trim the signal to the specified time range.

        Args:
            start: Start time (seconds)
            end: End time (seconds)

        Returns:
            New ChannelFrame containing the trimmed signal

        Raises:
            ValueError: If end time is earlier than start time
        """
        if end is None:
            end = self.duration
        if start > end:
            raise ValueError("start must be less than end")
        result = self.apply_operation("trim", start=start, end=end)
        return cast(T_Processing, result)

    def fix_length(
        self: T_Processing,
        length: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> T_Processing:
        """Adjust the signal to the specified length.

        Args:
            duration: Signal length in seconds
            length: Signal length in samples

        Returns:
            New ChannelFrame containing the adjusted signal
        """

        result = self.apply_operation("fix_length", length=length, duration=duration)
        return cast(T_Processing, result)

    def rms_trend(
        self: T_Processing,
        frame_length: int = 2048,
        hop_length: int = 512,
        dB: bool = False,  # noqa: N803
        Aw: bool = False,  # noqa: N803
    ) -> T_Processing:
        """Compute the RMS trend of the signal.

        This method calculates the root mean square value over a sliding window.

        Args:
            frame_length: Size of the sliding window in samples. Default is 2048.
            hop_length: Hop length between windows in samples. Default is 512.
            dB: Whether to return RMS values in decibels. Default is False.
            Aw: Whether to apply A-weighting. Default is False.

        Returns:
            New ChannelFrame containing the RMS trend
        """
        # Access _channel_metadata to retrieve reference values
        frame = cast(ProcessingFrameProtocol, self)

        # Ensure _channel_metadata exists before referencing
        ref_values = []
        if hasattr(frame, "_channel_metadata") and frame._channel_metadata:
            ref_values = [ch.ref for ch in frame._channel_metadata]

        result = self.apply_operation(
            "rms_trend",
            frame_length=frame_length,
            hop_length=hop_length,
            ref=ref_values,
            dB=dB,
            Aw=Aw,
        )

        # Sampling rate update is handled by the Operation class
        return cast(T_Processing, result)

    def channel_difference(
        self: T_Processing, other_channel: Union[int, str] = 0
    ) -> T_Processing:
        """Compute the difference between channels.

        Args:
            other_channel: Index or label of the reference channel. Default is 0.

        Returns:
            New ChannelFrame containing the channel difference
        """
        # label2index is a method of BaseFrame
        if isinstance(other_channel, str):
            if hasattr(self, "label2index"):
                other_channel = self.label2index(other_channel)

        result = self.apply_operation("channel_difference", other_channel=other_channel)
        return cast(T_Processing, result)

    def resampling(
        self: T_Processing,
        target_sr: float,
        **kwargs: Any,
    ) -> T_Processing:
        """Resample audio data.

        Args:
            target_sr: Target sampling rate (Hz)
            **kwargs: Additional resampling parameters

        Returns:
            Resampled ChannelFrame
        """
        return cast(
            T_Processing,
            self.apply_operation(
                "resampling",
                target_sr=target_sr,
                **kwargs,
            ),
        )

    def hpss_harmonic(
        self: T_Processing,
        kernel_size: Union[
            "_IntLike_co", tuple["_IntLike_co", "_IntLike_co"], list["_IntLike_co"]
        ] = 31,
        power: float = 2,
        margin: Union[
            "_FloatLike_co",
            tuple["_FloatLike_co", "_FloatLike_co"],
            list["_FloatLike_co"],
        ] = 1,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "_WindowSpec" = "hann",
        center: bool = True,
        pad_mode: "_PadModeSTFT" = "constant",
    ) -> T_Processing:
        """
        Extract harmonic components using HPSS
         (Harmonic-Percussive Source Separation).

        This method separates the harmonic (tonal) components from the signal.

        Args:
            kernel_size: Median filter size for HPSS.
            power: Exponent for the Weiner filter used in HPSS.
            margin: Margin size for the separation.
            n_fft: Size of FFT window.
            hop_length: Hop length for STFT.
            win_length: Window length for STFT.
            window: Window type for STFT.
            center: If True, center the frames.
            pad_mode: Padding mode for STFT.

        Returns:
            A new ChannelFrame containing the harmonic components.
        """
        result = self.apply_operation(
            "hpss_harmonic",
            kernel_size=kernel_size,
            power=power,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return cast(T_Processing, result)

    def hpss_percussive(
        self: T_Processing,
        kernel_size: Union[
            "_IntLike_co", tuple["_IntLike_co", "_IntLike_co"], list["_IntLike_co"]
        ] = 31,
        power: float = 2,
        margin: Union[
            "_FloatLike_co",
            tuple["_FloatLike_co", "_FloatLike_co"],
            list["_FloatLike_co"],
        ] = 1,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "_WindowSpec" = "hann",
        center: bool = True,
        pad_mode: "_PadModeSTFT" = "constant",
    ) -> T_Processing:
        """
        Extract percussive components using HPSS
        (Harmonic-Percussive Source Separation).

        This method separates the percussive (tonal) components from the signal.

        Args:
            kernel_size: Median filter size for HPSS.
            power: Exponent for the Weiner filter used in HPSS.
            margin: Margin size for the separation.

        Returns:
            A new ChannelFrame containing the harmonic components.
        """
        result = self.apply_operation(
            "hpss_percussive",
            kernel_size=kernel_size,
            power=power,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return cast(T_Processing, result)

    def loudness_zwtv(self: T_Processing, field_type: str = "free") -> T_Processing:
        """
        Calculate time-varying loudness using Zwicker method (ISO 532-1:2017).

        This method computes the loudness of non-stationary signals according to
        the Zwicker method, as specified in ISO 532-1:2017. The loudness is
        calculated in sones, where a doubling of sones corresponds to a doubling
        of perceived loudness.

        Args:
            field_type: Type of sound field. Options:
                - 'free': Free field (sound from a specific direction)
                - 'diffuse': Diffuse field (sound from all directions)
                Default is 'free'.

        Returns:
            New ChannelFrame containing time-varying loudness values in sones.
            Each channel is processed independently.
            The output sampling rate is adjusted based on the loudness
            calculation time resolution (typically ~500 Hz for 2ms steps).

        Raises:
            ValueError: If field_type is not 'free' or 'diffuse'

        Examples:
            Calculate loudness for a signal:
            >>> import wandas as wd
            >>> signal = wd.read_wav("audio.wav")
            >>> loudness = signal.loudness_zwtv(field_type="free")
            >>> loudness.plot(title="Time-varying Loudness")

            Compare free field and diffuse field:
            >>> loudness_free = signal.loudness_zwtv(field_type="free")
            >>> loudness_diffuse = signal.loudness_zwtv(field_type="diffuse")

        Notes:
            - The output contains time-varying loudness values in sones
            - Typical loudness: 1 sone ≈ 40 phon (loudness level)
            - The time resolution is approximately 2ms (determined by the algorithm)
            - For multi-channel signals, loudness is calculated per channel
            - The output sampling rate is updated to reflect the time resolution

        References:
            ISO 532-1:2017, "Acoustics — Methods for calculating loudness —
            Part 1: Zwicker method"
        """
        result = self.apply_operation("loudness_zwtv", field_type=field_type)

        # Sampling rate update is handled by the Operation class
        return cast(T_Processing, result)

    def loudness_zwst(self: T_Processing, field_type: str = "free") -> T_Processing:
        """
        Calculate steady-state loudness using Zwicker method (ISO 532-1:2017).

        This method computes the loudness of stationary (steady) signals according to
        the Zwicker method, as specified in ISO 532-1:2017. The loudness is
        calculated in sones, where a doubling of sones corresponds to a doubling
        of perceived loudness.

        This method is suitable for analyzing steady sounds such as fan noise,
        constant machinery sounds, or other stationary signals.

        Args:
            field_type: Type of sound field. Options:
                - 'free': Free field (sound from a specific direction)
                - 'diffuse': Diffuse field (sound from all directions)
                Default is 'free'.

        Returns:
            New ChannelFrame containing a single steady-state loudness value
            (in sones) for each channel. The output will have shape (channels, 1).

        Raises:
            ValueError: If field_type is not 'free' or 'diffuse'

        Examples:
            Calculate steady-state loudness for a fan noise:
            >>> import wandas as wd
            >>> signal = wd.read_wav("fan_noise.wav")
            >>> loudness = signal.loudness_zwst(field_type="free")
            >>> print(f"Steady-state loudness: {loudness.data[0, 0]:.2f} sones")

            Compare free field and diffuse field:
            >>> loudness_free = signal.loudness_zwst(field_type="free")
            >>> loudness_diffuse = signal.loudness_zwst(field_type="diffuse")
            >>> print(f"Free field: {loudness_free.data[0, 0]:.2f} sones")
            >>> print(f"Diffuse field: {loudness_diffuse.data[0, 0]:.2f} sones")

        Notes:
            - The output contains a single loudness value in sones per channel
            - Typical loudness: 1 sone ≈ 40 phon (loudness level)
            - For multi-channel signals, loudness is calculated independently per channel
            - This method is designed for stationary signals (constant sounds)
            - For time-varying signals, use loudness_zwtv() instead

        References:
            ISO 532-1:2017, "Acoustics — Methods for calculating loudness —
            Part 1: Zwicker method"
        """
        result = self.apply_operation("loudness_zwst", field_type=field_type)

        # No sampling rate update needed for steady-state loudness
        return cast(T_Processing, result)

