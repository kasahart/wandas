"""Module providing mixins related to signal processing."""

import logging
from typing import TYPE_CHECKING, Any, Optional, Union, cast

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

    def normalize(
        self: T_Processing, target_level: float = -20, channel_wise: bool = True
    ) -> T_Processing:
        """Normalize signal levels.

        This method adjusts the signal amplitude to reach the target RMS level.

        Args:
            target_level: Target RMS level (dB). Default is -20.
            channel_wise: If True, normalize each channel individually.
                If False, apply the same scaling to all channels.

        Returns:
            New ChannelFrame containing the normalized signal
        """
        logger.debug(
            f"Setting up normalize: target_level={target_level}, "
            f"channel_wise={channel_wise} (lazy)"
        )
        result = self.apply_operation(
            "normalize", target_level=target_level, channel_wise=channel_wise
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

    def sum(self: T_Processing) -> T_Processing:
        """Sum all channels.

        Returns:
            A new ChannelFrame with summed signal.
        """
        result = self.apply_operation("sum")
        return cast(T_Processing, result)

    def mean(self: T_Processing) -> T_Processing:
        """Average all channels.

        Returns:
            A new ChannelFrame with averaged signal.
        """
        result = self.apply_operation("mean")
        return cast(T_Processing, result)

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

        # Update sampling rate
        result_obj = cast(T_Processing, result)
        if hasattr(result_obj, "sampling_rate"):
            result_obj.sampling_rate = frame.sampling_rate / hop_length

        return result_obj

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
