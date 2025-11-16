"""Module providing mixins related to signal processing."""

import logging
from typing import TYPE_CHECKING, Any, Union, cast

from wandas.core.metadata import ChannelMetadata
from wandas.frames.roughness import RoughnessFrame
from wandas.processing import create_operation

from .protocols import ProcessingFrameProtocol, T_Processing

if TYPE_CHECKING:
    from librosa._typing import (
        _FloatLike_co,
        _IntLike_co,
        _PadModeSTFT,
        _WindowSpec,
    )

    from wandas.core.base_frame import BaseFrame
    from wandas.utils.types import NDArrayReal
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
        norm: float | None = float("inf"),
        axis: int | None = -1,
        threshold: float | None = None,
        fill: bool | None = None,
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

    def remove_dc(self: T_Processing) -> T_Processing:
        """Remove DC component (DC offset) from the signal.

        This method removes the DC (direct current) component by subtracting
        the mean value from each channel. This is equivalent to centering the
        signal around zero.

        Returns:
            New ChannelFrame with DC component removed

        Examples:
            >>> import wandas as wd
            >>> import numpy as np
            >>> # Create signal with DC offset
            >>> signal = wd.read_wav("audio.wav")
            >>> signal_with_dc = signal + 2.0  # Add DC offset
            >>> # Remove DC offset
            >>> signal_clean = signal_with_dc.remove_dc()
            >>> # Verify DC removal
            >>> assert np.allclose(signal_clean.data.mean(axis=1), 0, atol=1e-10)

        Notes:
            - This operation is performed per channel
            - Equivalent to applying a high-pass filter with very low cutoff
            - Useful for removing sensor drift or measurement offset
        """
        logger.debug("Setting up DC removal (lazy)")
        result = self.apply_operation("remove_dc")
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
        end: float | None = None,
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
        length: int | None = None,
        duration: float | None = None,
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
        self: T_Processing, other_channel: int | str = 0
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
        hop_length: int | None = None,
        win_length: int | None = None,
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
        hop_length: int | None = None,
        win_length: int | None = None,
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

            **Time axis convention:**
            The time axis in the returned frame represents the start time of
            each 2ms analysis step. This differs slightly from the MoSQITo
            library, which uses the center time of each step. For example:

            - wandas time: [0.000s, 0.002s, 0.004s, ...] (step start)
            - MoSQITo time: [0.001s, 0.003s, 0.005s, ...] (step center)

            The difference is very small (~1ms) and does not affect the loudness
            values themselves. This design choice ensures consistency with
            wandas's time axis convention across all frame types.

        References:
            ISO 532-1:2017, "Acoustics — Methods for calculating loudness —
            Part 1: Zwicker method"
        """
        result = self.apply_operation("loudness_zwtv", field_type=field_type)

        # Sampling rate update is handled by the Operation class
        return cast(T_Processing, result)

    def loudness_zwst(self: T_Processing, field_type: str = "free") -> "NDArrayReal":
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
            Loudness values in sones, one per channel. Shape: (n_channels,)

        Raises:
            ValueError: If field_type is not 'free' or 'diffuse'

        Examples:
            Calculate steady-state loudness for a fan noise:
            >>> import wandas as wd
            >>> signal = wd.read_wav("fan_noise.wav")
            >>> loudness = signal.loudness_zwst(field_type="free")
            >>> print(f"Channel 0 loudness: {loudness[0]:.2f} sones")
            >>> print(f"Mean loudness: {loudness.mean():.2f} sones")

            Compare free field and diffuse field:
            >>> loudness_free = signal.loudness_zwst(field_type="free")
            >>> loudness_diffuse = signal.loudness_zwst(field_type="diffuse")
            >>> print(f"Free field: {loudness_free[0]:.2f} sones")
            >>> print(f"Diffuse field: {loudness_diffuse[0]:.2f} sones")

        Notes:
            - Returns a 1D array with one loudness value per channel
            - Typical loudness: 1 sone ≈ 40 phon (loudness level)
            - For multi-channel signals, loudness is calculated independently
              per channel
            - This method is designed for stationary signals (constant sounds)
            - For time-varying signals, use loudness_zwtv() instead
            - Similar to the rms property, returns NDArrayReal for consistency

        References:
            ISO 532-1:2017, "Acoustics — Methods for calculating loudness —
            Part 1: Zwicker method"
        """
        # Treat self as a ProcessingFrameProtocol so mypy understands
        # where sampling_rate and data come from.
        from wandas.processing.psychoacoustic import LoudnessZwst
        from wandas.utils.types import NDArrayReal

        # Create operation instance
        operation = LoudnessZwst(self.sampling_rate, field_type=field_type)

        # Get data (triggers computation if lazy)
        data = self.data

        # Ensure data is 2D (n_channels, n_samples)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Process the array using the public API and materialize to NumPy
        result = operation.process_array(data).compute()

        # Squeeze to get 1D array (n_channels,)
        loudness_values: NDArrayReal = result.squeeze()

        # Ensure it's 1D even for single channel
        if loudness_values.ndim == 0:
            loudness_values = loudness_values.reshape(1)

        return loudness_values

    def roughness_dw(self: T_Processing, overlap: float = 0.5) -> T_Processing:
        """Calculate time-varying roughness using Daniel and Weber method.

        Roughness is a psychoacoustic metric that quantifies the perceived
        harshness or roughness of a sound, measured in asper. This method
        implements the Daniel & Weber (1997) standard calculation.

        The calculation follows the standard formula:
        R = 0.25 * sum(R'_i) for i=1 to 47 Bark bands

        Args:
            overlap: Overlapping coefficient for 200ms analysis windows (0.0 to 1.0).
                - overlap=0.5: 100ms hop → ~10 Hz output sampling rate
                - overlap=0.0: 200ms hop → ~5 Hz output sampling rate
                Default is 0.5.

        Returns:
            New ChannelFrame containing time-varying roughness values in asper.
            The output sampling rate depends on the overlap parameter.

        Raises:
            ValueError: If overlap is not in the range [0.0, 1.0]

        Examples:
            Calculate roughness for a motor noise:
            >>> import wandas as wd
            >>> signal = wd.read_wav("motor_noise.wav")
            >>> roughness = signal.roughness_dw(overlap=0.5)
            >>> roughness.plot(ylabel="Roughness [asper]")

            Analyze roughness statistics:
            >>> mean_roughness = roughness.data.mean()
            >>> max_roughness = roughness.data.max()
            >>> print(f"Mean: {mean_roughness:.2f} asper")
            >>> print(f"Max: {max_roughness:.2f} asper")

            Compare before and after modification:
            >>> before = wd.read_wav("motor_before.wav").roughness_dw()
            >>> after = wd.read_wav("motor_after.wav").roughness_dw()
            >>> improvement = before.data.mean() - after.data.mean()
            >>> print(f"Roughness reduction: {improvement:.2f} asper")

        Notes:
            - Returns a ChannelFrame with time-varying roughness values
            - Typical roughness values: 0-2 asper for most sounds
            - Higher values indicate rougher, harsher sounds
            - For multi-channel signals, roughness is calculated independently
              per channel
            - This is the standard-compliant total roughness (R)
            - For detailed Bark-band analysis, use roughness_dw_spec() instead

            **Time axis convention:**
            The time axis in the returned frame represents the start time of
            each 200ms analysis window. This differs from the MoSQITo library,
            which uses the center time of each window. For example:

            - wandas time: [0.0s, 0.1s, 0.2s, ...] (window start)
            - MoSQITo time: [0.1s, 0.2s, 0.3s, ...] (window center)

            The difference is constant (half the window duration = 100ms) and
            does not affect the roughness values themselves. This design choice
            ensures consistency with wandas's time axis convention across all
            frame types.

        References:
            Daniel, P., & Weber, R. (1997). "Psychoacoustical roughness:
            Implementation of an optimized model." Acustica, 83, 113-123.
        """
        logger.debug(f"Applying roughness_dw operation with overlap={overlap} (lazy)")
        result = self.apply_operation("roughness_dw", overlap=overlap)
        return cast(T_Processing, result)

    def roughness_dw_spec(self: T_Processing, overlap: float = 0.5) -> "RoughnessFrame":
        """Calculate specific roughness with Bark-band frequency information.

        This method returns detailed roughness analysis data organized by
        Bark frequency bands over time, allowing for frequency-specific
        roughness analysis. It uses the Daniel & Weber (1997) method.

        The relationship between total roughness and specific roughness:
        R = 0.25 * sum(R'_i) for i=1 to 47 Bark bands

        Args:
            overlap: Overlapping coefficient for 200ms analysis windows (0.0 to 1.0).
                - overlap=0.5: 100ms hop → ~10 Hz output sampling rate
                - overlap=0.0: 200ms hop → ~5 Hz output sampling rate
                Default is 0.5.

        Returns:
            RoughnessFrame containing:
                - data: Specific roughness by Bark band, shape (47, n_time)
                        for mono or (n_channels, 47, n_time) for multi-channel
                - bark_axis: Frequency axis in Bark scale (47 values, 0.5-23.5)
                - time: Time axis for each analysis frame
                - overlap: Overlap coefficient used
                - plot(): Method for Bark-Time heatmap visualization

        Raises:
            ValueError: If overlap is not in the range [0.0, 1.0]

        Examples:
            Analyze frequency-specific roughness:
            >>> import wandas as wd
            >>> import numpy as np
            >>> signal = wd.read_wav("motor.wav")
            >>> roughness_spec = signal.roughness_dw_spec(overlap=0.5)
            >>>
            >>> # Plot Bark-Time heatmap
            >>> roughness_spec.plot(cmap="viridis", title="Roughness Analysis")
            >>>
            >>> # Find dominant Bark band
            >>> dominant_idx = roughness_spec.data.mean(axis=1).argmax()
            >>> dominant_bark = roughness_spec.bark_axis[dominant_idx]
            >>> print(f"Most contributing band: {dominant_bark:.1f} Bark")
            >>>
            >>> # Extract specific Bark band time series
            >>> bark_10_idx = np.argmin(np.abs(roughness_spec.bark_axis - 10.0))
            >>> roughness_at_10bark = roughness_spec.data[bark_10_idx, :]
            >>>
            >>> # Verify standard formula
            >>> total_roughness = 0.25 * roughness_spec.data.sum(axis=-2)
            >>> # This should match signal.roughness_dw(overlap=0.5).data

        Notes:
            - Returns a RoughnessFrame (not ChannelFrame)
            - Contains 47 Bark bands from 0.5 to 23.5 Bark
            - Each Bark band corresponds to a critical band of hearing
            - Useful for identifying which frequencies contribute most to roughness
            - The specific roughness can be integrated to obtain total roughness
            - For simple time-series analysis, use roughness_dw() instead

            **Time axis convention:**
            The time axis represents the start time of each 200ms analysis
            window, consistent with roughness_dw() and other wandas methods.

        References:
            Daniel, P., & Weber, R. (1997). "Psychoacoustical roughness:
            Implementation of an optimized model." Acustica, 83, 113-123.
        """

        params = {"overlap": overlap}
        operation_name = "roughness_dw_spec"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # Create operation instance via factory
        operation = create_operation(operation_name, self.sampling_rate, **params)

        # Apply processing lazily to self._data (Dask)
        r_spec_dask = operation.process(self._data)

        # Get metadata updates (sampling rate, bark_axis)
        metadata_updates = operation.get_metadata_updates()

        # Build metadata and history
        new_metadata = {**self.metadata, **params}
        new_history = [
            *self.operation_history,
            {"operation": operation_name, "params": params},
        ]

        # Extract bark_axis with proper type handling
        bark_axis_value = metadata_updates.get("bark_axis")
        if bark_axis_value is None:
            raise ValueError("Operation did not provide bark_axis in metadata")

        # Create RoughnessFrame. operation.get_metadata_updates() should provide
        # sampling_rate and bark_axis
        roughness_frame = RoughnessFrame(
            data=r_spec_dask,
            sampling_rate=metadata_updates.get("sampling_rate", self.sampling_rate),
            bark_axis=bark_axis_value,
            overlap=overlap,
            label=f"{self.label}_roughness_spec" if self.label else "roughness_spec",
            metadata=new_metadata,
            operation_history=new_history,
            channel_metadata=self._channel_metadata,
            previous=cast("BaseFrame[NDArrayReal]", self),
        )

        logger.debug(
            "Created RoughnessFrame via operation %s, shape=%s, sampling_rate=%.2f Hz",
            operation_name,
            r_spec_dask.shape,
            roughness_frame.sampling_rate,
        )

        return roughness_frame

    def fade(self: T_Processing, fade_ms: float = 50) -> T_Processing:
        """Apply symmetric fade-in and fade-out to the signal using Tukey window.

        This method applies a symmetric fade-in and fade-out envelope to the signal
        using a Tukey (tapered cosine) window. The fade duration is the same for
        both the beginning and end of the signal.

        Args:
            fade_ms: Fade duration in milliseconds for each end of the signal.
                The total fade duration is 2 * fade_ms. Default is 50 ms.
                Must be positive and less than half the signal duration.

        Returns:
            New ChannelFrame containing the faded signal

        Raises:
            ValueError: If fade_ms is negative or too long for the signal

        Examples:
            >>> import wandas as wd
            >>> signal = wd.read_wav("audio.wav")
            >>> # Apply 10ms fade-in and fade-out
            >>> faded = signal.fade(fade_ms=10.0)
            >>> # Apply very short fade (almost no effect)
            >>> faded_short = signal.fade(fade_ms=0.1)

        Notes:
            - Uses SciPy's Tukey window for smooth fade transitions
            - Fade is applied symmetrically to both ends of the signal
            - The Tukey window alpha parameter is computed automatically
              based on the fade duration and signal length
            - For multi-channel signals, the same fade envelope is applied
              to all channels
            - Lazy evaluation is preserved - computation occurs only when needed
        """
        logger.debug(f"Setting up fade: fade_ms={fade_ms} (lazy)")
        result = self.apply_operation("fade", fade_ms=fade_ms)
        return cast(T_Processing, result)

    def sharpness_din(
        self: T_Processing,
        weighting: str = "din",
        field_type: str = "free",
    ) -> T_Processing:
        """Calculate sharpness using DIN 45692 method.

        This method computes the time-varying sharpness of the signal
        according to DIN 45692 standard, which quantifies the perceived
        sharpness of sounds.

        Parameters
        ----------
        weighting : str, default="din"
            Weighting type for sharpness calculation.
        field_type : str, default="free"
            Type of sound field. Options:
            - 'free': Free field (sound from a specific direction)
            - 'diffuse': Diffuse field (sound from all directions)

        Returns
        -------
        T_Processing
            New ChannelFrame containing sharpness time series in acum.
            The output sampling rate is approximately 500 Hz (2ms time steps).

        Raises
        ------
        ValueError
            If the signal sampling rate is not supported by the algorithm.

        Examples
        --------
        >>> import wandas as wd
        >>> signal = wd.read_wav("sharp_sound.wav")
        >>> sharpness = signal.sharpness_din(weighting="din", field_type="free")
        >>> print(f"Mean sharpness: {sharpness.data.mean():.2f} acum")

        Notes
        -----
        - Sharpness is measured in acum (acum = 1 when the sound has the
          same sharpness as a 2 kHz narrow-band noise at 60 dB SPL)
        - The calculation uses MoSQITo's implementation of DIN 45692
        - Output sampling rate is fixed at 500 Hz regardless of input rate
        - For multi-channel signals, sharpness is calculated per channel

        References
        ----------
        .. [1] DIN 45692:2009, "Measurement technique for the simulation of the
               auditory sensation of sharpness"
        """
        logger.debug(
            "Setting up sharpness DIN calculation with weighting=%s, "
            "field_type=%s (lazy)",
            weighting,
            field_type,
        )
        result = self.apply_operation(
            "sharpness_din",
            weighting=weighting,
            field_type=field_type,
        )
        return cast(T_Processing, result)

    def sharpness_din_st(
        self: T_Processing,
        weighting: str = "din",
        field_type: str = "free",
    ) -> "NDArrayReal":
        """Calculate steady-state sharpness using DIN 45692 method.

        This method computes the steady-state sharpness of the signal
        according to DIN 45692 standard, which quantifies the perceived
        sharpness of stationary sounds.

        Parameters
        ----------
        weighting : str, default="din"
            Weighting type for sharpness calculation.
        field_type : str, default="free"
            Type of sound field. Options:
            - 'free': Free field (sound from a specific direction)
            - 'diffuse': Diffuse field (sound from all directions)

        Returns
        -------
        NDArrayReal
            Sharpness values in acum, one per channel. Shape: (n_channels,)

        Raises
        ------
        ValueError
            If the signal sampling rate is not supported by the algorithm.

        Examples
        --------
        >>> import wandas as wd
        >>> signal = wd.read_wav("constant_tone.wav")
        >>> sharpness = signal.sharpness_din_st(weighting="din", field_type="free")
        >>> print(f"Steady-state sharpness: {sharpness[0]:.2f} acum")

        Notes
        -----
        - Sharpness is measured in acum (acum = 1 when the sound has the
          same sharpness as a 2 kHz narrow-band noise at 60 dB SPL)
        - The calculation uses MoSQITo's implementation of DIN 45692
        - Output is a single value per channel, suitable for stationary signals
        - For multi-channel signals, sharpness is calculated per channel

        References
        ----------
        .. [1] DIN 45692:2009, "Measurement technique for the simulation of the
               auditory sensation of sharpness"
        """
        from wandas.processing.psychoacoustic import SharpnessDinSt
        from wandas.utils.types import NDArrayReal

        # Create operation instance
        operation = SharpnessDinSt(
            self.sampling_rate, weighting=weighting, field_type=field_type
        )

        # Get data (triggers computation if lazy)
        data = self.data

        # Ensure data is 2D (n_channels, n_samples)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Process the array using the public API and materialize to NumPy
        result = operation.process_array(data).compute()

        # Squeeze to get 1D array (n_channels,)
        sharpness_values: NDArrayReal = result.squeeze()

        # Ensure it's 1D even for single channel
        if sharpness_values.ndim == 0:
            sharpness_values = sharpness_values.reshape(1)

        return sharpness_values
