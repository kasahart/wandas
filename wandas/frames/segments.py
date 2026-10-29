"""Fixed-length segments cut from Frames, such as the hits of an impact test.

A segment collection is not a Frame: it holds one Frame per segment, each with
its own lineage, plus the stable event IDs and trigger times needed to inspect,
exclude and average the segments. Numerical work stays in the existing Frame
operations; ensemble averages reuse the Welch cross-spectral operations by
treating every segment as one non-overlapping Welch segment.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

import dask.array as da
import numpy as np
import numpy.typing as npt

from wandas.core.base_frame import BaseFrame
from wandas.frames.channel import ChannelFrame
from wandas.processing.semantic import InputBinding, LineageNode, SemanticOperation, freeze_params, semantic_lineage

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.frames.pairwise import CoherenceFrame, TransferFunctionFrame

F = TypeVar("F", bound=BaseFrame[Any])
SegmentsT = TypeVar("SegmentsT", bound="FrameSegments[Any]")

_NOT_IN_RECIPE = "Segments are tied to positions in one recording and cannot be replayed as a Recipe step"


def _segments_lineage(operation_id: str, items: Sequence[BaseFrame[Any]], event_ids: Sequence[int]) -> LineageNode:
    """Return one lineage node whose inputs are every segment's lineage."""
    bindings = tuple(InputBinding(f"segment_{index}", "frame") for index in range(len(items)))
    operation = SemanticOperation(operation_id, 1, bindings, freeze_params({"event_ids": list(event_ids)}))
    return LineageNode(operation, tuple(item.lineage for item in items), recipe_error=_NOT_IN_RECIPE)


class FrameSegments(Generic[F]):
    """Equal-shaped Frames cut from recordings, one per event.

    Segments are created by :meth:`ChannelFrame.trigger` or
    :meth:`ChannelFrame.segment`, or by :meth:`apply` on existing segments.
    Every segment keeps a stable event ID, its position among the detected
    events, so IDs stay the same after selecting or dropping segments.

    Indexing with an integer returns that segment's Frame. Indexing with a
    slice, a list of positions or a boolean mask returns a new collection of the
    same type. The collection is immutable and keeps every Frame lazy; only
    :meth:`to_numpy` and plotting compute data.

    Args:
        items: Frames of the same type, shape and sampling rate.
        event_ids: Unique integer ID of each segment.
        trigger_times: Event time in seconds of each segment on its recording.

    Raises:
        TypeError: If the items are not Frames of one type.
        ValueError: If shapes, sampling rates or the number of IDs and times
            differ, or IDs repeat.
    """

    def __init__(
        self,
        items: Sequence[F],
        event_ids: Sequence[int] | npt.NDArray[np.integer[Any]],
        trigger_times: Sequence[float] | npt.NDArray[np.floating[Any]],
    ) -> None:
        frames = tuple(items)
        ids = np.asarray(event_ids, dtype=np.int64).reshape(-1)
        times = np.asarray(trigger_times, dtype=float).reshape(-1)
        if not (len(frames) == ids.size == times.size):
            raise ValueError(
                "Segments need one event ID and one trigger time per Frame\n"
                f"  Got: {len(frames)} Frames, {ids.size} IDs, {times.size} times"
            )
        if np.unique(ids).size != ids.size:
            raise ValueError(f"Segment event IDs must be unique\n  Got: {ids.tolist()}")
        if frames:
            first = frames[0]
            if not isinstance(first, BaseFrame):
                raise TypeError(f"Segments must be Frames\n  Got: {type(first).__name__}")
            for position, frame in enumerate(frames[1:], start=1):
                if type(frame) is not type(first):
                    raise TypeError(
                        f"Segment {position} has another Frame type\n"
                        f"  Got: {type(frame).__name__}\n  Expected: {type(first).__name__}"
                    )
                if frame.shape != first.shape or frame.sampling_rate != first.sampling_rate:
                    raise ValueError(
                        f"Segment {position} differs from the first segment\n"
                        f"  Got: shape {frame.shape} at {frame.sampling_rate} Hz\n"
                        f"  Expected: shape {first.shape} at {first.sampling_rate} Hz"
                    )
        ids.flags.writeable = False
        times.flags.writeable = False
        self._items = frames
        self._event_ids = ids
        self._trigger_times = times

    @property
    def event_ids(self) -> npt.NDArray[np.int64]:
        """Stable event ID of each segment, in segment order (read-only)."""
        return self._event_ids

    @property
    def trigger_times(self) -> npt.NDArray[np.float64]:
        """Event time in seconds of each segment on its recording (read-only)."""
        return self._trigger_times

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[F]:
        return iter(self._items)

    @overload
    def __getitem__(self, key: int) -> F: ...

    @overload
    def __getitem__(
        self: SegmentsT, key: slice | Sequence[int] | npt.NDArray[np.integer[Any]] | npt.NDArray[np.bool_]
    ) -> SegmentsT: ...

    def __getitem__(self, key: Any) -> Any:
        """Return one segment's Frame, or a subset of the segments.

        Args:
            key: Integer position for one Frame; a slice, a sequence of integer
                positions or a boolean mask of length ``len(self)`` for a subset.
                Positions count the segments currently held, not event IDs; use
                :meth:`drop` to exclude segments by event ID.

        Raises:
            IndexError: If a position is out of range.
            TypeError: If the key is of another type.
            ValueError: If a mask has the wrong length or positions repeat.
        """
        if isinstance(key, numbers.Integral) and not isinstance(key, (bool, np.bool_)):
            return self._items[int(key)]
        if isinstance(key, slice):
            return self._subset(list(range(len(self)))[key])
        selector = np.asarray(key)
        if selector.dtype == np.bool_:
            if selector.shape != (len(self),):
                raise ValueError(
                    f"Segment mask length does not match the number of segments\n"
                    f"  Got: shape {selector.shape}\n  Expected: ({len(self)},)"
                )
            return self._subset(np.flatnonzero(selector).tolist())
        if selector.ndim == 1 and (selector.size == 0 or np.issubdtype(selector.dtype, np.integer)):
            positions = [int(value) for value in selector]
            normalized = [value + len(self) if value < 0 else value for value in positions]
            if any(not 0 <= value < len(self) for value in normalized):
                raise IndexError(f"Segment position out of range\n  Got: {positions} for {len(self)} segments")
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"Segment positions must not repeat\n  Got: {positions}")
            return self._subset(normalized)
        raise TypeError(
            "Segments are indexed by an integer, a slice, integer positions or a boolean mask\n"
            f"  Got: {type(key).__name__}"
        )

    def drop(self: SegmentsT, event_ids: int | Sequence[int]) -> SegmentsT:
        """Return the segments without the given event IDs.

        Args:
            event_ids: Event ID or IDs to exclude, as listed in
                :attr:`event_ids` and shown by ``plot()``.

        Raises:
            KeyError: If an ID is not among the segments.
            ValueError: If an ID repeats.
        """
        dropped = (
            [int(event_ids)]
            if isinstance(event_ids, numbers.Integral)
            else [int(i) for i in cast(Sequence[int], event_ids)]
        )
        if len(set(dropped)) != len(dropped):
            raise ValueError(f"Event IDs to drop must not repeat\n  Got: {dropped}")
        missing = sorted(set(dropped) - set(self._event_ids.tolist()))
        if missing:
            raise KeyError(f"Unknown segment event IDs: {missing}; available: {self._event_ids.tolist()}")
        keep = ~np.isin(self._event_ids, dropped)
        return self._subset(np.flatnonzero(keep).tolist())

    def apply(self, func: Callable[[F], BaseFrame[Any]]) -> FrameSegments[Any]:
        """Apply a function to every segment and keep the results as segments.

        Use this to process each segment separately, for example
        ``hits.apply(lambda hit: hit.high_pass_filter(5))``. Filtering each
        segment differs from filtering the recording before cutting it, because
        each segment starts its own filter state; choose the order that matches
        the analysis. Event IDs and trigger times are kept, and each result
        records its own lineage.

        Args:
            func: Function that takes one segment's Frame and returns a Frame.
                Every result must have the same type and shape.

        Returns:
            :class:`ChannelFrameSegments` when the results are ChannelFrames,
                otherwise :class:`FrameSegments`.
        """
        results = [func(item) for item in self._items]
        if results and isinstance(results[0], ChannelFrame):
            pre = self.pre if isinstance(self, ChannelFrameSegments) else 0.0
            trigger_channel = self._trigger_index if isinstance(self, ChannelFrameSegments) else None
            if self._items and results[0].n_channels != self._items[0].n_channels:
                trigger_channel = None
            return ChannelFrameSegments(
                cast(list[ChannelFrame], results),
                self._event_ids,
                self._trigger_times,
                pre=pre,
                trigger_channel=trigger_channel,
            )
        return FrameSegments(results, self._event_ids, self._trigger_times)

    def to_numpy(self) -> npt.NDArray[Any]:
        """Compute every segment and stack them as ``(segment, channel, ...)``.

        The channel axis is kept even for single-channel segments, so a
        ChannelFrame collection gives ``(segment, channel, sample)``. Values are
        calibrated, as in each Frame's ``data``. This computes the data.

        Raises:
            ValueError: If there are no segments.
        """
        if not self._items:
            raise ValueError("There are no segments to convert")
        stacked = da.stack([item._effective_data for item in self._items])
        return np.asarray(stacked.compute())

    def _subset(self: SegmentsT, positions: Sequence[int]) -> SegmentsT:
        """Return the segments at ``positions`` with the same collection state."""
        index = np.asarray(positions, dtype=np.int64)
        return self._rebuild([self._items[i] for i in positions], self._event_ids[index], self._trigger_times[index])

    def _rebuild(
        self: SegmentsT, items: Sequence[Any], event_ids: npt.NDArray[np.int64], trigger_times: npt.NDArray[np.float64]
    ) -> SegmentsT:
        return cast(SegmentsT, FrameSegments(items, event_ids, trigger_times))

    def __repr__(self) -> str:
        frame_type = type(self._items[0]).__name__ if self._items else "Frame"
        shape = self._items[0].shape if self._items else ()
        return f"{type(self).__name__}(n_segments={len(self)}, frame={frame_type}, shape={shape})"


class ChannelFrameSegments(FrameSegments[ChannelFrame]):
    """Time segments of all channels, one per event, such as the hits of an impact test.

    Every segment is a ChannelFrame of the same channels and length. Index it
    to look at one hit, select or :meth:`drop` hits to exclude bad ones, and
    average the remaining hits with :meth:`transfer_function`,
    :meth:`coherence` and :meth:`average`.

    Detection by :meth:`ChannelFrame.trigger` computes the trigger channel;
    segments, selections and averages stay lazy until data is accessed.

    Args:
        items: ChannelFrames with the same channels, units, calibration, length
            and sampling rate.
        event_ids: Unique integer ID of each segment.
        trigger_times: Event time in seconds of each segment on its recording.
        pre: Time in seconds from the start of each segment to its event.
        trigger_channel: Index of the channel the events were detected on, or
            ``None`` when the times came from elsewhere.

    Raises:
        ValueError: If the segments differ in channels, units or calibration,
            or ``pre`` is negative or not finite.
    """

    def __init__(
        self,
        items: Sequence[ChannelFrame],
        event_ids: Sequence[int] | npt.NDArray[np.integer[Any]],
        trigger_times: Sequence[float] | npt.NDArray[np.floating[Any]],
        pre: float = 0.0,
        trigger_channel: int | None = None,
    ) -> None:
        super().__init__(items, event_ids, trigger_times)
        if not (np.isfinite(pre) and pre >= 0):
            raise ValueError(f"Pre-trigger time must be finite and non-negative\n  Got: {pre} s")
        if self._items:
            first = self._items[0]
            if not isinstance(first, ChannelFrame):
                raise TypeError(f"ChannelFrameSegments need ChannelFrames\n  Got: {type(first).__name__}")
            reference = _channel_signature(first)
            for position, frame in enumerate(self._items[1:], start=1):
                if _channel_signature(frame) != reference:
                    raise ValueError(
                        f"Segment {position} has other channels than the first segment\n"
                        f"  Got: {_channel_signature(frame)}\n  Expected: {reference}"
                    )
            if trigger_channel is not None and not 0 <= trigger_channel < first.n_channels:
                raise ValueError(f"Trigger channel {trigger_channel} is out of range for {first.n_channels} channels")
        self._pre = float(pre)
        self._trigger_index = trigger_channel

    @property
    def pre(self) -> float:
        """Time in seconds from the start of each segment to its event."""
        return self._pre

    @property
    def trigger_channel(self) -> str | None:
        """Label of the channel the events were detected on, if known."""
        if self._trigger_index is None or not self._items:
            return None
        return self._items[0].labels[self._trigger_index]

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the segments in Hz."""
        return self._first().sampling_rate

    @property
    def labels(self) -> list[str]:
        """Channel labels of the segments."""
        return self._first().labels

    @property
    def n_samples(self) -> int:
        """Samples per segment."""
        return self._first().n_samples

    @property
    def duration(self) -> float:
        """Segment length in seconds."""
        return self.n_samples / self.sampling_rate

    def transfer_function(
        self,
        input: int | str | None = None,  # noqa: A002
        output: int | str | Sequence[int | str] | None = None,
        n_fft: int | None = None,
        window: str = "boxcar",
        detrend: Literal["constant", "linear"] | None = None,
    ) -> TransferFunctionFrame:
        """Average the segments into one frequency response function (H1).

        The estimate is ``H1 = sum(G_xy) / sum(G_xx)`` over the segments,
        where ``x`` is the input channel and ``y`` an output channel: every
        segment is one spectral average with equal weight, so this is not the
        mean of per-segment transfer functions. Each segment is transformed
        whole, without overlap. The defaults follow impact testing (rectangular
        window, no detrending), unlike :meth:`ChannelFrame.transfer_function`,
        which averages overlapping Hann windows within one recording. Bins where
        the input has no energy are ``NaN``.

        Args:
            input: Excitation channel (index or label). ``None`` uses the
                trigger channel.
            output: Response channel or channels. ``None`` uses every channel
                except the input.
            n_fft: FFT length. ``None`` uses the segment length; a longer length
                zero-pads each segment.
            window: SciPy window name applied to each segment. Default
                ``"boxcar"`` (rectangular).
            detrend: ``None`` (default), ``"constant"`` or ``"linear"``.

        Returns:
            TransferFunctionFrame with one output/input pair per output channel.

        Raises:
            ValueError: If there are no segments, the channels are invalid or
                ``n_fft`` is shorter than a segment.
        """

        joined, n_channels, n_fft = self._joined_pairs(input, output, n_fft)
        result = joined.transfer_function(
            n_fft=n_fft,
            hop_length=self.n_samples,
            win_length=self.n_samples,
            window=window,
            detrend=cast(str, detrend),
        )
        return result.select_pairs([row * n_channels for row in range(1, n_channels)])

    def coherence(
        self,
        input: int | str | None = None,  # noqa: A002
        output: int | str | Sequence[int | str] | None = None,
        n_fft: int | None = None,
        window: str = "boxcar",
        detrend: Literal["constant", "linear"] | None = None,
    ) -> CoherenceFrame:
        """Coherence between the input and each output over the segments.

        Computed from the same averaged cross and auto spectra as
        :meth:`transfer_function`, with the same arguments. Values near 1 mean
        the output follows the input linearly in every segment. At least two
        segments are needed, because with one segment the coherence is always
        1.

        Args:
            input: Excitation channel (index or label). ``None`` uses the
                trigger channel.
            output: Response channel or channels. ``None`` uses every channel
                except the input.
            n_fft: FFT length. ``None`` uses the segment length.
            window: SciPy window name applied to each segment. Default
                ``"boxcar"``.
            detrend: ``None`` (default), ``"constant"`` or ``"linear"``.

        Returns:
            CoherenceFrame with one output/input pair per output channel.

        Raises:
            ValueError: If there are fewer than two segments, or as in
                :meth:`transfer_function`.
        """

        if len(self) < 2:
            raise ValueError(
                "Coherence needs at least two segments\n"
                f"  Got: {len(self)}\n"
                "  With one segment the coherence is 1 at every frequency."
            )
        joined, n_channels, n_fft = self._joined_pairs(input, output, n_fft)
        result = joined.coherence(
            n_fft=n_fft,
            hop_length=self.n_samples,
            win_length=self.n_samples,
            window=window,
            detrend=cast(str, detrend),
        )
        return result.select_pairs([row * n_channels for row in range(1, n_channels)])

    def average(self) -> ChannelFrame:
        """Average the segments sample by sample, aligned on their events.

        This is the time-domain mean of the waveforms (synchronous averaging),
        not an average spectrum. The result keeps the channels of the segments
        and is lazy.

        Raises:
            ValueError: If there are no segments.
        """
        first = self._first()
        lineage = _segments_lineage("wandas.segments.average", self._items, self._event_ids.tolist())
        data = da.stack([item._data for item in self._items]).mean(axis=0)
        with semantic_lineage(lineage):
            return first._create_new_instance(
                data=data, label=f"average of {len(self)} segments", lineage=lineage, previous=first
            )

    def plot(self, channel: int | str | None = None, ax: Axes | None = None) -> Axes | Iterator[Axes]:
        """Overlay every segment, one axes per channel, labelled by event ID.

        The time axis is in seconds from the event, so the trigger is at 0 and
        the pre-trigger part is negative. This computes the data.

        Args:
            channel: Channel to plot (index or label). ``None`` plots every
                channel on its own axes.
            ax: Axes to draw on; only allowed with one channel.

        Returns:
            The Axes for one channel, otherwise an iterator over the Axes.

        Raises:
            ValueError: If there are no segments, or ``ax`` is given for
                several channels.
        """
        import matplotlib.pyplot as plt

        first = self._first()
        indices = list(range(first.n_channels)) if channel is None else first._channel_indices(channel)
        if ax is not None and len(indices) != 1:
            raise ValueError("ax can only be given when plotting one channel")
        if ax is None:
            _, created = plt.subplots(len(indices), 1, sharex=True, squeeze=False)
            axes = list(created[:, 0])
        else:
            axes = [ax]
        values = self.to_numpy()
        time = np.arange(self.n_samples) / self.sampling_rate - self.pre
        for target, index in zip(axes, indices, strict=True):
            meta = first.channels[index]
            for event_id, segment in zip(self._event_ids, values, strict=True):
                target.plot(time, segment[index], label=f"#{event_id}")
            target.set_ylabel(f"{meta.label} [{meta.unit}]" if meta.unit else meta.label)
            target.legend(title="Event", fontsize="small", ncol=max(1, len(self) // 10))
        axes[-1].set_xlabel("Time from event [s]")
        return axes[0] if len(axes) == 1 else iter(axes)

    def _first(self) -> ChannelFrame:
        if not self._items:
            raise ValueError("There are no segments")
        return self._items[0]

    def _joined_pairs(
        self,
        input: int | str | None,  # noqa: A002
        output: int | str | Sequence[int | str] | None,
        n_fft: int | None,
    ) -> tuple[ChannelFrame, int, int]:
        """Join the input and output channels of every segment end to end.

        With the window and hop both equal to the segment length, Welch's
        segments are exactly the event segments, so the existing cross-spectral
        operations compute the ensemble average.
        """
        first = self._first()
        if input is None:
            if self._trigger_index is None:
                raise ValueError(
                    "Give the input channel: these segments were not cut by a trigger, so there is no default input"
                )
            input = self._trigger_index  # noqa: A001
        (input_index,) = first._channel_indices(input)
        if output is None:
            output_indices = [index for index in range(first.n_channels) if index != input_index]
        elif isinstance(output, (str, numbers.Integral)):
            output_indices = first._channel_indices(output)
        else:
            output_indices = [
                index for item in cast(Sequence[int | str], output) for index in first._channel_indices(item)
            ]
        if not output_indices:
            raise ValueError("There is no output channel besides the input channel")
        if input_index in output_indices or len(set(output_indices)) != len(output_indices):
            raise ValueError(
                "Output channels must be distinct and differ from the input channel\n"
                f"  Got: input {first.labels[input_index]!r}, outputs {[first.labels[i] for i in output_indices]}"
            )
        resolved_n_fft = self.n_samples if n_fft is None else n_fft
        if resolved_n_fft < self.n_samples:
            raise ValueError(
                "n_fft must not be shorter than a segment\n"
                f"  Got: {resolved_n_fft}\n  Expected: at least {self.n_samples} (the segment length)"
            )
        lineage = _segments_lineage("wandas.segments.join", self._items, self._event_ids.tolist())
        data = da.concatenate([item._data for item in self._items], axis=-1)
        with semantic_lineage(lineage):
            joined = first._create_new_instance(
                data=data, label=f"{len(self)} segments of {first.label}", lineage=lineage, previous=first
            )
        channels = [input_index, *output_indices]
        return joined[channels], len(channels), resolved_n_fft

    def _rebuild(
        self, items: Sequence[Any], event_ids: npt.NDArray[np.int64], trigger_times: npt.NDArray[np.float64]
    ) -> ChannelFrameSegments:
        return ChannelFrameSegments(items, event_ids, trigger_times, pre=self._pre, trigger_channel=self._trigger_index)

    def __repr__(self) -> str:
        if not self._items:
            return "ChannelFrameSegments(n_segments=0)"
        return (
            f"ChannelFrameSegments(n_segments={len(self)}, duration={self.duration:g} s, pre={self.pre:g} s, "
            f"sampling_rate={self.sampling_rate:g} Hz, channels={self.labels}, "
            f"trigger_channel={self.trigger_channel!r})"
        )


def _channel_signature(frame: ChannelFrame) -> list[tuple[str, str, float]]:
    """Return the label, unit and calibration factor of every channel."""
    return [(meta.label, meta.unit, meta.calibration.factor) for meta in frame.channels]
