# Source Time Offset Design

## Purpose

Wandas should preserve signal context across time-domain operations without breaking the existing local time API. Today `ChannelFrame.time` is always derived as `np.arange(n_samples) / sampling_rate`, so a frame created by `x.trim(10.0, 12.0)` cannot directly say that it represents seconds 10.0 through 12.0 of the source signal.

This design adds first-class source-time context using a lightweight offset model:

- `time` and `times` remain local, zero-based axes for backward compatibility.
- `source_time_offset` records where local time `0.0` sits in the source signal.
- `source_time` and `source_times` expose source-relative axes by adding the offset to the local axis.
- `source_time_range` exposes the source-relative time span of a frame.

The accepted behavior is:

```python
x = wd.read_wav("recording.wav")
y = x.trim(10.0, 12.0)

y.time[0]        # 0.0
y.source_time[0] # 10.0
```

## Public API

All frame types gain:

```python
frame.source_time_offset: float
frame.source_time_range: tuple[float, float]
```

Frames with a local time axis gain a source-time array:

```python
channel_frame.time          # existing local time axis
channel_frame.source_time   # channel_frame.time + source_time_offset

roughness_frame.time        # existing local time axis
roughness_frame.source_time # roughness_frame.time + source_time_offset

spectrogram_frame.times         # existing local spectrogram frame times
spectrogram_frame.source_times  # spectrogram_frame.times + source_time_offset
```

`SpectralFrame` and `NOctFrame` keep frequency as their main axis. They preserve `source_time_offset`. Their `source_time_range` is derived from the previous time-domain frame when available, because offset alone is not enough to infer a duration from frequency-bin data.

`to_dataframe()` stays backward compatible. For `ChannelFrame`, the index remains local `time`. Future Analyzer code should use `source_time` or `source_times` when it needs source-relative alignment.

## Internal Model

`BaseFrame.__init__` accepts:

```python
source_time_offset: float = 0.0
```

`BaseFrame._create_new_instance()` forwards the current offset unless a caller overrides it. Every concrete frame constructor that calls `BaseFrame.__init__` accepts and forwards the same argument.

The offset is not stored in `metadata`. It is a first-class frame attribute so users and future analyzers can rely on it without parsing mutable metadata.

The offset is measured in seconds and is source-relative, not wall-clock absolute time.

## Operation Rules

`trim(start, end)` keeps `start` and `end` as local-time arguments. The result offset is:

```python
result.source_time_offset = input.source_time_offset + start
```

This means:

```python
x.trim(10.0, 12.0).trim(0.5, 1.0)
```

represents source seconds 10.5 through 11.0.

`resampling(target_sr)` preserves `source_time_offset`. The output `sampling_rate`, sample count, `duration`, and local `time` must agree with the resampled data shape.

`rms_trend(frame_length, hop_length)` preserves `source_time_offset`. Its local `time` remains based on the output sampling rate created by the operation.

`stft(...)` preserves `source_time_offset`. `SpectrogramFrame.times` remains local STFT frame time, and `SpectrogramFrame.source_times` is `times + source_time_offset`.

Channel selection and channel reduction operations preserve `source_time_offset`.

Frequency-domain transforms preserve `source_time_offset` and keep frequency as their primary data axis. When the output has no time axis, `source_time_range` should delegate to the previous frame source range if that previous frame is available.

## Source Time Range

For frames with a local time axis and known duration:

```python
source_start = source_time_offset
source_end = source_time_offset + duration
```

For `ChannelFrame`, `duration` is `n_samples / sampling_rate`.

For time-frame outputs such as `SpectrogramFrame` and roughness-like frames, `source_time_range` should represent the source span covered by the frame's local duration. The implementation should use the frame's existing local time semantics rather than changing those semantics. For example, `SpectrogramFrame.times` remains based on `hop_length / sampling_rate`.

For frequency-only frames such as `SpectralFrame` and `NOctFrame`, `source_time_range` should return the previous frame source range when `previous` exists. If there is no previous frame and no time-axis duration can be inferred from the frame itself, the range should fall back to a zero-length range at `source_time_offset` instead of inventing a duration from frequency-bin count.

Empty frames should report a valid zero-length range:

```python
(source_time_offset, source_time_offset)
```

## Error Handling

`source_time_offset` must be a finite numeric value. Constructors should reject `NaN`, infinity, and non-numeric values with `ValueError` or `TypeError`.

Operations must not silently create a negative sample count or a source range whose end is earlier than its start. Existing trim validation remains local-time validation.

## Tests

Add focused tests for:

- `ChannelFrame.trim(10, 12)` keeps `time[0] == 0.0` and sets `source_time[0] == 10.0`.
- Chained trim `x.trim(10, 12).trim(0.5, 1.0)` reports source range 10.5 through 11.0.
- `resampling()` preserves offset and keeps `duration == n_samples / sampling_rate`.
- `stft()` preserves offset and reports `source_times == times + input.source_time_offset`.
- `rms_trend()` preserves offset and keeps its local time axis consistent with output sampling rate.
- `to_dataframe()` keeps local `time` as the index for backward compatibility.
- Constructor validation rejects invalid offsets.

## Non-Goals

This design does not change `time` or `times` to source-relative axes.

This design does not add wall-clock timestamps or calendar time.

This design does not make `trim(start, end)` accept source-time arguments. Trim arguments remain local time.

This design does not require storing full time arrays on frames.
