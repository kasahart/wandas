# ADR: Immutable Frame state updates

- **Status**: Accepted
- **Date**: 2026-07-21

Frame state has one of four contracts:

| State | Contract | Update path |
| --- | --- | --- |
| Label, user metadata, channel extra | Annotation | `with_label()`, `with_metadata()`, `with_channel_extra()` |
| Channel labels | Structural Recipe operation | `rename_channels()` |
| Source-time offsets | Analytical Recipe operation | `with_source_time_offset()` |
| Sampling rate | Numerical operation | `ChannelFrame.resampling()` |
| Calibration | Typed physical-domain state | `ChannelFrame.with_calibration()` and `ChannelCalibration.with_*()` |
| Derived-domain sampling rate or calibration reassignment | Unsupported | Return to the source `ChannelFrame` and perform the supported operation there |

Annotation updates return a new Frame while preserving its lazy data, channel IDs,
axes, calibration, runtime lineage, and derived `operation_history`. They are not
Recipe intent: replay applies processing intent to the runtime input annotations.
WDF stores the current annotation values but its display history remains non-executable.

`with_source_time_offset()` stores invocation intent in Recipe state. A scalar
remains scalar and broadcasts to the runtime Frame's channel count during replay.
An explicit vector remains a vector and is rejected when its length differs from
the runtime channel count.

Direct assignment to Frame state and channel metadata views is unsupported.
Metadata, channel-extra, and source-offset getters return owned snapshots, so
mutating a returned dictionary, list, or array cannot change the Frame.

All caller-owned mappings, lists, and arrays are copied at the public boundary.

## Validation ownership

Validation is owned by the state value, not by the input route that happens to
write it:

| State | Single owner | Adapters that reuse it |
| --- | --- | --- |
| Frame label | `BaseFrame` label normalizer | constructors and annotation reconstruction |
| Frame metadata | `BaseFrame` metadata snapshot normalizer | constructors and `with_metadata()` |
| Channel label and extra | `ChannelMetadata` normalizers | value objects, constructors, `with_channel_extra()`, `add_channel()`, and `rename_channels()` |
| Channel annotation selector | `BaseFrame` one-channel resolver | name/index channel-extra updates; stable-ID lookup uses `frame.channels.by_id()` |
| Unit, reference, and factor | `ChannelCalibration` | channel metadata, private xarray writer, immutable calibration update, calibration Recipe decode |
| Sampling rate | `validate_sampling_rate()` | Frame constructors, private writer, and axis-owning Frame overrides |
| Source-time offset | `BaseFrame` source-offset normalizer | constructors, private writer, immutable update, Recipe capture and replay |

Private writers only store already validated values and never provide a weaker
input contract. Recipe declarations add an exact persisted-shape decoder around
the same state owner. `RecipePlan.from_dict()` calls that decoder through
`validate_params`, and the handler reuses it before calling the public operation.
Consequently malformed rename keys or labels and malformed source-offset lists
are rejected while loading a plan; only runtime-dependent checks such as channel
count remain at apply time. No input boundary stringifies or integer-coerces an
invalid label, selector key, sampling rate, or source-offset value.
