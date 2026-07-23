# ADR: Immutable Frame state updates

- **Status**: Accepted
- **Date**: 2026-07-21

Frame state has one of four contracts:

| State | Contract | Update path |
| --- | --- | --- |
| Label, user metadata, channel extra | Annotation | `with_label()`, `with_metadata()`, `with_channel_extra()`, or atomic `with_annotations()` |
| Channel labels | Structural Recipe operation | `rename_channels()` |
| Source-time offsets | Analytical Recipe operation | `with_source_time_offset()` |
| Sampling rate | Numerical operation | `ChannelFrame.resampling()` |
| Calibration | Typed physical-domain state | `ChannelFrame.with_calibration()` and `ChannelCalibration.with_*()` |
| Derived-domain sampling rate or calibration reassignment | Unsupported | Return to the source `ChannelFrame` and perform the supported operation there |

Annotation updates return a new Frame while preserving its lazy data, channel IDs,
axes, calibration, runtime lineage, and derived `operation_history`. They are not
Recipe intent: replay applies processing intent to the runtime input annotations.
WDF stores the current annotation values but its display history remains non-executable.

`with_source_time_offset()` normalizes scalar broadcast to a complete per-channel
float list when Recipe intent is captured. Replay preserves that authored analytical
state and rejects a runtime Frame with a different channel count instead of
reinterpreting scalar broadcast against a new shape.

For v0.7.0, direct assignment to Frame annotation or analytical state remains a
compatibility path and emits `DeprecationWarning`, including mutations inside nested
metadata dictionaries and lists. v0.8.0 will make these public views read-only.

All caller-owned mappings, lists, and arrays are copied at the public boundary.

## Validation ownership

Validation is owned by the state value, not by the input route that happens to
write it:

| State | Single owner | Adapters that reuse it |
| --- | --- | --- |
| Frame label | `BaseFrame` label normalizer | constructors, compatibility setter, annotation reconstruction |
| Frame metadata | `BaseFrame` metadata snapshot normalizer | constructors, compatibility setter, individual and atomic annotation methods |
| Channel label and extra | `ChannelMetadata` normalizers | value objects, xarray-backed compatibility views, constructors, `add_channel()`, and `rename_channels()` |
| Channel annotation selector | `BaseFrame` one-channel resolver | individual and atomic channel-extra updates; stable-ID/label collisions are rejected as ambiguous |
| Unit, reference, and factor | `ChannelCalibration` | channel metadata, private xarray writer, immutable calibration update, calibration Recipe decode |
| Sampling rate | `validate_sampling_rate()` | Frame constructors, private writer, compatibility setters, and axis-owning Frame overrides |
| Source-time offset | `BaseFrame` source-offset normalizer | constructors, private writer, compatibility setter, immutable update, Recipe capture and replay |

Private writers only store already validated values and never provide a weaker
input contract. Recipe declarations add an exact persisted-shape decoder around
the same state owner. `RecipePlan.from_dict()` calls that decoder through
`validate_params`, and the handler reuses it before calling the public operation.
Operations with more than one input-kind pattern may additionally declare a
binding-aware parameter validator. Plan validation passes the already selected
pattern to that validator, so constraints such as raw-array-only source offsets
are rejected at load time without changing the persisted Recipe schema.
Consequently malformed rename keys or labels and malformed source-offset lists
are rejected while loading a plan; only runtime-dependent checks such as channel
count remain at apply time. No input boundary stringifies or integer-coerces an
invalid label, selector key, sampling rate, or source-offset value.
