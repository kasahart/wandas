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
