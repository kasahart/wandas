# ADR: Xarray Dimension Semantics Phase 2

- **Status**: Proposed
- **Date**: 2026-06-10
- **Context**: Phase 2 of the xarray migration, after introducing xarray as an internal data container.

## Context

Wandas now stores frame data in an internal `xarray.DataArray`, while operation execution,
metadata ownership, and I/O remain unchanged. Phase 2 uses xarray only for named dimensions
and the conservative `channel` coordinate.

The goal is to reduce Wandas-specific channel-axis bookkeeping without introducing
xarray-native execution, broad coordinate generation, or new input-shape semantics.

## Decision

Target frames declare semantic dimension suffixes:

```text
ChannelFrame:     ("channel", "time")
SpectralFrame:    ("channel", "frequency")
SpectrogramFrame: ("channel", "frequency", "time")
NOctFrame:        ("channel", "band")
```

`BaseFrame` applies these semantic dims only when the normalized data rank exactly matches
the suffix length. Higher-rank legacy inputs keep neutral names such as `dim_0`, `dim_1`,
and `dim_2`.

This intentionally avoids leading-dimension semantics in Phase 2. Existing channel selection
and chunking still assume channel-like data is axis 0, so declaring a nonzero `channel` dim
would create inconsistent behavior.

## Channel Coordinate

`BaseFrame` centralizes channel coordinate creation. A `channel` coord is attached only when:

- the frame has a declared `channel` dimension
- channel metadata label count matches the xarray channel size

If the counts differ, Wandas keeps `channel_metadata` unchanged and omits the xarray coord.
`unit`, `ref`, and other channel metadata remain Wandas-owned in this phase.

## Non-Goals

Phase 2 does not add:

- xarray-native operation dispatch
- `time`, `frequency`, `band`, or `bark` coordinates
- `from_xarray`
- NetCDF or Zarr I/O
- attrs-backed metadata
- accepted input-shape expansion

## Consequences

Positive:

- `ChannelFrame` no longer needs frame-specific xarray dim/coord refresh hooks.
- `n_channels` can prefer `self._xr.sizes["channel"]` for semantic target frames.
- 3D `NOctFrame` compatibility is preserved without pretending the channel axis is known.

Risks:

- `_channel_axis` remains as a fallback for neutral-dim and legacy frames until more frame
  layouts are migrated.
- Dense domain coordinates are still computed through existing Wandas properties until a
  later phase.

## Validation

The phase is validated by:

- frame dim tests for ChannelFrame, SpectralFrame, SpectrogramFrame, and NOctFrame
- channel coord creation and mismatch tests
- regression tests proving non-goals: no time/frequency/band coords and no input-shape expansion
- full repository checks: `ruff`, `ty`, and `pytest`
