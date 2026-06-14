# Public API Design for v0.3

## Context

The current top-level `wandas` API exposes convenience constructors such as
`read_wav`, `read_csv`, `from_numpy`, `from_ndarray`, `generate_sin`, and
`from_folder`, but the exported names do not describe the stable public surface
clearly. In particular, `ChannelFrame` is importable as `wandas.ChannelFrame`
but is missing from `wandas.__all__`.

The documentation also describes the frames module as containing
`ChannelFrame`, `SpectralFrame`, `SpectrogramFrame`, and `NOctFrame`, while
`wandas.frames.__all__` currently exports only `ChannelFrame` and
`RoughnessFrame`. This mismatch makes the public API harder to discover and
harder to explain.

## Stable Public API

For v0.3, the primary public API is:

```python
import wandas as wd

wd.read(...)
wd.read_wav(...)
wd.read_csv(...)
wd.load(...)
wd.from_numpy(...)
wd.from_folder(...)

wd.ChannelFrame
wd.SpectralFrame
wd.SpectrogramFrame
wd.NOctFrame
wd.ChannelFrameDataset
```

`from_ndarray` remains available for backward compatibility, but it stays
deprecated and is not presented as a primary API in the README or public API
overview. `generate_sin` remains available as an existing helper, but it is not
part of the primary v0.3 API list.

## `read` and `load` Responsibilities

`wd.read()` is the standard top-level entry point for creating a
`ChannelFrame` from external source data handled by `ChannelFrame.from_file()`,
including WAV, CSV, supported audio files, URLs, bytes, and file-like objects.

`wd.load()` is the standard top-level entry point for loading Wandas native WDF
files. It delegates to the existing WDF loader and returns a `ChannelFrame`.

WDF is intentionally not handled by `wd.read()`. If a caller passes a `.wdf`
path to `wd.read()`, Wandas raises a clear `ValueError` that points the
caller to `wd.load()` instead of falling through to a generic unsupported file
reader error.

Example error shape:

```text
WDF files are loaded with wd.load(), not wd.read()
  Path: example.wdf
  Use: wd.load("example.wdf")
```

`ChannelFrame.from_file()` remains the class-level implementation for external
source data. `wd.read()` is the user-facing top-level wrapper over that
behavior. `ChannelFrame.load()` and `wd.load()` remain responsible for WDF.

## Exports

`wandas.__all__` includes the primary stable names and retained
compatibility helpers:

- `ChannelFrame`
- `SpectralFrame`
- `SpectrogramFrame`
- `NOctFrame`
- `ChannelFrameDataset`
- `read`
- `read_wav`
- `read_csv`
- `load`
- `from_numpy`
- `from_folder`
- `from_ndarray`
- `generate_sin`

`wandas.frames.__all__` matches the documented frame classes:

- `ChannelFrame`
- `SpectralFrame`
- `SpectrogramFrame`
- `NOctFrame`
- `RoughnessFrame`

## Documentation

The README Quick Start uses `wd.read("audio.wav")`, because this is
the easiest path to explain as stable public API. The README can still
mention `read_wav` and `read_csv` as explicit-format convenience functions, but
the first path is `read`.

The API docs describe:

- `wd.read()` for external source data.
- `wd.load()` for WDF.
- `wd.read_wav()` and `wd.read_csv()` as explicit-format convenience functions.
- `from_ndarray()` as deprecated in favor of `from_numpy()`.
- The same frame classes that `wandas.frames.__all__` exports.

Docs, README, and `__all__` use the same names for the same concepts.

## Testing

Add or update tests to cover:

- `wandas.__all__` contains the stable public names.
- `wandas.frames.__all__` exports documented frame classes.
- `wd.ChannelFrame`, `wd.SpectralFrame`, `wd.SpectrogramFrame`,
  `wd.NOctFrame`, and `wd.ChannelFrameDataset` resolve to the expected classes.
- `wd.read()` reads WAV and CSV through the existing `ChannelFrame.from_file()`
  behavior.
- `wd.read()` rejects `.wdf` paths with an error that mentions `wd.load()`.
- `wd.load()` loads WDF files through the existing WDF round-trip path.
- Existing compatibility behavior for `from_ndarray` remains deprecated but
  functional.

## Acceptance Criteria

The README Quick Start API must be directly explainable as stable public API.
After this change, the top-level API, frames module exports, README, and docs
must agree on the v0.3 public names.
