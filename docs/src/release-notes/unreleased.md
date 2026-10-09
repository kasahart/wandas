# Unreleased changes

See [Wandas 0.8.1](v0.8.1.md) for the preceding audio inspection, physical
frame-center and WDF 0.5 compatibility guidance.

## Sparse STFT compatibility

STFT accepts `allow_sparse=True` for hops exceeding the window length.
Per-frame cepstrum and spectral-envelope transformations preserve these gaps.
Sparse results use WDF 0.6; other results retain their existing WDF version.
New STFT Recipes publish operation version 2 while version 1 remains replayable.
Upgrade readers before loading these new artifacts. ISTFT reconstruction still
requires overlapping windows.

## WAV output destinations

`ChannelFrame.to_wav` and `write_wav` now accept writable binary streams,
including `BytesIO`, with an explicit `format="WAV"`. Caller-owned streams
remain open. Strings and all `os.PathLike` paths continue to infer the format
from the extension; custom path objects are converted with `os.fspath`.
