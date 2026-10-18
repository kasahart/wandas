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

## Transfer function and coherence from a CSD

`CrossSpectralFrame.transfer_function()` and `CrossSpectralFrame.coherence()`
derive both quantities from an existing complete CSD without recomputing the
spectra, and both are Recipe operations. `ChannelFrame.transfer_function()` and
`ChannelFrame.coherence()` now compute one CSD matrix internally; their values
and signatures are unchanged.

`SpectrogramFrame.csd()` builds the same complete CSD from an existing STFT by
averaging its frames, so one cached spectrogram can feed the spectrogram view,
the CSD, the transfer function and the coherence. `stft()` accepts
`detrend="constant"` or `"linear"` to remove each frame's trend first, so the
result matches `ChannelFrame.csd()` with the same `detrend`; new STFT Recipe
version 2 nodes record it, and version 1 nodes reject it.

## WAV output destinations

`ChannelFrame.to_wav` and `write_wav` now accept writable binary streams,
including `BytesIO`, with an explicit `format="WAV"`. Caller-owned streams
remain open. Strings and all `os.PathLike` paths continue to infer the format
from the extension; custom path objects are converted with `os.fspath`.
