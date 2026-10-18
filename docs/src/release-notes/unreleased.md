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

## STFT default hop follows the window length

`ChannelFrame.stft()` and `ChannelFrameDataset.stft()` now default an omitted
`hop_length` to `win_length // 4`, as `welch()`, `csd()`, `coherence()`,
`transfer_function()` and the librosa/SciPy convention already do. They
previously used `n_fft // 4`.

This changes results for calls that pass a `win_length` smaller than `n_fft`
without `hop_length`. For example, `stft(n_fft=2048, win_length=512)` now uses
a hop of 128 instead of 512. The old hop equalled the window, so a Hann STFT had
no overlap and its ISTFT could not reconstruct the signal. A call that relied on
the old default and used a window shorter than `n_fft // 4` raised an error; it
now succeeds. Calls without `win_length`, or with `win_length=n_fft`, and calls
with an explicit `hop_length` are unchanged. To keep the old frames, pass
`hop_length=n_fft // 4`.

Saved spectrograms and WDF files store the resolved hop and are unaffected.
STFT Recipes saved by Wandas 0.8.1 (operation version 1) keep the `n_fft // 4`
default on replay. Recipes keep an omitted hop omitted, so version 2 Recipes
created from an unreleased development build replay with the new default.

## Transfer function and coherence from a CSD

`CrossSpectralFrame.transfer_function()` and `CrossSpectralFrame.coherence()`
derive both quantities from an existing complete CSD without recomputing the
spectra, and both are Recipe operations. `ChannelFrame.transfer_function()` and
`ChannelFrame.coherence()` now compute one CSD matrix internally; their values
and signatures are unchanged.

`SpectrogramFrame.csd()` builds the same complete CSD from an existing STFT by
averaging its frames, so one cached spectrogram can feed the spectrogram view,
the CSD, the transfer function and the coherence. Frames are not detrended.

## WAV output destinations

`ChannelFrame.to_wav` and `write_wav` now accept writable binary streams,
including `BytesIO`, with an explicit `format="WAV"`. Caller-owned streams
remain open. Strings and all `os.PathLike` paths continue to infer the format
from the extension; custom path objects are converted with `os.fspath`.
