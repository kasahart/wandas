# Public API and Compatibility Policy / 公開 API・互換性方針

Wandas is a 0.x project, so compatibility changes can still occur.
Wandasは0.xのプロジェクトのため、後方互換性を損なう変更が入る場合があります。

- A change to a stable public API normally emits a runtime deprecation warning
  and keeps the replacement API available during migration.
- Migration guidance is recorded in the applicable release notes.
- Readers reject unsupported WDF or Recipe schemas explicitly; they do not
  guess, silently upgrade, or reinterpret unknown data.
- A release replays the Recipe operation versions written by the previous
  release. Older operation versions may be removed; loading a Recipe that uses
  one fails with an "unregistered operation" error, and the Recipe must be
  saved again with a supported release.
- Signatures, parameters, returns, exceptions, units, and numerical behavior are
  authoritative in the generated [API Reference](../api/index.md) and its
  Python docstrings.
- `BaseFrame.astype(dtype)` is an additive 0.7.1 lazy, immutable numerical API. Version
  1 supports `float32`/`float64` for real or integer Frames and
  `complex64`/`complex128` for complex Frames. It records the normalized dtype in
  lineage and Recipe ID `wandas.frame.astype`; cross-domain and other output dtypes
  are rejected without changing WDF or Recipe schema versions.

安定した公開APIを変更する場合は、原則としてruntime deprecation warningを出し、
移行方法をrelease notesに記載します。未対応のWDF／Recipe schemaは推測せず明示的に失敗します。
Recipeの操作versionは直前のリリースが書いたものまで再生し、それより古いversionは削除する場合があります。
API詳細は生成された[API Reference](../api/index.md)とPython docstringを正本とします。

- `inspect` is additive audio-header preflight. It accepts built-in local audio,
  bytes and seekable binary streams, and creates neither PCM nor a Dask graph.
  CSV, WDF, URL downloads and custom readers are outside that guarantee.
- `frame_center_times` adds physical centers without changing `times`,
  `source_times`, plot, ISTFT or `get_frame_at` semantics. Unknown origin raises
  explicitly; history is never treated as persisted analysis state.
- WDF 0.5 is reserved for Spectrogram/Cepstrogram results with an explicit
  `frame_time_origin`. Other results still save as 0.4. The reader accepts both
  strict schemas; 0.4 has unknown physical centers. Older readers reject 0.5.
  Recipe schemas and operation versions are unchanged. See the
  [inspection and frame-time guide](../how-to/inspect-audio-and-frame-times.md).
- Element-wise NumPy ufuncs on a frame return a lazy frame. Arithmetic ufuncs
  record the operator Recipe operations, whose reverse forms accept an array
  input; other one- and two-input NumPy ufuncs record `wandas.numpy.ufunc`
  version 1 with the ufunc name. Ufuncs outside NumPy cannot be saved.
- Recipe schema 3 writes finite Python floats as plain JSON numbers. The reader
  accepts schemas 2 and 3; the writer emits 3. Older readers reject schema 3.
  Operation versions are unchanged.
- Recipe schema 4 adds a `steps` object naming other Frames of the graph. The
  writer emits it only for a plan with steps; a plan without steps is still
  written as schema 3. The reader accepts schemas 2 to 4. Older readers reject
  schema 4. Operation versions are unchanged.

- Sparse STFT requires the boolean opt-in `allow_sparse=True`. New STFT
  Recipes use operation version 2. Readers without version 2 reject new
  Recipes at load time. The Recipe document schema is unchanged.
- An omitted STFT `hop_length` now defaults to `win_length // 4`; version 2,
  released in 0.9.0, uses the new default. Operation versions are unchanged.
- FFT and cepstrum window the available samples before zero-padding to
  `n_fft`, so short inputs keep their amplitude. New Recipes use operation
  version 3; released version 2 replays its zero-pad-then-window numerics. `SpectralFrame` and `CepstralFrame` gain
  `win_length` (default `n_fft`), which `ifft()` uses to undo the window gain;
  Welch and STFT frames record theirs, and spectral envelopes inherit the
  cepstrum's. IFFT therefore also moves to version 3, and released version 2
  replays the `n_fft`-window gain. WDF keeps the original
  SpectralFrame and CepstralFrame fields when `win_length == n_fft`; a shorter
  window adds `win_length`, which older readers reject explicitly. Frame-frame
  arithmetic on either frame type rejects operands whose `win_length` differs.
- `noct_synthesis` converts the peak-amplitude bins to RMS and divides them by
  the square root of the analysis window's equivalent noise bandwidth before
  MoSQITo sums each band, using the frame's `window` and `win_length`. New
  Recipes use operation version 3; released version 2 replays the unconverted
  numerics. `SpectralFrame` frame-frame
  arithmetic rejects operands whose `window` differs.
- High-pass, low-pass and band-pass filters run as second-order sections.
  New Recipes use operation version 2; released version 1 replays the
  transfer-function `filtfilt` numerics, including their divergence at high
  orders and low cutoffs. Readers without version 2 reject new Recipes at load
  time. The Recipe document schema is unchanged.
- The A/B/C frequency-weighting filters keep the bilinear transform only for
  the low-frequency poles and fit the high-frequency section to the analog
  curve, so they no longer attenuate too much near Nyquist. New
  `a_weighting` Recipes use operation version 2, and `rms_trend` and
  `sound_level` use version 3; released `a_weighting` version 1 and
  `rms_trend`/`sound_level` version 2 replay the bilinear-only filter.
  Readers without the new versions reject new Recipes at load time. The
  Recipe document schema is unchanged.
- `ChannelFrame.trim()` rounds times to the nearest sample (halves up) instead
  of truncating them. New `wandas.frame.time_slice` Recipes use operation
  version 2; released version 1 replays the truncation. Readers without
  version 2 reject new Recipes at load time. The Recipe document schema is
  unchanged.
- Sparse Spectrogram/Cepstrogram results save as WDF 0.6 with explicit sparse
  constructor state. Non-sparse results retain WDF 0.4/0.5. Older readers reject
  0.6 explicitly. Cepstrum, liftering and spectral-envelope conversion preserve
  sparse time axes; ISTFT still requires overlapping windows.
