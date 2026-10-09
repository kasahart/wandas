# Public API and Compatibility Policy / 公開 API・互換性方針

Wandas is a 0.x project, so compatibility changes can still occur.
Wandasは0.xのプロジェクトのため、後方互換性を損なう変更が入る場合があります。

- A change to a stable public API normally emits a runtime deprecation warning
  and keeps the replacement API available during migration.
- Migration guidance is recorded in the applicable release notes.
- Readers reject unsupported WDF or Recipe schemas explicitly; they do not
  guess, silently upgrade, or reinterpret unknown data.
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

- Sparse STFT requires the boolean opt-in `allow_sparse=True`. New STFT
  Recipes use operation version 2; released version 1 remains replayable and
  retains strict hop validation. Readers without version 2 reject new Recipes
  at load time. The Recipe document schema is unchanged.
- Sparse Spectrogram/Cepstrogram results save as WDF 0.6 with explicit sparse
  constructor state. Non-sparse results retain WDF 0.4/0.5. Older readers reject
  0.6 explicitly. Cepstrum, liftering and spectral-envelope conversion preserve
  sparse time axes; ISTFT still requires overlapping windows.
