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
- `BaseFrame.astype()` is an additive 0.7.0 lazy, immutable numerical API. Version
  1 supports `float32`/`float64` for real or integer Frames and
  `complex64`/`complex128` for complex Frames. It records the normalized dtype in
  lineage and Recipe ID `wandas.frame.astype`; cross-domain and other output dtypes
  are rejected without changing WDF or Recipe schema versions.

安定した公開APIを変更する場合は、原則としてruntime deprecation warningを出し、
移行方法をrelease notesに記載します。未対応のWDF／Recipe schemaは推測せず明示的に失敗します。
API詳細は生成された[API Reference](../api/index.md)とPython docstringを正本とします。
