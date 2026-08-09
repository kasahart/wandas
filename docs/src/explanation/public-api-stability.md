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
- `BaseFrame.cache()` is an additive 0.7.0 API with no arguments. Its stable
  contract is synchronous local materialization into a new equivalent Frame while
  preserving lineage; cache management, status, release, capacity, scheduler,
  `persist()` aliases, and WDF or Recipe schema fields are not part of the API. A
  raw `np.ma.MaskedArray` compute result raises `ValueError`; mask preservation is
  not part of the contract because it varies across supported xarray versions.

安定した公開APIを変更する場合は、原則としてruntime deprecation warningを出し、
移行方法をrelease notesに記載します。未対応のWDF／Recipe schemaは推測せず明示的に失敗します。
API詳細は生成された[API Reference](../api/index.md)とPython docstringを正本とします。
