# Explanation

These pages explain concepts and guarantees that span more than one API. They
are not a second API reference or a catalog of every implementation detail.

## Concepts and guarantees

- [Scalability Contract](scalability-contract.md) describes the supported flow
  for bounded recordings, lazy graphs, and materialization boundaries.
- [Spectral Numerical Contracts](spectral-numerical-contracts.md) compares
  amplitude, units, normalization, and dB conversion across spectral APIs.

## Where to find other details

- Public classes, functions, methods, parameters, and return values are in the
  generated [API Reference](../api/index.md) and their Python docstrings.
- Current operation and Recipe extension procedures are in the
  [Contributor Guide](../contributing/frame-operation-extensions.md).
- Historical xarray and immutable-state decisions are preserved in the
  [xarray migration ADR](https://github.com/kasahart/wandas/blob/main/docs/design/2026-06-11-xarray-migration-consolidation.md)
  and [immutable state ADR](https://github.com/kasahart/wandas/blob/main/docs/design/2026-07-21-immutable-frame-state-updates.md).
- WDF implementation invariants for maintainers are reachable from the
  [Contributing Overview](../contributing.md), which links to the [I/O contract guide](../contributing/io-contracts.md).
- Operation-author implementation details belong to the
  [Frame and Operation extension guide](../contributing/frame-operation-extensions.md).

Use the [Tutorial](../tutorial/index.md) for a first success and
[How-to guides](../how-to/cepstral-analysis.md) for task-specific procedures.
