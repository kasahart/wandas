# Public API and Compatibility Policy / 公開 API・互換性方針

Wandas 0.6 keeps the user entry surface small while the project approaches its
1.0 compatibility promise. This page defines stability levels and change records;
it is not an inventory of every public symbol or a second API reference.

## Where API information belongs

- The authoritative public surface is each module's `__all__`, with top-level
  `wandas.__all__` as the main stable entry point.
- API-specific parameters, returns, exceptions, shapes, units, numerical
  contracts, and short examples belong in Google-style Python docstrings and
  the generated [API Reference](../api/index.md).
- Optional extras are summarized in the [README](https://github.com/kasahart/wandas#installation).
- Cross-API acoustic guarantees belong in the [spectral numerical contracts](spectral-numerical-contracts.md).
- WDF details belong in the [I/O contract guide](../contributing/io-contracts.md);
  Recipe details belong in [Recipe Design](pipeline-recipe-design.md) and its
  [task guides](../how-to/pipeline-recipes.md).

## Stability levels

| Level | Meaning |
| --- | --- |
| **Stable** | A documented user-facing API whose behavior is covered by the compatibility policy. |
| **Experimental** | A usable but explicitly changeable surface; changes remain visible and must not silently alter stored data or numerical meaning. |
| **Serialized** | A persisted WDF or Recipe schema whose readers and writers have an explicit compatibility contract. |
| **Internal-only** | An implementation detail not promised to users and not a supported import path. |

Changes to stable and serialized surfaces require tests, documentation, and a
deprecation period. During 0.x, a deprecation warning remains for at least one
feature release before removal. The feature release that first emits the warning
starts the window; the next feature release is the earliest normal removal release.
Patch releases do not consume the window, and the replacement remains available
through removal. Version 1.0 will define the longer support window.

Experimental APIs may change in a feature release without a warning release, but
the release note must identify the experimental surface, describe the migration or
state that there is no replacement, and name the version in which the change takes
effect. Internal-only changes use `not applicable` for deprecation.

## Compatibility exceptions and release records

An exception to the normal window requires a documented security, data-loss,
numerical-correctness, or adapter-retention reason in the tracking issue or PR.
Release notes record the affected surface, classification, deprecation start (or
`none`), migration, and removal/change version. Use the
[`release-notes/template.md`](../release-notes/template.md) for compatibility
changes; ordinary patch releases may state that no such changes occurred.

Serialized readers fail explicitly for unsupported future schemas rather than
guessing or silently upgrading. The format-specific contracts remain with the
WDF and Recipe documentation linked above.

## Related policies

The [Frame and Operation extension guide](../contributing/frame-operation-extensions.md)
defines completion requirements for new operations. Historical design decisions
remain in the [ADR collection](https://github.com/kasahart/wandas/tree/main/docs/design/),
while the implementation and tests remain the source of truth for current behavior.
