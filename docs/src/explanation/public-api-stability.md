# Public API and schema stability / 公開 API・schema 安定性

Wandas 0.6 keeps the user entry surface small and classifies the broader library before
the 1.0 compatibility promise.

## Stable user surface / 安定した user surface

- Top level: `read`, `from_numpy`, `from_folder`, `load`, `supported_formats`,
  `generate_sin`.
- Built-in Frame types and their primary workflow: immutable typed transforms,
  metadata/channel views, `frame.data` as the canonical NumPy-value boundary,
  `to_numpy()` and NumPy's array protocol as equivalent interoperability APIs,
  `plot`, `describe`, and `BaseFrame.save`.
- `RecipePlan.from_frame`, `apply`, `to_dict`, `from_dict`, `save`, and `load`.
- WDF 0.4 typed round-trip and Recipe schema 2 strict JSON.

Changes to this surface require tests, documentation, and a deprecation period. During
0.x, a deprecation warning remains for at least one feature release before removal.
1.0 will define the longer support window.

The feature release that first emits the warning is the start of the support window;
the next feature release is the earliest normal removal release. Patch releases do
not consume that window. The replacement must be documented when the warning starts
and remain available through removal.

## Experimental surface / 実験的 surface

- Recipe extension registries/decorators used to declare third-party operations.
- sklearn adapters in `wandas.pipeline.sklearn`.
- Internal xarray/Dask storage helpers and private attributes such as `_xr` and
  `_data`.

Experimental APIs may change in a feature release, but changes must still be explicit
and must not silently alter stored data or numerical meaning.

Experimental removal does not require a warning release. Its release note must still
identify the surface as experimental, describe the migration or state that there is
no replacement, and name the version in which the change takes effect.

## Optional-domain extensions / optional 領域

Effects, psychoacoustic metrics, ML tensor conversion, interactive marimo/IPython
display, and WDF/HDF5 dependencies are installed through extras. A missing extra must
fail with an actionable installation message; no optional operation may silently no-op.

## Serialization compatibility / serialization 互換性

| Artifact | Current write schema | Read compatibility | Meaning |
| --- | --- | --- | --- |
| WDF | format 0.4, Frame state 1 | exact format 0.4 | Concrete typed Frame and display history |
| Recipe JSON | `wandas.recipe` 2 | exact schema 2 | Reusable executable operation intent |

WDF 0.1–0.3 and future format versions fail explicitly instead of being guessed or
silently upgraded. WDF 0.4 stores one concrete built-in Frame's type, validated
constructor state, raw tensor values and dtype, semantic dimensions and represented
coordinates, sampling rate, labels, strict-JSON metadata, stable channel state,
source-time offsets, and display history. It does not store live lineage, `previous`
references, operation objects or callables, executable Recipe intent, Dask graphs,
chunk/task topology, scheduler state, or an open runtime backend.

A Frame loaded from WDF owns access to its source internally. Keep the source path
unchanged while that Frame or Frames derived from it are in use, and read NumPy
values through `frame.data` as with every other Frame. Users do not manage the
xarray/Dask backend directly.

Future Recipe schema versions also fail explicitly. Recipe JSON stores reusable
operation intent and named runtime input slots, not Frame samples, live lineage,
Dask graphs, or callables. WDF history is display-only and is not executable Recipe
intent: use WDF for a concrete typed result and Recipe JSON for replay.

## Compatibility decisions and release records

Every user-visible removal or incompatible semantic change is classified before
merge as **stable user surface**, **experimental surface**, **serialized schema or
operation version**, or **internal-only**. Release notes for the change record:

- the affected name, artifact, or operation ID and its classification;
- the version in which deprecation began, or the classification-appropriate
  no-deprecation value described below;
- the supported replacement or migration;
- the removal or behavior-change version; and
- for an exception, the reason and a link to the approving issue or PR.

Stable and supported serialized contracts may use `none` only when the linked
decision explicitly approves an exception. Experimental removals may use `none`
without an exception because they do not require a warning release; their
classification, migration, and change version are still recorded. Internal-only
changes use `not applicable` for the deprecation field.

Stable user surfaces and supported serialized contracts follow the normal warning
window above. A stable contract may bypass it only for a security issue, data-loss
risk, materially incorrect numerical meaning, or a compatibility adapter whose
retention is explicitly judged more harmful than removal. A maintainer must approve
the exception in the tracking issue or PR before release. The release note must make
the missing deprecation period visible; silence is not an exception process.

An exception does not reclassify a stable surface as experimental. It documents why
the normal policy was not followed, gives users the most direct migration, and keeps
the historical decision reviewable. Copy
[`release-notes/template.md`](../release-notes/template.md) when preparing any release
that contains a compatibility change so these fields are reviewed with the release.
An ordinary patch release with no compatibility change may instead state that none
occurred.

## Gate for new algorithms / 新規 algorithm の条件

A new algorithm is not complete merely because it computes a number. Its supported
contract must cover the relevant items below:

- a typed Frame result or an explicit scalar boundary;
- immutable input and synchronized sampling/channel/unit metadata;
- Dask laziness, or a documented eager reduction boundary;
- semantic lineage and either portable Recipe support or an explicit runtime-only rejection;
- notebook static visualization when the result is a new visual domain;
- reference/theoretical numerical tests and serialization behavior where applicable.

Existing FFT, STFT, Welch, fractional-octave, and spectral-level APIs follow the
documented [spectral numerical contracts](spectral-numerical-contracts.md).
Corrections to those contracts require reference-value and public round-trip tests;
terminology alone must not silently change amplitude into power or PSD.

This gate keeps Wandas focused on context-preserving analysis rather than matching the
raw function count of SciPy or librosa.
