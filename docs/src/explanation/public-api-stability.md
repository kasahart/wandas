# Public API and schema stability / 公開 API・schema 安定性

Wandas 0.6 keeps the user entry surface small and classifies the broader library before
the 1.0 compatibility promise.

## Stable user surface / 安定した user surface

- Top level: `read`, `from_numpy`, `from_folder`, `load`, `supported_formats`.
- Built-in Frame types and documented Frame methods, including `plot`, `describe`,
  typed transforms, and `BaseFrame.save`.
- `RecipePlan.from_frame`, `apply`, `to_dict`, `from_dict`, `save`, and `load`.
- WDF 0.3 typed round-trip and Recipe schema 2 strict JSON.

Changes to this surface require tests, documentation, and a deprecation period. During
0.x, a deprecation warning remains for at least one feature release before removal.
1.0 will define the longer support window.

## Experimental surface / 実験的 surface

- Recipe extension registries/decorators used to declare third-party operations.
- sklearn adapters in `wandas.pipeline.sklearn`.
- Internal xarray storage helpers and private attributes such as `_xr` and `_data`.

Experimental APIs may change in a feature release, but changes must still be explicit
and must not silently alter stored data or numerical meaning.

## Optional-domain extensions / optional 領域

Effects, psychoacoustic metrics, ML tensor conversion, interactive marimo/IPython
display, and WDF/HDF5 dependencies are installed through extras. A missing extra must
fail with an actionable installation message; no optional operation may silently no-op.

## Serialization compatibility / serialization 互換性

| Artifact | Current write schema | Read compatibility | Meaning |
| --- | --- | --- | --- |
| WDF | format 0.3, Frame state 1 | WDF 0.1–0.3 | Concrete typed Frame and display history |
| Recipe JSON | `wandas.recipe` 2 | exact schema 2 | Reusable executable operation intent |

Future schema versions fail explicitly. Live lineage, Dask graphs, callables, and Frame
samples are outside Recipe JSON. WDF history is display-only and is not executable Recipe intent.

## Gate for new algorithms / 新規 algorithm の条件

A new algorithm is not complete merely because it computes a number. Its supported
contract must cover the relevant items below:

- a typed Frame result or an explicit scalar boundary;
- immutable input and synchronized sampling/channel/unit metadata;
- Dask laziness, or a documented eager reduction boundary;
- semantic lineage and either portable Recipe support or an explicit runtime-only rejection;
- notebook static visualization when the result is a new visual domain;
- reference/theoretical numerical tests and serialization behavior where applicable.

This gate keeps Wandas focused on context-preserving analysis rather than matching the
raw function count of SciPy or librosa.
