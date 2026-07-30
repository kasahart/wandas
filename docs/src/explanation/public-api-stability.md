# Public API and schema stability / 公開 API・schema 安定性

Wandas 0.6 keeps the user entry surface small and classifies the broader library before
the 1.0 compatibility promise.

The machine-readable authority is
`wandas._public_api.PUBLIC_API_INVENTORY`. It assigns every package-level symbol in
`wandas`, `wandas.frames`, `wandas.frames.mixins`, `wandas.processing`,
`wandas.utils`, and `wandas.datasets` exactly one of four classifications and records
whether the name belongs in `__all__`. Documentation and export drift is a CI-tested
error.

- **stable public**: compatibility changes require documentation, tests, and the
  deprecation window below;
- **experimental public**: user-visible and documented, but may change in a feature
  release;
- **deprecated compatibility**: still supported with a documented replacement and
  earliest removal version;
- **private/internal**: may remain directly importable for implementation or migration
  reasons but is excluded from `__all__` and has no public compatibility promise.

Processing registry storage (`_OPERATION_MODULES`, `_OPERATION_REGISTRY`), lazy
registration, calibration plumbing, and the `wandas.utils` introspection and
optional-import helpers are private/internal and outside package `__all__` lists.
Direct importability does not promote them into the supported API.

Every deprecated inventory entry names its replacement and support window.
`from_ndarray` uses `from_numpy`; it has been deprecated since 0.2.0, remains
supported through 0.6.x, and is removable no earlier than 0.7.0. Direct processing
`Trim` uses `Frame.trim`; it is deprecated in 0.6.2, remains supported through 0.6.x,
and is removable no earlier than 0.7.0. The general feature-release window below still
governs any later change to these dates.

`wandas.datasets` exports no sample dataset or packaged audio asset. Repository
learning files are not an installed dataset API; use experimental `generate_sin` for
known signals, or stable `read`/`from_folder` for application-owned data.

## Stable user surface / 安定した user surface

- Top level: `read`, `from_numpy`, `from_folder`, `load`, `supported_formats`.
- Stable top-level compatibility conveniences outside `__all__`: `read_wav` and
  `read_csv`; new code normally uses `read`.
- Built-in Frame types and their primary workflow: immutable typed transforms,
  metadata/channel views, `frame.data` as the canonical NumPy-value boundary,
  `to_numpy()` and NumPy's array protocol as equivalent interoperability APIs,
  `plot`, `describe`, and `BaseFrame.save`.
- `RecipePlan.from_frame`, `apply`, `to_dict`, `from_dict`, `save`, and `load`.
- WDF 0.4 typed round-trip and Recipe schema 2 strict JSON.

Changes to this surface require tests, documentation, and a deprecation period. During
0.x, a deprecation warning remains for at least one feature release before removal.
1.0 will define the longer support window.

## Experimental surface / 実験的 surface

- Recipe extension registries/decorators used to declare third-party operations.
- sklearn adapters in `wandas.pipeline.sklearn`.
- Internal xarray/Dask storage helpers and private attributes such as `_xr` and
  `_data`.
- Top-level `generate_sin` and `setup_wandas_logging` conveniences outside
  `wandas.__all__`.
- Direct `wandas.processing` operation and extension contracts, including
  `AudioOperation`, `ChannelIndependentAudioOperation`, `create_operation`,
  `get_operation`, and `register_operation`.
- Exported Frame mixins, including `SpectralPropertiesMixin`.

Experimental APIs may change in a feature release, but changes must still be explicit
and must not silently alter stored data or numerical meaning.

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
silently upgraded. A Frame loaded from WDF owns access to its source internally. Keep
the source path unchanged while that Frame or Frames derived from it are in use, and
read NumPy values through `frame.data` as with every other Frame. Users do not manage
the xarray/Dask backend directly. Future Recipe schema versions also fail explicitly.
Live lineage, Dask graphs, callables, and Frame samples are outside Recipe JSON. WDF
history is display-only and is not executable Recipe intent.

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
