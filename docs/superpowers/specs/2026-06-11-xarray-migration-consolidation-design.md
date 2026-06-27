# Phase 5: xarray Migration Consolidation / Public API Cleanup

## Purpose

Phase 5 is a consolidation phase, not a feature phase. Its purpose is to make the xarray migration easier to understand by separating lasting design decisions from temporary implementation logs and by removing stale public wording from the repository.

The expected outcome is a smaller and clearer documentation surface:

- persistent architecture decisions live in `docs/design/`
- temporary implementation plans are removed after the work is complete
- public docs and docstrings describe the current xarray-backed model
- old concepts are verified with targeted `rg` checks

## Current State

The migration has already reached these points:

- `BaseFrame` stores data in `_xr: xarray.DataArray`
- `_data` remains a read-only compatibility alias to `_xr.data`
- xarray dims and channel coords are used for the supported frame schemas
- durable frame-level state is backed by `_xr.attrs`; operation provenance is runtime-only lineage with a derived `operation_history` compatibility view
- `FrameMetadata` has been removed
- `ChannelMetadata` is a standard-library dataclass
- operation execution is still the existing Wandas/Dask path

The remaining cleanup is mostly documentation and public API hygiene. The repository currently has large `docs/superpowers/plans/*.md` implementation plans that were useful during development, but they are too detailed to serve as long-term design references.

## Scope

Phase 5 will do the following:

1. Delete completed implementation plans from `docs/superpowers/plans/`.
2. Add a concise ADR under `docs/design/` for the current xarray migration boundary.
3. Add a short public documentation pointer in README or `docs/src/explanation/index.md`.
4. Remove stale public wording and comments from `BaseFrame`, `ChannelMetadata`, and docs.
5. Verify old concepts with repository searches.

The cleanup searches must include:

```bash
rg -n "Pydantic" README.md docs wandas tests pyproject.toml
rg -n "FrameMetadata" README.md docs wandas tests pyproject.toml
rg -n "metadata\.source_file" README.md docs wandas tests
rg -n "metadata\.merged" README.md docs wandas tests
rg -n "xarray bridge|xarray-bridge" README.md docs wandas tests
```

Some matches may remain in historical design/spec documents only if they are explicitly describing removed behavior. Public docs, production code, tests, and API docstrings should not describe those old concepts as current behavior.

## Persistent ADR

The new ADR should live in `docs/design/YYYY-MM-DD-xarray-migration-consolidation.md`.

It should be short and current-state oriented. It should not duplicate the implementation plans.

The ADR should document this responsibility boundary:

```text
xarray
  - data container
  - named dimensions
  - selected coordinates
  - frame-level attrs storage

Wandas
  - validation
  - waveform/audio domain semantics
  - operation history semantics
  - channel metadata objects
  - operation execution
```

It should also document these migration notes:

- use plain `dict` metadata instead of `FrameMetadata`
- source file metadata is stored as `metadata["_source_file"]`
- replace `metadata.merged(...)` with normal dict merging
- `ChannelMetadata` is a dataclass and has no Pydantic APIs
- explicit `ref` values, including `ref=1.0`, are preserved

## Public Documentation

README or `docs/src/explanation/index.md` should contain a short pointer to the ADR. The public wording should be brief: Wandas is using xarray internally as the labelled storage/state layer while preserving Wandas as the domain API and operation layer.

This should not become a long architecture guide.

## Non-Goals

Phase 5 will not introduce:

- `from_xarray`
- NetCDF or Zarr support
- xarray-native operation dispatch
- `xr.apply_ufunc` or `map_blocks` operation rewrites
- xarray-native channel metadata ownership or public APIs; Wandas still exposes and validates channel metadata through `ChannelMetadata` views
- dense time coordinates
- new frame schemas

## Documentation Retention Policy

`docs/superpowers/specs/` can contain design artifacts created during active work. `docs/superpowers/plans/` should not keep completed step-by-step implementation logs after the phase has landed.

Long-term architectural decisions belong in `docs/design/`.

## Verification

Because this phase is mostly documentation and cleanup, full numerical regression is not required by default. Verification should include:

```bash
rg -n "Pydantic|FrameMetadata|metadata\.source_file|metadata\.merged|xarray bridge|xarray-bridge" README.md docs wandas tests pyproject.toml
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/core/test_metadata.py tests/core/test_xarray_storage_only.py -q
```

If production code or public API behavior changes unexpectedly, run the full test suite.

## Success Criteria

Phase 5 is complete when:

1. `docs/superpowers/plans/` no longer contains completed implementation plans.
2. A concise ADR in `docs/design/` explains the current xarray-backed storage/state boundary.
3. Public docs point readers to the ADR instead of long implementation plans.
4. Current public docs and docstrings no longer present removed concepts as active API.
5. `rg` checks show no stale active references to Pydantic, `FrameMetadata`, `metadata.source_file`, `metadata.merged`, or the old xarray bridge framing.
6. Verification commands pass.
