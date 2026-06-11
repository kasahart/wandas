# Phase 4: ChannelMetadata Dataclass Simplification

## Goal

Replace `ChannelMetadata`'s Pydantic `BaseModel` implementation with a lightweight standard-library `dataclass`, while preserving the channel metadata concept and the existing Wandas channel metadata behavior.

This phase is a dependency and complexity reduction step. It does not move channel metadata storage into xarray coords or attrs.

## Context

After Phase 3, frame-level state is backed by `BaseFrame._xr.attrs`, and `FrameMetadata` has been removed. The remaining metadata-specific production type is `ChannelMetadata` in `wandas/core/metadata.py`.

`ChannelMetadata` currently depends on Pydantic for:

- `BaseModel`
- `Field(default_factory=dict)`
- `model_fields`
- `model_copy(deep=True)`
- `model_dump_json()`
- `model_validate_json()`
- `ValidationError` handling in `BaseFrame`

The fields are simple enough to represent with a dataclass: `label`, `unit`, `ref`, and `extra`.

## In Scope

- Convert `ChannelMetadata` from `pydantic.BaseModel` to `@dataclass`.
- Keep `label`, `unit`, `ref`, and `extra` fields.
- Keep dict-like access through `__getitem__` and `__setitem__`.
- Keep `matches_query()` behavior.
- Keep `to_json()` and `from_json()` using the standard `json` module.
- Replace internal `model_copy(deep=True)` calls with `copy.deepcopy()`.
- Replace `ChannelMetadata.model_fields` usage with a stable class-level field set.
- Remove `pydantic.ValidationError` handling from `BaseFrame`.
- Remove `pydantic` from runtime dependencies if no production or test code still imports it.

## Out of Scope

- Moving `_channel_metadata` storage into xarray coords or attrs.
- Moving `unit`, `ref`, or `extra` into xarray coords or attrs.
- Changing `channels` property mutability or return semantics.
- Adding proxy objects for channel metadata.
- Adding `from_xarray`, NetCDF, or Zarr support.
- Changing operation execution or xarray operation dispatch.
- Adding generated numeric coordinates.
- Keeping Pydantic compatibility shims such as `model_copy()`.

## Behavior to Preserve

`ChannelMetadata` should preserve these public behaviors:

- Default construction:

  ```python
  ChannelMetadata(label="", unit="", ref=1.0, extra={})
  ```

- Unit-based reference inference at construction time:

  ```python
  ChannelMetadata(unit="Pa").ref == unit_to_ref("Pa")
  ```

- Explicit `ref` wins during construction:

  ```python
  ChannelMetadata(unit="Pa", ref=0.5).ref == 0.5
  ```

- Unit assignment after construction updates `ref`:

  ```python
  ch = ChannelMetadata()
  ch.unit = "Pa"
  ch.ref == unit_to_ref("Pa")
  ```

- Dict-like access:

  ```python
  ch["label"]
  ch["unit"]
  ch["ref"]
  ch["custom"]
  ch["custom"] = value
  ```

- `extra` remains a mutable dictionary.
- `matches_query()` still supports literal comparisons and compiled regex patterns.
- JSON round-trip preserves `label`, `unit`, `ref`, and `extra`.

## Behavior to Remove

Phase 4 intentionally removes Pydantic-specific APIs from `ChannelMetadata`:

- `model_copy()`
- `model_fields`
- `model_dump_json()`
- `model_validate_json()`
- Pydantic `ValidationError`

Internal code should use standard-library mechanisms instead. External code that used Pydantic-specific APIs must migrate to `copy.deepcopy()`, `ChannelMetadata._MODEL_FIELDS`, `to_json()`, or `from_json()` as appropriate.

## Dataclass Design

The implementation should use `dataclasses.dataclass` and `field(default_factory=dict)`.

Recommended shape:

```python
@dataclass
class ChannelMetadata:
    label: str = ""
    unit: str = ""
    ref: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)
    _initialized: bool = field(default=False, init=False, repr=False)

    _MODEL_FIELDS = frozenset({"label", "unit", "ref", "extra"})
```

`__post_init__()` should infer `ref` only when `unit` is non-empty and `ref` is still the default value. It should then set `_initialized` to `True`.

`__setattr__()` should update `ref` when `unit` changes after initialization. The `_initialized` guard is required so dataclass initialization does not overwrite an explicitly provided `ref`.

## BaseFrame Changes

`BaseFrame` should stop importing Pydantic `ValidationError`.

When converting channel metadata dictionaries, invalid dictionaries should raise a clear `ValueError` with the channel index, the original dict, and the construction error. This replaces the current Pydantic error formatting.

Query validation should stop using `ChannelMetadata.model_fields` and use the dataclass field set instead.

## Copy Semantics

Any internal use of `model_copy(deep=True)` should become `copy.deepcopy(channel_metadata)`.

This keeps existing isolation behavior for nested `extra` dictionaries and avoids introducing a Pydantic-style compatibility shim.

## JSON Semantics

`to_json()` should serialize a plain object containing exactly:

- `label`
- `unit`
- `ref`
- `extra`

`from_json()` should parse JSON with `json.loads()`, require the decoded value to be a dict, and raise a clear `ValueError` if it is not.

Unknown JSON keys should follow normal dataclass constructor behavior through `ChannelMetadata(**data)`, producing a clear construction error rather than being silently accepted.

## Dependency Cleanup

After replacing all Pydantic usage, remove `pydantic>=...` from `pyproject.toml` and update the lockfile with the repository's package manager.

Before removal, verify no production or test code imports Pydantic:

```bash
rg -n "pydantic|ValidationError|model_fields|model_copy|model_dump|model_validate" wandas tests pyproject.toml
```

Expected final result: no `pydantic` imports or Pydantic-specific API usage in `wandas` or `tests`.

## Tests

Add or update tests for:

- `ChannelMetadata` is not a Pydantic `BaseModel`.
- `unit="Pa"` infers `ref` when ref is default.
- Explicit `ref` is preserved during initialization.
- Assigning `unit` after initialization updates `ref`.
- `copy.deepcopy(ChannelMetadata(...))` isolates nested `extra` values.
- Invalid channel metadata dictionaries raise a clear `ValueError`.
- Query validation uses the dataclass field set.
- JSON round-trip still works.
- Non-object JSON passed to `from_json()` raises a clear `ValueError`.

Run focused tests first:

```bash
uv run pytest tests/core/test_metadata.py tests/core/test_base_frame.py::TestBaseFrameChannelMetadata -q
```

Then run the full verification set:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check wandas tests
uv run pytest
```

## Success Criteria

Phase 4 is complete when:

1. `ChannelMetadata` is a standard-library dataclass, not a Pydantic model.
2. `pydantic` is removed from runtime dependencies.
3. `BaseFrame` no longer imports `ValidationError`.
4. Production code no longer uses `model_fields`, `model_copy`, `model_dump_json`, or `model_validate_json`.
5. Existing channel metadata behavior is preserved except for intentionally removed Pydantic-specific APIs.
6. `_channel_metadata` storage remains unchanged.
7. No xarray coords/attrs expansion is introduced.
8. Operation execution remains unchanged.
9. Full lint, format, type, and test verification passes.

## Next Phase Candidates

After this phase, consider a separate analysis phase for channel metadata storage:

- whether channel labels, units, and references should be represented in xarray coords;
- how to preserve mutable `channels` semantics if storage moves;
- whether `extra` belongs in coords, attrs, or a separate Wandas-owned structure.

That storage migration is deliberately not part of Phase 4.
