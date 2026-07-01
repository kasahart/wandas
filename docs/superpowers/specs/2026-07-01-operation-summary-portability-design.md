# Operation Summary Portability Design

## Context

PR #250 adds `operation_summaries` as a lightweight projection of operation lineage.
New review feedback found three cases where a summary can claim `portable: True`
while either losing parameter values or emitting values that strict JSON cannot
persist:

- Dask arrays with unknown shape or chunks can include `NaN` in descriptor fields.
- `set` and `frozenset` parameters currently fall back to type-only descriptors.
- Opaque objects such as `Path("ir.wav")` also fall back to type-only descriptors.

The chosen contract is strict: a summary is portable only when it preserves enough
parameter information to interpret or persist the operation configuration, not
merely when the resulting summary can be encoded as JSON.

## Goals

- Keep `operation_summaries` strict-JSON safe with `json.dumps(..., allow_nan=False)`.
- Preserve values for supported scalar and container parameter types.
- Mark summaries as non-portable when parameter values are represented only by
  opaque type descriptors.
- Keep frame methods thin and keep summary behavior in `wandas/processing`.
- Avoid sharing `operation_history` internals directly with summary generation,
  because summaries use lightweight display descriptors rather than full history
  snapshots.

## Non-Goals

- Do not add WDF persistence for `operation_summaries` in this change.
- Do not remove or redesign `previous`, `operation_history`, or lineage storage.
- Do not introduce a new public `json_safe` field in this PR.
- Do not attempt to serialize arbitrary Python objects by value.

## Design

`_summary_value(value)` remains responsible for producing a lightweight,
JSON-safe representation:

- Existing scalar handling stays in place, including sentinel objects for
  non-finite numbers.
- Dask array descriptors should recursively normalize `shape` and `chunks`
  entries through `_summary_value()` so unknown dimensions such as `np.nan` do
  not leak raw non-finite floats.
- `set` and `frozenset` should be represented as deterministic lists. Each item
  is converted through `_summary_value()`, then sorted by a stable JSON-based
  sort key so output is repeatable.
- `callable` values remain callable descriptors with a qualified name.
- Opaque values continue to fall back to `{"type": type(value).__name__}` for
  display, but that fallback is not considered portable.

`_summary_is_portable(value)` remains responsible for deciding whether the
original parameter value is sufficiently represented:

- Scalars, NumPy arrays, Dask arrays, mappings, lists, tuples, sets, and
  frozensets are portable when their nested values are portable.
- Callable values are not portable.
- Unknown opaque objects are not portable.
- Type-only descriptors do not make a value portable.

`AudioOperation.to_summary()` should stay thin:

1. Capture `params = self.to_params()` once.
2. Build `params` with `_summary_value()`.
3. Set `portable` with `_summary_is_portable()` over the original parameter
   values.

## Data Flow

```text
AudioOperation.to_params()
  -> raw params
  -> _summary_value() for display / JSON-safe params
  -> _summary_is_portable() for strict portability
  -> OperationSummary
```

This keeps the display projection and portability decision separate while
ensuring they follow the same supported-type boundary.

## Error Handling

Summary generation should not raise for unsupported parameter objects. Unknown
objects should produce a type descriptor and mark the summary non-portable. This
keeps `operation_summaries` usable for inspection even when the operation cannot
be faithfully persisted.

## Testing

Add focused regression tests in `tests/processing/test_base_operations.py`:

- A Dask array parameter with unknown shape or chunks is strict-JSON safe and
  keeps `portable: True` when descriptor fields are normalized.
- `set` and `frozenset` parameters preserve their contents as deterministic
  lists and remain portable when their items are portable.
- `Path` or another opaque non-callable object uses a type descriptor and marks
  the operation summary as non-portable.

Run the existing relevant validation set after implementation:

- `uv run pytest tests/core/test_base_frame_lineage.py tests/processing/test_base_operations.py tests/processing/test_custom_operation.py tests/frames/test_channel_processing.py`
- `uv run ruff check wandas tests --config=pyproject.toml -v`
- `uv run ty check wandas tests`
