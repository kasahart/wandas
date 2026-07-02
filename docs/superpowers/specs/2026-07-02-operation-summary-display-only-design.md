# Operation Summary Display-Only Design

## Background

PR #250 grew around `operation_summaries` and `to_summary()` as if they were a
future persistence contract. The `portable` flag then required increasingly
precise claims about whether each parameter could be faithfully represented and
restored. Review feedback kept finding edge cases where that claim was too broad:
non-finite numbers, callables, exact rationals, arrays, mapping keys, container
subclasses, and unlabeled frame operands.

That direction is larger than the immediate need. The current feature should
help users inspect processing history without computing lazy data. It should not
promise save/restore semantics.

## Decision

`operation_summaries` and `to_summary()` are display-only APIs.

They return a lightweight, strict-JSON-serializable summary that a human can read
to understand the operation history. They do not define a lossless
serialization, restoration, migration, or WDF persistence contract.

The summary shape is intentionally small:

```python
{
    "operation": "normalize",
    "params": {
        "norm": {"type": "float", "value": "inf"}
    },
}
```

The summary must include:

- `operation`: a human-readable operation name or symbol.
- `params`: JSON-safe display values derived from operation parameters.

The summary must not include:

- `schema_version`
- `portable`
- fields that imply restoration or persistence guarantees

## PR Strategy

PR #250 should be closed instead of continuing to revise it. Its commit history
and review threads are dominated by the old portability contract, so continuing
there would keep the design confusing.

Before closing PR #250:

- Remove issue-closing language from the PR body so it no longer closes or
  claims ownership of linked issues.
- Leave a closing comment explaining that the persistence-oriented design was
  withdrawn and the feature is being restarted as display-only.
- Close the PR.

The replacement work should start from `develop` on a new branch. The new PR
should describe `operation_summaries` as display-only and explicitly state that
save/restore will require a separate `to_spec()` / `from_spec()` design if it is
needed later.

## Summary Value Rules

The conversion helper should prioritize readable, JSON-safe display output:

- Plain JSON scalars stay plain where safe: `None`, `str`, `bool`, finite
  `int` / `float`.
- Non-finite floats become descriptors such as
  `{"type": "float", "value": "inf"}` so strict JSON encoding succeeds.
- Complex values become descriptors with JSON-safe real and imaginary parts.
- Mappings become JSON objects with string keys and recursively converted
  values. Key conversion is display-only and may be lossy.
- Lists, tuples, sets, and frozensets become JSON arrays. Sets and frozensets
  should use deterministic ordering.
- NumPy arrays and Dask arrays become descriptors with type, shape, dtype, and
  chunks where available. Values are not embedded.
- Dask metadata must be read without triggering compute.
- Callables become descriptors such as
  `{"type": "callable", "name": "module.qualname"}`.
- Unknown objects become `{"type": "ClassName"}`.

These rules are intentionally lossy. Lossiness is acceptable because summaries
are for inspection, not restoration.

## Component Design

`AudioOperation.to_summary()` should stay thin:

1. Ask the operation for `to_params()`.
2. Convert each parameter through the JSON-safe display conversion helper.
3. Return `{"operation": self.name, "params": converted_params}`.

`BinaryOperation.to_summary()` should follow the same shape:

1. Build display params from `symbol`, `operand_kind`, and any operand descriptor.
2. Return `{"operation": self.symbol, "params": converted_params}`.

`CustomOperation.to_summary()` may keep an `implementation` display field because
it helps humans understand the callable that was used. It should not mark the
summary as non-portable because portability is no longer part of the contract.

`BaseFrame.operation_summaries` should remain a read-only projection from
existing lineage. It should not mutate lineage, preserve additional state, or
trigger Dask compute.

Fallback summaries for operations without `to_summary()` should use the same
display-only shape. Their params should also pass through the JSON-safe display
conversion helper:

```python
{
    "operation": operation_name,
    "params": converted_operation_params,
}
```

## Removed Design

The replacement implementation should remove the persistence-oriented pieces
introduced by PR #250:

- `SUMMARY_SCHEMA_VERSION`
- `_summary_is_portable()`
- `_operand_descriptor_is_portable()`
- `portable` assertions and behavior
- tests whose only purpose is proving whether a value is losslessly represented
- documentation that presents summaries as portable or persistence-ready

Tests for strict JSON safety, no Dask compute, operation ordering, multi-input
lineage, and readable descriptors should remain.

## Future Persistence Contract

If Wandas later needs save/restore, it should not extend `to_summary()`.

Persistence should be designed as a separate explicit contract, likely named
`to_spec()` / `from_spec()`, with operation-specific lossless parameter schemas.
That design should define versioning, validation, unsupported parameter handling,
and migration behavior independently from display summaries.

## Testing

The replacement PR should test:

- `AudioOperation.to_summary()` returns only `operation` and `params`.
- `BinaryOperation.to_summary()` returns only `operation` and `params`.
- `CustomOperation.to_summary()` includes a display implementation reference but
  no `portable` flag.
- `BaseFrame.operation_summaries` preserves lineage order and multi-input
  lineage flattening.
- `operation_summaries` is strict-JSON-serializable with
  `json.dumps(..., allow_nan=False)`.
- Dask arrays are summarized by metadata without compute.
- Arrays, callables, opaque objects, sets, mappings, and non-finite numbers have
  readable descriptors.
- No tests assert save/restore, schema versioning, or portability.

Relevant verification commands:

```bash
uv run pytest tests/core/test_base_frame_lineage.py tests/processing/test_base_operations.py tests/processing/test_custom_operation.py tests/frames/test_channel_processing.py
uv run ruff check wandas tests --config=pyproject.toml
uv run ty check wandas tests
```
