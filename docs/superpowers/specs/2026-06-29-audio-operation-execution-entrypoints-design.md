# AudioOperation Execution Entrypoint Simplification

## Goal

Simplify `AudioOperation` execution after PR #247 by making `process()` the only public Dask Array execution path and making the private numerical kernel contract unambiguous.

The refactor should reduce the number of execution entrypoints, remove misleading delayed-only APIs, and keep Dask shape/dtype metadata behavior covered in one place.

## Context

Issue #248 is a follow-up to PR #247. PR #247 adds variadic Dask input support so `AddWithSNR` can receive clean and noise arrays as runtime Dask graph inputs instead of storing the noise array in operation config.

That PR clarifies multi-input lineage, but it also leaves several execution layers:

- `_process_array()` for concrete single-input kernels.
- `_process_inputs()` as the single-input/multi-input adapter.
- `_delayed()` for Dask delayed task construction.
- `process_array()` as a delayed-only method whose name suggests it returns an array.
- `process()` as the public Dask Array result builder.

The issue is not double execution. The risk is that future fixes for Dask dtype metadata, shape metadata, arity checks, dependency checks, or lineage markers can land in only one path.

## Scope

This design assumes PR #247 has landed or its equivalent changes are present.

In scope:

- Make `AudioOperation.process(data, *inputs)` the only public Dask Array execution API.
- Remove `process_array()`.
- Remove `_delayed()`.
- Remove `_process_inputs()`.
- Rename the private concrete kernel to `_process(*arrays)`.
- Keep `_expected_input_count` as a small early-validation contract.
- Add `calculate_output_dtype(input_dtype)` to centralize dtype metadata for delayed-backed operations.
- Keep Dask-native stats operations as `process()` overrides.
- Update tests so numerical kernel tests call `_process()` and Dask/laziness tests call `process()`.

Out of scope:

- Expanding `CustomOperation` to support multiple runtime inputs.
- Moving Dask-native stats operations onto delayed execution.
- Adding public frame APIs.
- Changing frame metadata, lineage, or immutability semantics.
- Adding compatibility shims for `process_array()`.

## Public Contract

`process(data, *inputs)` is the only supported public execution method for Dask-backed operation execution.

It is responsible for:

- validating the runtime input count,
- building the Dask delayed task,
- wrapping the task with `da.from_delayed()`,
- applying output shape metadata,
- applying output dtype metadata.

`process_array()` is removed instead of deprecated. It returned a delayed object despite its name, and preserving it would keep the ambiguous public-ish path that this refactor is meant to remove.

## Private Kernel Contract

`_process(*arrays)` is the only private concrete numerical kernel entrypoint.

Single-input operations implement:

```python
def _process(self, x):
    ...
```

Multi-input operations implement:

```python
def _process(self, clean, noise):
    ...
```

The Dask task executor calls:

```python
operation._process(*inputs)
```

The method name is intentionally neutral. It avoids the single-array implication of `_process_array()` and avoids a second adapter method such as `_process_inputs()`.

## Base Operation Design

The base shape should be:

```python
class AudioOperation:
    _expected_input_count = 1

    def process(self, data, *inputs):
        self._validate_process_input_count(1 + len(inputs))
        delayed_result = delayed(
            _execute_wandas_operation,
            name=self.name,
            pure=self.pure,
        )(self, data, *inputs)
        return da.from_delayed(
            delayed_result,
            shape=self.calculate_output_shape(data.shape),
            dtype=self.calculate_output_dtype(data.dtype),
        )

    def _process(self, *arrays):
        raise NotImplementedError("Subclasses must implement this method.")

    def calculate_output_shape(self, input_shape):
        return input_shape

    def calculate_output_dtype(self, input_dtype):
        return input_dtype
```

`_expected_input_count` remains because it enables early errors before Dask compute. Without it, generic frame paths such as `_apply_operation_instance()` could build an invalid graph for a multi-input operation and fail only at `.compute()`.

The arity validation message should explain:

- what operation was called,
- how many runtime inputs were expected,
- how many were supplied,
- how to call an operation-specific method when multiple runtime inputs are required.

## Dtype Metadata

Delayed-backed operations should use `calculate_output_dtype(input_dtype)` instead of overriding `process()` only to pass a custom dtype to `da.from_delayed()`.

Operations currently expected to move from `process()` override to `calculate_output_dtype()` include:

- `Normalize`
- `ReSampling`
- `RmsTrend`
- `SoundLevel`

Default behavior returns the input dtype unchanged.

Shape metadata remains owned by `calculate_output_shape(input_shape)`.

## Process Overrides

Most operations should rely on base `process()`.

`process()` overrides remain valid for two narrow cases:

1. Dask-native operations that deliberately avoid delayed execution and preserve Dask chunk behavior, such as `ABS`, `Power`, `Sum`, `Mean`, and `ChannelDifference`.
2. Thin dependency preflight wrappers, such as psychoacoustic and N-octave operations that call `ensure_dependencies()` and then delegate to `super().process(data, *inputs)`.

Overrides should still accept `data, *inputs` and preserve early input-count validation. Retain the `__init_subclass__` validation wrapper from PR #247 so override implementations cannot accidentally bypass `_validate_process_input_count()`.

## CustomOperation

`CustomOperation` remains single-input.

Its implementation moves from `_process_array(x)` to `_process(x)`, preserving:

- function snapshot behavior,
- defensive parameter snapshots,
- `dask_pure` behavior,
- `output_shape_func`,
- display-name behavior.

Multi-input custom functions are out of scope.

## AddWithSNR

After PR #247, `AddWithSNR` should remain a multi-input operation with clean and noise arrays passed as runtime graph inputs.

It should implement:

```python
_expected_input_count = 2

def _process(self, clean, noise):
    ...
```

Its params should remain limited to `{"snr": ...}`. The noise input should not be stored in operation config or lineage params.

Frame-level `add(..., snr=...)` should continue to call:

```python
operation.process(self._data, other._data)
```

The resulting lineage graph should keep both parent branches.

## Test Strategy

Update tests in the affected areas instead of preserving compatibility behavior:

- Replace `process_array()` tests with `process()` tests when checking Dask laziness, graph construction, shape metadata, dtype metadata, or delayed purity.
- Replace `process_array()` tests with `_process()` tests when checking pure numerical kernels on concrete arrays.
- Update base operation tests to cover:
  - `process()` returns a Dask Array,
  - no eager compute occurs,
  - delayed task names and `pure` are passed correctly,
  - shape metadata comes from `calculate_output_shape()`,
  - dtype metadata comes from `calculate_output_dtype()`,
  - arity mismatches fail before compute.
- Update `AddWithSNR` tests for:
  - `process(clean, noise)`,
  - two-parent lineage graphs,
  - clean operation params,
  - Dask laziness.
- Update `CustomOperation` tests for:
  - `_process()` direct kernel behavior,
  - `process()` Dask behavior,
  - `dask_pure` forwarding.
- Update psychoacoustic, spectral, and fade tests that currently call `process_array()`.

Relevant validation commands:

```bash
uv run pytest tests/processing/test_base_operations.py tests/processing/test_custom_operation.py tests/processing/test_effect_operations.py tests/processing/test_temporal_operations.py tests/processing/test_spectral_operations.py tests/processing/test_psychoacoustic_operations.py tests/frames/test_channel_processing.py tests/core/test_base_frame_lineage.py
uv run ruff check wandas tests --config=pyproject.toml -v
uv run ty check wandas tests
```

Run narrower pytest subsets during implementation as needed, then run the listed validation before handoff.

## Risks

Removing `process_array()` is intentionally breaking for callers who reached into operation objects directly. This is acceptable for this refactor because the method name was misleading and the issue calls for removing or reducing it rather than preserving another public path.

Relying only on Python `TypeError` for wrong input counts is not sufficient. Keep `_expected_input_count` so generic frame operation paths can reject multi-input operations before compute and provide Wandas-style error messages.

Do not convert Dask-native stats operations to delayed execution. Their current `process()` overrides preserve Dask-native graph behavior and chunk semantics while marking lineage explicitly.

## Acceptance Criteria

- `AudioOperation` has one public execution path: `process(data, *inputs)`.
- `process_array()` is removed.
- `_delayed()` is removed.
- `_process_inputs()` is removed.
- Concrete operation kernels implement `_process()`.
- Delayed-backed shape and dtype metadata are centralized through `calculate_output_shape()` and `calculate_output_dtype()`.
- `AddWithSNR` remains lazy, multi-input, and free of Dask array config/lineage params.
- Generic single-input frame application rejects multi-input operations before compute.
- Tests describe the clarified contract.
- Relevant pytest, ruff, and ty checks pass.
