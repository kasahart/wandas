# ADR: Channel-independent graph construction with `map_blocks`

- **Status**: Proposed
- **Date**: 2026-10-23
- **Issue**: #505

This ADR owns how `ChannelIndependentAudioOperation` builds its lazy graph. The
numerical contract of channel-independent operations, and the cases that use the
whole-frame graph, stay as described in
`docs/src/contributing/frame-operation-extensions.md`.

## Context

`_try_build_channelwise_graph()` slices each channel (`data[i:i+1]`), passes the
slice to `dask.delayed(_execute_wandas_operation)`, wraps each result with
`da.from_delayed`, and concatenates the channels.

`delayed()` finalizes a Dask array argument. It optimizes the argument's graph and
gives the tasks new names for that one call, so two channels of the same input do
not share upstream tasks. Each chained stage copies the previous stage's graph again.

With 8 channels, the graph of a few chained filters already holds tens of
thousands of keys, and graph construction takes seconds to minutes before any
computation starts. The computed values are correct, so this is a graph-size and
graph-construction problem, not a numerical one. The measurements are recorded in
#505 and in the pull requests that introduce this ADR and its implementation.

## Decision

Build the channel-wise graph with one `da.map_blocks` call over the input rechunked
to one block per channel. Each block holds one complete channel:

```python
source = data.rechunk({0: 1, **{axis: -1 for axis in range(1, data.ndim)}})
return da.map_blocks(
    _execute_wandas_operation,
    operation,
    source,
    chunks=((1,) * channel_count, *((length,) for length in output_shape[1:])),
    drop_axis=list(range(1, source.ndim)),
    new_axis=list(range(1, len(output_shape))),
    dtype=output_dtype,
    meta=np.empty((0,) * len(output_shape), dtype=output_dtype),
    name=task_name,
)
```

Each stage then adds one task per channel and references the previous stage's
blocks by key. When the input is not already one block per channel, for example a
single `(C, n)` block or an input chunked along time, the rechunk adds its own
tasks. The graph still grows linearly with channels × stages.

### Contracts kept

- **Eligibility.** The same conditions as today: one input, a known positive
  channel count, and an output whose first axis has the same channel count. Runtime
  inputs, unknown or zero channel counts, and channel-axis-changing outputs keep the
  whole-frame graph.
- **Kernel input.** The kernel receives one channel with shape `(1, ...)` and the
  complete non-channel axes, as the slice-and-finalize path gives it now. An input
  chunked along time is merged per channel by the rechunk, which is what
  finalizing the slice did.
- **Output shape and dimensions.** `chunks` is taken from
  `calculate_output_shape()`. Only the channel axis is aligned between input and
  output. All other input axes are dropped and all other output axes are new, so
  operations that change length (`resampling`, `noct_spectrum`) and a future
  channel-independent operation that changes the number of dimensions use one
  rule. The output chunks are
  `((1,) * channels, (n,), ...)`, the same as the current concatenation.
- **Dtype.** `dtype` comes from `calculate_output_dtype()`. The kernel result is
  not cast, as with `from_delayed` now.
- **No kernel call while building the graph.** `meta` is passed explicitly, so
  `map_blocks` does not call the kernel on empty arrays to infer it. Graph
  construction stays lazy, and kernels that reject empty input are not affected.
- **Task names.** For `pure=True` the name is
  `f"{operation.name}-channel-{tokenize(operation, source, output_shape, output_dtype)}"`,
  so equal operations on the same input share keys, as they do now. For
  `pure=False` the token is a fresh `uuid4().hex`, so every call gets its own
  tasks, as `delayed(pure=False)` gives now.
- **Validation and errors.** `process()` still validates inputs before building
  the graph. Kernel exceptions are raised at compute time, per channel, as now.
- **Frame contract.** Metadata, lineage, Recipe behavior and the lazy boundary are
  unchanged because only `_try_build_channelwise_graph()` changes.

### Alternatives considered

- **`to_delayed(optimize_graph=False)` per block, then `from_delayed`.** Also
  linear, but it still creates one `from_delayed` layer and one concatenate per
  stage, and it needs the same naming rules. `map_blocks` expresses the same thing in one layer.
- **`map_overlap` or splitting the time axis.** Rejected. Many kernels (IIR
  filters, resampling, `noct_spectrum`) need each channel's whole time axis, and splitting it
  would change state, edge handling and phase. This ADR does not change time-axis
  dependency.
- **Keep `delayed`, but cull or rename upstream keys by hand.** Rejected. It
  depends on Dask's internal finalize naming.

## Consequences

- The graph size for chained channel-independent operations becomes linear in
  channels × stages, and graph construction time falls with it.
- Task names change from per-channel `"_execute_wandas_operation-<token>"` delayed
  keys plus `finalize`, `from-value` and `concatenate` layers to one
  `"<name>-channel-<token>"` blockwise layer per stage. Task names and counts are
  not public promises.

## Verification for the implementation PR

- Graph keys for chained operations grow linearly: the key increase from one
  extra stage is the same at every depth and proportional to the channel count.
- Through Frame methods, the channel-wise result equals
  `_build_whole_frame_graph()` exactly for operations that take the channel-wise
  path: a filter, `a_weighting`, `resampling`, `noct_spectrum`, `normalize` with
  integer input, and `hpss_*`.
- Called directly on `FFT`, `STFT` and `RmsTrend`, the channel-wise graph builder
  equals `_build_whole_frame_graph()` exactly. This covers length and
  dimension changes that no channel-independent operation has today.
- The kernel is not called during graph construction.
- A time-chunked input reaches the kernel as complete `(1, n)` channels.
- `pure=True` gives equal keys for equal operations on the same input;
  `pure=False` gives different keys.
- The existing whole-frame fallback tests still pass.
- Run `scripts/scalability_benchmark.py` with the representative matrix on main
  and on the candidate, and record the task counts and timings in the PR.
