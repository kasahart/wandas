# AudioOperation execution dependencies

Channel dependency and time dependency are separate dimensions. An operation can
be independent across channels while still requiring one complete, continuous
time series for every channel. The base class expresses the numerical contract;
the graph builder only implements that contract.

## Choose the operation base

Use `ChannelIndependentAudioOperation` only when every output channel depends on
the corresponding input channel and the kernel satisfies:

```text
op(all_channels) == concatenate(op(channel) for each channel)
```

Its `_process()` implementation must be correct for both a single-channel input
and a complete multi-channel tensor. The contract does not promise one task per
channel, a scheduler, or a particular Dask graph.

Use `AudioOperation` when channels interact, runtime configuration can change the
dependency, the operation has multiple inputs, or the independence proof does not
hold for every supported parameter. A conservative whole-Frame fallback may be
less parallel, but it must not change the numerical or metadata meaning.

## Decision procedure for a new operation

1. Read the numerical kernel and identify its channel and time dependencies.
2. Choose the narrowest valid base class; do not choose based on current chunk
   layout or an expected scheduler topology.
3. Check that `_process()` accepts the shapes used by one-channel and
   multi-channel execution, including zero or unknown channel counts where the
   public path permits them.
4. Keep parameter-dependent eligibility in the operation that owns the
   parameters. Do not add a family-specific branch to a central graph builder.
5. Test exact equivalence with the conservative whole-Frame path, output shape and
   dtype before compute, metadata, source-time offsets, lineage, and laziness.
6. If the operation is portable, test its public Recipe extraction and replay
   through the existing declaration and registry path.

## Representative choices

The following examples explain the decision method; they are not a state ledger.

| Situation | Base or path | Reason |
| --- | --- | --- |
| `remove_dc`, a per-channel filter, or resampling | `ChannelIndependentAudioOperation` | Each channel uses its own complete time series; resampling may update time length and sampling rate. |
| Cross-channel mean, difference, or coherence | `AudioOperation` or a Dask-native path | An output depends on more than its corresponding input channel. |
| `normalize` with parameter-dependent eligibility | `AudioOperation` | The operation owns the predicate for configurations that are channel-independent. |
| FFT, STFT, or Welch | Existing conservative path | Windowing, temporary arrays, and output semantics require numerical evidence before changing the boundary. |
| User-defined `custom` operation | Conservative `AudioOperation` default | User code may have unknown channel dependencies, shape behavior, or state. |

## Private execution boundary

`AudioOperation.process()` validates inputs and output shape/dtype, then delegates
graph construction. Channel-independent execution may call one kernel per channel;
the fallback may call one kernel for the complete tensor. The graph hook, Dask
topology, chunk layout, scheduler, and task count are private implementation
details. Do not document them as user controls or fixed guarantees.

Whole-frame fallback must preserve the public result, metadata, lineage, and
Recipe behavior. A new execution form is valid only when it preserves those
contracts and the `_process()` implementation remains correct for both input
arities.

## Focused evidence

Keep numerical and Frame tests close to the operation. Compare an independent
reference or the forced conservative path, and include representative channel
counts and parameter boundaries. When a performance question matters, record a
focused benchmark and its environment in the change discussion; timing, RSS,
task counts, and worker topology are evidence for a decision, not public promises.
