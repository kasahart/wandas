# AudioOperation execution dependencies

Wandas separates storage chunking from numerical-kernel execution. Frame data is
normally chunked with one channel per Dask chunk, but that topology alone does not
make a delayed NumPy/SciPy kernel run once per channel. The operation's internal
graph-building specialization owns that decision.

Channel dependency and time dependency are independent dimensions. An operation can
be channel-independent while still requiring one complete, continuous time series for
each channel. Channel-wise execution therefore reduces the number of channels that a
kernel task materializes; it does not introduce time-axis distribution.

## Internal graph-builder contract

`AudioOperation.process()` validates inputs and calculates output shape and dtype,
then delegates graph construction through one polymorphic internal hook:

- The base implementation is conservative whole-frame execution. One delayed kernel receives the complete
  channel-first tensor, preserving all existing operations and custom extensions.
- The channel-independent specialization builds one delayed kernel call for each
  channel, where the kernel input retains shape `(1, ...)`, then concatenates the
  outputs along the channel axis.

The prototype contract is deliberately narrow: channel-wise execution currently
applies to unary operations with a known positive channel count that preserve the
channel axis. Zero-channel, unknown-channel-count, multi-input, and channel-axis-changing
inputs fall back to the base whole-frame graph. This makes channel-wise execution an
opportunistic optimization without strengthening the `AudioOperation` input contract.
New execution forms override the graph-building hook instead of adding cases to a
central dispatcher. The hook, Dask graph, chunks, scheduler, and xarray container
remain private implementation details; the public Frame workflow is unchanged.

## Built-in operation classification

The table classifies the registered numerical families. “Whole signal/axis” means the
operation needs the complete relevant axis for one channel; it does not mean that all
channels depend on one another.

| Built-in family | Channel dependency | Time or analysis-axis dependency | Current execution |
| --- | --- | --- | --- |
| `remove_dc` | Independent | Whole time series per channel for the mean | **Channel-wise prototype** |
| `abs`, `power` | Independent | Pointwise/time-local | Existing Dask-native graph override |
| `normalize` | Parameter-dependent: `axis=-1` is independent; `axis=None` or a channel axis is cross-channel | Whole selected norm axis | Whole-frame |
| `trim`, `fix_length` | Independent | Indexed/padded time-local transform with output-shape change | Whole-frame |
| `fade` | Independent | Needs the full signal length to define the envelope | Whole-frame |
| high-pass, low-pass, band-pass, A-weighting | Independent | Stateful/whole continuous time series per channel | Whole-frame |
| resampling | Independent | Stateful/whole continuous time series per channel | Whole-frame |
| RMS trend, sound level | Independent | Window/overlap-sensitive; weighting can add filter state | Whole-frame |
| FFT, IFFT, cepstrum, lifter, spectral envelope, N-octave analysis/synthesis | Independent | Whole transform axis per channel | Whole-frame |
| STFT, ISTFT, Welch, spectrogram cepstrum, HPSS | Independent | Window/overlap-sensitive or full analysis-axis context | Whole-frame |
| loudness, roughness, sharpness | Independent | Standard algorithms require complete or overlapping per-channel context | Whole-frame |
| `add_with_snr` | Corresponding channels from two inputs | Whole time series for RMS scaling | Whole-frame; multi-input is outside the prototype |
| `sum`, `mean`, `channel_difference` | Cross-channel | Pointwise after combining channels | Existing cross-channel Dask graph |
| coherence, CSD, transfer function | Cross-channel | Window/overlap-sensitive cross-spectral analysis | Whole-frame |
| `custom` | Unknown by construction | User-defined | Whole-frame |

This classification does not authorize time chunking for filters, resampling, FFT,
STFT, Welch, psychoacoustic algorithms, or other continuity-sensitive transforms.
Those operations remain whole-signal per channel until they have an explicit state or
overlap contract.

## Prototype: RemoveDC

`RemoveDC` is unary, shape-preserving, and numerically independent across channels. It
is the first operation to use the channel-independent specialization. For known,
positive channel counts, each task still receives the complete time series for one
channel, so subtracting that channel's mean is identical to the previous whole-frame
kernel call. Indeterminate or inapplicable inputs retain the historical whole-frame
behavior.

The Frame boundary is unchanged: calibration factors are applied lazily before the
operation, consumed exactly once in output channel metadata, and channel IDs, labels,
units, references, extra metadata, source-time offsets, semantic lineage, and Recipe
replay continue through the existing construction path. Shape and dtype are calculated
before execution as before.

Unsupported and cross-channel operations retain the default graph builder. This
fail-safe default is also used by third-party `AudioOperation` subclasses, so the
prototype does not silently reinterpret existing kernels as channel-independent.

## Benchmark interpretation

The scalability benchmark accepts multiple values for `--channels` and runs both the
forced `whole-frame` baseline and the `channel-wise` prototype in isolated workers. It
computes the same `remove_dc` kernel and reports numerical evidence, graph task count,
operation compute time, absolute process peak RSS observed immediately after operation
execution, and the existing WDF metrics. Recipe extraction happens only after the
operation graph and operation-lifetime RSS measurements; `recipe_nodes` is therefore a
separate structural probe rather than part of the operation timing or allocation
window.

For a fixed sample count, compare rows with the same channel count and different
`execution_path` values, then compare increasing channel counts within each path. More
channel tasks and graph-building allocation are the expected tradeoff for avoiding one
kernel task that receives every channel. Timings and absolute RSS are meaningful only
for reruns on the same machine, Python environment, and dependency lock; the benchmark
does not define a platform-independent RSS ceiling.

## Prototype benchmark evidence

The issue-328 comparison used one fixed 480,000-sample signal per channel and increased
the channel count through 1, 2, 4, and 8. The base revision was run through the same
candidate benchmark harness as a bridge, selecting only `whole-frame`; the candidate
ran both paths. The complete candidate matrix was repeated once because timing and RSS
are environment-sensitive.

| Channels | Tasks, whole → channel | Operation peak RSS MB, whole → channel | Rerun RSS MB, whole → channel |
| ---: | ---: | ---: | ---: |
| 1 | 6 → 6 | 158.3 → 158.0 | 158.1 → 157.3 |
| 2 | 12 → 14 | 173.3 → 165.6 | 173.7 → 165.7 |
| 4 | 20 → 28 | 196.1 → 189.5 | 197.2 → 181.4 |
| 8 | 36 → 56 | 242.9 → 220.2 | 242.4 → 220.0 |

Every paired `output_l2_squared` value is exactly equal. At eight channels, the two
candidate runs reduced the observed operation-lifetime peak by about 22.4–22.7 MB
while adding 20 graph tasks. The four-channel RSS delta varied more between runs,
which is why these values characterize the observed tradeoff rather than define a
portable memory budget. Channel-wise compute time was lower for 2, 4, and 8 channels
in both runs, but timing remains descriptive and has no pass/fail threshold.

Environment and revisions:

- base: `e5c7c4f8a47e60fb79eef996d9595260579ea6c3`;
- candidate: `27939e9a8285e108cff0e6ea47ed936cb868bb02`;
- Linux `6.17.0-40-generic` x86-64, glibc 2.36;
- CPython 3.10.20, Clang 22.1.3;
- one shared virtual environment and lock file, `uv.lock` SHA-256
  `8f22e9d43bb9a4f1ec476219fb57464bd29929f8e7e30bc0d03c32f728414107`.

Commands (absolute checkout prefixes are shown because the bridge run intentionally
loaded the base library while executing the candidate harness):

```bash
PYTHONPATH=/workspaces/wandas /workspaces/wandas/.venv/bin/python \
  /workspaces/wandas/.worktrees/issue-328/scripts/scalability_benchmark.py \
  --channels 1 2 4 8 --samples 480000 --chunk-samples 480000 \
  --sampling-rate 48000 --execution-paths whole-frame

/workspaces/wandas/.venv/bin/python scripts/scalability_benchmark.py \
  --channels 1 2 4 8 --samples 480000 --chunk-samples 480000 \
  --sampling-rate 48000
```

The second command was run twice without concurrent benchmark activity. Material RSS
or timing differences must be rerun in the same environment and lock; cross-platform
absolute RSS comparisons are invalid. The committed raw reports are intentional
evidence artifacts:

- [base whole-frame JSON](../assets/benchmarks/issue-328/base-e5c7c4f8.json)
- [candidate JSON](../assets/benchmarks/issue-328/candidate-27939e9a.json)
- [candidate rerun JSON](../assets/benchmarks/issue-328/candidate-rerun-27939e9a.json)
