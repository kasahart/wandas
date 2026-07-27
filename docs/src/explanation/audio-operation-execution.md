# AudioOperation execution dependencies

Wandas separates storage chunking from numerical-kernel execution. Frame data is
normally chunked with one channel per Dask chunk, but that topology alone does not
make a delayed NumPy/SciPy kernel run once per channel. The operation's internal
graph-building specialization owns that decision.

Channel dependency and time dependency are independent dimensions. An operation can
be channel-independent while still requiring one complete, continuous time series for
each channel. Channel-wise execution therefore reduces the number of channels that a
kernel task materializes; it does not introduce time-axis distribution.

## Public extension contracts

Choose the base class from the numerical dependency, before considering the current
graph implementation:

- `AudioOperation` permits cross-channel dependencies and uses conservative
  whole-frame execution by default.
- `ChannelIndependentAudioOperation` declares that every output channel depends only
  on the corresponding input channel. Subclasses must preserve
  `op(all_channels) == concatenate(op(channel) for channel in each_channel)`.

The second contract is semantic. It does not promise one task per channel, a scheduler,
or a particular Dask graph. Its `_process()` implementation must remain correct when
given either one channel or a complete multi-channel tensor. A gain operation is
channel-independent; common-mode removal, which subtracts a mean across channels, is
cross-channel and must use `AudioOperation`.

## Internal graph-builder contract

`AudioOperation.process()` validates inputs and calculates output shape and dtype,
then delegates graph construction through one polymorphic internal hook:

- The base implementation is conservative whole-frame execution. One delayed kernel receives the complete
  channel-first tensor, preserving all existing operations and custom extensions.
- `ChannelIndependentAudioOperation` can build one delayed kernel call for each
  channel, where the kernel input retains shape `(1, ...)`, then concatenates the
  outputs along the channel axis.

The current optimization is deliberately narrow: channel-wise execution applies to
unary operations with a known positive channel count that preserve the channel axis.
Zero-channel, unknown-channel-count, multi-input, and channel-axis-changing inputs fall
back to the base whole-frame graph. The fallback changes execution topology, not the
public channel-independence contract.
New execution forms override the graph-building hook instead of adding cases to a
central dispatcher. The hook, Dask graph, chunks, scheduler, and xarray container
remain private implementation details; the public Frame workflow is unchanged.

## Built-in operation classification

The table classifies the registered numerical families. “Whole signal/axis” means the
operation needs the complete relevant axis for one channel; it does not mean that all
channels depend on one another.

| Built-in family | Channel dependency | Time or analysis-axis dependency | Current execution |
| --- | --- | --- | --- |
| `remove_dc` | Independent | Whole time series per channel for the mean | **Channel-wise** |
| `abs`, `power` | Independent | Pointwise/time-local | Existing Dask-native graph override |
| `normalize` | Parameter-dependent: a non-`None` norm over the last axis is independent; `axis=None` or a channel axis is cross-channel | Whole selected norm axis | **Channel-wise when eligible** |
| `trim`, `fix_length` | Independent | Indexed/padded time-local transform with output-shape change | Whole-frame |
| `fade` | Independent | Needs the full signal length to define the envelope | Whole-frame |
| high-pass, low-pass, band-pass | Independent | Stateful/whole continuous time series per channel | **Channel-wise** |
| A-weighting | Independent | Stateful/whole continuous time series per channel | Whole-frame |
| resampling | Independent | Whole time series per channel for the resampling transform | **Channel-wise** |
| RMS trend, sound level | Independent | Window/overlap-sensitive; weighting can add filter state | Whole-frame |
| FFT, IFFT, cepstrum, lifter, spectral envelope, N-octave analysis/synthesis | Independent | Whole transform axis per channel | Whole-frame |
| STFT, ISTFT, Welch, spectrogram cepstrum | Independent | Window/overlap-sensitive or full analysis-axis context | Whole-frame |
| HPSS harmonic, percussive | Independent | Whole time series per channel for the internal STFT, median filters, and inverse STFT | **Channel-wise** |
| loudness, roughness, sharpness | Independent | Standard algorithms require complete or overlapping per-channel context | Whole-frame |
| `add_with_snr` | Corresponding channels from two inputs | Whole time series for RMS scaling | Whole-frame; multi-input is outside the prototype |
| `sum`, `mean`, `channel_difference` | Cross-channel | Pointwise after combining channels | Existing cross-channel Dask graph |
| coherence, CSD, transfer function | Cross-channel | Window/overlap-sensitive cross-spectral analysis | Whole-frame |
| `custom` | Unknown by construction | User-defined | Whole-frame |

This classification does not authorize time chunking for filters, resampling, FFT,
STFT, Welch, psychoacoustic algorithms, or other continuity-sensitive transforms.
Those operations remain whole-signal per channel until they have an explicit state or
overlap contract.

## Adopted operation: RemoveDC

`RemoveDC` is unary, shape-preserving, and numerically independent across channels. It
uses `ChannelIndependentAudioOperation`. For known, positive channel counts, each task
still receives the complete time series for one channel, so subtracting that channel's
mean is identical to the whole-frame kernel call. Indeterminate or inapplicable inputs
retain whole-frame execution.

The Frame boundary is unchanged: calibration factors are applied lazily before the
operation, consumed exactly once in output channel metadata, and channel IDs, labels,
units, references, extra metadata, source-time offsets, semantic lineage, and Recipe
replay continue through the existing construction path. Shape and dtype are calculated
before execution as before.

Unsupported and cross-channel operations retain the default graph builder. Third-party
extensions remain whole-frame when they subclass `AudioOperation`; extensions that
choose `ChannelIndependentAudioOperation` explicitly accept and pass its independence
contract to their subclasses.

## Adopted family: Butterworth filters

The shared `_ButterworthFilter` kernel used by the high-pass, low-pass, and band-pass
operations applies `scipy.signal.filtfilt(..., axis=1)`. Each output row depends on one
complete input row but not on any other channel, so the family uses the public
`ChannelIndependentAudioOperation` contract without a new graph-builder mechanism.
Filter coefficients, output shape and `float64` dtype, Frame metadata, lineage, and
Recipe declarations are unchanged.

The final LowPass comparison for issue #343 used the normal materialization path:
`BaseFrame.data` calls `BaseFrame._compute()`, which calls Dask Array `compute()`
without overriding its default threaded scheduler. Eight channels with 1,000,000
float64 samples each ran whole-frame and channel-wise in alternating order across five
isolated processes per path, without explicitly setting the scheduler, worker count,
or native thread count. Median end-to-end time decreased from 121.1 ms to 52.1 ms
(56.9%), median process peak RSS decreased from 508.4 MiB to 469.9 MiB (7.6%), and
task count increased from 36 to 56. Numerical values, shape, dtype, chunks, Frame and
Recipe behavior, and fallback behavior were unchanged; focused tests require exact
array equality with forced whole-frame execution for all three filters. These
same-environment measurements explain the adoption decision, not a portable
performance guarantee.

## Adopted operation: ReSampling

`ReSampling` preserves channel count while changing the complete time-axis length and
sampling-rate metadata. Its polyphase or exact-length fallback kernel is independent
across channels, so each channel-wise task receives one complete time series. The
existing exact output-length calculation, dtype rules, sampling-rate update, Frame
time coordinates, source offsets, lineage, and Recipe declaration remain unchanged.

The issue #346 evaluation used 1,000,000 samples per channel at 48 kHz resampled to
16 kHz. At eight channels, tasks increased from 20 to 56, median compute time changed
from 0.0659 s to 0.0729 s, and same-environment median peak RSS decreased from about
299.5 MiB to 278.9 MiB. Every paired numerical checksum was equal. These measurements
describe the task/time/memory tradeoff and do not define a portable performance limit.

## Parameter-dependent operation: Normalize

`Normalize` is parameter-dependent, so it subclasses `AudioOperation`, not
`ChannelIndependentAudioOperation`. It owns its eligibility semantics rather than
extending the common graph builder with an operation-specific dependency language.
A non-`None` norm over the last axis reuses private generic channel-wise graph
mechanics. Global normalization (`axis=None`), a channel axis, `norm=None`, and invalid
or inapplicable configurations retain conservative whole-frame execution. Threshold,
fill, dtype, Frame metadata, lineage, and Recipe behavior remain unchanged.

The issue #348 evaluation used L2 normalization over 1,000,000 samples per channel. At
eight channels, tasks increased from 20 to 56, median compute time changed from about
0.0441 s to 0.0373 s, and same-environment median peak RSS decreased from about
393.4 MiB to 347.7 MiB. All paired arrays were exactly equal. These figures describe
the observed tradeoff rather than define a portable threshold.

## Adopted family: HPSS harmonic and percussive extraction

The shared `_HpssBase` kernel delegates to `librosa.effects.harmonic` or
`librosa.effects.percussive`. Librosa applies the transform independently to each
leading input row, and both Wandas operations preserve channel count, sample count,
and dtype. The family therefore uses `ChannelIndependentAudioOperation`: each
channel-wise task receives shape `(1, n_samples)` with the complete time series needed
for the internal STFT, median filtering, and inverse STFT. Scalar or tuple
`kernel_size` and `margin` values configure the time/frequency separation filters;
they are common operation parameters, not per-channel configuration. Zero or unknown
channel counts retain the generic whole-frame fallback; extra runtime inputs remain
rejected by the existing unary input contract.

The public Frame methods, optional-dependency check, librosa call, Frame metadata,
calibration consumption, lineage, and Recipe declarations are unchanged. Focused
tests compare both family members exactly with forced whole-frame execution and the
direct librosa authority for 1, 2, 4, and 8 float32 or float64 channels. Integer input
continues to raise librosa's existing `ParameterError`.

The formal 2026-07-27 comparison used base
`9d758ad82cd7fbc4a814d37b0a6ff094ab0eb9f8` and committed candidate
`e9ca186b2e2ecbc419374a146ce80da19777d691`. It measured eight float64 channels
with 96,000 samples each. Whole-frame and channel-wise paths ran in separate
processes for three runs per path, interleaved in the order base 1, candidate 1,
candidate 2, base 2, base 3, candidate 3 for each operation and boundary. Dask's
default scheduler was used without a scheduler override or reported native-thread
environment overrides. The parameters were
`kernel_size=31`, `power=2`, `margin=1`, `n_fft=1024`, `hop_length=256`,
`win_length=1024`, `window="hann"`, `center=True`, and `pad_mode="constant"`.

| Public `Frame.data` path | Tasks, whole → channel | Median graph build, whole → channel | Median materialization, whole → channel | Median process peak RSS, whole → channel |
| --- | ---: | ---: | ---: | ---: |
| harmonic | 36 → 56 | 0.00416 s → 0.00866 s | 1.2149 s → 0.5485 s | 488.5 MiB → 440.5 MiB |
| percussive | 36 → 56 | 0.00422 s → 0.00832 s | 1.2182 s → 0.5462 s | 488.4 MiB → 444.6 MiB |

At the direct operation boundary, harmonic tasks increased from 20 to 56 while
median graph build changed from 0.00105 s to 0.00527 s, median compute time changed
from 1.2044 s to 0.5273 s, and median process peak RSS changed from 488.3 MiB to
444.0 MiB. Percussive tasks likewise increased from 20 to 56 while median graph
build changed from 0.00098 s to 0.00529 s, median compute time changed from 1.2101 s
to 0.5308 s, and median process peak RSS changed from 487.6 MiB to 439.0 MiB. Across
all 24 workers, each operation had one output shape, dtype, SHA-256 checksum, and
squared-L2 value.

The deterministic input was created in memory, so these RSS figures cover the worker
process and operation materialization but do not characterize a file-reader boundary
or isolate only the librosa kernel's temporary arrays. The observation used Linux
`7.0.0-28-generic` x86-64 with glibc 2.36, CPython 3.10.20, and `uv.lock` SHA-256
`8f22e9d43bb9a4f1ec476219fb57464bd29929f8e7e30bc0d03c32f728414107`.
The orchestration command was:

```bash
bash /tmp/run-hpss-formal-benchmark.sh
```

Each worker expanded to the following exact command form, with the revision worktree,
operation, boundary, label, and run number recorded per case in the raw evidence:

```bash
PYTHONPATH=<revision-worktree> \
  uv run --no-sync --project /workspaces/wandas python \
  /tmp/wandas-channelwise-small-benchmark.py \
  --operation <operation> --boundary <boundary> \
  --channels 8 --samples 96000 --dtype float64 --label <label>
```

These same-environment figures support the adoption decision but are not a portable
timing, task-count, or memory guarantee. The revision-addressable
[formal raw JSON](../assets/benchmarks/hpss-channelwise/base-9d758ad8-candidate-e9ca186b.json)
contains all 24 expanded commands and measurements. It also embeds the exact worker
and orchestration script sources used for the run. Their recorded SHA-256 values are
`1a15b7e706e1a0acaf29d02b6cb8d2de8f239d9dd0ac555f575ebcf0eaf39103` and
`f70c7c338fc46e73dd8dbd42fcd96360054a9492fac1f80b014722c3fe1b6068`,
respectively.

The measured production candidate remains
`e9ca186b2e2ecbc419374a146ce80da19777d691`. Post-measurement review changes are
limited to tests, documentation, and the benchmark evidence asset; no production
source differs from that measured candidate.

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
| 1 | 6 → 6 | 158.1 → 158.0 | 157.1 → 157.9 |
| 2 | 12 → 14 | 172.6 → 165.6 | 173.1 → 165.4 |
| 4 | 20 → 28 | 195.6 → 181.3 | 195.7 → 181.2 |
| 8 | 36 → 56 | 241.2 → 230.0 | 242.7 → 227.8 |

Every paired `output_l2_squared` value is exactly equal. At eight channels, the two
candidate runs reduced the observed operation-lifetime peak by about 11.2–15.0 MB
while adding 20 graph tasks. At four channels, both runs reduced it by about 14.2–14.4
MB. These values characterize the observed tradeoff rather than define a portable
memory budget. Channel-wise compute time was lower for 4 and 8 channels in both runs;
the two-channel timing changed direction, so timing remains descriptive and has no
pass/fail threshold.

Environment and revisions:

- base: `e5c7c4f8a47e60fb79eef996d9595260579ea6c3`;
- candidate: `71ea46cabd747666b10a44991735a3c566caa7a3`;
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

PYTHONPATH=/workspaces/wandas/.worktrees/issue-328 \
  /workspaces/wandas/.venv/bin/python scripts/scalability_benchmark.py \
  --channels 1 2 4 8 --samples 480000 --chunk-samples 480000 \
  --sampling-rate 48000
```

The second command was run twice without concurrent benchmark activity. Material RSS
or timing differences must be rerun in the same environment and lock; cross-platform
absolute RSS comparisons are invalid. The committed raw reports are intentional
evidence artifacts:

- [base whole-frame JSON](../assets/benchmarks/issue-328/base-e5c7c4f8.json)
- [candidate JSON](../assets/benchmarks/issue-328/candidate-71ea46ca.json)
- [candidate rerun JSON](../assets/benchmarks/issue-328/candidate-rerun-71ea46ca.json)
