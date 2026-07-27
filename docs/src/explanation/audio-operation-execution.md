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

The scope snapshot is all 41 built-in registry entries reachable from the processing
modules at base `9d758ad82cd7fbc4a814d37b0a6ff094ab0eb9f8`, including `custom`,
which registers when `wandas.processing.custom` is imported. Every entry is classified
once: **A** means the existing direct contract is sufficient, **B** means the operation
owns parameter-dependent eligibility, **C** means a new contract is required, and
**D** means keep the current whole-frame or Dask-native path. Counts are **A=10,
B=1, C=4, D=26**.

“Execution at snapshot” describes merged `main`, not the adoption decision. Thus
RemoveDC, Butterworth, ReSampling, and eligible Normalize are already channel-wise
through merged PRs
[#339](https://github.com/kasahart/wandas/pull/339),
[#344](https://github.com/kasahart/wandas/pull/344),
[#347](https://github.com/kasahart/wandas/pull/347), and
[#349](https://github.com/kasahart/wandas/pull/349). The public semantic extension
contract was merged in [#352](https://github.com/kasahart/wandas/pull/352).
Review-ready but unmerged PRs remain whole-frame in this snapshot.

| Registered operation / implementation | Class | Semantic and measured evidence | Execution at snapshot | Decision / tracking |
| --- | --- | --- | --- | --- |
| `highpass_filter` / `HighPassFilter` | A | Shared Butterworth `filtfilt(axis=1)` needs one complete row and never another channel; exact family parity. | **Channel-wise** | Adopted with issue [#343](https://github.com/kasahart/wandas/issues/343) / PR [#344](https://github.com/kasahart/wandas/pull/344). |
| `lowpass_filter` / `LowPassFilter` | A | Same shared kernel. The 8-channel default-threaded representative changed 36→56 tasks, 121.1→52.1 ms, and 508.4→469.9 MiB. | **Channel-wise** | Adopted with issue [#343](https://github.com/kasahart/wandas/issues/343) / PR [#344](https://github.com/kasahart/wandas/pull/344). |
| `bandpass_filter` / `BandPassFilter` | A | Same coefficients, shape/dtype contract, whole-row state, and exact per-channel equivalence as the Butterworth family. | **Channel-wise** | Adopted with issue [#343](https://github.com/kasahart/wandas/issues/343) / PR [#344](https://github.com/kasahart/wandas/pull/344). |
| `remove_dc` / `RemoveDC` | A | Subtracts each row's own time mean. The original 8-channel prototype retained exact L2² while reducing same-environment peak RSS with bounded task growth. | **Channel-wise** | Adopted in PR [#339](https://github.com/kasahart/wandas/pull/339). |
| `resampling` / `ReSampling` | A | Unary complete-row resampling preserves channel count while changing time length and sampling-rate metadata; exact parity. | **Channel-wise** | Adopted with issue [#346](https://github.com/kasahart/wandas/issues/346) / PR [#347](https://github.com/kasahart/wandas/pull/347). |
| `a_weighting` / `AWeighting` | A | One shared filter and complete time axis per channel; exact parity. Default-threaded materialization improved with an explained concurrency/RSS trade-off. | Whole-frame | Review-ready PR [#356](https://github.com/kasahart/wandas/pull/356); do not describe as merged. |
| `trim` / `Trim` | A | Final-axis slicing is independent and channel-preserving. The candidate also aligns shape and source-time provenance with NumPy slice normalization; 8-channel `Frame.data` improved 25.1→15.3 ms with flat RSS. | Whole-frame | Review-ready PR [#357](https://github.com/kasahart/wandas/pull/357). |
| `hpss_harmonic` / `HpssHarmonic` | A | Librosa HPSS uses the trailing complete time axis; leading channels are independent across scalar/tuple/list settings. Exact parity; 8-channel `Frame.data` improved about 1.21→0.55 s and 488.5→440.5 MiB. | Whole-frame | Review-ready family PR [#358](https://github.com/kasahart/wandas/pull/358). |
| `hpss_percussive` / `HpssPercussive` | A | Same shared HPSS base and independence proof; exact parity. The representative changed about 1.22→0.55 s and 488.4→444.6 MiB. | Whole-frame | Review-ready family PR [#358](https://github.com/kasahart/wandas/pull/358). |
| `noct_spectrum` / `NOctSpectrum` | A | MoSQITo analyzes signal columns independently. Reshaping with the known input-channel and returned-frequency counts preserves single/zero-band shapes, including `(0, 1)` and `(0, 0)`; exact direct/whole/channel parity. The 8-channel `Frame.data` candidate changed 36→56 tasks, 0.2636→0.1270 s, and 270.9→257.2 MiB. | Whole-frame | Review-ready PR [#359](https://github.com/kasahart/wandas/pull/359); do not describe as merged. |
| `normalize` / `Normalize` | B | A non-`None` norm over the last axis is independent; global/channel-axis norms and `norm=None` are not eligible. The operation owns this predicate. | **Channel-wise when eligible** | Adopted with issue [#348](https://github.com/kasahart/wandas/issues/348) / PR [#349](https://github.com/kasahart/wandas/pull/349). |
| `add_with_snr` / `AddWithSNR` | C | The kernel is row-independent only after corresponding-input pairing or declared mono broadcast. A prototype was exact and favorable, but the unary helper intentionally rejects runtime inputs. | Whole-frame | Defer to multi-input binding issue [#351](https://github.com/kasahart/wandas/issues/351). |
| `rms_trend` / `RmsTrend` | C | Windowed RMS is row-local, but `dB=True` can bind one reference per original channel; a shared operation cannot slice that immutable config today. | Whole-frame | Defer to per-channel configuration issue [#350](https://github.com/kasahart/wandas/issues/350). |
| `sound_level` / `SoundLevel` | C | A/C/Z and Fast/Slow state are row-local, but public calibrated Frames supply per-channel references. Scalar-reference parity is insufficient to preserve that path. | Whole-frame | Defer to per-channel configuration issue [#350](https://github.com/kasahart/wandas/issues/350). |
| `roughness_dw_spec` / `RoughnessDwSpec` | C | MoSQITo is row-independent, but a one-channel kernel returns `(bark, time)` while multi-channel returns `(channel, bark, time)`. The helper has no semantic shape adapter for restoring the leading channel axis. | Whole-frame | Defer until an explicit operation-owned shape-adapter contract exists; do not alter shared `_NOctBase`-style mechanics ad hoc. |
| `abs` / `ABS` | D | Pointwise and already implemented as a Dask-native array graph; wrapping it in delayed channel kernels adds no useful boundary. | Dask-native | Keep current path. |
| `power` / `Power` | D | Pointwise and already Dask-native; helper migration would add tasks and materialization overhead. | Dask-native | Keep current path. |
| `sum` / `Sum` | D | Reduces across the channel axis and is intrinsically cross-channel. | Cross-channel Dask-native | Keep current path. |
| `mean` / `Mean` | D | Reduces across the channel axis and is intrinsically cross-channel. | Cross-channel Dask-native | Keep current path. |
| `channel_difference` / `ChannelDifference` | D | Each output depends on multiple input channels. | Cross-channel Dask-native | Keep current path. |
| `coherence` / `Coherence` | D | Pairwise cross-spectral output depends on channel pairs and complete window/overlap context. | Whole-frame | Keep current path; unary channel execution is semantically invalid. |
| `csd` / `CSD` | D | Pairwise cross-spectral density is intrinsically cross-channel. | Whole-frame | Keep current path. |
| `transfer_function` / `TransferFunction` | D | Cross-spectral numerator/denominator bind channel pairs. | Whole-frame | Keep current path. |
| `custom` / `CustomOperation` | D | User code has unknown channel dependencies, shape behavior, inputs, and state. | Whole-frame | Preserve the conservative third-party-compatible default. |
| `loudness_zwst` / `LoudnessZwst` | D | The public method intentionally materializes and returns a 1-D NumPy scalar-per-channel result, outside lazy Frame/Recipe output metadata. The NumPy kernel already loops per channel. | Eager public scalar path; operation default is whole-frame | Keep; changing the public result/metadata contract is out of scope. |
| `sharpness_din_st` / `SharpnessDinSt` | D | Same eager scalar-per-channel public boundary as steady loudness, with no lazy Frame/Recipe result to improve. | Eager public scalar path; operation default is whole-frame | Keep; public result ownership must not change here. |
| `stft` / `STFT` | D | Numerically expressible and exact, but issue #345 measured 8-channel compute at 0.1193→0.3773 s for only about 5.3 MiB RSS reduction; smaller channel counts regressed in RSS. | Whole-frame | Keep per the closed non-adoption decision [#345](https://github.com/kasahart/wandas/issues/345). |
| `fft` / `FFT` | D | The 8-channel prototype increased direct RSS 485.8→691.4 MiB and did not improve direct compute; public RSS also increased to 692.6 MiB. | Whole-frame | Keep; output/temporary growth outweighs task parallelism. |
| `ifft` / `IFFT` | D | The 8-channel prototype changed direct RSS 478.2→707.1 MiB and compute 0.1515→0.1741 s; the public path also regressed. | Whole-frame | Keep. |
| `istft` / `ISTFT` | D | The public prototype reduced RSS but changed compute 0.0961→0.3325 s; the large unexplained runtime regression is not acceptable. | Whole-frame | Keep. |
| `welch` / `Welch` | D | Window/overlap analysis changes the trailing axis, and the prototype did not preserve strict array equality for supported `detrend="linear"` and `"median"` settings. No normal-materialization benefit was established to justify relaxing that numerical contract. | Whole-frame | Keep until materially different evidence exists. |
| `cepstrum` / `Cepstrum` | D | The 8-channel public prototype improved compute modestly but increased RSS 699.6→750.6 MiB; direct RSS increased 623.1→710.2 MiB. | Whole-frame | Keep. |
| `lifter` / `Lifter` | D | Supplemental 8-channel public timing improved 0.0425→0.0291 s, but RSS increased 421.2→482.1 MiB; direct timing was effectively flat. | Whole-frame | Keep; material memory regression outweighs the public timing observation. |
| `spectral_envelope` / `SpectralEnvelope` | D | The 8-channel public prototype improved 0.1922→0.1428 s but increased RSS 543.2→604.5 MiB; direct RSS rose about 122 MiB. | Whole-frame | Keep. |
| `spectrogram_cepstrum` / `SpectrogramCepstrum` | D | Rank-3 analysis input/output requires the complete transform axis; no representative normal-materialization benefit was established. | Whole-frame | Keep; correctness alone does not justify graph growth. |
| `noct_synthesis` / `NOctSynthesis` | D | The 8-channel public prototype changed compute 0.3342→0.9286 s and RSS 308.9→399.4 MiB. | Whole-frame | Keep; spectrum adoption does not change shared synthesis behavior. |
| `fix_length` / `FixLength` | D | Padding/slicing is cheap. The 8-channel `Frame.data` prototype improved 0.0461→0.0286 s but increased RSS 417.2→478.8 MiB. | Whole-frame | Keep; the memory regression outweighs this small transform's timing. |
| `fade` / `Fade` | D | Multiplication by one shared full-length envelope is cheap. The 8-channel `Frame.data` prototype improved 0.0539→0.0400 s but increased RSS 409.3→569.8 MiB. | Whole-frame | Keep. |
| `loudness_zwtv` / `LoudnessZwtv` | D | Exact row-wise execution was about 2.1× slower on the 8-channel public path and increased RSS 172.6→207.4 MiB. | Whole-frame | Keep; optional MoSQITo call/task overhead dominates. |
| `roughness_dw` / `RoughnessDw` | D | The 8-channel public prototype was time-neutral (about 0.451→0.453 s) while RSS increased 181.2→245.2 MiB. | Whole-frame | Keep. |
| `sharpness_din` / `SharpnessDin` | D | Exact row-wise execution was about 2× slower on the 8-channel public path and increased RSS 173.1→207.7 MiB. | Whole-frame | Keep. |

Prototype timings and RSS above are 2026-07-27 same-environment observations from
isolated workers using one dependency lock; sample counts and kernels differ by
family. They support within-family decisions only and are not portable thresholds or
cross-operation rankings. The psychoacoustic rows also preserve two distinct public
boundaries: time-varying metrics return lazy Frames with changed analysis metadata,
whereas steady-state metrics eagerly return NumPy scalars. `RoughnessDwSpec` additionally
needs a rank-aware channel adapter before its `RoughnessFrame` metadata can be
constructed safely.

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
