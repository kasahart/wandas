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
modules at integrated base `edd9d253280ff548e62263acd6ec3d7597e4c2f2`, including `custom`,
which registers when `wandas.processing.custom` is imported. Every entry is classified
once: **A** means the existing direct contract is sufficient, **B** means the operation
owns parameter-dependent eligibility, **C** means a new contract is required, and
**D** means keep the current whole-frame or Dask-native path. Counts are **A=9,
B=1, C=4, D=27**.

“Execution at snapshot” describes merged `main`, not the adoption decision. Thus
RemoveDC, Butterworth, ReSampling, A-weighting, HPSS harmonic/percussive extraction,
N-octave spectrum, and eligible Normalize are already channel-wise through merged PRs
[#339](https://github.com/kasahart/wandas/pull/339),
[#344](https://github.com/kasahart/wandas/pull/344),
[#347](https://github.com/kasahart/wandas/pull/347), and
[#349](https://github.com/kasahart/wandas/pull/349), followed by
[#356](https://github.com/kasahart/wandas/pull/356),
[#358](https://github.com/kasahart/wandas/pull/358), and
[#359](https://github.com/kasahart/wandas/pull/359). The public semantic extension
contract was merged in [#352](https://github.com/kasahart/wandas/pull/352).

| Registered operation / implementation | Class | Semantic and measured evidence | Execution at snapshot | Decision / tracking |
| --- | --- | --- | --- | --- |
| `highpass_filter` / `HighPassFilter` | A | Shared Butterworth `filtfilt(axis=1)` needs one complete row and never another channel; exact family parity. | **Channel-wise** | Adopted with issue [#343](https://github.com/kasahart/wandas/issues/343) / PR [#344](https://github.com/kasahart/wandas/pull/344). |
| `lowpass_filter` / `LowPassFilter` | A | Same shared kernel. The 8-channel default-threaded representative changed 36→56 tasks, 121.1→52.1 ms, and 508.4→469.9 MiB. | **Channel-wise** | Adopted with issue [#343](https://github.com/kasahart/wandas/issues/343) / PR [#344](https://github.com/kasahart/wandas/pull/344). |
| `bandpass_filter` / `BandPassFilter` | A | Same coefficients, shape/dtype contract, whole-row state, and exact per-channel equivalence as the Butterworth family. | **Channel-wise** | Adopted with issue [#343](https://github.com/kasahart/wandas/issues/343) / PR [#344](https://github.com/kasahart/wandas/pull/344). |
| `remove_dc` / `RemoveDC` | A | Subtracts each row's own time mean. The original 8-channel prototype retained exact L2² while reducing same-environment peak RSS with bounded task growth. | **Channel-wise** | Adopted in PR [#339](https://github.com/kasahart/wandas/pull/339). |
| `resampling` / `ReSampling` | A | Unary complete-row resampling preserves channel count while changing time length and sampling-rate metadata; exact parity. | **Channel-wise** | Adopted with issue [#346](https://github.com/kasahart/wandas/issues/346) / PR [#347](https://github.com/kasahart/wandas/pull/347). |
| `a_weighting` / `AWeighting` | A | One shared filter and complete time axis per channel; exact parity. Default-threaded materialization improved with an explained concurrency/RSS trade-off. | **Channel-wise** | Adopted in PR [#356](https://github.com/kasahart/wandas/pull/356). |
| `hpss_harmonic` / `HpssHarmonic` | A | Librosa HPSS uses the trailing complete time axis; leading channels are independent across scalar/tuple/list settings. Exact parity; 8-channel `Frame.data` improved about 1.21→0.55 s and 488.5→440.5 MiB. | **Channel-wise** | Adopted in family PR [#358](https://github.com/kasahart/wandas/pull/358). |
| `hpss_percussive` / `HpssPercussive` | A | Same shared HPSS base and independence proof; exact parity. The representative changed about 1.22→0.55 s and 488.4→444.6 MiB. | **Channel-wise** | Adopted in family PR [#358](https://github.com/kasahart/wandas/pull/358). |
| `noct_spectrum` / `NOctSpectrum` | A | MoSQITo analyzes signal columns independently. Reshaping with the known input-channel and returned-frequency counts preserves single/zero-band shapes, including `(0, 1)` and `(0, 0)`; exact direct/whole/channel parity. The 8-channel `Frame.data` candidate changed 36→56 tasks, 0.2636→0.1270 s, and 270.9→257.2 MiB. | **Channel-wise** | Adopted in PR [#359](https://github.com/kasahart/wandas/pull/359). |
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
| `trim` / `Trim` | D | The public `Frame.trim()` path is now a structural Dask-native time-axis slice. Only the deprecated direct processing class remains registered and whole-frame for released Recipe compatibility. | Deprecated whole-frame operation; public path is Dask-native | Keep the compatibility operation through its deprecation period; issue [#357](https://github.com/kasahart/wandas/issues/357) was completed by PR [#361](https://github.com/kasahart/wandas/pull/361). |
| `loudness_zwst` / `LoudnessZwst` | D | The public method intentionally materializes and returns a 1-D NumPy scalar-per-channel result, outside lazy Frame/Recipe output metadata. The NumPy kernel already loops per channel. | Eager public scalar path; operation default is whole-frame | Keep; changing the public result/metadata contract is out of scope. |
| `sharpness_din_st` / `SharpnessDinSt` | D | Same eager scalar-per-channel public boundary as steady loudness, with no lazy Frame/Recipe result to improve. | Eager public scalar path; operation default is whole-frame | Keep; public result ownership must not change here. |
| `stft` / `STFT` | D | Numerically expressible and exact, but issue #345 measured 8-channel compute at 0.1193→0.3773 s for only about 5.3 MiB RSS reduction; smaller channel counts regressed in RSS. | Whole-frame | Keep per the closed non-adoption decision [#345](https://github.com/kasahart/wandas/issues/345). |
| `fft` / `FFT` | D | Formal 8-channel `Frame._compute` materialization improved 187.3→163.9 ms, but peak RSS increased 547.7→753.4 MiB. | Whole-frame | Keep; output/temporary growth outweighs the timing observation. |
| `ifft` / `IFFT` | D | Formal 8-channel `Frame._compute` materialization improved 199.6→180.8 ms, but peak RSS increased 562.9→791.4 MiB. | Whole-frame | Keep; the material memory regression is not acceptable. |
| `istft` / `ISTFT` | D | Formal 8-channel `Frame._compute` materialization regressed 98.6→203.5 ms with essentially flat peak RSS (394.6→394.9 MiB). | Whole-frame | Keep. |
| `welch` / `Welch` | D | Window/overlap analysis changes the trailing axis. `detrend="constant"` is strictly equal with both `average="mean"` and `"median"`, but supported `detrend="linear"` differs at machine precision for 2/4/8 channels and therefore fails the current strict-parity contract. | Whole-frame | Keep until materially different evidence exists. |
| `cepstrum` / `Cepstrum` | D | Formal 8-channel `Frame._compute` materialization improved 344.1→270.0 ms, but peak RSS increased 684.5→787.9 MiB. | Whole-frame | Keep; the memory regression outweighs the timing improvement. |
| `lifter` / `Lifter` | D | Formal 8-channel `Frame._compute` materialization improved 35.0→28.4 ms, but peak RSS increased 386.9→448.7 MiB. | Whole-frame | Keep; material memory regression outweighs the small timing observation. |
| `spectral_envelope` / `SpectralEnvelope` | D | Formal 8-channel `Frame._compute` materialization improved 223.5→141.4 ms, but peak RSS increased 555.0→677.4 MiB. | Whole-frame | Keep. |
| `spectrogram_cepstrum` / `SpectrogramCepstrum` | D | Formal 8-channel `Frame._compute` materialization improved 100.6→47.2 ms, but peak RSS increased 498.4→586.3 MiB. | Whole-frame | Keep; the rank-3 timing benefit does not offset the material memory regression. |
| `noct_synthesis` / `NOctSynthesis` | D | Formal 8-channel `Frame._compute` materialization regressed 1.431→5.728 s and peak RSS increased 428.8→867.4 MiB. | Whole-frame | Keep; spectrum adoption does not change shared synthesis behavior. |
| `fix_length` / `FixLength` | D | Padding/slicing is cheap. Formal 8-channel `Frame._compute` materialization improved 30.2→25.4 ms, but peak RSS increased 387.6→449.1 MiB. | Whole-frame | Keep; the memory regression outweighs this small transform's timing. |
| `fade` / `Fade` | D | Multiplication by one shared full-length envelope is cheap. Formal 8-channel `Frame._compute` materialization regressed 33.8→38.5 ms while peak RSS increased 409.5→570.0 MiB. | Whole-frame | Keep. |
| `loudness_zwtv` / `LoudnessZwtv` | D | Formal 8-channel `Frame._compute` materialization regressed 6.014→9.937 s and peak RSS increased 232.5→486.8 MiB. | Whole-frame | Keep; optional MoSQITo call/task overhead dominates. |
| `roughness_dw` / `RoughnessDw` | D | Formal 8-channel `Frame._compute` materialization regressed 1.375→1.523 s and peak RSS increased 188.6→254.7 MiB. | Whole-frame | Keep. |
| `sharpness_din` / `SharpnessDin` | D | Formal 8-channel `Frame._compute` materialization regressed 6.088→9.834 s and peak RSS increased 229.2→485.5 MiB. | Whole-frame | Keep. |

The performance-based D rows are backed by one revision-addressable
[formal evidence asset](../assets/benchmarks/channelwise-inventory-decisions/base-edd9d253-adapter.json).
It records the exact worker/orchestrator source, source and lock identities,
environment, commands, run order, parity matrix, and every isolated raw observation.
Measurements support within-family decisions only and are not portable thresholds or
cross-operation rankings. The psychoacoustic rows also preserve two distinct public
boundaries: time-varying metrics return lazy Frames with changed analysis metadata,
whereas steady-state metrics eagerly return NumPy scalars. `RoughnessDwSpec` additionally
needs a rank-aware channel adapter before its `RoughnessFrame` metadata can be
constructed safely.

`Frame.trim()` does not execute through a registered numerical operation. It is a
structural, Dask-native time-axis slice that shares the Frame indexing contract. It
preserves channel calibration and descriptors, advances `source_time_offset` to the
first selected sample, and records `wandas.frame.time_slice` version 1 so the
time-valued bounds are recomputed for each Recipe runtime input.
Previously saved `wandas.audio.trim` version 1 plans retain their released
array-operation replay contract.
The legacy `wandas.processing.Trim` class and `trim` registry key remain available
with a deprecation warning for one feature-release compatibility period; Frame
execution does not use them.

`wandas.audio.rms_trend` and `wandas.audio.sound_level` also retain exact Recipe
version 1 replay handlers. Those handlers preserve the released direct
square/divide arithmetic and physical-unit metadata. Public calls now capture
operation version 2, whose dB-only kernels use range-safe logarithmic ordering and
whose output metadata names the original reference. Both versions keep the same
whole-frame Dask execution boundary; the linear version 2 paths are unchanged.

This classification does not authorize time chunking for filters, resampling, FFT,
STFT, Welch, psychoacoustic algorithms, or other continuity-sensitive transforms.
Those operations remain whole-signal per channel until they have an explicit state or
overlap contract.

### Acoustic quantity boundary

The whole-frame classification above is independent of the returned quantity.
`RmsTrend(dB=False)` returns linear windowed RMS; `dB=True` returns
`20 log10(RMS/ref)`. `SoundLevel` first applies A/C/Z frequency weighting and
then a first-order exponential smoother to squared samples, using 125 ms for
Fast or 1 s for Slow. Its linear result is the square root of smoothed power;
its dB result is `10 log10(smoothed_power/ref²)`.

The dB paths return finite lower bounds for silence: `RmsTrend` floors the
amplitude ratio at `1e-12` (-240 dB), and `SoundLevel` floors the smoothed-power
ratio at `1e-20` (-200 dB).

Calibration is applied lazily at the Frame boundary before either operation.
Pa-domain input with a `2e-5 Pa` channel reference yields dB SPL. A reference of
1 yields relative dB re 1 input unit and must not be labeled dB SPL. Frequency
weighting, time weighting, and reference conversion are independent parts of
the contract.

The repository verifies formula-level frequency response, steady-state power,
and discrete-time Fast/Slow step response. It does not validate the full
tolerance, detector, calibration, environmental, or directional-response
requirements of an IEC/JIS sound-level meter. Operation display names such as
`LAF` identify selected implementation parameters, not instrument certification.

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

## Adopted operation: AWeighting

`AWeighting` applies the same second-order-section filter independently along the
complete time axis of each channel. Its output preserves the leading channel count
and always has `float64` dtype. The operation therefore uses
`ChannelIndependentAudioOperation`; each eligible kernel receives one complete
channel. Zero or unknown channel counts retain the conservative whole-frame fallback,
while an extra runtime input is rejected. The numerical kernel, public Frame method,
calibration consumption, metadata, lineage, and Recipe declaration are unchanged.

The revision-addressable adoption evaluation compared base
`9d758ad82cd7fbc4a814d37b0a6ff094ab0eb9f8` with candidate
`100955fe3f7c693038bc54721f1cf5d00ea6211a`. It used eight channels with
1,000,000 float64 samples per channel at 48 kHz. For each boundary, separate
processes ran in the interleaved order `base1`, `candidate1`, `candidate2`, `base2`,
`base3`, `candidate3`. No scheduler, worker-count, or native-thread override was
set, so normal `Frame.data` materialization used the default threaded scheduler.

| Boundary | Tasks, base → candidate | Median build, ms | Median materialization, ms | Median peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
| Operation | 20 → 56 | 0.959 → 5.094 | 68.148 → 37.936 | 448.066 → 508.996 |
| `Frame.data` | 36 → 56 | 4.207 → 7.947 | 74.915 → 41.288 | 448.574 → 509.348 |

All 12 outputs had shape `(8, 1000000)`, `float64` dtype, and exact SHA-256
checksum
`829133cbe9536fefe7fd21e68b06dcc92d170cee17d1e404a413041917541504`.
The exact expanded worker commands and every raw observation are recorded in the
[base report](../assets/benchmarks/a-weighting-channelwise/base-9d758ad8.json)
and
[candidate report](../assets/benchmarks/a-weighting-channelwise/candidate-100955fe.json).
The orchestration command was:

```bash
bash /tmp/run-wandas-a-weighting-formal-benchmark.sh
```

The RSS observation includes the resident in-memory source, concurrent filter
temporaries, and final NumPy output; it is not a one-channel memory claim. The
reproducible normal-materialization speedup justified adoption despite that measured
memory tradeoff. These observations were made on Linux 7.0.0-28-generic x86-64 with
glibc 2.36, CPython 3.10.20 from `/workspaces/wandas/.venv/bin/python3`, and
`uv.lock` SHA-256
`8f22e9d43bb9a4f1ec476219fb57464bd29929f8e7e30bc0d03c32f728414107`; they are not
portable performance guarantees.

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

## Adopted operation: N-octave spectrum

`NOctSpectrum` passes the complete time axis to MoSQITo and preserves the leading
channel count while changing the second axis to fractional-octave bands. MoSQITo
analyzes each signal column independently, so the Wandas kernel satisfies the direct
channel-independent contract for its shared `fmin`, `fmax`, `n`, `G`, and `fr`
configuration. Each channel task therefore receives shape `(1, n_samples)` and the
complete time axis. `NOctSynthesis` continues to use conservative whole-frame
execution; the spectrum adoption does not change the shared `_NOctBase`, synthesis
behavior, or the common graph helper.

The numerical result is RMS amplitude in each band, in the calibrated input unit;
its level uses `20 log10(band_rms / channel_ref)`. `G` selects MoSQITo's exact
center-frequency ratio convention (base 10 or base 2) and is not a decibel gain.
The complete quantity and reference definitions are in the
[spectral numerical contracts](spectral-numerical-contracts.md).

MoSQITo returns `float64` N-octave spectra for supported integer, `float32`, and
`float64` inputs. `NOctSpectrum` now advertises that actual dtype directly instead of
inheriting input dtype metadata. The correction is local to the spectrum operation.
Focused tests require exact equality among channel-wise execution, forced whole-frame
execution, and direct MoSQITo output for 1, 2, 4, and 8 channels, both `n=1` and `n=3`,
and all three input dtype families. They also cover MoSQITo's scalar/one-dimensional
return for a single requested band, preserving `(channels, 1)` by normalizing with the
known input channel count and returned frequency count. The same reshape preserves a
zero-channel single-band result as `(0, 1)` and a zero-channel empty-band result as
`(0, 0)`. A direct operation call with an empty band range preserves the exact MoSQITo
empty result as `(channels, 0)`; the public Frame path continues to reject
`fmin > fmax` during lazy Frame construction, before materializing samples, at the
existing `NOctFrame` validation boundary. Invalid
octave bases and denominators retain MoSQITo's graph-time exception, and bands above
the supported Nyquist design range retain its kernel-time exception, with exact type
and message parity on both execution paths. The existing optional-dependency failure,
whole-frame fallbacks, lazy `ChannelFrame` to `NOctFrame` transition, calibration
consumption, metadata, axes, lineage, and Recipe round trip remain unchanged.

The formal 2026-07-27 comparison used base
`9d758ad82cd7fbc4a814d37b0a6ff094ab0eb9f8` and committed candidate
`1666ab175bd489b2d3896435e796f9d4354d2fee`. It measured 240,000 float64 samples per
channel at both 4 and 8 channels. The direct operation and public `Frame.data`
boundaries each ran in fresh, serial worker processes, with three runs per revision in
the interleaved order base 1, candidate 1, candidate 2, base 2, base 3, candidate 3.
Dask's default threaded scheduler was used without a scheduler or native-thread
environment override.

| Boundary and channels | Tasks, base → candidate | Median graph build, base → candidate | Median materialization, base → candidate | Median peak RSS, base → candidate |
| --- | ---: | ---: | ---: | ---: |
| direct operation, 4 | 12 → 28 | 0.1827 s → 0.1822 s | 0.1279 s → 0.0620 s | 218.9 MiB → 212.9 MiB |
| direct operation, 8 | 20 → 56 | 0.1796 s → 0.1860 s | 0.2680 s → 0.1175 s | 270.2 MiB → 259.4 MiB |
| public `Frame.data`, 4 | 20 → 28 | 0.1823 s → 0.1837 s | 0.1249 s → 0.0545 s | 218.7 MiB → 211.7 MiB |
| public `Frame.data`, 8 | 36 → 56 | 0.1852 s → 0.1866 s | 0.2636 s → 0.1270 s | 270.9 MiB → 257.2 MiB |

Every base/candidate pair had exactly the same shape, dtype, SHA-256 checksum, and
float64 squared-L2 value. The eight-channel output had shape `(8, 19)`, `float64`
dtype, 1,216 bytes, checksum
`4fc3a40e416eff5e562e1f22b59161ac5df46c74e7a1628c894292c1ea8a90f0`, and
squared-L2 value `4.1062500944182165`. The deterministic input was created in memory,
so RSS includes the complete worker and materialization boundary; it does not
characterize a file reader or isolate only MoSQITo temporary allocations.

The orchestration command was:

```bash
bash /tmp/run-noct-spectrum-formal-benchmark.sh
```

Every expanded worker command, all 24 raw cases, the exact worker and orchestrator
source, and their SHA-256 hashes are stored in the
[formal raw JSON](../assets/benchmarks/noct-spectrum-channelwise/base-9d758ad8-candidate-1666ab17.json).
The orchestrator checked both source worktrees were clean and at their expected
commits before starting; every worker independently reported the same actual and
expected revision and a clean source status.
The shared environment was Linux `7.0.0-28-generic` x86-64 with glibc 2.36,
CPython 3.10.20, NumPy 2.2.6, SciPy 1.15.3, Dask 2025.11.0, MoSQITo 1.2.1, and
`uv.lock` SHA-256
`8f22e9d43bb9a4f1ec476219fb57464bd29929f8e7e30bc0d03c32f728414107`.
These measurements explain the same-environment adoption decision; they do not define
a portable timing, task-count, or memory guarantee.

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
