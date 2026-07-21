# AudioOperation execution dependencies

Wandas separates storage chunking from numerical-kernel execution. Frame data is
normally chunked with one channel per Dask chunk, but that topology alone does not
make a delayed NumPy/SciPy kernel run once per channel. The operation's internal
execution strategy owns that decision.

Channel dependency and time dependency are independent dimensions. An operation can
be channel-independent while still requiring one complete, continuous time series for
each channel. Channel-wise execution therefore reduces the number of channels that a
kernel task materializes; it does not introduce time-axis distribution.

## Internal strategy contract

`AudioOperation` has two internal strategies:

- `whole-frame` is the conservative default. One delayed kernel receives the complete
  channel-first tensor, preserving all existing operations and custom extensions.
- `channel-wise` builds one delayed kernel call for each channel, where the kernel input
  retains shape `(1, ...)`, then concatenates the outputs along the channel axis.

The prototype contract is deliberately narrow: channel-wise execution currently
supports unary operations that preserve the channel axis. A contradictory declaration
fails while the graph is built. The strategy, Dask graph, chunks, scheduler, and xarray
container remain private implementation details; the public Frame workflow is
unchanged.

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
is the first operation to select `channel-wise`. Each task still receives the complete
time series for one channel, so subtracting that channel's mean is identical to the
previous whole-frame kernel call.

The Frame boundary is unchanged: calibration factors are applied lazily before the
operation, consumed exactly once in output channel metadata, and channel IDs, labels,
units, references, extra metadata, source-time offsets, semantic lineage, and Recipe
replay continue through the existing construction path. Shape and dtype are calculated
before execution as before.

Unsupported and cross-channel operations retain the default strategy. This fail-safe
default is also used by third-party `AudioOperation` subclasses, so the prototype does
not silently reinterpret existing kernels as channel-independent.

## Benchmark interpretation

The scalability benchmark accepts multiple values for `--channels` and runs both the
forced `whole-frame` baseline and the `channel-wise` prototype in isolated workers. It
computes the same `remove_dc` kernel and reports numerical evidence, graph task count,
operation compute time, absolute process peak RSS observed immediately after operation
execution, and the existing WDF metrics.

For a fixed sample count, compare rows with the same channel count and different
`execution_path` values, then compare increasing channel counts within each path. More
channel tasks and graph-building allocation are the expected tradeoff for avoiding one
kernel task that receives every channel. Timings and absolute RSS are meaningful only
for reruns on the same machine, Python environment, and dependency lock; the benchmark
does not define a platform-independent RSS ceiling.

