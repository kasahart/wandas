# Spectral numerical contracts / スペクトル数値契約

Wandas spectral Frames carry amplitude quantities unless an API explicitly says
otherwise. Calibration is applied before analysis, so `unit` below means the
linear measurement unit of the input channel. Canonically decoded audio has the
explicit unit `FS`; a generic empty-unit Frame is dimensionless but is not
inferred to be full scale.

| Result | Stored value and normalization | Unit | Level conversion |
| --- | --- | --- | --- |
| `ChannelFrame.fft()` | Complex one-sided peak-amplitude spectrum. The input is truncated to `n_fft`, its `min(n_samples, n_fft)` samples are multiplied by a window of that length and zero-padded to `n_fft`, the result is divided by that window's coherent gain (its sum), and non-DC/non-Nyquist positive bins are doubled. The window length is stored as `SpectralFrame.win_length`. | Input unit | `20 log10(magnitude / channel_ref)` |
| `ChannelFrame.stft()` | Complex one-sided peak-amplitude spectrum for each time frame. SciPy coherent-gain magnitude scaling is used and non-DC/non-Nyquist positive bins are doubled. | Input unit | `20 log10(magnitude / channel_ref)` |
| `ChannelFrame.welch()` | Real one-sided peak-amplitude spectrum. Welch segment **power spectra** use `scaling="spectrum"`, are averaged, and are then converted to peak amplitude. This is not PSD and is not per hertz. | Input unit | `20 log10(amplitude / channel_ref)` |
| `ChannelFrame.noct_spectrum()` | RMS amplitude in each fractional-octave band, as returned by MoSQITo. `G` selects the exact-center-frequency ratio convention (`10` for base `10**(3/10)`, `2` for base 2); it is not a gain. | Input unit | `20 log10(band_rms / channel_ref)` |
| `SpectralFrame.noct_synthesis()` | RMS amplitude in each fractional-octave band, from MoSQITo's frequency-domain band filters. The stored peak-amplitude bins are first converted to RMS (non-DC/non-Nyquist bins divided by `sqrt(2)`) and divided by `sqrt(ENBW)`, where `ENBW = n_fft * sum(w**2) / sum(w)**2` for the frame's `win_length`-sample `window`. Band levels therefore match `noct_spectrum()` for stationary signals. | Input unit | `20 log10(band_rms / channel_ref)` |

`SpectralFrame.magnitude` is the absolute value of its stored spectral quantity;
for the FFT and Welch rows above, that quantity is amplitude.
`SpectralFrame.power` is exactly `magnitude**2`; for an amplitude result it has
squared input units. The name is a compatibility property and does not turn an
arbitrary result into physical power or power spectral density. The amplitude and
squared-amplitude forms give the same level when their references are paired
correctly:

```text
20 log10(amplitude / reference)
  = 10 log10(amplitude**2 / reference**2)
```

Each channel's `channel_ref` comes from its calibration metadata. For example,
`Pa` defaults to `20 µPa`, while reader-created canonical audio explicitly uses
`FS` with reference `1`. Identity calibration with an empty unit also has a
numeric reference of `1`, but its level label remains generic `dB re 1 input
unit`, never dBFS. All amplitude conversions floor the magnitude/reference
ratio at `1e-12`, so zero returns `-240 dB`.
For the amplitude results above, `dBA` adds the IEC 61672 A-weighting curve to
that amplitude level; it does not change the underlying stored amplitude.

## Pairwise spectral contracts

The pairwise contracts below are the mathematical authority for the dedicated
`CoherenceFrame`, `CrossSpectralFrame`, and `TransferFunctionFrame`.
The generic `SpectralFrame` is not a semantic owner for these quantities; its
amplitude `dB`, `dBA`, and `ifft` behavior must not be used to infer or render
CSD or transfer meaning.

### Ordered channel pairs

For `n_channels` input channels, every pair uses output-major, input-minor
ordering:

```text
pair_index = output_index * n_channels + input_index
```

The pair at that index is always `(output, input)`. Pair labels, role metadata,
source-time offsets, and numerical fixtures use this same order. The source-time
offset is the input-role offset because the input signal is the reference for a
cross quantity or transfer denominator.

The public CSD and coherence declarations are Recipe version 2 because their
labels now spell out `(output, input)`. Transfer version 2 is the canonical
input-denominator contract. Version 1 of these operations was written only
before 0.9.0 and is no longer replayed.

### Coherence

`ChannelFrame.coherence()` computes magnitude-squared coherence through the
processing-layer numerical definition. Its genuine operation result is
dimensionless and lies in `[0, 1]`, apart from NaN bins where coherence is
undefined. Processing tests compare this result with SciPy and own that
mathematical contract.

`CoherenceFrame` construction is a structural boundary, not a numerical data
scanner. Direct construction, WDF decoding, and external Dask input preserve
the supplied real numeric values without range checks, infinity checks,
clipping, rounding, or replacement. The constructor still validates rank,
frequency-bin count, real numeric dtype, typed pair state, source identity,
metadata, and the remaining constructor state. This distinction allows lazy
graphs and externally produced arrays to retain their exact data while the
Wandas coherence operation remains responsible for producing mathematically
valid coherence values.

### Cross-spectrum / CSD

`ChannelFrame.csd()` stores the complex one-sided quantity returned by the
SciPy-domain definition

```python
P_out_in = scipy.signal.csd(
    x=input_signal,
    y=output_signal,
    ...,
)[1]
```

Therefore `P_out_in` is conceptually `conj(X_input) * X_output`. Its phase is
`angle(P_out_in)` in radians and its magnitude is `abs(P_out_in)`.

| Scaling | Stored unit | Pair reference | Explicit level |
| --- | --- | --- | --- |
| `spectrum` | `input_unit * output_unit` | `input_ref * output_ref` | `10 * log10(abs(P_out_in) / pair_reference)` |
| `density` | `input_unit * output_unit / Hz` | `input_ref * output_ref / Hz` (numeric reference per 1 Hz) | `10 * log10(abs(P_out_in) / pair_reference)` |

For an auto-spectrum, the reference is the square of the channel amplitude
reference, so the CSD power level equals the corresponding amplitude level:
`10 * log10(abs(P) / ref**2) == 20 * log10(amplitude / ref)`. CSD must not use
the generic amplitude `20 * log10` rule for cross values.

### Transfer function

The stored quantity is the complex ordered transfer ratio

```python
H_out_in = P_out_in / P_in_in
```

where `P_out_in` is the CSD above and `P_in_in` is the input auto-spectrum from
`scipy.signal.welch(input_signal, ...)`, using the same spectral configuration.
`H[output, input]` therefore means the response observed on `output` for the
signal on `input`. Its magnitude is `abs(H_out_in)` and its phase is
`angle(H_out_in)` in radians.

| Quantity | Contract |
| --- | --- |
| Unit | `output_unit / input_unit` (`1` when the physical units are equal) |
| Reference ratio | `output_ref / input_ref` |
| Transfer level | `20 * log10(abs(H_out_in) / (output_ref / input_ref))` |
| Same unit and same reference | Ordinary gain level `20 * log10(abs(H_out_in))` |

For unlike units the result is a unit ratio, not a dimensionless gain. Its
level is still defined only relative to the explicit output/input reference
ratio.

### Deriving transfer and coherence from a computed CSD

`CrossSpectralFrame.transfer_function()` and `CrossSpectralFrame.coherence()`
derive the canonical quantities from the stored rows without repeating the
segment FFTs or the averaging:

```python
H_out_in = P_out_in / P_in_in
gamma2_out_in = abs(P_out_in) ** 2 / (P_out_out * P_in_in)
```

`P_in_in` is the real diagonal row of the same CSD, which equals the Welch
auto-spectrum for the same configuration. `ChannelFrame.transfer_function()`
and `ChannelFrame.coherence()` use this same definition internally, so both
routes give the same values. Coherence is independent of the CSD `scaling`; a
derived coherence reuses the CSD's `average` and `detrend` settings, while
`ChannelFrame.coherence()` always uses mean averaging. A median-averaged CSD
does not give magnitude-squared coherence and can produce values above 1, so
deriving coherence from a CSD whose lineage records `average="median"` emits a
`UserWarning`; the values are returned unchanged.

Both methods need every output/input pair, so call them before
`select_pair()`; a selected subset raises the `KeyError` of the first missing
pair. The result always uses canonical output-major/input-minor row order and
keeps the represented frequency bins. Caching the CSD once lets several
derived quantities share it:

```python
csd = frame.csd(n_fft=4096).cache()
h = csd.transfer_function().select_pair(1, 0)
g = csd.coherence().select_pair(1, 0)
```

### CSD from a spectrogram

`SpectrogramFrame.csd(scaling=...)` averages `conj(X_in) * X_out` over every
time frame the spectrogram holds and returns the same `CrossSpectralFrame` as
`ChannelFrame.csd()`. The STFT stores amplitude-normalized values, so the
method undoes that normalization: `scaling="spectrum"` halves the doubled
one-sided bins, and `scaling="density"` also multiplies by
`sum(w) ** 2 / (fs * sum(w ** 2))` for the frame's window `w`.

Frames are mean-averaged. The CSD cannot remove a trend from a computed STFT
frame, so detrend when computing the STFT: `stft(detrend="constant")` or
`stft(detrend="linear")` removes each frame's trend before windowing, as SciPy's
`ShortTimeFFT.stft_detrend` does. The result then equals
`ChannelFrame.csd(detrend=...)` with the same `detrend` over the same segments.
The default `stft()` does not detrend, so it matches `csd(detrend=False)`, not
the `csd()` default of `"constant"`. A detrended STFT cannot reconstruct the signal with
`istft()`. The STFT pads the signal, so its first and last
frames contain zeros; slice the time axis to choose the averaged frames. Each
pair's `source_time_offset` is the start of the first averaged window, matching
`ChannelFrame.csd()` over the same segment:

```python
spec = frame.stft(n_fft=4096, hop_length=1024, detrend="constant").cache()
h = spec[:, :, 4:-4].csd().transfer_function().select_pair(1, 0)
```

### Pairwise edge cases and plotting handoff

An exact zero input auto-spectrum is not silently floored or regularized:
the corresponding transfer value is complex NaN. A nonzero near-zero
denominator remains a finite, potentially large gain. Non-finite inputs and
results remain non-finite and are not clipped. Diagnostics, when added by a
caller, must identify the affected input pair and frequency bin.

A-weighting is unsupported for typed coherence, CSD, and transfer quantities
because there is no unambiguous generic choice of which channel(s) to weight or
how many times to apply the weighting. The architecture-neutral
`reject_pairwise_a_weighting()` helper is the enforcement hook for those
consumers; requests must fail explicitly rather than silently altering a
pairwise value. #401 intentionally does not route the current generic
Generic `SpectralFrame.dBA` and its amplitude plot `Aw` path remain available
for amplitude spectra. Dedicated pairwise Frames reject `Aw` at their typed
plot boundary; no operation-history quantity inference is used.

The dedicated Frames own the typed plotting implementation. Their frequency
and matrix projections use linear `abs(P_out_in)` for CSD and linear
`abs(H_out_in)` for transfer by default, with unit-aware labels. Explicit phase
views show radians. Explicit level views use the CSD `10 * log10` rule or the
transfer `20 * log10` rule, respectively; A-weighting remains unsupported.

The property surface is intentionally quantity-specific: a `CoherenceFrame`
exposes raw `data` and `coherence`; a `CrossSpectralFrame` exposes raw complex
`data`, `magnitude`, `phase`, and an explicit CSD `level_db`; a
`TransferFunctionFrame` exposes raw complex `data`, `gain`, `phase`, `gain_db`
for the ordinary same-unit gain case, and `transfer_level_db` for the explicit
output/input reference ratio (including unlike units). Pair roles, `scaling`,
derived unit, reference, and the Transfer v1 denominator definition remain
typed Frame state. These properties are not added to the generic
`SpectralFrame`.

## Window and hop defaults

`ChannelFrame.stft()`, `welch()`, `csd()`, `coherence()` and
`transfer_function()` share one segmentation rule. An omitted `win_length`
defaults to `n_fft`, and an omitted `hop_length` defaults to `win_length // 4`,
the librosa convention. The hop is a fraction of the analysis window,
not of the zero-padded FFT size, so `stft(n_fft=2048, win_length=512)` uses a
hop of 128 and keeps the 75 % Hann overlap that ISTFT needs. A `win_length`
below 4 has no default hop; pass `hop_length` explicitly.
`SpectrogramFrame` and its WDF file store the resolved hop.

A Recipe keeps an omitted `hop_length` omitted. STFT Recipe operation version 2
therefore resolves it with this rule on replay.

## FFT inverse guarantee

`SpectralFrame.ifft()` inverts Wandas' one-sided amplitude normalization. For a
spectrum produced by `ChannelFrame.fft()` with its stored `n_fft`, `window` and
`win_length`, the result is the input truncated to `n_fft` samples, multiplied
by a `win_length`-sample window and zero-padded to `n_fft`, within
floating-point tolerance. `win_length` is `min(n_samples, n_fft)`. A boxcar window
therefore round-trips the prepared waveform. A tapered window such as Hann
round-trips the **windowed** waveform: Wandas does not divide by the window and
cannot reconstruct samples erased at its zeros.

These transforms build Dask graphs lazily. Accessing a NumPy-value property such
as `frame.data`, `magnitude`, `power`, `dB`, or plotting is the materialization
boundary.
