# Spectral numerical contracts / スペクトル数値契約

Wandas spectral Frames carry amplitude quantities unless an API explicitly says
otherwise. Calibration is applied before analysis, so `unit` below means the
physical unit of the input channel (or dimensionless full scale when no unit is
set).

| Result | Stored value and normalization | Unit | Level conversion |
| --- | --- | --- | --- |
| `ChannelFrame.fft()` | Complex one-sided peak-amplitude spectrum. The input is truncated or zero-padded to `n_fft`, multiplied by the selected window, divided by the window coherent gain (its sum), and non-DC/non-Nyquist positive bins are doubled. | Input unit | `20 log10(magnitude / channel_ref)` |
| `ChannelFrame.stft()` | Complex one-sided peak-amplitude spectrum for each time frame. SciPy coherent-gain magnitude scaling is used and non-DC/non-Nyquist positive bins are doubled. | Input unit | `20 log10(magnitude / channel_ref)` |
| `ChannelFrame.welch()` | Real one-sided peak-amplitude spectrum. Welch segment **power spectra** use `scaling="spectrum"`, are averaged, and are then converted to peak amplitude. This is not PSD and is not per hertz. | Input unit | `20 log10(amplitude / channel_ref)` |
| `ChannelFrame.noct_spectrum()` | RMS amplitude in each fractional-octave band, as returned by MoSQITo. `G` selects the exact-center-frequency ratio convention (`10` for base `10**(3/10)`, `2` for base 2); it is not a gain. | Input unit | `20 log10(band_rms / channel_ref)` |

`SpectralFrame.magnitude` is the absolute value of its stored complex amplitude.
`SpectralFrame.power` is exactly `magnitude**2`, with squared input units; the
name is a compatibility property and does not turn the result into power spectral
density. The amplitude and power forms give the same level when their references
are paired correctly:

```text
20 log10(amplitude / reference)
  = 10 log10(amplitude**2 / reference**2)
```

Each channel's `channel_ref` comes from its calibration metadata. For example,
`Pa` defaults to `20 µPa`, while uncalibrated full-scale data defaults to `1`.
`dBA` adds the IEC 61672 A-weighting curve to that amplitude level; it does not
change the underlying stored amplitude.

## FFT inverse guarantee

`SpectralFrame.ifft()` inverts Wandas' one-sided amplitude normalization. For a
spectrum produced by `ChannelFrame.fft()` with its stored matching `n_fft` and
`window`, the result is exactly the truncated-or-zero-padded analysis input
multiplied by that window, within floating-point tolerance. A boxcar window
therefore round-trips the prepared waveform. A tapered window such as Hann
round-trips the **windowed** waveform: Wandas does not divide by the window and
cannot reconstruct samples erased at its zeros.

These transforms build Dask graphs lazily. Accessing a NumPy-value property such
as `frame.data`, `magnitude`, `power`, `dB`, or plotting is the materialization
boundary.
