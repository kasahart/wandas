---
description: "Processing test patterns: numerical accuracy, frequency-domain verification, and reference library comparison"
applyTo: "tests/processing/**"
---
# Wandas Test Policy: Processing (`tests/processing/`)

Processing tests verify **numerical computation accuracy** and **acoustic physics validity**.
`wandas/processing/` is a pure numerical logic layer that does not handle frame metadata.
These tests focus on **whether numbers are physically correct** and **agreement with external authorities**.

**Prerequisite**: This file is applied together with [test-grand-policy.instructions.md](test-grand-policy.instructions.md).

---

## Common Fixtures for Processing Tests

Processing-layer fixtures return `(DaskArray, int)` tuples (named with `_dask` suffix).
Used for pure numerical verification without going through frames. Create channel-wise chunks of `chunks=(1, -1)` with `wandas.utils.dask_helpers.da_from_array`.

Define the following standard fixtures in `conftest.py`:

- **`sine_1khz_48k_dask`**: 1 kHz pure tone (SR=48000 Hz, 1 second). Standard signal for psychoacoustic tests.
- **`calibrated_sine_1khz_70dB_dask`**: 1 kHz pure tone at 70 dB SPL (SR=48000 Hz). For IEC-compliant psychoacoustic tests. Compute amplitude from `p_ref=2e-5 Pa` as `amplitude = p_ref * 10**(70/20) * sqrt(2)`.
- **`dual_tone_16k_dask`**: Composite tone of 50 Hz + 1000 Hz (SR=16000 Hz). Used to verify filter pass/stop characteristics.
- **`impulse_16k_dask`**: Unit impulse (SR=16000 Hz). Used to verify filter impulse responses.

---

## Processing Module Test Strategy

### AudioOperation Base Class
Verify the following behavior:
- Parameter validation in `__init__` (WHAT/WHY/HOW error messages)
- `process()` accepts a DaskArray and returns a DaskArray
- `get_metadata_updates()` returns the correct dict
- `get_display_name()` returns a correctly formatted string

### Per-Module Requirements

| Module | Reference Library | Key Verification |
|--------|------------------|-----------------|
| `filters.py` | `scipy.signal` | Frequency response characteristics, stopband attenuation |
| `spectral.py` | `scipy.signal`, `librosa` | FFT peak position, STFT shape |
| `psychoacoustic.py` | `mosqito` | Loudness/roughness/sharpness values |
| `weighting.py` | `scipy.signal` | A/C weighting frequency response |
| `temporal.py` | `numpy` | RMS, zero-crossing rate, and other statistics |
| `stats.py` | `numpy`, `scipy.stats` | Statistical accuracy |
| `effects.py` | None (verify against theoretical values) | Fade, clipping, etc. |

---

## Filter Tests: Frequency Domain Verification

For filter tests, verify **attenuation in the frequency domain** rather than time-domain waveforms.
Obtain the FFT spectrum after applying the filter, and verify that the amplitude ratio of each frequency component matches the designed theoretical value within a relative error of 1E-6.

Also compare against `scipy.signal.filtfilt(b, a, x)` as Wrapper Equivalence. Since `LowPassFilter` generates coefficients with `scipy.signal.butter(order, cutoff/nyquist, btype="low")` and applies them with `scipy.signal.filtfilt(b, a, x, axis=1)`, the same call yields an exact match.

### Filter Edge Cases
- Verify that a `ValueError` is raised when the cutoff frequency is ≤ 0 or ≥ the Nyquist frequency (`sampling_rate / 2`).

---

## Spectral Tests: Known-Signal Verification

Use analytically predictable signals for FFT/STFT tests.

Note: Class names are `FFT` and `STFT` (not `FFTOperation` / `STFTOperation`).

- **FFT peak verification**: Compute the FFT of a 1 kHz pure tone with `n_fft=sr` (1 Hz/bin) and verify that the peak bin is within 1000 ±1.
- **STFT shape verification**: Compute STFT with `n_fft=1024` and verify that the number of frequency bins is `1024 // 2 + 1 = 513`.

---

## Psychoacoustic Tests: MoSQITo Reference Verification

Since Wandas wraps MoSQITo, psychoacoustic computation results must **exactly match** the MoSQITo reference implementation.
Use IEC-compliant calibrated signals (known sound pressure level, 1 second or more) and verify that Wandas computation results exactly match the results of direct calls to MoSQITo's `loudness_zwtv`, etc.

---

## A-Weighting Tests: Known Frequency Response

The A-weighting filter is defined as 0 dB at 1 kHz. Tests should define the "theoretical value" as the frequency response, and verify using one of the following approaches:

- Compute the frequency response from coefficients obtained via `A_weighting(fs, output="sos")` using `scipy.signal.sosfreqz`/`freqz`, and verify that the gain matches the theoretical A-weighting curve within sufficiently small error (e.g., relative error ~1e-6).
- When using a known-frequency pure tone as the input signal, note that `sosfilt` is a causal filter; exclude the transient rise period from evaluation, compute the RMS ratio of input and output in dB, and verify it matches the theoretical frequency response within sufficiently small error (explicitly document the evaluation window and tolerance in the test).
---

## Operation Registration & Display Name Tests

- The key for `create_operation` is the **registry name** (e.g., `"lowpass_filter"`), which differs from the frame method name (`"low_pass_filter"`). Verify that `create_operation("lowpass_filter", ...)` returns something other than `None`.
- `LowPassFilter.get_display_name()` returns `"lpf"` (no parameters included).

---

## Anti-Patterns Specific to Processing Tests

Avoid the following patterns as they undermine the value of Processing tests:

- **Time-domain-only verification**: Verifying filter results only with `assert result is not None` in the time domain guarantees nothing about filter quality. Always verify attenuation in the frequency domain.
- **Self-referential expected values**: Computing expected values with your own implementation is equivalent to "grading your own homework" and cannot detect bugs. Use external libraries (SciPy, librosa, MoSQITo, etc.) as the "authority".
- **Skipping MoSQITo comparison in psychoacoustic tests**: Verification like `result.mean() > 0` does not guarantee correctness. Always compare numerically against the MoSQITo reference implementation.

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [processing-api.instructions.md](processing-api.instructions.md) — Processing layer architecture
