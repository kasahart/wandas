---
description: "Frame test patterns: immutability, metadata propagation, lazy evaluation, chaining, and domain transitions"
applyTo: "tests/frames/**"
---
# Wandas Test Policy: Frames (`tests/frames/`)

Frame tests verify **data structure correctness** and **metadata consistency across domain transitions**.
Frames are at the core of Wandas' public API and are the objects users directly manipulate.

**Prerequisite**: This file is applied together with [test-grand-policy.instructions.md](test-grand-policy.instructions.md).
Follow the 4 pillars of the Grand Policy, then apply the additional guidelines below.

---

## Common Fixtures for Frame Tests

Define the following fixtures in `conftest.py`. Use deterministic signals rather than random data.

- **`channel_frame`**: Standard multi-channel frame. Use deterministic signals with known analytical solutions.
- **`mono_frame`**: Single-channel frame. Use deterministic signals with known analytical solutions.

Seeded random data is acceptable for immutability/structural tests where signal content is irrelevant, but deterministic signals are preferred.

---

## Frame Types & Their Test Concerns

### ChannelFrame (time domain)
- **Immutability**: Verify that the return value differs from the original instance and that original data is unchanged for all operations
- **Channel operations**: Verify label and data consistency after `add_channel`, `remove_channel`, `rename_channels`
- **Slicing**: Verify that correct channels are returned for int / slice / bool mask / label-based indexing
- **Dask laziness**: Verify that `_data` is a `DaskArray` instance after operations

### SpectralFrame (frequency domain)
- **Complex data type**: Verify that `.data` is a complex array
- **Derived properties**: Verify that `magnitude`, `phase`, `power`, `dB`, `dBA` are mathematically correct
- **Frequency axis**: Verify that the `frequency` array is correctly constructed up to the Nyquist frequency

### SpectrogramFrame (time-frequency domain)
- **3D shape**: Verify that shape is `(n_channels, n_freq_bins, n_time_frames)`
- **ISTFT round-trip**: Verify that the original time-domain signal can be recovered within `atol=1e-6` by the inverse transform
- **Time and frequency axes**: Verify that metadata for both axes is accurate

### RoughnessFrame, NOctFrame (specialized analysis)
- **MoSQITo equivalence**: Since Wandas wraps MoSQITo, verify that results match the external library exactly
- **Physical quantity metadata**: Verify that units (sone, asper, etc.) are correctly set

---

## Required Test Categories per Frame Operation

When adding a new Frame method, always include the following 4 categories:

**1. Immutability** — the original frame must not be modified:
Copy the original data before the operation, then verify with `assert result is not channel_frame` that the original instance's data has not changed and that the return value is a new instance.

**2. Metadata Propagation** — sampling rate and history are correctly updated:
Verify that the sampling rate is preserved after the operation. For methods designed to record history, verify that the `operation_history` count increases by exactly 1. For structural operations that only modify the channel collection (`add_channel` / `remove_channel` / `rename_channels`, etc.), verify that `operation_history` remains unchanged.

**3. Lazy Evaluation** — Dask array is preserved:
Verify that `result._data` is an instance of `dask.array.core.Array` after the operation.

**4. Chaining** — supports chained calls:
Verify that subsequent calls like `.normalize()` can be made on the operation result, and that the return type is `ChannelFrame`.

---

## Domain Transition Test Patterns

Domain-changing operations (`fft()`, `stft()`, `loudness()`, etc.) require additional verification:

- Verify that the return value type is converted to the correct frame type (e.g., `SpectralFrame`)
- Verify that the frequency bin count matches the theoretical value (N/2+1 for FFT)
- Verify that the sampling rate is correctly inherited

---

## Channel Collection Test Patterns

Key considerations for multi-channel operation tests:

- Label and data order must be consistent after `add_channel`
- Duplicate labels must raise `ValueError`
- Adding data with different length must raise `ValueError`

---

## Indexing Test Matrix

Test coverage for ChannelFrame index access is important:

| Index Type | Example | Expected Behavior |
|-----------|---------|-------------------|
| `int` | `cf[0]` | Single channel ChannelFrame |
| Negative `int` | `cf[-1]` | Last channel |
| `slice` | `cf[0:2]` | Multi-channel subset |
| `bool mask` | `cf[cf.rms > 0.5]` | Channels matching condition |
| `str` label | `cf["ch0"]` | Channel by label |
| `list[str]` | `cf[["ch0", "ch1"]]` | Multiple channels by label |
| `list[int]` | `cf[[0, 2]]` | Multiple channels by index |
| Out of range | `cf[999]` | IndexError |

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [frames-design.instructions.md](frames-design.instructions.md) — Frame architecture and immutability rules
