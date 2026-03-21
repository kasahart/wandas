---
description: "I/O test patterns: round-trip fidelity, metadata preservation, and format-specific contracts (WAV/WDF/CSV)"
applyTo: "tests/io/**"
---
# Wandas Test Policy: I/O (`tests/io/`)

I/O tests verify **data integrity** and **complete metadata round-trip fidelity**.
No data loss or metadata loss is permitted during file read/write operations.

**Prerequisite**: This file is applied together with [test-grand-policy.instructions.md](test-grand-policy.instructions.md).

---

## Common Fixtures for I/O Tests

Define the following fixtures in `conftest.py`.

- **`known_signal_frame`**: A 2-channel frame for round-trip verification (SR=44100 Hz, ch_labels=`["left", "right"]`, ch_units=`["Pa", "Pa"]`). Since I/O tests use "data equality" as the verification criterion, seeded random data (`np.random.default_rng(42)`) is acceptable.
- **`create_test_wav`**: A factory fixture that accepts `tmp_path`. Takes `sr`, `n_channels`, `n_samples` as arguments, writes a WAV file as int16 PCM using `scipy.io.wavfile.write`, and returns a Path. For mono (n_channels=1), write as a 1D array; for stereo (n_channels=2), write as a 2D array.

---

## I/O Test Strategy

### Format-Specific Requirements

| Format | Module | Key Concerns |
|--------|--------|-------------|
| WAV | `wav_io.py` | PCM precision, normalization, sampling rate, channel count |
| WDF (HDF5) | `wdf_io.py` | Full metadata preservation, compression, dtype conversion |
| CSV | `readers.py` | Time column interpretation, delimiter, header handling |

---

## Core Pattern: Round-Trip Test

The "write → read → compare with original" pattern is mandatory for all I/O formats.

For WDF round-trips, verify that all of the following are preserved:
- Numerical data (`np.testing.assert_allclose` rtol=1e-6)
- Sampling rate
- Channel labels
- `operation_history`

---

## WAV I/O Tests

### Float Round-Trip
`ChannelFrame.to_wav` writes as IEEE FLOAT subtype when data is floating-point and `max(abs(data)) <= 1`, so no PCM quantization error occurs. Read with `normalize=True` and verify the result matches the original data within `atol=1e-6`.

### PCM Round-Trip
A WAV file written as int16 using `scipy.io.wavfile.write` should be read back with `normalize=False` and verified to match the original integer sample values (compare using `cf.data[0].astype(np.int16)`).

### Channel Count
- Verify that a `ChannelFrame` loaded from a stereo WAV (n_channels=2) has `n_channels` equal to 2.
- Verify that a `ChannelFrame` loaded from a mono WAV (n_channels=1) has `n_channels` equal to 1.

---

## WDF I/O Tests

WDF is Wandas' native format and guarantees full metadata preservation.

- **Channel metadata**: Verify that ch_labels and ch_units are saved and restored.
- **Overwrite protection**: Verify that `FileExistsError` is raised when attempting to save to an existing file with `overwrite=False` (the default).

---

## CSV I/O Tests

Read a CSV with a header (time column + data columns) and verify that the channel count is correctly obtained (matches the number of data columns excluding the time column).

---

## File Error Handling Tests

Verify that calling `ChannelFrame.from_file()` with a non-existent path raises `FileNotFoundError`.

---

## Lazy Loading Verification

Verify that `cf._data` after WAV load is an instance of `dask.array.core.Array`, confirming that data is not immediately loaded into memory.

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [io-contracts.instructions.md](io-contracts.instructions.md) — I/O metadata preservation contracts
