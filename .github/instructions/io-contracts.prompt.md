# Wandas I/O Contracts Prompt

Use this prompt when working on `wandas/io/` or anything that reads/writes external data (WAV, WDF, CSV, sample data).

## Responsibilities
- `wandas/io/` handles:
  - reading/writing WAV files with correct sampling rate and channel layout,
  - WDF (HDF5-based) storage with full metadata preservation,
  - simple CSV-based time series loading,
  - helpers for sample data and datasets.

## Invariants & contracts
- Preserve **sampling rate**, **channel labels**, and **time/frequency axes** across I/O operations.
- Ensure metadata round-trips for formats that support it (especially WDF):
  - user/recording metadata,
  - `operation_history` when applicable.
- For WAV/CSV, define clear rules for what metadata is retained or reconstructed.

## Design guidelines
- Keep I/O functions thin wrappers around libraries (e.g. `scipy.io`, `soundfile`, `h5py`) and Wandas frames.
- Avoid embedding heavy signal processing logic in I/O modules; delegate to `processing/` or frame methods when necessary.
- Provide clear error messages (WHAT/WHY/HOW) when files are missing, malformed, or incompatible.

Use this prompt to maintain reliable, predictable I/O behavior and metadata integrity.
