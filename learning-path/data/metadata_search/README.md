# Metadata search demo dataset

This tiny synthetic dataset supports `08_metadata_driven_dataset_search.py`.
It contains three mono WAV files (8 kHz, 0.1 seconds each) arranged in a
DCASE-style `machine/split/filename.wav` layout, plus a sidecar metadata CSV.

| Relative path | Tone | Purpose |
| --- | ---: | --- |
| `fan/train/section_00_source.wav` | 440 Hz | fan training/source example |
| `fan/test/section_01_target.wav` | 660 Hz | fan test/target example |
| `pump/train/section_00_source.wav` | 880 Hz | pump training/source example |

`recordings.csv` adds `load` and `rpm` fields keyed by relative path. All
signals are deterministic synthetic tones created for this repository; no
external recordings are included.
