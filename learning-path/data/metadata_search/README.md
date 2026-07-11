# Metadata search demo dataset

This tiny synthetic dataset supports `08_metadata_driven_dataset_search.py`.
It contains three mono WAV files (8 kHz, 0.1 seconds each) arranged in a
generic `group/batch/filename.wav` layout, plus a sidecar metadata CSV.

| Relative path | Tone | Purpose |
| --- | ---: | --- |
| `group_a/batch_01/recording_001.wav` | 440 Hz | reference example |
| `group_a/batch_02/recording_002.wav` | 660 Hz | variant example |
| `group_b/batch_01/recording_003.wav` | 880 Hz | second-group example |

`recordings.csv` adds `condition` and `priority` fields keyed by relative path. All
signals are deterministic synthetic tones created for this repository; no
external recordings are included.
