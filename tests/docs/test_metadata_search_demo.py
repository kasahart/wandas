import csv
from pathlib import Path

from scipy.io import wavfile

DEMO_ROOT = Path(__file__).parents[2] / "learning-path" / "data" / "metadata_search"


def test_metadata_search_demo_files_match_sidecar() -> None:
    with (DEMO_ROOT / "recordings.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    wav_paths = sorted(path.relative_to(DEMO_ROOT).as_posix() for path in DEMO_ROOT.rglob("*.wav"))

    assert len(rows) == 3
    assert sorted(row["path"] for row in rows) == wav_paths
    assert {row["load"] for row in rows} == {"low", "high"}
    assert {int(row["rpm"]) for row in rows} == {1_000, 1_500, 2_000}


def test_metadata_search_demo_wavs_are_tiny_and_consistent() -> None:
    wav_paths = sorted(DEMO_ROOT.rglob("*.wav"))

    assert len(wav_paths) == 3
    for path in wav_paths:
        sampling_rate, data = wavfile.read(path)
        assert sampling_rate == 8_000
        assert data.ndim == 1
        assert data.shape == (800,)
        assert path.stat().st_size < 4_000
