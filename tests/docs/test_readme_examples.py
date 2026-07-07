import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import wandas as wd

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATHS = (REPO_ROOT / "README.md", REPO_ROOT / "README.ja.md")


def _python_blocks(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)\n```", markdown, flags=re.S)


def _github_main_paths(markdown: str) -> Iterator[str]:
    pattern = r"https://github\.com/kasahart/wandas/(?:blob|tree)/main/([^)#?]+)"
    yield from re.findall(pattern, markdown)


@pytest.fixture()
def readme_example_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sr = 48_000
    t = np.arange(sr) / sr
    samples = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    sf.write(tmp_path / "recording.wav", samples, sr)
    sf.write(tmp_path / "audio.wav", samples, sr)

    recordings = tmp_path / "recordings"
    recordings.mkdir()
    sf.write(recordings / "tone.wav", samples, sr)

    pytest.importorskip("h5py", reason="README WDF example requires the io extra")
    pytest.importorskip("IPython.display", reason="README describe example requires the marimo extra")
    pytest.importorskip("mosqito.sound_level_meter", reason="README N-octave example requires the psychoacoustic extra")
    pytest.importorskip("mosqito.sq_metrics", reason="README psychoacoustic examples require the psychoacoustic extra")
    wd.from_numpy(samples, sampling_rate=sr, label="saved tone").save(tmp_path / "analysis.wdf")

    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.filterwarnings("ignore:More than 20 figures have been opened:RuntimeWarning")
def test_readme_python_code_blocks_execute(readme_example_workspace: Path) -> None:
    """README Python examples should stay executable as the public API changes."""
    del readme_example_workspace

    for path in README_PATHS:
        namespace: dict[str, object] = {"__name__": f"readme_example_{path.stem}"}
        for index, block in enumerate(_python_blocks(path.read_text(encoding="utf-8")), start=1):
            try:
                exec(compile(block, f"{path.name}:python-block-{index}", "exec"), namespace)
            except Exception as exc:
                raise AssertionError(f"{path.name} Python block {index} failed: {exc}") from exc


def test_readme_github_repository_links_target_existing_paths() -> None:
    """README links into the repository should not point at missing files."""
    missing: list[str] = []
    for path in README_PATHS:
        for target in _github_main_paths(path.read_text(encoding="utf-8")):
            if not (REPO_ROOT / target).exists():
                missing.append(f"{path.name}: {target}")

    assert missing == []


def test_readme_optional_dependency_examples_are_labeled() -> None:
    """README examples should name extras needed beyond the recommended install."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")

    assert "N-octave spectra require the psychoacoustic extra." in english
    assert "SPL-style dB plots require pressure calibration." in english
    assert "オクターブバンド解析には psychoacoustic extra が必要です。" in japanese
    assert "SPL として dB 表示する場合は、先に音圧校正を設定します。" in japanese


def test_readme_recording_workflow_keeps_spl_data_calibrated_and_filter_bounded() -> None:
    """README recording examples should not invalidate later SPL or filter examples."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")

    assert "normalize=True" not in english
    assert "normalize=True" not in japanese
    assert ".band_pass_filter(80, min(8_000, 0.45 * signal.sampling_rate))" in english
    assert ".band_pass_filter(80, min(8_000, 0.45 * signal.sampling_rate))" in japanese
