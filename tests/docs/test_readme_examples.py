import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATHS = (REPO_ROOT / "README.md", REPO_ROOT / "README.ja.md")
README_DESCRIBE_FIGURES = (
    REPO_ROOT / "images" / "readme_known_signal_describe_0.png",
    REPO_ROOT / "images" / "readme_known_signal_describe_1.png",
)
README_PLOT_FIGURES = (
    REPO_ROOT / "images" / "readme_known_signal_waveform.png",
    REPO_ROOT / "images" / "readme_known_signal_spectrum.png",
)
README_SIGNAL_FIGURES = (*README_DESCRIBE_FIGURES, *README_PLOT_FIGURES)


def _python_blocks(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)\n```", markdown, flags=re.S)


def _github_main_paths(markdown: str) -> Iterator[str]:
    pattern = r"https://github\.com/kasahart/wandas/(?:blob|tree)/main/([^)#?]+)"
    yield from re.findall(pattern, markdown)


def _execute_known_signal_example(path: Path, workdir: Path) -> dict[str, object]:
    plt.close("all")
    namespace: dict[str, object] = {"__name__": f"known_signal_{path.stem}"}
    block = _python_blocks(path.read_text(encoding="utf-8"))[0]
    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        exec(compile(block, f"{path.name}:known-signal-block", "exec"), namespace)
    finally:
        os.chdir(old_cwd)
    return namespace


@pytest.fixture()
def readme_example_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sr = 48_000
    t = np.arange(sr) / sr
    samples = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    sf.write(tmp_path / "recording.wav", samples, sr)
    (tmp_path / "images").mkdir()

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
                plt.close("all")
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
    """README should label optional features without making them the main path."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")

    assert "`wandas[io]`" in english
    assert "`wandas[psychoacoustic]`" in english
    assert "`wandas[io]`" in japanese
    assert "`wandas[psychoacoustic]`" in japanese
    assert "loudness = " not in english
    assert "roughness = " not in english
    assert "loudness = " not in japanese
    assert "roughness = " not in japanese


def test_readme_positions_wandas_as_reviewable_analysis_workflow() -> None:
    """README should include the collaboration and review value proposition."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")

    assert "shared with teammates or AI agents" in english
    assert "implementation and review easier" in english
    assert "Reviewable workflows" in english
    assert "チームや AI エージェントと解析を共有・レビュー" in japanese
    assert "実装内容の確認とレビュー" in japanese
    assert "レビューしやすい解析フロー" in japanese


def test_readme_real_data_path_stays_compact() -> None:
    """README real-data path should be a short next step, not the main proof."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")

    assert "normalize=True" not in english
    assert "normalize=True" not in japanese
    assert 'recording = wd.read("recording.wav", start=0, end=10)' in english
    assert 'recording = wd.read("recording.wav", start=0, end=10)' in japanese
    assert english.count("```python") <= 3
    assert japanese.count("```python") <= 3


def test_readme_leads_with_verified_synthetic_signal_and_figure() -> None:
    """README should lead with a known signal and Wandas-produced figures."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")

    assert "Known-signal check" in english
    assert "既知信号で確認する" in japanese
    assert "images/readme_known_signal_describe_0.png" in english
    assert "images/readme_known_signal_describe_1.png" in english
    assert "images/readme_known_signal_describe_0.png" in japanese
    assert "images/readme_known_signal_describe_1.png" in japanese
    assert "images/readme_known_signal_waveform.png" in english
    assert "images/readme_known_signal_spectrum.png" in english
    assert "images/readme_known_signal_waveform.png" in japanese
    assert "images/readme_known_signal_spectrum.png" in japanese
    assert "The first figures are Wandas `describe()` output from the original frame" in english
    assert "最初の図は元の frame から Wandas の `describe()` が出力したものです" in japanese
    assert "the DC offset disappears after `remove_dc()`" in english
    assert "DC オフセットが消えていることが分かります" in japanese
    assert "750 Hz and 1500 Hz for the first channel" in english
    assert "1 つ目のチャンネルの 750 Hz / 1500 Hz" in japanese
    for figure in README_SIGNAL_FIGURES:
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_known_signal_readme_result_matches_executed_example(tmp_path: Path) -> None:
    """The known-signal narrative should be backed by the README code itself."""
    namespace = _execute_known_signal_example(REPO_ROOT / "README.md", tmp_path)
    clean = cast(Any, namespace["clean"])
    spectrum = cast(Any, namespace["spectrum"])

    clean_data = np.asarray(clean.to_numpy(), dtype=np.float64)
    means = clean_data.mean(axis=1)
    amplitudes = np.asarray(spectrum.to_numpy())
    freqs = spectrum.freqs
    peak_freqs = freqs[np.argmax(amplitudes, axis=1)]

    def bin_at(freq: float) -> int:
        return int(np.argmin(np.abs(freqs - freq)))

    assert np.max(np.abs(means)) < 1e-6
    np.testing.assert_allclose(peak_freqs, [750.0, 1500.0])
    np.testing.assert_allclose(amplitudes[0, bin_at(750)], 0.20)
    np.testing.assert_allclose(amplitudes[0, bin_at(1500)], 0.05)
    np.testing.assert_allclose(amplitudes[1, bin_at(1500)], 0.10)
    np.testing.assert_allclose(amplitudes[1, bin_at(3000)], 0.02)
    plt.close("all")


def test_known_signal_readme_plots_have_expected_semantics(tmp_path: Path) -> None:
    """README figures should be backed by stable Wandas plot semantics."""
    _execute_known_signal_example(REPO_ROOT / "README.md", tmp_path)

    generated_describe_figures = (
        tmp_path / "readme_known_signal_describe_0.png",
        tmp_path / "readme_known_signal_describe_1.png",
    )
    for figure in generated_describe_figures:
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    figures = [plt.figure(number) for number in plt.get_fignums()]
    assert len(figures) == len(README_PLOT_FIGURES)

    waveform_ax = figures[0].axes[0]
    spectrum_ax = figures[1].axes[0]
    assert waveform_ax.get_title() == "Known signal after remove_dc()"
    assert spectrum_ax.get_title() == "Welch spectrum of the known signal"
    np.testing.assert_allclose(waveform_ax.get_xlim(), (0, 0.02))
    np.testing.assert_allclose(spectrum_ax.get_xlim(), (0, 4_000))

    waveform_lines = waveform_ax.get_lines()
    spectrum_lines = spectrum_ax.get_lines()
    assert [line.get_label() for line in waveform_lines] == ["750 Hz + 1500 Hz", "1500 Hz + 3000 Hz"]
    assert [line.get_label() for line in spectrum_lines] == ["750 Hz + 1500 Hz", "1500 Hz + 3000 Hz"]
    for line in waveform_lines:
        y_data = np.asarray(line.get_ydata(), dtype=np.float64)
        np.testing.assert_allclose(y_data.mean(), 0.0, atol=1e-6)

    peak_freqs: list[float] = []
    for line in spectrum_lines:
        x_data = np.asarray(line.get_xdata(), dtype=np.float64)
        y_data = np.asarray(line.get_ydata(), dtype=np.float64)
        peak_freqs.append(float(x_data[np.argmax(y_data)]))
    np.testing.assert_allclose(peak_freqs, [750.0, 1500.0])

    plt.close("all")
