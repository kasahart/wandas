import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATHS = (REPO_ROOT / "README.md", REPO_ROOT / "README.ja.md")
README_SAMPLE_AUDIO_FIGURES = (
    REPO_ROOT / "images" / "readme_sample_audio_describe_0.png",
    REPO_ROOT / "images" / "readme_sample_audio_describe_1.png",
)
README_PLOT_FIGURES = (
    REPO_ROOT / "images" / "readme_known_signal_waveform.png",
    REPO_ROOT / "images" / "readme_known_signal_spectrum.png",
)
README_FIGURES = (*README_SAMPLE_AUDIO_FIGURES, *README_PLOT_FIGURES)


def _python_blocks(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)\n```", markdown, flags=re.S)


def _github_main_paths(markdown: str) -> Iterator[str]:
    pattern = r"https://github\.com/kasahart/wandas/(?:blob|tree)/main/([^)#?]+)"
    yield from re.findall(pattern, markdown)


def _python_block_containing(path: Path, marker: str) -> str:
    for block in _python_blocks(path.read_text(encoding="utf-8")):
        if marker in block:
            return block
    raise AssertionError(f"{path.name} has no Python block containing {marker!r}")


def _execute_readme_example(path: Path, workdir: Path, marker: str) -> dict[str, object]:
    plt.close("all")
    namespace: dict[str, object] = {"__name__": f"readme_example_{path.stem}"}
    block = _python_block_containing(path, marker)
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
    sample_dir = tmp_path / "learning-path"
    sample_dir.mkdir()
    shutil.copyfile(REPO_ROOT / "learning-path" / "sample_audio.wav", sample_dir / "sample_audio.wav")
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

    assert "review the analysis as a team" in english
    assert "ask an AI agent to check" in english
    assert "reviewable workflow" in english
    assert "チームで解析をレビュー" in japanese
    assert "AI エージェントに確認を頼む" in japanese
    assert "レビューしやすい" in japanese
    assert "データの文脈" in japanese


def test_readme_places_install_before_examples() -> None:
    """README should let users install Wandas before running examples."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")

    assert english.index("## Installation") < english.index("## Inspect the Sample Audio")
    assert japanese.index("## インストール") < japanese.index("## サンプル音声を確認する")
    assert english.index('pip install "wandas[io]"') < english.index("## Inspect the Sample Audio")
    assert japanese.index('pip install "wandas[io]"') < japanese.index("## サンプル音声を確認する")


def test_readme_real_data_path_stays_compact() -> None:
    """README real-data path should be a short next step, not the main proof."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")

    assert 'recording = wd.read("recording.wav", end=15)' in english
    assert 'recording = wd.read("recording.wav", end=15)' in japanese
    assert "spectrum = clean.welch()" in english
    assert "spectrum = clean.welch()" in japanese
    assert english.count("```python") <= 3
    assert japanese.count("```python") <= 3


def test_readme_leads_with_wav_describe_and_verified_signal_figures() -> None:
    """README should lead with WAV describe output before known-signal checks."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")

    assert "learning-path/sample_audio.wav" in english
    assert "learning-path/sample_audio.wav" in japanese
    assert "images/readme_sample_audio_describe_0.png" in english
    assert "images/readme_sample_audio_describe_1.png" in english
    assert "images/readme_sample_audio_describe_0.png" in japanese
    assert "images/readme_sample_audio_describe_1.png" in japanese
    assert "Validate with a Known Signal" in english
    assert "既知信号で確認する" in japanese
    assert "images/readme_known_signal_waveform.png" in english
    assert "images/readme_known_signal_spectrum.png" in english
    assert "images/readme_known_signal_waveform.png" in japanese
    assert "images/readme_known_signal_spectrum.png" in japanese
    assert english.index("learning-path/sample_audio.wav") < english.index("Validate with a Known Signal")
    assert english.index("## Validate with a Known Signal") < english.index("## Use Your Own Data")
    assert japanese.index("learning-path/sample_audio.wav") < japanese.index("既知信号で確認する")
    assert japanese.index("## 既知信号で確認する") < japanese.index("## 手元のデータで使う")
    assert "DC offset disappears" in english
    assert "DC オフセットが消え" in japanese
    assert "750 Hz component" in english
    assert "1500 Hz component" in english
    assert "750 Hz 成分" in japanese
    assert "1500 Hz 成分" in japanese
    assert "signal.add_channel" in english
    assert "signal.add_channel" in japanese
    assert "signal = wd.from_numpy" in japanese
    for figure in README_FIGURES:
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_readme_describe_examples_keep_committed_color_range() -> None:
    """README describe examples should use the color range shown by the figures."""
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (REPO_ROOT / "README.ja.md").read_text(encoding="utf-8")
    english_sample = _python_block_containing(REPO_ROOT / "README.md", "sample_source")
    japanese_sample = _python_block_containing(REPO_ROOT / "README.ja.md", "sample_source")

    sample_call = (
        'recording.describe(fmin=20, fmax=8_000, vmin=-80, vmax=-20, image_save="readme_sample_audio_describe.png")'
    )
    own_data_call = 'clean.describe(fmin=20, fmax=fmax, vmin=-80, vmax=-20, image_save="recording_overview.png")'

    assert sample_call in english_sample
    assert sample_call in japanese_sample
    assert own_data_call in english
    assert own_data_call in japanese


def test_readme_sample_audio_supports_checkout_and_installed_users() -> None:
    """The first example should work inside and outside a repository checkout."""
    expected_url = "https://raw.githubusercontent.com/kasahart/wandas/main/learning-path/sample_audio.wav"
    for path in README_PATHS:
        block = _python_block_containing(path, "sample_source")
        assert 'Path("learning-path/sample_audio.wav")' in block
        assert expected_url in block
        assert "recording = wd.read(sample_source, end=15, normalize=True)" in block


def test_readme_documents_frame_context_and_boundaries() -> None:
    """README claims should include the important execution and analysis boundaries."""
    english = README_PATHS[0].read_text(encoding="utf-8")
    japanese = README_PATHS[1].read_text(encoding="utf-8")

    for text in (english, japanese):
        assert "`previous`" in text
        assert "`operation_history`" in text
        assert "`wd.load()`" in text
        assert "`wandas[ml]`" in text
        assert "lazy_loading=True" not in text
        assert 'wd.from_folder("recordings/", recursive=True)' in text

    assert "calibrated" in english
    assert "校正" in japanese


def test_known_signal_readme_result_matches_executed_example(tmp_path: Path) -> None:
    """The known-signal narrative should be backed by the README code itself."""
    namespace = _execute_readme_example(REPO_ROOT / "README.md", tmp_path, "known signal")
    signal = cast(Any, namespace["signal"])
    processed = cast(Any, namespace["processed"])
    comparison = cast(Any, namespace["comparison"])
    spectrum = comparison.fft(n_fft=48_000)

    assert comparison.labels == ["Original", "After DC removal + 1 kHz low-pass"]
    assert [entry["operation"] for entry in processed.operation_history][-3:] == [
        "remove_dc",
        "lowpass_filter",
        "rename_channels",
    ]
    np.testing.assert_allclose(np.asarray(signal.to_numpy()).mean(), 0.25, atol=1e-6)
    np.testing.assert_allclose(np.asarray(processed.to_numpy()).mean(), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(comparison.to_numpy()).mean(axis=1), [0.25, 0.0], atol=1e-5)

    amplitudes = np.abs(np.asarray(spectrum.to_numpy()))
    freqs = spectrum.freqs

    def bin_at(freq: float) -> int:
        return int(np.argmin(np.abs(freqs - freq)))

    np.testing.assert_allclose(amplitudes[0, bin_at(750)], 0.20, atol=1e-6)
    assert amplitudes[1, bin_at(750)] > 0.15
    assert amplitudes[1, bin_at(1500)] < amplitudes[0, bin_at(1500)] * 0.25
    assert amplitudes[0, bin_at(0)] > 0.20
    assert amplitudes[1, bin_at(0)] < 1e-6
    plt.close("all")


def test_known_signal_readme_uses_automatic_overlay_labels() -> None:
    """The example should demonstrate channel-label propagation without plot boilerplate."""
    for path in README_PATHS:
        block = _python_block_containing(path, "known signal")
        assert "label=comparison.labels" not in block
        assert "fft(n_fft=sr)" not in block
        assert "comparison.fft().plot(" in block


def test_known_signal_readme_plots_have_expected_semantics(tmp_path: Path) -> None:
    """README figures should be backed by stable Wandas plot semantics."""
    _execute_readme_example(REPO_ROOT / "README.md", tmp_path, "known signal")

    figures = [plt.figure(number) for number in plt.get_fignums()]
    assert len(figures) == len(README_PLOT_FIGURES)

    waveform_ax = figures[0].axes[0]
    spectrum_ax = figures[1].axes[0]
    assert waveform_ax.get_title() == "Original vs processed"
    assert spectrum_ax.get_title() == "FFT: original vs processed"
    np.testing.assert_allclose(waveform_ax.get_xlim(), (0, 0.02))
    np.testing.assert_allclose(spectrum_ax.get_xlim(), (0, 4_000))
    assert spectrum_ax.get_xscale() == "linear"

    waveform_lines = waveform_ax.get_lines()
    spectrum_lines = spectrum_ax.get_lines()
    expected_labels = ["Original", "After DC removal + 1 kHz low-pass"]
    assert [line.get_label() for line in waveform_lines] == expected_labels
    assert [line.get_label() for line in spectrum_lines] == expected_labels
    np.testing.assert_allclose(
        [np.asarray(line.get_ydata()).mean() for line in waveform_lines],
        [0.25, 0.0],
        atol=1e-5,
    )

    freqs = np.asarray(spectrum_lines[0].get_xdata())
    bin_750 = int(np.argmin(np.abs(freqs - 750)))
    bin_1500 = int(np.argmin(np.abs(freqs - 1500)))
    original_db = np.asarray(spectrum_lines[0].get_ydata())
    processed_db = np.asarray(spectrum_lines[1].get_ydata())
    np.testing.assert_allclose(spectrum_ax.get_ylim(), (30, 90))
    assert processed_db[bin_750] > original_db[bin_750] - 3
    assert processed_db[bin_1500] < original_db[bin_1500] - 10

    plt.close("all")


def test_sample_audio_describe_example_creates_readme_figures(tmp_path: Path) -> None:
    """The sample-audio describe example should create the committed overview figures."""
    sample_dir = tmp_path / "learning-path"
    sample_dir.mkdir()
    shutil.copyfile(REPO_ROOT / "learning-path" / "sample_audio.wav", sample_dir / "sample_audio.wav")

    _execute_readme_example(REPO_ROOT / "README.md", tmp_path, "learning-path/sample_audio.wav")

    generated_figures = (
        tmp_path / "readme_sample_audio_describe_0.png",
        tmp_path / "readme_sample_audio_describe_1.png",
    )
    for figure in generated_figures:
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    plt.close("all")
