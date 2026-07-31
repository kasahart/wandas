"""Keep duplicated spectral terminology aligned with the numerical contract."""

from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_canonical_contract_defines_value_unit_normalization_and_reference() -> None:
    contract = (REPO_ROOT / "docs/src/explanation/spectral-numerical-contracts.md").read_text(encoding="utf-8")

    assert "one-sided peak-amplitude spectrum" in contract
    assert "window coherent gain" in contract
    assert "This is not PSD and is not per hertz" in contract
    assert "RMS amplitude in each fractional-octave band" in contract
    assert "20 log10(amplitude / reference)" in contract
    assert "windowed" in contract


def test_learning_material_does_not_relabel_levels_as_raw_magnitude_or_psd() -> None:
    lessons = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "learning-path/03_signal_processing_basics.py",
            "learning-path/04_advanced_processing.py",
        )
    )

    assert "Magnitude Spectrum" not in lessons
    assert "Welch PSD" not in lessons
    assert "パワースペクトル密度" not in lessons
    assert "Spectrum level [dB re 1 FS]" not in lessons
    assert "Amplitude level Spectrum" not in lessons
    assert "Band RMS level [dB re 1 FS]" in lessons
