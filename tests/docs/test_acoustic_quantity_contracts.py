"""Documentation contracts for acoustic quantities and conformance wording."""

import inspect
from pathlib import Path

from wandas.frames.channel import ChannelFrame
from wandas.frames.mixins.channel_processing_mixin import ChannelProcessingMixin
from wandas.processing.temporal import RmsTrend, SoundLevel
from wandas.processing.weighting import A_weighting

REPO_ROOT = Path(__file__).resolve().parents[2]
LEARNING_00 = REPO_ROOT / "learning-path/00_why_wandas.py"
LEARNING_04 = REPO_ROOT / "learning-path/04_advanced_processing.py"
FRAMES_API = REPO_ROOT / "docs/src/api/frames.md"
PROCESSING_API = REPO_ROOT / "docs/src/api/processing.md"
EXECUTION_DOC = REPO_ROOT / "docs/src/explanation/audio-operation-execution.md"
STABILITY_DOC = REPO_ROOT / "docs/src/explanation/public-api-stability.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_linear_rms_examples_are_not_labeled_as_levels() -> None:
    learning_00 = _read(LEARNING_00)

    assert "Linear RMS amplitudes" in learning_00
    assert "線形RMS音圧 [Pa]" in learning_00
    assert "Sound pressure levels" not in learning_00
    assert "A特性音圧レベル (RMS)" not in learning_00


def test_db_spl_learning_example_names_reference_and_weightings() -> None:
    learning_04 = _read(LEARNING_04)

    assert 'freq_weighting="A"' in learning_04
    assert 'time_weighting="Fast"' in learning_04
    assert 'time_weighting="Slow"' in learning_04
    assert "dB SPL re {reference_pressure_pa:g} Pa" in learning_04
    assert "sound_data.channels[0].ref" in learning_04
    assert "sound_data._channel_metadata" not in learning_04


def test_conformance_claims_are_bounded_to_implemented_behavior() -> None:
    sources = "\n".join(
        (
            _read(LEARNING_04),
            _read(FRAMES_API),
            _read(PROCESSING_API),
            _read(EXECUTION_DOC),
            _read(STABILITY_DOC),
            inspect.getdoc(SoundLevel) or "",
            inspect.getdoc(A_weighting) or "",
        )
    )

    assert "騒音計の特性を忠実に再現" not in sources
    assert "class 1-compliant filter" not in sources
    assert "complete IEC/JIS" in sources
    assert "not instrument certification" in sources


def test_public_docstrings_distinguish_quantity_reference_and_eagerness() -> None:
    rms = inspect.getdoc(ChannelFrame.rms) or ""
    rms_trend = inspect.getdoc(ChannelProcessingMixin.rms_trend) or ""
    sound_level = inspect.getdoc(ChannelProcessingMixin.sound_level) or ""
    rms_operation = inspect.getdoc(RmsTrend) or ""
    level_operation = inspect.getdoc(SoundLevel) or ""

    assert "linear RMS amplitude" in rms
    assert "immediate computation" in rms
    assert "20 * log10(window_rms / channel_ref)" in rms_trend
    assert "dB SPL only" in rms_trend
    assert "10 * log10(smoothed_power / channel_ref**2)" in sound_level
    assert "125 ms (Fast)" in sound_level
    assert "1 s (Slow)" in sound_level
    assert "20 * log10(RMS / ref)" in rms_operation
    assert "pressure in Pa with ``ref=2e-5``" in level_operation


def test_api_docs_publish_the_same_acoustic_contract() -> None:
    frames = _read(FRAMES_API)
    processing = _read(PROCESSING_API)
    stability = _read(STABILITY_DOC)

    assert "`frame.rms` | one linear RMS amplitude per channel" in frames
    assert "`10 log10(smoothed_power / channel_ref²)`" in frames
    assert "`10 log10(smoothed_power / ref²)`" in processing
    assert "`rms` never performs logarithmic conversion" in " ".join(stability.split())
