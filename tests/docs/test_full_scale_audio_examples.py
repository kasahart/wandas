from pathlib import Path

import numpy as np
import soundfile as sf
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

import wandas as wd
from scripts.documentation_audio_examples import (
    COMPARISON_NOCT_FMAX,
    COMPARISON_NOCT_YLIM,
    COMPARISON_PEAK_FS,
    COMPARISON_WELCH_YLIM,
    DB_REFERENCE_FS,
    comparison_signals,
    read_comparison_pcm16,
    write_comparison_pcm16,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pcm16_comparison_uses_exact_canonical_decode(tmp_path: Path) -> None:
    paths = write_comparison_pcm16(tmp_path)
    frames = read_comparison_pcm16(tmp_path)

    assert np.max(np.abs(comparison_signals())) == COMPARISON_PEAK_FS
    for path, frame in zip(paths, frames, strict=True):
        expected, _ = sf.read(path, dtype="float64", always_2d=True)
        assert sf.info(path).subtype == "PCM_16"
        assert frame.to_numpy().dtype == np.float64
        np.testing.assert_array_equal(frame.to_numpy(), expected[:, 0])


def test_comparison_spectra_are_finite_and_visible(tmp_path: Path) -> None:
    frames = read_comparison_pcm16(tmp_path)

    assert DB_REFERENCE_FS == 1.0
    for frame in frames:
        for spectrum, ylim in (
            (frame.welch(n_fft=2048, hop_length=1024), COMPARISON_WELCH_YLIM),
            (frame.noct_spectrum(fmin=25, fmax=COMPARISON_NOCT_FMAX, n=3), COMPARISON_NOCT_YLIM),
        ):
            values = np.asarray(spectrum.dB)
            assert np.isfinite(values).all()
            peak_db = float(np.max(values))
            assert ylim[0] + 5 < peak_db < ylim[1] - 5


def test_float_wav_round_trip_preserves_full_scale_values(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "round_trip.wav"
    sf.write(source, np.array([-0.75, -0.25, 0.25, 0.75]), 8_000, subtype="PCM_16")

    original = wd.read(source)
    original.to_wav(output)
    encoded, _ = sf.read(output, dtype="float64", always_2d=True)
    reloaded = wd.read(output)

    assert sf.info(output).subtype == "FLOAT"
    assert reloaded.to_numpy().dtype == np.float64
    np.testing.assert_array_equal(reloaded.to_numpy(), encoded[:, 0])
    np.testing.assert_array_equal(reloaded.to_numpy(), original.to_numpy())


def test_documented_describe_figure_has_full_scale_data_and_visible_axes(tmp_path: Path) -> None:
    source = REPO_ROOT / "learning-path" / "sample_audio.wav"
    audio = wd.read(source, end=15).get_channel(0)
    expected, _ = sf.read(source, dtype="float64", always_2d=True, stop=15 * int(audio.sampling_rate))

    assert audio.to_numpy().dtype == np.float64
    np.testing.assert_array_equal(audio.to_numpy(), expected[:, 0])
    assert -1.0 <= float(audio.to_numpy().min()) < 0 < float(audio.to_numpy().max()) <= 1.0

    output = tmp_path / "read_wav_describe.png"
    audio.describe(fmin=20, fmax=8_000, vmin=-80, vmax=-20, image_save=output)
    generated = mpimg.imread(output)
    committed_docs = mpimg.imread(REPO_ROOT / "docs/src/assets/images/read_wav_describe.png")
    committed_root = mpimg.imread(REPO_ROOT / "images/read_wav_describe.png")
    assert generated.ndim == committed_docs.ndim == committed_root.ndim == 3
    assert generated.shape == committed_docs.shape == committed_root.shape
    assert float(np.std(generated)) > 0.05
    np.testing.assert_array_equal(committed_docs, committed_root)
    plt.close("all")


def test_mkdocs_uses_one_current_describe_command_and_no_stale_figure() -> None:
    index = (REPO_ROOT / "docs/src/index.md").read_text(encoding="utf-8")
    tutorial = (REPO_ROOT / "docs/src/tutorial/index.md").read_text(encoding="utf-8")
    command = 'audio.describe(fmin=20, fmax=8_000, vmin=-80, vmax=-20, image_save="read_wav_describe.png")'
    read_call = "audio = wd.read(url, end=15).get_channel(0)"

    for markdown in (index, tutorial):
        assert command in markdown
        assert read_call in markdown
        assert "summer_streets1.wav" not in markdown
    assert not (REPO_ROOT / "docs/src/assets/images/read_wav_describe_set_config.png").exists()
    assert not (REPO_ROOT / "images/read_wav_describe_set_config.png").exists()
