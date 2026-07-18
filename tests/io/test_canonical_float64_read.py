"""Canonical external-reader numeric contracts."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from scipy.io import wavfile

import wandas as wd
from wandas.frames.channel import ChannelFrame


@pytest.mark.parametrize("subtype", ["PCM_U8", "PCM_16", "PCM_24", "PCM_32"])
def test_wav_integer_subtypes_match_soundfile_full_scale(tmp_path: Path, subtype: str) -> None:
    path = tmp_path / f"integer-{subtype}.wav"
    source = np.array([[-0.75], [0.0], [0.5]], dtype=np.float64)
    sf.write(path, source, 8_000, subtype=subtype)
    expected, _ = sf.read(path, dtype="float64", always_2d=True)

    frame = wd.read(path)

    assert frame._data.dtype == np.dtype("float64")
    assert frame.compute().dtype == np.dtype("float64")
    np.testing.assert_array_equal(frame.compute(), expected.T)


@pytest.mark.parametrize("subtype", ["FLOAT", "DOUBLE"])
def test_wav_float_subtypes_preserve_values_above_full_scale(tmp_path: Path, subtype: str) -> None:
    path = tmp_path / f"float-{subtype}.wav"
    source = np.array([[-2.0], [0.1], [1.5]], dtype=np.float64)
    sf.write(path, source, 8_000, subtype=subtype)
    encoded = source.astype(np.float32).astype(np.float64) if subtype == "FLOAT" else source

    frame = wd.read(path)

    np.testing.assert_array_equal(frame.compute(), encoded.T)
    assert frame.compute().dtype == np.dtype("float64")


def test_pcm_u8_is_zero_centered(tmp_path: Path) -> None:
    path = tmp_path / "unsigned.wav"
    wavfile.write(path, 8_000, np.array([0, 128, 255], dtype=np.uint8))

    np.testing.assert_array_equal(wd.read(path).compute(), [[-1.0, 0.0, 127.0 / 128.0]])


def test_same_audio_content_matches_across_path_and_memory_transports(tmp_path: Path) -> None:
    path = tmp_path / "transport.wav"
    sf.write(path, np.array([[-0.5], [0.0], [0.5]]), 8_000, subtype="PCM_16")
    content = path.read_bytes()
    expected = wd.read(path).compute()

    for source in (content, bytearray(content), memoryview(content), io.BytesIO(content)):
        np.testing.assert_array_equal(wd.read(source, file_type=".wav").compute(), expected)


def test_csv_preserves_numeric_values_as_float64_and_rejects_text(tmp_path: Path) -> None:
    valid = tmp_path / "valid.csv"
    pd.DataFrame({"time": [0.0, 0.5], "count": [1, 2], "value": [1.25, -3.5]}).to_csv(valid, index=False)
    frame = wd.read(valid)

    assert frame._data.dtype == np.dtype("float64")
    np.testing.assert_array_equal(frame.compute(), [[1.0, 2.0], [1.25, -3.5]])

    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"time": [0.0, 0.5], "sensor": ["ok", "bad"]}).to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="CSV data channels must be numeric"):
        wd.read(invalid).compute()


def test_read_time_normalize_argument_is_removed(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros((2, 1)), 8_000)

    with pytest.raises(TypeError, match="normalize"):
        wd.read(path, normalize=True)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError, match="normalize"):
        ChannelFrame.from_file(path, normalize=True)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError, match="normalize"):
        ChannelFrame.read_wav(path, normalize=True)  # ty: ignore[unknown-argument]
