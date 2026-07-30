from __future__ import annotations

import io
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import soundfile as sf

import wandas as wd
from tests.frame_helpers import channel_first_values

SAMPLE_RATE = 8_000
INTEGER_SUBTYPES = ("PCM_U8", "PCM_16", "PCM_24", "PCM_32")
FLOAT_SUBTYPES = ("FLOAT", "DOUBLE")


def _frame_values(frame: wd.ChannelFrame) -> np.ndarray:
    assert isinstance(frame._data, da.Array)
    return np.asarray(channel_first_values(frame))


def _wav_bytes(values: np.ndarray, subtype: str) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, values, SAMPLE_RATE, format="WAV", subtype=subtype)
    return buffer.getvalue()


def test_pyodide_imports_wandas_and_soundfile() -> None:
    assert wd.__version__
    assert sf.__version__
    assert "WAV" in sf.available_formats()
    print(f"WAV runtime: wandas={wd.__version__}, soundfile={sf.__version__}")


@pytest.mark.parametrize("n_channels", [1, 2], ids=["mono", "multichannel"])
def test_wav_metadata_and_channel_first_shape(n_channels: int) -> None:
    samples = np.arange(8, dtype=np.float64) / 8
    source = samples if n_channels == 1 else np.column_stack((samples, -samples))
    content = _wav_bytes(source, "DOUBLE")
    info = sf.info(io.BytesIO(content))

    frame = wd.read(content, file_type=".wav")

    assert info.samplerate == SAMPLE_RATE
    assert info.channels == n_channels
    assert info.frames == 8
    assert frame.sampling_rate == SAMPLE_RATE
    assert frame.n_channels == n_channels
    assert frame.n_samples == 8
    assert _frame_values(frame).shape == (n_channels, 8)
    assert _frame_values(frame).dtype == np.dtype("float64")
    expected = source.reshape(-1, 1).T if n_channels == 1 else source.T
    np.testing.assert_array_equal(_frame_values(frame), expected)


@pytest.mark.parametrize("subtype", INTEGER_SUBTYPES)
def test_wav_integer_pcm_scaling(subtype: str) -> None:
    # Binary-fraction levels are exactly representable in every tested PCM subtype.
    source = np.array([-0.75, -0.25, 0.0, 0.5], dtype=np.float64)
    content = _wav_bytes(source, subtype)

    frame = wd.read(content, file_type=".wav")

    assert frame.sampling_rate == SAMPLE_RATE
    assert frame.n_channels == 1
    assert _frame_values(frame).dtype == np.dtype("float64")
    np.testing.assert_array_equal(_frame_values(frame), source[None, :])


@pytest.mark.parametrize("subtype", FLOAT_SUBTYPES)
def test_wav_floating_point_subtypes_preserve_full_scale_values(subtype: str) -> None:
    source = np.array([-2.0, -0.25, 0.0, 1.5], dtype=np.float64)
    content = _wav_bytes(source, subtype)
    expected = source.astype(np.float32).astype(np.float64) if subtype == "FLOAT" else source

    frame = wd.read(content, file_type=".wav")

    assert _frame_values(frame).dtype == np.dtype("float64")
    np.testing.assert_array_equal(_frame_values(frame), expected[None, :])


@pytest.mark.parametrize(
    "transport",
    [
        pytest.param(lambda content: content, id="bytes"),
        pytest.param(io.BytesIO, id="file-like"),
    ],
)
def test_wav_in_memory_transports(transport) -> None:
    source = np.column_stack(
        (
            np.array([-0.5, 0.0, 0.5], dtype=np.float64),
            np.array([0.25, 0.0, -0.25], dtype=np.float64),
        )
    )
    content = _wav_bytes(source, "PCM_16")

    frame = wd.read(transport(content), file_type=".wav")

    assert frame.shape == (2, 3)
    np.testing.assert_array_equal(_frame_values(frame), source.T)


def test_wav_partial_read_preserves_channel_first_shape() -> None:
    source = np.column_stack(
        (
            np.arange(8, dtype=np.float64) / 8,
            -np.arange(8, dtype=np.float64) / 8,
        )
    )
    content = _wav_bytes(source, "DOUBLE")

    frame = wd.read(
        content,
        file_type=".wav",
        start=2 / SAMPLE_RATE,
        end=6 / SAMPLE_RATE,
    )

    assert frame.sampling_rate == SAMPLE_RATE
    assert frame.n_channels == 2
    assert frame.n_samples == 4
    assert frame.shape == (2, 4)
    np.testing.assert_array_equal(_frame_values(frame), source[2:6].T)


def test_wav_public_write_read_round_trip(tmp_path: Path) -> None:
    source = np.array(
        [
            [-0.75, -0.25, 0.0, 0.5],
            [0.5, 0.0, -0.25, -0.75],
        ],
        dtype=np.float64,
    )
    frame = wd.from_numpy(source, sampling_rate=SAMPLE_RATE)
    output = tmp_path / "wandas-round-trip.wav"

    frame.to_wav(output)
    loaded = wd.read(output)
    info = sf.info(output)

    assert info.format == "WAV"
    assert info.subtype == "FLOAT"
    assert info.samplerate == SAMPLE_RATE
    assert info.channels == 2
    assert loaded.sampling_rate == SAMPLE_RATE
    assert loaded.n_channels == 2
    assert loaded.n_samples == 4
    assert _frame_values(loaded).dtype == np.dtype("float64")
    # Wandas intentionally writes bounded floating-point audio as WAV FLOAT.
    expected = source.astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(_frame_values(loaded), expected)
    np.testing.assert_array_equal(_frame_values(frame), source)
