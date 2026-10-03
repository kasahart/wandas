"""Header-only preflight never constructs a Frame or decodes audio."""

import io
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import wandas as wd
from wandas.io.readers import SoundFileReader


def audio_bytes(format: str = "WAV") -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.zeros((123, 2)), 8000, format=format)
    return stream.getvalue()


@pytest.mark.parametrize("format", ["WAV", "FLAC", "OGG", "AIFF"])
def test_inspect_audio_has_header_metadata_without_frame_graph_or_pcm(format: str) -> None:
    with (
        patch.object(wd.ChannelFrame, "from_file", side_effect=AssertionError("Frame")),
        patch("dask.array.from_delayed", side_effect=AssertionError("graph")),
        patch.object(SoundFileReader, "get_data", side_effect=AssertionError("PCM")),
        patch.object(sf.SoundFile, "read", side_effect=AssertionError("PCM")),
    ):
        info = wd.inspect(audio_bytes(format), file_type=format.lower())
    assert info["samplerate"] == 8000 and info["channels"] == 2 and info["frames"] == 123
    assert info["duration"] == 123 / 8000
    assert info["format"] == format and info["subtype"]


@pytest.mark.parametrize("factory", [bytes, bytearray, memoryview, io.BytesIO])
def test_inspect_anonymous_buffers_default_to_wav(factory: Any) -> None:
    assert wd.inspect(factory(audio_bytes()))["frames"] == 123


def test_inspect_local_path_and_name_inference(tmp_path: Path) -> None:
    path = tmp_path / "source.flac"
    path.write_bytes(audio_bytes("FLAC"))
    assert wd.inspect(path)["format"] == "FLAC"
    assert wd.inspect(str(path))["frames"] == 123
    stream = io.BytesIO(path.read_bytes())
    stream.name = "source.flac"
    assert wd.inspect(stream, source_name="wrong.csv")["format"] == "FLAC"
    assert wd.inspect(path.read_bytes(), source_name="source.flac")["format"] == "FLAC"
    assert wd.inspect(path.read_bytes(), source_name="wrong.csv", file_type=".FLAC")["format"] == "FLAC"


@pytest.mark.parametrize("broken", [False, True])
def test_inspect_borrowed_stream_restores_position_and_never_closes(broken: bool) -> None:
    stream = io.BytesIO(b"not audio" if broken else audio_bytes())
    stream.seek(5)
    if broken:
        with pytest.raises(sf.LibsndfileError):
            wd.inspect(stream)
    else:
        assert wd.inspect(stream)["frames"] == 123
    assert stream.tell() == 5 and not stream.closed
    if not broken:
        assert wd.read(stream).n_samples == 123


def test_inspect_rejects_nonseekable_and_text_streams_without_consuming() -> None:
    class NonSeekable(io.BytesIO):
        def seekable(self) -> bool:
            return False

    stream = NonSeekable(audio_bytes())
    with pytest.raises(ValueError, match="seekable"):
        wd.inspect(stream)
    assert stream.tell() == 0
    with pytest.raises(TypeError, match="binary"):
        wd.inspect(io.StringIO("text"))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("source", ["https://example.invalid/a.wav", "http://example.invalid/a.wav"])
def test_inspect_rejects_urls_without_network(source: str) -> None:
    with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
        with pytest.raises(ValueError, match="local"):
            wd.inspect(source)


@pytest.mark.parametrize("extension", ["csv", "wdf", "unknown"])
def test_inspect_rejects_non_audio_formats(extension: str) -> None:
    with pytest.raises(ValueError):
        wd.inspect(b"time,value\n0,1\n", file_type=extension)


def test_inspect_missing_and_invalid_audio_errors_are_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        wd.inspect(tmp_path / "missing.wav")
    with pytest.raises(sf.LibsndfileError):
        wd.inspect(b"bad wav")
    with pytest.raises(TypeError):
        wd.inspect(np.zeros(4))  # ty: ignore[invalid-argument-type]


def test_inspect_does_not_fall_back_to_current_position_when_rewind_fails() -> None:
    class CannotRewind(io.BytesIO):
        def seek(self, offset: int, whence: int = 0) -> int:
            if offset == 0 and whence == 0:
                raise OSError("cannot rewind")
            return super().seek(offset, whence)

    stream = CannotRewind(audio_bytes())
    stream.seek(5)
    with pytest.raises(ValueError, match="rewind"):
        wd.inspect(stream)
    assert stream.tell() == 5 and not stream.closed
