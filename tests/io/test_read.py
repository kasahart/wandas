import io
from pathlib import Path
from typing import Any, cast

import pytest

from wandas.frames.channel import ChannelFrame
from wandas.io import read


def test_read_is_exported_from_io_package() -> None:
    from wandas.io.read import read as module_read

    assert read is module_read


def test_read_defaults_in_memory_source_to_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_from_file(cls: type[ChannelFrame], path: object, **kwargs: object) -> object:
        captured["path"] = path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(ChannelFrame, "from_file", classmethod(fake_from_file))

    result = read(b"not real wav")

    assert result is sentinel
    assert captured["path"] == b"not real wav"
    assert captured["file_type"] == ".wav"
    assert captured["source_name"] is None


def test_read_infers_named_file_like_type_and_source_name(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_from_file(cls: type[ChannelFrame], path: object, **kwargs: object) -> object:
        captured["path"] = path
        captured.update(kwargs)
        return sentinel

    buffer = io.BytesIO(b"time,left\n0.0,1.0\n")
    buffer.name = "folder/source.csv"
    monkeypatch.setattr(ChannelFrame, "from_file", classmethod(fake_from_file))

    result = read(buffer)

    assert result is sentinel
    assert captured["path"] is buffer
    assert captured["file_type"] == ".csv"
    assert captured["source_name"] == "folder/source.csv"


@pytest.mark.parametrize("file_type", [".wdf", "wdf"])
def test_read_rejects_wdf_file_type_with_load_guidance(file_type: str) -> None:
    with pytest.raises(ValueError, match="wd.load"):
        read(b"not a real wdf", file_type=file_type)


def test_read_preserves_explicit_source_name_for_named_stream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = object()
    path = tmp_path / "sensor.csv"
    path.write_text("time,left\n0.0,1.0\n", encoding="utf-8")
    captured_kwargs: dict[str, object] = {}

    def fake_from_file(path: object, **kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return expected

    monkeypatch.setattr(ChannelFrame, "from_file", staticmethod(fake_from_file))

    with path.open("rb") as file_obj:
        assert read(file_obj, source_name="explicit.csv") is expected

    assert captured_kwargs["file_type"] == ".csv"
    assert captured_kwargs["source_name"] == "explicit.csv"


def test_read_forwards_non_path_non_memory_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = object()
    source = cast(Any, object())
    captured: dict[str, object] = {}
    captured_kwargs: dict[str, object] = {}

    def fake_from_file(path: object, **kwargs: object) -> object:
        captured["path"] = path
        captured_kwargs.update(kwargs)
        return expected

    monkeypatch.setattr(ChannelFrame, "from_file", staticmethod(fake_from_file))

    assert read(source) is expected
    assert captured["path"] is source
    assert captured_kwargs["file_type"] is None


@pytest.mark.parametrize("scheme", ["https", "HTTPS"])
def test_read_rejects_wdf_url_with_query_using_load_guidance(scheme: str) -> None:
    url = f"{scheme}://example.com/analysis.wdf?signature=abc#section"

    with pytest.raises(ValueError, match="wd.load") as exc_info:
        read(url)

    message = str(exc_info.value)
    assert "WDF files are loaded with wd.load(), not wd.read()" in message
    assert url in message
