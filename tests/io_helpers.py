"""Explicit test doubles shared by Wandas I/O contract tests."""

import io
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


@contextmanager
def mock_urlopen_stream(
    content: bytes,
    *,
    forbid_unbounded_read: bool = False,
    include_content_length: bool = True,
    expected_chunk_size: int | None = None,
    max_chunk_bytes: int | None = None,
) -> Iterator[MagicMock]:
    """Patch ``urlopen`` with a bounded in-memory byte stream.

    Args:
        content: Bytes returned by the simulated response.
        forbid_unbounded_read: Fail if production code requests ``read(-1)``.
        include_content_length: Add the HTTP ``Content-Length`` header.
        expected_chunk_size: Require every bounded read to request this size.
        max_chunk_bytes: Return at most this many bytes per read, even when the
            caller requests more. This simulates partial network reads.

    Yields:
        The patched ``urllib.request.urlopen`` mock.
    """
    stream = io.BytesIO(content)
    response = MagicMock()
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    response.headers = {"Content-Length": str(len(content))} if include_content_length else {}

    def read(size: int = -1) -> bytes:
        if forbid_unbounded_read and size < 0:
            raise AssertionError("URL reads must request bounded chunks")
        if expected_chunk_size is not None and size >= 0:
            assert size == expected_chunk_size, f"Expected chunk size {expected_chunk_size}, got {size}"
        returned_size = min(size, max_chunk_bytes) if max_chunk_bytes is not None and size >= 0 else size
        return stream.read(returned_size)

    response.read = MagicMock(side_effect=read)
    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        yield urlopen
