"""Read and rewrite HTML fragments serialized in static marimo exports."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping
from typing import Any

_MOUNT_CONFIG = re.compile(
    r"Object\.defineProperty\(\s*window\s*,\s*['\"]__MARIMO_MOUNT_CONFIG__['\"]\s*,\s*"
    r"\{\s*value\s*:\s*Object\.freeze\(\s*"
)
_SESSION = re.compile(r'"session"\s*:\s*')
_HTML_MIME_TYPES = {"text/html", "text/markdown"}
_MIME_BUNDLE = "application/vnd.marimo+mimebundle"


def _session_span(document: str) -> tuple[int, int, dict[str, Any]] | None:
    """Return the rendered-session span from one marimo mount config."""
    mount_match = _MOUNT_CONFIG.search(document)
    if mount_match is None:
        return None
    script_end = document.find("</script>", mount_match.end())
    if script_end < 0:
        raise ValueError("marimo mount config script is not closed")
    invalid_session: json.JSONDecodeError | None = None
    for match in _SESSION.finditer(document, mount_match.end(), script_end):
        try:
            value, consumed = json.JSONDecoder().raw_decode(document[match.end() :])
        except json.JSONDecodeError as exc:
            invalid_session = exc
            continue
        if isinstance(value, dict) and isinstance(value.get("cells"), list):
            return match.end(), match.end() + consumed, value
    if invalid_session is not None:
        raise ValueError(f"marimo mount config has an invalid serialized session: {invalid_session}")
    raise ValueError("marimo mount config has no serialized session object")


def _data_mappings(session: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield mutable MIME mappings from the supported marimo session shape."""
    cells = session.get("cells", [])
    if not isinstance(cells, list):
        return
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            continue
        for output in outputs:
            if not isinstance(output, Mapping):
                continue
            data = output.get("data")
            if isinstance(data, dict):
                yield data


def _mime_bundle(value: object) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def marimo_html_outputs(document: str) -> list[str]:
    """Return rendered HTML fragments from one static marimo export."""
    span = _session_span(document)
    if span is None:
        return []
    fragments: list[str] = []
    for data in _data_mappings(span[2]):
        for mime_type in _HTML_MIME_TYPES:
            value = data.get(mime_type)
            if isinstance(value, str):
                fragments.append(value)
        bundle = _mime_bundle(data.get(_MIME_BUNDLE))
        if bundle is not None:
            for mime_type in _HTML_MIME_TYPES:
                value = bundle.get(mime_type)
                if isinstance(value, str):
                    fragments.append(value)
    return fragments


def rewrite_marimo_html_outputs(document: str, transform: Callable[[str], str]) -> str:
    """Apply *transform* to every rendered HTML fragment in a static export."""
    span = _session_span(document)
    if span is None:
        return document
    start, end, session = span
    for data in _data_mappings(session):
        for mime_type in _HTML_MIME_TYPES:
            value = data.get(mime_type)
            if isinstance(value, str):
                data[mime_type] = transform(value)
        bundle = _mime_bundle(data.get(_MIME_BUNDLE))
        if bundle is not None:
            changed = False
            for mime_type in _HTML_MIME_TYPES:
                value = bundle.get(mime_type)
                if isinstance(value, str):
                    bundle[mime_type] = transform(value)
                    changed = True
            if changed:
                data[_MIME_BUNDLE] = json.dumps(bundle, ensure_ascii=True, separators=(",", ":"))

    serialized = json.dumps(session, ensure_ascii=True, separators=(",", ":"))
    serialized = serialized.replace("<", r"\u003C").replace(">", r"\u003E").replace("&", r"\u0026")
    return document[:start] + serialized + document[end:]
