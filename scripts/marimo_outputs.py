"""Read and rewrite HTML fragments serialized in static marimo exports."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping
from typing import Any, cast

_MOUNT_CONFIG = re.compile(
    r"Object\.defineProperty\(\s*window\s*,\s*['\"]__MARIMO_MOUNT_CONFIG__['\"]\s*,\s*"
    r"\{\s*value\s*:\s*Object\.freeze\(\s*"
)
_SESSION = re.compile(r'"session"\s*:\s*')
_NOTEBOOK = re.compile(r'"notebook"\s*:\s*')
_HTML_MIME_TYPES = {"text/html", "text/markdown"}
_MIME_BUNDLE = "application/vnd.marimo+mimebundle"


def _mount_bounds(document: str, *, required: bool) -> tuple[int, int] | None:
    mount_match = _MOUNT_CONFIG.search(document)
    if mount_match is None:
        if required:
            raise ValueError("learning export has no recognizable marimo mount config")
        return None
    script_end = document.find("</script>", mount_match.end())
    if script_end < 0:
        raise ValueError("marimo mount config script is not closed")
    return mount_match.end(), script_end


def _object_member_span(
    document: str,
    pattern: re.Pattern[str],
    bounds: tuple[int, int],
    *,
    label: str,
) -> tuple[int, int, dict[str, Any]]:
    invalid_value: json.JSONDecodeError | None = None
    for match in pattern.finditer(document, bounds[0], bounds[1]):
        try:
            value, consumed = json.JSONDecoder().raw_decode(document[match.end() :])
        except json.JSONDecodeError as exc:
            invalid_value = exc
            continue
        if isinstance(value, dict) and isinstance(value.get("cells"), list):
            return match.end(), match.end() + consumed, value
    if invalid_value is not None:
        raise ValueError(f"marimo mount config has an invalid serialized {label}: {invalid_value}")
    raise ValueError(f"marimo mount config has no serialized {label} object")


def _session_span(document: str, *, required: bool = False) -> tuple[int, int, dict[str, Any]] | None:
    """Return the rendered-session span from one marimo mount config."""
    bounds = _mount_bounds(document, required=required)
    if bounds is None:
        return None
    return _object_member_span(document, _SESSION, bounds, label="session")


def _require_source_code(document: str) -> None:
    bounds = _mount_bounds(document, required=True)
    assert bounds is not None
    _, _, notebook = _object_member_span(document, _NOTEBOOK, bounds, label="notebook")
    cells = notebook["cells"]
    if not all(isinstance(cell, Mapping) and isinstance(cell.get("code"), str) for cell in cells):
        raise ValueError("marimo mount config has an unrecognized notebook cell schema")
    if not any(cell["code"].strip() for cell in cells):
        raise ValueError("learning export does not include notebook source code")


def _data_mappings(session: Mapping[str, Any], *, strict: bool = False) -> Iterator[dict[str, Any]]:
    """Yield mutable MIME mappings from the supported marimo session shape."""
    cells = session.get("cells", [])
    if not isinstance(cells, list):
        return
    for cell_index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            if strict:
                raise ValueError(f"marimo session cell {cell_index} has an unrecognized schema")
            continue
        cell_mapping = cast(dict[str, Any], cell)
        if strict and "outputs" not in cell_mapping:
            raise ValueError(f"marimo session cell {cell_index} has no outputs container")
        outputs = cell_mapping.get("outputs", [])
        if not isinstance(outputs, list):
            if strict:
                raise ValueError(f"marimo session cell {cell_index} has unrecognized outputs")
            continue
        for output_index, output in enumerate(outputs):
            if not isinstance(output, dict):
                if strict:
                    raise ValueError(
                        f"marimo session cell {cell_index} output {output_index} has an unrecognized schema"
                    )
                continue
            output_mapping = cast(dict[str, Any], output)
            data = output_mapping.get("data")
            if isinstance(data, dict):
                yield data
            elif strict:
                raise ValueError(f"marimo session cell {cell_index} output {output_index} has unrecognized data")


def _mime_bundle(value: object, *, strict: bool = False) -> dict[str, Any] | None:
    if not isinstance(value, str):
        if strict and value is not None:
            raise ValueError("marimo MIME bundle has an unrecognized schema")
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        if strict:
            raise ValueError(f"marimo MIME bundle is invalid JSON: {exc}") from exc
        return None
    if isinstance(decoded, dict):
        return decoded
    if strict:
        raise ValueError("marimo MIME bundle has an unrecognized schema")
    return None


def marimo_html_outputs(
    document: str,
    *,
    require_session: bool = False,
    require_source: bool = False,
) -> list[str]:
    """Return rendered HTML fragments from one static marimo export."""
    span = _session_span(document, required=require_session)
    if span is None:
        return []
    if require_source:
        _require_source_code(document)
    fragments: list[str] = []
    for data in _data_mappings(span[2], strict=require_session):
        for mime_type in _HTML_MIME_TYPES:
            if mime_type not in data:
                continue
            value = data[mime_type]
            if not isinstance(value, str):
                if require_session:
                    raise ValueError(f"marimo {mime_type} output has an unrecognized schema")
                continue
            fragments.append(value)
        bundle = _mime_bundle(data.get(_MIME_BUNDLE), strict=require_session)
        if bundle is not None:
            for mime_type in _HTML_MIME_TYPES:
                if mime_type not in bundle:
                    continue
                value = bundle[mime_type]
                if not isinstance(value, str):
                    if require_session:
                        raise ValueError(f"marimo MIME bundle {mime_type} output has an unrecognized schema")
                    continue
                fragments.append(value)
    return fragments


def rewrite_marimo_html_outputs(
    document: str,
    transform: Callable[[str], str],
    *,
    require_session: bool = False,
) -> str:
    """Apply *transform* to every rendered HTML fragment in a static export."""
    span = _session_span(document, required=require_session)
    if span is None:
        return document
    start, end, session = span
    for data in _data_mappings(session, strict=require_session):
        for mime_type in _HTML_MIME_TYPES:
            if mime_type not in data:
                continue
            value = data[mime_type]
            if not isinstance(value, str):
                if require_session:
                    raise ValueError(f"marimo {mime_type} output has an unrecognized schema")
                continue
            data[mime_type] = transform(value)
        bundle = _mime_bundle(data.get(_MIME_BUNDLE), strict=require_session)
        if bundle is not None:
            changed = False
            for mime_type in _HTML_MIME_TYPES:
                if mime_type not in bundle:
                    continue
                value = bundle[mime_type]
                if not isinstance(value, str):
                    if require_session:
                        raise ValueError(f"marimo MIME bundle {mime_type} output has an unrecognized schema")
                    continue
                bundle[mime_type] = transform(value)
                changed = True
            if changed:
                data[_MIME_BUNDLE] = json.dumps(bundle, ensure_ascii=True, separators=(",", ":"))

    serialized = json.dumps(session, ensure_ascii=True, separators=(",", ":"))
    serialized = serialized.replace("<", r"\u003C").replace(">", r"\u003E").replace("&", r"\u0026")
    return document[:start] + serialized + document[end:]
