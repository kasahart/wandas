"""Crawl a generated documentation site and validate its deployment contract."""

from __future__ import annotations

import argparse
import posixpath
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urljoin, urlparse
from xml.etree import ElementTree

if __package__:
    from .marimo_outputs import marimo_html_outputs
else:
    from marimo_outputs import marimo_html_outputs

IGNORED_SCHEMES = {"data", "javascript", "mailto", "tel", "blob"}
EDIT_PREFIX = "https://github.com/kasahart/wandas/edit/main/docs/src/"
_LEARNING_EXPORT = re.compile(r"learning-path/0[0-8]_[^/]+\.html\Z")


@dataclass(frozen=True)
class Reference:
    attribute: str
    target: str
    line: int
    kind: Literal["resource", "stylesheet"] = "resource"


@dataclass
class HtmlDocument:
    path: Path
    ids: set[str] = field(default_factory=set)
    references: list[Reference] = field(default_factory=list)
    canonicals: list[str] = field(default_factory=list)
    edit_links: list[str] = field(default_factory=list)
    base_hrefs: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


def _css_comment_end(content: str, position: int) -> int:
    end = content.find("*/", position + 2)
    if end < 0:
        raise ValueError("unterminated CSS comment")
    return end + 2


def _css_string(content: str, position: int) -> tuple[str, int]:
    quote = content[position]
    start = position + 1
    position = start
    while position < len(content):
        if content[position] == "\\":
            position += 2
            continue
        if content[position] == quote:
            return content[start:position], position + 1
        position += 1
    raise ValueError("unterminated CSS string")


def _decode_css_escapes(value: str) -> str:
    """Decode CSS escapes in one URL or import token."""
    decoded: list[str] = []
    position = 0
    while position < len(value):
        if value[position] != "\\":
            decoded.append(value[position])
            position += 1
            continue
        position += 1
        if position >= len(value):
            raise ValueError("unterminated CSS escape")
        if value[position] in "\r\n\f":
            if value[position] == "\r" and position + 1 < len(value) and value[position + 1] == "\n":
                position += 1
            position += 1
            continue
        hex_start = position
        while position < len(value) and position - hex_start < 6 and value[position].lower() in "0123456789abcdef":
            position += 1
        if position > hex_start:
            codepoint = int(value[hex_start:position], 16)
            if codepoint == 0 or codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
                decoded.append("\N{REPLACEMENT CHARACTER}")
            else:
                decoded.append(chr(codepoint))
            if position < len(value) and value[position].isspace():
                if value[position] == "\r" and position + 1 < len(value) and value[position + 1] == "\n":
                    position += 1
                position += 1
            continue
        decoded.append(value[position])
        position += 1
    return "".join(decoded)


def _css_trivia_end(content: str, position: int) -> int:
    while position < len(content):
        if content[position].isspace():
            position += 1
        elif content.startswith("/*", position):
            position = _css_comment_end(content, position)
        else:
            break
    return position


def _css_url(content: str, position: int) -> tuple[str, int] | None:
    after_name = position + 3
    if position and (content[position - 1].isalnum() or content[position - 1] in "_-"):
        return None
    after_name = _css_trivia_end(content, after_name)
    if after_name >= len(content) or content[after_name] != "(":
        return None
    value_start = _css_trivia_end(content, after_name + 1)
    if value_start >= len(content):
        raise ValueError("unterminated CSS url()")
    if content[value_start] in {'"', "'"}:
        value, position = _css_string(content, value_start)
        position = _css_trivia_end(content, position)
        if position >= len(content) or content[position] != ")":
            raise ValueError("unterminated CSS url()")
        return _decode_css_escapes(value), position + 1
    position = value_start
    while position < len(content):
        if content[position] == "\\":
            position += 2
            continue
        if content[position] == ")":
            raw_value = content[value_start:position].strip()
            return _decode_css_escapes(raw_value), position + 1
        position += 1
    raise ValueError("unterminated CSS url()")


def _parse_css_content(content: str, *, line_offset: int = 0) -> list[Reference]:
    """Return dependency candidates declared by CSS source text."""
    references: list[Reference] = []
    lowered = content.lower()
    position = 0
    while position < len(content):
        if content.startswith("/*", position):
            position = _css_comment_end(content, position)
            continue
        if content[position] in {'"', "'"}:
            _, position = _css_string(content, position)
            continue
        if lowered.startswith("@import", position) and (
            position + 7 == len(content) or not (content[position + 7].isalnum() or content[position + 7] in "_-")
        ):
            start = position
            value_start = _css_trivia_end(content, position + 7)
            parsed_url = _css_url(content, value_start) if lowered.startswith("url", value_start) else None
            if parsed_url is not None:
                target, position = parsed_url
            elif value_start < len(content) and content[value_start] in {'"', "'"}:
                target, position = _css_string(content, value_start)
                target = _decode_css_escapes(target)
            else:
                position = value_start + 1
                continue
            references.append(
                Reference(
                    "css-import",
                    target.strip(),
                    line_offset + content.count("\n", 0, start) + 1,
                    "stylesheet",
                )
            )
            continue
        if lowered.startswith("url", position):
            parsed_url = _css_url(content, position)
            if parsed_url is not None:
                target, end = parsed_url
                references.append(
                    Reference(
                        "css-url",
                        target.strip(),
                        line_offset + content.count("\n", 0, position) + 1,
                    )
                )
                position = end
                continue
        if content[position] == "\\":
            position += 2
        else:
            position += 1
    return references


def _srcset_targets(srcset: str) -> list[str]:
    """Return srcset URL tokens without splitting commas inside URL data."""
    targets: list[str] = []
    position = 0
    length = len(srcset)
    while position < length:
        while position < length and (srcset[position].isspace() or srcset[position] == ","):
            position += 1
        start = position
        while position < length and not srcset[position].isspace():
            position += 1
        target = srcset[start:position]
        if not target:
            break
        if target.endswith(","):
            target = target.rstrip(",")
        else:
            parentheses = 0
            while position < length:
                character = srcset[position]
                if character == "(":
                    parentheses += 1
                elif character == ")" and parentheses:
                    parentheses -= 1
                elif character == "," and not parentheses:
                    position += 1
                    break
                position += 1
        if target:
            targets.append(target)
    return targets


class DocumentParser(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.document = HtmlDocument(path)
        self._style_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value for name, value in attrs if value is not None}
        tag = tag.lower()
        rel = values.get("rel", "").lower().split()
        css_mime = values.get("type", "").partition(";")[0].strip().lower() == "text/css"
        stylesheet_link = tag == "link" and ("stylesheet" in rel or values.get("as", "").lower() == "style" or css_mime)
        if tag == "style":
            self._style_depth += 1
        if element_id := values.get("id"):
            self.document.ids.add(element_id)
        if tag == "a" and (anchor_name := values.get("name")):
            self.document.ids.add(anchor_name)

        if tag == "base" and (base_href := values.get("href")):
            self.document.base_hrefs.append(base_href)

        for attribute in ("href", "src", "poster", "data"):
            if tag == "base" and attribute == "href":
                continue
            if target := values.get(attribute):
                kind = "stylesheet" if stylesheet_link and attribute == "href" else "resource"
                self.document.references.append(Reference(attribute, target, self.getpos()[0], kind))
        if srcset := values.get("srcset"):
            for target in _srcset_targets(srcset):
                self.document.references.append(Reference("srcset", target, self.getpos()[0]))
        if style := values.get("style"):
            self._record_css_references(style, line_offset=self.getpos()[0] - 1)

        if tag == "link" and "canonical" in rel and (canonical := values.get("href")):
            self.document.canonicals.append(canonical)
        if tag == "a" and (href := values.get("href", "")).startswith(EDIT_PREFIX):
            self.document.edit_links.append(href)

    def handle_data(self, data: str) -> None:
        if self._style_depth:
            self._record_css_references(data, line_offset=self.getpos()[0] - 1)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "style" and self._style_depth:
            self._style_depth -= 1

    def _record_css_references(self, content: str, *, line_offset: int) -> None:
        try:
            self.document.references.extend(_parse_css_content(content, line_offset=line_offset))
        except ValueError as exc:
            self.document.parse_errors.append(f"CSS on line {line_offset + 1}: {exc}")


def _public_path(relative_html: PurePosixPath) -> str:
    value = relative_html.as_posix()
    if value == "index.html":
        return ""
    if value.endswith("/index.html"):
        return value[: -len("index.html")]
    return value


def _output_path(site_dir: Path, public_path: str) -> Path:
    decoded = unquote(public_path).lstrip("/")
    candidate = site_dir / decoded
    if public_path.endswith("/"):
        return candidate / "index.html"
    if candidate.is_file():
        return candidate
    return candidate / "index.html"


def _parse_html(path: Path, *, require_marimo_session: bool = False) -> HtmlDocument:
    parser = DocumentParser(path)
    content = path.read_text(encoding="utf-8")
    parser.feed(content)
    try:
        for fragment in marimo_html_outputs(
            content,
            require_session=require_marimo_session,
            require_source=require_marimo_session,
        ):
            parser.feed(fragment)
    except ValueError as exc:
        parser.document.parse_errors.append(str(exc))
    parser.close()
    return parser.document


def _parse_css(path: Path) -> list[Reference]:
    """Return local-dependency candidates declared by one stylesheet."""
    return _parse_css_content(path.read_text(encoding="utf-8"))


def check_site(site_dir: Path, source_dir: Path, site_url: str) -> list[str]:
    """Return every generated-site contract violation."""
    errors: list[str] = []
    site_dir = site_dir.resolve()
    source_dir = source_dir.resolve()
    base_url = site_url.rstrip("/") + "/"
    base = urlparse(base_url)
    base_host = (base.hostname or "").lower()
    base_port = base.port or (443 if base.scheme == "https" else 80)
    project_prefix = base.path.rstrip("/") + "/"

    html_paths = sorted(site_dir.rglob("*.html"))
    documents = {
        path.resolve(): _parse_html(
            path,
            require_marimo_session=bool(_LEARNING_EXPORT.fullmatch(path.relative_to(site_dir).as_posix())),
        )
        for path in html_paths
    }
    if not html_paths:
        errors.append(f"{site_dir}: no HTML files found")

    css_to_visit: list[Path] = []
    visited_css: set[Path] = set()

    def check_reference(
        origin: PurePosixPath,
        origin_url: str,
        reference: Reference,
    ) -> Path | None:
        raw_target = reference.target.strip()
        parsed_raw = urlparse(raw_target)
        if not raw_target or parsed_raw.scheme.lower() in IGNORED_SCHEMES or raw_target == "#":
            return None
        if parsed_raw.scheme.lower() == "file":
            errors.append(f"{origin}:{reference.line}: {reference.attribute} {raw_target!r} uses forbidden file URL")
            return None
        resolved = urlparse(urljoin(origin_url, raw_target))
        resolved_host = (resolved.hostname or "").lower()
        if resolved_host == base_host and base.scheme == "https" and resolved.scheme == "http":
            errors.append(f"{origin}:{reference.line}: {reference.attribute} {raw_target!r} uses mixed-content HTTP")
            return None
        try:
            resolved_port = resolved.port or (443 if resolved.scheme == "https" else 80)
        except ValueError:
            errors.append(f"{origin}:{reference.line}: {reference.attribute} {raw_target!r} has an invalid port")
            return None
        if resolved.scheme != base.scheme or resolved_host != base_host or resolved_port != base_port:
            return None
        decoded_path = unquote(resolved.path)
        if not decoded_path.startswith(project_prefix):
            errors.append(
                f"{origin}:{reference.line}: {reference.attribute} {raw_target!r} escapes "
                f"project prefix {project_prefix!r}"
            )
            return None
        target_public_path = decoded_path[len(project_prefix) :]
        target_path = _output_path(site_dir, target_public_path).resolve()
        try:
            target_relative = target_path.relative_to(site_dir)
        except ValueError:
            errors.append(f"{origin}:{reference.line}: {raw_target!r} escapes the generated site")
            return None
        if not target_path.is_file():
            errors.append(
                f"{origin}:{reference.line}: {reference.attribute} {raw_target!r} targets missing {target_relative}"
            )
            return None
        if resolved.fragment and target_path.suffix.lower() == ".html":
            target_document = documents.get(target_path)
            fragment = unquote(resolved.fragment)
            if target_document is None or fragment not in target_document.ids:
                errors.append(f"{origin}:{reference.line}: {raw_target!r} targets missing fragment #{fragment}")
        return target_path

    for path, document in documents.items():
        relative = PurePosixPath(path.relative_to(site_dir).as_posix())
        errors.extend(f"{relative}: {error}" for error in document.parse_errors)
        document_url = urljoin(base_url, _public_path(relative))
        reference_url = document_url
        if document.base_hrefs:
            base_href = document.base_hrefs[0]
            candidate = urlparse(urljoin(document_url, base_href))
            candidate_host = (candidate.hostname or "").lower()
            try:
                candidate_port = candidate.port or (443 if candidate.scheme == "https" else 80)
            except ValueError:
                candidate_port = -1
            candidate_path = unquote(candidate.path)
            normalized_candidate_path = posixpath.normpath(candidate_path)
            if (
                candidate.scheme == base.scheme
                and candidate_host == base_host
                and candidate_port == base_port
                and (
                    normalized_candidate_path == project_prefix.rstrip("/")
                    or normalized_candidate_path.startswith(project_prefix)
                )
            ):
                reference_url = candidate.geturl()
            else:
                errors.append(f"{relative}: base href {base_href!r} is outside the deployment origin or project prefix")
        expected_canonical = document_url
        if relative.as_posix() != "404.html":
            if len(document.canonicals) != 1:
                errors.append(f"{relative}: expected exactly one canonical link, found {len(document.canonicals)}")
            elif urljoin(reference_url, document.canonicals[0]) != expected_canonical:
                errors.append(f"{relative}: canonical {document.canonicals[0]!r} does not match {expected_canonical!r}")

        if relative.as_posix() == "index.html":
            source_relative = PurePosixPath("index.md")
        elif relative.name == "index.html":
            flat_source = relative.parent.with_suffix(".md")
            nested_source = relative.parent / "index.md"
            source_relative = flat_source if (source_dir / flat_source).is_file() else nested_source
        else:
            source_relative = relative.with_suffix(".md")
        if (source_dir / source_relative).is_file():
            expected_edit = EDIT_PREFIX + source_relative.as_posix()
            if document.edit_links != [expected_edit]:
                errors.append(f"{relative}: expected edit link {expected_edit!r}, found {document.edit_links!r}")

        for reference in document.references:
            target_path = check_reference(relative, reference_url, reference)
            if target_path is not None and reference.kind == "stylesheet":
                css_to_visit.append(target_path)

    while css_to_visit:
        css_path = css_to_visit.pop()
        if css_path in visited_css:
            continue
        visited_css.add(css_path)
        css_relative = PurePosixPath(css_path.relative_to(site_dir).as_posix())
        css_url = urljoin(base_url, css_relative.as_posix())
        try:
            references = _parse_css(css_path)
        except ValueError as exc:
            errors.append(f"{css_relative}: {exc}")
            continue
        for reference in references:
            target_path = check_reference(css_relative, css_url, reference)
            if target_path is not None and reference.kind == "stylesheet":
                css_to_visit.append(target_path)

    sitemap = site_dir / "sitemap.xml"
    if not sitemap.is_file():
        errors.append("sitemap.xml: missing")
    else:
        try:
            root = ElementTree.parse(sitemap).getroot()
        except ElementTree.ParseError as exc:
            errors.append(f"sitemap.xml: invalid XML: {exc}")
        else:
            locations = [element.text or "" for element in root.iter() if element.tag.endswith("loc")]
            if not locations:
                errors.append("sitemap.xml: contains no locations")
            for location in locations:
                parsed = urlparse(location)
                if (
                    parsed.scheme != base.scheme
                    or parsed.netloc != base.netloc
                    or not parsed.path.startswith(project_prefix)
                ):
                    errors.append(f"sitemap.xml: location {location!r} is outside {base_url!r}")
                    continue
                decoded_path = unquote(parsed.path)
                target = _output_path(site_dir, decoded_path[len(project_prefix) :]).resolve()
                try:
                    target_relative = target.relative_to(site_dir)
                except ValueError:
                    errors.append(f"sitemap.xml: location {location!r} escapes the generated site")
                    continue
                if not target.is_file():
                    errors.append(f"sitemap.xml: location {location!r} targets missing {target_relative}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_dir", type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path("docs/src"))
    parser.add_argument("--site-url", required=True)
    args = parser.parse_args()
    errors = check_site(args.site_dir, args.source_dir, args.site_url)
    if errors:
        print("Generated documentation site is invalid:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Generated documentation site is valid: {args.site_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
