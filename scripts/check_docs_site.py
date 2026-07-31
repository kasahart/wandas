"""Crawl a generated documentation site and validate its deployment contract."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse
from xml.etree import ElementTree

IGNORED_SCHEMES = {"data", "javascript", "mailto", "tel", "blob"}
EDIT_PREFIX = "https://github.com/kasahart/wandas/edit/main/docs/src/"
_CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", flags=re.IGNORECASE)
_CSS_IMPORT = re.compile(r"@import\s+(['\"])(.*?)\1", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Reference:
    attribute: str
    target: str
    line: int


@dataclass
class HtmlDocument:
    path: Path
    ids: set[str] = field(default_factory=set)
    references: list[Reference] = field(default_factory=list)
    canonicals: list[str] = field(default_factory=list)
    edit_links: list[str] = field(default_factory=list)
    base_hrefs: list[str] = field(default_factory=list)


def _parse_css_content(content: str, *, line_offset: int = 0) -> list[Reference]:
    """Return dependency candidates declared by CSS source text."""
    references: list[Reference] = []
    for attribute, pattern in (("css-url", _CSS_URL), ("css-import", _CSS_IMPORT)):
        for match in pattern.finditer(content):
            references.append(
                Reference(
                    attribute,
                    match.group(2).strip(),
                    line_offset + content.count("\n", 0, match.start()) + 1,
                )
            )
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
                self.document.references.append(Reference(attribute, target, self.getpos()[0]))
        if srcset := values.get("srcset"):
            for target in _srcset_targets(srcset):
                self.document.references.append(Reference("srcset", target, self.getpos()[0]))
        if style := values.get("style"):
            self.document.references.extend(_parse_css_content(style, line_offset=self.getpos()[0] - 1))

        rel = values.get("rel", "").lower().split()
        if tag == "link" and "canonical" in rel and (canonical := values.get("href")):
            self.document.canonicals.append(canonical)
        if tag == "a" and (href := values.get("href", "")).startswith(EDIT_PREFIX):
            self.document.edit_links.append(href)

    def handle_data(self, data: str) -> None:
        if self._style_depth:
            self.document.references.extend(_parse_css_content(data, line_offset=self.getpos()[0] - 1))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "style" and self._style_depth:
            self._style_depth -= 1


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


def _parse_html(path: Path) -> HtmlDocument:
    parser = DocumentParser(path)
    parser.feed(path.read_text(encoding="utf-8"))
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
    project_prefix = base.path.rstrip("/") + "/"

    html_paths = sorted(site_dir.rglob("*.html"))
    documents = {path.resolve(): _parse_html(path) for path in html_paths}
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
        if resolved.scheme not in {"http", "https"} or resolved.netloc != base.netloc:
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
        document_url = urljoin(base_url, _public_path(relative))
        reference_url = document_url
        if document.base_hrefs:
            base_href = document.base_hrefs[0]
            candidate = urlparse(urljoin(document_url, base_href))
            if candidate.scheme in {"http", "https"} and candidate.netloc:
                reference_url = candidate.geturl()
            else:
                errors.append(f"{relative}: base href {base_href!r} is not a deployable HTTP(S) URL")
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
            if target_path is not None and target_path.suffix.lower() == ".css":
                css_to_visit.append(target_path)

    while css_to_visit:
        css_path = css_to_visit.pop()
        if css_path in visited_css:
            continue
        visited_css.add(css_path)
        css_relative = PurePosixPath(css_path.relative_to(site_dir).as_posix())
        css_url = urljoin(base_url, css_relative.as_posix())
        for reference in _parse_css(css_path):
            target_path = check_reference(css_relative, css_url, reference)
            if target_path is not None and target_path.suffix.lower() == ".css":
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
