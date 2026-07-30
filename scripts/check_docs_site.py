"""Crawl a generated documentation site and validate its deployment contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse
from xml.etree import ElementTree

IGNORED_SCHEMES = {"data", "javascript", "mailto", "tel", "blob"}
EDIT_PREFIX = "https://github.com/kasahart/wandas/edit/main/docs/src/"


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


class DocumentParser(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.document = HtmlDocument(path)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value for name, value in attrs if value is not None}
        if element_id := values.get("id"):
            self.document.ids.add(element_id)
        if tag.lower() == "a" and (anchor_name := values.get("name")):
            self.document.ids.add(anchor_name)

        for attribute in ("href", "src", "poster", "data"):
            if target := values.get(attribute):
                self.document.references.append(Reference(attribute, target, self.getpos()[0]))
        if srcset := values.get("srcset"):
            for candidate in srcset.split(","):
                target = candidate.strip().split(maxsplit=1)[0]
                if target:
                    self.document.references.append(Reference("srcset", target, self.getpos()[0]))

        rel = values.get("rel", "").lower().split()
        if tag.lower() == "link" and "canonical" in rel and (canonical := values.get("href")):
            self.document.canonicals.append(canonical)
        if tag.lower() == "a" and (href := values.get("href", "")).startswith(EDIT_PREFIX):
            self.document.edit_links.append(href)


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

    for path, document in documents.items():
        relative = PurePosixPath(path.relative_to(site_dir).as_posix())
        document_url = urljoin(base_url, _public_path(relative))
        expected_canonical = document_url
        if relative.as_posix() != "404.html":
            if len(document.canonicals) != 1:
                errors.append(f"{relative}: expected exactly one canonical link, found {len(document.canonicals)}")
            elif urljoin(document_url, document.canonicals[0]) != expected_canonical:
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
            raw_target = reference.target.strip()
            parsed_raw = urlparse(raw_target)
            if not raw_target or parsed_raw.scheme.lower() in IGNORED_SCHEMES or raw_target == "#":
                continue
            resolved = urlparse(urljoin(document_url, raw_target))
            if resolved.scheme not in {"http", "https"} or resolved.netloc != base.netloc:
                continue
            decoded_path = unquote(resolved.path)
            if not decoded_path.startswith(project_prefix):
                errors.append(
                    f"{relative}:{reference.line}: {reference.attribute} {raw_target!r} escapes "
                    f"project prefix {project_prefix!r}"
                )
                continue
            target_public_path = decoded_path[len(project_prefix) :]
            target_path = _output_path(site_dir, target_public_path).resolve()
            try:
                target_path.relative_to(site_dir)
            except ValueError:
                errors.append(f"{relative}:{reference.line}: {raw_target!r} escapes the generated site")
                continue
            if not target_path.is_file():
                errors.append(
                    f"{relative}:{reference.line}: {reference.attribute} {raw_target!r} "
                    f"targets missing {target_path.relative_to(site_dir)}"
                )
                continue
            if resolved.fragment and target_path.suffix.lower() == ".html":
                target_document = documents.get(target_path)
                fragment = unquote(resolved.fragment)
                if target_document is None or fragment not in target_document.ids:
                    errors.append(f"{relative}:{reference.line}: {raw_target!r} targets missing fragment #{fragment}")

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
                target = _output_path(site_dir, unquote(parsed.path)[len(project_prefix) :])
                if not target.is_file():
                    errors.append(f"sitemap.xml: location {location!r} targets missing {target.relative_to(site_dir)}")

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
