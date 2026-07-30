"""Finalize exported learning applications for the GitHub Pages project site."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin

LEARNING_HTML_GLOB = "0[0-8]_*.html"
PYTHON_HREF = re.compile(
    r"""(?P<prefix>\bhref\s*=\s*["'])(?P<target>[^"'#?]+)\.py"""
    r"""(?P<suffix>[#?][^"']*)?(?P<quote>["'])"""
)
CANONICAL_LINK = re.compile(
    r"""<link\b[^>]*\brel\s*=\s*["'][^"']*\bcanonical\b[^"']*["'][^>]*>""",
    flags=re.IGNORECASE,
)


def finalize_learning_html(site_dir: Path, site_url: str) -> list[Path]:
    """Rewrite source-only links and install canonical metadata."""
    learning_dir = site_dir / "learning-path"
    html_paths = sorted(learning_dir.glob(LEARNING_HTML_GLOB))
    if len(html_paths) != 9:
        raise ValueError(f"expected 9 exported learning applications in {learning_dir}, found {len(html_paths)}")

    base_url = site_url.rstrip("/") + "/"
    for html_path in html_paths:
        html = html_path.read_text(encoding="utf-8")
        html = PYTHON_HREF.sub(
            lambda match: (
                f"{match.group('prefix')}{match.group('target')}.html"
                f"{match.group('suffix') or ''}{match.group('quote')}"
            ),
            html,
        )
        canonical_url = urljoin(base_url, f"learning-path/{html_path.name}")
        canonical = f'<link rel="canonical" href="{canonical_url}">'
        if CANONICAL_LINK.search(html):
            html = CANONICAL_LINK.sub(canonical, html, count=1)
        elif re.search(r"</head\s*>", html, flags=re.IGNORECASE):
            html = re.sub(r"</head\s*>", f"  {canonical}\n</head>", html, count=1, flags=re.IGNORECASE)
        else:
            raise ValueError(f"{html_path} has no HTML head")
        html_path.write_text(html, encoding="utf-8")

    return html_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_dir", type=Path)
    parser.add_argument("--site-url", required=True)
    args = parser.parse_args()
    finalized = finalize_learning_html(args.site_dir, args.site_url)
    print(f"Finalized {len(finalized)} learning applications")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
