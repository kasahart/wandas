import json
from pathlib import Path

import pytest

from scripts.check_docs_site import check_site
from scripts.finalize_learning_html import finalize_learning_html
from scripts.marimo_outputs import marimo_html_outputs

SITE_URL = "https://kasahart.github.io/wandas/"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _marimo_mount(fragment: str, *, bundle_fragment: str | None = None) -> str:
    data = {"text/markdown": fragment}
    if bundle_fragment is not None:
        data["application/vnd.marimo+mimebundle"] = json.dumps({"text/html": bundle_fragment})
    session = {
        "cells": [
            {
                "outputs": [
                    {
                        "data": data,
                        "type": "data",
                    }
                ]
            }
        ]
    }
    serialized = json.dumps(session).replace("<", r"\u003C").replace(">", r"\u003E")
    return (
        '<script data-marimo="true">Object.defineProperty(window, '
        '"__MARIMO_MOUNT_CONFIG__", {value: Object.freeze({'
        f'"session": {serialized}'
        "})});</script>"
    )


@pytest.fixture()
def valid_site(tmp_path: Path) -> tuple[Path, Path]:
    site = tmp_path / "site"
    source = tmp_path / "src"
    _write(source / "index.md", "# Home")
    _write(source / "page.md", "# Page")
    _write(site / "assets/app.js", "console.log('ok')")
    _write(
        site / "index.html",
        """
        <html><head>
          <link rel="canonical" href="https://kasahart.github.io/wandas/">
          <script src="/wandas/assets/app.js"></script>
        </head><body id="home">
          <a href="/wandas/page/#section">Page</a>
          <a href="/wandas/learning-path/00_intro.html">Learn</a>
          <a href="https://github.com/kasahart/wandas/edit/main/docs/src/index.md">Edit</a>
        </body></html>
        """,
    )
    _write(
        site / "page/index.html",
        """
        <html><head>
          <link rel="canonical" href="https://kasahart.github.io/wandas/page/">
        </head><body id="section">
          <a href="/wandas/#home">Home</a>
          <a href="https://github.com/kasahart/wandas/edit/main/docs/src/page.md">Edit</a>
        </body></html>
        """,
    )
    _write(
        site / "learning-path/00_intro.html",
        """
        <html><head>
          <link rel="canonical" href="https://kasahart.github.io/wandas/learning-path/00_intro.html">
        </head><body><a href="../page/#section">Page</a></body></html>
        """,
    )
    _write(
        site / "sitemap.xml",
        """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://kasahart.github.io/wandas/</loc></url>
          <url><loc>https://kasahart.github.io/wandas/page/</loc></url>
        </urlset>
        """,
    )
    return site, source


def test_generated_site_contract_accepts_valid_project_site(valid_site: tuple[Path, Path]) -> None:
    site, source = valid_site
    assert check_site(site, source, SITE_URL) == []


def test_generated_site_contract_accepts_data_url_comma_in_srcset(
    valid_site: tuple[Path, Path],
) -> None:
    site, source = valid_site
    index = site / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            "</body>",
            '<img srcset="data:image/svg+xml,%3Csvg%3E%3C/svg%3E 1x, /wandas/assets/app.js 2x"></body>',
        ),
        encoding="utf-8",
    )

    assert check_site(site, source, SITE_URL) == []


@pytest.mark.parametrize(
    ("relative_path", "old", "new", "message"),
    [
        ("index.html", "/wandas/assets/app.js", "/wandas/assets/missing.js", "targets missing"),
        ("index.html", "/wandas/assets/app.js", "file:///tmp/app.js", "forbidden file URL"),
        ("index.html", "/wandas/page/#section", "/wandas/page/#missing", "missing fragment"),
        ("index.html", "/wandas/page/#section", "/outside/page/", "escapes project prefix"),
        (
            "page/index.html",
            '<link rel="canonical" href="https://kasahart.github.io/wandas/page/">',
            '<link rel="canonical" href="https://example.com/page/">',
            "does not match",
        ),
        (
            "page/index.html",
            "https://github.com/kasahart/wandas/edit/main/docs/src/page.md",
            "https://github.com/kasahart/wandas/edit/main/docs/src/wrong.md",
            "expected edit link",
        ),
        (
            "sitemap.xml",
            "https://kasahart.github.io/wandas/page/",
            "https://wandas.github.io/page/",
            "outside",
        ),
    ],
)
def test_generated_site_contract_rejects_broken_fixture(
    valid_site: tuple[Path, Path],
    relative_path: str,
    old: str,
    new: str,
    message: str,
) -> None:
    site, source = valid_site
    path = site / relative_path
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    errors = check_site(site, source, SITE_URL)

    assert any(message in error for error in errors), errors


def test_generated_site_contract_rejects_duplicate_canonical(valid_site: tuple[Path, Path]) -> None:
    site, source = valid_site
    path = site / "page/index.html"
    html = path.read_text(encoding="utf-8").replace(
        "</head>",
        '<link rel="canonical" href="https://kasahart.github.io/wandas/page/"></head>',
    )
    path.write_text(html, encoding="utf-8")

    assert any("exactly one canonical" in error for error in check_site(site, source, SITE_URL))


def test_generated_site_contract_rejects_encoded_sitemap_traversal(
    valid_site: tuple[Path, Path],
) -> None:
    site, source = valid_site
    _write(site.parent / "outside.md", "This file is not deployed.")
    sitemap = site / "sitemap.xml"
    sitemap.write_text(
        sitemap.read_text(encoding="utf-8").replace(
            "https://kasahart.github.io/wandas/page/",
            "https://kasahart.github.io/wandas/%2e%2e/outside.md",
        ),
        encoding="utf-8",
    )

    errors = check_site(site, source, SITE_URL)

    assert any("sitemap.xml" in error and "escapes the generated site" in error for error in errors), errors


def test_generated_site_contract_recursively_checks_stylesheet_dependencies(
    valid_site: tuple[Path, Path],
) -> None:
    site, source = valid_site
    index = site / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            "</head>",
            '<link rel="stylesheet" href="/wandas/assets/main.css"></head>',
        ),
        encoding="utf-8",
    )
    _write(site / "assets/main.css", '@import "nested/theme.css";')
    _write(site / "assets/nested/theme.css", 'body { background: url("../../images/missing.png"); }')

    errors = check_site(site, source, SITE_URL)

    assert any("assets/nested/theme.css" in error and "images/missing.png" in error for error in errors), errors

    _write(site / "images/missing.png", "placeholder")
    assert check_site(site, source, SITE_URL) == []


def test_generated_site_contract_respects_first_valid_html_base(
    valid_site: tuple[Path, Path],
) -> None:
    site, source = valid_site
    index = site / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8")
        .replace("</head>", '<base href="/wandas/assets/"></head>')
        .replace('src="/wandas/assets/app.js"', 'src="app.js"'),
        encoding="utf-8",
    )

    assert check_site(site, source, SITE_URL) == []


def test_generated_site_contract_checks_inline_css_dependencies(
    valid_site: tuple[Path, Path],
) -> None:
    site, source = valid_site
    index = site / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8")
        .replace("</head>", '<style>.hero { background: url("/wandas/images/style.png"); }</style></head>')
        .replace('<body id="home">', '<body id="home" style="mask: url(\'/wandas/images/attribute.svg\')">'),
        encoding="utf-8",
    )

    errors = check_site(site, source, SITE_URL)

    assert any("images/style.png" in error and "targets missing" in error for error in errors), errors
    assert any("images/attribute.svg" in error and "targets missing" in error for error in errors), errors

    _write(site / "images/style.png", "placeholder")
    _write(site / "images/attribute.svg", "placeholder")
    assert check_site(site, source, SITE_URL) == []


def test_generated_site_contract_checks_serialized_marimo_outputs(
    valid_site: tuple[Path, Path],
) -> None:
    site, source = valid_site
    lesson = site / "learning-path/00_intro.html"
    lesson.write_text(
        lesson.read_text(encoding="utf-8").replace(
            "</body>",
            _marimo_mount(
                '<a href="/wandas/page/#missing-output">Broken fragment</a>'
                '<img src="/wandas/images/output.png">'
                "<span style=\"mask:url('/wandas/images/output.svg')\"></span>",
                bundle_fragment='<img src="/wandas/images/bundle.png">',
            )
            + "</body>",
        ),
        encoding="utf-8",
    )

    errors = check_site(site, source, SITE_URL)

    assert any("missing fragment #missing-output" in error for error in errors), errors
    assert any("images/output.png" in error and "targets missing" in error for error in errors), errors
    assert any("images/output.svg" in error and "targets missing" in error for error in errors), errors
    assert any("images/bundle.png" in error and "targets missing" in error for error in errors), errors


def test_finalize_learning_html_rewrites_navigation_and_adds_canonical(tmp_path: Path) -> None:
    site = tmp_path / "site"
    for index in range(9):
        _write(
            site / "learning-path" / f"0{index}_lesson.html",
            '<html><head></head><body><a href="01_lesson.py#part">Next</a>'
            + _marimo_mount(
                '<a href="./01_lesson.py?mode=read#part">Rendered next</a>'
                '<a href="https://example.com/demo.py">External source</a>',
                bundle_fragment='<a href="../learning-path/01_lesson.py">Bundle next</a>',
            )
            + "</body></html>",
        )

    finalized = finalize_learning_html(site, SITE_URL)

    assert len(finalized) == 9
    for index, path in enumerate(finalized):
        html = path.read_text(encoding="utf-8")
        assert 'href="01_lesson.html#part"' in html
        rendered = marimo_html_outputs(html)
        assert len(rendered) == 2
        assert any('href="./01_lesson.html?mode=read#part"' in fragment for fragment in rendered)
        assert any('href="https://example.com/demo.py"' in fragment for fragment in rendered)
        assert any('href="../learning-path/01_lesson.html"' in fragment for fragment in rendered)
        assert f'<link rel="canonical" href="{SITE_URL}learning-path/0{index}_lesson.html">' in html
        assert ".py#" not in html
