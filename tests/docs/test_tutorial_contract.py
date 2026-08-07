import os
import posixpath
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

from scripts.learning_path_i18n import load_manifest, output_path

REPO_ROOT = Path(__file__).resolve().parents[2]
MKDOCS_CONFIG = REPO_ROOT / "docs/mkdocs.yml"


class _CodeTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_code = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "code":
            self._in_code = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "code":
            self._in_code = False

    def handle_data(self, data: str) -> None:
        if self._in_code:
            self.parts.append(data)


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.hrefs.extend(value for name, value in attrs if name == "href" and value is not None)


@pytest.fixture(scope="module")
def built_site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the docs once for the executable and navigation contracts."""
    pytest.importorskip("mkdocs", reason="MkDocs is installed only in the docs dependency group")
    site_dir = tmp_path_factory.mktemp("mkdocs-site")
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "-f",
            str(MKDOCS_CONFIG),
            "-d",
            str(site_dir),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    return site_dir


def _hrefs(path: Path) -> list[str]:
    assert path.exists(), f"Generated page is missing: {path}"
    parser = _HrefParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser.hrefs


def _has_href(hrefs: list[str], fragment: str) -> bool:
    return any(fragment in href for href in hrefs)


def _normalized_learning_path_target(site_dir: Path, page: Path, href: str) -> str | None:
    """Normalize a local Learning Path href against its generated page."""
    if href.startswith(("http://", "https://")) or "learning-path/" not in href:
        return None
    href_without_fragment = href.split("#", 1)[0]
    relative_page = page.relative_to(site_dir)
    return posixpath.normpath(posixpath.join(relative_page.parent.as_posix(), href_without_fragment))


def test_tutorial_build_contains_executed_code_and_inline_svg(built_site: Path) -> None:
    """The tutorial's executable source and generated figures are the docs contract."""
    site_dir = built_site

    tutorial = site_dir / "tutorial" / "index.html"
    assert tutorial.exists()
    html = tutorial.read_text(encoding="utf-8")
    code = _CodeTextParser()
    code.feed(html)
    code_text = "".join(code.parts)

    assert "wd.from_numpy" in code_text
    assert "low_pass_filter" in code_text
    assert "concat_frame" in code_text
    assert "comparison.plot" in code_text
    assert "comparison.fft().plot" in code_text
    tutorial_figures = re.findall(
        r'<figure class="tutorial-figure">.*?</figure>',
        html,
        flags=re.DOTALL,
    )
    assert len(tutorial_figures) == 2
    assert all("<svg" in figure for figure in tutorial_figures)
    assert "<figcaption>Time waveform: original and filtered</figcaption>" in html
    assert "<figcaption>FFT spectrum: original and filtered</figcaption>" in html
    assert "Original" in html
    assert "After 1 kHz low-pass" in html


def test_learning_path_navigation_contract(built_site: Path) -> None:
    """The beginner path and the legacy RecipePlan URL remain discoverable."""
    config = (REPO_ROOT / "docs/mkdocs.yml").read_text(encoding="utf-8")
    assert "tutorial/pipeline-recipes.md" not in config
    assert "Five-minute Tutorial: tutorial/index.md" in config

    home_source = (REPO_ROOT / "docs/src/index.md").read_text(encoding="utf-8")
    assert home_source.index("Try Wandas first") < home_source.index("Continue learning in English")

    home = built_site / "index.html"
    home_hrefs = _hrefs(home)
    assert _has_href(home_hrefs, "tutorial/")
    assert _has_href(home_hrefs, "learning-path/00_why_wandas.html")
    assert _has_href(home_hrefs, "en/learning-path/00_why_wandas.html")

    tutorial = built_site / "tutorial" / "index.html"
    tutorial_hrefs = _hrefs(tutorial)
    assert _has_href(tutorial_hrefs, "learning-path/00_why_wandas.html")
    assert _has_href(tutorial_hrefs, "en/learning-path/00_why_wandas.html")
    assert not any("tutorial/pipeline-recipes" in href for href in tutorial_hrefs)

    manifest_targets = {
        output_path(Path(), lesson, locale).as_posix() for lesson in load_manifest() for locale in lesson.locales
    }
    for page, hrefs in ((home, home_hrefs), (tutorial, tutorial_hrefs)):
        targets = {
            target for href in hrefs if (target := _normalized_learning_path_target(built_site, page, href)) is not None
        }
        assert targets <= manifest_targets

    legacy_source = REPO_ROOT / "docs/src/tutorial/pipeline-recipes.md"
    legacy_text = legacy_source.read_text(encoding="utf-8")
    assert "exec=" not in legacy_text
    assert "\nassert " not in legacy_text

    legacy = built_site / "tutorial" / "pipeline-recipes" / "index.html"
    assert legacy.exists()
    legacy_html = legacy.read_text(encoding="utf-8")
    legacy_hrefs = _hrefs(legacy)
    assert "Reusable Pipeline Recipes Learning Path" in legacy_html
    assert "RecipePlan How-to" in legacy_html
    assert "Pipeline API Reference" in legacy_html
    assert _has_href(legacy_hrefs, "06_reusable_pipeline_recipes.html")
    assert _has_href(legacy_hrefs, "how-to/pipeline-recipes")
    assert _has_href(legacy_hrefs, "api/pipeline")

    how_to_hrefs = _hrefs(built_site / "how-to" / "pipeline-recipes" / "index.html")
    assert _has_href(how_to_hrefs, "06_reusable_pipeline_recipes.html")
    assert not any("tutorial/pipeline-recipes" in href for href in how_to_hrefs)

    api_hrefs = _hrefs(built_site / "api" / "pipeline" / "index.html")
    assert _has_href(api_hrefs, "06_reusable_pipeline_recipes.html")
    assert _has_href(api_hrefs, "how-to/pipeline-recipes")
    assert not any("tutorial/pipeline-recipes" in href for href in api_hrefs)
