import os
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

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


def test_tutorial_build_contains_executed_code_and_inline_svg(tmp_path) -> None:
    """The tutorial's executable source and generated figures are the docs contract."""
    site_dir = tmp_path / "site"
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
