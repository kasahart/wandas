"""Run the Python blocks of the how-to guides on synthetic recordings."""

import importlib.util
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import soundfile as sf
from matplotlib import pyplot as plt

from wandas.core.default_dtype import get_default_dtype, set_default_dtype

REPO_ROOT = Path(__file__).resolve().parents[2]
HOW_TO_DIR = REPO_ROOT / "docs" / "src" / "how-to"
RULE = (
    "Rule: how-to Python blocks run as written on synthetic recordings "
    "(see docs/src/contributing.md, Documentation Structure)."
)

# Pages that this test does not run, with the reason.
EXCLUDED_PAGES = {
    "custom-processing.md": "tests/docs/test_custom_processing_guide.py runs it in a fresh process.",
    "pyodide-browser.md": "It needs Pyodide (micropip, top-level await, pyodide.http).",
}

# Blocks that cannot run here, keyed by (page, text found in exactly one block).
# This list may only shrink: fix the page instead of adding an entry when you can.
SKIPPED_BLOCKS = {
    ("pipeline-recipes.md", "plan.apply_files("): (
        "It starts a 4-process pool; tests/pipeline/test_recipe_apply_files.py covers apply_files."
    ),
    ("recording-catalogs.md", 'base_dir="/data"'): "It reads recordings from an absolute /data folder.",
    ("recording-catalogs.md", 'channel="force"'): "It needs recordings with a channel labeled force.",
}

# Blocks that need an optional extra, keyed like SKIPPED_BLOCKS, mapped to the
# module the extra installs. They run when that module is installed.
EXTRA_BLOCKS = {
    ("pipeline-recipes.md", "noct_spectrum("): "mosqito",
}

# Names a page expects the reader to already have, defined before its blocks run.
PAGE_SETUP = {
    "cepstral-analysis.md": 'another_audio = wd.read("voice.wav")',
}

SHORT_WAVS = (
    "audio.wav",
    "motor.wav",
    "long-recording.wav",
    "reference.wav",
    "voice.wav",
    "root_a/001.wav",
    "root_b/001.wav",
)


def _python_blocks(page: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)\n```", page.read_text(encoding="utf-8"), flags=re.S)


def _pages() -> list[str]:
    return sorted(path.name for path in HOW_TO_DIR.glob("*.md") if path.name not in EXCLUDED_PAGES)


def _write_tone(path: Path, seconds: float, sr: int = 8_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)


@pytest.fixture()
def how_to_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in SHORT_WAVS:
        _write_tone(tmp_path / name, 1.0)
    # inspect-audio-and-frame-times.md trims 1 min 30 s to 2 min.
    _write_tone(tmp_path / "recording.wav", 125.0)
    (tmp_path / "catalog.csv").write_text("audio_file\nroot_a/001.wav\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("page_name", _pages())
@pytest.mark.filterwarnings("ignore:More than 20 figures have been opened:RuntimeWarning")
def test_how_to_python_blocks_run(page_name: str, how_to_workspace: Path) -> None:
    del how_to_workspace
    page = HOW_TO_DIR / page_name
    blocks = _python_blocks(page)
    namespace: dict[str, object] = {"__name__": f"how_to_{page.stem.replace('-', '_')}"}
    setup = PAGE_SETUP.get(page_name)
    if setup is not None:
        exec(compile("import wandas as wd\n" + setup, f"{page_name}:setup", "exec"), namespace)
    missing_extras: list[str] = []
    dtype = get_default_dtype()
    try:
        for index, block in enumerate(blocks, start=1):
            if any(page_name == name and text in block for name, text in SKIPPED_BLOCKS):
                continue
            extras = EXTRA_BLOCKS.items()
            module = next((mod for (name, text), mod in extras if name == page_name and text in block), None)
            if module is not None and importlib.util.find_spec(module) is None:
                missing_extras.append(f"block {index} needs {module}")
                continue
            try:
                exec(compile(block, f"{page_name}:python-block-{index}", "exec"), namespace)
            except Exception as exc:
                raise AssertionError(
                    f"docs/src/how-to/{page_name} Python block {index} failed: {type(exc).__name__}: {exc}\n"
                    f"{RULE}\n"
                    f"Fix: update that block to the current API. If it cannot run on synthetic data, "
                    f"add it to SKIPPED_BLOCKS in tests/docs/test_how_to_examples.py with a reason."
                ) from exc
            finally:
                plt.close("all")
    finally:
        set_default_dtype(dtype)
    if missing_extras:
        pytest.skip(f"{page_name}: optional extra not installed ({', '.join(missing_extras)})")


def test_how_to_block_lists_match_the_pages() -> None:
    stale = []
    for page_name, text in [*SKIPPED_BLOCKS, *EXTRA_BLOCKS]:
        matches = sum(text in block for block in _python_blocks(HOW_TO_DIR / page_name))
        if matches != 1:
            stale.append(f"{page_name}: {text!r} is in {matches} blocks")
    stale += [f"{name}: no such page" for name in [*EXCLUDED_PAGES, *PAGE_SETUP] if not (HOW_TO_DIR / name).is_file()]
    assert not stale, (
        "Entries in tests/docs/test_how_to_examples.py no longer match the how-to pages:\n"
        + "\n".join(stale)
        + f"\n{RULE}\n"
        "Fix: remove the entry if the block was fixed or deleted, or change its text so it is found in exactly "
        "one block of that page."
    )
