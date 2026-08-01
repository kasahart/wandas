import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATHS = (REPO_ROOT / "README.md", REPO_ROOT / "README.ja.md")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REPOSITORY_IMAGE_PREFIXES = (
    "https://raw.githubusercontent.com/kasahart/wandas/main/",
    "https://github.com/kasahart/wandas/blob/main/",
    "https://github.com/kasahart/wandas/raw/main/",
    "https://github.com/kasahart/wandas/raw/refs/heads/main/",
)
OPTIONAL_CODE_PATTERNS = (
    re.compile(
        r"^\s*(?:from|import)\s+(?:h5netcdf|h5py|IPython|librosa|marimo|mosqito|sklearn|tensorflow|torch|"
        r"wandas\.pipeline\.sklearn)\b",
        re.M,
    ),
    re.compile(r"\bWandasOperationTransformer\s*\("),
    re.compile(r"\.\s*(?:hpss_harmonic|hpss_percussive|noct_spectrum)\s*\("),
    re.compile(r"\.\s*(?:loudness|roughness|sharpness)_[a-z0-9_]+\s*\("),
    re.compile(r"\bto_tensor\s*\(\s*framework\s*=\s*[\"'](?:tensorflow|torch)[\"']"),
    re.compile(r"\.\s*(?:load|save)\s*\("),
)


def _python_blocks(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)\n```", markdown, flags=re.S)


def _github_main_paths(markdown: str) -> Iterator[str]:
    pattern = r"https://github\.com/kasahart/wandas/(?:blob|tree)/main/([^)#?]+)"
    yield from re.findall(pattern, markdown)


def _repository_links(markdown: str) -> Iterator[str]:
    """Yield repository-relative links without becoming a general Markdown parser."""
    yield from _github_main_paths(markdown)
    for target in re.findall(r"(?<!!)\[[^]]*\]\(([^)]+)\)", markdown):
        target = target.split(maxsplit=1)[0].split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]
        if target and not re.match(r"(?:[a-z][a-z0-9+.-]*:|/)", target):
            yield target


def _readme_image_paths(markdown: str) -> Iterator[Path]:
    for target in re.findall(r"!\[[^]]*\]\(([^)]+)\)", markdown):
        target = target.split(maxsplit=1)[0].split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]
        for prefix in REPOSITORY_IMAGE_PREFIXES:
            if target.startswith(prefix):
                target = target.removeprefix(prefix)
                break
        else:
            if re.match(r"(?:[a-z][a-z0-9+.-]*:|/)", target):
                continue

        candidate = (REPO_ROOT / target).resolve()
        try:
            candidate.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise AssertionError(f"README image target escapes the repository: {target}") from exc
        yield candidate


def _is_minimally_valid_image(path: Path) -> bool:
    with path.open("rb") as stream:
        data = stream.read(12)
    if path.suffix.lower() == ".png":
        return data.startswith(PNG_SIGNATURE)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return data.startswith(b"\xff\xd8\xff")
    if path.suffix.lower() == ".gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    if path.suffix.lower() == ".webp":
        return data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    return bool(data)


@pytest.fixture()
def readme_example_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sr = 48_000
    t = np.arange(sr) / sr
    samples = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    sf.write(tmp_path / "recording.wav", samples, sr)
    sample_dir = tmp_path / "learning-path"
    sample_dir.mkdir()
    shutil.copyfile(REPO_ROOT / "learning-path" / "sample_audio.wav", sample_dir / "sample_audio.wav")

    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.filterwarnings("ignore:More than 20 figures have been opened:RuntimeWarning")
def test_readme_python_code_blocks_execute(readme_example_workspace: Path) -> None:
    """README Python examples should remain runnable public workflows."""
    del readme_example_workspace

    for path in README_PATHS:
        namespace: dict[str, object] = {"__name__": f"readme_example_{path.stem}"}
        for index, block in enumerate(_python_blocks(path.read_text(encoding="utf-8")), start=1):
            try:
                exec(compile(block, f"{path.name}:python-block-{index}", "exec"), namespace)
                plt.close("all")
            except Exception as exc:
                raise AssertionError(f"{path.name} Python block {index} failed: {exc}") from exc


def test_readme_english_and_japanese_python_examples_stay_aligned() -> None:
    """The translated README should not silently teach a different Python workflow."""
    examples = [
        [_block.strip() for _block in _python_blocks(path.read_text(encoding="utf-8"))] for path in README_PATHS
    ]
    english, japanese = examples
    assert english == japanese


def test_readme_repository_links_target_existing_paths() -> None:
    """README links into the repository should not point at missing files."""
    missing: list[str] = []
    for path in README_PATHS:
        for target in _repository_links(path.read_text(encoding="utf-8")):
            candidate = (REPO_ROOT / target).resolve()
            try:
                candidate.relative_to(REPO_ROOT.resolve())
            except ValueError:
                missing.append(f"{path.name}: {target}")
            else:
                if not candidate.exists():
                    missing.append(f"{path.name}: {target}")

    assert missing == []


def test_readme_referenced_images_exist_and_have_valid_headers() -> None:
    """README image references should resolve to minimally valid checked-in assets."""
    references = {path for readme in README_PATHS for path in _readme_image_paths(readme.read_text(encoding="utf-8"))}

    assert references
    assert all(path.is_file() and _is_minimally_valid_image(path) for path in references)


def test_readme_python_examples_do_not_require_optional_extras() -> None:
    """Core README examples must not accidentally become optional-dependency examples."""
    offenders = [
        f"{path.name}: {pattern.pattern}"
        for path in README_PATHS
        for block in _python_blocks(path.read_text(encoding="utf-8"))
        for pattern in OPTIONAL_CODE_PATTERNS
        if pattern.search(block)
    ]

    assert offenders == []
