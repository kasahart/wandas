import inspect
from pathlib import Path

import pytest

from wandas.utils.frame_dataset import ChannelFrameDataset, FrameDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_DOC = REPO_ROOT / "docs" / "src" / "api" / "utils.md"


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (0, 0),
        (1, 1),
        (9, 1),
        (10, 1),
        (99, 9),
        (100, 10),
        (101, 10),
        (250, 10),
    ],
)
def test_documented_default_sample_formula_matches_runtime(
    tmp_path: Path,
    total: int,
    expected: int,
) -> None:
    for index in range(total):
        (tmp_path / f"{index:03}.wav").write_bytes(b"")
    dataset = ChannelFrameDataset(
        str(tmp_path),
        file_extensions=[".wav"],
        lazy_loading=True,
    )

    sampled = dataset.sample(seed=7)

    assert len(sampled) == expected
    assert sampled.get_metadata()["loaded_count"] == 0


def test_sample_formula_is_explicit_in_docstring_and_api_guide() -> None:
    formula = "max(1, min(10, int(len(self) * 0.1)))"
    docstring = inspect.getdoc(FrameDataset.sample)
    api_guide = UTILS_DOC.read_text(encoding="utf-8")

    assert docstring is not None
    assert formula in docstring
    assert "max(1, min(10, int(len(dataset) * 0.1)))" in api_guide
    assert "For an empty dataset, `sample()` returns an empty dataset." in api_guide


def test_unsupported_save_contract_is_visible() -> None:
    docstring = inspect.getdoc(FrameDataset.save)
    api_guide = UTILS_DOC.read_text(encoding="utf-8")

    assert docstring is not None
    assert docstring.startswith("Unsupported:")
    assert "Always raised" in docstring
    assert "`FrameDataset.save()` is not a supported persistence API" in api_guide


def test_get_by_label_deprecation_names_replacement_and_support_window(
    tmp_path: Path,
) -> None:
    docstring = inspect.getdoc(FrameDataset.get_by_label)
    api_guide = UTILS_DOC.read_text(encoding="utf-8")
    dataset = ChannelFrameDataset(
        str(tmp_path),
        file_extensions=[".wav"],
        lazy_loading=True,
    )

    assert docstring is not None
    assert "Deprecated since 0.2.0" in docstring
    assert "get_all_by_label" in docstring
    assert "no earlier than version 0.7.0" in " ".join(docstring.split())
    assert "planned for removal no earlier than 0.7.0" in " ".join(api_guide.split())
    with pytest.warns(
        DeprecationWarning,
        match=r"deprecated since 0\.2\.0.*removal no earlier than 0\.7\.0",
    ):
        assert dataset.get_by_label("missing.wav") is None


def test_api_examples_guard_optional_integer_access() -> None:
    api_guide = UTILS_DOC.read_text(encoding="utf-8")

    assert "first_file = dataset[0]\nif first_file is None:" in api_guide
    assert "spec_frame = spec_dataset[0]\nif spec_frame is not None:" in api_guide
    assert "processing history is not exposed" in api_guide
