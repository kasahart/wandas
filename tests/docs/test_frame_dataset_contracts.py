import inspect
from pathlib import Path

import pytest

from wandas.utils.frame_dataset import ChannelFrameDataset, FrameDataset


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


def test_sample_formula_is_explicit_in_docstring() -> None:
    formula = "max(1, min(10, int(len(self) * 0.1)))"
    docstring = inspect.getdoc(FrameDataset.sample)

    assert docstring is not None
    assert formula in docstring


def test_unsupported_save_contract_is_in_docstring() -> None:
    docstring = inspect.getdoc(FrameDataset.save)

    assert docstring is not None
    assert docstring.startswith("Unsupported:")
    assert "Always raised" in docstring


def test_get_by_label_deprecation_names_replacement_and_support_window(
    tmp_path: Path,
) -> None:
    docstring = inspect.getdoc(FrameDataset.get_by_label)
    dataset = ChannelFrameDataset(
        str(tmp_path),
        file_extensions=[".wav"],
        lazy_loading=True,
    )

    assert docstring is not None
    assert "Deprecated since 0.2.0" in docstring
    assert "get_all_by_label" in docstring
    assert "no earlier than version 0.7.0" in " ".join(docstring.split())
    with pytest.warns(
        DeprecationWarning,
        match=r"deprecated since 0\.2\.0.*removal no earlier than 0\.7\.0",
    ):
        assert dataset.get_by_label("missing.wav") is None
