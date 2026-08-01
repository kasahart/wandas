from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEARNING_PATH = REPO_ROOT / "learning-path"
PRIVATE_API_TOKENS = (
    ".compute(",
    ".persist(",
    ".xr",
    "to_xarray(",
    "._data",
    "._xr",
    "._channel_metadata",
    "wandas.frames.spectrogram",
    "wandas.utils.frame_dataset",
    "wandas.visualization.types",
)


def test_learning_path_uses_frame_data_as_the_value_boundary() -> None:
    """Learner examples should not turn Frame backend details into user APIs."""
    findings = {
        path.relative_to(REPO_ROOT).as_posix(): [token for token in PRIVATE_API_TOKENS if token in text]
        for path in sorted(LEARNING_PATH.glob("*.py"))
        if path.name != "06_skill_validation.py"
        if (text := path.read_text(encoding="utf-8"))
        if any(token in text for token in PRIVATE_API_TOKENS)
    }

    assert findings == {}
