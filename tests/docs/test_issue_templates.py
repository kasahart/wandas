from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).parents[2]
TEMPLATE_DIRECTORY = REPOSITORY_ROOT / ".github" / "ISSUE_TEMPLATE"
TEMPLATE_PATHS = (
    TEMPLATE_DIRECTORY / "bug_report.yml",
    TEMPLATE_DIRECTORY / "feature_request.yml",
)


def _load_template(path: Path) -> dict[str, Any]:
    template = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(template, dict)
    assert isinstance(template.get("body"), list)
    return template


def _body_item(template: dict[str, Any], item_id: str) -> dict[str, Any]:
    for item in template["body"]:
        if isinstance(item, dict) and item.get("id") == item_id:
            return item
    raise AssertionError(f"Missing issue template body item: {item_id}")


def test_issue_template_examples_use_recommended_read_api() -> None:
    bug_template = _load_template(TEMPLATE_PATHS[0])
    feature_template = _load_template(TEMPLATE_PATHS[1])

    reproduction = _body_item(bug_template, "reproduction")
    solution = _body_item(feature_template, "solution")

    assert 'wd.read("audio.wav")' in reproduction["attributes"]["placeholder"]
    assert 'wd.read("audio.wav")' in solution["attributes"]["placeholder"]


def test_issue_templates_reject_compatibility_helpers_and_pinned_versions() -> None:
    template_text = "\n".join(path.read_text(encoding="utf-8") for path in TEMPLATE_PATHS)

    for compatibility_helper in ("wd.read_wav(", "wd.read_csv(", "wd.from_ndarray("):
        offenders = [path.name for path in TEMPLATE_PATHS if compatibility_helper in path.read_text(encoding="utf-8")]
        assert compatibility_helper not in template_text, (
            f"Issue templates {offenders} use the compatibility helper {compatibility_helper!r}.\n"
            "Rule: issue template examples use the primary API in wandas.__all__, such as wd.read() "
            "(see docs/src/api/index.md).\n"
            "Fix: replace the helper with wd.read(), wd.from_numpy(), or another primary API in those templates."
        )

    bug_template = _load_template(TEMPLATE_PATHS[0])
    environment = _body_item(bug_template, "environment")["attributes"]["value"]
    wandas_version_line = next(line for line in environment.splitlines() if "Wandas version:" in line)
    assert "installed version" in wandas_version_line
    assert not any(character.isdigit() for character in wandas_version_line)


def test_feature_template_areas_match_current_module_boundaries() -> None:
    feature_template = _load_template(TEMPLATE_PATHS[1])
    options = _body_item(feature_template, "area")["attributes"]["options"]

    for module in ("frames/", "processing/", "io/", "pipeline/", "visualization/", "datasets/", "utils/"):
        assert any(module in option for option in options), (
            f".github/ISSUE_TEMPLATE/feature_request.yml has no 'area' option for {module}.\n"
            "Rule: the feature request areas follow the current wandas/ package boundaries "
            "(see the module pages under docs/src/api/).\n"
            f"Fix: add an area option naming {module} to that template, or update this list if the package moved."
        )
    assert any("Recipe" in option for option in options)
    assert any("calibration" in option.lower() for option in options)
