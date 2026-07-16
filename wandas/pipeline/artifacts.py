"""Filesystem boundary for portable Recipe JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandas.pipeline.errors import RecipeSerializationError

if TYPE_CHECKING:
    from wandas.pipeline.model import RecipePlan
    from wandas.pipeline.registry import RecipeRegistry


RECIPE_ARTIFACT_SUFFIX = ".recipe.json"


def recipe_artifact_path(path: str | Path) -> Path:
    """Return the canonical path for a standalone Recipe artifact."""
    normalized = Path(path)
    if not normalized.name.endswith(RECIPE_ARTIFACT_SUFFIX):
        normalized = normalized.with_name(f"{normalized.name}{RECIPE_ARTIFACT_SUFFIX}")
    return normalized


def save_recipe_artifact(
    plan: RecipePlan,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one validated Recipe plan as deterministic strict JSON."""
    target = recipe_artifact_path(path)
    encoded = json.dumps(
        plan.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        separators=(",", ": "),
    )
    mode = "w" if overwrite else "x"
    with target.open(mode, encoding="utf-8", newline="\n") as artifact:
        artifact.write(encoded)
        artifact.write("\n")
    return target


def _reject_nonfinite_json_number(value: str) -> Any:
    """Reject JavaScript-style non-finite constants at the JSON boundary."""
    raise RecipeSerializationError(
        "Recipe artifact must use strict JSON\n"
        f"  Got: {value}\n"
        "Replace non-finite values or recreate the Recipe with a current Wandas version."
    )


def load_recipe_artifact(
    path: str | Path,
    *,
    registry: RecipeRegistry | None = None,
) -> RecipePlan:
    """Read and validate one standalone Recipe JSON artifact."""
    from wandas.pipeline.model import RecipePlan

    target = recipe_artifact_path(path)
    try:
        decoded = json.loads(
            target.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json_number,
        )
    except RecipeSerializationError:
        raise
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecipeSerializationError(
            "Invalid Recipe JSON artifact\n"
            f"  Path: {target}\n"
            f"  Cause: {exc}\n"
            "Use a UTF-8 .recipe.json file produced by RecipePlan.save()."
        ) from exc
    return RecipePlan.from_dict(decoded, registry=registry)


__all__ = [
    "RECIPE_ARTIFACT_SUFFIX",
    "load_recipe_artifact",
    "recipe_artifact_path",
    "save_recipe_artifact",
]
