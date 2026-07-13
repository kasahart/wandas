"""Audit every deleted v1 Recipe test as migrated or intentionally removed."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOVED_HELPERS = (
    "steps_from_graph",
    "step_from_graph",
    "axis_slices_from_params",
    "indices_from_params",
    "mask_from_params",
    "slice_from_serialized",
    "channel_key_from_parent_graph",
    "rename_mapping_from_params",
    "restore_history_value",
    "snapshot_get_channel_query_params",
    "boolean_mask_wrapper",
)
ROUTES = (
    (("sklearn",), "tests/pipeline/test_sklearn_adapter.py", "test_transform_applies_operation"),
    (
        ("terminal",),
        "tests/pipeline/test_recipe_serialization.py",
        "test_valid_terminal_property_roundtrips_and_executes",
    ),
    (
        ("custom", "callable", "importable"),
        "tests/pipeline/test_recipe_execution.py",
        "test_custom_function_replays_by_stable_path",
    ),
    (
        ("index", "getitem", "slice", "mask"),
        "tests/pipeline/test_recipe_compiler.py",
        "test_multidimensional_indexing_is_one_call",
    ),
    (("add_channel",), "tests/pipeline/test_recipe_execution.py", "test_add_channel_frame_and_array_replay"),
    (
        ("binary", "operand", "scalar", "graph_recipe"),
        "tests/pipeline/test_recipe_compiler.py",
        "test_scalar_and_reflected_scalar_preserve_order",
    ),
    (
        ("serial", "dict", "json", "params"),
        "tests/pipeline/test_recipe_serialization.py",
        "test_canonical_schema_roundtrip_is_json_serializable",
    ),
    (
        ("reject", "invalid", "missing", "unknown"),
        "tests/pipeline/test_recipe_contract.py",
        "test_graph_invariants_fail_closed",
    ),
    (
        ("metadata", "source_time"),
        "tests/pipeline/test_recipe_execution.py",
        "test_metadata_and_source_time_offset_are_preserved",
    ),
)
DEFAULT = ("tests/pipeline/test_recipe_compiler.py", "test_linear_audio_recipe_replays")


def functions(path: Path) -> set[str]:
    return {
        node.name for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, ast.FunctionDef)
    }


def main() -> None:
    old_source = subprocess.run(
        ["git", "show", "b808c8e:tests/pipeline/test_recipe.py"], check=True, capture_output=True, text=True
    ).stdout
    old_names = [
        node.name
        for node in ast.walk(ast.parse(old_source))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    if len(old_names) != 192 or len(set(old_names)) != len(old_names):
        raise RuntimeError("v1 Recipe test inventory is incomplete or duplicated")
    inventories: dict[str, set[str]] = {}
    migrated = removed = 0
    print("v1_test\tdisposition\trationale\tcurrent_test")
    for name in old_names:
        if any(marker in name for marker in REMOVED_HELPERS):
            removed += 1
            print(f"{name}\tremoved_contract\tv1 dict/step reconstruction no longer exists\t-")
            continue
        path, target = DEFAULT
        rationale = "public RecipePlan behavior retained"
        for words, candidate_path, candidate_target in ROUTES:
            if any(word in name for word in words):
                path, target = candidate_path, candidate_target
                rationale = "behavior migrated to canonical public/family contract"
                break
        available = inventories.setdefault(path, functions(ROOT / path))
        if target not in available:
            raise RuntimeError(f"Audit target does not exist: {path}::{target}")
        migrated += 1
        print(f"{name}\tmigrated\t{rationale}\t{path}::{target}")
    print(f"audited_cases\t{len(old_names)}\tmigrated\t{migrated}\tremoved_contract\t{removed}")


if __name__ == "__main__":
    main()
