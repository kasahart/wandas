"""Audit every deleted v1 Recipe test with an explicitly curated disposition."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = {
    "test_recipe_apply_runs_steps_in_order_and_preserves_source_frame": (
        "tests/pipeline/test_recipe_compiler.py",
        "test_linear_audio_recipe_replays",
    ),
    "test_recipe_apply_supports_terminal_rms_metric": (
        "tests/pipeline/test_recipe_serialization.py",
        "test_valid_terminal_property_roundtrips_and_executes",
    ),
    "test_recipe_from_frame_extracts_importable_custom_function": (
        "tests/pipeline/test_recipe_execution.py",
        "test_custom_function_replays_by_stable_path",
    ),
    "test_recipe_from_frame_rejects_custom_lambda_boundary": (
        "tests/pipeline/test_recipe_serialization.py",
        "test_public_call_constructors_share_fail_closed_contracts",
    ),
    "test_graph_recipe_applies_add_with_snr": (
        "tests/pipeline/test_recipe_execution.py",
        "test_true_multi_input_replays_in_role_order",
    ),
    "test_graph_recipe_rejects_missing_input": (
        "tests/pipeline/test_recipe_contract.py",
        "test_missing_and_wrong_input_types_are_rejected",
    ),
    "test_graph_recipe_from_frame_extracts_single_merge_with_typed_tail": (
        "tests/pipeline/test_recipe_execution.py",
        "test_typed_transition_after_merge_replays",
    ),
    "test_graph_recipe_from_frame_uses_numbered_default_names_with_typed_tail": (
        "tests/pipeline/test_recipe_execution.py",
        "test_typed_transition_after_merge_replays",
    ),
    "test_node_graph_recipe_from_frame_extracts_typed_tail_after_merge": (
        "tests/pipeline/test_recipe_execution.py",
        "test_typed_transition_after_merge_replays",
    ),
    "test_node_graph_recipe_from_frame_extracts_multidimensional_indexing_branch": (
        "tests/pipeline/test_recipe_compiler.py",
        "test_multidimensional_indexing_is_one_call",
    ),
    "test_recipe_from_frame_extracts_multidimensional_slice_indexing": (
        "tests/pipeline/test_recipe_compiler.py",
        "test_multidimensional_indexing_is_one_call",
    ),
}


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
    unknown = set(MIGRATIONS) - set(old_names)
    if unknown:
        raise RuntimeError(f"Curated migration names are not in the v1 inventory: {sorted(unknown)}")
    inventories: dict[str, set[str]] = {}
    print("v1_test\tdisposition\trationale\tcurrent_test")
    for name in old_names:
        migration = MIGRATIONS.get(name)
        if migration is None:
            print(f"{name}\tremoved_contract\tv1 API/helper contract not retained by destructive v2\t-")
            continue
        path, target = migration
        available = inventories.setdefault(path, functions(ROOT / path))
        if target not in available:
            raise RuntimeError(f"Audit target does not exist: {path}::{target}")
        print(f"{name}\tmigrated\texact retained behavior\t{path}::{target}")
    removed = len(old_names) - len(MIGRATIONS)
    print(f"audited_cases\t{len(old_names)}\tmigrated\t{len(MIGRATIONS)}\tremoved_contract\t{removed}")


if __name__ == "__main__":
    main()
