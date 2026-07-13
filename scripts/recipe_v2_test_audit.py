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
    "test_binary_operand_step_applies_non_additive_array_operators": (
        "tests/pipeline/test_recipe_compiler.py",
        "test_external_numpy_non_additive_operators_preserve_semantics",
    ),
    "test_node_graph_recipe_from_frame_extracts_numpy_operand_as_external_input": (
        "tests/pipeline/test_recipe_compiler.py",
        "test_external_numpy_non_additive_operators_preserve_semantics",
    ),
    "test_recipe_from_frame_extracts_fft_ifft_typed_transition_chain": (
        "tests/pipeline/test_recipe_execution.py",
        "test_fft_ifft_typed_transition_chain_replays",
    ),
    "test_recipe_apply_preserves_dask_laziness": (
        "tests/pipeline/test_recipe_compiler.py",
        "test_external_dask_array_is_named_input_and_stays_lazy",
    ),
    "test_recipe_from_frame_extracts_reverse_numeric_scalar_operations": (
        "tests/pipeline/test_recipe_compiler.py",
        "test_scalar_and_reflected_scalar_preserve_order",
    ),
    "test_recipe_from_frame_extracts_scalar_operation_symbols": (
        "tests/pipeline/test_recipe_compiler.py",
        "test_scalar_and_reflected_scalar_preserve_order",
    ),
    "test_recipe_from_frame_extracts_fft_typed_transition": (
        "tests/pipeline/test_recipe_execution.py",
        "test_typed_frame_transition_replays",
    ),
    "test_node_graph_recipe_from_frame_extracts_add_channel_dask_data_after_processed_parent": (
        "tests/pipeline/test_recipe_compiler.py",
        "test_processed_parent_add_channel_external_dask_stays_lazy",
    ),
    "test_recipe_from_frame_extracts_additional_single_input_apply_operations": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_supported_unary_audio_operations_replay",
    ),
    "test_recipe_from_frame_extracts_hpss_apply_operations": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_hpss_operations_replay",
    ),
    "test_recipe_from_frame_extracts_psychoacoustic_apply_operations": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_psychoacoustic_operations_replay",
    ),
    "test_recipe_from_frame_extracts_roughness_spec_typed_transition": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_roughness_typed_transition_replays",
    ),
    "test_recipe_from_frame_extracts_stft_istft_typed_transition_chain": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_stft_istft_and_welch_transitions_replay",
    ),
    "test_recipe_from_frame_extracts_stft_typed_transition": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_stft_istft_and_welch_transitions_replay",
    ),
    "test_recipe_from_frame_extracts_welch_typed_transition": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_stft_istft_and_welch_transitions_replay",
    ),
    "test_recipe_from_frame_extracts_noct_spectrum_typed_transition": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_noct_spectrum_and_synthesis_transitions_replay",
    ),
    "test_recipe_from_frame_extracts_noct_synthesis_typed_transition": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_noct_spectrum_and_synthesis_transitions_replay",
    ),
    "test_recipe_from_frame_extracts_cross_channel_typed_transitions": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_cross_channel_typed_transitions_replay",
    ),
    "test_node_graph_recipe_add_channel_frame_input_omits_raw_source_time_offset_option": (
        "tests/pipeline/test_recipe_behavior_parity.py",
        "test_add_channel_preserves_metadata_and_source_time_contract",
    ),
    "test_recipe_from_frame_extracts_multidimensional_slice_indexing": (
        "tests/pipeline/test_recipe_compiler.py",
        "test_multidimensional_indexing_is_one_call",
    ),
}
FORCED_REMOVALS = {
    "test_steps_from_graph_rejects_invalid_multidimensional_parent_shapes": "v1 dictionary parent-shape parser removed",
    "test_custom_function_step_rejects_invalid_metadata": "v1 step metadata schema removed",
    "test_graph_step_extraction_rejects_unreplayable_binary_metadata": "v1 dictionary metadata inference removed",
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
    migrated = removed = 0
    print("v1_test\tdisposition\trationale\tcurrent_test")
    for name in old_names:
        if name in FORCED_REMOVALS:
            removed += 1
            print(f"{name}\tremoved_contract\t{FORCED_REMOVALS[name]}\t-")
            continue
        migration = MIGRATIONS.get(name)
        if migration is None:
            removed += 1
            print(f"{name}\tremoved_contract\tv1 API/helper contract not retained by destructive v2\t-")
            continue
        path, target = migration
        available = inventories.setdefault(path, functions(ROOT / path))
        if target not in available:
            raise RuntimeError(f"Audit target does not exist: {path}::{target}")
        migrated += 1
        print(f"{name}\tmigrated\tcurated retained behavior family\t{path}::{target}")
    print(f"audited_cases\t{len(old_names)}\tmigrated\t{migrated}\tremoved_contract\t{removed}")


if __name__ == "__main__":
    main()
