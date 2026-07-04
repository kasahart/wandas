from __future__ import annotations

import numbers
from collections.abc import Mapping
from typing import Any, cast

from wandas.pipeline.errors import RecipeExtractionError
from wandas.pipeline.params import _BooleanMask, _restore_history_value
from wandas.pipeline.registry import (
    _REPLAYABLE_APPLY_OPERATIONS,
    _REPLAYABLE_GETITEM_INDEXING,
    _REPLAYABLE_METHOD_OPERATIONS,
    _REPLAYABLE_SCALAR_OPERATIONS,
    _REPLAYABLE_TYPED_METHOD_OPERATIONS,
)
from wandas.pipeline.steps import (
    AddChannelDataStep,
    AddChannelStep,
    BinaryFrameStep,
    BinaryOperandStep,
    CustomFunctionStep,
    IndexingStep,
    MethodStep,
    OperationSpec,
    RecipeStep,
    ScalarOperationStep,
    TypedMethodStep,
    _load_importable_frame_class,
)


def _validate_replayable_operation(operation: str) -> None:
    try:
        from wandas.processing import get_operation

        operation_class = get_operation(operation)
    except ValueError as exc:
        raise RecipeExtractionError(
            "Operation is outside the Stage 1 recipe allowlist\n"
            f"  Operation: {operation}\n"
            "  Current RecipeSpec can only replay selected single-input Wandas operations. "
            "Graph, method, domain-transition, and callable recipes require the next recipe model."
        ) from exc

    expected_input_count = getattr(operation_class, "_expected_input_count", 1)
    if isinstance(expected_input_count, int) and expected_input_count != 1:
        raise RecipeExtractionError(
            "Graph operation requires graph recipe support\n"
            f"  Operation: {operation}\n"
            f"  Runtime inputs: {expected_input_count}\n"
            "  Current RecipeSpec can only replay single-input linear operations."
        )
    if operation not in _REPLAYABLE_APPLY_OPERATIONS:
        raise RecipeExtractionError(
            "Operation is outside the Stage 1 recipe allowlist\n"
            f"  Operation: {operation}\n"
            "  Current RecipeSpec can only replay selected single-input Wandas operations. "
            "Graph, method, domain-transition, and callable recipes require the next recipe model."
        )


def _method_params(params: Mapping[str, Any], param_names: Mapping[str, str] | None) -> dict[str, Any]:
    if param_names is None:
        return dict(params)
    return {param_names[key]: value for key, value in params.items() if key in param_names}


def _method_step_from_graph(operation: str, params: Mapping[str, Any]) -> MethodStep:
    if operation == "get_channel" and "query_kind" in params:
        raise RecipeExtractionError(
            "Channel selection recipe extraction only supports explicit indices or exact label queries\n"
            f"  Query kind: {params['query_kind']!r}\n"
            "  Callable, regex, and dict channel queries need a selection recipe model that can preserve query intent."
        )
    if operation == "rename_channels":
        return MethodStep("rename_channels", {"mapping": _rename_mapping_from_params(params)})
    method, param_names = _REPLAYABLE_METHOD_OPERATIONS[operation]
    return MethodStep(method, _method_params(params, param_names))


def _custom_function_step_from_graph(
    params: Mapping[str, Any],
    custom_metadata: Mapping[str, Any] | None,
) -> CustomFunctionStep:
    if custom_metadata is None:
        raise RecipeExtractionError(
            "Custom operation recipe extraction requires importable module-level functions\n"
            "  Supported custom operations: module-level functions importable as 'module.function'.\n"
            "  Unsupported custom operations: lambdas, closures, nested functions, callable objects, "
            "bound methods, functools.partial, and __main__ functions."
        )
    function = custom_metadata.get("function")
    if not isinstance(function, str):
        raise RecipeExtractionError(
            f"Custom operation recipe extraction requires an importable function path\n  Metadata: {custom_metadata!r}"
        )
    output_shape_function = custom_metadata.get("output_shape_function")
    if output_shape_function is not None and not isinstance(output_shape_function, str):
        raise RecipeExtractionError(
            "Custom operation recipe extraction requires an importable output_shape_func path\n"
            f"  Metadata: {custom_metadata!r}"
        )
    dask_pure = custom_metadata.get("dask_pure", True)
    if not isinstance(dask_pure, bool):
        raise RecipeExtractionError(
            f"Custom operation recipe extraction requires a boolean dask_pure flag\n  Metadata: {custom_metadata!r}"
        )
    output_frame_class = custom_metadata.get("output_frame_class")
    if output_frame_class is not None and not isinstance(output_frame_class, str):
        raise RecipeExtractionError(
            "Custom operation recipe extraction requires an importable output frame class path\n"
            f"  Metadata: {custom_metadata!r}"
        )
    if output_frame_class is not None:
        try:
            _load_importable_frame_class(output_frame_class)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            raise RecipeExtractionError(
                "Custom operation recipe extraction requires an importable output frame class\n"
                f"  Output frame class: {output_frame_class!r}"
            ) from exc
    output_frame_kwargs = custom_metadata.get("output_frame_kwargs", {})
    if not isinstance(output_frame_kwargs, Mapping):
        raise RecipeExtractionError(
            f"Custom operation recipe extraction requires output_frame_kwargs mapping\n  Metadata: {custom_metadata!r}"
        )
    try:
        return CustomFunctionStep(
            function,
            params,
            output_shape_function=output_shape_function,
            dask_pure=dask_pure,
            output_frame_class=output_frame_class,
            output_frame_kwargs=output_frame_kwargs,
        )
    except TypeError as exc:
        raise RecipeExtractionError(
            "Custom operation recipe extraction requires recipe-literal params and output_frame_kwargs\n"
            f"  Metadata: {custom_metadata!r}"
        ) from exc


def _rename_mapping_from_params(params: Mapping[str, Any]) -> dict[int | str, str]:
    raw_items = params.get("mapping_items")
    if not isinstance(raw_items, list | tuple):
        raise RecipeExtractionError(
            f"rename_channels recipe extraction requires typed mapping items\n  Got: {type(raw_items).__name__}"
        )
    mapping: dict[int | str, str] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, list | tuple) or len(raw_item) != 2:
            raise RecipeExtractionError(
                f"rename_channels recipe extraction requires key/value mapping items\n  Got: {raw_item!r}"
            )
        key, value = raw_item
        if not isinstance(key, int | str) or not isinstance(value, str):
            raise RecipeExtractionError(
                "rename_channels recipe extraction only supports int/str keys and str labels\n"
                f"  Got key/value types: {type(key).__name__}/{type(value).__name__}"
            )
        mapping[key] = value
    return mapping


def _typed_method_step_from_graph(operation: str, params: Mapping[str, Any], kind: str | None) -> TypedMethodStep:
    if operation == "welch" and params.get("detrend", "constant") != "constant":
        raise RecipeExtractionError(
            "Welch recipe extraction only supports public welch parameters\n"
            f"  Operation detrend: {params.get('detrend')!r}\n"
            "  ChannelFrame.welch() does not expose detrend, so non-default values cannot be replayed safely."
        )
    if kind != "method":
        raise RecipeExtractionError(
            "Typed operation requires frame method lineage\n"
            f"  Operation: {operation}\n"
            "  Call the typed frame method, such as frame.fft(), instead of generic apply_operation(). "
            "Recipe replay delegates typed domain transitions to frame methods so output frame metadata stays correct."
        )
    method, param_names = _REPLAYABLE_TYPED_METHOD_OPERATIONS[operation]
    return TypedMethodStep(method, _method_params(params, param_names))


def _scalar_operand_from_params(operation: str, params: Mapping[str, Any]) -> int | float:
    if params.get("operand_kind") != "operand":
        raise RecipeExtractionError(
            "Graph operation requires graph recipe support\n"
            f"  Operation: {operation}\n"
            "  ScalarOperationStep can only replay a single numeric operand stored in the operation graph."
        )

    operand = params.get("operand")
    if isinstance(operand, int | float) and not isinstance(operand, bool):
        return operand
    if not isinstance(operand, Mapping) or set(operand) != {"type", "value"}:
        raise RecipeExtractionError(
            f"Scalar operation requires a numeric scalar operand\n  Operation: {operation}\n  Operand: {operand!r}"
        )

    operand_type = operand["type"]
    value = operand["value"]
    if operand_type == "bool" or isinstance(value, bool):
        raise RecipeExtractionError(
            "Scalar operation requires a numeric scalar operand\n"
            f"  Operation: {operation}\n"
            f"  Operand type: {operand_type!r}"
        )
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    raise RecipeExtractionError(
        "Scalar operation requires a numeric scalar operand\n"
        f"  Operation: {operation}\n"
        f"  Operand type: {operand_type!r}"
    )


def _scalar_step_from_graph(operation: str, params: Mapping[str, Any]) -> ScalarOperationStep:
    symbol = params.get("symbol", operation)
    if symbol != operation:
        raise RecipeExtractionError(
            f"Scalar operation graph has inconsistent operator metadata\n  Operation: {operation}\n  Symbol: {symbol!r}"
        )
    operand_position = params.get("operand_position", "right")
    if operand_position not in {"left", "right"}:
        raise RecipeExtractionError(
            "Scalar operation graph has invalid operand position metadata\n"
            f"  Operation: {operation}\n"
            f"  Operand position: {operand_position!r}"
        )
    try:
        return ScalarOperationStep(
            operation,
            _scalar_operand_from_params(operation, params),
            reverse=operand_position == "left",
        )
    except TypeError as exc:
        raise RecipeExtractionError(
            "Scalar operation requires a stable numeric scalar operand\n"
            f"  Operation: {operation}\n"
            "  NaN operands are not replayable because recipe equality must remain stable."
        ) from exc


def _optional_int_from_params(params: Mapping[str, Any], key: str) -> int | None:
    if key not in params:
        raise RecipeExtractionError(
            f"Channel slice recipe extraction requires explicit slice params\n  Missing parameter: {key}"
        )
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise RecipeExtractionError(
            "Channel slice recipe extraction requires integer slice bounds\n"
            f"  Parameter: {key}\n"
            f"  Got: {type(value).__name__}"
        )
    return int(value)


def _slice_from_serialized(value: Any, *, context: str) -> slice:
    if not isinstance(value, Mapping):
        raise RecipeExtractionError(f"{context} requires serialized slice objects\n  Got: {type(value).__name__}")
    if not {"start", "stop", "step"}.issubset(value):
        raise RecipeExtractionError(f"{context} requires explicit start/stop/step slice params\n  Got: {value!r}")
    return slice(
        _optional_int_from_params(value, "start"),
        _optional_int_from_params(value, "stop"),
        _optional_int_from_params(value, "step"),
    )


def _axis_slices_from_params(params: Mapping[str, Any]) -> tuple[slice, ...]:
    raw_axis_slices = params.get("axis_slices")
    if not isinstance(raw_axis_slices, list | tuple) or not raw_axis_slices:
        raise RecipeExtractionError(
            f"Multidimensional slice recipe extraction requires non-empty axis_slices\n  Got: {raw_axis_slices!r}"
        )
    return tuple(
        _slice_from_serialized(raw_axis_slice, context="Multidimensional slice recipe extraction")
        for raw_axis_slice in raw_axis_slices
    )


def _indices_from_params(params: Mapping[str, Any]) -> list[int]:
    indices = params.get("indices")
    if not isinstance(indices, list | tuple) or not indices:
        raise RecipeExtractionError(f"Integer-list indexing recipe extraction requires indices\n  Got: {indices!r}")
    if not all(isinstance(index, numbers.Integral) and not isinstance(index, bool) for index in indices):
        raise RecipeExtractionError(
            f"Integer-list indexing recipe extraction requires integer indices\n  Got: {indices!r}"
        )
    return [int(index) for index in indices]


def _mask_from_params(params: Mapping[str, Any]) -> _BooleanMask:
    mask = params.get("mask")
    if not isinstance(mask, list | tuple) or not mask:
        raise RecipeExtractionError(f"Boolean-mask indexing recipe extraction requires mask values\n  Got: {mask!r}")
    if not all(
        isinstance(value, bool) or type(value).__module__ == "numpy" and type(value).__name__ == "bool"
        for value in mask
    ):
        raise RecipeExtractionError(f"Boolean-mask indexing recipe extraction requires bool values\n  Got: {mask!r}")
    return _BooleanMask(tuple(bool(value) for value in mask))


def _channel_key_from_getitem_params(params: Mapping[str, Any]) -> slice | list[int] | list[str] | _BooleanMask:
    indexing = params.get("indexing")
    if indexing == "channel_slice":
        return slice(
            _optional_int_from_params(params, "start"),
            _optional_int_from_params(params, "stop"),
            _optional_int_from_params(params, "step"),
        )
    if indexing == "boolean_mask":
        return _mask_from_params(params)
    if indexing == "integer_list":
        return _indices_from_params(params)
    if indexing == "label_list":
        labels = params.get("labels")
        if isinstance(labels, list | tuple) and all(isinstance(label, str) for label in labels):
            return list(labels)
    raise RecipeExtractionError(
        "Multidimensional indexing recipe extraction only supports slice, integer-list, "
        "or label-list channel selectors\n"
        f"  Parent indexing kind: {indexing!r}"
    )


def _channel_key_from_parent_graph(parent: Mapping[str, Any]) -> int | slice | list[int] | list[str] | _BooleanMask:
    operation = str(parent["operation"])
    params = cast(Mapping[str, Any], _restore_history_value(parent.get("params", {})))
    if operation == "get_channel":
        channel_idx = params.get("channel_idx")
        if isinstance(channel_idx, numbers.Integral) and not isinstance(channel_idx, bool):
            return int(channel_idx)
        raise RecipeExtractionError(
            "Multidimensional indexing recipe extraction only supports single integer get_channel parents\n"
            f"  Parent params: {params!r}"
        )
    if operation == "__getitem__":
        return _channel_key_from_getitem_params(params)
    raise RecipeExtractionError(
        "Multidimensional indexing recipe extraction requires a replayable channel-selection parent\n"
        f"  Parent operation: {operation}"
    )


def _getitem_step_from_graph(params: Mapping[str, Any]) -> IndexingStep:
    indexing = params.get("indexing")
    if indexing not in _REPLAYABLE_GETITEM_INDEXING:
        raise RecipeExtractionError(
            "Indexing recipe extraction only supports channel-only slice, label list, "
            "and multidimensional slice selection\n"
            f"  Indexing kind: {indexing!r}\n"
            "  Multidimensional, callable, regex, dict, and array indexing need a selection recipe model "
            "that can preserve full indexing intent."
        )
    if indexing == "channel_slice":
        return IndexingStep(
            slice(
                _optional_int_from_params(params, "start"),
                _optional_int_from_params(params, "stop"),
                _optional_int_from_params(params, "step"),
            )
        )
    if indexing == "integer_list":
        return IndexingStep(_indices_from_params(params))
    if indexing == "boolean_mask":
        return IndexingStep(_mask_from_params(params))
    if indexing == "multidimensional_slice":
        return IndexingStep((slice(None), *_axis_slices_from_params(params)))
    labels = params.get("labels")
    if not isinstance(labels, list | tuple) or not all(isinstance(label, str) for label in labels):
        raise RecipeExtractionError(f"Label-list indexing recipe extraction requires string labels\n  Got: {labels!r}")
    return IndexingStep(list(labels))


def _multidimensional_steps_from_graph(
    params: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> tuple[RecipeStep, ...]:
    axis_slices = _axis_slices_from_params(params)
    parent_inputs = tuple(parent.get("inputs", ()))
    if len(parent_inputs) > 1:
        raise RecipeExtractionError(
            "Graph operation requires graph recipe support\n"
            f"  Operation: {parent['operation']}\n"
            f"  Parent count: {len(parent_inputs)}\n"
            "  Current RecipeSpec can only replay one linear parent chain."
        )
    grandparent_steps = _steps_from_graph(cast(Mapping[str, Any], parent_inputs[0])) if parent_inputs else ()
    return (*grandparent_steps, IndexingStep((_channel_key_from_parent_graph(parent), *axis_slices)))


def _step_from_graph(
    operation: str,
    params: Mapping[str, Any],
    kind: str | None,
    custom_metadata: Mapping[str, Any] | None = None,
) -> RecipeStep:
    if operation == "__getitem__":
        return _getitem_step_from_graph(params)
    if operation == "custom":
        return _custom_function_step_from_graph(params, custom_metadata)
    if operation in _REPLAYABLE_METHOD_OPERATIONS:
        return _method_step_from_graph(operation, params)
    if operation in _REPLAYABLE_TYPED_METHOD_OPERATIONS:
        return _typed_method_step_from_graph(operation, params, kind)
    if operation in _REPLAYABLE_SCALAR_OPERATIONS:
        return _scalar_step_from_graph(operation, params)
    _validate_replayable_operation(operation)
    return OperationSpec(operation, params)


def _steps_from_graph(graph: Mapping[str, Any]) -> tuple[RecipeStep, ...]:
    operation = str(graph["operation"])
    kind = cast(str | None, graph.get("kind"))
    if kind == "source":
        return ()
    inputs = tuple(graph.get("inputs", ()))
    if operation == "add_channel":
        raise RecipeExtractionError(
            "add_channel recipe extraction requires external input support\n"
            "  Current RecipeSpec can only replay one existing input frame. "
            "Adding channels needs a graph recipe or external data/input reference."
        )
    if len(inputs) > 1:
        raise RecipeExtractionError(
            "Graph operation requires graph recipe support\n"
            f"  Operation: {operation}\n"
            f"  Parent count: {len(inputs)}\n"
            "  Current RecipeSpec can only replay one linear parent chain."
        )

    params = cast(Mapping[str, Any], _restore_history_value(graph.get("params", {})))
    if operation == "__getitem__" and params.get("indexing") == "multidimensional_slice":
        if len(inputs) != 1:
            raise RecipeExtractionError(
                "Multidimensional indexing recipe extraction requires one channel-selection parent\n"
                f"  Parent count: {len(inputs)}"
            )
        return _multidimensional_steps_from_graph(params, cast(Mapping[str, Any], inputs[0]))
    parent_steps = _steps_from_graph(cast(Mapping[str, Any], inputs[0])) if inputs else ()
    step = _step_from_graph(
        operation,
        params,
        kind,
        cast(Mapping[str, Any] | None, graph.get("custom")),
    )
    return (*parent_steps, step)


def _binary_frame_step_from_graph(operation: str, params: Mapping[str, Any], left: str, right: str) -> BinaryFrameStep:
    if operation == "add_with_snr":
        return BinaryFrameStep("add_with_snr", left, right, {"snr": params["snr"]})
    if operation in _REPLAYABLE_SCALAR_OPERATIONS and params.get("operand_kind") == "frame":
        return BinaryFrameStep(operation, left, right)
    raise RecipeExtractionError(
        "GraphRecipeSpec extraction only supports root binary frame operations\n"
        f"  Operation: {operation}\n"
        f"  Params: {params!r}\n"
        "  Supported roots: frame-frame '+', '-', '*', '/', '**', and add_with_snr."
    )


def _is_external_operand_graph(operation: str, params: Mapping[str, Any]) -> bool:
    if operation not in _REPLAYABLE_SCALAR_OPERATIONS or params.get("operand_kind") != "operand":
        return False
    operand = params.get("operand")
    return isinstance(operand, Mapping) and operand.get("type") in {"ndarray", "dask.array"}


def _binary_operand_step_from_graph(
    operation: str,
    params: Mapping[str, Any],
    frame: str,
    operand: str,
) -> BinaryOperandStep:
    symbol = params.get("symbol", operation)
    if symbol != operation:
        raise RecipeExtractionError(
            f"Binary operand graph has inconsistent operator metadata\n  Operation: {operation}\n  Symbol: {symbol!r}"
        )
    if not _is_external_operand_graph(operation, params):
        raise RecipeExtractionError(
            "Binary operand recipe extraction only supports external ndarray and dask.array operands\n"
            f"  Operation: {operation}\n"
            f"  Params: {params!r}"
        )
    return BinaryOperandStep(operation, frame, operand)


def _add_channel_step_from_graph(params: Mapping[str, Any], base: str, added: str) -> AddChannelStep:
    if params.get("input_kind") != "frame":
        raise RecipeExtractionError(
            "add_channel recipe extraction only supports ChannelFrame inputs\n"
            f"  Input kind: {params.get('input_kind')!r}\n"
            "  Raw ndarray and dask.array inputs need an explicit data serialization or external-input policy."
        )
    return AddChannelStep(
        base,
        added,
        {
            "align": params.get("align", "strict"),
            "label": params.get("label"),
            "suffix_on_dup": params.get("suffix_on_dup"),
        },
    )


def _is_external_add_channel_data_graph(operation: str, params: Mapping[str, Any]) -> bool:
    return operation == "add_channel" and params.get("input_kind") in {"ndarray", "dask.array"}


def _add_channel_data_step_from_graph(params: Mapping[str, Any], base: str, data: str) -> AddChannelDataStep:
    if not _is_external_add_channel_data_graph("add_channel", params):
        raise RecipeExtractionError(
            "add_channel data recipe extraction only supports external ndarray and dask.array inputs\n"
            f"  Input kind: {params.get('input_kind')!r}"
        )
    return AddChannelDataStep(
        base,
        data,
        {
            "align": params.get("align", "strict"),
            "label": params.get("label"),
            "suffix_on_dup": params.get("suffix_on_dup"),
            "source_time_offset": params.get("source_time_offset"),
        },
    )


def _split_graph_at_binary_merge(graph: Mapping[str, Any]) -> tuple[Mapping[str, Any], tuple[RecipeStep, ...]]:
    inputs = tuple(graph.get("inputs", ()))
    if len(inputs) == 2:
        return graph, ()
    if len(inputs) != 1:
        raise RecipeExtractionError(
            "GraphRecipeSpec extraction requires one binary merge\n"
            f"  Operation: {graph.get('operation')!r}\n"
            f"  Parent count: {len(inputs)}\n"
            "  Supported shape: two replayable linear input chains, one binary merge, and one replayable linear tail."
        )

    merge_graph, tail_steps = _split_graph_at_binary_merge(cast(Mapping[str, Any], inputs[0]))
    operation = str(graph["operation"])
    params = cast(Mapping[str, Any], _restore_history_value(graph.get("params", {})))
    kind = cast(str | None, graph.get("kind"))
    return merge_graph, (
        *tail_steps,
        _step_from_graph(operation, params, kind, cast(Mapping[str, Any] | None, graph.get("custom"))),
    )


steps_from_graph = _steps_from_graph
