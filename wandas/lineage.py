"""Extract Wandas operation lineage from lazy Dask graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from wandas.processing.base import AudioOperation, BinaryOperation, _execute_wandas_operation, _mark_wandas_operation

_OPERATION_MARKER_FUNCTIONS = {_execute_wandas_operation, _mark_wandas_operation}
OperationMarker = AudioOperation[Any, Any] | BinaryOperation


@dataclass(frozen=True)
class _OperationNode:
    key: Any
    operation: OperationMarker
    dependencies: frozenset[Any]


@dataclass(frozen=True)
class _SubgraphTask:
    task: Any
    substitutions: Mapping[Any, Any]
    key_map: Mapping[Any, Any]

    @property
    def dependencies(self) -> frozenset[Any]:
        dependencies = getattr(self.task, "dependencies", None)
        if dependencies is None:
            return frozenset()
        resolved: set[Any] = set()
        for dependency in dependencies:
            value = _resolve_subgraph_value(dependency, self.substitutions, self.key_map)
            try:
                hash(value)
            except TypeError:
                continue
            resolved.add(value)
        return frozenset(resolved)


@dataclass(frozen=True)
class _SubgraphOutputTask:
    task: Any
    output_dependency: Any

    @property
    def dependencies(self) -> frozenset[Any]:
        return frozenset({self.output_dependency})


def extract_operations(collection: Any) -> tuple[OperationMarker, ...]:
    """Return Wandas operations embedded in an unoptimized Dask collection graph."""
    graph = _collection_graph(collection)
    nodes = _operation_nodes(graph)
    if not nodes:
        return ()
    return tuple(node.operation for node in _ordered_operation_nodes(nodes, graph))


def _collection_graph(collection: Any) -> Mapping[Any, Any]:
    dask_graph_method = getattr(collection, "__dask_graph__", None)
    if not callable(dask_graph_method):
        raise TypeError(f"Expected a Dask collection with __dask_graph__(), got {type(collection).__name__}")
    dask_graph = dask_graph_method()
    if hasattr(dask_graph, "to_dict"):
        graph = dask_graph.to_dict()
    else:
        graph = dict(dask_graph)
    if not isinstance(graph, Mapping):
        raise TypeError(f"Dask graph must be a mapping, got {type(graph).__name__}")
    return _flatten_graph(graph)


def _flatten_graph(graph: Mapping[Any, Any]) -> Mapping[Any, Any]:
    flattened: dict[Any, Any] = dict(graph)
    stack = list(graph.items())
    seen_tasks = {id(task) for task in graph.values()}
    while stack:
        key, task = stack.pop()
        subgraph_tasks, output_dependency = _subgraph_tasks(key, task)
        if output_dependency is not None:
            flattened[key] = _SubgraphOutputTask(task, output_dependency)
        for nested_key, nested_task in subgraph_tasks.items():
            if nested_key not in flattened or _prefers_nested_task(flattened[nested_key], nested_task):
                flattened[nested_key] = nested_task
                stack.append((nested_key, nested_task))
        if output_dependency is not None:
            continue

        _, args = _task_func_and_args(task)
        for value in _walk_values(args, include_containers=True):
            if _looks_like_dask_task(value) and id(value) not in seen_tasks:
                seen_tasks.add(id(value))
                stack.append((_dependency_key(value), value))
            if not _is_graph_mapping(value):
                continue
            for key, nested_task in value.items():
                if key not in flattened or _prefers_nested_task(flattened[key], nested_task):
                    flattened[key] = nested_task
                    stack.append((key, nested_task))
    return flattened


def _subgraph_tasks(parent_key: Any, task: Any) -> tuple[Mapping[Any, _SubgraphTask], Any | None]:
    func, args = _task_func_and_args(task)
    if getattr(func, "__name__", None) != "_execute_subgraph" or len(args) < 2 or not isinstance(args[0], Mapping):
        return {}, None
    subgraph = args[0]
    key_map = {key: (parent_key, key) for key in subgraph}
    substitutions = _subgraph_substitutions(args)
    output_dependency = key_map.get(_dependency_key(args[1]))
    return {
        key_map[key]: _SubgraphTask(nested_task, substitutions=substitutions, key_map=key_map)
        for key, nested_task in subgraph.items()
    }, output_dependency


def _prefers_nested_task(existing_task: Any, nested_task: Any) -> bool:
    return _operation_marker(existing_task) is None and _operation_marker(nested_task) is not None


def _is_graph_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(_looks_like_dask_task(item) for item in value.values())


def _looks_like_dask_task(value: Any) -> bool:
    return (
        hasattr(value, "func")
        or hasattr(value, "target")
        or value.__class__.__module__.startswith("dask.")
        or isinstance(value, tuple)
    )


def _operation_nodes(graph: Mapping[Any, Any]) -> dict[Any, _OperationNode]:
    nodes: dict[Any, _OperationNode] = {}
    operation_keys: dict[int, Any] = {}
    for key, task in graph.items():
        marker = _operation_marker(task)
        if marker is None:
            continue
        dependencies = frozenset(_task_dependencies(task, graph))
        operation_id = id(marker)
        existing_key = operation_keys.get(operation_id)
        if existing_key is not None:
            continue
        operation_keys[operation_id] = key
        nodes[key] = _OperationNode(key=key, operation=marker, dependencies=dependencies)
    return nodes


def _operation_marker(task: Any) -> OperationMarker | None:
    func, args = _task_func_and_args(task)
    if getattr(func, "__name__", None) == "_execute_subgraph":
        return None
    if not any(func is marker_func for marker_func in _OPERATION_MARKER_FUNCTIONS):
        return None
    return _operation_from_args(args)


def _subgraph_substitutions(args: tuple[Any, ...]) -> dict[Any, Any]:
    if len(args) < 3 or not isinstance(args[2], tuple):
        return {}
    return dict(zip(args[2], args[3:], strict=False))


def _operation_from_args(
    args: tuple[Any, ...],
    substitutions: Mapping[Any, Any] | None = None,
) -> OperationMarker | None:
    substitutions = substitutions or {}
    for arg in args:
        value = _literal_value(arg)
        if not isinstance(value, AudioOperation | BinaryOperation):
            value = _literal_value(substitutions.get(_dependency_key(arg), value))
        if isinstance(value, AudioOperation | BinaryOperation):
            return value
    return None


def _literal_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _task_func_and_args(task: Any) -> tuple[Any, tuple[Any, ...]]:
    if isinstance(task, _SubgraphTask):
        func, args = _task_func_and_args(task.task)
        return (
            func,
            tuple(_resolve_subgraph_value(arg, task.substitutions, task.key_map) for arg in args),
        )
    func = getattr(task, "func", None)
    args = getattr(task, "args", None)
    if func is not None and args is not None:
        return func, tuple(args)
    if isinstance(task, tuple) and task:
        return task[0], tuple(task[1:])
    return None, ()


def _task_dependencies(task: Any, graph: Mapping[Any, Any]) -> set[Any]:
    graph_keys = set(graph)
    task_args = _task_func_and_args(task)[1]
    discovered = {
        key
        for key in (_reference_key(value, graph_keys) for value in _walk_references(task_args, graph_keys))
        if key is not None
    }
    discovered.update(_inline_task_output_dependencies(task, graph_keys))

    dependencies = getattr(task, "dependencies", None)
    if dependencies is not None:
        return {_dependency_key(dep) for dep in dependencies} | discovered

    return discovered


def _inline_task_output_dependencies(task: Any, graph_keys: set[Any]) -> set[Any]:
    dependencies: set[Any] = set()
    for value in _walk_values(_task_func_and_args(task)[1]):
        if not _looks_like_dask_task(value):
            continue
        func, args = _task_func_and_args(value)
        if getattr(func, "__name__", None) != "_execute_subgraph" or len(args) < 2:
            continue
        output_key = _reference_key(args[1], graph_keys)
        if output_key is not None:
            dependencies.add(output_key)
    return dependencies


def _ordered_operation_nodes(
    nodes: Mapping[Any, _OperationNode],
    graph: Mapping[Any, Any],
) -> list[_OperationNode]:
    direct_operation_deps = {
        key: _reachable_operation_dependencies(node.dependencies, nodes, graph) - {key} for key, node in nodes.items()
    }

    ordered: list[_OperationNode] = []
    remaining = dict(nodes)
    while remaining:
        progressed = False
        for key, node in list(remaining.items()):
            if direct_operation_deps[key].isdisjoint(remaining):
                ordered.append(node)
                del remaining[key]
                progressed = True
        if not progressed:
            ordered.extend(remaining.values())
            break
    return ordered


def _reachable_operation_dependencies(
    start_keys: Iterable[Any],
    nodes: Mapping[Any, _OperationNode],
    graph: Mapping[Any, Any],
) -> set[Any]:
    found: set[Any] = set()
    seen: set[Any] = set()
    stack = list(start_keys)
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        if key in nodes:
            found.add(key)
            continue
        task = graph.get(key)
        if task is not None:
            stack.extend(_task_dependencies(task, graph))
    return found


def _dependency_key(value: Any) -> Any:
    return getattr(value, "key", value)


def _resolve_subgraph_value(value: Any, substitutions: Mapping[Any, Any], key_map: Mapping[Any, Any]) -> Any:
    key = _dependency_key(value)
    try:
        if key in substitutions:
            return substitutions[key]
        if key in key_map:
            return key_map[key]
    except TypeError:
        pass
    return value


def _reference_key(value: Any, graph_keys: set[Any]) -> Any | None:
    try:
        if value in graph_keys:
            return value
    except TypeError:
        pass
    key = getattr(value, "key", None)
    try:
        if key in graph_keys:
            return key
    except TypeError:
        pass
    return None


def _walk_references(values: Any, graph_keys: set[Any]) -> Iterable[Any]:
    if _reference_key(values, graph_keys) is not None:
        yield values
        return
    if isinstance(values, Mapping):
        for key, value in values.items():
            yield from _walk_references(key, graph_keys)
            yield from _walk_references(value, graph_keys)
        return
    if isinstance(values, (tuple, list, set, frozenset)):
        for value in values:
            yield from _walk_references(value, graph_keys)
        return
    yield values


def _walk_values(values: Any, *, include_containers: bool = False) -> Iterable[Any]:
    if include_containers:
        yield values
    if isinstance(values, Mapping):
        for key, value in values.items():
            yield key
            yield from _walk_values(value, include_containers=include_containers)
        return
    if isinstance(values, (tuple, list, set, frozenset)):
        if isinstance(values, tuple):
            yield values
        for value in values:
            yield from _walk_values(value, include_containers=include_containers)
        return
    yield values
