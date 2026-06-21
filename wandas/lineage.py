"""Extract Wandas operation lineage from lazy Dask graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from wandas.processing.base import AudioOperation, _execute_wandas_operation


@dataclass(frozen=True)
class _OperationNode:
    key: Any
    operation: AudioOperation[Any, Any]
    dependencies: frozenset[Any]


def extract_operations(collection: Any) -> tuple[AudioOperation[Any, Any], ...]:
    """Return Wandas operations embedded in an unoptimized Dask collection graph."""
    graph = _collection_graph(collection)
    nodes = _operation_nodes(graph)
    if not nodes:
        return ()
    return tuple(node.operation for node in _ordered_operation_nodes(nodes, graph))


def _collection_graph(collection: Any) -> Mapping[Any, Any]:
    dask_graph = collection.__dask_graph__()
    if hasattr(dask_graph, "to_dict"):
        graph = dask_graph.to_dict()
    else:
        graph = dict(dask_graph)
    if not isinstance(graph, Mapping):
        raise TypeError(f"Dask graph must be a mapping, got {type(graph).__name__}")
    return _flatten_graph(graph)


def _flatten_graph(graph: Mapping[Any, Any]) -> Mapping[Any, Any]:
    flattened: dict[Any, Any] = dict(graph)
    stack = list(graph.values())
    seen_tasks = {id(task) for task in stack}
    while stack:
        task = stack.pop()
        _, args = _task_func_and_args(task)
        for value in _walk_values(args, include_containers=True):
            if _looks_like_dask_task(value) and id(value) not in seen_tasks:
                seen_tasks.add(id(value))
                stack.append(value)
            if not _is_graph_mapping(value):
                continue
            for key, nested_task in value.items():
                if key not in flattened:
                    flattened[key] = nested_task
                    stack.append(nested_task)
    return flattened


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
    for key, task in graph.items():
        marker = _operation_marker(task)
        if marker is None:
            continue
        nodes[key] = _OperationNode(
            key=key,
            operation=marker,
            dependencies=frozenset(_task_dependencies(task, graph)),
        )
    return nodes


def _operation_marker(task: Any) -> AudioOperation[Any, Any] | None:
    func, args = _task_func_and_args(task)
    if func is not _execute_wandas_operation:
        return None
    for arg in args:
        if isinstance(arg, AudioOperation):
            return arg
    return None


def _task_func_and_args(task: Any) -> tuple[Any, tuple[Any, ...]]:
    func = getattr(task, "func", None)
    args = getattr(task, "args", None)
    if func is not None and args is not None:
        return func, tuple(args)
    if isinstance(task, tuple) and task:
        return task[0], tuple(task[1:])
    return None, ()


def _task_dependencies(task: Any, graph: Mapping[Any, Any]) -> set[Any]:
    graph_keys = set(graph)
    discovered = {
        key
        for key in (_reference_key(value, graph_keys) for value in _walk_values(_task_func_and_args(task)[1]))
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
        key: _reachable_operation_dependencies(node.dependencies, nodes, graph) for key, node in nodes.items()
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


def _walk_values(values: Any, *, include_containers: bool = False) -> Iterable[Any]:
    if include_containers:
        yield values
    if isinstance(values, Mapping):
        for key, value in values.items():
            yield key
            yield from _walk_values(value, include_containers=include_containers)
        return
    if isinstance(values, tuple | list | set | frozenset):
        for value in values:
            yield from _walk_values(value, include_containers=include_containers)
        return
    yield values
