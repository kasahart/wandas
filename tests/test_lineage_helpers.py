from __future__ import annotations

from typing import Any

import pytest

from wandas.lineage import (
    _collection_graph,
    _flatten_graph,
    _operation_from_args,
    _operation_marker,
    _OperationNode,
    _ordered_operation_nodes,
    _prefers_nested_task,
    _reference_key,
    _subgraph_substitutions,
    _SubgraphTask,
    _walk_references,
    _walk_values,
)
from wandas.processing.base import AudioOperation, _execute_wandas_operation
from wandas.utils.types import NDArrayReal


class _LineageNoop(AudioOperation[NDArrayReal, NDArrayReal]):
    name = "_lineage_noop"

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        return x


def _noop(*args: object) -> tuple[object, ...]:
    return args


def test_collection_graph_requires_mapping_result() -> None:
    class BadGraph:
        def to_dict(self) -> list[object]:
            return []

    class BadCollection:
        def __dask_graph__(self) -> BadGraph:
            return BadGraph()

    with pytest.raises(TypeError, match="Dask graph must be a mapping"):
        _collection_graph(BadCollection())


def test_subgraph_task_without_dependencies_reports_empty_dependency_set() -> None:
    task = _SubgraphTask(task=object(), substitutions={}, key_map={})

    assert task.dependencies == frozenset()


def test_flatten_graph_discovers_nested_graph_mapping_tasks() -> None:
    operation = _LineageNoop(16000)
    graph = {
        "root": (_noop, {"nested": (_execute_wandas_operation, operation, "source")}),
    }

    flattened = _flatten_graph(graph)

    assert "nested" in flattened
    assert _operation_marker(flattened["nested"]) is operation


def test_lineage_helper_predicates_and_fallbacks() -> None:
    operation = _LineageNoop(16000)

    def _execute_subgraph() -> None:
        return None

    assert _prefers_nested_task((_noop, "source"), (_execute_wandas_operation, operation, "source")) is True
    assert _operation_marker((_execute_subgraph, {}, "output")) is None
    assert _subgraph_substitutions((_noop, "output")) == {}
    assert _operation_from_args(("source",)) is None


def test_ordered_operation_nodes_falls_back_when_dependencies_cycle() -> None:
    first = _LineageNoop(16000)
    second = _LineageNoop(16000)
    nodes = {
        "first": _OperationNode("first", first, frozenset({"second"})),
        "second": _OperationNode("second", second, frozenset({"first"})),
    }

    assert _ordered_operation_nodes(nodes, {}) == list(nodes.values())


def test_reference_and_walk_helpers_ignore_unhashable_values() -> None:
    class Reference:
        key: Any = []

    assert _reference_key([], set()) is None
    assert _reference_key(Reference(), set()) is None
    assert list(_walk_references({"left": ["right"]}, {"left", "right"})) == ["left", "right"]
    assert list(_walk_values({"left": ["right"]}, include_containers=True)) == [
        {"left": ["right"]},
        "left",
        ["right"],
        "right",
        "right",
    ]
