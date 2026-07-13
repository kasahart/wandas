"""Stable callable contracts used by Recipe persistence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from wandas.processing.semantic import OperationContract

MultiInputHandler = Callable[[tuple[Any, ...], Mapping[str, Any]], Any]


def replay_method(operation_id: str, *, version: int = 1) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    contract = OperationContract(operation_id, version, True, ())

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        setattr(method, "__wandas_replay_contract__", contract)
        return method

    return decorate


def multi_input_handler(
    operation_id: str, *, version: int, roles: tuple[str, ...]
) -> Callable[[MultiInputHandler], MultiInputHandler]:
    contract = OperationContract(operation_id, version, True, ())

    def decorate(handler: MultiInputHandler) -> MultiInputHandler:
        setattr(handler, "__wandas_multi_input_contract__", (contract, roles))
        return handler

    return decorate
