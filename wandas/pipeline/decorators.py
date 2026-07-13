"""Stable callable contracts used by Recipe persistence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from wandas.processing.semantic import OperationContract, replay_method, terminal_method

__all__ = ["multi_input_handler", "replay_method", "terminal_method"]

MultiInputHandler = Callable[[tuple[Any, ...], Mapping[str, Any]], Any]


def multi_input_handler(
    operation_id: str, *, version: int, roles: tuple[str, ...]
) -> Callable[[MultiInputHandler], MultiInputHandler]:
    contract = OperationContract(operation_id, version, True, ())

    def decorate(handler: MultiInputHandler) -> MultiInputHandler:
        setattr(handler, "__wandas_multi_input_contract__", (contract, roles))
        return handler

    return decorate
