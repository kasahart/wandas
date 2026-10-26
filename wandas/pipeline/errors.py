from __future__ import annotations

from typing import Any


class RecipeExtractionError(ValueError):
    """Raised when a frame lineage cannot be represented by a recipe."""


class RecipeSerializationError(ValueError):
    """Raised when a Recipe payload violates the canonical schema."""


class RecipeValidationError(ValueError):
    """Raised when a Recipe graph or registered operation contract is invalid."""


class RecipeExecutionError(RuntimeError):
    """Raised when a validated Recipe cannot be applied to runtime inputs.

    Attributes:
        node_id: Id of the Recipe node that failed, or ``None`` when the error is
            not tied to one node (for example, mismatched input names).
        operation: Registered operation id of that node (``RecipeNode.operation``,
            for example ``"wandas.audio.lowpass_filter"``), or ``None``.
        cause: The exception raised by the operation, the same object as
            ``__cause__``, or ``None`` when the node failed a Recipe check itself.
            Like ``__cause__``, it is not pickled, so it is ``None`` after the error
            crosses a process boundary; ``node_id``, ``operation`` and the message
            are kept.
    """

    def __init__(
        self,
        message: str,
        *,
        node_id: str | None = None,
        operation: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.node_id = node_id
        self.operation = operation
        self.cause = cause

    def __reduce__(self) -> tuple[Any, ...]:
        # The cause may not survive pickling (an exception whose __init__ does not take
        # its args), so leave it out like __cause__.
        return (
            _rebuild_recipe_execution_error,
            (type(self), self.args, self.node_id, self.operation),
        )


def _rebuild_recipe_execution_error(
    cls: type[RecipeExecutionError],
    args: tuple[Any, ...],
    node_id: str | None,
    operation: str | None,
) -> RecipeExecutionError:
    error = cls.__new__(cls)
    error.args = args
    error.node_id = node_id
    error.operation = operation
    error.cause = None
    return error
