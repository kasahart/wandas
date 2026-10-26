from __future__ import annotations


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
