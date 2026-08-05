"""Importable eager Operation used to exercise module reload registration."""

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal


class ReloadableOperation(AudioOperation[NDArrayReal, NDArrayReal]):
    """Operation whose class identity changes when this module is reloaded."""

    name = "test_reloadable_operation"

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        return x


register_operation(ReloadableOperation)
