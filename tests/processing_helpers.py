"""Shared Dask execution adapters for processing-layer tests."""

from typing import Any, Protocol

import numpy as np
from dask.array.core import Array as DaArray

from wandas.utils.dask_helpers import da_from_array


class ProcessingOperation(Protocol):
    """Structural contract required by the processing test runners."""

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        """Build a lazy processing graph for channel-first inputs."""
        ...


def as_operation_dask(data: Any) -> DaArray:
    """Return operation input as channel-wise Dask data.

    Existing Dask arrays retain their caller-selected chunks. Concrete 1D
    arrays receive the channel axis required by ``AudioOperation.process``.
    """
    if isinstance(data, DaArray):
        return data
    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    chunks = (1, *(-1,) * (array.ndim - 1))
    return da_from_array(array, chunks=chunks)


def run_operation_lazy(
    operation: ProcessingOperation,
    data: Any,
    *inputs: Any,
) -> DaArray:
    """Build an operation graph from concrete or existing Dask inputs."""
    return operation.process(
        as_operation_dask(data),
        *(as_operation_dask(input_data) for input_data in inputs),
    )


def run_operation_eager(
    operation: ProcessingOperation,
    data: Any,
    *inputs: Any,
) -> Any:
    """Compute an operation result after applying the shared input contract."""
    return run_operation_lazy(operation, data, *inputs).compute()
