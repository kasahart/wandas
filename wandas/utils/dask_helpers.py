from typing import Any

import dask.array as da
from dask.array.core import Array as DaArray


def da_from_array(data: Any, chunks: Any | None = None, **kwargs: Any) -> DaArray:
    """Wrapper for dask.array.from_array that accepts Any for chunks.

    This helper hides typing mismatches from mypy for chunk tuples such as
    (1, -1) while preserving runtime behavior.
    """
    return da.from_array(data, chunks=chunks, **kwargs)
