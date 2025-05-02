# wandas/utils/types.py
from typing import Any

import numpy as np
import numpy.typing as npt

# np.floating and np.complexfloating are generic types,
# so we specify Any as the type parameter
Real = np.number[Any]
Complex = np.complexfloating[Any, Any]

# Type alias for NumPy arrays with real number elements
NDArrayReal = npt.NDArray[Real]
# Type alias for NumPy arrays with complex number elements
NDArrayComplex = npt.NDArray[Complex]
