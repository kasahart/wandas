# wandas/utils/types.py
from typing import Any

import numpy as np
import numpy.typing as npt

# np.floating, np.complexfloating はジェネリック型なので、Any を型パラメータとして指定
Real = np.number[Any]
Complex = np.complexfloating[Any, Any]

# 実数型の要素を持つ NumPy 配列のエイリアス
NDArrayReal = npt.NDArray[Real]
# 複素数型の要素を持つ NumPy 配列のエイリアス
NDArrayComplex = npt.NDArray[Complex]

# np.floating, np.complexfloating はジェネリック型なので、Any を型パラメータとして指定
