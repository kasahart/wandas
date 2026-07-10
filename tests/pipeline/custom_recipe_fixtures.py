from functools import partial

import numpy as np


def custom_scale(data: np.ndarray, gain: float) -> np.ndarray:
    return data * gain


class CallableScale:
    def __call__(self, data: np.ndarray, gain: float) -> np.ndarray:
        return data * gain


callable_scale = CallableScale()
partial_scale = partial(custom_scale, gain=2.0)


def same_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape


def custom_rfft(data: np.ndarray) -> np.ndarray:
    return np.fft.rfft(data, axis=-1)


def rfft_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return (shape[0], shape[1] // 2 + 1)
