from __future__ import annotations

from typing import cast

from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq as _mosqito_center_freq

from wandas.utils.types import NDArrayReal


def center_freq(
    fmin: float,
    fmax: float,
    n: int = 3,
    G: int = 10,  # noqa: N803
    fr: int = 1000,
) -> tuple[NDArrayReal, NDArrayReal]:
    return cast(
        tuple[NDArrayReal, NDArrayReal],
        _mosqito_center_freq(fmin=fmin, fmax=fmax, n=n, G=G, fr=fr),
    )
