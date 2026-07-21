"""Private storage keys for xarray-backed channel metadata."""

from typing import Final

_CHANNEL_IDS_ATTR: Final[str] = "channel_ids"
_CHANNEL_LABEL_KEY: Final[str] = "channel_label"
_CHANNEL_UNIT_KEY: Final[str] = "channel_unit"
_CHANNEL_REF_KEY: Final[str] = "channel_ref"
_CHANNEL_CALIBRATION_FACTOR_KEY: Final[str] = "channel_calibration_factor"
_CHANNEL_EXTRA_ATTR: Final[str] = "channel_extra"

# Vector metadata uses coordinates on channel-aware Frames and attrs only as a
# fallback. ``channel_extra`` remains an attr in both representations.
_CHANNEL_COORD_FALLBACK_ATTRS: Final[tuple[str, ...]] = (
    _CHANNEL_IDS_ATTR,
    _CHANNEL_LABEL_KEY,
    _CHANNEL_UNIT_KEY,
    _CHANNEL_REF_KEY,
    _CHANNEL_CALIBRATION_FACTOR_KEY,
)
