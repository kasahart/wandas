from wandas.core._channel_schema import (
    _CHANNEL_CALIBRATION_FACTOR_KEY,
    _CHANNEL_COORD_FALLBACK_ATTRS,
    _CHANNEL_EXTRA_ATTR,
    _CHANNEL_IDS_ATTR,
    _CHANNEL_LABEL_KEY,
    _CHANNEL_REF_KEY,
    _CHANNEL_UNIT_KEY,
)


def test_private_channel_storage_keys_preserve_existing_schema() -> None:
    assert _CHANNEL_IDS_ATTR == "channel_ids"
    assert _CHANNEL_LABEL_KEY == "channel_label"
    assert _CHANNEL_UNIT_KEY == "channel_unit"
    assert _CHANNEL_REF_KEY == "channel_ref"
    assert _CHANNEL_CALIBRATION_FACTOR_KEY == "channel_calibration_factor"
    assert _CHANNEL_EXTRA_ATTR == "channel_extra"


def test_channel_coord_fallback_attrs_exclude_canonical_extra_storage() -> None:
    assert _CHANNEL_COORD_FALLBACK_ATTRS == (
        "channel_ids",
        "channel_label",
        "channel_unit",
        "channel_ref",
        "channel_calibration_factor",
    )
    assert "channel_extra" not in _CHANNEL_COORD_FALLBACK_ATTRS
