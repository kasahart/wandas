import copy
import json
from dataclasses import is_dataclass
from typing import Any, cast

import pytest

from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.utils.util import unit_to_ref


class TestChannelMetadata:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"factor": True},
            {"factor": "bad"},
            {"unit": 1},
            {"ref": True},
            {"ref": "bad"},
        ],
    )
    def test_channel_calibration_rejects_non_numeric_or_mistyped_fields(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(TypeError, match="Invalid channel calibration"):
            ChannelCalibration(**kwargs)

    def test_channel_calibration_is_immutable_and_validated(self) -> None:
        calibration = ChannelCalibration(factor=0.02, unit="Pa")

        assert calibration.factor == 0.02
        assert calibration.unit == "Pa"
        assert calibration.ref == 2e-5
        with pytest.raises(AttributeError):
            calibration.factor = 2.0  # ty: ignore[invalid-assignment]

    @pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf")])
    def test_channel_calibration_rejects_non_positive_or_non_finite_factor(self, factor: float) -> None:
        with pytest.raises(ValueError, match="Invalid channel calibration factor"):
            ChannelCalibration(factor=factor)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"unit": "   "},
            {"ref": 0.0},
            {"ref": float("nan")},
            {"ref": float("inf")},
        ],
    )
    def test_channel_calibration_rejects_invalid_unit_or_reference(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="Invalid channel calibration"):
            ChannelCalibration(**kwargs)

    def test_channel_metadata_owns_calibration_as_a_typed_field(self) -> None:
        calibration = ChannelCalibration(9.81, "m/s^2", 1.0)
        metadata = ChannelMetadata(label="accelerometer", calibration=calibration)

        assert metadata.calibration is calibration
        assert metadata.unit == "m/s^2"
        assert metadata.ref == 1.0
        assert "calibration" not in metadata.extra

    @pytest.mark.parametrize(
        "value",
        [
            None,
            {"factor": 1.0, "unit": "Pa"},
            {"factor": 1.0, "unit": "Pa", "ref": 2e-5, "extra": True},
        ],
    )
    def test_channel_calibration_from_dict_requires_exact_snapshot(self, value: object) -> None:
        with pytest.raises(ValueError, match="Invalid channel calibration snapshot"):
            ChannelCalibration.from_dict(value)

    def test_channel_metadata_rejects_conflicting_or_untyped_calibration(self) -> None:
        with pytest.raises(TypeError, match="calibration must be a ChannelCalibration"):
            ChannelMetadata(calibration=cast(Any, "bad"))
        with pytest.raises(ValueError, match="Conflicting channel calibration metadata"):
            ChannelMetadata(unit="Pa", calibration=ChannelCalibration())

        metadata = ChannelMetadata()
        with pytest.raises(TypeError, match="calibration must be a ChannelCalibration"):
            metadata.calibration = cast(Any, "bad")

        object.__setattr__(metadata, "calibration", "bad")
        with pytest.raises(TypeError, match="calibration must be a ChannelCalibration"):
            metadata.__post_init__()

    def test_channel_metadata_legacy_setters_validate_and_initialize_defensively(self) -> None:
        metadata = ChannelMetadata()
        with pytest.raises(TypeError, match="Invalid channel calibration unit"):
            metadata.unit = cast(Any, 1)
        with pytest.raises(TypeError, match="Invalid channel calibration reference"):
            metadata.ref = cast(Any, "bad")

        unit_only = ChannelMetadata.__new__(ChannelMetadata)
        unit_only.unit = "Pa"
        assert unit_only.calibration == ChannelCalibration(unit="Pa")

        ref_only = ChannelMetadata.__new__(ChannelMetadata)
        ref_only.ref = 2.0
        assert ref_only.calibration == ChannelCalibration(ref=2.0)

    def test_channel_metadata_is_dataclass(self) -> None:
        """ChannelMetadata uses stdlib dataclass semantics."""
        metadata = ChannelMetadata()

        assert is_dataclass(metadata)

    def test_init_default_values_empty_strings_and_dict(self) -> None:
        """Test initialization with default values returns empty strings and dict."""
        metadata: ChannelMetadata = ChannelMetadata()
        assert metadata.label == ""
        assert metadata.unit == ""
        assert metadata.extra == {}

    def test_init_custom_values_preserves_all_fields(self) -> None:
        """Test initialization with custom values preserves all fields."""
        metadata: ChannelMetadata = ChannelMetadata(
            label="test_label",
            unit="Hz",
            extra={"source": "microphone", "calibrated": True},
        )
        assert metadata.label == "test_label"
        assert metadata.unit == "Hz"
        assert metadata.extra == {"source": "microphone", "calibrated": True}

    def test_getitem_main_fields_returns_correct_values(self) -> None:
        """Test dictionary-like access for main fields returns correct values."""
        metadata: ChannelMetadata = ChannelMetadata(label="test_label", unit="Hz", ref=0.5)
        assert metadata["label"] == "test_label"
        assert metadata["unit"] == "Hz"
        assert metadata["ref"] == 0.5

    def test_getitem_extra_field_returns_value_or_none(self) -> None:
        """Test dictionary-like access for extra fields returns value or None."""
        metadata: ChannelMetadata = ChannelMetadata(extra={"source": "microphone", "calibrated": True})
        assert metadata["source"] == "microphone"
        assert metadata["calibrated"] is True
        # Non-existent key should return None
        assert metadata["nonexistent"] is None

    def test_setitem_main_fields_updates_values(self) -> None:
        """Test dictionary-like assignment for main fields updates values."""
        metadata: ChannelMetadata = ChannelMetadata()
        metadata["label"] = "new_label"
        metadata["unit"] = "dB"
        metadata["ref"] = 0.75
        assert metadata.label == "new_label"
        assert metadata.unit == "dB"
        assert metadata.ref == 0.75

    def test_setitem_extra_fields_stores_in_extra_dict(self) -> None:
        """Test dictionary-like assignment for extra fields stores in extra dict."""
        metadata: ChannelMetadata = ChannelMetadata()
        metadata["source"] = "microphone"
        metadata["calibrated"] = True
        assert metadata.extra == {"source": "microphone", "calibrated": True}

    def test_ref_auto_set_pa_unit_sets_2e5(self) -> None:
        """Test that ref is automatically set based on unit (Pa -> 2e-5)."""
        # Case 1: Initialize with unit "Pa" should set ref to 2e-5
        metadata: ChannelMetadata = ChannelMetadata(unit="Pa")
        assert metadata.unit == "Pa"
        assert metadata.ref == 2e-5

        # Case 2: Initialize with unit "Hz" should keep default ref (1.0)
        metadata2: ChannelMetadata = ChannelMetadata(unit="Hz")
        assert metadata2.unit == "Hz"
        assert metadata2.ref == 1.0

        # Case 3: Change unit via __setitem__ should update ref
        metadata3: ChannelMetadata = ChannelMetadata()
        metadata3["unit"] = "Pa"
        assert metadata3.unit == "Pa"
        assert metadata3.ref == 2e-5

        # Case 4: Change unit via property setter should update ref
        metadata3b: ChannelMetadata = ChannelMetadata()
        metadata3b.unit = "Pa"
        assert metadata3b.unit == "Pa"
        assert metadata3b.ref == 2e-5

        # Case 5: Property setter should work for other units too
        metadata3c: ChannelMetadata = ChannelMetadata()
        metadata3c.unit = "V"
        assert metadata3c.unit == "V"
        assert metadata3c.ref == 1.0  # Should be default for "V"

        # Case 6: Explicitly setting both unit and ref should keep specified ref
        metadata4: ChannelMetadata = ChannelMetadata(unit="Pa", ref=0.5)
        assert metadata4.unit == "Pa"
        assert metadata4.ref == 0.5  # ref should not be overridden

        metadata4b: ChannelMetadata = ChannelMetadata(unit="Pa", ref=1.0)
        assert metadata4b.unit == "Pa"
        assert metadata4b.ref == 1.0  # explicit ref should not be overridden

        # Case 7: Test with other units
        other_units = ["V", "m/s", "g"]
        for unit in other_units:
            expected_ref = unit_to_ref(unit)  # Should be 1.0 for these units
            metadata5: ChannelMetadata = ChannelMetadata(unit=unit)
            assert metadata5.unit == unit
            assert metadata5.ref == expected_ref

    def test_to_json_serializes_all_fields(self) -> None:
        """Test serialization to JSON includes all fields."""
        metadata: ChannelMetadata = ChannelMetadata(
            label="test_label",
            unit="Hz",
            extra={"source": "microphone", "calibrated": True},
        )
        json_data: str = metadata.to_json()
        # Validate it's proper JSON
        parsed: dict[str, Any] = json.loads(json_data)
        assert parsed["label"] == "test_label"
        assert parsed["calibration"] == {"factor": 1.0, "unit": "Hz", "ref": 1.0}
        assert parsed["extra"]["source"] == "microphone"
        assert parsed["extra"]["calibrated"] is True

    def test_from_json_deserializes_all_fields(self) -> None:
        """Test deserialization from JSON restores all fields."""
        json_data: str = """
        {
            "label": "test_label",
            "unit": "Hz",
            "extra": {
                "source": "microphone",
                "calibrated": true,
                "notes": "Test recording"
            }
        }
        """
        metadata: ChannelMetadata = ChannelMetadata.from_json(json_data)
        assert metadata.label == "test_label"
        assert metadata.unit == "Hz"
        assert metadata.extra["source"] == "microphone"
        assert metadata.extra["calibrated"] is True
        assert metadata.extra["notes"] == "Test recording"

    def test_from_json_preserves_explicit_unit_ref_one(self) -> None:
        """JSON with explicit ref=1.0 preserves that value."""
        metadata = ChannelMetadata.from_json('{"unit": "Pa", "ref": 1.0, "extra": {}}')

        assert metadata.unit == "Pa"
        assert metadata.ref == 1.0

    def test_from_json_rejects_mixed_current_and_legacy_calibration_fields(self) -> None:
        payload = {
            "unit": "Pa",
            "calibration": {"factor": 1.0, "unit": "Pa", "ref": 2e-5},
        }

        with pytest.raises(ValueError, match="must not combine calibration"):
            ChannelMetadata.from_json(json.dumps(payload))

    def test_from_json_rejects_explicit_null_calibration(self) -> None:
        with pytest.raises(ValueError, match="Invalid channel calibration snapshot"):
            ChannelMetadata.from_json('{"calibration": null}')

    def test_from_json_rejects_non_object_json(self) -> None:
        """ChannelMetadata JSON must decode to an object."""
        with pytest.raises(ValueError, match="ChannelMetadata JSON must decode to an object"):
            ChannelMetadata.from_json('["not", "an", "object"]')

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"label": 1}, "Channel label must be a string"),
            ({"unit": 1}, "Invalid channel calibration unit"),
            ({"ref": "bad"}, "Invalid channel calibration reference"),
            ({"ref": True}, "Invalid channel calibration reference"),
            ({"extra": []}, "Channel extra must be a mapping"),
            ({"extra": None}, "Channel extra must be a mapping"),
        ],
    )
    def test_init_rejects_invalid_field_types(self, kwargs: dict[str, Any], message: str) -> None:
        """ChannelMetadata rejects invalid dataclass field types."""
        with pytest.raises(TypeError, match=message):
            ChannelMetadata(**kwargs)

    def test_extra_dict_is_copied_from_caller_input(self) -> None:
        """Caller-owned extra dictionaries are not stored directly."""
        extra: dict[str, Any] = {"nested": {"gain": 10}}
        metadata = ChannelMetadata(extra=extra)

        extra["nested"]["gain"] = 99
        extra["new"] = "value"

        assert metadata.extra == {"nested": {"gain": 10}}

    def test_from_json_rejects_explicit_null_extra(self) -> None:
        """JSON extra must be a mapping when explicitly provided."""
        with pytest.raises(ValueError, match="Channel extra must be a mapping"):
            ChannelMetadata.from_json('{"extra": null}')

    def test_init_converts_numeric_ref_to_float(self) -> None:
        """ChannelMetadata stores valid numeric ref values as floats."""
        metadata = ChannelMetadata(ref=2)

        assert metadata.ref == 2.0
        assert isinstance(metadata.ref, float)

    def test_from_json_wraps_malformed_json(self) -> None:
        """Malformed JSON raises a ChannelMetadata-specific ValueError."""
        with pytest.raises(ValueError, match="Invalid ChannelMetadata JSON:"):
            ChannelMetadata.from_json("{")

    def test_deepcopy_independent_from_original(self) -> None:
        """Standard deepcopy is independent from original."""
        metadata = ChannelMetadata(
            label="test_label",
            unit="Hz",
            extra={"source": "microphone", "nested": {"gain": 10}},
        )
        copied = copy.deepcopy(metadata)

        assert copied.label == metadata.label
        assert copied.unit == metadata.unit
        assert copied.extra == metadata.extra

        metadata.label = "modified_label"
        metadata.extra["new_key"] = "new_value"
        metadata.extra["nested"]["gain"] = 99

        assert copied.label == "test_label"
        assert "new_key" not in copied.extra
        assert copied.extra["nested"]["gain"] == 10

    def test_unicode_and_special_chars_roundtrip(self) -> None:
        """Test Unicode and special characters survive JSON round-trip."""
        metadata: ChannelMetadata = ChannelMetadata(
            label="测试标签",  # Chinese characters
            unit="°C",  # Degree symbol
            extra={"note": "Special chars: !@#$%^&*()"},
        )

        # Test serialization and deserialization with special chars
        json_data: str = metadata.to_json()
        deserialized: ChannelMetadata = ChannelMetadata.from_json(json_data)

        assert deserialized.label == "测试标签"
        assert deserialized.unit == "°C"
        assert deserialized.extra["note"] == "Special chars: !@#$%^&*()"

    def test_nested_extra_data_roundtrip(self) -> None:
        """Test nested structures in extra field survive JSON round-trip."""
        nested_data: dict[str, Any] = {
            "config": {"sampling": {"rate": 44100, "bits": 24}},
            "tags": ["audio", "speech", "raw"],
        }

        metadata: ChannelMetadata = ChannelMetadata(extra=nested_data)
        json_data: str = metadata.to_json()
        deserialized: ChannelMetadata = ChannelMetadata.from_json(json_data)

        assert deserialized.extra["config"]["sampling"]["rate"] == 44100
        assert deserialized.extra["config"]["sampling"]["bits"] == 24
        assert deserialized.extra["tags"] == ["audio", "speech", "raw"]

    def test_getitem_ref_returns_correct_value(self) -> None:
        """Test dictionary-like access for ref field returns correct value."""
        # Case 1: Initialize with custom ref value
        metadata: ChannelMetadata = ChannelMetadata(ref=0.25)
        assert metadata["ref"] == 0.25
        assert metadata.ref == 0.25

        # Case 2: Get ref field after setting unit (automatic ref update)
        metadata2: ChannelMetadata = ChannelMetadata()
        metadata2["unit"] = "Pa"
        assert metadata2["ref"] == 2e-5  # Should be updated based on unit

        # Case 3: Get ref field from default instance
        metadata3: ChannelMetadata = ChannelMetadata()
        assert metadata3["ref"] == 1.0  # Default value

    def test_setitem_ref_field_updates_ref(self) -> None:
        """Test dictionary-like assignment for ref field updates ref."""
        # Case 1: Set ref directly
        metadata: ChannelMetadata = ChannelMetadata()
        metadata["ref"] = 0.5
        assert metadata.ref == 0.5

        # Case 2: Set ref after unit was specified
        metadata2: ChannelMetadata = ChannelMetadata(unit="Pa")
        assert metadata2.ref == 2e-5  # Initially set by unit
        metadata2["ref"] = 0.75  # Override auto-set value
        assert metadata2.ref == 0.75

        # Case 3: Set ref and then set unit
        metadata3: ChannelMetadata = ChannelMetadata()
        metadata3["ref"] = 0.3
        assert metadata3.ref == 0.3
        metadata3["unit"] = "Pa"  # Setting unit should override ref
        assert metadata3.ref == 2e-5  # Should be updated based on unit

    def test_property_getters_return_correct_values(self) -> None:
        """Test property getter methods return correct values."""
        metadata: ChannelMetadata = ChannelMetadata(
            label="test_label",
            unit="Pa",
            ref=0.5,
            extra={"source": "microphone", "calibrated": True},
        )

        # Test direct field access
        assert metadata.label == "test_label"
        assert metadata.unit == "Pa"
        assert metadata.ref == 0.5
        assert metadata.extra == {"source": "microphone", "calibrated": True}

    def test_property_getters_default_values(self) -> None:
        """Test property getter methods return correct default values."""
        metadata: ChannelMetadata = ChannelMetadata()

        assert metadata.label == ""
        assert metadata.unit == ""
        assert metadata.ref == 1.0
        assert metadata.extra == {}

    def test_property_getters_after_modification(self) -> None:
        """Test property getters reflect modifications."""
        metadata: ChannelMetadata = ChannelMetadata()

        # Modify values and check properties
        metadata.label = "modified_label"
        metadata.unit = "Hz"
        metadata.ref = 0.75
        metadata.extra["new_key"] = "new_value"

        assert metadata.label == "modified_label"
        assert metadata.unit == "Hz"
        assert metadata.ref == 0.75
        assert metadata.extra["new_key"] == "new_value"
