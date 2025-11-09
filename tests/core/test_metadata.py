import json
from typing import Any

from wandas.core.metadata import ChannelMetadata
from wandas.utils.util import unit_to_ref

# filepath: wandas/core/test_channel_metadata.py


class TestChannelMetadata:
    def test_init_default_values(self) -> None:
        """Test initialization with default values"""
        metadata: ChannelMetadata = ChannelMetadata()
        assert metadata.label == ""
        assert metadata.unit == ""
        assert metadata.extra == {}

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values"""
        metadata: ChannelMetadata = ChannelMetadata(
            label="test_label",
            unit="Hz",
            extra={"source": "microphone", "calibrated": True},
        )
        assert metadata.label == "test_label"
        assert metadata.unit == "Hz"
        assert metadata.extra == {"source": "microphone", "calibrated": True}

    def test_getitem_main_fields(self) -> None:
        """Test dictionary-like access for main fields"""
        metadata: ChannelMetadata = ChannelMetadata(
            label="test_label", unit="Hz", ref=0.5
        )
        assert metadata["label"] == "test_label"
        assert metadata["unit"] == "Hz"
        assert metadata["ref"] == 0.5

    def test_getitem_extra_field(self) -> None:
        """Test dictionary-like access for extra fields"""
        metadata: ChannelMetadata = ChannelMetadata(
            extra={"source": "microphone", "calibrated": True}
        )
        assert metadata["source"] == "microphone"
        assert metadata["calibrated"] is True
        # Non-existent key should return None
        assert metadata["nonexistent"] is None

    def test_setitem_main_fields(self) -> None:
        """Test dictionary-like assignment for main fields"""
        metadata: ChannelMetadata = ChannelMetadata()
        metadata["label"] = "new_label"
        metadata["unit"] = "dB"
        metadata["ref"] = 0.75
        assert metadata.label == "new_label"
        assert metadata.unit == "dB"
        assert metadata.ref == 0.75

    def test_setitem_extra_fields(self) -> None:
        """Test dictionary-like assignment for extra fields"""
        metadata: ChannelMetadata = ChannelMetadata()
        metadata["source"] = "microphone"
        metadata["calibrated"] = True
        assert metadata.extra == {"source": "microphone", "calibrated": True}

    def test_ref_auto_set_when_unit_specified(self) -> None:
        """Test that ref is automatically set when unit is specified"""
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

        # Case 7: Test with other units
        other_units = ["V", "m/s", "g"]
        for unit in other_units:
            expected_ref = unit_to_ref(unit)  # Should be 1.0 for these units
            metadata5: ChannelMetadata = ChannelMetadata(unit=unit)
            assert metadata5.unit == unit
            assert metadata5.ref == expected_ref

    def test_to_json(self) -> None:
        """Test serialization to JSON"""
        metadata: ChannelMetadata = ChannelMetadata(
            label="test_label",
            unit="Hz",
            extra={"source": "microphone", "calibrated": True},
        )
        json_data: str = metadata.to_json()
        # Validate it's proper JSON
        parsed: dict[str, Any] = json.loads(json_data)
        assert parsed["label"] == "test_label"
        assert parsed["unit"] == "Hz"
        assert parsed["extra"]["source"] == "microphone"
        assert parsed["extra"]["calibrated"] is True

    def test_from_json(self) -> None:
        """Test deserialization from JSON"""
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

    def test_copy(self) -> None:
        """Test deep copying of metadata"""
        metadata: ChannelMetadata = ChannelMetadata(
            label="test_label",
            unit="Hz",
            extra={"source": "microphone", "calibrated": True},
        )
        copy_mata: ChannelMetadata = metadata.model_copy(deep=True)

        # Verify all fields are equal
        assert copy_mata.label == metadata.label
        assert copy_mata.unit == metadata.unit
        assert copy_mata.extra == metadata.extra

        # Verify it's a deep copy by modifying the original
        metadata.label = "modified_label"
        metadata.extra["new_key"] = "new_value"

        # The copy should remain unchanged
        assert copy_mata.label == "test_label"
        assert "new_key" not in copy_mata.extra

    def test_unicode_and_special_chars(self) -> None:
        """Test handling of Unicode and special characters"""
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

    def test_nested_extra_data(self) -> None:
        """Test handling of nested structures in extra field"""
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

    def test_getitem_ref_field(self) -> None:
        """Test dictionary-like access for ref field"""
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

    def test_setitem_ref_field(self) -> None:
        """Test dictionary-like assignment for ref field"""
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

    def test_property_methods(self) -> None:
        """Test property getter methods"""
        metadata: ChannelMetadata = ChannelMetadata(
            label="test_label",
            unit="Pa",
            ref=0.5,
            extra={"source": "microphone", "calibrated": True},
        )

        # Test property getters
        assert metadata.label_value == "test_label"
        assert metadata.unit_value == "Pa"
        assert metadata.ref_value == 0.5
        assert metadata.extra_data == {"source": "microphone", "calibrated": True}

    def test_property_methods_default_values(self) -> None:
        """Test property getter methods with default values"""
        metadata: ChannelMetadata = ChannelMetadata()

        assert metadata.label_value == ""
        assert metadata.unit_value == ""
        assert metadata.ref_value == 1.0
        assert metadata.extra_data == {}

    def test_property_methods_after_modification(self) -> None:
        """Test property getter methods after modifying values"""
        metadata: ChannelMetadata = ChannelMetadata()

        # Modify values and check properties
        metadata.label = "modified_label"
        metadata.unit = "Hz"
        metadata.ref = 0.75
        metadata.extra["new_key"] = "new_value"

        assert metadata.label_value == "modified_label"
        assert metadata.unit_value == "Hz"
        assert metadata.ref_value == 0.75
        assert metadata.extra_data["new_key"] == "new_value"
