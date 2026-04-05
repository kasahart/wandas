import json
from typing import Any

from wandas.core.metadata import ChannelMetadata, FrameMetadata
from wandas.utils.util import unit_to_ref

# filepath: wandas/core/test_channel_metadata.py


class TestChannelMetadata:
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
        assert parsed["unit"] == "Hz"
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

    def test_copy_deep_independent_from_original(self) -> None:
        """Test deep copy is independent from original."""
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

        # Test direct field access
        assert metadata.label == "test_label"
        assert metadata.unit == "Pa"
        assert metadata.ref == 0.5
        assert metadata.extra == {"source": "microphone", "calibrated": True}

    def test_property_methods_default_values(self) -> None:
        """Test property getter methods with default values"""
        metadata: ChannelMetadata = ChannelMetadata()

        assert metadata.label == ""
        assert metadata.unit == ""
        assert metadata.ref == 1.0
        assert metadata.extra == {}

    def test_property_methods_after_modification(self) -> None:
        """Test property getter methods after modifying values"""
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


class TestFrameMetadata:
    """Tests for FrameMetadata."""

    def test_init_default(self) -> None:
        """Empty FrameMetadata behaves like an empty dict."""
        meta = FrameMetadata()
        assert meta.source_file is None
        assert dict(meta) == {}

    def test_init_with_source_file(self) -> None:
        meta = FrameMetadata(source_file="audio.wav")
        assert meta.source_file == "audio.wav"
        assert dict(meta) == {}

    def test_init_with_dict_content(self) -> None:
        meta = FrameMetadata({"gain": 0.5, "device": "mic"}, source_file="test.wav")
        assert meta.source_file == "test.wav"
        assert meta["gain"] == 0.5
        assert meta["device"] == "mic"

    def test_dict_operations(self) -> None:
        meta = FrameMetadata({"a": 1})
        meta["b"] = 2
        assert "a" in meta
        assert "b" in meta
        assert meta.get("a") == 1
        assert meta.get("missing") is None
        assert list(meta.items()) == [("a", 1), ("b", 2)]

    def test_equality_with_dict(self) -> None:
        meta = FrameMetadata({"x": 10}, source_file="file.wav")
        assert meta == {"x": 10}

    def test_copy_preserves_source_file(self) -> None:
        meta = FrameMetadata({"key": "val"}, source_file="orig.wav")
        copied = meta.copy()
        assert isinstance(copied, FrameMetadata)
        assert copied.source_file == "orig.wav"
        assert copied["key"] == "val"
        # Modifying copy does not affect original
        copied["key"] = "changed"
        assert meta["key"] == "val"

    def test_deepcopy_preserves_source_file(self) -> None:
        import copy

        meta = FrameMetadata({"nested": {"n": 1}}, source_file="deep.wav")
        cloned = copy.deepcopy(meta)
        assert isinstance(cloned, FrameMetadata)
        assert cloned.source_file == "deep.wav"
        assert cloned["nested"] == {"n": 1}
        # Modifying nested content in clone does not affect original
        cloned["nested"]["n"] = 99
        assert meta["nested"]["n"] == 1

    def test_json_serializable(self) -> None:
        meta = FrameMetadata({"rate": 44100}, source_file="audio.wav")
        # json.dumps should only serialize the dict portion
        data = json.loads(json.dumps(dict(meta)))
        assert data == {"rate": 44100}

    def test_repr(self) -> None:
        meta = FrameMetadata({"k": 1}, source_file="f.wav")
        r = repr(meta)
        assert "FrameMetadata" in r
        assert "f.wav" in r

    def test_bool_empty(self) -> None:
        assert not FrameMetadata()
        assert FrameMetadata({"a": 1})

    def test_unpack_operator(self) -> None:
        meta = FrameMetadata({"a": 1, "b": 2}, source_file="s.wav")
        merged = {**meta, "c": 3}
        assert merged == {"a": 1, "b": 2, "c": 3}
