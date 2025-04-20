import json
from typing import Any

from wandas.core.lazy.channel_metadata import ChannelMetadata

# filepath: wandas/core/lazy/test_channel_metadata.py


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
        metadata: ChannelMetadata = ChannelMetadata(label="test_label", unit="Hz")
        assert metadata["label"] == "test_label"
        assert metadata["unit"] == "Hz"

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
        assert metadata.label == "new_label"
        assert metadata.unit == "dB"

    def test_setitem_extra_fields(self) -> None:
        """Test dictionary-like assignment for extra fields"""
        metadata: ChannelMetadata = ChannelMetadata()
        metadata["source"] = "microphone"
        metadata["calibrated"] = True
        assert metadata.extra == {"source": "microphone", "calibrated": True}

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
        copy_mata: ChannelMetadata = metadata.copy(deep=True)

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
