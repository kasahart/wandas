"""
Tests for ChannelCollectionMixin
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import dask.array as da
import numpy as np
import pytest

from wandas.frames.mixins.channel_collection_mixin import ChannelCollectionMixin
from wandas.utils.types import NDArrayReal


class ConcreteChannelCollection(ChannelCollectionMixin):
    """Concrete implementation of ChannelCollectionMixin for testing"""

    def __init__(self) -> None:
        self.add_channel_called = False
        self.remove_channel_called = False
        self.add_channel_args: dict[str, Any] = {}
        self.remove_channel_args: dict[str, Any] = {}

    def add_channel(
        self,
        data: np.ndarray[Any, Any] | da.Array | ConcreteChannelCollection,
        label: str | None = None,
        align: Literal["strict", "pad", "truncate"] = "strict",
        suffix_on_dup: str | None = None,
        source_time_offset: float | Sequence[float] | NDArrayReal | None = None,
        **kwargs: Any,
    ) -> ConcreteChannelCollection:
        """Implementation of abstract method for testing"""
        self.add_channel_called = True
        self.add_channel_args = {
            "data": data,
            "label": label,
            "align": align,
            "suffix_on_dup": suffix_on_dup,
            "source_time_offset": source_time_offset,
            "kwargs": kwargs,
        }
        return ConcreteChannelCollection()

    def remove_channel(self, key: int | str) -> ConcreteChannelCollection:
        """Implementation of abstract method for testing"""
        self.remove_channel_called = True
        self.remove_channel_args = {"key": key}
        return ConcreteChannelCollection()


class TestChannelCollectionMixin:
    """Tests for ChannelCollectionMixin"""

    def test_add_channel_method(self) -> None:
        """Test the add_channel method"""
        collection = ConcreteChannelCollection()
        data = np.ones(10)
        result = collection.add_channel(data, label="test_channel", align="strict")

        assert collection.add_channel_called
        assert collection.add_channel_args["data"] is data
        assert collection.add_channel_args["label"] == "test_channel"
        assert collection.add_channel_args["align"] == "strict"
        assert collection.add_channel_args["suffix_on_dup"] is None
        assert collection.add_channel_args["source_time_offset"] is None
        assert isinstance(result, ConcreteChannelCollection)
        assert result is not collection

    def test_add_channel_with_source_time_offset(self) -> None:
        """Test the add_channel method with source_time_offset."""
        collection = ConcreteChannelCollection()
        data = np.ones(10)
        collection.add_channel(data, label="test_channel", source_time_offset=1.25)

        assert collection.add_channel_called
        assert collection.add_channel_args["source_time_offset"] == 1.25

    def test_add_channel_with_kwargs(self) -> None:
        """Test the add_channel method with additional kwargs"""
        collection = ConcreteChannelCollection()
        data = np.ones(10)
        collection.add_channel(data, label="test_channel", custom_kwarg="value")

        assert collection.add_channel_called
        assert collection.add_channel_args["kwargs"]["custom_kwarg"] == "value"

    def test_remove_channel_method(self) -> None:
        """Test the remove_channel method"""
        collection = ConcreteChannelCollection()
        result = collection.remove_channel("test_channel")

        assert collection.remove_channel_called
        assert collection.remove_channel_args["key"] == "test_channel"
        assert isinstance(result, ConcreteChannelCollection)
        assert result is not collection

    def test_remove_channel_with_index(self) -> None:
        """Test the remove_channel method with an index"""
        collection = ConcreteChannelCollection()
        _ = collection.remove_channel(0)

        assert collection.remove_channel_called
        assert collection.remove_channel_args["key"] == 0

    def test_align_parameter_options(self) -> None:
        """Test all valid options for the align parameter"""
        collection = ConcreteChannelCollection()
        data = np.ones(10)

        # Test each valid align option
        for align_option in ["strict", "pad", "truncate"]:
            collection.add_channel(data, label="test_channel", align=align_option)  # ty: ignore[invalid-argument-type]
            assert collection.add_channel_args["align"] == align_option

    def test_add_channel_with_metadata(self) -> None:
        """Test the add_channel method with extra metadata"""
        collection = ConcreteChannelCollection()
        data = np.ones(10)
        metadata = {"source": "test", "calibration": 0.5}

        collection.add_channel(data, label="test_channel", metadata=metadata)
        assert collection.add_channel_args["kwargs"]["metadata"] == metadata

    def test_base_mixin_raises_not_implemented(self) -> None:
        """Test that base mixin methods raise NotImplementedError if not implemented"""

        # Create a class that inherits but doesn't implement the methods
        class IncompleteImplementation(ChannelCollectionMixin):
            pass

        incomplete = IncompleteImplementation()

        # Test add_channel method
        with pytest.raises(NotImplementedError):
            incomplete.add_channel(np.ones(10), label="test")

        # Test remove_channel method
        with pytest.raises(NotImplementedError):
            incomplete.remove_channel(0)

    def test_add_channel_with_suffix_on_dup(self) -> None:
        """Test the add_channel method with suffix_on_dup parameter"""
        collection = ConcreteChannelCollection()
        data = np.ones(10)

        # Test with a suffix value
        collection.add_channel(data, label="test_channel", suffix_on_dup="_duplicate")

        assert collection.add_channel_called
        assert collection.add_channel_args["suffix_on_dup"] == "_duplicate"

        # Test with None value
        collection.add_channel(data, label="test_channel", suffix_on_dup=None)
        assert collection.add_channel_args["suffix_on_dup"] is None

    def test_add_channel_with_none_label(self) -> None:
        """Test the add_channel method with None label"""
        collection = ConcreteChannelCollection()
        data = np.ones(10)

        collection.add_channel(data, label=None)
        assert collection.add_channel_args["label"] is None
