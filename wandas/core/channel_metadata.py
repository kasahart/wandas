from __future__ import annotations

import copy
import numbers
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast, overload

from ._channel_schema import (
    _CHANNEL_CALIBRATION_FACTOR_KEY,
    _CHANNEL_EXTRA_ATTR,
    _CHANNEL_LABEL_KEY,
    _CHANNEL_REF_KEY,
    _CHANNEL_UNIT_KEY,
)
from .metadata import (
    ChannelCalibration,
    ChannelMetadata,
    _normalize_channel_label,
)

if TYPE_CHECKING:
    from .base_frame import BaseFrame


class ChannelMetadataView(ChannelMetadata):
    """Read-only xarray-backed view for one channel's metadata."""

    def __init__(self, frame: BaseFrame[Any], index: int) -> None:
        object.__setattr__(self, "_frame", frame)
        object.__setattr__(self, "_index", index)

    def __getattribute__(self, name: str) -> Any:
        if name in {"id", "label", "calibration", "unit", "ref", "extra"}:
            try:
                frame = object.__getattribute__(self, "_frame")
                index = object.__getattribute__(self, "_index")
            except AttributeError:
                return super().__getattribute__(name)
            if name == "id":
                return frame._channel_id_at(index)
            if name == "label":
                return _normalize_channel_label(frame._get_channel_coord_value(_CHANNEL_LABEL_KEY, index))
            if name == "calibration":
                return ChannelCalibration(
                    factor=frame._get_channel_coord_value(_CHANNEL_CALIBRATION_FACTOR_KEY, index),
                    unit=frame._get_channel_coord_value(_CHANNEL_UNIT_KEY, index),
                    ref=frame._get_channel_coord_value(_CHANNEL_REF_KEY, index),
                )
            if name == "unit":
                return self.calibration.unit
            if name == "ref":
                return self.calibration.ref
            channel_extra = frame._xr.attrs.get(_CHANNEL_EXTRA_ATTR, {})
            channel_id = frame._channel_id_at(index)
            existing = channel_extra.get(channel_id, {})
            if not isinstance(existing, dict):
                return {}
            return copy.deepcopy(existing)
        return super().__getattribute__(name)

    def __getattr__(self, name: str) -> Any:
        if name in {"id", "label", "calibration", "unit", "ref", "extra"}:
            try:
                object.__getattribute__(self, "_frame")
            except AttributeError as exc:
                raise AttributeError(name) from exc
            return self.__getattribute__(name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"label", "calibration", "unit", "ref", "extra"}:
            try:
                object.__getattribute__(self, "_frame")
            except AttributeError:
                super().__setattr__(name, value)
                return
        if name in {"label", "calibration", "unit", "ref", "extra"}:
            raise AttributeError(f"Channel metadata view field {name!r} is read-only")
        super().__setattr__(name, value)

    def __getitem__(self, key: str) -> Any:
        if key in {"id", "label", "calibration", "unit", "ref", "extra"}:
            return getattr(self, key)
        return self.extra.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("Channel metadata views are read-only")

    def matches_query(self, query: dict[str, Any]) -> bool:
        return super().matches_query(query)

    def model_copy(self, *, deep: bool = False, **_: Any) -> ChannelMetadata:
        metadata = self.to_metadata()
        return copy.deepcopy(metadata) if deep else copy.copy(metadata)

    def to_metadata(self) -> ChannelMetadata:
        return ChannelMetadata(
            label=self.label,
            calibration=self.calibration,
            extra=self._frame._channel_extra_at(self._index),
        )


class ChannelMetadataIndexer(Sequence[ChannelMetadataView]):
    """Sequence-like access to xarray-backed channel metadata views."""

    def __init__(self, frame: BaseFrame[Any]) -> None:
        self._frame = frame

    def __len__(self) -> int:
        return self._frame.n_channels

    def __iter__(self) -> Iterator[ChannelMetadataView]:
        for index in range(len(self)):
            yield ChannelMetadataView(self._frame, index)

    @overload
    def __getitem__(self, key: int) -> ChannelMetadataView: ...

    @overload
    def __getitem__(self, key: slice) -> list[ChannelMetadataView]: ...

    @overload
    def __getitem__(self, key: str) -> ChannelMetadataView: ...

    def __getitem__(self, key: int | slice | str) -> ChannelMetadataView | list[ChannelMetadataView]:
        if isinstance(key, numbers.Integral):
            index = int(key)
            index = index + len(self) if index < 0 else index
            if not (0 <= index < len(self)):
                raise IndexError(f"channel index {key} out of range")
            return ChannelMetadataView(self._frame, index)
        if isinstance(key, slice):
            return [ChannelMetadataView(self._frame, i) for i in range(len(self))[key]]
        if isinstance(key, str):
            labels = self._frame.labels
            if key in labels:
                return ChannelMetadataView(self._frame, labels.index(key))
            raise KeyError(f"Channel '{key}' not found.")
        raise TypeError(f"Invalid channel metadata key type: {type(key).__name__}")

    def by_id(self, channel_id: str) -> ChannelMetadataView:
        """Return one channel through the explicit opaque stable-ID path."""
        if not isinstance(channel_id, str):
            raise TypeError("Channel id must be a string")
        ids = self._frame._channel_ids
        if channel_id not in ids:
            raise KeyError(f"Channel id {channel_id!r} not found.")
        return ChannelMetadataView(self._frame, ids.index(channel_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or isinstance(other, (str, bytes)):
            return False
        snapshots: list[ChannelMetadata] = []
        for item in other:
            to_metadata = getattr(item, "to_metadata", None)
            if callable(to_metadata):
                snapshots.append(cast(ChannelMetadata, to_metadata()))
            elif isinstance(item, ChannelMetadata):
                snapshots.append(item)
            else:
                return False
        return self.to_list() == snapshots

    def __repr__(self) -> str:
        return repr(self.to_list())

    def __add__(self, other: list[ChannelMetadata]) -> list[ChannelMetadata]:
        return self.to_list() + other

    def __radd__(self, other: list[ChannelMetadata]) -> list[ChannelMetadata]:
        return other + self.to_list()

    def to_list(self) -> list[ChannelMetadata]:
        return [view.to_metadata() for view in self]
