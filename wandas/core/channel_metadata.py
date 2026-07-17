from __future__ import annotations

import copy
import numbers
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast, overload

from .metadata import ChannelCalibration, ChannelMetadata

if TYPE_CHECKING:
    from .base_frame import BaseFrame


class ChannelMetadataView(ChannelMetadata):
    """Mutable xarray-backed view for one channel's metadata."""

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
                return str(frame._get_channel_coord_value("channel_label", index))
            if name == "calibration":
                return ChannelCalibration(
                    factor=frame._get_channel_coord_value("channel_calibration_factor", index),
                    unit=str(frame._get_channel_coord_value("channel_unit", index)),
                    ref=frame._get_channel_coord_value("channel_ref", index),
                    sample_scale=frame._get_channel_coord_value("channel_calibration_sample_scale", index),
                )
            if name == "unit":
                return self.calibration.unit
            if name == "ref":
                return self.calibration.ref
            channel_extra = frame._xr.attrs.setdefault("channel_extra", {})
            channel_id = frame._channel_id_at(index)
            existing = channel_extra.setdefault(channel_id, {})
            if not isinstance(existing, dict):
                existing = {}
                channel_extra[channel_id] = existing
            return existing
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
        if name == "label":
            if not isinstance(value, str):
                raise TypeError("ChannelMetadata label must be a string")
            self._frame._set_channel_coord_value("channel_label", self._index, value)
            return
        if name == "calibration":
            raise AttributeError(
                "Channel calibration cannot be replaced through a metadata view; use frame.with_calibration(...)"
            )
        if name == "unit":
            if not isinstance(value, str):
                raise TypeError("ChannelMetadata unit must be a string")
            current = self.calibration
            replacement = (
                ChannelCalibration(
                    factor=current.factor,
                    unit=value,
                    sample_scale=current.sample_scale,
                )
                if value
                else ChannelCalibration(
                    factor=current.factor,
                    unit="",
                    ref=current.ref,
                    sample_scale=current.sample_scale,
                )
            )
            self._frame._set_channel_calibration(self._index, replacement)
            return
        if name == "ref":
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError("ChannelMetadata ref must be a number")
            current = self.calibration
            self._frame._set_channel_calibration(
                self._index,
                ChannelCalibration(
                    factor=current.factor,
                    unit=current.unit,
                    ref=value,
                    sample_scale=current.sample_scale,
                ),
            )
            return
        if name == "extra":
            if not isinstance(value, dict):
                raise TypeError("channel extra must be a dictionary")
            self._frame._xr.attrs.setdefault("channel_extra", {})[self.id] = value
            return
        super().__setattr__(name, value)

    def __getitem__(self, key: str) -> Any:
        if key in {"id", "label", "calibration", "unit", "ref", "extra"}:
            return getattr(self, key)
        return self.extra.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in {"label", "calibration", "unit", "ref", "extra"}:
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def matches_query(self, query: dict[str, Any]) -> bool:
        return self.to_metadata().matches_query(query)

    def model_copy(self, *, deep: bool = False, **_: Any) -> ChannelMetadata:
        metadata = self.to_metadata()
        return copy.deepcopy(metadata) if deep else copy.copy(metadata)

    def to_metadata(self) -> ChannelMetadata:
        return ChannelMetadata(
            label=self.label,
            calibration=self.calibration,
            extra=copy.deepcopy(self.extra),
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
            ids = self._frame._channel_ids
            if key in ids:
                return ChannelMetadataView(self._frame, ids.index(key))
            labels = self._frame.labels
            if key in labels:
                return ChannelMetadataView(self._frame, labels.index(key))
            raise KeyError(f"Channel '{key}' not found.")
        raise TypeError(f"Invalid channel metadata key type: {type(key).__name__}")

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
