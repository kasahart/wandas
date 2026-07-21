"""Ownership and copy-count contracts for xarray-backed channel metadata."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import dask.array as da
import pytest
from dask.array.core import Array as DaArray

from wandas.core._channel_schema import (
    _CHANNEL_CALIBRATION_FACTOR_KEY,
    _CHANNEL_LABEL_KEY,
    _CHANNEL_REF_KEY,
    _CHANNEL_UNIT_KEY,
)
from wandas.core.base_frame import BaseFrame
from wandas.core.channel_metadata import ChannelMetadataIndexer
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.processing.semantic import (
    InputBinding,
    LineageNode,
    SemanticOperation,
    freeze_params,
    semantic_lineage,
)


class _DeepcopyProbe:
    """Count defensive copies without introducing another mutable payload."""

    copies: ClassVar[int] = 0

    def __deepcopy__(self, memo: dict[int, object]) -> "_DeepcopyProbe":
        type(self).copies += 1
        return self


def _frame(*, channel_count: int = 1) -> ChannelFrame:
    frame = ChannelFrame(
        data=da.ones((channel_count, 8), chunks=(1, -1)),
        sampling_rate=8_000,
        metadata={"nested": _DeepcopyProbe()},
        channel_metadata=[
            ChannelMetadata(label=f"sensor-{index:03d}", extra={"nested": _DeepcopyProbe()})
            for index in range(channel_count)
        ],
        channel_ids=[f"sensor-id-{index:03d}" for index in range(channel_count)],
    )
    _DeepcopyProbe.copies = 0
    return frame


@pytest.mark.parametrize(
    ("operation", "expected_copies"),
    [
        (lambda frame: frame.channels.to_list(), 1),
        (lambda frame: frame._create_new_instance(data=frame._data), 2),
        (lambda frame: frame.with_calibration([2.0]), 2),
        (lambda frame: frame.abs(), 2),
        (
            lambda frame: frame.channels[0].matches_query({"nested": frame.channels[0].extra["nested"]}),
            0,
        ),
    ],
)
def test_channel_metadata_copy_count_matches_ownership_boundaries(
    operation: Callable[[ChannelFrame], object],
    expected_copies: int,
) -> None:
    frame = _frame()

    operation(frame)

    # One copy each is allowed for Frame metadata and channel ``extra`` when a
    # new Frame takes ownership. Public snapshots copy only channel ``extra``.
    assert _DeepcopyProbe.copies == expected_copies


def test_create_new_instance_does_not_evaluate_explicit_metadata_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    explicit_channels = [ChannelMetadata(label="explicit", extra={"owned": True})]

    def fail_to_list(self: ChannelMetadataIndexer) -> list[ChannelMetadata]:
        raise AssertionError("channel metadata default was evaluated")

    monkeypatch.setattr(ChannelMetadataIndexer, "to_list", fail_to_list)
    result = frame._create_new_instance(
        data=frame._data,
        metadata={"explicit": True},
        channel_metadata=explicit_channels,
    )

    assert result.metadata == {"explicit": True}
    assert result.channels[0].label == "explicit"
    assert _DeepcopyProbe.copies == 0


class _NormalizeCountingFrame(ChannelFrame):
    normalize_calls: int

    def _normalize_channel_metadata_for_count(
        self,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]] | None,
        channel_count: int,
    ) -> list[ChannelMetadata]:
        self.normalize_calls = getattr(self, "normalize_calls", 0) + 1
        return super()._normalize_channel_metadata_for_count(channel_metadata, channel_count)


class _ContainsCountingIds(list[str]):
    def __init__(self, values: Sequence[str], owner: "_MembershipCountingFrame") -> None:
        super().__init__(values)
        self._owner = owner

    def __contains__(self, value: object) -> bool:
        self._owner.contains_calls += 1
        return super().__contains__(value)


class _MembershipCountingFrame(ChannelFrame):
    contains_calls: int = 0

    @property
    def _channel_ids(self) -> list[str]:
        return _ContainsCountingIds(super()._channel_ids, self)


def test_constructor_normalizes_channel_metadata_once() -> None:
    frame = _NormalizeCountingFrame(
        data=da.ones((1, 8), chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=[ChannelMetadata(label="sensor")],
    )

    assert frame.normalize_calls == 1


def test_replace_data_writes_its_exclusive_snapshot_without_renormalizing() -> None:
    caller_extra = {"nested": {"gain": 1}}
    frame = _NormalizeCountingFrame(
        data=da.ones((1, 8), chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=[ChannelMetadata(label="sensor", extra=caller_extra)],
    )

    frame._replace_data(da.zeros((1, 8), chunks=(1, -1)))
    caller_extra["nested"]["gain"] = 2
    detached = frame.channels[0].to_metadata()
    detached.extra["nested"]["gain"] = 3

    assert frame.normalize_calls == 1
    assert frame.channels[0].extra == {"nested": {"gain": 1}}


def test_normalized_writer_rejects_wrong_channel_count() -> None:
    frame = _frame()

    with pytest.raises(ValueError, match="Normalized channel metadata length"):
        frame._write_normalized_channel_metadata([])


def test_normalization_does_not_trust_caller_snapshot_aliases() -> None:
    class AliasingSnapshot(ChannelMetadata):
        def to_metadata(self) -> ChannelMetadata:
            return self

    source = AliasingSnapshot(label="sensor", extra={"nested": {"gain": 1}})
    frame = ChannelFrame(
        data=da.ones((1, 8), chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=[source],
    )

    source.extra["nested"]["gain"] = 2

    assert frame.channels[0].extra == {"nested": {"gain": 1}}


def test_channel_extra_setter_detaches_caller_owned_nested_values() -> None:
    frame = _frame()
    caller_extra = {"nested": {"gain": 1}}

    frame.channels[0].extra = caller_extra
    caller_extra["nested"]["gain"] = 2

    assert frame.channels[0].extra == {"nested": {"gain": 1}}


@pytest.mark.parametrize(
    "channel_metadata",
    [
        [ChannelMetadata(label="sensor", extra={"nested": {"gain": 1}})],
        [{"label": "sensor", "extra": {"nested": {"gain": 1}}}],
    ],
)
def test_frame_constructor_detaches_caller_owned_channel_metadata(
    channel_metadata: list[ChannelMetadata | dict[str, Any]],
) -> None:
    frame = ChannelFrame(
        data=da.ones((1, 8), chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=channel_metadata,
    )

    source_extra = (
        channel_metadata[0].extra if isinstance(channel_metadata[0], ChannelMetadata) else channel_metadata[0]["extra"]
    )
    source_extra["nested"]["gain"] = 2

    assert frame.channels[0].extra == {"nested": {"gain": 1}}


def test_derived_frame_does_not_share_nested_extra_with_source() -> None:
    frame = ChannelFrame(
        data=da.ones((1, 8), chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=[ChannelMetadata(label="sensor", extra={"nested": {"gain": 1}})],
    )

    result = frame.with_calibration([2.0])
    result.channels[0].extra["nested"]["gain"] = 2

    assert frame.channels[0].extra == {"nested": {"gain": 1}}


def test_private_calibration_replacements_preserve_domain_rules() -> None:
    calibration = ChannelCalibration(factor=2.0, unit="V", ref=0.25)

    assert calibration._with_unit("Pa") == ChannelCalibration(factor=2.0, unit="Pa")
    assert calibration._with_unit("") == ChannelCalibration(factor=2.0, unit="", ref=0.25)
    assert calibration._with_ref(0.5) == ChannelCalibration(factor=2.0, unit="V", ref=0.5)


def test_with_calibration_resolves_once_and_reads_source_coordinates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame(channel_count=100)
    getter_counts: Counter[str] = Counter()
    resolver_calls = 0
    original_getter = BaseFrame._get_channel_coord_value
    original_resolver = ChannelFrame._resolve_calibration_updates

    def counted_getter(self: BaseFrame[Any], name: str, index: int) -> Any:
        if self is frame:
            getter_counts[name] += 1
        return original_getter(self, name, index)

    def counted_resolver(
        self: ChannelFrame,
        values: Sequence[float | ChannelCalibration] | Mapping[str | int, float | ChannelCalibration] | Any,
    ) -> list[tuple[int, ChannelCalibration]]:
        nonlocal resolver_calls
        if self is frame:
            resolver_calls += 1
        return original_resolver(self, values)

    monkeypatch.setattr(BaseFrame, "_get_channel_coord_value", counted_getter)
    monkeypatch.setattr(ChannelFrame, "_resolve_calibration_updates", counted_resolver)

    result = frame.with_calibration([2.0] * 100)

    assert resolver_calls == 1
    assert getter_counts == {
        _CHANNEL_CALIBRATION_FACTOR_KEY: 100,
        _CHANNEL_LABEL_KEY: 100,
        _CHANNEL_REF_KEY: 100,
        _CHANNEL_UNIT_KEY: 100,
    }
    assert isinstance(result._data, DaArray)
    assert result.operation_history[-1]["operation"] == "wandas.channel.with_calibration"
    assert [channel.calibration.factor for channel in frame.channels] == [1.0] * 100


def test_with_calibration_does_not_linearly_scan_ids_for_each_update() -> None:
    frame = _MembershipCountingFrame(
        data=da.ones((100, 8), chunks=(1, -1)),
        sampling_rate=8_000,
        channel_ids=[f"sensor-id-{index:03d}" for index in range(100)],
    )

    result = frame.with_calibration([2.0] * 100)

    assert frame.contains_calls == 0
    assert [channel.calibration.factor for channel in result.channels] == [2.0] * 100


def _outer_lineage(frame: ChannelFrame, operation_id: str) -> LineageNode:
    operation = SemanticOperation(
        operation_id,
        1,
        (InputBinding("frame", "frame"),),
        freeze_params({}),
    )
    return LineageNode(operation, (frame.lineage,))


def test_with_calibration_falls_back_to_runtime_values_inside_other_semantic_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    outer = _outer_lineage(frame, "tests.outer")
    resolver_calls = 0
    original_resolver = ChannelFrame._resolve_calibration_updates

    def counted_resolver(
        self: ChannelFrame,
        values: Sequence[float | ChannelCalibration] | Mapping[str | int, float | ChannelCalibration] | Any,
    ) -> list[tuple[int, ChannelCalibration]]:
        nonlocal resolver_calls
        resolver_calls += 1
        return original_resolver(self, values)

    monkeypatch.setattr(ChannelFrame, "_resolve_calibration_updates", counted_resolver)

    with semantic_lineage(outer):
        result = frame.with_calibration([3.0])

    assert resolver_calls == 1
    assert result.lineage is outer
    assert result.channels[0].calibration.factor == 3.0


class _DisplayOperation:
    params: ClassVar[dict[str, object]] = {}

    def __init__(self, display_name: str | None) -> None:
        self._display_name = display_name

    def process(self, data: DaArray) -> DaArray:
        return data

    def get_metadata_updates(self) -> dict[str, object]:
        return {}

    def get_display_name(self) -> str | None:
        return self._display_name


@pytest.mark.parametrize("display_name", [None, "", "alias"])
def test_operation_instance_preserves_relabel_contract_for_every_display_name_class(
    display_name: str | None,
) -> None:
    frame = _frame()
    outer = _outer_lineage(frame, "tests.operation")
    expected = [channel.label for channel in frame._relabel_channels("operation_name", display_name)]

    with semantic_lineage(outer):
        result = frame._apply_operation_instance(
            _DisplayOperation(display_name),
            operation_name="operation_name",
        )

    assert result.labels == expected
    assert result.lineage is outer
    assert result.channels[0].extra == frame.channels[0].extra
    assert result.channels[0].extra is not frame.channels[0].extra
