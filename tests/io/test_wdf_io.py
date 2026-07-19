"""Contract tests for the xarray-backed typed WDF 0.4 boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import wandas as wd
from wandas.core.base_frame import BaseFrame
from wandas.frames.cepstral import CepstralFrame
from wandas.frames.cepstrogram import CepstrogramFrame
from wandas.frames.channel import ChannelFrame
from wandas.frames.noct import NOctFrame
from wandas.frames.roughness import RoughnessFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.io import wdf_frames, wdf_io


def _typed_frames() -> list[BaseFrame[Any]]:
    source = ChannelFrame.from_numpy(
        np.array([[0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]], dtype=np.float32),
        8.0,
        label="source",
        metadata={"recording": "fixture"},
        ch_labels=["mic"],
        ch_units=["Pa"],
    ).normalize()
    return [
        source,
        source.fft(n_fft=8),
        source.stft(n_fft=8, hop_length=2),
        source.cepstrum(n_fft=8),
        CepstrogramFrame(
            da.from_array(np.arange(24, dtype=float).reshape(1, 8, 3), chunks=(1, -1, -1)),
            sampling_rate=8.0,
            n_fft=8,
            hop_length=2,
            win_length=8,
            window="hann",
            channel_metadata=[{"label": "mic", "unit": "Pa"}],
        ),
        NOctFrame(
            da.from_array(np.arange(6, dtype=float).reshape(1, 6), chunks=(1, -1)),
            sampling_rate=48_000.0,
            fmin=100.0,
            fmax=4_000.0,
            n=3,
            G=10,
            fr=1_000,
            channel_metadata=[{"label": "mic", "unit": "Pa"}],
        ),
        RoughnessFrame(
            da.from_array(np.arange(141, dtype=float).reshape(47, 3), chunks=(-1, -1)),
            sampling_rate=10.0,
            bark_axis=np.linspace(0.5, 23.5, 47),
            overlap=0.5,
            channel_metadata=[{"label": "mic", "unit": "asper"}],
        ),
    ]


@pytest.mark.parametrize("frame", _typed_frames(), ids=lambda frame: type(frame).__name__)
def test_wdf_roundtrips_all_exact_builtin_types(frame: BaseFrame[Any], tmp_path: Path) -> None:
    path = tmp_path / f"{type(frame).__name__}.wdf"
    frame.save(path)
    loaded = wd.load(path)

    assert type(loaded) is type(frame)
    assert isinstance(loaded._data, da.Array)
    assert loaded._data.dtype == frame._data.dtype
    assert loaded._data.ndim == frame._data.ndim
    assert loaded._xr.dims == frame._xr.dims
    np.testing.assert_array_equal(loaded.compute(), frame.compute())
    assert loaded.sampling_rate == frame.sampling_rate
    assert loaded.label == frame.label
    assert loaded.metadata == frame.metadata
    assert loaded.channels.to_list() == frame.channels.to_list()
    assert loaded.operation_history == frame.operation_history
    loaded_state = loaded._get_additional_init_kwargs()
    original_state = frame._get_additional_init_kwargs()
    assert loaded_state.keys() == original_state.keys()
    for key, expected in original_state.items():
        actual = loaded_state[key]
        np.testing.assert_array_equal(actual, expected) if isinstance(expected, np.ndarray) else assert_equal(
            actual, expected
        )


def assert_equal(actual: object, expected: object) -> None:
    assert actual == expected


def test_wdf_layout_uses_root_attrs_variables_and_data_dims(tmp_path: Path) -> None:
    frame = ChannelFrame.from_numpy(np.arange(8, dtype=float).reshape(2, 4), 8_000.0)
    path = tmp_path / "layout.wdf"
    frame.save(path)

    with xr.open_dataset(path, engine="h5netcdf", decode_cf=False) as dataset:
        assert dataset.attrs["version"] == "0.4"
        assert set(dataset.attrs) == {
            "version",
            "frame_type",
            "sampling_rate",
            "label",
            "constructor_json",
            "metadata_json",
            "operation_history_json",
        }
        assert set(dataset.data_vars) == {
            "data",
            "channel_label",
            "channel_unit",
            "channel_ref",
            "channel_calibration_factor",
            "source_time_offset",
            "channel_extra_json",
        }
        assert set(dataset.coords) == {"channel"}
        assert dataset["data"].dims == ("channel", "time")


def test_wdf_preserves_raw_data_and_applies_calibration_once(tmp_path: Path) -> None:
    raw = np.array([[1.0, 2.0], [3.0, 4.0]])
    frame = ChannelFrame.from_numpy(raw, 8_000.0).with_calibration(
        [wd.ChannelCalibration(0.02, "Pa"), wd.ChannelCalibration(9.81, "m/s^2", 1.0)]
    )
    path = tmp_path / "calibrated.wdf"
    frame.save(path)

    with xr.open_dataset(path, engine="h5netcdf", decode_cf=False) as dataset:
        np.testing.assert_array_equal(dataset["data"].values, raw)
        np.testing.assert_array_equal(dataset["channel_calibration_factor"].values, [0.02, 9.81])
    loaded = wd.load(path)
    np.testing.assert_array_equal(loaded._data.compute(), raw)
    np.testing.assert_array_equal(loaded.data, frame.data)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SpectralFrame(da.from_array(np.arange(129, dtype=float).reshape(1, -1)), 8_000.0, n_fft=256),
        lambda: ChannelFrame.from_numpy(np.arange(48, dtype=float).reshape(1, -1), 24.0)
        .stft(n_fft=8, hop_length=2)
        .abs(),
    ],
)
def test_wdf_roundtrips_real_analysis_tensors(factory: Callable[[], BaseFrame[Any]], tmp_path: Path) -> None:
    frame = factory()
    path = tmp_path / "real-analysis.wdf"
    frame.save(path)
    loaded = wd.load(path)
    assert type(loaded) is type(frame)
    assert np.issubdtype(loaded._data.dtype, np.floating)
    np.testing.assert_allclose(loaded.compute(), frame.compute())


@pytest.mark.parametrize("reverse", [False, True])
def test_wdf_roundtrips_sliced_dimension_coordinate(reverse: bool, tmp_path: Path) -> None:
    frame = ChannelFrame.from_numpy(np.arange(24, dtype=float).reshape(1, -1), 24.0).cepstrum(n_fft=24)
    frame = frame[:, 9:1:-1] if reverse else frame[:, 2:10]
    path = tmp_path / "coordinate.wdf"
    frame.save(path)
    loaded = cast(CepstralFrame, wd.load(path))
    np.testing.assert_array_equal(loaded.quefrencies, frame.quefrencies)
    np.testing.assert_array_equal(loaded.compute(), frame.compute())


@pytest.mark.parametrize("frame_type", [NOctFrame, RoughnessFrame])
def test_wdf_roundtrips_supported_three_dimensional_domain_frames(
    frame_type: type[NOctFrame] | type[RoughnessFrame], tmp_path: Path
) -> None:
    data = da.from_array(np.arange(6, dtype=float).reshape(1, 2, 3), chunks=(1, -1, -1))
    if frame_type is NOctFrame:
        frame = NOctFrame(data, 48_000.0, fmin=100.0, fmax=4_000.0)
    else:
        roughness_data = da.from_array(np.arange(141, dtype=float).reshape(1, 47, 3), chunks=(1, -1, -1))
        frame = RoughnessFrame(roughness_data, 10.0, bark_axis=np.linspace(0.5, 23.5, 47), overlap=0.5)
    path = tmp_path / "domain-3d.wdf"
    frame.save(path)
    loaded = wd.load(path)
    assert type(loaded) is frame_type
    assert loaded._xr.dims == frame._xr.dims
    np.testing.assert_array_equal(loaded.compute(), frame.compute())


def test_save_does_not_call_frame_data_compute(tmp_path: Path) -> None:
    frame = ChannelFrame.from_numpy(np.arange(8, dtype=float).reshape(1, -1), 8.0)
    with patch.object(frame._data, "compute", side_effect=AssertionError("eager compute")):
        frame.save(tmp_path / "lazy-save.wdf")


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (
            lambda: SpectralFrame(
                da.ones((1, 5)),
                8_000.0,
                n_fft=cast(Any, 8.0),
            ),
            "n_fft",
        ),
        (
            lambda: NOctFrame(
                da.ones((1, 2)),
                8_000.0,
                n=cast(Any, 3.5),
            ),
            "n",
        ),
        (
            lambda: RoughnessFrame(
                da.ones((1, 47, 2)),
                8_000.0,
                bark_axis=np.arange(47, dtype=float),
                overlap=cast(Any, True),
            ),
            "overlap",
        ),
    ],
)
def test_save_rejects_constructor_state_that_load_would_reject(
    factory: Callable[[], BaseFrame[Any]], field: str, tmp_path: Path
) -> None:
    path = tmp_path / "invalid-constructor.wdf"

    with pytest.raises(ValueError, match=rf"Field: {field}"):
        factory().save(path)

    assert not path.exists()


def test_loaded_backend_remains_computable_after_load_returns(tmp_path: Path) -> None:
    expected = np.arange(64, dtype=float).reshape(2, 32)
    path = tmp_path / "lazy-load.wdf"
    ChannelFrame.from_numpy(expected, 8_000.0).save(path)
    loaded = wd.load(path)
    assert isinstance(loaded._data, da.Array)
    np.testing.assert_array_equal(loaded._data.compute(), expected)


def test_suffix_compression_and_overwrite_contract(tmp_path: Path) -> None:
    frame = ChannelFrame.from_numpy(np.ones((1, 8)), 8.0)
    target = tmp_path / "artifact"
    frame.save(target, compress=None)
    path = target.with_suffix(".wdf")
    assert path.exists()
    with pytest.raises(FileExistsError):
        frame.save(target)
    frame.save(target, overwrite=True)


def test_channel_frame_load_rejects_other_typed_frame(tmp_path: Path) -> None:
    path = tmp_path / "spectrum.wdf"
    ChannelFrame.from_numpy(np.arange(8, dtype=float).reshape(1, -1), 8.0).fft(n_fft=8).save(path)
    with pytest.raises(TypeError, match=r"wd\.load"):
        ChannelFrame.load(path)


def _rewrite(path: Path, mutate: Callable[[xr.Dataset], None]) -> None:
    with xr.open_dataset(path, engine="h5netcdf", decode_cf=False, mask_and_scale=False) as opened:
        dataset = opened.load()
    mutate(dataset)
    dataset.to_netcdf(path, engine="h5netcdf", mode="w", invalid_netcdf=True)


@pytest.mark.parametrize("version", [None, "0.1", "0.2", "0.3", "0.5", "99.0"])
def test_wdf_rejects_missing_legacy_and_future_versions(version: str | None, tmp_path: Path) -> None:
    path = tmp_path / "version.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)

    def mutate(dataset: xr.Dataset) -> None:
        if version is None:
            del dataset.attrs["version"]
        else:
            dataset.attrs["version"] = version

    _rewrite(path, mutate)
    with pytest.raises(ValueError, match="Unsupported WDF format version"):
        wd.load(path)


@pytest.mark.parametrize("name", sorted(wdf_io._ROOT_ATTRS))
def test_wdf_rejects_missing_required_attribute(name: str, tmp_path: Path) -> None:
    path = tmp_path / "attribute.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.attrs.pop(name))
    expected = "Unsupported WDF format version" if name == "version" else "Invalid WDF root attribute schema"
    with pytest.raises(ValueError, match=expected):
        wd.load(path)


@pytest.mark.parametrize("name", sorted(wdf_io._DATA_VARIABLES))
def test_wdf_rejects_missing_required_variable(name: str, tmp_path: Path) -> None:
    path = tmp_path / "variable.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.__delitem__(name))
    with pytest.raises(ValueError, match="Invalid WDF data variable schema"):
        wd.load(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("constructor_json", "{invalid"),
        ("metadata_json", "{invalid"),
        ("operation_history_json", "{invalid"),
        ("label", "{invalid"),
        ("metadata_json", '{"bad":NaN}'),
    ],
)
def test_wdf_rejects_invalid_json(field: str, value: str, tmp_path: Path) -> None:
    path = tmp_path / "json.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.attrs.__setitem__(field, value))
    with pytest.raises(ValueError, match="Invalid strict JSON"):
        wd.load(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("constructor_json", "[]", "constructor_json"),
        ("metadata_json", "[]", "metadata_json"),
        ("label", "1", "label"),
        ("operation_history_json", "{}", "operation history"),
        ("operation_history_json", "[1]", "operation history"),
        ("operation_history_json", '[{"operation":"","version":1,"params":{}}]', "operation history"),
    ],
)
def test_wdf_rejects_wrong_json_shapes(field: str, value: str, message: str, tmp_path: Path) -> None:
    path = tmp_path / "json-shape.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.attrs.__setitem__(field, value))
    with pytest.raises(ValueError, match=message):
        wd.load(path)


@pytest.mark.parametrize(("value", "message"), [(True, "got"), (np.nan, "finite")])
def test_wdf_rejects_invalid_sampling_rate(value: object, message: str, tmp_path: Path) -> None:
    path = tmp_path / "sampling-rate.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.attrs.__setitem__("sampling_rate", value))
    with pytest.raises(ValueError, match=message):
        wd.load(path)


def test_wdf_rejects_non_text_json_attribute(tmp_path: Path) -> None:
    path = tmp_path / "json-type.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.attrs.__setitem__("metadata_json", 1))
    with pytest.raises(ValueError, match="expected text"):
        wd.load(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_type", "FutureFrame", "Unsupported WDF frame type"),
        ("constructor_json", '{"extra":1}', "Invalid typed WDF Frame state"),
    ],
)
def test_wdf_rejects_invalid_typed_state(field: str, value: str, message: str, tmp_path: Path) -> None:
    path = tmp_path / "typed.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.attrs.__setitem__(field, value))
    with pytest.raises(ValueError, match=message):
        wd.load(path)


def test_wdf_rejects_non_text_frame_type(tmp_path: Path) -> None:
    path = tmp_path / "frame-type.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.attrs.__setitem__("frame_type", 1))
    with pytest.raises(ValueError, match="frame_type"):
        wd.load(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda dataset: dataset.__setitem__("channel_label", (("channel", "bad"), np.array([["x"]]))),
        lambda dataset: dataset.__setitem__("channel_ref", ("channel", np.array(["bad"]))),
        lambda dataset: dataset.__setitem__("source_time_offset", ("channel", np.array([np.nan]))),
    ],
)
def test_wdf_rejects_invalid_channel_variables(mutate: Callable[[xr.Dataset], None], tmp_path: Path) -> None:
    path = tmp_path / "channels.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, mutate)
    with pytest.raises(ValueError, match="Invalid WDF channel variable"):
        wd.load(path)


def test_wdf_rejects_non_text_channel_values_and_non_object_extra(tmp_path: Path) -> None:
    path = tmp_path / "channel-types.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.__setitem__("channel_label", ("channel", np.array([1]))))
    with pytest.raises(ValueError, match="expected text values"):
        wd.load(path)

    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path, overwrite=True)
    _rewrite(path, lambda dataset: dataset.__setitem__("channel_extra_json", ("channel", np.array(["[]"]))))
    with pytest.raises(ValueError, match="expected JSON objects"):
        wd.load(path)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1], dtype=np.int64),
        np.array([0.0, np.nan]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.3]),
    ],
)
def test_wdf_rejects_invalid_dimension_coordinate(values: np.ndarray[Any, Any], tmp_path: Path) -> None:
    frame = ChannelFrame.from_numpy(np.arange(8, dtype=float).reshape(1, -1), 8.0).cepstrum(n_fft=8)[:, :2]
    path = tmp_path / "coordinate.wdf"
    frame.save(path)

    def mutate(dataset: xr.Dataset) -> None:
        dataset.coords["quefrency"] = values

    _rewrite(path, mutate)
    with pytest.raises(ValueError, match="WDF coordinate|Invalid WDF coordinate schema"):
        wd.load(path)


def test_wdf_rejects_multidimensional_coordinate(tmp_path: Path) -> None:
    path = tmp_path / "coordinate-rank.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.coords.__setitem__("bad", (("bad0", "bad1"), np.ones((1, 2)))))
    with pytest.raises(ValueError, match="Invalid WDF (coordinate|root attribute) schema"):
        wd.load(path)


def test_wdf_rejects_missing_and_unexpected_dimension_coordinates(tmp_path: Path) -> None:
    coordinate_path = tmp_path / "missing-coordinate.wdf"
    ChannelFrame.from_numpy(np.arange(8, dtype=float).reshape(1, -1), 8.0).cepstrum(n_fft=8).save(coordinate_path)
    _rewrite(coordinate_path, lambda dataset: dataset.__delitem__("quefrency"))
    with pytest.raises(ValueError, match="Incomplete WDF Frame coordinates"):
        wd.load(coordinate_path)

    generic_path = tmp_path / "unexpected-coordinate.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(generic_path)
    _rewrite(
        generic_path,
        lambda dataset: dataset.coords.__setitem__("time", ("time", np.arange(dataset.sizes["time"], dtype=float))),
    )
    with pytest.raises(ValueError, match="coordinate schema"):
        wd.load(generic_path)


def test_wdf_rejects_semantic_dimension_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "dims.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.__setitem__("data", dataset["data"].rename({"time": "sample"})))
    with pytest.raises(ValueError, match="semantic dimensions"):
        wd.load(path)


def test_wdf_rejects_data_rank_dtype_and_dims(tmp_path: Path) -> None:
    path = tmp_path / "tensor.wdf"
    ChannelFrame.from_numpy(np.ones((1, 4)), 8.0).save(path)
    _rewrite(path, lambda dataset: dataset.__setitem__("data", (("channel", "time"), np.ones((1, 4), dtype=complex))))
    with pytest.raises(ValueError, match="tensor dtype"):
        wd.load(path)


def test_save_validates_json_before_opening_destination(tmp_path: Path) -> None:
    frame = ChannelFrame.from_numpy(np.ones((1, 4)), 8.0, metadata={"bad": object()})
    path = tmp_path / "invalid.wdf"
    with pytest.raises(ValueError, match="metadata_json"):
        frame.save(path)
    assert not path.exists()


def test_missing_h5netcdf_has_io_extra_hint(tmp_path: Path) -> None:
    frame = ChannelFrame.from_numpy(np.ones((1, 4)), 8.0)
    with patch.object(wdf_io, "require_h5netcdf", side_effect=ImportError('Install it with: pip install "wandas[io]"')):
        with pytest.raises(ImportError, match=r"wandas\[io\]"):
            frame.save(tmp_path / "missing.wdf")


def test_url_is_not_a_wdf_load_source() -> None:
    with pytest.raises(FileNotFoundError):
        wd.load("https://example.com/audio.wdf")


def test_operation_history_remains_display_history_not_recipe_lineage(tmp_path: Path) -> None:
    frame = ChannelFrame.from_numpy(np.array([[1.0, -1.0, 0.5]]), 16_000.0).normalize()
    path = tmp_path / "history.wdf"
    frame.save(path)
    loaded = wd.load(path)
    assert loaded.operation_history == frame.operation_history
    assert loaded.previous is None
    json.loads(cast(str, xr.load_dataset(path, engine="h5netcdf").attrs["operation_history_json"]))


@pytest.mark.parametrize(
    ("helper", "state", "field", "frame_type"),
    [
        (wdf_frames._positive_integer, {"value": 0}, "value", "SpectralFrame"),
        (wdf_frames._nonblank_string, {"value": " "}, "value", "SpectralFrame"),
        (wdf_frames._finite_number, {"value": np.nan}, "value", "NOctFrame"),
    ],
)
def test_codec_scalar_helpers_reject_invalid_constructor_values(
    helper: Callable[..., object], state: dict[str, object], field: str, frame_type: str
) -> None:
    with pytest.raises(ValueError, match="Invalid WDF Frame constructor value"):
        helper(state, field, frame_type)


@pytest.mark.parametrize(
    "state",
    [
        {"fmin": -1.0, "fmax": 4_000.0, "n": 3, "G": 10, "fr": 1_000},
        {"fmin": 100.0, "fmax": 99.0, "n": 3, "G": 10, "fr": 1_000},
    ],
)
def test_noct_codec_rejects_invalid_frequency_bounds(state: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Invalid WDF Frame constructor value"):
        wdf_frames._validated_noct_constructor_state(state)


@pytest.mark.parametrize(
    "state",
    [
        {"bark_axis": [1.0], "overlap": 0.5},
        {"bark_axis": [1.0] * 46 + [np.nan], "overlap": 0.5},
        {"bark_axis": [1.0] * 47, "overlap": 2.0},
    ],
)
def test_roughness_codec_rejects_invalid_constructor_state(state: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Invalid WDF Frame constructor value"):
        wdf_frames._validated_roughness_constructor_state(state)


def test_roughness_codec_rejects_bark_axis_tensor_mismatch() -> None:
    state = {"bark_axis": [1.0] * 47, "overlap": 0.5}
    with pytest.raises(ValueError, match="Invalid WDF Frame constructor value"):
        wdf_frames._roughness_decode({"data": da.ones((1, 46, 3))}, state)


def test_codec_rejects_unsupported_rank_and_exact_subclass(tmp_path: Path) -> None:
    codec = wdf_frames._codecs_by_name()["ChannelFrame"]
    with pytest.raises(ValueError, match="tensor rank"):
        wdf_frames._validate_codec_tensor(codec, da.ones((1, 2, 3)))

    class CustomChannelFrame(ChannelFrame):
        pass

    frame = CustomChannelFrame.from_numpy(np.ones((1, 4)), 8.0)
    with pytest.raises(TypeError, match="Unsupported Frame type"):
        frame.save(tmp_path / "custom.wdf")


def test_generic_coordinate_helpers_cover_singleton_length_and_unexpected_coordinate() -> None:
    frame = ChannelFrame.from_numpy(np.arange(8, dtype=float).reshape(1, -1), 8.0).cepstrum(n_fft=8)[:, :1]
    assert wdf_frames._coordinate_spacing(frame, "quefrency") is None
    with pytest.raises(ValueError, match="coordinate length"):
        wdf_frames._validate_coordinate_values(frame, "quefrency", np.array([0.0, 0.125]), 1)

    channel = ChannelFrame.from_numpy(np.ones((1, 4)), 8.0)
    with pytest.raises(ValueError, match="Invalid WDF coordinate dimension"):
        wdf_frames.restore_frame_coordinates(channel, {"sample": np.arange(4, dtype=float)})


def test_spectrogram_rejects_nonpositive_win_length() -> None:
    with pytest.raises(ValueError, match="win_length must be positive"):
        SpectrogramFrame(da.ones((1, 5, 2)), 8.0, n_fft=8, hop_length=2, win_length=0)
