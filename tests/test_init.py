import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import wandas
from wandas.frames.channel import ChannelFrame
from wandas.frames.noct import NOctFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.utils.frame_dataset import ChannelFrameDataset


@pytest.fixture(autouse=True)
def reset_logger():
    """各テストの前にロガーのハンドラーをリセット"""
    logger = logging.getLogger("wandas")
    original_level = logger.level
    original_disabled = logger.disabled
    original_propagate = logger.propagate
    original_handlers = logger.handlers[:]
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    yield
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(original_level)
    logger.disabled = original_disabled
    logger.propagate = original_propagate
    for handler in original_handlers:
        logger.addHandler(handler)


def test_default_settings():
    """デフォルト設定でロガーが正しく設定されるか確認"""
    logger = wandas.setup_wandas_logging()

    assert logger.name == "wandas"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_custom_level_string():
    """文字列でログレベルを指定した場合"""
    logger = wandas.setup_wandas_logging(level="DEBUG")

    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1


def test_custom_level_int():
    """整数値でログレベルを指定した場合"""
    logger = wandas.setup_wandas_logging(level=logging.ERROR)

    assert logger.level == logging.ERROR
    assert len(logger.handlers) == 1


def test_invalid_level_string():
    """無効なログレベル文字列を指定した場合、デフォルトのINFOになる"""
    logger = wandas.setup_wandas_logging(level="INVALID_LEVEL")

    assert logger.level == logging.INFO


def test_no_handler():
    """add_handler=Falseを指定した場合"""
    logger = wandas.setup_wandas_logging(add_handler=False)

    assert logger.level == logging.INFO
    assert len(logger.handlers) == 0


def test_supported_formats_public_api() -> None:
    formats = wandas.supported_formats()

    assert formats == [".aif", ".aiff", ".csv", ".flac", ".ogg", ".snd", ".wav"]
    assert ".mp3" not in formats


def test_existing_handler():
    """すでにハンドラーがある場合、新しいハンドラーは追加されない"""
    # 事前にハンドラーを追加
    logger = logging.getLogger("wandas")
    mock_handler = logging.StreamHandler()
    logger.addHandler(mock_handler)

    # setup_wandas_logging を呼び出し
    result_logger = wandas.setup_wandas_logging()

    # 1つのハンドラーだけ存在すること
    assert len(result_logger.handlers) == 1
    assert result_logger.handlers[0] == mock_handler


def test_formatter():
    """フォーマッターが正しく設定されているか"""
    logger = wandas.setup_wandas_logging()
    handler = logger.handlers[0]

    # フォーマッターのフォーマット文字列をチェック
    formatter = handler.formatter
    format_str = formatter._style._fmt if formatter is not None and hasattr(formatter, "_style") else str(formatter)
    assert "%(asctime)s" in format_str
    assert "%(name)s" in format_str
    assert "%(levelname)s" in format_str
    assert "%(message)s" in format_str


def test_top_level_all_is_curated_primary_api() -> None:
    assert wandas.__all__ == [
        "ChannelFrame",
        "SpectralFrame",
        "SpectrogramFrame",
        "NOctFrame",
        "ChannelFrameDataset",
        "read",
        "load",
        "from_numpy",
        "from_folder",
    ]


def test_top_level_frame_classes_are_public() -> None:
    assert wandas.ChannelFrame is ChannelFrame
    assert wandas.SpectralFrame is SpectralFrame
    assert wandas.SpectrogramFrame is SpectrogramFrame
    assert wandas.NOctFrame is NOctFrame
    assert wandas.ChannelFrameDataset is ChannelFrameDataset


def test_frames_module_all_matches_documented_frames() -> None:
    import wandas.frames as frames
    from wandas.frames.roughness import RoughnessFrame

    assert frames.__all__ == [
        "ChannelFrame",
        "SpectralFrame",
        "SpectrogramFrame",
        "NOctFrame",
        "RoughnessFrame",
    ]
    assert frames.ChannelFrame is ChannelFrame
    assert frames.SpectralFrame is SpectralFrame
    assert frames.SpectrogramFrame is SpectrogramFrame
    assert frames.NOctFrame is NOctFrame
    assert frames.RoughnessFrame is RoughnessFrame


def test_compatibility_helpers_remain_importable_but_outside_all() -> None:
    assert callable(wandas.read_wav)
    assert callable(wandas.read_csv)
    assert callable(wandas.from_ndarray)
    assert callable(wandas.generate_sin)
    assert isinstance(ChannelFrame.__dict__["read_wav"], classmethod)
    assert isinstance(ChannelFrame.__dict__["read_csv"], classmethod)
    assert isinstance(ChannelFrame.__dict__["from_ndarray"], classmethod)
    assert "read_wav" not in wandas.__all__
    assert "read_csv" not in wandas.__all__
    assert "from_ndarray" not in wandas.__all__
    assert "generate_sin" not in wandas.__all__


def test_from_ndarray_remains_deprecated_compatibility_helper() -> None:
    with pytest.warns(DeprecationWarning, match="from_ndarray is deprecated"):
        signal = wandas.from_ndarray(np.array([[0.0, 1.0]], dtype=np.float32), sampling_rate=1000)

    assert isinstance(signal, ChannelFrame)
    assert signal.sampling_rate == 1000


def test_from_folder(tmp_path: Path) -> None:
    """wd.from_folder が ChannelFrameDataset を返すことを確認"""
    from wandas.utils.frame_dataset import ChannelFrameDataset

    # Create a minimal WAV file for the dataset
    sr = 16000
    data = np.zeros((sr, 1), dtype=np.float32)
    sf.write(str(tmp_path / "test.wav"), data, sr)

    dataset = wandas.from_folder(str(tmp_path), file_extensions=[".wav"])
    assert isinstance(dataset, ChannelFrameDataset)
    assert len(dataset) == 1


def test_from_folder_default_extensions_exclude_mp3(tmp_path: Path) -> None:
    (tmp_path / "readable.csv").write_text("time,ch1\n0.0,1.0\n", encoding="utf-8")
    (tmp_path / "ignored.mp3").write_bytes(b"")

    dataset = wandas.from_folder(str(tmp_path))

    names = [lazy_frame.file_path.name for lazy_frame in dataset._lazy_frames]
    assert names == ["readable.csv"]
    assert ".mp3" not in dataset.file_extensions


def test_from_folder_same_as_class_method(tmp_path: Path) -> None:
    """wd.from_folder が ChannelFrameDataset.from_folder と同じ結果を返すことを確認"""
    from wandas.utils.frame_dataset import ChannelFrameDataset

    sr = 16000
    data = np.zeros((sr, 1), dtype=np.float32)
    sf.write(str(tmp_path / "test.wav"), data, sr)

    ds1 = wandas.from_folder(str(tmp_path), sampling_rate=8000, file_extensions=[".wav"])
    ds2 = ChannelFrameDataset.from_folder(str(tmp_path), sampling_rate=8000, file_extensions=[".wav"])
    assert type(ds1) is type(ds2)
    assert ds1.sampling_rate == ds2.sampling_rate
    assert len(ds1) == len(ds2)


def test_read_loads_wav_like_read_wav(tmp_path: Path) -> None:
    sr = 16000
    data = np.zeros((sr, 1), dtype=np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), data, sr)

    signal = wandas.read(path)

    assert isinstance(signal, ChannelFrame)
    assert signal.sampling_rate == sr
    assert signal.n_channels == 1
    assert signal.label == "test"


def test_read_loads_csv_like_read_csv(tmp_path: Path) -> None:
    path = tmp_path / "sensor.csv"
    path.write_text("time,left,right\n0.0,1.0,2.0\n0.1,3.0,4.0\n", encoding="utf-8")

    signal = wandas.read(path)

    assert isinstance(signal, ChannelFrame)
    assert signal.sampling_rate == 10
    assert signal.n_channels == 2
    assert signal.labels == ["left", "right"]


def test_read_rejects_wdf_with_load_guidance(tmp_path: Path) -> None:
    path = tmp_path / "analysis.wdf"
    path.write_bytes(b"not a real wdf")

    with pytest.raises(ValueError, match="wd.load") as exc_info:
        wandas.read(path)

    message = str(exc_info.value)
    assert "WDF files are loaded with wd.load(), not wd.read()" in message
    assert "analysis.wdf" in message


def test_read_rejects_wdf_file_type_with_load_guidance() -> None:
    with pytest.raises(ValueError, match="wd.load"):
        wandas.read(b"not a real wdf", file_type=".wdf")


def test_load_reads_wdf(tmp_path: Path) -> None:
    sr = 8000
    source = ChannelFrame.from_numpy(
        np.array([[0.0, 0.5, -0.5]], dtype=np.float32),
        sampling_rate=sr,
        ch_labels=["source"],
    )
    path = tmp_path / "analysis.wdf"
    source.save(path)

    loaded = wandas.load(path)

    assert isinstance(loaded, ChannelFrame)
    assert loaded.sampling_rate == sr
    assert loaded.labels == ["source"]
    np.testing.assert_allclose(loaded.compute(), source.compute())


def test_channel_frame_dataset_attr() -> None:
    """wandas.ChannelFrameDataset が __getattr__ 経由でアクセスできることを確認"""
    from wandas.utils.frame_dataset import ChannelFrameDataset

    assert wandas.ChannelFrameDataset is ChannelFrameDataset


def test_unknown_attr_raises() -> None:
    """存在しない属性へのアクセスが AttributeError を送出することを確認"""
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = wandas.no_such_attribute
