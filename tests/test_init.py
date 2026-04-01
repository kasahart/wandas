import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import wandas


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


def test_channel_frame_dataset_attr() -> None:
    """wandas.ChannelFrameDataset が __getattr__ 経由でアクセスできることを確認"""
    from wandas.utils.frame_dataset import ChannelFrameDataset

    assert wandas.ChannelFrameDataset is ChannelFrameDataset


def test_unknown_attr_raises() -> None:
    """存在しない属性へのアクセスが AttributeError を送出することを確認"""
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = wandas.no_such_attribute
