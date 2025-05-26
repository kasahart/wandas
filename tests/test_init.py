import logging

import pytest

import wandas


@pytest.fixture(autouse=True)
def reset_logger():
    """各テストの前にロガーのハンドラーをリセット"""
    logger = logging.getLogger("wandas")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    yield
    # テスト後も同様にリセット
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


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
    format_str = (
        formatter._style._fmt
        if formatter is not None and hasattr(formatter, "_style")
        else str(formatter)
    )
    assert "%(asctime)s" in format_str
    assert "%(name)s" in format_str
    assert "%(levelname)s" in format_str
    assert "%(message)s" in format_str
