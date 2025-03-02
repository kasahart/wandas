# tests/io/test_wav_io.py
import os

import numpy as np
import pytest
from scipy.io.wavfile import write

from wandas.io import read_wav


@pytest.fixture  # type: ignore [misc, unused-ignore]
def create_test_wav(tmpdir: str) -> str:
    """
    テスト用の一時的な WAV ファイルを作成するフィクスチャ。
    テスト後に自動で削除されます。
    """
    # 一時ディレクトリに WAV ファイルを作成
    filename = os.path.join(tmpdir, "test_file.wav")

    # サンプルデータを作成
    sampling_rate = 44100
    duration = 1.0  # 1秒

    # 左右に振幅差をつけた直流データを生成
    data_left = (
        np.ones(int(sampling_rate * duration)) * 0.5
    )  # 左チャンネル (直流信号、振幅0.5)
    data_right = np.ones(
        int(sampling_rate * duration)
    )  # 右チャンネル (直流信号、振幅1.0)

    stereo_data = np.column_stack((data_left, data_right))

    # WAV ファイルを書き出し
    write(filename, sampling_rate, stereo_data)

    return filename


def test_read_wav(create_test_wav: str) -> None:
    # テスト用の WAV ファイルを読み込む
    signal = read_wav(create_test_wav)

    # チャンネル数の確認
    assert len(signal.channels) == 2

    # サンプリングレートの確認
    assert signal.sampling_rate == 44100

    # チャンネルデータの確認
    assert np.allclose(signal.channels[0].data, 0.5)
    assert np.allclose(signal.channels[1].data, 1.0)
