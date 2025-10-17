<h1 align="center">
    <img src="https://github.com/kasahart/wandas/blob/main/images/logo.png?raw=true" alt="Wandas logo" width="300"/>
</h1>

[![PyPi](https://img.shields.io/pypi/v/wandas)](https://pypi.org/project/wandas/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wandas)
[![CI](https://github.com/kasahart/wandas/actions/workflows/ci.yml/badge.svg)](https://github.com/kasahart/wandas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kasahart/wandas/graph/badge.svg?token=53NPNQQZZ8)](https://codecov.io/gh/kasahart/wandas)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kasahart/wandas/blob/main/LICENSE)
[![Typing](https://img.shields.io/pypi/types/wandas)](https://pypi.org/project/wandas/)

**Wandas** (**W**aveform **An**alysis **Da**ta **S**tructures) is an open-source Python library for efficient signal analysis. It provides comprehensive signal processing functionalities and seamless integration with Matplotlib for visualization.

**Wandas** (**W**aveform **An**alysis **Da**ta **S**tructures)は、Pythonによる効率的な信号解析のためのオープンソースライブラリです。
Wandas は、信号処理のための包括的な機能を提供し、Matplotlibとのシームレスな統合を実現しています。

## Features / 機能

- **Comprehensive Signal Processing**:
  Easily perform basic signal processing operations such as filtering (low-pass, high-pass, band-pass, A-weighting), Fourier transforms (FFT, STFT, ISTFT), spectral analysis (Welch, CSD, Coherence, Transfer Function), N-octave analysis, and more.

  **包括的な信号処理機能**:
  フィルタリング（ローパス、ハイパス、バンドパス、A特性）、
  フーリエ変換（FFT、STFT、ISTFT）、
  スペクトル分析（Welch法、CSD、コヒーレンス、伝達関数）、
  Nオクターブ分析など、基本的な信号処理操作を簡単に実行可能。

- **Psychoacoustic Metrics**:
  Calculate loudness (ISO 532-1:2017) using the MoSQITo library integration. Supports both time-varying and stationary loudness calculations with free-field and diffuse-field conditions.

  **心理音響メトリクス**:
  MoSQIToライブラリ統合によるラウドネス計算（ISO 532-1:2017）。
  時間変化および定常状態のラウドネス計算、自由音場と拡散音場の両方に対応。

- **Intuitive Data Structures**:
  Utilizes `ChannelFrame` for time-domain data, `SpectralFrame` for frequency-domain data, and `SpectrogramFrame` for time-frequency data, offering a pandas-like experience.

  **直感的なデータ構造**:
  時間領域データには `ChannelFrame`、
  周波数領域データには `SpectralFrame`、
  時間周波数領域データには `SpectrogramFrame` を使用し、
  pandasライクな操作感を提供。

- **Visualization Integration**:
  Seamless integration with Matplotlib for easy and customizable data visualization. The `.plot()` and `.describe()` methods offer quick insights into your data.

  **可視化ライブラリとの統合**:
  Matplotlibとシームレスに統合し、
  簡単かつカスタマイズ可能なデータ可視化を実現。
  `.plot()` や `.describe()` メソッドで迅速にデータ概要を把握可能。

- **Efficient Large Data Handling**:
  Leverages lazy evaluation with Dask for efficient processing of large datasets.

  **効率的な大規模データ処理**:
  Daskを活用した遅延評価により、大規模データセットを効率的に処理。

- **Flexible I/O**:
  Supports reading and writing WAV and CSV files. Additionally, it features its own WDF (Wandas Data File) format based on HDF5 for complete data and metadata preservation.

  **柔軟なI/O**:
  WAVおよびCSVファイルの読み書きをサポート。
  さらに、データとメタデータを完全に保存するHDF5ベースの独自形式WDF (Wandas Data File) も搭載。

- **Metadata and History Tracking**:
  Keeps track of processing history and metadata associated with the signals.

  **メタデータと処理履歴の追跡**:
  信号に関連する処理履歴とメタデータを記録・管理。

- **Extensible API**:
  Designed for extensibility, allowing users to add custom processing functions.

  **拡張可能なAPI**:
  ユーザーがカスタム処理関数を追加しやすいように拡張性を考慮した設計。

- **Sample Datasets**:
  Includes sample datasets for testing and demonstration purposes.

## Installation
<!-- ## インストール -->

```bash
pip install git+https://github.com/endolith/waveform-analysis.git@master
pip install wandas
```

## Quick Start
<!-- ## クイックスタート -->

```python
import wandas as wd

# To run this example, place 'data/summer_streets1.wav' at the root of the repository,
# or change the path accordingly (e.g., 'examples/data/summer_streets1.wav').
# この例を実行するには、リポジトリのルートに 'data/summer_streets1.wav' を配置するか、
# 'examples/data/summer_streets1.wav' のようにパスを適宜変更してください。
cf = wd.read_wav("data/summer_streets1.wav")
cf.describe()
```

![cf.describe](https://github.com/kasahart/wandas/blob/main/images/read_wav_describe.png?raw=true)

```python
cf.describe(
    axis_config={
        "time_plot": {"xlim": (0, 15), "ylim": (-30000, 30000)},
        "freq_plot": {"xlim": (60, 120), "ylim": (0, 16000)},
    },
    cbar_config={"vmin": 10, "vmax": 70},
)
```

![cf.describe](https://github.com/kasahart/wandas/blob/main/images/read_wav_describe_set_config.png?raw=true)

```python
cf = wd.read_csv("data/test_signals.csv", time_column="Time")
cf.plot(title="Plot of test_signals.csv using wandas", overlay=False)
```

![cf.plot](https://github.com/kasahart/wandas/blob/main/images/plot_csv_using_wandas.png?raw=true)

### Signal Processing Example
<!-- ### 信号処理の例 -->

```python
# Example of applying a low-pass filter and plotting its FFT
# ローパスフィルタを適用し、そのFFTをプロットする例
signal = wd.generate_sin(freqs=[5000, 1000], duration=1, sampling_rate=44100)
filtered_signal = signal.low_pass_filter(cutoff=1000)
filtered_signal.fft().plot(title="FFT of Low-pass Filtered Signal")
```

![signal.low_pass_filter](https://github.com/kasahart/wandas/blob/main/images/low_pass_filter.png?raw=true)

```python
# Save the filtered signal as a WAV file
# フィルタ済み信号を WAV ファイルに保存
signal.low_pass_filter(cutoff=1000).to_wav('filtered_audio.wav')
# Display audio control
# Audioコントロール表示
signal.to_audio()
```

## Documentation
<!-- ## ドキュメント -->

For more detailed information, API reference, and tutorials, please visit the [official documentation site](https://kasahart.github.io/wandas/).

より詳細な情報やAPIリファレンス、チュートリアルについては、[公式ドキュメントサイト](https://kasahart.github.io/wandas/) をご覧ください。

## Tutorial

For practical usage and advanced examples, see the [Tutorial](tutorial/00_setup.ipynb) and the [Tutorial Index](tutorial/).

より実践的な使い方や応用例については、[チュートリアル](tutorial/00_setup.ipynb) および [チュートリアル一覧](tutorial/) をご覧ください。

## Supported Data Formats
<!-- ## 対応データ形式 -->

- **Audio Files**: WAV
    <!-- **音声ファイル**: WAV -->
- **Data Files**: CSV
    <!-- **データファイル**: CSV -->
- **Wandas Data Files**: WDF (HDF5-based)
    <!-- **Wandasデータファイル**: WDF (HDF5ベース) -->

## Bug Reports and Feature Requests

- **Bug Reports**: Please provide details in the [Issue Tracker](https://github.com/kasahart/wandas/issues).

- **Feature Requests**: Feel free to open an Issue if you have new features or improvement suggestions.

バグ報告と機能リクエスト

- **バグ報告**: [Issue Tracker](https://github.com/kasahart/wandas/issues) に詳細を記載してください。

- **機能リクエスト**: 新機能や改善案があれば、気軽に Issue をオープンしてください。

## License
<!-- ## ライセンス -->

This project is licensed under the [MIT License](LICENSE).
<!-- このプロジェクトは [MIT ライセンス](LICENSE) の下で公開されています。 -->

---

Experience efficient signal analysis with Wandas!

Wandas を使って効率的な信号解析体験を！
