---
applyTo: '.py, .ipynb'
---

# Wandas プロジェクト開発ガイドライン

## プロジェクト概要
Wandas (**W**aveform **An**alysis **Da**ta **S**tructures) は、音響信号・波形解析に特化したPythonライブラリです。
pandasライクなAPIで信号処理、スペクトル解析、可視化を提供します。

## 設計原則

これらの原則はすべてのコード変更において最優先で遵守してください：

1. **Pandasライクなインターフェース**: ユーザーがpandasの操作感で信号処理できるようにする
2. **型安全性**: mypyの厳格モードに準拠し、実行時エラーを防ぐ
3. **チェインメソッド**: メソッドチェーンで複数の処理を直感的に記述できるようにする
4. **遅延評価**: Dask配列を活用し、大規模データでもメモリ効率的に処理する
5. **拡張性**: ユーザーが独自の処理を追加しやすい設計にする
6. **テスタビリティ**: すべての機能が独立してテスト可能な設計にする
7. **ドキュメント駆動**: コードの意図が明確に伝わるドキュメントを提供する

## コーディング規約

### 1. 型ヒントとType Safety
- **すべての関数・メソッドに型ヒントを必須で付与**してください
- mypyの厳格モード (`strict = true`) に準拠してください
- 型エイリアスは `wandas.utils.types` から使用してください（例: `NDArrayReal`, `NDArrayComplex`）
- 戻り値の型も必ず明示してください（`None` を含む）

```python
# 良い例
def process_signal(data: NDArrayReal, sampling_rate: float) -> NDArrayComplex:
    ...

# 悪い例
def process_signal(data, sampling_rate):  # 型ヒントなし
    ...
```

### 2. NumPy/Dask配列の扱い
- **Dask配列を優先的に使用**して遅延評価を活用してください
- NumPy配列との相互運用性を保ってください
- 配列操作は軸（axis）を明示的に指定してください
- 形状（shape）の検証を行い、期待する次元数をチェックしてください

```python
# 良い例
import dask.array as da

def apply_filter(data: DaskArray, axis: int = -1) -> DaskArray:
    if data.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D array, got {data.ndim}D")
    result = da.some_operation(data, axis=axis)
    return result
```

### 3. 信号処理の実装
- **AudioOperation基底クラス**を継承して新しい処理を実装してください
- `_process_array` メソッドで実際の処理ロジックを実装してください
- 処理は`@register_operation`デコレータで登録してください
- サンプリングレート、FFTサイズ、窓関数などのパラメータを適切に管理してください
- **不変性の原則**: 元のデータを変更せず、新しいフレームを返してください

```python
from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal

@register_operation
class MyCustomFilter(AudioOperation[NDArrayReal, NDArrayReal]):
    """
    Custom filter for signal processing.

    This filter applies a custom algorithm to the input signal
    while preserving the original data structure.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the input signal in Hz.
    cutoff : float
        Cutoff frequency in Hz. Must be less than Nyquist frequency.

    Raises
    ------
    ValueError
        If cutoff frequency is greater than or equal to Nyquist frequency.
    """

    name = "my_custom_filter"

    def __init__(self, sampling_rate: float, cutoff: float) -> None:
        if cutoff >= sampling_rate / 2:
            raise ValueError(
                f"Cutoff frequency ({cutoff} Hz) must be less than "
                f"Nyquist frequency ({sampling_rate / 2} Hz)"
            )
        self.cutoff = cutoff
        super().__init__(sampling_rate, cutoff=cutoff)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process the input array.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array.

        Returns
        -------
        NDArrayReal
            Filtered signal array.
        """
        # 実装
        ...
```

### 4. メタデータと処理履歴
- **処理履歴を必ず記録**してください（`operation_history`）
- チャンネルメタデータ（`ChannelMetadata`）を適切に管理してください
- `previous` 属性で前処理のフレームへの参照を保持してください
- **トレーサビリティ**: どの処理がどの順序で適用されたかを追跡可能にしてください

```python
from wandas.core.metadata import ChannelMetadata, OperationRecord
from typing import Optional

def create_processed_frame(
    self,
    data: NDArrayReal,
    operation_name: str,
    **params: Any
) -> "ChannelFrame":
    """
    Create a new frame with operation history.

    Parameters
    ----------
    data : NDArrayReal
        Processed data array.
    operation_name : str
        Name of the operation applied.
    **params : Any
        Parameters used in the operation.

    Returns
    -------
    ChannelFrame
        New frame with updated metadata and operation history.
    """
    new_history = self.operation_history.copy()
    new_history.append(
        OperationRecord(
            name=operation_name,
            params=params,
            timestamp=datetime.now()
        )
    )

    return ChannelFrame(
        data=data,
        sampling_rate=self.sampling_rate,
        channel_metadata=self.channel_metadata,
        operation_history=new_history,
        previous=self
    )
```

### 5. ドキュメンテーション
- **docstringは英語で記述**してください
- **NumPy/Google形式のdocstring**を使用してください
- **すべてのパラメータに説明を記載**してください（型、デフォルト値、説明を含む）
- 引数、戻り値、Raises、Examples セクションを含めてください
- 数式はLaTeX記法で記述してください（MkDocsでレンダリング）

```python
def fft(self, n_fft: Optional[int] = None, window: str = "hann") -> "SpectralFrame":
    """
    Apply Fast Fourier Transform to the signal.

    Parameters
    ----------
    n_fft : int, optional
        Size of FFT. If None, it will be determined based on the data length.
        Must be a power of 2 for optimal performance.
    window : str, default="hann"
        Window function type to apply before FFT.
        Supported values: 'hann', 'hamming', 'blackman', 'bartlett', 'boxcar'.

    Returns
    -------
    SpectralFrame
        A new SpectralFrame containing the frequency domain representation
        of the input signal.

    Raises
    ------
    ValueError
        If n_fft is not a positive integer or if window type is not supported.

    Examples
    --------
    >>> signal = wd.read_wav("audio.wav")
    >>> spectrum = signal.fft(n_fft=2048)
    >>> spectrum.plot()
    """
    ...
```

### 6. テスト
- **pytest**を使用してテストを記述してください
- 各モジュールに対応する `tests/` ディレクトリにテストファイルを配置してください
- フィクスチャを活用してテストデータを共有してください
- **カバレッジ100%を目標**にしてください（最低90%以上）
- **テストの独立性**: 各テストは他のテストに依存せず、単独で実行可能にしてください
- **テストの可読性**: テスト名は「何をテストしているか」が明確にわかるようにしてください
- **数値検証の原則**: 数値のチェックを行う場合は、理論値と照合して検証してください
  - 単なる非ゼロチェックや範囲チェックではなく、期待される理論値（数式から導出される値）と比較してください
  - 浮動小数点数の比較には適切な許容誤差を設定してください（`np.allclose()`, `pytest.approx()` など）
  - 理論値の根拠をコメントやdocstringに記載してください

```python
import pytest
import numpy as np
from wandas.frames.channel import ChannelFrame
import wandas as wd

@pytest.fixture
def sample_signal() -> ChannelFrame:
    """
    Generate a sample signal for testing.

    Returns
    -------
    ChannelFrame
        A 1-second signal containing 440 Hz and 880 Hz sine waves.
    """
    return wd.generate_sin(freqs=[440, 880], duration=1.0, sampling_rate=44100)

@pytest.fixture
def sample_signal_with_noise(sample_signal: ChannelFrame) -> ChannelFrame:
    """
    Generate a noisy sample signal for testing.

    Parameters
    ----------
    sample_signal : ChannelFrame
        Clean sample signal.

    Returns
    -------
    ChannelFrame
        Signal with added Gaussian noise (SNR = 20 dB).
    """
    noise = np.random.normal(0, 0.01, sample_signal.shape)
    return sample_signal + noise

def test_low_pass_filter_preserves_shape(sample_signal: ChannelFrame) -> None:
    """Test that low-pass filter preserves the signal shape."""
    filtered = sample_signal.low_pass_filter(cutoff=1000)
    assert filtered.n_samples == sample_signal.n_samples
    assert filtered.n_channels == sample_signal.n_channels

def test_low_pass_filter_preserves_sampling_rate(sample_signal: ChannelFrame) -> None:
    """Test that low-pass filter preserves the sampling rate."""
    filtered = sample_signal.low_pass_filter(cutoff=1000)
    assert filtered.sampling_rate == sample_signal.sampling_rate

def test_low_pass_filter_records_operation_history(sample_signal: ChannelFrame) -> None:
    """Test that low-pass filter records operation in history."""
    filtered = sample_signal.low_pass_filter(cutoff=1000)
    assert len(filtered.operation_history) == len(sample_signal.operation_history) + 1
    assert filtered.operation_history[-1].name == "low_pass_filter"

def test_low_pass_filter_with_invalid_cutoff_raises_error(
    sample_signal: ChannelFrame
) -> None:
    """Test that invalid cutoff frequency raises ValueError."""
    with pytest.raises(ValueError, match="Cutoff frequency.*must be less than"):
        sample_signal.low_pass_filter(cutoff=50000)  # > Nyquist frequency

def test_low_pass_filter_attenuates_high_frequencies(
    sample_signal: ChannelFrame
) -> None:
    """Test that low-pass filter actually attenuates high frequencies."""
    filtered = sample_signal.low_pass_filter(cutoff=600)

    # 元の信号のスペクトル
    original_spectrum = sample_signal.fft()
    filtered_spectrum = filtered.fft()

    # 880 Hz成分が減衰していることを確認
    # 理論値: カットオフ周波数600Hzのローパスフィルタは880Hz成分を大きく減衰させるはず
    freq_880_idx = np.argmin(np.abs(original_spectrum.freqs - 880))
    assert (
        np.abs(filtered_spectrum.data[freq_880_idx]) <
        np.abs(original_spectrum.data[freq_880_idx]) * 0.5
    )

def test_fft_preserves_energy(sample_signal: ChannelFrame) -> None:
    """Test that FFT preserves signal energy (Parseval's theorem)."""
    # Parseval's theorem: sum(|x[n]|^2) = (1/N) * sum(|X[k]|^2)
    # 時間領域のエネルギー
    time_energy = np.sum(np.abs(sample_signal.data) ** 2)

    # 周波数領域のエネルギー
    spectrum = sample_signal.fft()
    freq_energy = np.sum(np.abs(spectrum.data) ** 2) / len(sample_signal.data)

    # 理論値: 両者は等しいはず（Parsevalの定理）
    np.testing.assert_allclose(time_energy, freq_energy, rtol=1e-10)

def test_normalize_produces_unit_maximum(sample_signal: ChannelFrame) -> None:
    """Test that normalization produces signal with maximum amplitude of 1.0."""
    normalized = sample_signal.normalize()

    # 理論値: 正規化後の最大振幅は1.0になるはず
    assert np.abs(np.max(np.abs(normalized.data)) - 1.0) < 1e-10
```

### 7. ロギング
- **標準ライブラリのlogging**を使用してください
- デバッグ情報は`logger.debug()`で、重要な情報は`logger.info()`で記録してください
- エラーメッセージは明確で実用的な内容にしてください

```python
import logging

logger = logging.getLogger(__name__)

def process_data(data: NDArrayReal) -> NDArrayReal:
    logger.debug(f"Processing data with shape: {data.shape}")
    try:
        result = some_operation(data)
        logger.debug(f"Processing completed, result shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        raise
```

### 8. パフォーマンス
- 大規模データ処理では**Daskの遅延評価**を活用してください
- 不要な `.compute()` 呼び出しを避けてください
- メモリ効率を意識した実装を心がけてください
- ベクトル化されたNumPy演算を優先してください
- **プロファイリング**: パフォーマンスが重要な処理では計測を行ってください

```python
import dask.array as da
from typing import Union
import numpy as np

def efficient_processing(
    data: Union[np.ndarray, da.Array],
    chunk_size: int = 1000
) -> da.Array:
    """
    Process data efficiently using Dask for large arrays.

    Parameters
    ----------
    data : Union[np.ndarray, da.Array]
        Input data array.
    chunk_size : int, default=1000
        Chunk size for Dask array processing.

    Returns
    -------
    da.Array
        Processed data as Dask array (not computed).

    Notes
    -----
    This function returns a Dask array without computing it,
    allowing users to chain operations efficiently.
    """
    # NumPy配列の場合はDask配列に変換
    if isinstance(data, np.ndarray):
        data = da.from_array(data, chunks=chunk_size)

    # ベクトル化された演算を使用
    result = da.fft.fft(data, axis=-1)

    # compute()を呼ばずにDask配列を返す
    return result
```

### 9. 可視化
- **Matplotlibとの統合**を維持してください
- `.plot()` メソッドは `matplotlib.axes.Axes` を返すようにしてください
- 日本語表示には `japanize-matplotlib` を活用してください
- カスタマイズ可能なプロットパラメータを提供してください
- **アクセシビリティ**: カラーブラインド対応の配色を使用してください

```python
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def plot(
    self,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Time [s]",
    ylabel: str = "Amplitude",
    color: str = "#1f77b4",  # デフォルトの青（カラーブラインド対応）
    **kwargs: Any
) -> Axes:
    """
    Plot the signal in time domain.

    Parameters
    ----------
    ax : Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    title : str, optional
        Plot title. If None, no title is displayed.
    xlabel : str, default="Time [s]"
        Label for the x-axis.
    ylabel : str, default="Amplitude"
        Label for the y-axis.
    color : str, default="#1f77b4"
        Line color in hex or named color format.
    **kwargs : Any
        Additional keyword arguments passed to matplotlib.pyplot.plot.

    Returns
    -------
    Axes
        The matplotlib axes object containing the plot.

    Examples
    --------
    >>> signal = wd.read_wav("audio.wav")
    >>> ax = signal.plot(title="My Signal", color="red")
    >>> plt.show()
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    time = np.arange(self.n_samples) / self.sampling_rate
    ax.plot(time, self.data, color=color, **kwargs)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    return ax
```

### 10. エラーハンドリング
- **明確なエラーメッセージ**: ユーザーが問題を理解し、解決できるメッセージを提供してください
- **適切な例外型**: 状況に応じた適切な例外を使用してください
  - `ValueError`: 不正な値やパラメータ
  - `TypeError`: 型の不一致
  - `RuntimeError`: 実行時の予期しない状態
  - カスタム例外: ドメイン固有のエラー
- **入力検証**: 関数の先頭で入力値を検証してください
- **リソース管理**: ファイルやネットワーク接続は適切にクローズしてください

```python
from pathlib import Path
from typing import Union

class WandasError(Exception):
    """Base exception for wandas library."""
    pass

class InvalidSamplingRateError(WandasError):
    """Raised when sampling rate is invalid."""
    pass

def read_wav(
    filepath: Union[str, Path],
    sampling_rate: Optional[float] = None
) -> "ChannelFrame":
    """
    Read WAV file and create ChannelFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the WAV file to read.
    sampling_rate : float, optional
        Expected sampling rate. If provided and doesn't match the file,
        raises InvalidSamplingRateError.

    Returns
    -------
    ChannelFrame
        Audio data from the WAV file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    InvalidSamplingRateError
        If the file's sampling rate doesn't match the expected rate.
    ValueError
        If the file format is invalid or corrupted.

    Examples
    --------
    >>> signal = wd.read_wav("audio.wav")
    >>> signal = wd.read_wav("audio.wav", sampling_rate=44100)
    """
    filepath = Path(filepath)

    # 入力検証
    if not filepath.exists():
        raise FileNotFoundError(
            f"WAV file not found: {filepath}\n"
            f"Please check the file path and try again."
        )

    if not filepath.suffix.lower() == ".wav":
        raise ValueError(
            f"Expected WAV file, got: {filepath.suffix}\n"
            f"This function only supports .wav files."
        )

    # ファイル読み込み（with文でリソース管理）
    try:
        with open(filepath, "rb") as f:
            # 読み込み処理
            file_sr, data = _read_wav_data(f)
    except Exception as e:
        raise ValueError(
            f"Failed to read WAV file: {filepath}\n"
            f"The file may be corrupted or in an unsupported format.\n"
            f"Error details: {e}"
        ) from e

    # サンプリングレートの検証
    if sampling_rate is not None and file_sr != sampling_rate:
        raise InvalidSamplingRateError(
            f"Sampling rate mismatch:\n"
            f"  Expected: {sampling_rate} Hz\n"
            f"  File contains: {file_sr} Hz\n"
            f"Consider using resample() method to convert the sampling rate."
        )

    return ChannelFrame(data=data, sampling_rate=file_sr)
```

### 11. 互換性
- **Python 3.9以上**をサポートしてください
- 後方互換性を保つよう注意してください
- 破壊的変更は慎重に検討し、適切に文書化してください
- **非推奨化**: 機能を削除する前に少なくとも1バージョンは非推奨警告を出してください

```python
import warnings
from typing import Optional

def old_method(self, param: float) -> "ChannelFrame":
    """
    Old method (deprecated).

    .. deprecated:: 0.2.0
        Use :meth:`new_method` instead. This method will be removed in v0.4.0.

    Parameters
    ----------
    param : float
        Some parameter.

    Returns
    -------
    ChannelFrame
        Processed frame.
    """
    warnings.warn(
        "old_method is deprecated and will be removed in v0.4.0. "
        "Use new_method instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.new_method(param)
```

## コード変更時の手順

コードを変更する際は、以下の手順を**必ず**順守してください：

### 1. 変更プランの作成
- **変更プランを記載したMarkdownファイルを作成**してください
- ファイル名: `PLAN_<機能名または変更内容>.md`（例: `PLAN_add_bandpass_filter.md`）
- プランには以下の内容を**すべて**含めてください：
  - **変更の目的と背景**: なぜこの変更が必要か
  - **影響を受けるファイルとモジュール**: 変更するファイルのリスト
  - **実装方針と技術的な詳細**: どのように実装するか
  - **テスト戦略**: どのようなテストケースが必要か
  - **想定されるリスクと対応策**: 何が問題になりうるか
  - **後方互換性**: 既存コードへの影響
  - **ドキュメント更新計画**: どのドキュメントを更新するか
- **変更作業中は常にこのプランファイルを参照**してください
- プランファイルは変更完了後もリポジトリに残してください（将来の参照用）

### 2. 変更プランのレビュー
実装前に以下の観点で**必ず**レビューを実施してください：

#### 設計チェックリスト
- [ ] 設計原則（Pandasライクなインターフェース、型安全性など）に沿っているか
- [ ] 後方互換性は保たれているか（破壊的変更の場合は非推奨化の計画があるか）
- [ ] パフォーマンスへの影響は考慮されているか
- [ ] メモリ効率は最適か（大規模データでの動作を考慮）
- [ ] エラーハンドリングは適切か
- [ ] 関連する既存コードへの影響は洗い出されているか

#### ドキュメントチェックリスト
- [ ] docstringは英語で記述されているか
- [ ] すべてのパラメータに説明があるか
- [ ] 戻り値の説明があるか
- [ ] 発生する可能性のある例外が記載されているか
- [ ] 使用例（Examples）が含まれているか

### 3. テスト駆動開発
- **変更部分のテストケースを先に作成**してください
- **カバレッジ100%を達成**するようにテストを設計してください
- 以下のテストパターンを**すべて**含めてください：
  - **正常系のテスト**: 期待通りの動作を確認
  - **異常系のテスト**: エラーハンドリングを確認
  - **境界値のテスト**: 極端な値での動作を確認
  - **エッジケースのテスト**: 特殊なケースでの動作を確認
  - **統合テスト**: 他のモジュールとの連携を確認
- テストが**失敗することを確認**してから実装を開始してください（Red-Green-Refactorサイクル）

#### テスト作成の具体例
```python
# tests/processing/test_new_feature.py

def test_new_feature_normal_case():
    """正常系: 基本的な動作を確認"""
    ...

def test_new_feature_with_edge_values():
    """境界値: 最小値・最大値での動作を確認"""
    ...

def test_new_feature_raises_error_on_invalid_input():
    """異常系: 不正な入力でエラーが発生することを確認"""
    ...

def test_new_feature_preserves_metadata():
    """メタデータが保持されることを確認"""
    ...

def test_new_feature_records_operation_history():
    """処理履歴が記録されることを確認"""
    ...
```

### 4. 実装とテストの反復
- **小さな単位**で実装とテストを繰り返してください
- 各ステップで**テストが通ることを確認**してください
- コミットは**意味のある単位**で行ってください
  - Good: `feat: Add bandpass filter implementation`
  - Bad: `update`, `fix`, `changes`
- コミットメッセージは[Conventional Commits](https://www.conventionalcommits.org/)形式を推奨

### 5. 最終確認
実装完了後、以下を**すべて**確認してください：

#### 品質チェックリスト
- [ ] すべてのテストが通ることを確認（`uv run pytest`）
- [ ] カバレッジレポートで100%達成を確認（`uv run pytest --cov`）
- [ ] 型チェックが通ることを確認（`uv run mypy --config-file=pyproject.toml`）
- [ ] リントが通ることを確認（`uv run ruff check wandas tests`）
- [ ] フォーマットが整っていることを確認（`uv run ruff format wandas tests`）

#### ドキュメントチェックリスト
- [ ] 変更したコードにdocstringが記載されているか
- [ ] APIドキュメント（`docs/`）が更新されているか
- [ ] README.mdが更新されているか（必要な場合）
- [ ] CHANGELOGが更新されているか
- [ ] 使用例が追加されているか（`examples/`）

#### 統合チェックリスト
- [ ] 他のモジュールとの統合テストが通るか
- [ ] 既存の機能に影響がないか（リグレッションテスト）
- [ ] パフォーマンステストが通るか（該当する場合）

## ツールとワークフロー

### 開発環境セットアップ
```bash
# uvを使用した環境構築
uv venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows

# 依存関係のインストール（開発用含む）
uv pip install -e ".[dev]"
```

### コード品質チェック
```bash
# Ruffでリント・フォーマット
uv run ruff check wandas tests --fix
uv run ruff format wandas tests

# mypyで型チェック
uv run mypy --config-file=pyproject.toml

# テスト実行
uv run pytest

# カバレッジ付きテスト実行
uv run pytest --cov=wandas --cov-report=html --cov-report=term

# 特定のファイルのカバレッジ確認
uv run pytest tests/frames/test_channel_frame.py --cov=wandas.frames.channel --cov-report=term-missing
```

### 継続的インテグレーション
- すべてのプルリクエストで自動テストが実行されます
- カバレッジが90%未満の場合は警告が表示されます
- 型チェックとリントのエラーはマージをブロックします

### プリコミットフック
`.pre-commit-config.yaml` を使用して、コミット前に自動チェックが実行されます。

```bash
# pre-commitのインストール
pre-commit install

# 手動実行
pre-commit run --all-files
```

## ベストプラクティス集

### 1. メソッドチェーンの実装
```python
# 良い例: メソッドチェーンが可能
signal = (
    wd.read_wav("audio.wav")
    .normalize()
    .low_pass_filter(cutoff=1000)
    .resample(target_rate=16000)
)

# 実装時は常にselfまたは新しいフレームを返す
def normalize(self) -> "ChannelFrame":
    """Normalize signal amplitude to [-1, 1]."""
    normalized_data = self.data / np.max(np.abs(self.data))
    return self._create_new_frame(normalized_data, operation_name="normalize")
```

### 2. 型ヒントの活用
```python
from typing import Union, Optional, Literal, overload
from wandas.utils.types import NDArrayReal, DaskArray

# オーバーロードで戻り値の型を明確にする
@overload
def process(self, mode: Literal["numpy"]) -> NDArrayReal: ...

@overload
def process(self, mode: Literal["dask"]) -> DaskArray: ...

def process(
    self,
    mode: Literal["numpy", "dask"] = "numpy"
) -> Union[NDArrayReal, DaskArray]:
    """Process data with specified backend."""
    ...
```

### 3. リソース管理
```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def open_audio_file(filepath: Path) -> Generator[AudioFile, None, None]:
    """
    Context manager for audio file handling.

    Ensures proper resource cleanup even if an error occurs.
    """
    file = AudioFile(filepath)
    try:
        yield file
    finally:
        file.close()

# 使用例
with open_audio_file("audio.wav") as f:
    data = f.read()
```

### 4. 設定可能なデフォルト値
```python
from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class FilterConfig:
    """Global configuration for filters."""

    default_order: ClassVar[int] = 5
    default_window: ClassVar[str] = "hann"

    # インスタンス変数
    order: int = field(default_factory=lambda: FilterConfig.default_order)
    window: str = field(default_factory=lambda: FilterConfig.default_window)

# ユーザーがグローバル設定を変更可能
FilterConfig.default_order = 8
```

### 5. デバッグ情報の提供
```python
def __repr__(self) -> str:
    """Provide detailed string representation for debugging."""
    return (
        f"{self.__class__.__name__}("
        f"n_samples={self.n_samples}, "
        f"n_channels={self.n_channels}, "
        f"sampling_rate={self.sampling_rate}, "
        f"duration={self.duration:.2f}s"
        ")"
    )

def _repr_html_(self) -> str:
    """Provide HTML representation for Jupyter notebooks."""
    return f"""
    <div>
        <strong>{self.__class__.__name__}</strong>
        <ul>
            <li>Samples: {self.n_samples:,}</li>
            <li>Channels: {self.n_channels}</li>
            <li>Sampling Rate: {self.sampling_rate:,} Hz</li>
            <li>Duration: {self.duration:.2f} s</li>
        </ul>
    </div>
    """
```

## よくある質問（FAQ）

### Q: NumPy配列とDask配列のどちらを使うべきか？
A: 遅延評価のため、Dask配列を使用してください。

### Q: カバレッジ100%が難しい場合は？
A: 以下の場合は例外的に許容されます：
- プラットフォーム依存のコード（# pragma: no cover）
- デバッグ用のコード
- 型チェック用のコード（if TYPE_CHECKING:）

ただし、これらは最小限に抑えてください。

### Q: 破壊的変更が必要な場合は？
A: 以下の手順を踏んでください：
1. 非推奨化（Deprecation）警告を出す（最低1バージョン）
2. ドキュメントに移行ガイドを記載
3. CHANGELOGに詳細を記載
4. メジャーバージョンアップ時に削除

### Q: パフォーマンステストはどう書くべきか？
A: pytest-benchmarkを使用してください：
```python
def test_fft_performance(benchmark, sample_signal):
    """Test FFT performance."""
    result = benchmark(sample_signal.fft)
    assert result is not None
```

## トラブルシューティング

### 型チェックエラーが解決できない
1. `reveal_type()` を使って型を確認
2. `cast()` を使って明示的に型を指定（最終手段）
3. 必要に応じて `# type: ignore` を使用（理由をコメントで記載）

### テストが不安定
1. 乱数のシードを固定（`np.random.seed(42)`）
2. 浮動小数点数の比較には `np.allclose()` を使用
3. テストの独立性を確認（他のテストに依存していないか）

### カバレッジが上がらない
1. `pytest --cov-report=html` でHTMLレポートを確認
2. カバーされていない行を特定
3. その行を実行するテストケースを追加

## 参考資料
- **プロジェクトドキュメント**: https://kasahart.github.io/wandas/
- **リポジトリ**: https://github.com/kasahart/wandas
- **Issue Tracker**: https://github.com/kasahart/wandas/issues
- **NumPy Docstring Guide**: https://numpydoc.readthedocs.io/
- **Python Type Hints**: https://docs.python.org/3/library/typing.html
- **pytest Documentation**: https://docs.pytest.org/

## コントリビューション

このガイドラインは進化し続けます。改善提案がある場合は：
1. Issueで議論を開始
2. このガイドライン自体の変更プラン（PLAN_update_guidelines.md）を作成
3. プルリクエストを送信

---
