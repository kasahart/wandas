# Wandas 実装レビューと改善提案

## 概要

Wandas のコードベースを包括的に分析し、アーキテクチャ、実装パターン、品質、そして改善が必要な領域について詳細にレビューしました。

**日付**: 2026-02-25
**バージョン**: 0.1.11
**解析対象ファイル数**: 34 ソースファイル、35 テストファイル

---

## 1. プロジェクト概要

### 1.1 基本情報

- **目的**: pandas ライクなデータ構造を用いた効率的な信号解析ライブラリ
- **主な機能**:
  - 時間領域・周波数領域・時間 - 周波数領域のデータ処理
  - フィルタリング、FFT、STFT、Welch 法などの信号処理
  - ラウドネス、粗さ、鮮明度などの心理音響指標計算
  - Dask を用いた遅延評価による大規模データ処理

### 1.2 プロジェクト構造

```
wandas/
├── core/                    # コア機能
│   ├── base_frame.py        # BaseFrame クラス (1058 行)
│   └── metadata.py          # ChannelMetadata クラス
├── frames/                  # データフレーム実装
│   ├── channel.py           # ChannelFrame (1327 行)
│   ├── spectrogram.py       # SpectrogramFrame
│   ├── spectral.py          # SpectralFrame
│   ├── noct.py              # NOctFrame
│   ├── roughness.py         # RoughnessFrame
│   └── mixins/              # ミックスインクラス群
├── processing/              # 信号処理ロジック
│   ├── base.py              # AudioOperation ベースクラス
│   ├── filters.py           # フィルタ実装
│   ├── spectral.py          # 周波数解析
│   ├── temporal.py          # 時間領域処理
│   ├── psychoacoustic.py    # 心理音響指標
│   ├── effects.py           # エフェクト処理
│   └── stats.py             # 統計処理
├── io/                      # I/O 機能
│   ├── readers.py           # ファイルリーダー
│   ├── wav_io.py            # WAV入出力
│   └── wdf_io.py            # WDF(HDF5) 入出力
├── visualization/           # 可視化
│   ├── plotting.py          # プロット戦略 (781 行)
│   └── types.py             # 型定義
└── utils/                   # ユーティリティ
    ├── introspection.py     # シグネチャ分析
    ├── dask_helpers.py      # Dask ヘルパー
    └── frame_dataset.py     # データセット操作
```

---

## 2. アーキテクチャ評価

### 2.1 強み

#### 2.1.1 優れた設計パターン

1. **フレーム不変性 (Immutability)**
   - 全処理で新しいフレームオブジェクトを返す設計
   - `operation_history` で完全なトレーサビリティ確保
   - [base_frame.py:620-654](wandas/core/base_frame.py#L620-L654) の `_create_new_instance` で統一管理

2. **遅延評価 (Lazy Evaluation)**
   - Dask を活用した効率的な大規模データ処理
   - 計算グラフの可視化機能 `visualize_graph()` ([base_frame.py:663-715](wandas/core/base_frame.py#L663-L715))

3. **戦略パターン (Strategy Pattern)**
   - プロット戦略：`PlotStrategy` クラス階層
   - 処理操作：`AudioOperation` クラス階層
   - 拡張性の高いプラグイン型アーキテクチャ

4. **型安全な設計**
   - Python 3.10+ の型ヒントを積極採用
   - Protocol を用いたダックタイピング ([protocols.py](wandas/frames/mixins/protocols.py))

### 2.2 メリットのある実装詳細

#### 2.2.1 メタデータ管理

```python
# ChannelMetadata クラスは pydantic を活用
class ChannelMetadata(BaseModel):
    label: str = ""
    unit: str = ""
    ref: float = 1.0
    extra: dict[str, Any] = Field(default_factory=dict)
```

- [metadata.py:8-85](wandas/core/metadata.py#L8-L85): pydantic でバリデーション、JSON シリアライズ対応

#### 2.2.2 演算登録システム

```python
_OPERATION_REGISTRY: dict[str, type[AudioOperation]] = {}

def register_operation(operation_class: type) -> None:
    _OPERATION_REGISTRY[operation_class.name] = operation_class
```

- [base.py:263-287](wandas/processing/base.py#L263-L287): 自動登録、型安全なファクトリパターン

#### 2.2.3 引数フィルタリングユーティリティ

```python
def filter_kwargs(func, kwargs, *, strict_mode=False) -> dict[str, Any]:
    """シグネチャ introspection で関数に渡せる引数を自動抽出"""
```

- [introspection.py:46-87](wandas/utils/introspection.py#L46-L87): matplotlib 等の複雑な API への柔軟対応

---

## 3. 改善提案

### 3.1 型システム関連

#### 3.1.1 Protocol の強化（重要度：高）

**現状**: `ProcessingFrameProtocol` がほぼ空のプレースホルダー

```python
@runtime_checkable
class ProcessingFrameProtocol(BaseFrameProtocol, Protocol):
    """Protocol that defines operations related to signal processing."""
    pass
```

- [protocols.py:79-86](wandas/frames/mixins/protocols.py#L79-L86)

**問題点**:

- Protocol が実際のメソッドシグネチャを定義していない
- 型チェックの恩恵が限定的

**改善提案**:

```python
@runtime_checkable
class ProcessingFrameProtocol(BaseFrameProtocol, Protocol):
    """Protocol that defines operations related to signal processing."""

    def high_pass_filter(self, cutoff: float, order: int = 4) -> "ProcessingFrameProtocol": ...
    def low_pass_filter(self, cutoff: float, order: int = 4) -> "ProcessingFrameProtocol": ...
    def normalize(self, norm: float | None = float("inf")) -> "ProcessingFrameProtocol": ...
    # 他の処理メソッドも追加

T_Processing = TypeVar("T_Processing", bound=ProcessingFrameProtocol)
```

#### 3.1.2 戻り値型の厳密化（重要度：中）

**現状**: 多くのメソッドで `Any` や過度に一般的な型を使用

```python
# Example: base_frame.py
def _create_new_instance(self, data: DaArray, **kwargs: Any) -> S:
    # ... 実装
```

**改善提案**:

- `TypeVar` のバウンドをより具体的に
- Generics を活用した型推論の向上

### 3.2 コード品質関連

#### 3.2.1 ドキュメント戦略（重要度：高）

**現状**:

- docstring は充実しているが、日本語/英語が混在
- 例え話が豊富な一方、重要な情報が埋もれている場合がある

**改善提案**:

```python
def normalize(
    self,
    norm: float | None = float("inf"),
    axis: int | None = -1,
    threshold: float | None = None,
    fill: bool | None = None,
) -> T_Processing:
    """Normalize signal levels using librosa.util.normalize.

    Parameters
    ----------
    norm : float or None
        Norm type for normalization.
        - np.inf: Peak normalization (default)
        - 2.0: L2 norm
        - None: No normalization

    Returns
    -------
    T_Processing
        New frame with normalized signal

    Examples
    --------
    >>> signal = wd.read_wav("audio.wav")
    >>> normalized = signal.normalize()  # Peak normalization
    """
```

#### 3.2.2 エラーメッセージの統一（重要度：中）

**現状**: エラーメッセージは多岐にわたり、パターンが一定でない

**改善提案**: コントリビューターガイドラインとして WHAT/WHY/HOW パターンを厳格に:

```python
raise ValueError(
    f"{error_type} (WHAT)\n"
    f"  Got: {actual_value}\n"
    f"  Expected: {expected_value}\n"
    f"Why this matters: {impact_description}\n"
    f"How to fix: {solution_hint}"
)
```

### 3.3 パフォーマンス関連

#### 3.3.1 Dask チャンキングの最適化（重要度：中）

**現状**: [base_frame.py:99-121](wandas/core/base_frame.py#L99-L121) で自動リチャンク

```python
# Default rechunk: prefer channel-wise chunking
chunks = tuple([1] + [-1] * (normalized.ndim - 1))
```

**改善点**:

- ユーザーが使用パターンに応じて最適化可能にするオプションを追加
- メモリ使用量と並列性のバランスを調整できる API

#### 3.3.2 中間計算のキャッシュ（重要度：低）

**現状**: 同じ操作の繰り返しで不要な計算が発生する可能性

**改善提案**:

```python
class ChannelFrame:
    @functools.cached_property
    def _cached_fft(self) -> SpectralFrame:
        """Cache frequently computed transforms"""
        ...
```

### 3.4 API 設計関連

#### 3.4.1 メソッド名の統一（重要度：中）

**現状**:

- `low_pass_filter` (snake_case)
- `rms_trend` (snake_case)
- `hpss_harmonic` (snake_case)
- `a_weighting` vs `A_weighting`

**改善提案**: 一貫した命名規則の確立と移行パス:

```python
# 統一例
def low_pass_filter() -> ...      # ✓ 現在のままで OK
def rms_trend() -> ...             # ✓ 現在のままで OK
def a_weighting() -> ...           # ✓ 現在のままで OK
```

#### 3.4.2 チェイン対応の改善（重要度：低）

**現状**: 多くのメソッドでチェイン可能だが、戻り値型が必ずしも一致しない場合がある

**改善提案**: `Self` type の活用 (Python 3.11+):

```python
from typing import Self

def normalize(self, norm: float | None = None) -> Self:
    return self.apply_operation("normalize", norm=norm)
```

### 3.5 テスト関連

#### 3.5.1 カバレッジ拡大（重要度：高）

**現状**:

- テストファイルは 35 ファイル存在する
- 主要モジュールのテストはあるが、エッジケースの網羅性は不明

**改善提案**:

```python
# test_coverage_plan.md の作成:
# - Dask lazy evaluation の検証テスト
# - メタデータ継承の完全性テスト
# - 多チャンネル処理のエッジケース
# - 数値安定性の境界値テスト
```

#### 3.5.2 プロパティベーステスト（重要度：中）

**提案**: Hypothesis 等の導入:

```python
from hypothesis import given, strategies as st

@given(
    data=st.arrays(dtype=float, shape=st.tuples(st.integers(1, 10), st.integers(100, 1000))),
    sr=st.floats(min_value=8000, max_value=96000)
)
def test_normalize_preserves_shape(data, sr):
    frame = ChannelFrame.from_numpy(data, sr)
    normalized = frame.normalize()
    assert normalized.shape == frame.shape
```

### 3.6 ドキュメント関連

#### 3.6.1 Quick Start の拡充（重要度：高）

**現状**: README に基本的な使用例がある

**改善提案**: より多様なユースケース:

```python
# example_multichannel.py
import wandas as wd

# 複数チャンネルの読み込みと選択
cf = wd.read_wav("stereo.wav")
left_only = cf.get_channel(0)
right_only = cf.get_channel(1)

# クエリによるチャンネル選択
cf_with_labels = cf.add_channel(sine_wave, label="reference_440hz")
refs = cf_with_labels.get_channel(query=lambda ch: "ref" in ch.label)

# 心理音響解析
loudness = cf.loudness_zwtv(field_type="free")
roughness = cf.roughness_dw(overlap=0.5)

# 結果の保存
cf.save("processed.wdf")  # メタデータ完全保存
loaded = wd.ChannelFrame.load("processed.wdf")
```

#### 3.6.2 API リファレンスの生成（重要度：中）

**提案**: mkdocstrings を活用した自動生成:

```yaml
# mkdocs.yml に追加
plugins:
  - mkdocstrings:
      handlers:
        python:
          paths: [wandas]
          options:
            show_root_heading: true
            show_source: false
```

### 3.7 エラーハンドリング関連

#### 3.7.1 カスタム例外の追加（重要度：中）

**現状**: 標準的な Exception が多用されている

**改善提案**: ドメイン固有例外クラス:

```python
class WandasError(Exception):
    """Base exception for wandas"""
    pass

class FrameTypeError(WandasError):
    """Invalid frame type operation"""
    pass

class MetadataError(WandasError):
    """Metadata validation failed"""
    pass

class DaskGraphError(WandasError):
    """Dask computation graph error"""
    pass
```

#### 3.7.2 リカバリ機能の追加（重要度：低）

**提案**: エラーから自動的に回復するオプション:

```python
def low_pass_filter(self, cutoff: float, order: int = 4,
                    on_error: str = "raise") -> Self:
    """
    Parameters
    ----------
    on_error : {'raise', 'warn', 'skip'}
        How to handle errors. 'warn' logs warning and returns self.
    """
    try:
        return self.apply_operation("lowpass_filter", cutoff=cutoff, order=order)
    except ValueError as e:
        if on_error == "warn":
            warnings.warn(f"Filter failed: {e}, returning original frame")
            return self
        elif on_error == "skip":
            return self
        raise
```

---

## 4. 具体的な実装改善案

### 4.1 Protocol の完全な実装（優先度：高）

**対象ファイル**: `wandas/frames/mixins/protocols.py`

**現状**:

```python
@runtime_checkable
class ProcessingFrameProtocol(BaseFrameProtocol, Protocol):
    """Protocol that defines operations related to signal processing."""
    pass
```

**改善案**:

```python
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable
import numpy.typing as npt

T_Processing = TypeVar("T_Processing", bound="ProcessingFrameProtocol")

@runtime_checkable
class ProcessingFrameProtocol(BaseFrameProtocol, Protocol):
    """Protocol defining the interface for processing frames."""

    # 基本処理メソッド
    def normalize(
        self: T_Processing,
        norm: float | None = float("inf"),
        axis: int | None = -1,
    ) -> T_Processing: ...

    def high_pass_filter(
        self: T_Processing,
        cutoff: float,
        order: int = 4
    ) -> T_Processing: ...

    def low_pass_filter(
        self: T_Processing,
        cutoff: float,
        order: int = 4
    ) -> T_Processing: ...

    def band_pass_filter(
        self: T_Processing,
        low_cutoff: float,
        high_cutoff: float,
        order: int = 4,
    ) -> T_Processing: ...

    # 心理音響メソッド
    def loudness_zwtv(
        self: T_Processing,
        field_type: str = "free",
    ) -> T_Processing: ...

    def loudness_zwst(self, field_type: str = "free") -> npt.NDArray[Any]: ...

    # 演算メソッド
    def __add__(self: T_Processing, other: Any) -> T_Processing: ...
    def __sub__(self: T_Processing, other: Any) -> T_Processing: ...
    def __mul__(self: T_Processing, other: Any) -> T_Processing: ...

    # プロパティ
    @property
    def rms(self) -> npt.NDArray[Any]: ...
    @property
    def time(self) -> npt.NDArray[Any]: ...
```

### 4.2 エラーハンドリングの改善（優先度：中）

**対象ファイル**: `wandas/core/base_frame.py`

**現状のエラー例**:

```python
if channel is None:
    raise TypeError("Either 'channel_idx' or 'query' must be provided.")
```

**改善案**:

```python
class FrameQueryError(TypeError):
    """Raised when frame query parameters are invalid."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{super().__str__()}\nHint: {self.hint}"
        return super().__str__()

def get_channel(
    self,
    channel_idx: int | list[int] | None = None,
    query: QueryType | None = None,
) -> Self:
    if channel_idx is None and query is None:
        raise FrameQueryError(
            "Either 'channel_idx' or 'query' must be provided",
            "Use get_channel(0) for first channel, or "
            "get_channel(query='ch0') for label-based selection"
        )
    # ... rest of implementation
```

### 4.3 ドキュメントの改善（優先度：高）

**対象ファイル**: `wandas/processing/base.py`

**改善案**:

```python
class AudioOperation(Generic[InputArrayType, OutputArrayType]):
    """Base class for all audio processing operations.

    All processing operations in wandas inherit from this class. It provides:

    - Lazy evaluation via Dask
    - Automatic metadata management
    - Operation history tracking
    - Consistent error handling

    To create a new operation:

    1. Subclass AudioOperation
    2. Set the `name` class variable
    3. Implement `_process_array()` with actual processing logic
    4. Optionally implement `calculate_output_shape()` for efficiency

    Args:
        sampling_rate: float
            Sampling rate in Hz. Provided automatically by the frame.
        pure: bool, default=True
            If True, Dask will cache results for identical inputs (recommended).
            Set False only for operations with side effects.
        **params: Any
            Operation-specific parameters stored for metadata tracking.

    Attributes:
        name: str
            Unique identifier for this operation type.
        sampling_rate: float
            The sampling rate used for processing.
        params: dict
            Dictionary of operation-specific parameters.
        history: list
            List of applied operations for reproducibility.

    Examples:
        Create a custom filter operation:

        >>> from wandas.processing.base import AudioOperation, register_operation
        >>>
        >>> class MyFilter(AudioOperation):
        ...     name = "my_filter"
        ...
        ...     def __init__(self, sampling_rate: float, cutoff: float):
        ...         self.cutoff = cutoff
        ...         super().__init__(sampling_rate, cutoff=cutoff)
        ...
        ...     def _process_array(self, x):
        ...         # Apply filter logic here
        ...         return filtered_x
        >>>
        >>> register_operation(MyFilter)  # Auto-register the operation

        Use the registered operation:

        >>> frame = ChannelFrame(data, sampling_rate=44100)
        >>> processed = frame.apply("my_filter", cutoff=2000)

    See Also:
        register_operation : Register a new operation type for global access.
        create_operation : Create an operation instance from name and params.
    """
```

---

## 5. リファクタリング提案

### 5.1 コード重複の除去（重要度：中）

**現状**: 類似の実装が複数箇所に存在

**例**: ChannelFrame と各 Frame クラスで `_binary_op` が個別に実装されている

**改善案**:

```python
# core/base_frame.py に統一実装を移動
class BaseFrame(ABC, Generic[T]):
    def _generic_binary_op(
        self: S,
        other: S | int | float | NDArrayReal | DaArray,
        op: Callable[[DaArray, Any], DaArray],
        symbol: str,
        frame_class: type[S],
    ) -> S:
        """Generic binary operation implementation

        All frames can delegate to this method for common operations.
        """
        # Common logic for handling other types, metadata updates, etc.
        pass

# Frame クラスでは必要最小限のオーバーライドのみ
class ChannelFrame(BaseFrame):
    def _binary_op(self, other, op, symbol) -> "ChannelFrame":
        return self._generic_binary_op(
            other, op, symbol, frame_class=ChannelFrame
        )
```

### 5.2 型定義の一元化（重要度：中）

**現状**: 型エイリアスが複数箇所に散在

**改善案**: `wandas/utils/types.py` に集約:

```python
# wandas/utils/types.py
from typing import TypeAlias
import numpy as np
import numpy.typing as npt

NDArrayReal: TypeAlias = npt.NDArray[np.float64 | np.float32]
NDArrayComplex: TypeAlias = npt.NDArray[np.complex128 | np.complex64]
FrameData: TypeAlias = NDArrayReal | NDArrayComplex
ProcessingConfig: TypeAlias = dict[str, int | float | str | bool]

# 共通の Exception クラス群
class WandasException(Exception): ...
class ValidationError(WandasException): ...
class ConfigurationError(WandasException): ...
```

---

## 6. チェックリスト

### 6.1 リリース前の確認項目

- [ ] Protocol の完全な実装完了か？
- [ ] エラーメッセージの統一化完了か？
- [ ] docstring の英語/日本語統一完了か？
- [ ] テストカバレッジ 80% 以上達成か？
- [ ] プロパティベーステストの実装完了か？
- [ ] パフォーマンスプロファイル作成完了か？

### 6.2 コードレビューチェックリスト

- [ ] Frame の不変性遵守されているか？
- [ ] `operation_history` が正しく更新されているか？
- [ ] メタデータが適切に継承されているか？
- [ ] Lazy evaluation が維持されているか？
- [ ] 型ヒントが正確か？
- [ ] docstring の Google スタイル準拠か？

---

## 7. まとめ

### 7.1 優れた点

1. **アーキテクチャ**: フレーム不変性、遅延評価、戦略パターンなど、現代的な設計パターンの適切な適用
2. **拡張性**: Plugin アーキテクチャによる容易な機能追加
3. **型安全性**: Python の型システムを積極的に活用
4. **ドキュメント**: 詳細な docstring と使用例の充実

### 7.2 優先的に改善すべき領域

| 優先度 | 領域 | 具体的アクション |
|--------|------|-----------------|
| 高 | Protocol の完全実装 | メソッドシグネチャを追加し型推論を強化 |
| 高 | テストカバレッジ | プロパティベーステスト、境界値テスト追加 |
| 中 | エラーハンドリング | カスタム例外クラス、統一メッセージ形式 |
| 中 | API ドキュメント | ユースケース例の拡充、リファレンス自動生成 |
| 低 | パフォーマンス最適化 | キャッシュ機能、チャンキングオプション |

### 7.3 次期リリース推奨事項

1. **v0.2.0**: Protocol の完全実装、テストカバレッジ向上
2. **v0.3.0**: エラーハンドリング統一、カスタム例外追加
3. **v1.0.0**: API安定性保証、公式リリース

---

## 付録

### A. 引用元ファイル一覧

- `wandas/core/base_frame.py` (1058 行)
- `wandas/core/metadata.py` (86 行)
- `wandas/frames/channel.py` (1327 行)
- `wandas/frames/mixins/protocols.py` (110 行)
- `wandas/processing/base.py` (288 行)
- `wandas/processing/filters.py` (多数)
- `wandas/io/wdf_io.py` (258 行)
- `wandas/visualization/plotting.py` (781 行)
- `wandas/utils/introspection.py` (88 行)

### B. 参考文献

- [copilot-instructions.md](.github/copilot-instructions.md): プロジェクト固有の設計指針
- [README.md](README.md): ユーザー向けドキュメント
- [pyproject.toml](pyproject.toml): 依存関係とツール設定

---

**レポート作成日**: 2026-02-25
**レビュー担当者**: AI Code Assistant
