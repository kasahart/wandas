# API改善ガイド

**最終更新**: 2025年10月

## 概要

wandas APIの使いやすさ向上のための改善パターン集。

## describe()メソッドの改善

### 問題

`**kwargs`で隠されていたパラメータはIDE補完が効かず、型安全性に欠けていた。

### 解決策

頻繁に使用されるパラメータを明示的な引数として定義:

```python
def describe(
    self,
    normalize: bool = True,
    is_close: bool = True,
    *,  # 以降はキーワード専用引数
    fmin: float = 0,
    fmax: Optional[float] = None,
    cmap: str = "jet",
    Aw: bool = False,
    **kwargs: Any,
) -> None:
    """
    Display signal visualization with waveform and spectrum.

    Parameters
    ----------
    normalize : bool, default=True
        Normalize signal amplitude.
    is_close : bool, default=True
        Close plot after display.
    fmin : float, default=0
        Minimum frequency for spectrum display (Hz).
    fmax : float, optional
        Maximum frequency for spectrum display (Hz).
    cmap : str, default="jet"
        Colormap for spectrum visualization.
    Aw : bool, default=False
        Apply A-weighting to spectrum.
    **kwargs : Any
        Additional matplotlib parameters.
    """
```

### TypedDict定義

設定の再利用のため`wandas.visualization.types`にTypedDict型を定義:

```python
from typing import TypedDict, Any, Optional

class WaveformConfig(TypedDict, total=False):
    """Waveform plot configuration."""
    title: str
    xlabel: str
    ylabel: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]

class SpectralConfig(TypedDict, total=False):
    """Spectral plot configuration."""
    fmin: float
    fmax: float
    cmap: str
    vmin: float
    vmax: float
    Aw: bool

class DescribeParams(TypedDict, total=False):
    """All parameters for describe method."""
    normalize: bool
    is_close: bool
    fmin: float
    fmax: Optional[float]
    cmap: str
    Aw: bool
    waveform: WaveformConfig
    spectral: SpectralConfig
```

### 使用例

```python
import wandas as wd

# 基本的な使い方
signal = wd.read_wav("audio.wav")
signal.describe(fmin=100, fmax=5000, Aw=True)

# TypedDict使用（設定の再利用）
from wandas.visualization.types import DescribeParams

config: DescribeParams = {
    "fmin": 100,
    "fmax": 5000,
    "Aw": True,
}
signal.describe(**config)
```

## plot()メソッドの改善

頻繁に使用される引数を明示化:

```python
def plot(
    self,
    ax: Optional[Axes] = None,
    *,
    title: Optional[str] = None,
    xlabel: str = "Time [s]",
    ylabel: str = "Amplitude",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    alpha: float = 1.0,
    overlay: bool = False,
    **kwargs: Any
) -> Axes:
    """
    Plot signal waveform.

    Parameters
    ----------
    ax : Axes, optional
        Matplotlib axes to plot on.
    title : str, optional
        Plot title.
    xlabel : str, default="Time [s]"
        X-axis label.
    ylabel : str, default="Amplitude"
        Y-axis label.
    xlim : tuple[float, float], optional
        X-axis limits.
    ylim : tuple[float, float], optional
        Y-axis limits.
    alpha : float, default=1.0
        Line transparency (0.0-1.0).
    overlay : bool, default=False
        Overlay multiple channels.
    **kwargs : Any
        Additional matplotlib parameters.

    Returns
    -------
    Axes
        The matplotlib axes object.
    """
```

## メリット

### IDE補完

- ✅ パラメータ名が補完される
- ✅ 型情報が表示される
- ✅ ドキュメントがポップアップで表示される

### 型安全性

- ✅ mypyで型チェック可能
- ✅ 誤った引数の使用を事前に検出

### 後方互換性

- ✅ `**kwargs`は残しているため既存コードは動作し続ける
- ✅ 段階的な移行が可能

## ベストプラクティス

### 1. パラメータの明示化基準

**明示化すべきパラメータ**:

- 頻繁に使用される（使用頻度 > 30%）
- ドメイン固有（信号処理特有の概念）
- 型が複雑（辞書、タプルなど）

**kwargsに残すべきパラメータ**:

- 稀にしか使用されない
- matplotlib等の下位ライブラリのパラメータ
- 高度なカスタマイズ用

### 2. TypedDict活用

設定の再利用や型安全性が必要な場合はTypedDictを定義:

```python
# 定義
class FilterConfig(TypedDict, total=False):
    cutoff: float
    order: int
    filter_type: str

# 使用
config: FilterConfig = {
    "cutoff": 1000.0,
    "order": 5,
    "filter_type": "butterworth",
}
signal.low_pass_filter(**config)
```

### 3. キーワード専用引数

混乱を避けるため、オプション引数は`*`の後に配置:

```python
def process(
    self,
    required_param: float,  # 位置引数
    *,
    optional_param: int = 10,  # キーワード専用引数
    **kwargs: Any,
) -> None:
    """Must call as: process(1.0, optional_param=20)"""
```

## 関連ドキュメント

- TypedDict活用ガイド: `docs/src/improvements/typeddict_usage_guide.md`
- 詳細説明: `docs/src/improvements/describe_parameters_visibility.md`
- APIリファレンス: `docs/src/api/`
