# メタデータ更新のカプセル化パターン

**最終更新**: 2025年10月

## 概要

Operation層がメタデータ更新の責任を持つ設計パターン。

## 設計原則

### 問題

Mixin層が直接`result.sampling_rate`を変更していた:

```python
# ❌ 問題のあったコード
def loudness_zwtv(...) -> T_Processing:
    result = self.apply_operation("loudness_zwtv", ...)
    result.sampling_rate = 500.0  # Mixin層でビジネスロジック
    return result
```

**問題点**:

- カプセル化違反
- 責任の所在が不明確
- テストが困難
- 拡張性が低い

### 解決策

Operation層が`get_metadata_updates()`でメタデータ更新を返す:

```python
# ✅ 改善後のコード
class LoudnessZwtv(AudioOperation):
    def get_metadata_updates(self) -> dict[str, Any]:
        """Zwicker method uses 2ms time steps → 500 Hz"""
        return {"sampling_rate": 500.0}
```

## アーキテクチャ

### Before: Mixin層がメタデータを直接変更

```text
Mixin層 → 直接sampling_rate変更 ❌
```

### After: Operation層が責任を持つ

```text
Mixin層 → Framework層 → Operation.get_metadata_updates() ✅
```

## 実装パターン

### 基底クラス

```python
# wandas/processing/base.py
class AudioOperation:
    def get_metadata_updates(self) -> dict[str, Any]:
        """
        Get metadata updates to apply after processing.

        Returns
        -------
        dict
            Dictionary of metadata updates.

        Notes
        -----
        Design principle: Operations should use parameters provided at
        initialization (via __init__). All necessary information should be
        available as instance variables.
        """
        return {}
```

### 具体的な実装例

```python
# 例1: 固定値を返す
class LoudnessZwtv(AudioOperation):
    def get_metadata_updates(self) -> dict[str, Any]:
        return {"sampling_rate": 500.0}

# 例2: インスタンス変数から計算
class RmsTrend(AudioOperation):
    def __init__(self, sampling_rate: float, hop_length: int):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

    def get_metadata_updates(self) -> dict[str, Any]:
        new_sr = self.sampling_rate / self.hop_length
        return {"sampling_rate": new_sr}

# 例3: メタデータ更新なし
class LowPassFilter(AudioOperation):
    def get_metadata_updates(self) -> dict[str, Any]:
        return {}  # サンプリングレートは変わらない
```

### Framework層の実装

```python
# wandas/frames/channel.py
def _apply_operation_impl(
    self,
    operation: AudioOperation,
    **kwargs: Any,
) -> Self:
    # 処理実行
    result_data = operation.process(self.data)

    # メタデータ更新を取得して適用
    metadata_updates = operation.get_metadata_updates()
    for key, value in metadata_updates.items():
        setattr(result, key, value)

    return result
```

## YAGNI原則の適用

### 設計の洗練過程

**Phase 1**: `get_metadata_updates(input_metadata)`として実装

```python
def get_metadata_updates(self, input_metadata: dict[str, Any]) -> dict[str, Any]:
    original_sr = input_metadata.get("sampling_rate")
    return {"sampling_rate": original_sr / self.hop_length}
```

**発見**: どのOperationも`input_metadata`を実際には使用していなかった。
必要な情報はすべて`__init__`で受け取っている。

**Phase 2**: YAGNI原則を適用し、不要なパラメータを削除

```python
def get_metadata_updates(self) -> dict[str, Any]:
    # self.sampling_rate を使用（input_metadataは不要）
    return {"sampling_rate": self.sampling_rate / self.hop_length}
```

### 学んだこと

- **使われていない機能は削除する**: 「将来使うかもしれない」は実装の理由にならない
- **必要になったときに追加する**: 後方互換性を保ちながら拡張可能
- **自己完結したオブジェクト**: 必要な情報は`__init__`で受け取る

## 拡張例

### チャンネル数の変更

```python
class MonoConverter(AudioOperation):
    def get_metadata_updates(self) -> dict[str, Any]:
        return {
            "n_channels": 1,
            "channel_names": ["Mono"]
        }
```

### 周波数領域への変換

```python
class FFTOperation(AudioOperation):
    def get_metadata_updates(self) -> dict[str, Any]:
        return {
            "domain": "frequency",
            "n_fft": self.n_fft
        }
```

### 複数のメタデータ更新

```python
class ComplexOperation(AudioOperation):
    def get_metadata_updates(self) -> dict[str, Any]:
        return {
            "sampling_rate": self.new_sr,
            "n_channels": self.output_channels,
            "unit": "Pa",
        }
```

## テスト戦略

### Unit Test（Operation単体）

```python
def test_operation_metadata_updates(self) -> None:
    """Operation が正しいメタデータ更新を返すことを確認"""
    operation = MyOperation(sampling_rate=44100, param=10)
    updates = operation.get_metadata_updates()
    assert updates["sampling_rate"] == expected_value
```

### Integration Test（フレームワーク統合）

```python
def test_framework_applies_metadata_updates(self) -> None:
    """Framework がメタデータ更新を正しく適用することを確認"""
    result = self.frame.my_operation(param=10)
    assert result.sampling_rate == expected_value
```

## メリット

### カプセル化の改善

- ✅ サンプリングレート計算ロジックが各Operationに集約
- ✅ Mixin層からビジネスロジックが排除
- ✅ 責任の所在が明確

### 設計原則の強化

- ✅ 単一責任の原則
- ✅ 開放閉鎖の原則
- ✅ 依存性逆転の原則
- ✅ YAGNI原則

### 拡張性の向上

- ✅ 新しい操作を追加してもフレームワーク層は変更不要
- ✅ メタデータ更新が必要なら`get_metadata_updates()`をオーバーライドするだけ
- ✅ 一貫性のあるパターン

### 保守性の向上

- ✅ テストが容易
- ✅ コードの意図が明確
- ✅ 変更の影響範囲が限定的

## 新しいOperationの実装ガイド

```python
from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal
from typing import Any

@register_operation
class MyOperation(AudioOperation[NDArrayReal, NDArrayReal]):
    """
    Description of the operation.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the input signal.
    my_param : float
        Description of my_param.
    """

    name = "my_operation"

    def __init__(self, sampling_rate: float, my_param: float) -> None:
        self.sampling_rate = sampling_rate
        self.my_param = my_param
        super().__init__(sampling_rate, my_param=my_param)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Process the input array."""
        # 処理ロジック
        result = ...
        return result

    def get_metadata_updates(self) -> dict[str, Any]:
        """
        Get metadata updates (override only if needed).

        Returns
        -------
        dict
            Dictionary of metadata updates.
        """
        # メタデータを変更する場合のみオーバーライド
        return {
            "sampling_rate": self.new_sampling_rate,
        }
```

## 関連ドキュメント

- プロジェクトガイドライン: `.github/copilot-instructions.md`
- APIドキュメント: `docs/src/api/`
- 使用例: `examples/`
