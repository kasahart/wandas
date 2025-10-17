# Plan: Add Loudness Calculation using MoSQITo

## 変更の目的と背景

MoSQIToライブラリを使用して、音響信号のラウドネス（Loudness）計算機能をWandasに統合する。
ラウドネスは音の大きさの知覚的な尺度であり、ISO 532-1:2017に基づくZwickerの手法で計算される。

参考: https://mosqito.readthedocs.io/en/latest/source/user_guide/scope.html#sq-metrics

## 影響を受けるファイルとモジュール

### 新規作成
- `wandas/processing/psychoacoustics.py` - ラウドネス計算のための新しいモジュール
- `tests/processing/test_psychoacoustics.py` - ラウドネス計算のテスト

### 変更
- `wandas/processing/__init__.py` - 新しいLoudness操作をエクスポートに追加
- `wandas/frames/mixins/channel_processing.py` - ChannelFrameに`.loudness()`メソッドを追加（必要な場合）

## 実装方針と技術的な詳細

### 1. Loudness操作クラスの実装
MoSQIToの2つの主要なラウドネス関数を統合:
- `loudness_zwtv` - 時間変化するラウドネス（Time-varying）
- `loudness_zwst` - 定常状態のラウドネス（Stationary）

### 2. AudioOperationの継承
`wandas.processing.base.AudioOperation`を継承して実装:
- 入力: `NDArrayReal` (時間領域の信号、単位: Pa)
- 出力: 辞書形式（ラウドネス値、特定ラウドネス、時間軸、周波数軸）

### 3. API設計
```python
from wandas.processing.psychoacoustics import LoudnessZwtv, LoudnessZwst

# Time-varying loudness
loudness_op = LoudnessZwtv(sampling_rate=44100, field_type="free")
result = loudness_op.process(signal_data)

# Stationary loudness  
loudness_st_op = LoudnessZwst(sampling_rate=44100, field_type="free")
result = loudness_st_op.process(signal_data)
```

### 4. 戻り値の形式
MoSQIToの戻り値を保持しつつ、Wandasの規約に準拠:
- `loudness_zwtv` returns: (N, N_spec, bark_axis, time_axis)
- `loudness_zwst` returns: (N, N_spec, bark_axis)

処理結果は辞書として返し、後続処理で利用可能にする。

## テスト戦略

### 1. 単体テスト
- 初期化テスト（パラメータ検証）
- MoSQIToとの比較テスト（同じ入力で同じ出力を得る）
- 異なるfield_type（"free", "diffuse"）での動作確認
- エラーハンドリング（不正な入力、サンプリングレート）

### 2. 統合テスト
- ChannelFrameからの呼び出し
- 処理履歴の記録確認
- メタデータの保持確認

### 3. MoSQIToのテストケースとの比較
MoSQIToのリポジトリにあるテストケースを参考にして、同等の結果が得られることを検証:
- ISO 532-1 Annex B4, B5のテスト信号
- 理論値との比較（許容誤差内での一致）

## 想定されるリスクと対応策

### リスク1: MoSQIToの入力形式
MoSQIToは単位がPa（パスカル）を想定している可能性がある。
- 対応: ドキュメントに単位を明記し、必要に応じて変換関数を提供

### リスク2: マルチチャンネル対応
MoSQIToがモノラル信号のみ対応の可能性。
- 対応: 各チャンネルを個別に処理し、結果を統合

### リスク3: 大規模データ処理
ラウドネス計算は計算量が多い可能性がある。
- 対応: Daskの遅延評価を活用し、必要に応じてチャンク処理

## 後方互換性

新機能の追加であり、既存コードには影響しない。

## ドキュメント更新計画

### 1. docstring
- 英語で記述
- NumPy形式
- パラメータ、戻り値、Raises、Examplesを含む
- ISO 532-1への参照を含める

### 2. README更新
- Featuresセクションに「Psychoacoustic Metrics (Loudness)」を追加

### 3. チュートリアル（オプション）
- 将来的にラウドネス計算のチュートリアルを追加

## 実装手順

1. ✅ MoSQIToのAPIを調査・理解
2. ⬜ `psychoacoustics.py`に基本的なLoudness操作クラスを実装
3. ⬜ テストファイルを作成し、MoSQIToとの比較テストを実装
4. ⬜ 型チェックとリント
5. ⬜ ドキュメント更新
6. ⬜ 最終的な統合テスト
