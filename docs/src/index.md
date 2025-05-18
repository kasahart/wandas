# Wandas: **W**aveform **An**alysis **Da**ta **S**tructures

**Wandas** は、Pythonによる効率的な信号解析のためのオープンソースライブラリです。Wandas は、信号処理のための包括的な機能を提供し、Matplotlibとのシームレスな統合を実現しています。

## 機能

- **包括的な信号処理機能**: フィルタリング、フーリエ変換、STFTなど、基本的な信号処理操作を簡単に実行可能
- **可視化ライブラリとの統合**: Matplotlibとシームレスに統合してデータを簡単に可視化可能
- **遅延評価**: daskを活用した効率的な大規模データ処理
- **多様な分析ツール**: 周波数分析、オクターブバンド分析、時間-周波数分析など

## 使用例

### 音声ファイルの読み込みと可視化

```python
import wandas as wd

# docs/docs/ja/index.md からの相対パスでサンプルデータを指定
# 実際の使用時は適切なパスに変更してください
# cf = wd.read_wav("../../examples/data/summer_streets1.wav")
# cf.describe()
```

![波形とスペクトログラムの表示](../assets/images/read_wav_describe.png)

### フィルタ処理

```python
# import wandas as wd
# import numpy as np
# signal = wd.generate_sin(freqs=[5000, 1000], duration=1)
# ローパスフィルタを適用
# filtered_signal = signal.low_pass_filter(cutoff=1000)
# filtered_signal.fft().plot()
```

![ローパスフィルタの適用結果](../assets/images/low_pass_filter.png)

詳細なドキュメントや使用例については、[チュートリアル](tutorial/index.md)をご覧ください。

## ドキュメント構成

- [チュートリアル](tutorial/index.md) - 5分で始められる入門ガイドと一般的なタスクのレシピ集
- [APIリファレンス](api/index.md) - 詳細なAPI仕様
- [理論背景/アーキテクチャ](explanation/index.md) - 設計思想とアルゴリズムの解説
- [貢献ガイド](contributing.md) - コントリビューションのルールと方法

## ライセンス

このプロジェクトは [MITライセンス](https://opensource.org/licenses/MIT) の下で公開されています。
