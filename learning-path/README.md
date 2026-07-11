# Wandas 学習パス

## 信号処理・音響解析のための包括的なガイド

この学習パスは、Wandasを使って信号処理・音響解析を始めるための**統合されたステップバイステップガイド**です。チュートリアルと実践例を統合し、ユーザーにWandasを使うメリットを体感してもらうことを目的としています。

## 🎯 この学習パスの目的

**信号処理の初心者から上級者まで**、Wandasを使って以下を達成できるようにします：

- ✅ **なぜWandasを使うのか**を理解する
- ✅ **基本的な信号処理**をpandasライクなAPIで実装
- ✅ **大規模データ**を効率的に処理
- ✅ **簡易に可視化**を作成

## 📚 学習の流れ

この学習パスは以下のmarimoアプリで構成されています。各アプリは独立しており、順番に進めることも、興味のある部分だけを学ぶことも可能です。

### 1. [00_why_wandas.py](00_why_wandas.py) - Wandasとは何か

**なぜWandasが必要なのか？**

- 信号処理の課題と従来のアプローチの問題点
- Wandasの特徴と利点
- どんな問題を解決できるか

### 2. [01_getting_started.py](01_getting_started.py) - 環境構築とウォームアップ

**Wandasを動かしてみよう**

- インストールと環境設定
- 最初の信号生成と可視化
- marimo環境での探索的な確認

### 3. [02_working_with_data.py](02_working_with_data.py) - データの読み込みと操作

**現実のデータをWandasで扱う**

- WAV/CSVファイルの読み込み
- ChannelFrameデータ構造の理解
- データのアクセスと基本操作

### 4. [03_signal_processing_basics.py](03_signal_processing_basics.py) - 信号処理の基礎

**周波数分析とフィルタリング**

- FFTによる周波数領域変換
- ローパス/ハイパス/バンドパスフィルター
- 実践的なオーディオ処理例

### 5. [04_advanced_processing.py](04_advanced_processing.py) - 高度な信号処理

**スペクトログラムと時間周波数分析**

- STFTとスペクトログラム
- 時間周波数解析の応用
- 信号の特徴抽出

### 6. [05_custom_functions.py](05_custom_functions.py) - custom function

**独自処理をWandasの操作として扱う**

- custom operationの最小例
- frame操作としての組み込み
- 再利用しやすい処理単位の作り方

### 7. [06_pipeline_recipe_ux.py](06_pipeline_recipe_ux.py) - Frame-first Recipe UX

**探索解析を別データで再現する基本導線**

- 通常のFrame method chainからRecipeを抽出
- 別Frameへのreplayによる再現実験
- `operation_history` による処理履歴の確認
- 明示Recipe構築とsklearn adapterは任意セクションとして確認

### 8. [08_metadata_driven_dataset_search.py](08_metadata_driven_dataset_search.py) - メタデータ駆動のファイル検索

> 07はRecipe UXの設計検討用補助資料で、初学者向けの本線ではありません。そのため本線は06から08へ進みます。

**波形を読む前に対象ファイルを絞り込む**

- `path_metadata=True` でフォルダ階層からメタデータを自動推論
- `dataset.select()` による完全一致・AND検索
- Dataset全体への処理チェーンと処理後のメタデータ選択
- 外部属性が必要な場合だけsidecar CSVをlookupとして接続

## 🚀 学習を始める前に

### 必要な環境

- Python 3.10+
- marimo
- 基本的な信号処理の知識（なくてもOK）

## 📖 各marimoアプリの特徴

| marimoアプリ | 学習目標 | 実践要素 | 動機付け |
|-------------|---------|---------|---------|
| 00_why_wandas | Wandas理解 | 概念説明 | なぜ必要か |
| 01_getting_started | 環境構築 | インストール | すぐに始める |
| 02_working_with_data | データ操作 | ファイルIO | データ活用 |
| 03_signal_processing_basics | 基本処理 | フィルタリング | 信号改善 |
| 04_advanced_processing | 高度処理 | スペクトログラム | 特徴抽出 |
| 05_custom_functions | custom処理 | custom operation | 処理の再利用 |
| 06_pipeline_recipe_ux | frame-first Recipe基本 | extract / replay | 探索解析の再現 |
| 08_metadata_driven_dataset_search | Dataset検索 | path_metadata / select | 大量ファイルの事前絞り込み |

## 🔗 関連リソース

- [公式ドキュメント](https://kasahart.github.io/wandas/)
- [GitHub リポジトリ](https://github.com/kasahart/wandas)
- [API リファレンス](https://kasahart.github.io/wandas/api/)
- [07_frame_centric_recipe_ux.py](07_frame_centric_recipe_ux.py) - Recipe UX設計検討用marimo。初学者向けの本線ではなく、graph recipe、terminal値、将来facade案を確認するための補助資料。

## 🤝 貢献とフィードバック

この学習パスは継続的に改善しています。改善提案やバグ報告は[GitHub Issues](https://github.com/kasahart/wandas/issues)へお願いします。

---
