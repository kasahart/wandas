# Wandas: **W**aveform **An**alysis **Da**ta **S**tructures

[![CI](https://github.com/kasahart/wandas/actions/workflows/ci.yml/badge.svg)](https://github.com/kasahart/wandas/actions/workflows/ci.yml)

**Wandas** は、Pythonによる効率的な信号解析のためのオープンソースライブラリです。Wandas は、信号処理のための包括的な機能を提供し、Matplotlibとのシームレスな統合を実現しています。

## 機能

- **包括的な信号処理機能**: フィルタリング、フーリエ変換、STFTなど、基本的な信号処理操作を簡単に実行可能
- **可視化ライブラリとの統合**: Matplotlibとシームレスに統合してデータを簡単に可視化可能。

## インストール

```bash
pip install git+https://github.com/kasahart/wandas
```

## クイックスタート

```python
import wandas as wd

# WAV ファイルの読み込み
signal = wd.read_wav('audio_sample.wav')

# 信号をプロット
signal.plot()

# ローパスフィルタを適用
filtered_signal = signal.low_pass_filter(cutoff=1000)

# スペクトル解析のためにフーリエ変換を実行
spectrum = filtered_signal.fft()

# スペクトルをプロット
spectrum.plot()

# フィルタ済み信号を WAV ファイルに保存
filtered_signal.to_wav('filtered_audio.wav')
```

## ドキュメント

詳細な使用方法は`/exsampls`を参照してください

## 対応データ形式

- **音声ファイル**: WAV
- **データファイル**: CSV

## バグ報告と機能リクエスト

- **バグ報告**: [Issue Tracker](https://github.com/kasahart/wandas/issues) に詳細を記載してください。
- **機能リクエスト**: 新機能や改善案があれば、気軽に Issue をオープンしてください。

## 貢献方法

1. このリポジトリをフォークします。
2. フィーチャーブランチを作成します。(`git checkout -b feature/new_feature`)
3. 変更をコミットします。(`git commit -m 'Add new_feature'`)
4. ブランチをプッシュします。(`git push origin feature/new_feature`)
5. プルリクエストを送信します。

## ライセンス

このプロジェクトは [MIT ライセンス](LICENSE) の下で公開されています。

---

Wandas を使って効率的な信号解析体験を！
