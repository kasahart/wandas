# Performance Benchmarks

このディレクトリには、wandasの主要な信号処理操作のパフォーマンスベンチマークが含まれています。

## 概要

pytest-benchmarkを使用して、以下の操作のパフォーマンスを継続的に監視します：

- **FFT (Fast Fourier Transform)** - 高速フーリエ変換
- **STFT (Short-Time Fourier Transform)** - 短時間フーリエ変換
- **Welch法** - パワースペクトル密度推定
- **Low-pass Filter** - ローパスフィルタ
- **High-pass Filter** - ハイパスフィルタ

## ローカルでの実行方法

### 基本的な実行

```bash
# すべてのベンチマークを実行
uv run pytest tests/benchmarks/ --benchmark-only

# 最小ラウンド数を指定
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-min-rounds=10

# ウォームアップを有効にして実行
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-warmup=on

# JSON形式で結果を保存
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json
```

### 詳細な実行

```bash
# 詳細な統計情報を表示
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-verbose

# 特定のベンチマークのみを実行
uv run pytest tests/benchmarks/test_performance.py::TestFFTPerformance -v --benchmark-only

# ヒストグラムを表示
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-histogram=histogram
```

### ベンチマーク結果の比較

```bash
# 前回の結果と比較
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare

# 保存された結果と比較
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=0001

# 失敗の閾値を設定（例: 150%遅い場合は失敗）
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare --benchmark-compare-fail=mean:150%
```

## CI/CDでの実行

GitHub Actionsワークフロー（`.github/workflows/performance.yml`）が自動的に以下を実行します：

1. **mainブランチへのプッシュ時**
   - ベンチマークを実行
   - 結果をartifactとして保存（90日間保持）

2. **プルリクエスト時**
   - ベンチマークを実行
   - 結果をPRコメントとして投稿
   - パフォーマンス劣化をチェック（将来実装予定）

## ベンチマーク結果の見方

### 出力の説明

```
Name (time in us)             Min       Max      Mean   StdDev    Median      IQR  Outliers  OPS (Kops/s)  Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------
test_fft_performance     208.8400  276.7970  223.0674  13.1178  217.5565  17.7425     59;10        4.4830     316           1
```

- **Min/Max**: 最小/最大実行時間
- **Mean**: 平均実行時間
- **StdDev**: 標準偏差（安定性の指標）
- **Median**: 中央値
- **IQR**: 四分位範囲
- **Outliers**: 外れ値の数
- **OPS**: 1秒あたりの操作回数
- **Rounds**: 実行回数

### パフォーマンスの目安

- **Mean < 1ms**: 非常に高速
- **Mean < 10ms**: 高速
- **Mean < 100ms**: 許容範囲
- **Mean > 100ms**: 最適化が必要な可能性

## ベンチマークの追加方法

新しい操作のベンチマークを追加する場合：

1. `test_performance.py`に新しいテストクラスを追加
2. テストメソッドは`test_*_performance`の命名規則に従う
3. `benchmark`フィクスチャを使用して操作を測定

例：

```python
class TestNewOperationPerformance:
    """Benchmark tests for new operation."""

    def test_new_operation_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark new operation.

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(benchmark_signal.new_operation, param=value)
        assert result is not None
```

## トラブルシューティング

### ベンチマークが遅すぎる

```bash
# ラウンド数を減らす
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-min-rounds=3

# タイムアウトを増やす
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-max-time=10
```

### メモリ不足

```bash
# 小さいデータセットを使用するようフィクスチャを変更
# conftest.pyでdurationやsampling_rateを調整
```

## 参考資料

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [wandas documentation](https://kasahart.github.io/wandas/)
- [GitHub Actions workflow](.github/workflows/performance.yml)
