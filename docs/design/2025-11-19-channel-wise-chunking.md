# ADR: Channel-Wise Dask Chunking Strategy / チャンネル単位のDaskチャンク分割戦略

- **Status**: Accepted / Implemented
- **Date**: 2025-11-19
- **Context**: Optimization of parallel processing and API simplification for Dask-backed frames.
  - **コンテキスト**: Daskベースのフレームにおける並列処理の最適化とAPIの簡素化。

## Context & Problem Statement / 背景と問題点

Wandas uses Dask arrays to handle large waveform data lazily. Previously, the default chunking strategy was inconsistent or defaulted to a single chunk for the entire array (`chunks=-1`).
Wandasは大規模な波形データを遅延処理するためにDask配列を使用しています。以前は、デフォルトのチャンク分割戦略が一貫していないか、配列全体に対して単一のチャンク（`chunks=-1`）がデフォルトとなっていました。

This approach had several drawbacks:
このアプローチにはいくつかの欠点がありました：

1. **Inefficient Parallelism**: Operations that could be parallelized per channel (e.g., filtering, spectral analysis) were often treated as a single task, failing to utilize Dask's potential.
   - **非効率な並列性**: チャンネルごとに並列化できる操作を単一のタスクとして扱うことを強制するもので、Daskの潜在能力を活用できていませんでした。
2. **API Ambiguity**: The `ChannelFrame.from_file` method exposed a `chunk_size` parameter that allowed users to fragment the data arbitrarily, potentially breaking channel-wise assumptions.
   - **APIの曖昧さ**: `ChannelFrame.from_file` メソッドは `chunk_size` パラメータを公開しており、ユーザーがデータを任意に断片化することを許可していましたが、これによりチャンネル単位の前提が崩れる可能性がありました。
3. **Type Safety Issues**: Direct calls to `da.from_array` caused friction with static type checkers (mypy) due to strict type stubs for the `chunks` parameter.
   - **型安全性の問題**: `da.from_array` を直接呼び出すと、`chunks` パラメータに対する厳密な型スタブのため、静的型チェッカー（mypy）との摩擦が生じていました。

## Decisions / 決定事項

### 1. Enforce Channel-Wise Chunking / チャンネル単位チャンク分割の強制

We decided to enforce a "Channel-First" chunking policy across the entire codebase. The channel axis (axis 0) must always have a chunk size of `1`.
コードベース全体で「チャンネルファースト」のチャンク分割ポリシーを強制することに決定しました。チャンネル軸（軸0）のチャンクサイズは常に `1` でなければなりません。

- **2D Frames (Time Domain)**: `chunks=(1, -1)`
  - Each channel is its own chunk. The sample axis (time) remains a single chunk by default to preserve contiguous memory for FFT/filtering operations.
  - **2Dフレーム（時間領域）**: 各チャンネルが独自のチャンクとなります。サンプル軸（時間）は、FFTやフィルタリング操作のために連続したメモリを維持するため、デフォルトで単一のチャンクのままとなります。
- **3D Frames (Spectrograms)**: `chunks=(1, -1, -1)`
  - Each channel is independent. Frequency and time axes remain contiguous by default.
  - **3Dフレーム（スペクトログラム）**: 各チャンネルは独立しています。周波数軸と時間軸はデフォルトで連続したままとなります。

### 2. Simplify I/O API (Breaking Change) / I/O APIの簡素化（破壊的変更）

We removed the `chunk_size` parameter from `ChannelFrame.from_file`.
`ChannelFrame.from_file` から `chunk_size` パラメータを削除しました。

- **Rationale**: Providing a chunking argument at creation time led to confusion about which axis was being chunked.
  - **理由**: 作成時にチャンク分割の引数を提供すると、どの軸がチャンク分割されているかについて混乱が生じていました。
- **New Pattern**: Frames are always created with the default channel-wise chunking.
  - **新しいパターン**: フレームは常にデフォルトのチャンネル単位のチャンク分割で作成されます。

### 3. Centralize Dask Array Creation / Dask配列生成の集約

We introduced a centralized helper `wandas.utils.dask_helpers.da_from_array`.
集約されたヘルパー `wandas.utils.dask_helpers.da_from_array` を導入しました。

- **Rationale**: To isolate type-checking suppressions and ensure consistent Dask array instantiation patterns.
  - **理由**: 型チェックの抑制を分離し、一貫したDask配列のインスタンス化パターンを保証するため。
- **Implementation**: A thin wrapper around `dask.array.from_array` that accepts `Any` for the chunks parameter, resolving mypy conflicts.
  - **実装**: `chunks` パラメータに `Any` を受け入れる `dask.array.from_array` の薄いラッパーで、mypyの競合を解決します。

### 4. Maintain Single Delayed Operations for Processing (Default) / 処理における単一遅延操作の維持（デフォルト）

Existing functions are implemented to avoid Python loops by using a single delayed operation for the entire data structure. However, the design assumes that it is possible to use per-channel iteration selectively depending on the choice.
既存の関数はPythonのforループを使わないように、データ構造全体に対して単一の遅延操作を使用するよう実装されています。しかし、設計上は選択に応じてチャンネルごとの反復処理を使い分けることが可能であることを想定しています。

- **Rationale**: To avoid Python-loop overhead by default. Dask handles the graph generation efficiently based on the underlying chunks.
  - **理由**: デフォルトではPythonループのオーバーヘッドを回避するため。Daskは基礎となるチャンクに基づいてグラフ生成を効率的に処理します。

## Detailed Design / 詳細設計

### BaseFrame Auto-Rechunking / BaseFrameの自動リチャンク

The `BaseFrame` class automatically enforces the chunking policy upon initialization.
`BaseFrame` クラスは初期化時にチャンク分割ポリシーを自動的に強制します。

```python
# Logic in BaseFrame.__init__
if data.ndim == 2:
    self._data = data.rechunk((1, -1))
elif data.ndim >= 3:
    # e.g., (channels, freq, time)
    self._data = data.rechunk((1, -1, -1))
```

This ensures that even if a user (or an internal method) provides a Dask array with a different chunking scheme, it is normalized to the channel-wise format immediately.
これにより、ユーザー（または内部メソッド）が異なるチャンク分割スキームを持つDask配列を提供した場合でも、即座にチャンネル単位の形式に正規化されることが保証されます。

## Consequences / 影響

### Positive / ポジティブな影響

- **Improved Parallelism**: Downstream Dask operations can now automatically parallelize across channels without additional configuration.
  - **並列性の向上**: 下流のDask操作は、追加の設定なしで自動的にチャンネル間で並列化できるようになります。
- **Memory Efficiency**: Processing single channels does not require loading the entire dataset into memory.
  - **メモリ効率**: 単一チャンネルの処理において、データセット全体をメモリにロードする必要がなくなります。
- **Consistency**: All frame types (Time, Spectral, Spectrogram) now follow the same structural rules.
  - **一貫性**: すべてのフレームタイプ（時間、スペクトル、スペクトログラム）が同じ構造ルールに従うようになります。
- **Type Safety**: The codebase is now compliant with strict mypy settings without scattered `# type: ignore` comments.
  - **型安全性**: コードベースは、散在する `# type: ignore` コメントなしで厳密なmypy設定に準拠するようになります。

### Negative / Risks / ネガティブな影響・リスク

- **Breaking Change**: The removal of `chunk_size` breaks backward compatibility.
  - **破壊的変更**: `chunk_size` の削除により、後方互換性が失われます。
- **Overhead**: The auto-rechunking in `BaseFrame` might introduce a negligible graph construction overhead if the input is already chunked differently.
  - **オーバーヘッド**: `BaseFrame` での自動リチャンクは、入力がすでに異なってチャンク分割されている場合、無視できる程度のグラフ構築オーバーヘッドを導入する可能性があります。

## Validation Strategy / 検証戦略

The design is validated by:
設計は以下によって検証されます：

1. **Unit Tests**: Specifically `tests/frames/test_channel_chunking.py`, which asserts the chunk structure of created frames.
   - **ユニットテスト**: 具体的には `tests/frames/test_channel_chunking.py` で、作成されたフレームのチャンク構造をアサートします。
2. **Integration Tests**: Verifying that WDF I/O round-trips preserve the chunk structure.
   - **統合テスト**: WDF I/Oのラウンドトリップがチャンク構造を維持していることを検証します。
3. **Static Analysis**: Passing `mypy` with `pydantic` plugin enabled.
   - **静的解析**: `pydantic` プラグインを有効にした状態で `mypy` を通過すること。
