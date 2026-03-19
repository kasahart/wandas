---
description: "Visualization test patterns: plot strategy dispatch, axes return types, parameter forwarding, and memory leak prevention"
applyTo: "tests/visualization/**"
---
# Wandas Test Policy: Visualization (`tests/visualization/`)

Visualization テストは「プロット生成の正しさ」と「フレームメソッドとの整合性」を検証する層です。
数値計算の正確性は Processing テストで担保されるため、Visualization テストでは
**「正しいデータが正しい形式でプロットに渡されているか」** に集中します。

**前提**: このファイルは [test-grand-policy.instructions.md](test-grand-policy.instructions.md) と同時に適用されます。

---

## Common Fixtures for Visualization Tests

`conftest.py` に以下の fixture を定義すること。非インタラクティブバックエンド（`matplotlib.use("Agg")`）をインポート前に設定すること。

- **`channel_frame`**: プロットテスト用の標準モノラルフレーム（440 Hz 正弦波、SR=16000 Hz、label="test_signal"）。
- **`stereo_frame`**: マルチチャンネルプロットテスト用の 2 チャンネルフレーム（440 Hz と 880 Hz の正弦波、SR=16000 Hz、ch_labels=`["440Hz", "880Hz"]`）。
- **`cleanup_plots`（autouse=True）**: 各テスト後に全 Figure のクリア（`fig.clf()`）と `plt.close("all")` を実行するクリーンアップ fixture。メモリリーク防止のためすべてのテストに自動適用すること。

---

## Visualization Test Strategy

### What to Test (and What NOT to Test)

**Test (Visualization の責任):**
- PlotStrategy の dispatch が正しいこと（`create_operation("waveform")` が正しい Strategy を返す）
- Axes オブジェクトが返されること
- 正しい数のサブプロットが生成されること（チャンネル数に対応）
- 軸ラベル、タイトル、凡例が設定されていること
- `overlay=True/False` で挙動が変わること

**Do NOT Test (Processing/Frame の責任):**
- プロットされるデータの数値的正確性（これは Processing テストで担保）
- FFT や STFT の計算結果（spectral テストで担保）
- メタデータの伝播（frame テストで担保）

---

## PlotStrategy Dispatch Tests

`wandas.visualization.plotting.create_operation` を使ってプロット戦略の dispatch を検証すること:

- `"waveform"` を渡すと `plot` メソッドを持つ Strategy オブジェクトが返ること
- `"frequency"`, `"spectrogram"`, `"describe"` 等の登録済みプロット種別がすべて dispatch できること
- 未登録の文字列を渡すと `ValueError` または `KeyError` が送出されること

---

## Axes Return Type Tests

Matplotlib の非インタラクティブバックエンド（Agg）を使用してテストすること:

- `channel_frame.plot()` が `matplotlib.axes.Axes` または `Iterator` を返すこと
- `stereo_frame.plot(overlay=True)` が単一の `Axes` を返し、そこに `n_channels` 本以上のラインが描画されていること
- `stereo_frame.plot(overlay=False)` が複数のサブプロットを生成し、チャンネル数に対応した `Axes` の数を持つこと

---

## Describe Method Tests

`describe()` は複合プロットメソッドであり、特別な検証が必要:

- `describe(is_close=False)` が `plt.Figure` オブジェクトのリストを返し、件数がチャンネル数と一致すること
- `describe(is_close=True)` が `None` を返し、Figure が閉じられること（デフォルト動作）
- `describe(image_save=path)` が指定パスにファイルを保存すること
- マルチチャンネルの場合、ファイル名にチャンネルインデックスのサフィックスが付与されること（例: `test_0.png`, `test_1.png`）

---

## Plot Parameter Forwarding Tests

以下のパラメータがプロット結果に正しく反映されることを検証すること:

- `title`: `ax.get_title()` が指定した文字列と一致すること
- `xlabel` / `ylabel`: `ax.get_xlabel()` / `ax.get_ylabel()` が指定した文字列と一致すること
- `xlim` / `ylim`: `ax.get_xlim()` / `ax.get_ylim()` が指定した範囲と `pytest.approx` で一致すること

---

## Memory Leak Prevention

Visualization テストでは Figure のメモリリークを防ぐため:

- 個別テストで Figure を取得した場合は `fig.clf()` で内部状態をクリアし、その後 `plt.close(fig)` でウィンドウを閉じること。`plt.close("all")` のみでは内部状態がクリアされず不十分。
- `autouse=True` の `cleanup_plots` fixture を使用することで、すべてのテストに自動的にクリーンアップを適用できる（推奨）。

---

## Cross-References
- [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars and test pyramid
- [frames-design.instructions.md](frames-design.instructions.md) — Frame method behavior that plots depend on
