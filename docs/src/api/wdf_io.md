# WDF File I/O / WDFファイル入出力

WDF 0.4 is an xarray-backed, HDF5-based artifact for exact typed round-trips of
Wandas' seven built-in Frame classes. WDF 0.4 は、Wandas の7種類の built-in
Frameを型付きで往復する、xarray backedのHDF5 artifactです。

## Contract / 契約

- `BaseFrame.save(path, *, compress="gzip", overwrite=False)` saves WDF 0.4.
- `wd.load(path)` restores the exact stored built-in Frame type.
- `ChannelFrame.load(path)` additionally requires the stored type to be
  `ChannelFrame`.
- Loading accepts local `str` and `Path` values only. URL download is not part of
  the WDF API.
- WDF 0.1 through 0.3 and future versions are explicitly unsupported. There is no
  fallback or migration layer.

WDF 0.4 persists:

- the exact built-in Frame type and its validated constructor state;
- the raw tensor values, dtype, semantic dimension names, and stored
  one-dimensional represented-axis coordinates (frequency and local time are
  derived instead);
- sampling rate, Frame label, strict-JSON user/recording metadata, stable channel
  IDs, channel labels, calibration factors, units, references, channel extras,
  and per-channel source-time offsets;
- canonical operation history as a display-only compatibility view.

WDF 0.4 が保存するものは次のとおりです。

- exact built-in Frame type と検証済み constructor state
- raw tensor の値と dtype、semantic dimension 名、および保存対象の1次元
  represented-axis coordinate（frequency と local time は代わりに導出）
- sampling rate、Frame label、strict-JSON の user／recording metadata、安定した
  channel ID、channel label、校正係数、単位、基準値、channel extra、channel ごとの
  source-time offset
- display 専用の互換 view である canonical operation history

WDF intentionally does **not** persist live lineage, `previous` Frame references,
live operation objects or callables, executable `RecipePlan` intent, Dask graphs,
task/chunk topology, scheduler state, or an open runtime storage backend. A loaded
Frame reconstructs a new persistence-boundary lineage and uses the stored history
only for display. Saving a processed result therefore does not make its workflow
replayable.

WDF は live lineage、`previous` Frame reference、live operation object／callable、
実行可能な `RecipePlan` intent、Dask graph、task／chunk topology、scheduler state、
open 中の runtime storage backend を意図的に保存しません。load 後の Frame は
persistence boundary の新しい lineage を構築し、保存済み history は表示だけに使います。
したがって、処理結果を保存しても workflow を再実行可能にはできません。

Recipe JSON is the complementary boundary: schema 2 stores reusable operation
intent and named input slots, but not Frame samples, live lineage, Dask graphs, or
callables. Use WDF for one concrete typed result and Recipe JSON for a workflow to
apply to runtime inputs.

Recipe JSON は補完的な境界です。schema 2 は再利用可能な operation intent と名前付き
input slot を保存しますが、Frame sample、live lineage、Dask graph、callable は保存しません。
1つの具体的な型付き結果には WDF、runtime input に適用する workflow には Recipe JSON を
使います。

Root attributes are `version`, `frame_type`, `sampling_rate`, `label`,
`constructor_json`, `metadata_json`, and `operation_history_json`. The xarray Dataset
contains these data variables:

```text
data
channel_label
channel_unit
channel_ref
channel_calibration_factor
source_time_offset
channel_extra_json
```

The stable `channel` IDs are a dimension coordinate. Other persisted represented
axes are ordinary one-dimensional xarray dimension coordinates; the I/O layer does
not give one coordinate name a separate storage mechanism. `data.dims` is the sole
source of semantic dimension names. Frequency and local time are derived from
`sampling_rate`, `n_fft`, and `hop_length`, so they are not stored.

Raw tensor values and calibration are stored separately. This prevents calibration
from being applied twice after load. Runtime lineage, live operation objects, Recipe
artifacts, and Dask graphs are outside WDF; `operation_history_json` is display
history only.

raw tensor value と calibration は別々に保存されるため、load 後に calibration が
二重適用されません。runtime lineage、live operation object、Recipe artifact、
Dask graph は WDF の対象外で、`operation_history_json` は表示専用 history です。

Saving completes synchronously, but Wandas passes internal chunks to the writer
without first materializing the complete tensor as a NumPy array. While a loaded
Frame or a Frame derived from it remains in use, do not move, delete, or overwrite
its source WDF file. Read NumPy values through `frame.data`; users do not manage
the xarray/Dask backend directly.

保存は同期的に完了しますが、Wandasは内部でchunkをwriterへ渡し、事前にtensor全体を
NumPy化しません。WDFから読み込んだFrameと、そのFrameから派生したFrameを利用して
いる間は、元のWDFファイルを移動・削除・上書きしないでください。数値は他のFrameと
同じく`frame.data`からNumPy配列として取得します。xarrayやDask backendを利用者が
管理する必要はありません。

## Saving / 保存

::: wandas.io.wdf_io.save

## Loading / 読み込み

::: wandas.io.wdf_io.load

```python
import wandas as wd

frame = wd.read("audio.wav").stft(n_fft=2048)
frame.save("analysis.wdf", compress="gzip", overwrite=True)
restored = wd.load("analysis.wdf")
```
