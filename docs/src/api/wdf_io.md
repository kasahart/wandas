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

保存は同期的に完了しますが、Wandasは事前にtensor全体を
`frame._data.compute()`でNumPy化せず、Dask arrayをxarrayへ直接渡します。
読み込みはbackend-backed Dask arrayを返します。`compute()`または`persist()`が
完了するまでは、元のWDFファイルを移動・削除・上書きしないでください。

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
