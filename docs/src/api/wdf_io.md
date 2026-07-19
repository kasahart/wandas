# WDF File I/O / WDFファイル入出力

The `wandas.io.wdf_io` module saves and loads built-in typed Frames in the WDF (Wandas Data File) format.
`wandas.io.wdf_io` モジュールは built-in typed Frame を WDF (Wandas Data File) 形式で保存・読み込みします。

The WDF format is based on HDF5 and preserves not only the data but also all metadata such as sampling rate, units, and channel labels.
WDFフォーマットは HDF5 をベースとし、データだけでなくサンプリングレート、単位、チャンネルラベルなどのメタデータも完全に保存します。

## WDF Format Overview / WDFフォーマット概要

The WDF format has the following features:
WDFフォーマットは以下の特徴を持ちます:

- HDF5-based hierarchical data structure.
  HDF5ベースの階層的なデータ構造。
- Typed round-trip for Channel, spectral, spectrogram, cepstral, cepstrogram,
  N-octave, and roughness Frames.
  Channel、spectrum、spectrogram、cepstrum、cepstrogram、N-octave、roughness Frame の型付き往復。
- Size optimization through data compression and chunking.
  データ圧縮とチャンク化によるサイズ最適化。
- Version management for future extensions.
  将来の拡張に対応するバージョン管理。

File structure / ファイル構造:

```
[root attributes]: WDF, Frame-state, and display-history schemas
/data           : Complete rank-preserving Frame tensor / Frame tensor 全体
/channels/{i}   : Channel metadata / channel metadata
/coordinates    : Explicit represented axes such as quefrency / quefrency などの明示的な表現軸
/meta           : Frame-level metadata (JSON) / Frame metadata (JSON)
```

Schema values are HDF5 attributes on the file root (`f.attrs`), not an `/attrs`
group. / Schema 値は `/attrs` group ではなく、file root の HDF5 attribute
(`f.attrs`) として保存されます。

WDF 0.3 is the only supported WDF schema. Older and future versions fail explicitly;
there is no compatibility fallback or migration layer. It restores the exact built-in
Frame type and its analysis parameters.
Runtime lineage and Dask graphs are not restored; `operation_history` is display-only.
Coordinates that are part of an explicit Frame contract, such as represented
quefrencies, are persisted as finite ordered values on the Frame's sampling grid.
`SpectralFrame` accepts both complex FFT results and real Welch power spectra;
other typed domains retain their real- or complex-valued dtype contract.
`SpectralFrame` and `SpectrogramFrame` always contain the complete canonical one-sided
frequency axis. Their frequency and local-time values are derived from `sampling_rate`,
`n_fft`, and `hop_length`; frequency-axis slicing is not supported. Time slicing keeps
local `times` zero-based while absolute placement remains in `source_time_offset` and
`source_times`.

## Saving WDF Files / WDFファイル保存

::: wandas.io.wdf_io.save

## Loading WDF Files / WDFファイル読み込み

::: wandas.io.wdf_io.load

## Usage Examples / 利用例

```python
import wandas as wd

# Any built-in typed Frame can save itself
frame = wd.read("audio.wav").stft(n_fft=2048)
frame.save("analysis.wdf")

# Specifying options when saving
# 保存時のオプション指定
frame.save(
    "high_quality.wdf",
    compress="gzip",  # Compression method / 圧縮方式
    overwrite=True    # Allow overwriting / 上書き許可
)

# Restore the concrete stored type (SpectrogramFrame here)
restored = wd.load("analysis.wdf")
```

WDF save currently materializes the complete Frame before writing, and load reads the
stored tensor before wrapping it in Dask.
The tensor dtype is stored without conversion and restored exactly.
