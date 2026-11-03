# Unreleased changes

Changes after [Wandas 0.13.0](v0.13.0.md).

0.13.0以降の変更です。

## New features / 新機能

- `ChannelFrame.peak` and `ChannelFrame.kurtosis` return the peak absolute
  value and the Pearson kurtosis of each channel as NumPy arrays, like `rms`.
  `kurtosis` uses `dask.array.stats.kurtosis`.
  `ChannelFrame.peak`と`ChannelFrame.kurtosis`は、`rms`と同じように各チャンネルの絶対値の最大と
  ピアソンの尖度をNumPy配列で返します。`kurtosis`は`dask.array.stats.kurtosis`で計算します。
- `SpectralFrame.find_peaks()` lists the spectral peaks of each channel as a
  pandas table (frequency, level, amplitude, prominence), using
  `scipy.signal.find_peaks` on the dB level, with optional parabolic
  interpolation between bins.
  `SpectralFrame.find_peaks()`は、dBレベルに`scipy.signal.find_peaks`を使って各チャンネルのスペクトルピーク
  （周波数・レベル・振幅・突出度）をpandasの表で返します。ビン間の放物線補間も選べます。
- `SpectrogramFrame.spectral_kurtosis` returns the spectral kurtosis of each
  channel over time, aligned with `freqs`, to find the band where impulses
  are strongest.
  `SpectrogramFrame.spectral_kurtosis`は各チャンネルの時間方向のスペクトル尖度を`freqs`に揃えて返します。
  衝撃成分が最も強い帯域を探すのに使えます。
  Only the STFT frames that lie wholly inside the signal are used; the new
  `SpectrogramFrame.signal_length` records the analysed length for this.
  信号の内側に収まるSTFTフレームだけを使います。そのために解析した信号の長さを新しい`SpectrogramFrame.signal_length`に保持します。
  Spectrograms from `stft()` therefore save as WDF 0.7, which Wandas 0.13.0 and
  earlier cannot read.
  そのため`stft()`のスペクトログラムはWDF 0.7で保存され、Wandas 0.13.0以前では読めません。
