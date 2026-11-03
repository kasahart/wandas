# Unreleased changes

Changes after [Wandas 0.13.0](v0.13.0.md).

0.13.0以降の変更です。

## New features / 新機能

- `SpectralFrame.find_peaks()` lists the spectral peaks of each channel as a
  pandas table (frequency, level, amplitude, prominence), using
  `scipy.signal.find_peaks` on the dB level, with optional parabolic
  interpolation between bins.
  `SpectralFrame.find_peaks()`は、dBレベルに`scipy.signal.find_peaks`を使って各チャンネルのスペクトルピーク
  （周波数・レベル・振幅・突出度）をpandasの表で返します。ビン間の放物線補間も選べます。
