# Unreleased changes

Changes after [Wandas 0.12.0](v0.12.0.md).

0.12.0以降の変更です。

## New features / 新機能

- `SpectrogramFrame.spectral_kurtosis` returns the spectral kurtosis of each
  channel over time, aligned with `freqs`, to find the band where impulses
  are strongest.
  `SpectrogramFrame.spectral_kurtosis`は各チャンネルの時間方向のスペクトル尖度を`freqs`に揃えて返します。
  衝撃成分が最も強い帯域を探すのに使えます。
