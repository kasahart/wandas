# Unreleased changes

Changes after [Wandas 0.13.0](v0.13.0.md).

0.13.0以降の変更です。

## New features / 新機能

- `SpectrogramFrame.spectral_kurtosis` returns the spectral kurtosis of each
  channel over time, aligned with `freqs`, to find the band where impulses
  are strongest.
  `SpectrogramFrame.spectral_kurtosis`は各チャンネルの時間方向のスペクトル尖度を`freqs`に揃えて返します。
  衝撃成分が最も強い帯域を探すのに使えます。
  Only the STFT frames that lie wholly inside the signal are used; the new
  `SpectrogramFrame.signal_length` records the analysed length for this.
  信号の内側に収まるSTFTフレームだけを使います。そのために解析した信号の長さを新しい`SpectrogramFrame.signal_length`に保持します。
