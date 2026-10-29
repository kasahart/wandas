# Unreleased changes

Changes after [Wandas 0.11.0](v0.11.0.md).

## Added / 追加

- `ChannelFrame.envelope()` returns the amplitude envelope of each channel,
  `abs(scipy.signal.hilbert(x))` over the whole time axis, as a lazy
  `ChannelFrame` with the same sampling rate, length and calibrated unit.
  float32 input stays float32, and it is a Recipe step
  (`wandas.audio.envelope`). For an envelope spectrum, band-pass first:
  `frame.band_pass_filter(2000, 4000).envelope().remove_dc().fft()`.
  The first and last samples carry SciPy's edge effects.
  `ChannelFrame.envelope()`で各チャンネルの振幅包絡（時間軸全体での
  `abs(scipy.signal.hilbert(x))`）を、サンプリング周波数・長さ・校正済み単位が同じ遅延
  `ChannelFrame`として得られるようになりました。float32入力はfloat32のままで、Recipeの段
  （`wandas.audio.envelope`）にもなります。包絡スペクトルは帯域通過フィルタの後に
  `frame.band_pass_filter(2000, 4000).envelope().remove_dc().fft()`で求められます。
  先頭と末尾のサンプルにはSciPyの端の影響が残ります。
