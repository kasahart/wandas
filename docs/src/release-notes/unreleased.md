# Unreleased changes

Changes after [Wandas 0.11.0](v0.11.0.md).

## Added / 追加

- `ChannelFrame.cross_correlation(max_lag=None)` returns the normalized
  cross-correlation of every output/input channel pair as a new
  `CrossCorrelationFrame` with a lag axis in seconds (positive when the output
  lags the input). Each row is `scipy.signal.correlate(x_out, x_in)` divided by
  `sqrt(E_out * E_in)`, a coefficient in [-1, 1]. `delays` gives each pair's
  lag of the largest `abs(coefficient)` and `peak_coefficients` the signed
  value there; `select_pair()`, contiguous lag slicing, `plot()` (lag on x),
  `to_dataframe()`, WDF save/load and Recipe replay work as for the pairwise
  spectral Frames. `max_lag` keeps only lags within `+-max_lag` seconds.
  `ChannelFrame.cross_correlation(max_lag=None)`で、全ての出力／入力チャンネルの組の正規化相互相関を、
  秒単位のラグ軸（出力が入力より遅れるとき正）を持つ新しい`CrossCorrelationFrame`として得られる
  ようになりました。各行は`scipy.signal.correlate(x_out, x_in)`を`sqrt(E_out * E_in)`で割った
  [-1, 1]の係数です。`delays`は各組で`abs(係数)`が最大になるラグ、`peak_coefficients`はそこでの
  符号付きの値です。`select_pair()`、連続したラグの切り出し、`plot()`（横軸がラグ）、
  `to_dataframe()`、WDFの保存・読み込み、Recipeの再実行は、ペアのスペクトルFrameと同じように
  使えます。`max_lag`を指定すると`±max_lag`秒以内のラグだけを残します。
