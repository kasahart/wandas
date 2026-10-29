# Unreleased changes

Changes after [Wandas 0.11.0](v0.11.0.md).

## Added / 追加

- `ChannelFrame.transfer_function()` and `CrossSpectralFrame.transfer_function()`
  take `estimator="H1"` (default, unchanged) or `estimator="H2"`. H2 is
  `P_out_out / P_in_out` and is not biased by noise on the input channel, where
  H1 (`P_out_in / P_in_in`) is underestimated; with `average="mean"` (the
  default) `H1 / H2` equals the coherence.
  The result is the same `TransferFunctionFrame` with the same unit and
  reference, plus an `estimator` property. WDF files store an H2 result with
  the new `definition` value `h2_input_cross_denominator`, which older Wandas
  versions cannot load; H1 files are unchanged. Recipes record `estimator`
  only when it is given, so existing Recipes replay as before.
  `ChannelFrame.transfer_function()`と`CrossSpectralFrame.transfer_function()`で
  `estimator="H1"`（既定、従来どおり）または`estimator="H2"`を選べるようになりました。H2は
  `P_out_out / P_in_out`で、入力チャンネル側の雑音で過小評価されるH1（`P_out_in / P_in_in`）と違い、
  入力側の雑音の影響を受けません。`average="mean"`（既定）のとき`H1 / H2`はコヒーレンスに等しくなります。結果は単位と基準が
  同じ`TransferFunctionFrame`で、`estimator`プロパティが増えました。H2の結果をWDFに保存すると
  `definition`は新しい値`h2_input_cross_denominator`になり、古いWandasでは読み込めません。
  H1のファイルは変わりません。Recipeには`estimator`を指定したときだけ記録するため、既存の
  Recipeはこれまでどおり再実行できます。
