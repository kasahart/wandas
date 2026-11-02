# Unreleased changes

Changes after [Wandas 0.12.0](v0.12.0.md).

0.12.0以降の変更です。

## New features / 新機能

- `ChannelFrame.integrate()` integrates each channel in the frequency domain
  within a band, for example acceleration to velocity, and moves the unit
  along `m/s^2 -> m/s -> m`. It is a Recipe step.
  `ChannelFrame.integrate()`は各チャンネルを周波数領域で帯域を限って積分し（例: 加速度から速度）、
  単位を`m/s^2 -> m/s -> m`と進めます。Recipeの手順として記録されます。
