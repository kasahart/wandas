# Unreleased changes

Changes after [Wandas 0.13.0](v0.13.0.md).

0.13.0以降の変更です。

## New features / 新機能

- `ChannelFrame.integrate()` integrates each channel in time with the
  trapezoidal rule, for example acceleration to velocity, and moves the unit
  along `m/s^2 -> m/s -> m`. It is a Recipe step. Remove the drift with
  `high_pass_filter()` afterwards.
  `ChannelFrame.integrate()`は各チャンネルを台形則で時間積分し（例: 加速度から速度）、
  単位を`m/s^2 -> m/s -> m`と進めます。Recipeの手順として記録されます。
  ドリフトは後から`high_pass_filter()`で取り除きます。
