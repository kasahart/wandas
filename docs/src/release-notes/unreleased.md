# Unreleased changes

Changes after [Wandas 0.12.0](v0.12.0.md).

## Added / 追加

- Level-trigger workflows can be saved and replayed as Recipes
  ([#630](https://github.com/kasahart/wandas/issues/630)). `ChannelFrame.trigger()`
  and `segment()`, `drop()` (and selections in event order, recorded as the
  dropped event IDs), and `average()`, `transfer_function()` and `coherence()` of
  `ChannelFrameSegments` are Recipe steps, so
  `RecipePlan.from_frame(cf.trigger(10.0, duration=1.0, pre=0.01).drop([2]).transfer_function())`
  works. A Recipe records the trigger conditions, not the detected positions:
  replaying it on another recording detects that recording's events again, and
  drops the events with the same IDs there. Segments now carry a `lineage` like
  a Frame, and each segment's Frame keeps its own lineage as a time slice of the
  source. `apply(func)`, selections out of event order, and segments built
  directly from Frames still cannot be saved, with a message that says why. A
  Recipe's inputs, output and named steps stay Frames; segments only appear
  between its nodes. `default_recipe_registry().describe()` lists the new steps
  with `ChannelFrameSegments` or `FrameSegments` as their receiver or result.
  レベルトリガーの処理をRecipeとして保存・再生できるようになりました。
  `ChannelFrame.trigger()`と`segment()`、`drop()`（イベント順の選択は除いたイベントIDとして記録）、
  `ChannelFrameSegments`の`average()`・`transfer_function()`・`coherence()`がRecipeのstepになります。
  Recipeには検出した位置ではなくトリガーの条件を記録するので、別の録音で再生するとその録音で
  検出し直し、同じIDのイベントを除きます。Segmentsも Frame と同じく`lineage`を持ち、各セグメントの
  Frameは元の録音の時間スライスとして自分のlineageを保ちます。`apply(func)`、イベント順でない選択、
  Frameから直接作ったSegmentsは今までどおり保存できず、理由をメッセージで示します。Recipeの入力・
  出力・名前付きstepはFrameのままで、Segmentsはnodeの間にだけ現れます。
