# Unreleased changes

Changes after [Wandas 0.11.0](v0.11.0.md).

## Added / 追加

- `ChannelFrame.leq(freq_weighting="Z")` returns the equivalent continuous
  level of each channel, and `ChannelFrame.percentile_levels(n, freq_weighting,
  time_weighting)` the levels exceeded `n` % of the time (L_5, L_50, L_95, ...)
  of the `sound_level(dB=True)` time series. Both are scalar reductions like
  `rms`: they compute immediately and return NumPy arrays, not Frames.
  `leq("A")` of a calibrated Pa channel is L_Aeq in dB SPL and equals
  `channels[i].level_reference.to_level(frame.a_weighting().rms[i])`.
  `ChannelFrame.leq(freq_weighting="Z")`で各チャンネルの等価騒音レベルを、
  `ChannelFrame.percentile_levels(n, freq_weighting, time_weighting)`で
  `sound_level(dB=True)`の時系列から時間率レベル（`n` %の時間超えるレベル。L_5、L_50、
  L_95など）を求められるようになりました。どちらも`rms`と同じくその場で計算し、Frameではなく
  NumPy配列を返します。校正済みのPaチャンネルの`leq("A")`はdB SPLのL_Aeqで、
  `channels[i].level_reference.to_level(frame.a_weighting().rms[i])`と等しくなります。
