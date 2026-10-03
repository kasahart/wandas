# Unreleased changes after Wandas 0.8.0

These changes are not part of the published 0.8.0 tag. The development package
version remains 0.8.0 until a separately prepared release; no release is published
by this feature PR.

- Add header-only `wd.inspect()` without PCM decoding or Frame graph construction.
- Add explicit physical `frame_center_times` and optional `frame_time_origin`,
  including the `SpectrogramFrame.from_numpy()` factory. Existing `times` and
  `source_times` retain their contracts.

## WDF compatibility and migration

Saving SpectrogramFrame/CepstrogramFrame with a known physical origin writes
WDF 0.5. Other supported results, including unknown origins, retain WDF 0.4.
Readers from released 0.8.0 and earlier reject 0.5; upgrade to a release containing
this extension before reopening these files. Coordinate release upgrades for
producers and consumers before exchanging known-origin results.

The new reader still loads WDF 0.4 without changing stored data or legacy axes.
Physical centers remain unknown for those files. Recompute the STFT from the
original audio or supply a known origin from the original processing settings;
do not infer it from display history. Existing 0.4 artifacts need no bulk migration.

公開済み0.8.0の互換性契約は変更しません。上記は未公開開発版の変更です。
既知起点の結果を交換する場合は、作成側・読込側を0.5対応版へ揃えてください。
