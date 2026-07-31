# Datasets Module / データセットモジュール

`wandas.datasets` currently exports no sample datasets, catalog, or packaged audio
assets; its `__all__` is empty. The repository checkout contains files used by its
learning material, but installed applications must not depend on those as a package
dataset API.
`wandas.datasets` は現在、sample dataset、catalog、package同梱audio assetをexportせず、
`__all__` は空です。repository checkoutにはlearning material用のfileがありますが、
installされたapplicationがpackage dataset APIとして依存することはできません。

For a self-contained known signal, use the stable top-level
`wd.generate_sin()` helper documented in [Utilities](utils.md). For recordings owned
by an application, use stable `wd.read()` or create a lazy `ChannelFrameDataset` with
`wd.from_folder()`.
自己完結した既知信号には、[Utilities](utils.md)に記載したstableなtop-level helper
`wd.generate_sin()`を使用します。application所有のrecordingにはstableな`wd.read()`を使うか、
`wd.from_folder()`で遅延`ChannelFrameDataset`を作成します。

`wandas.datasets.sample_data.load_sample_signal` is an internal implementation name,
not a public export or a promise of shipped audio.
`wandas.datasets.sample_data.load_sample_signal` は内部実装名であり、public exportでも
同梱audioの約束でもありません。
