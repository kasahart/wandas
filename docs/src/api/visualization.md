# Visualization Module / 可視化モジュール

The `wandas.visualization` module provides functionality for visually representing audio data.
`wandas.visualization` モジュールは、オーディオデータを視覚的に表現するための機能を提供します。

## Plotting / プロッティング

Provides plotting functions for visualizing audio data.
オーディオデータを視覚化するためのプロッティング関数を提供します。

::: wandas.visualization.plotting

## Describe presentation / Describe presentation

`ChannelFrame.describe()` is the public entry point. It delegates Figure creation,
image saving, and lifecycle handling to `wandas.visualization.describe`; optional
IPython Figure/Audio display is isolated in `wandas.visualization.notebook`.

`ChannelFrame.describe()` が公開 entry point です。Figure 作成・画像保存・lifecycle は
`wandas.visualization.describe`、optional IPython Figure/Audio display は
`wandas.visualization.notebook` に分離されています。
