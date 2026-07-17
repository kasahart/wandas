# Core Module / コアモジュール

The `wandas.core` module provides the foundation components of the Wandas library.
`wandas.core` モジュールは、Wandasライブラリの基盤となるコンポーネントを提供します。

## BaseFrame

BaseFrame is the base class for all Wandas frames. It defines the basic data structure and operations.
BaseFrameはすべてのWandasフレームの基底クラスです。これは基本的なデータ構造と操作を定義します。

::: wandas.core.base_frame.BaseFrame

## ChannelMetadata

The ChannelMetadata class manages metadata related to audio channels.
ChannelMetadataクラスはオーディオデータのチャンネルに関連するメタデータを管理します。

::: wandas.core.metadata.ChannelMetadata

## ChannelCalibration

`ChannelCalibration` stores the immutable raw-to-physical factor, unit, and
level reference for one channel. See the
<a href="../learning-path/07_per_channel_calibration.html">per-channel calibration
learning app</a> for list, mapping, CSV, and 100-channel examples.

::: wandas.core.metadata.ChannelCalibration
