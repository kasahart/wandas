import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    # この教材で使う公開APIと表形式の確認用ライブラリを読み込む
    import io
    import pathlib
    import tempfile

    import marimo as mo
    import numpy as np
    import pandas as pd

    import wandas as wd
    from wandas import pipeline as pipeline_api

    return io, mo, np, pathlib, pd, pipeline_api, tempfile, wd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 既知の校正値をチャンネルへ設定する

    マイクや加速度計の校正値は、同時に収録した校正信号ではなく、センサ証明書、
    設備台帳、CSVなどで別々に管理されていることがあります。`with_calibration()`を使うと、
    その既知係数をチャンネルへ対応付け、元のFrameを変更せずに物理値を計算できます。

    この教材では次を確認します。

    1. 音と加速度へ異なる単位・基準値・係数を設定する
    2. CSVをラベル辞書へ変換し、並び順に依存せず適用する
    3. 係数列のNumPy配列と100chの管理表を一括適用する
    4. 校正後の値を`frame.data`で取得し、RecipeやWDFでも再現する

    Wandasは、ここで渡す係数を校正信号から推定しません。証明書や管理系で確定した
    **収録値から物理値への変換係数**をFrameへ設定するところから始めます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 収録値と物理値の関係を作る

    各チャンネルの数値処理には `physical = recorded * factor` を使います。
    `unit`は結果の物理単位、`ref`はレベル値を求めるときの基準値です。
    `ref`は収録値へ掛ける係数ではありません。

    まず、マイクと加速度計を同じFrameに持つ小さな2ch信号を作ります。
    """)
    return


@app.cell
def _(mo, np, pd, wd):
    # 音と加速度の校正前サンプルを持つ2ch Frameを作る
    recorded_signal = wd.from_numpy(
        np.array(
            [
                [10.0, 20.0, 30.0, 40.0],
                [0.1, 0.2, 0.3, 0.4],
            ]
        ),
        sampling_rate=8_000,
        ch_labels=["microphone", "accelerometer"],
    )
    _recorded_preview = pd.DataFrame(
        recorded_signal.data.T,
        columns=recorded_signal.labels,
    )

    mo.vstack([mo.md("**校正前の収録値**"), _recorded_preview])
    return (recorded_signal,)


@app.cell
def _(mo, np, pd, recorded_signal, wd):
    # 2つのセンサへ、それぞれの既知係数と物理領域を設定する
    configured_signal = recorded_signal.with_calibration(
        [
            wd.ChannelCalibration(factor=0.02, unit="Pa"),
            wd.ChannelCalibration(factor=9.81, unit="m/s^2", ref=1.0),
        ]
    )

    _recorded_values = recorded_signal.data
    _physical_values = configured_signal.data
    np.testing.assert_allclose(
        _physical_values,
        _recorded_values * np.array([[0.02], [9.81]]),
    )
    _calibration_summary = pd.DataFrame(
        {
            "channel": configured_signal.labels,
            "recorded (first sample)": _recorded_values[:, 0],
            "factor": [channel.calibration.factor for channel in configured_signal.channels],
            "physical (first sample)": _physical_values[:, 0],
            "unit": [channel.unit for channel in configured_signal.channels],
            "ref": [channel.ref for channel in configured_signal.channels],
        }
    )

    mo.vstack([mo.md("**収録値 × factor が物理値になることを確認**"), _calibration_summary])
    return (configured_signal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    リストと1次元NumPy配列は、**現在のチャンネル順に完全置換**します。
    長さがチャンネル数と一致しなければエラーです。`Pa`の`ref`は省略すると
    `2e-5`になり、加速度のように別の基準値を使う場合は明示します。

    数値だけを渡した場合は、現在の`unit`と`ref`を維持してfactorだけを置き換えます。
    前のfactorとの掛け算にはなりません。元のFrameも変更されません。
    """)
    return


@app.cell
def _(configured_signal, mo, pd):
    # 再発行された係数でfactorだけを置換し、元Frameとの違いを確認する
    _reissued_signal = configured_signal.with_calibration([0.025, 9.75])
    assert configured_signal.channels[0].calibration.factor == 0.02
    assert _reissued_signal.channels[0].unit == "Pa"
    assert _reissued_signal.channels[1].ref == 1.0

    _replacement_summary = pd.DataFrame(
        {
            "channel": configured_signal.labels,
            "before": [channel.calibration.factor for channel in configured_signal.channels],
            "after": [channel.calibration.factor for channel in _reissued_signal.channels],
            "unit preserved": [channel.unit for channel in _reissued_signal.channels],
        }
    )
    mo.vstack([mo.md("**factorは置換され、元Frameは不変**"), _replacement_summary])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CSV管理表をラベルで適用する

    長期運用では、順序に依存しないラベル指定が基本です。CSVの行を
    `label -> ChannelCalibration` の辞書へ変換すれば、CSV、データベース、
    設備管理APIのどれを使ってもFrame側の契約は変わりません。

    次のCSVは、Frameとは逆の順序で加速度計を先に記録しています。
    """)
    return


@app.cell
def _(io, mo, pd):
    # Frameとは逆順のCSVを読み、外部管理表の内容を見える形にする
    calibration_table = pd.read_csv(
        io.StringIO("channel,factor,unit,ref\naccelerometer,9.79,m/s^2,1.0\nmicrophone,0.021,Pa,0.00002\n")
    )
    mo.vstack([mo.md("**センサごとに管理された校正値**"), calibration_table])
    return (calibration_table,)


@app.cell
def _(calibration_table, mo, pd, recorded_signal, wd):
    # CSVの各行を完全な校正値へ変換し、チャンネルラベルで対応付ける
    _calibration_by_label = {
        str(row.channel): wd.ChannelCalibration(
            factor=float(row.factor),
            unit=str(row.unit),
            ref=float(row.ref),
        )
        for row in calibration_table.itertuples(index=False)
    }
    _csv_configured_signal = recorded_signal.with_calibration(_calibration_by_label)
    _csv_result = pd.DataFrame(
        {
            "frame order": _csv_configured_signal.labels,
            "factor": [channel.calibration.factor for channel in _csv_configured_signal.channels],
            "unit": [channel.unit for channel in _csv_configured_signal.channels],
        }
    )

    assert _csv_result["factor"].tolist() == [0.021, 9.79]
    mo.vstack([mo.md("**CSVの行順ではなくラベルで対応した結果**"), _csv_result])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    unitとrefがすでに正しいFrameなら、係数列だけを1次元NumPy配列として渡せます。
    配列は現在のチャンネル順である必要があるため、管理表をラベルで並べ直してから取り出します。
    """)
    return


@app.cell
def _(calibration_table, configured_signal, mo, pd):
    # CSVの係数列をFrameの順序へ並べ、NumPy配列として完全置換する
    _ordered_factors = calibration_table.set_index("channel").loc[configured_signal.labels, "factor"].to_numpy()
    _array_configured_signal = configured_signal.with_calibration(_ordered_factors)
    _array_result = pd.DataFrame(
        {
            "channel": _array_configured_signal.labels,
            "factor from array": [channel.calibration.factor for channel in _array_configured_signal.channels],
            "unit preserved": [channel.unit for channel in _array_configured_signal.channels],
        }
    )

    mo.vstack([mo.md("**係数列を現在のチャンネル順で適用**"), _array_result])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 100chを1回の操作で設定する

    チャンネル数が増えても、chごとのメソッド呼出しは不要です。全chなら生成したリスト、
    一部の証明書だけ更新された場合は対象ラベルの辞書を1回渡します。
    リストも辞書もch数に対して線形に検証されます。ここでは同じ物理領域を持つ
    100台の加速度計を想定し、初回はfactor、unit、refを含む完全な校正値を設定します。
    """)
    return


@app.cell
def _(mo, np, pd, wd):
    # 管理表から生成した100校正値を一括設定し、10chごとのfactor更新を重ねる
    _channel_count = 100
    _labels = [f"sensor-{index:03d}" for index in range(_channel_count)]
    _hundred_recorded = wd.from_numpy(
        np.ones((_channel_count, 16)),
        sampling_rate=8_000,
        ch_labels=_labels,
    )
    _all_factors = 1.0 + np.arange(_channel_count) / 1_000
    _all_calibrations = [wd.ChannelCalibration(factor=float(factor), unit="m/s^2", ref=1.0) for factor in _all_factors]
    _configured_hundred = _hundred_recorded.with_calibration(_all_calibrations)
    _partially_updated = _configured_hundred.with_calibration(
        {f"sensor-{index:03d}": 2.0 for index in range(0, _channel_count, 10)}
    )

    _hundred_values = _partially_updated.data
    assert _partially_updated.n_channels == 100
    assert _hundred_values.shape == (100, 16)
    _inspect_indices = [0, 1, 10, 90, 91]
    _hundred_result = pd.DataFrame(
        {
            "index": _inspect_indices,
            "channel": [_partially_updated.labels[index] for index in _inspect_indices],
            "factor": [_partially_updated.channels[index].calibration.factor for index in _inspect_indices],
            "physical (first sample)": _hundred_values[_inspect_indices, 0],
            "unit": [_partially_updated.channels[index].unit for index in _inspect_indices],
        }
    )

    mo.vstack([mo.md("**100chのうち代表5chを確認**"), _hundred_result])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 校正後の値を`data`で利用する

    利用者が数値を取り出す入口は`frame.data`です。`with_calibration()`は元のFrameを
    変更せず、新しいFrameを返します。FFTなどの後続処理にも校正後の物理値が自動的に渡ります。
    """)
    return


@app.cell
def _(configured_signal, mo, np, pd, recorded_signal, wd):
    # 元Frameの不変性と、後続処理が校正済みの値を使うことを確認する
    _recorded_values = recorded_signal.data
    _calibrated_values = configured_signal.data
    np.testing.assert_array_equal(_recorded_values[0], [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(_calibrated_values[0], [0.2, 0.4, 0.6, 0.8])
    _spectrum = configured_signal.fft(n_fft=4, window="boxcar")
    _expected_spectrum = wd.from_numpy(
        _calibrated_values,
        sampling_rate=configured_signal.sampling_rate,
        ch_labels=configured_signal.labels,
        ch_units=[channel.unit for channel in configured_signal.channels],
    ).fft(n_fft=4, window="boxcar")
    np.testing.assert_allclose(_spectrum.data, _expected_spectrum.data)

    _boundary_result = pd.DataFrame(
        {
            "確認対象": ["校正前Frameの先頭値", "校正後Frameの先頭値", "FFTが物理値を使用"],
            "結果": [_recorded_values[0, 0], _calibrated_values[0, 0], "yes"],
        }
    )
    mo.vstack([mo.md("**`data`から後続処理まで同じ物理値を使用**"), _boundary_result])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RecipeとWDFで対応関係を持ち運ぶ

    Recipeは一時的な位置ではなく、解決済みの安定ch IDと校正値のスナップショットを保持します。
    WDFはFrameと現在の校正メタデータをまとめて保存します。
    """)
    return


@app.cell
def _(mo, np, pd, pipeline_api, recorded_signal, wd):
    # 並べ替え後の校正操作をRecipe化し、元入力へ同じ対応関係で再実行する
    _workflow = recorded_signal.get_channel([1, 0]).with_calibration(
        [
            wd.ChannelCalibration(9.81, "m/s^2", 1.0),
            wd.ChannelCalibration(0.02, "Pa"),
        ]
    )
    _plan = pipeline_api.RecipePlan.from_frame(_workflow, input_names=("signal",))
    _replayed = _plan.apply({"signal": recorded_signal})
    np.testing.assert_allclose(_replayed.data, _workflow.data)

    _recipe_result = pd.DataFrame(
        {
            "channel": _replayed.labels,
            "factor": [channel.calibration.factor for channel in _replayed.channels],
            "unit": [channel.unit for channel in _replayed.channels],
        }
    )
    mo.vstack([mo.md("**並べ替えを含むRecipe replayの結果**"), _recipe_result])
    return


@app.cell
def _(configured_signal, mo, np, pathlib, pd, tempfile, wd):
    # 一時WDFを往復し、dataと校正値が復元されることを確認する
    with tempfile.TemporaryDirectory() as _temporary_directory:
        _wdf_path = pathlib.Path(_temporary_directory) / "calibrated.wdf"
        configured_signal.save(_wdf_path)
        _loaded_signal = wd.load(_wdf_path)
        _loaded_physical = _loaded_signal.data
        _loaded_factors = [channel.calibration.factor for channel in _loaded_signal.channels]

    np.testing.assert_allclose(_loaded_physical, configured_signal.data)
    _wdf_result = pd.DataFrame(
        {
            "preserved": ["frame.data", "calibration factors"],
            "result": ["yes", str(_loaded_factors)],
        }
    )
    mo.vstack([mo.md("**WDF round-tripで保持された内容**"), _wdf_result])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    - 証明書やCSVで確定した収録値から物理値への係数を`with_calibration()`へ渡す
    - 最初の設定では`ChannelCalibration`でfactor、unit、refをまとめて指定する
    - 長期運用や部分更新ではラベル辞書、完全置換では現在順のリスト／1次元配列を使う
    - 100chでも管理表から値を生成し、1回の操作で設定する
    - 校正後の物理値は`frame.data`からNumPy配列として取得する
    - 元のFrameは変わらず、派生Frame、Recipe、WDFでも同じ物理値を再現できる

    API signatureは[Frames API reference](../api/frames/)を、WDFの詳細は
    [WDF File I/O](../api/wdf_io/)を参照してください。
    """)
    return


if __name__ == "__main__":
    app.run()
