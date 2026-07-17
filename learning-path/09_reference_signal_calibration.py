import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    # この教材で使う公開APIと結果確認用ライブラリを読み込む
    import marimo as mo
    import numpy as np
    import pandas as pd

    import wandas as wd

    return mo, np, pd, wd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 校正信号からチャンネル係数を求める

    センサ証明書にfactorが直接記載されていない場合でも、既知の物理値を発生する
    校正器の収録からfactorを求められます。マイクと加速度計の校正を同時に収録する
    必要はありません。センサごとに別の時刻・別のファイルで収録し、同じチャンネル
    ラベルを使って測定データへ対応付けます。

    この教材では次を確認します。

    1. 94 dBの音響校正信号からマイクのfactorを求める
    2. 既知RMSの振動校正信号から加速度計のfactorを求める
    3. 別々に求めたラベル辞書をまとめて測定Frameへ設定する
    4. 校正後の物理値を`frame.data`から取得する
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 別々に収録した校正信号を用意する

    factorは `known physical RMS / recorded RMS` です。音響校正器は既知レベルを、
    振動校正器は既知の物理RMSを指定できます。ここでは計算を目で確認できるよう、
    RMSがそれぞれ`0.5`と`0.25`になる短い信号を使います。
    """)
    return


@app.cell
def _(mo, np, pd, wd):
    # マイクと加速度計の校正信号を別々のFrameとして作る
    microphone_reference = wd.from_numpy(
        np.array([0.5, -0.5, 0.5, -0.5]),
        sampling_rate=8_000,
        ch_labels=["microphone"],
    )
    accelerometer_reference = wd.from_numpy(
        np.array([0.25, -0.25, 0.25, -0.25]),
        sampling_rate=8_000,
        ch_labels=["accelerometer"],
    )
    _reference_summary = pd.DataFrame(
        {
            "channel": ["microphone", "accelerometer"],
            "recorded RMS": [microphone_reference.rms[0], accelerometer_reference.rms[0]],
            "known target": ["94 dB re 20 µPa", "1.0 m/s^2 RMS"],
        }
    )

    mo.vstack([mo.md("**別々に収録した校正信号**"), _reference_summary])
    return accelerometer_reference, microphone_reference


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 既知値からfactorを導出する

    `derive_calibration()`は各チャンネルのRMSを測定し、ラベルをキーにした
    `ChannelCalibration`辞書を返します。音圧レベルを指定した場合は
    `target RMS = ref × 10 ** (level / 20)`で物理RMSへ変換します。

    戻り値はそのまま`with_calibration()`へ渡せます。複数回の校正結果は、通常の
    Python辞書としてまとめられます。
    """)
    return


@app.cell
def _(accelerometer_reference, microphone_reference, mo, np, pd):
    # 音と加速度の既知値からfactorを別々に求め、ラベル辞書を統合する
    microphone_calibration = microphone_reference.derive_calibration(
        target_level=94.0,
        unit="Pa",
    )
    accelerometer_calibration = accelerometer_reference.derive_calibration(
        target_rms=1.0,
        unit="m/s^2",
        ref=1.0,
    )
    derived_calibrations = {
        **microphone_calibration,
        **accelerometer_calibration,
    }
    _measured_rms_by_label = {
        "microphone": microphone_reference.rms[0],
        "accelerometer": accelerometer_reference.rms[0],
    }
    calibration_table = pd.DataFrame(
        [
            {
                "channel": label,
                "recorded RMS": _measured_rms_by_label[label],
                "factor": calibration.factor,
                "physical RMS": _measured_rms_by_label[label] * calibration.factor,
                "unit": calibration.unit,
                "ref": calibration.ref,
            }
            for label, calibration in derived_calibrations.items()
        ]
    )

    _microphone_target_rms = 2e-5 * 10 ** (94.0 / 20.0)
    np.testing.assert_allclose(calibration_table["physical RMS"], [_microphone_target_rms, 1.0])
    mo.vstack([mo.md("**校正信号から導出したラベル別factor**"), calibration_table])
    return (derived_calibrations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    この表はCSVや設備台帳へ保存できる形です。後日factorを再利用する場合は
    [既知の校正値を設定する教材](07_per_channel_calibration.html)と同じ
    `label -> ChannelCalibration`辞書へ戻します。

    次は、校正信号とは別に収録した2ch測定データへ、導出結果を一度に設定します。
    """)
    return


@app.cell
def _(derived_calibrations, mo, np, pd, wd):
    # ラベルで音・加速度のfactorを対応付け、校正後の物理値をdataから読む
    measurement = wd.from_numpy(
        np.array(
            [
                [1.0, -1.0],
                [2.0, -2.0],
            ]
        ),
        sampling_rate=8_000,
        ch_labels=["microphone", "accelerometer"],
    )
    calibrated_measurement = measurement.with_calibration(derived_calibrations)
    _recorded_values = measurement.data
    _physical_values = calibrated_measurement.data
    _result_table = pd.DataFrame(
        {
            "channel": calibrated_measurement.labels,
            "recorded first sample": _recorded_values[:, 0],
            "physical first sample": _physical_values[:, 0],
            "unit": [channel.unit for channel in calibrated_measurement.channels],
        }
    )

    assert [channel.unit for channel in calibrated_measurement.channels] == ["Pa", "m/s^2"]
    np.testing.assert_array_equal(measurement.data, _recorded_values)
    mo.vstack([mo.md("**別収録の測定データへ適用した結果**"), _result_table])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    - 校正信号のチャンネルラベルを測定データのラベルと一致させる
    - 既知RMSまたは既知レベルを`derive_calibration()`へ渡す
    - センサごとに別々に導出した辞書は`{**a, **b}`でまとめられる
    - 導出した辞書を測定Frameの`with_calibration()`へ渡す
    - 校正後の物理値は`frame.data`から取得する

    `derive_calibration()`はfactorを求めるために校正信号のRMSを読みます。
    `with_calibration()`は元の測定Frameを変更せず、新しいFrameを返します。
    """)
    return


if __name__ == "__main__":
    app.run()
