# Convert Linear Measurements to Reference-Relative Levels

Use each linear channel's `level_reference` as the single source for measurement
level conversion and labels.

各linear channelの`level_reference`を、measurement level変換とlabelの唯一の正本として
使用します。

```python
import wandas as wd

frame = wd.read("audio.wav")
reference = frame.channels[0].level_reference

rms = frame.rms[0]
rms_db = reference.to_level(rms)

spectrum_db = frame.fft().dB
spectrogram_db = frame.stft().dB
```

`reference_value` and `reference_unit` describe the linear amplitude
reference. `unit` is `dBFS`, `dB SPL`, or generic `dB`; `label` is
canonical display text.

`reference_value`と`reference_unit`はlinear amplitude referenceを表します。
`unit`は`dBFS`、`dB SPL`、generic `dB`のいずれかで、`label`はcanonicalな
表示textです。

```python
measurement = {
    "referenceValue": reference.reference_value,
    "referenceUnit": reference.reference_unit,
    "levelUnit": reference.unit,
    "levelReferenceLabel": reference.label,
}
```

`to_level()` accepts amplitudes that are already in the channel's linear
domain. It does not apply `ChannelCalibration.factor` again. It uses
`20 * log10(abs(amplitude) / reference)`, with the amplitude/reference ratio
floored at `1e-12`; zero is therefore `-240 dB`. Scalar input returns a
`float`; array-like input preserves its NumPy shape. Signed and complex input
uses magnitude.

`to_level()`はchannelのlinear domainにすでにある振幅を受け取り、
`ChannelCalibration.factor`を再適用しません。振幅／reference比を`1e-12`でfloorした
`20 * log10(abs(amplitude) / reference)`を使用するため、zeroは`-240 dB`です。
scalar入力は`float`、array-like入力はshapeを維持したNumPy arrayになり、signed／complex
入力にはmagnitudeを使います。

## Calibrated channels / 校正済みchannel

The same API applies after calibration:

校正後も同じAPIを使用します。

```python
calibrated = frame.with_calibration(
    {
        "mic": wd.ChannelCalibration(
            factor=0.42,
            unit="Pa",
        )
    }
)

reference = calibrated.channels[0].level_reference
assert reference.label == "dB SPL re 20 µPa"

rms_db = reference.to_level(calibrated.rms[0])
```

Pa relative to 20 µPa uses `dB SPL`. Only the explicit `FS` / reference 1
metadata set by SoundFile-backed reading uses `dBFS`. `from_numpy()`, CSV,
and default identity calibration remain generic `dB re 1 input unit`.

Pa／20 µPaは`dB SPL`です。SoundFile-backed readが設定する明示的な`FS`／reference 1
だけが`dBFS`になります。`from_numpy()`、CSV、default identity calibrationはgenericな
`dB re 1 input unit`のままです。

FFT and STFT `.dB` use the same internal amplitude-level numerical contract as
`to_level()` without changing their existing public shapes. This contract is
for linear channel context; it does not define persistence or reinterpretation
of materialized level-domain Frames.

FFT／STFTの`.dB`は、既存public shapeを変えずに`to_level()`と同じ内部amplitude-level
数値契約を使用します。この契約はlinear channel context向けであり、materialized level-domain
Frameの永続化や再解釈は定義しません。
