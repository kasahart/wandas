# Use Linear Measurements and Reference-Relative Levels

Keep calibration on the `ChannelFrame`, then use the same channel context for
linear RMS/peak values and level values. Do not pass a separate `unit` or `ref`
to every analysis call, and do not construct a level-scale object yourself.

校正は`ChannelFrame`に保持し、線形RMS／peak値とlevel値の両方で同じchannel文脈を
使用します。解析のたびに別の`unit`や`ref`を渡したり、level scale objectを組み立てたり
する必要はありません。

```python
import wandas as wd

audio = wd.read("measurement.wav", ch_labels=["mic"])
calibrated = audio.with_calibration(
    {
        "mic": wd.ChannelCalibration(factor=0.42, unit="Pa"),
    }
)

linear_rms = calibrated.rms
rms_levels = calibrated.rms_level
peak_levels = calibrated.peak_level

spectrum = calibrated.fft()
spectral_levels = spectrum.dB

reference = calibrated.channels[0].level_reference
print(reference.unit)             # dB SPL
print(reference.reference_value)  # 2e-05
print(reference.reference_unit)   # Pa
print(reference.label)            # dB SPL re 20 µPa
```

`rms` and `peak` remain linear amplitudes in each channel's calibrated unit.
`rms_level`, `peak_level`, FFT `.dB`, and STFT `.dB` use
`20 * log10(magnitude / reference)` with the magnitude/reference ratio floored
at `1e-12`; zero therefore returns `-240 dB`. These eager NumPy properties do
not add lineage or Recipe nodes. FFT and STFT construction remains Dask-lazy;
accessing `.dB` materializes their data.

`rms`と`peak`は各channelの校正済みunitにおける線形振幅のままです。`rms_level`、
`peak_level`、FFTの`.dB`、STFTの`.dB`は、振幅／基準値の比を`1e-12`でfloorした
`20 * log10(magnitude / reference)`を使用するため、zeroは`-240 dB`になります。
これらのeager NumPy propertyはlineageやRecipe nodeを追加しません。FFT／STFTの構築は
Dask-lazyのままで、`.dB`へのアクセスがdataを実体化します。

## Read the level descriptor / level descriptorを読む

`level_reference` is structured so applications can keep numbers and display
text separate:

`level_reference`は構造化されているため、applicationは数値と表示textを分けて扱えます。

| Field | Pa / 20 µPa | acceleration / 1 m/s² | canonical audio / FS |
| --- | --- | --- | --- |
| `unit` | `dB SPL` | `dB` | `dBFS` |
| `reference_value` | `2e-5` | `1.0` | `1.0` |
| `reference_unit` | `Pa` | `m/s^2` | `FS` |
| `label` | `dB SPL re 20 µPa` | `dB re 1 m/s^2` | `dBFS` |

Canonical labels use readable engineering-prefix text, so a `2e-5 V`
reference is displayed as `dB re 20 µV`. Other values use a stable
significant-digit representation rather than exposing binary floating-point
noise. This affects only `label`; `reference_value` retains the exact normalized
float supplied by the channel calibration.

canonical labelは読みやすいengineering-prefix表記を使うため、`2e-5 V`のreferenceは
`dB re 20 µV`と表示されます。その他の値もbinary floating-point noiseを露出しない
安定した有効桁表現になります。この表示整形は`label`だけに適用され、
`reference_value`はchannel calibrationに指定された正規化済みfloatをそのまま保持します。

`wd.read()` marks SoundFile-decoded audio as `FS` because that reader returns
canonical full-scale floats. `wd.from_numpy()`, CSV input, and a default
`ChannelCalibration()` keep an empty unit. Wandas never infers dBFS merely from
`factor=1`, an empty unit, and `ref=1`.

`wd.read()`はSoundFileでdecodeしたaudioを`FS`として明示します。このreaderがcanonical
full-scale floatを返すためです。`wd.from_numpy()`、CSV入力、defaultの
`ChannelCalibration()`は空unitのままです。Wandasは`factor=1`、空unit、`ref=1`だけを
根拠にdBFSと推測しません。

For an already-calibrated external amplitude, use the descriptor directly:
すでに校正済みの外部振幅にはdescriptorを直接使用できます。

```python
reported_peak = calibrated.channels[0].level_reference.to_level(0.2)
```

The argument is unambiguously in the channel's linear domain; the method does
not apply `ChannelCalibration.factor` again. Scalar input returns a `float`.
Array-like input preserves its NumPy broadcast shape, signs and complex phases
are ignored through magnitude, and invalid non-positive references are rejected
when the descriptor is created.

引数はchannelの線形domainにある値として明確に解釈され、
`ChannelCalibration.factor`を再適用しません。scalar入力は`float`を返し、array-like入力は
NumPy broadcast shapeを維持します。符号とcomplex phaseはmagnitudeによって除かれ、
0以下の不正なreferenceはdescriptor作成時に拒否されます。
