# Error Message Inventory

**Generated**: 2025-10-22  
**Total Errors**: 100  
**Analysis Source**: Automated scan of wandas codebase

## Summary Statistics

### By Priority
- **HIGH Priority** (Score 0-1): 70 errors - Need complete rewrite
- **MEDIUM Priority** (Score 2): 28 errors - Add missing element
- **LOW Priority** (Score 3): 2 errors - Already good

### By Error Type
- **FileExistsError**: 1
- **FileNotFoundError**: 3
- **IndexError**: 5
- **KeyError**: 2
- **NotImplementedError**: 13
- **TypeError**: 16
- **ValueError**: 60

### By Module (Top 10)
- **frames.channel**: 23 errors (avg quality: 0.7/3)
- **core.base_frame**: 13 errors (avg quality: 1.3/3)
- **frames.roughness**: 7 errors (avg quality: 1.0/3)
- **visualization.plotting**: 7 errors (avg quality: 1.0/3)
- **utils.frame_dataset**: 7 errors (avg quality: 0.6/3)
- **io.wdf_io**: 5 errors (avg quality: 0.6/3)
- **processing.filters**: 5 errors (avg quality: 2.0/3)
- **processing.base**: 5 errors (avg quality: 0.8/3)
- **frames.spectrogram**: 4 errors (avg quality: 0.0/3)
- **io.readers**: 4 errors (avg quality: 0.5/3)

---

## HIGH Priority Errors (Need Complete Rewrite)

These errors lack critical information (score 0-1) and need to be rewritten following the 3-element rule (WHAT/WHY/HOW).

### frames.roughness - Line 127
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `            raise ValueError(                 fExpected 47 Bark bands; got {data.shape[-2]} "      `
- **Location**: `wandas/frames/roughness.py:127`

### frames.roughness - Line 133
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `            raise ValueError(fbark_axis must have 47 elements; got {len(bark_axis)}")`
- **Location**: `wandas/frames/roughness.py:133`

### frames.roughness - Line 299
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(                     fSampling rates do not match: {self.sampling_`
- **Location**: `wandas/frames/roughness.py:299`

### frames.roughness - Line 305
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(                     fShape mismatch: {self._data.shape} vs {other`
- **Location**: `wandas/frames/roughness.py:305`

### frames.mixins.channel_processing_mixin - Line 186
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `            raise ValueError(fUnsupported reduction operation: {op}")`
- **Location**: `wandas/frames/mixins/channel_processing_mixin.py:186`

### frames.mixins.channel_processing_mixin - Line 756
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `            raise ValueError(Operation did not provide bark_axis in metadata")`
- **Location**: `wandas/frames/mixins/channel_processing_mixin.py:756`

### frames.spectrogram - Line 122
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `            raise ValueError(                 fデータは2次元または3次元である必要があります。形状: {data.shape}"           `
- **Location**: `wandas/frames/spectrogram.py:122`

### frames.spectrogram - Line 126
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `            raise ValueError(                 fデータの形状が無効です。周波数ビン数は {n_fft // 2 + 1} である必要があります。"  #`
- **Location**: `wandas/frames/spectrogram.py:126`

### frames.spectrogram - Line 373
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(                     サンプリングレートが一致していません。演算できません。"                 `
- **Location**: `wandas/frames/spectrogram.py:373`

### frames.noct - Line 248
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHAT, HOW
- **Current**: `            raise ValueError(freqs is not numpy array.")`
- **Location**: `wandas/frames/noct.py:248`

### frames.spectral - Line 328
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `                raise ValueError(                     Sampling rates do not match. Cannot perform o`
- **Location**: `wandas/frames/spectral.py:328`

### frames.spectral - Line 583
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHAT, WHY
- **Current**: `            raise ValueError(                 noct_synthesis can only be used with a sampling rate `
- **Location**: `wandas/frames/spectral.py:583`

### frames.channel - Line 219
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `                raise ValueError(                     Sampling rates do not match. Cannot perform o`
- **Location**: `wandas/frames/channel.py:219`

### frames.channel - Line 305
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `                raise ValueError(                     Sampling rates do not match. Cannot perform o`
- **Location**: `wandas/frames/channel.py:305`

### frames.channel - Line 613
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(                     Number of channel labels does not match the n`
- **Location**: `wandas/frames/channel.py:613`

### frames.channel - Line 623
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(                     Number of channel units does not match the nu`
- **Location**: `wandas/frames/channel.py:623`

### frames.channel - Line 750
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(                     fChannel specification is out of range: {chan`
- **Location**: `wandas/frames/channel.py:750`

### frames.channel - Line 758
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                    raise ValueError(                         fChannel specification is out of rang`
- **Location**: `wandas/frames/channel.py:758`

### frames.channel - Line 787
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `                raise ValueError(Unexpected data type after reading file")`
- **Location**: `wandas/frames/channel.py:787`

### frames.channel - Line 823
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(                     Number of channel labels does not match the n`
- **Location**: `wandas/frames/channel.py:823`

### frames.channel - Line 1013
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(sampling_rate不一致")`
- **Location**: `wandas/frames/channel.py:1013`

### frames.channel - Line 1044
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                    raise ValueError(データ長不一致: align指定を確認")`
- **Location**: `wandas/frames/channel.py:1044`

### frames.channel - Line 1056
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                        raise ValueError(flabel重複: {new_label}")`
- **Location**: `wandas/frames/channel.py:1056`

### frames.channel - Line 1106
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(データ長不一致: align指定を確認")`
- **Location**: `wandas/frames/channel.py:1106`

### frames.channel - Line 1113
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(label重複")`
- **Location**: `wandas/frames/channel.py:1113`

### io.readers - Line 113
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `                raise ValueError(Unexpected data type after reading file")`
- **Location**: `wandas/io/readers.py:113`

### io.readers - Line 219
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise ValueError(fRequested channels {channels} out of range")`
- **Location**: `wandas/io/readers.py:219`

### io.readers - Line 231
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `            raise ValueError(Unexpected data type after reading file")`
- **Location**: `wandas/io/readers.py:231`

### io.readers - Line 254
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `    raise ValueError(fNo suitable file reader found for {path_str}")`
- **Location**: `wandas/io/readers.py:254`

### io.wdf_io - Line 231
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `            raise ValueError(No channel data found in the file")`
- **Location**: `wandas/io/wdf_io.py:231`

### core.base_frame - Line 285
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                    raise ValueError(                         fBoolean mask length {len(key)} does `
- **Location**: `wandas/core/base_frame.py:285`

### core.base_frame - Line 302
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `                raise ValueError(Cannot index with an empty list")`
- **Location**: `wandas/core/base_frame.py:302`

### core.base_frame - Line 379
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `            raise ValueError(fInvalid key length: {len(key)} for shape {self.shape}")`
- **Location**: `wandas/core/base_frame.py:379`

### core.base_frame - Line 474
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHAT, HOW
- **Current**: `            raise ValueError(fComputed result is not a np.ndarray: {type(result)}")`
- **Location**: `wandas/core/base_frame.py:474`

### processing.spectral - Line 388
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHAT, HOW
- **Current**: `            raise ValueError(                 Welch operation requires a Dask array; but received a`
- **Location**: `wandas/processing/spectral.py:388`

### processing.temporal - Line 301
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `                raise ValueError(A_weighting returned an unexpected type.")`
- **Location**: `wandas/processing/temporal.py:301`

### processing.base - Line 153
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `        raise ValueError(fUnknown operation type: {name}")`
- **Location**: `wandas/processing/base.py:153`

### visualization.plotting - Line 435
- **Type**: ValueError
- **Score**: 1/3
- **Missing**: WHAT, HOW
- **Current**: `            raise ValueError(Overlay is not supported for SpectrogramPlotStrategy.")`
- **Location**: `wandas/visualization/plotting.py:435`

### visualization.plotting - Line 782
- **Type**: ValueError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `        raise ValueError(fUnknown plot type: {name}")`
- **Location**: `wandas/visualization/plotting.py:782`

### frames.roughness - Line 367
- **Type**: NotImplementedError
- **Score**: 1/3
- **Missing**: WHAT, HOW
- **Current**: `        raise NotImplementedError(             fOperation '{operation_name}' is not supported for R`
- **Location**: `wandas/frames/roughness.py:367`

### frames.noct - Line 274
- **Type**: NotImplementedError
- **Score**: 1/3
- **Missing**: WHAT, HOW
- **Current**: `        raise NotImplementedError(             fOperation {symbol} is not implemented for NOctFrame`
- **Location**: `wandas/frames/noct.py:274`

### frames.noct - Line 284
- **Type**: NotImplementedError
- **Score**: 1/3
- **Missing**: WHAT, HOW
- **Current**: `        raise NotImplementedError(             fOperation {operation_name} is not implemented for N`
- **Location**: `wandas/frames/noct.py:284`

### utils.frame_dataset - Line 250
- **Type**: NotImplementedError
- **Score**: 1/3
- **Missing**: WHAT, HOW
- **Current**: `        raise NotImplementedError(The save method is not currently implemented.")`
- **Location**: `wandas/utils/frame_dataset.py:250`

### utils.frame_dataset - Line 385
- **Type**: NotImplementedError
- **Score**: 1/3
- **Missing**: WHAT, WHY
- **Current**: `        raise NotImplementedError(_SampledFrameDataset does not load files directly.")`
- **Location**: `wandas/utils/frame_dataset.py:385`

### utils.frame_dataset - Line 633
- **Type**: NotImplementedError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `        raise NotImplementedError(             No method defined for directly loading SpectrogramFr`
- **Location**: `wandas/utils/frame_dataset.py:633`

### io.wdf_io - Line 68
- **Type**: NotImplementedError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `        raise NotImplementedError(             fFormat {format} not supported. Only 'hdf5' is curre`
- **Location**: `wandas/io/wdf_io.py:68`

### io.wdf_io - Line 168
- **Type**: NotImplementedError
- **Score**: 1/3
- **Missing**: WHAT, HOW
- **Current**: `        raise NotImplementedError(fFormat '{format}' is not supported")`
- **Location**: `wandas/io/wdf_io.py:168`

### processing.base - Line 97
- **Type**: NotImplementedError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `        raise NotImplementedError(Subclasses must implement this method.")`
- **Location**: `wandas/processing/base.py:97`

### processing.base - Line 120
- **Type**: NotImplementedError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `        raise NotImplementedError(Subclasses must implement this method.")`
- **Location**: `wandas/processing/base.py:120`

### visualization.plotting - Line 757
- **Type**: NotImplementedError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `        raise NotImplementedError()`
- **Location**: `wandas/visualization/plotting.py:757`

### frames.spectrogram - Line 636
- **Type**: IndexError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `            raise IndexError(                 f時間インデックス {time_idx} が範囲外です。有効範囲: 0-{self.n_frames - `
- **Location**: `wandas/frames/spectrogram.py:636`

### frames.channel - Line 1138
- **Type**: IndexError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `                raise IndexError(findex {key} out of range")`
- **Location**: `wandas/frames/channel.py:1138`

### utils.frame_dataset - Line 185
- **Type**: IndexError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `            raise IndexError(                 fIndex {index} is out of range (0-{len(self._lazy_fra`
- **Location**: `wandas/utils/frame_dataset.py:185`

### utils.frame_dataset - Line 378
- **Type**: IndexError
- **Score**: 1/3
- **Missing**: WHAT, WHY
- **Current**: `            raise IndexError(                 Indices are out of range for the original dataset. Or`
- **Location**: `wandas/utils/frame_dataset.py:378`

### utils.frame_dataset - Line 394
- **Type**: IndexError
- **Score**: 1/3
- **Missing**: WHAT, WHY
- **Current**: `            raise IndexError(                 fIndex {index} is out of range for the sampled datase`
- **Location**: `wandas/utils/frame_dataset.py:394`

### frames.channel - Line 563
- **Type**: TypeError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `                raise TypeError(                     fUnexpected type for plot result: {type(_ax)}.`
- **Location**: `wandas/frames/channel.py:563`

### frames.channel - Line 1086
- **Type**: TypeError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `            raise TypeError(add_channel: ndarray/dask/同型Frameのみ対応")`
- **Location**: `wandas/frames/channel.py:1086`

### core.base_frame - Line 318
- **Type**: TypeError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `                raise TypeError(                     fList must contain all str or all int; got mix`
- **Location**: `wandas/core/base_frame.py:318`

### core.base_frame - Line 339
- **Type**: TypeError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `        raise TypeError(             fInvalid key type: {type(key).__name__}. "             f"Expec`
- **Location**: `wandas/core/base_frame.py:339`

### core.base_frame - Line 391
- **Type**: TypeError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `            raise TypeError(                 fInvalid channel key type in tuple: {type(channel_key)`
- **Location**: `wandas/core/base_frame.py:391`

### processing.base - Line 143
- **Type**: TypeError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `        raise TypeError(Strategy class must inherit from AudioOperation.")`
- **Location**: `wandas/processing/base.py:143`

### processing.base - Line 145
- **Type**: TypeError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `        raise TypeError(Cannot register abstract AudioOperation class.")`
- **Location**: `wandas/processing/base.py:145`

### visualization.plotting - Line 767
- **Type**: TypeError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `        raise TypeError(Strategy class must inherit from PlotStrategy.")`
- **Location**: `wandas/visualization/plotting.py:767`

### visualization.plotting - Line 769
- **Type**: TypeError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `        raise TypeError(Cannot register abstract PlotStrategy class.")`
- **Location**: `wandas/visualization/plotting.py:769`

### frames.channel - Line 720
- **Type**: FileNotFoundError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `            raise FileNotFoundError(fFile not found: {path}")`
- **Location**: `wandas/frames/channel.py:720`

### utils.frame_dataset - Line 92
- **Type**: FileNotFoundError
- **Score**: 0/3
- **Missing**: WHAT, WHY, HOW
- **Current**: `            raise FileNotFoundError(fFolder does not exist: {self.folder_path}")`
- **Location**: `wandas/utils/frame_dataset.py:92`

### io.wdf_io - Line 172
- **Type**: FileNotFoundError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `        raise FileNotFoundError(fFile not found: {path}")`
- **Location**: `wandas/io/wdf_io.py:172`

### frames.channel - Line 1143
- **Type**: KeyError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `                raise KeyError(flabel {key} not found")`
- **Location**: `wandas/frames/channel.py:1143`

### core.base_frame - Line 428
- **Type**: KeyError
- **Score**: 1/3
- **Missing**: WHY, HOW
- **Current**: `        raise KeyError(fChannel label '{label}' not found.")`
- **Location**: `wandas/core/base_frame.py:428`

### io.wdf_io - Line 62
- **Type**: FileExistsError
- **Score**: 1/3
- **Missing**: WHAT, WHY
- **Current**: `        raise FileExistsError(             fFile {path} already exists. Set overwrite=True to overw`
- **Location**: `wandas/io/wdf_io.py:62`


---

## MEDIUM Priority Errors (Add Missing Element)

These errors have 2/3 elements and need only one addition (usually HOW).

### frames.roughness - Line 121
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(                 fData must be 2D or 3D (mono or multi-channel); got {`
- **Location**: `wandas/frames/roughness.py:121`

### frames.roughness - Line 137
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(foverlap must be in [0.0; 1.0]; got {overlap}")`
- **Location**: `wandas/frames/roughness.py:137`

### frames.mixins.channel_processing_mixin - Line 251
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(start must be less than end")`
- **Location**: `wandas/frames/mixins/channel_processing_mixin.py:251`

### frames.spectral - Line 123
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(                 fData must be 1-dimensional or 2-dimensional. Shape: `
- **Location**: `wandas/frames/spectral.py:123`

### frames.channel - Line 69
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(                 fData must be 1-dimensional or 2-dimensional. Shape: `
- **Location**: `wandas/frames/channel.py:69`

### frames.channel - Line 598
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(                 fData must be 1-dimensional or 2-dimensional. Shape: `
- **Location**: `wandas/frames/channel.py:598`

### frames.channel - Line 805
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `                raise ValueError(Chunk size must be a positive integer")`
- **Location**: `wandas/frames/channel.py:805`

### utils.generate_sample - Line 80
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `        raise ValueError(freqs must be a float or a list of floats.")`
- **Location**: `wandas/utils/generate_sample.py:80`

### io.wav_io - Line 93
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `        raise ValueError(target must be a ChannelFrame object.")`
- **Location**: `wandas/io/wav_io.py:93`

### processing.filters - Line 39
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(fCutoff frequency must be between 0 Hz and {limit} Hz")`
- **Location**: `wandas/processing/filters.py:39`

### processing.filters - Line 89
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(                 fCutoff frequency must be between 0 Hz and {self.samp`
- **Location**: `wandas/processing/filters.py:89`

### processing.filters - Line 153
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(                 fLower cutoff frequency must be between 0 Hz and {nyq`
- **Location**: `wandas/processing/filters.py:153`

### processing.filters - Line 157
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(                 fHigher cutoff frequency must be between 0 Hz and {ny`
- **Location**: `wandas/processing/filters.py:157`

### processing.filters - Line 161
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(                 fLower cutoff frequency ({self.low_cutoff} Hz) must b`
- **Location**: `wandas/processing/filters.py:161`

### processing.psychoacoustic - Line 497
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(foverlap must be in [0.0; 1.0]; got {self.overlap}")`
- **Location**: `wandas/processing/psychoacoustic.py:497`

### processing.psychoacoustic - Line 645
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(foverlap must be in [0.0; 1.0]; got {self.overlap}")`
- **Location**: `wandas/processing/psychoacoustic.py:645`

### processing.temporal - Line 164
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `                raise ValueError(Either length or duration must be provided.")`
- **Location**: `wandas/processing/temporal.py:164`

### visualization.plotting - Line 438
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise ValueError(ax must be None when n_channels > 1.")`
- **Location**: `wandas/visualization/plotting.py:438`

### visualization.plotting - Line 499
- **Type**: ValueError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `                raise ValueError(fig must be a matplotlib Figure object.")`
- **Location**: `wandas/visualization/plotting.py:499`

### frames.mixins.channel_collection_mixin - Line 40
- **Type**: NotImplementedError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `        raise NotImplementedError(add_channel() must be implemented in subclasses")`
- **Location**: `wandas/frames/mixins/channel_collection_mixin.py:40`

### frames.mixins.channel_collection_mixin - Line 57
- **Type**: NotImplementedError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `        raise NotImplementedError(remove_channel() must be implemented in subclasses")`
- **Location**: `wandas/frames/mixins/channel_collection_mixin.py:57`

### frames.channel - Line 316
- **Type**: TypeError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise TypeError(                 Addition target with SNR must be a ChannelFrame or "  `
- **Location**: `wandas/frames/channel.py:316`

### frames.channel - Line 764
- **Type**: TypeError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise TypeError(channel must be int; list; or None")`
- **Location**: `wandas/frames/channel.py:764`

### core.base_frame - Line 295
- **Type**: TypeError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `                raise TypeError(                     fNumPy array must be of integer or boolean typ`
- **Location**: `wandas/core/base_frame.py:295`

### core.base_frame - Line 507
- **Type**: TypeError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `        #     raise TypeError(Sampling rate must be an integer")`
- **Location**: `wandas/core/base_frame.py:507`

### core.base_frame - Line 511
- **Type**: TypeError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise TypeError(Label must be a string")`
- **Location**: `wandas/core/base_frame.py:511`

### core.base_frame - Line 515
- **Type**: TypeError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise TypeError(Metadata must be a dictionary")`
- **Location**: `wandas/core/base_frame.py:515`

### core.base_frame - Line 521
- **Type**: TypeError
- **Score**: 2/3
- **Missing**: HOW
- **Current**: `            raise TypeError(Channel metadata must be a list")`
- **Location**: `wandas/core/base_frame.py:521`


---

## Implementation Roadmap

### Phase 1: Investigation (Current - Week 1-2)
- ✅ Extract all error messages
- ✅ Categorize and analyze
- ✅ Create guidelines document
- ✅ Update contribution guide

### Phase 2: High Priority Fixes (Week 3-6)
Focus on modules with most critical errors:

1. **frames.channel** (23 errors, avg: 0.7/3)
   - Most used module
   - Highest impact on user experience
   
2. **frames.spectrogram** (4 errors, avg: 0.0/3)
   - All errors score 0
   - Critical for spectral analysis

3. **utils.frame_dataset** (7 errors, avg: 0.6/3)
   - Important for data handling
   
4. **io.wdf_io** (5 errors, avg: 0.6/3)
   - File I/O errors are user-facing

### Phase 3: Medium Priority (Week 7-8)
- Enhance errors with score 2
- Add missing HOW elements
- Quick wins for better UX

### Phase 4: Quality Assurance (Week 9-10)
- Review all changed errors
- Add/update tests
- Documentation updates
- User feedback collection

---

## Guidelines Reference

For detailed guidelines on writing error messages, see:
- **[Error Message Guide](error_message_guide.md)** - Comprehensive guidelines
- **[Copilot Instructions](../../.github/copilot-instructions.md)** - Quick reference

---

**Note**: This inventory is automatically generated. Re-run analysis after making improvements to track progress.
