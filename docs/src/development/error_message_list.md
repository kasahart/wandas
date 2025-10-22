# Complete Error Message List

This document lists all error messages in the Wandas codebase for reference.

**Total Errors**: 100
**Files**: 19

---


## wandas/frames/channel.py

**Error Count**: 23


### FileNotFoundError (1)

- **Line 720**: `raise FileNotFoundError(f"File not found: {path}")`

### IndexError (1)

- **Line 1138**: `raise IndexError(f"index {key} out of range")`

### KeyError (1)

- **Line 1143**: `raise KeyError(f"label {key} not found")`

### TypeError (4)

- **Line 316**: `raise TypeError(`
- **Line 563**: `raise TypeError(`
- **Line 764**: `raise TypeError("channel must be int, list, or None")`
- **Line 1086**: `raise TypeError("add_channel: ndarray/dask/同型Frameのみ対応")`

### ValueError (16)

- **Line 69**: `raise ValueError(`
- **Line 219**: `raise ValueError(`
- **Line 305**: `raise ValueError(`
- **Line 598**: `raise ValueError(`
- **Line 613**: `raise ValueError(`
- **Line 623**: `raise ValueError(`
- **Line 750**: `raise ValueError(`
- **Line 758**: `raise ValueError(`
- **Line 787**: `raise ValueError("Unexpected data type after reading file")`
- **Line 805**: `raise ValueError("Chunk size must be a positive integer")`
- **Line 823**: `raise ValueError(`
- **Line 1013**: `raise ValueError("sampling_rate不一致")`
- **Line 1044**: `raise ValueError("データ長不一致: align指定を確認")`
- **Line 1056**: `raise ValueError(f"label重複: {new_label}")`
- **Line 1106**: `raise ValueError("データ長不一致: align指定を確認")`
- **Line 1113**: `raise ValueError("label重複")`


## wandas/core/base_frame.py

**Error Count**: 13


### KeyError (1)

- **Line 428**: `raise KeyError(f"Channel label '{label}' not found.")`

### TypeError (8)

- **Line 295**: `raise TypeError(`
- **Line 318**: `raise TypeError(`
- **Line 339**: `raise TypeError(`
- **Line 391**: `raise TypeError(`
- **Line 507**: `#     raise TypeError("Sampling rate must be an integer")`
- **Line 511**: `raise TypeError("Label must be a string")`
- **Line 515**: `raise TypeError("Metadata must be a dictionary")`
- **Line 521**: `raise TypeError("Channel metadata must be a list")`

### ValueError (4)

- **Line 285**: `raise ValueError(`
- **Line 302**: `raise ValueError("Cannot index with an empty list")`
- **Line 379**: `raise ValueError(f"Invalid key length: {len(key)} for shape {self.shape}")`
- **Line 474**: `raise ValueError(f"Computed result is not a np.ndarray: {type(result)}")`


## wandas/frames/roughness.py

**Error Count**: 7


### NotImplementedError (1)

- **Line 367**: `raise NotImplementedError(`

### ValueError (6)

- **Line 121**: `raise ValueError(`
- **Line 127**: `raise ValueError(`
- **Line 133**: `raise ValueError(f"bark_axis must have 47 elements, got {len(bark_axis)}")`
- **Line 137**: `raise ValueError(f"overlap must be in [0.0, 1.0], got {overlap}")`
- **Line 299**: `raise ValueError(`
- **Line 305**: `raise ValueError(`


## wandas/utils/frame_dataset.py

**Error Count**: 7


### FileNotFoundError (1)

- **Line 92**: `raise FileNotFoundError(f"Folder does not exist: {self.folder_path}")`

### IndexError (3)

- **Line 185**: `raise IndexError(`
- **Line 378**: `raise IndexError(`
- **Line 394**: `raise IndexError(`

### NotImplementedError (3)

- **Line 250**: `raise NotImplementedError("The save method is not currently implemented.")`
- **Line 385**: `raise NotImplementedError("_SampledFrameDataset does not load files directly.")`
- **Line 633**: `raise NotImplementedError(`


## wandas/visualization/plotting.py

**Error Count**: 7


### NotImplementedError (1)

- **Line 757**: `raise NotImplementedError()`

### TypeError (2)

- **Line 767**: `raise TypeError("Strategy class must inherit from PlotStrategy.")`
- **Line 769**: `raise TypeError("Cannot register abstract PlotStrategy class.")`

### ValueError (4)

- **Line 435**: `raise ValueError("Overlay is not supported for SpectrogramPlotStrategy.")`
- **Line 438**: `raise ValueError("ax must be None when n_channels > 1.")`
- **Line 499**: `raise ValueError("fig must be a matplotlib Figure object.")`
- **Line 782**: `raise ValueError(f"Unknown plot type: {name}")`


## wandas/io/wdf_io.py

**Error Count**: 5


### FileExistsError (1)

- **Line 62**: `raise FileExistsError(`

### FileNotFoundError (1)

- **Line 172**: `raise FileNotFoundError(f"File not found: {path}")`

### NotImplementedError (2)

- **Line 68**: `raise NotImplementedError(`
- **Line 168**: `raise NotImplementedError(f"Format '{format}' is not supported")`

### ValueError (1)

- **Line 231**: `raise ValueError("No channel data found in the file")`


## wandas/processing/filters.py

**Error Count**: 5


### ValueError (5)

- **Line 39**: `raise ValueError(f"Cutoff frequency must be between 0 Hz and {limit} Hz")`
- **Line 89**: `raise ValueError(`
- **Line 153**: `raise ValueError(`
- **Line 157**: `raise ValueError(`
- **Line 161**: `raise ValueError(`


## wandas/processing/base.py

**Error Count**: 5


### NotImplementedError (2)

- **Line 97**: `raise NotImplementedError("Subclasses must implement this method.")`
- **Line 120**: `raise NotImplementedError("Subclasses must implement this method.")`

### TypeError (2)

- **Line 143**: `raise TypeError("Strategy class must inherit from AudioOperation.")`
- **Line 145**: `raise TypeError("Cannot register abstract AudioOperation class.")`

### ValueError (1)

- **Line 153**: `raise ValueError(f"Unknown operation type: {name}")`


## wandas/frames/spectrogram.py

**Error Count**: 4


### IndexError (1)

- **Line 636**: `raise IndexError(`

### ValueError (3)

- **Line 122**: `raise ValueError(`
- **Line 126**: `raise ValueError(`
- **Line 373**: `raise ValueError(`


## wandas/io/readers.py

**Error Count**: 4


### ValueError (4)

- **Line 113**: `raise ValueError("Unexpected data type after reading file")`
- **Line 219**: `raise ValueError(f"Requested channels {channels} out of range")`
- **Line 231**: `raise ValueError("Unexpected data type after reading file")`
- **Line 254**: `raise ValueError(f"No suitable file reader found for {path_str}")`


## wandas/processing/psychoacoustic.py

**Error Count**: 4


### ValueError (4)

- **Line 99**: `raise ValueError(`
- **Line 301**: `raise ValueError(`
- **Line 497**: `raise ValueError(f"overlap must be in [0.0, 1.0], got {self.overlap}")`
- **Line 645**: `raise ValueError(f"overlap must be in [0.0, 1.0], got {self.overlap}")`


## wandas/frames/mixins/channel_processing_mixin.py

**Error Count**: 3


### ValueError (3)

- **Line 186**: `raise ValueError(f"Unsupported reduction operation: {op}")`
- **Line 251**: `raise ValueError("start must be less than end")`
- **Line 756**: `raise ValueError("Operation did not provide bark_axis in metadata")`


## wandas/frames/noct.py

**Error Count**: 3


### NotImplementedError (2)

- **Line 274**: `raise NotImplementedError(`
- **Line 284**: `raise NotImplementedError(`

### ValueError (1)

- **Line 248**: `raise ValueError("freqs is not numpy array.")`


## wandas/frames/spectral.py

**Error Count**: 3


### ValueError (3)

- **Line 123**: `raise ValueError(`
- **Line 328**: `raise ValueError(`
- **Line 583**: `raise ValueError(`


## wandas/frames/mixins/channel_collection_mixin.py

**Error Count**: 2


### NotImplementedError (2)

- **Line 40**: `raise NotImplementedError("add_channel() must be implemented in subclasses")`
- **Line 57**: `raise NotImplementedError("remove_channel() must be implemented in subclasses")`


## wandas/processing/temporal.py

**Error Count**: 2


### ValueError (2)

- **Line 164**: `raise ValueError("Either length or duration must be provided.")`
- **Line 301**: `raise ValueError("A_weighting returned an unexpected type.")`


## wandas/utils/generate_sample.py

**Error Count**: 1


### ValueError (1)

- **Line 80**: `raise ValueError("freqs must be a float or a list of floats.")`


## wandas/io/wav_io.py

**Error Count**: 1


### ValueError (1)

- **Line 93**: `raise ValueError("target must be a ChannelFrame object.")`


## wandas/processing/spectral.py

**Error Count**: 1


### ValueError (1)

- **Line 388**: `raise ValueError(`

