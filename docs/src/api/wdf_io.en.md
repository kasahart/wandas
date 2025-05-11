# WDF File I/O

The `wandas.io.wdf_io` module provides functionality for saving and loading `ChannelFrame` objects in the WDF (Wandas Data File) format.
The WDF format is based on HDF5 and preserves not only the data but also all metadata such as sampling rate, units, and channel labels.

## WDF Format Overview

The WDF format has the following features:

- HDF5-based hierarchical data structure
- Complete preservation of channel data and metadata
- Size optimization through data compression and chunking
- Version management for future extensions

File structure:

```
/meta           : Frame-level metadata (JSON format)
/channels/{i}   : Individual channel data and metadata
    ├─ data           : Waveform data (numpy array)
    └─ attrs          : Channel attributes (labels, units, etc.)
```

## Saving WDF Files

::: wandas.io.wdf_io.save

## Loading WDF Files

::: wandas.io.wdf_io.load

## Usage Examples

```python
# Save a ChannelFrame in WDF format
cf = wd.read_wav("audio.wav")
cf.save("audio_data.wdf")

# Specifying options when saving
cf.save(
    "high_quality.wdf",
    compress="gzip",  # Compression method
    dtype="float64",  # Data type
    overwrite=True    # Allow overwriting
)

# Load a ChannelFrame from a WDF file
cf2 = wd.ChannelFrame.load("audio_data.wdf")
```

For detailed usage examples, see the [File I/O Guide](/en/how_to/file_io).
