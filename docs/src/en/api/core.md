# Core Module

This page explains the core module of Wandas. The core module provides the basic data structures and functionality of Wandas.

## BaseFrame

`BaseFrame` is the base class for all frame classes in Wandas. You don't instantiate this class directly but use it through derived classes (like ChannelFrame).

```python
from wandas.core.base_frame import BaseFrame
```

::: wandas.core.base_frame.BaseFrame
    handler: python
    selection:
      members:
        - __init__
        - data
        - metadata
        - shape
        - n_channels
        - channels
        - sampling_rate
        - compute
        - plot
        - persist

## ChannelMetadata

`ChannelMetadata` is a class for managing metadata for each channel. It holds channel names, units, and other additional information.

```python
from wandas.core.metadata import ChannelMetadata

# Usage example
metadata = ChannelMetadata(label="Left Channel", unit="dB", extra={"device": "Microphone"})
```

::: wandas.core.metadata.ChannelMetadata
    handler: python
    selection:
      members:
        - __init__
        - __getitem__
        - __setitem__
        - to_json
        - from_json
