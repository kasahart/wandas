# Error Message Improvement Examples

**Document Purpose**: Show practical before/after examples for common error patterns in wandas  
**Target Audience**: Developers implementing Phase 2 improvements  
**Related**: [Error Message Guide](error_message_guide.md)

## Overview

This document provides concrete examples of improving actual error messages from the wandas codebase. Each example shows:
- Current message (with quality score)
- Issues identified
- Improved message following the 3-element rule
- Implementation code

## Common Patterns

### Pattern 1: Dimension Validation

**Current Issues**: Missing WHY and HOW elements

#### Example 1.1: Channel Frame Dimension Error

**Current** (Score: 1/3)
```python
# Location: wandas/frames/channel.py:69
raise ValueError(
    f"Data must be 1-dimensional or 2-dimensional. Shape: {data.shape}"
)
```

**Issues:**
- ✓ Has WHAT (dimension requirement)
- ✗ Missing WHY (what each dimension represents)
- ✗ Missing HOW (how to fix the issue)

**Improved** (Score: 3/3)
```python
raise ValueError(
    f"Invalid data shape for ChannelFrame\n"
    f"  Got: {data.shape} ({data.ndim}D)\n"
    f"  Expected: (n_samples,) for mono or (n_channels, n_samples) for multi-channel\n"
    f"Reshape your data:\n"
    f"  - For mono: data.flatten() or data.reshape(-1)\n"
    f"  - For multi-channel: data.reshape(n_channels, -1)"
)
```

#### Example 1.2: Spectrogram Dimension Error

**Current** (Score: 0/3)
```python
# Location: wandas/frames/spectrogram.py:122
raise ValueError(
    f"データは2次元または3次元である必要があります。形状: {data.shape}"
)
```

**Issues:**
- ✗ Japanese text (should be English)
- ✗ Missing WHAT (context about spectrogram)
- ✗ Missing WHY (dimension meaning)
- ✗ Missing HOW (conversion method)

**Improved** (Score: 3/3)
```python
raise ValueError(
    f"Invalid data shape for SpectrogramFrame\n"
    f"  Got: {data.shape} ({data.ndim}D)\n"
    f"  Expected: (n_freqs, n_frames) for mono or (n_channels, n_freqs, n_frames) for multi-channel\n"
    f"If you have time-domain data, convert to spectrogram first:\n"
    f"  Example: signal.stft(n_fft=2048, hop_length=512)"
)
```

### Pattern 2: Sampling Rate Validation

**Current Issues**: Missing specific values and actionable solutions

#### Example 2.1: Sampling Rate Mismatch (Japanese Text)

**Current** (Score: 0/3)
```python
# Location: wandas/frames/channel.py:1013
raise ValueError("sampling_rate不一致")
```

**Issues:**
- ✗ Japanese text
- ✗ No actual values shown
- ✗ No context about which signals
- ✗ No solution provided

**Improved** (Score: 3/3)
```python
raise ValueError(
    f"Sampling rate mismatch between signals\n"
    f"  Current signal: {self.sampling_rate} Hz\n"
    f"  New signal: {data.sampling_rate} Hz\n"
    f"Operations require matching sampling rates. Options:\n"
    f"  1. Resample new signal: new_signal.resample({self.sampling_rate})\n"
    f"  2. Resample current: self.resample({data.sampling_rate})\n"
    f"  3. Resample both to common rate: both.resample(44100)"
)
```

#### Example 2.2: Roughness Sampling Rate Mismatch

**Current** (Score: 0/3)
```python
# Location: wandas/frames/roughness.py:299
raise ValueError(
    f"Sampling rates do not match: {self.sampling_rate} vs {other.sampling_rate}"
)
```

**Issues:**
- ✓ Shows actual values
- ✗ Missing WHAT (which operation)
- ✗ Missing WHY (why it matters)
- ✗ Missing HOW (solution)

**Improved** (Score: 3/3)
```python
raise ValueError(
    f"Cannot perform roughness calculation with mismatched sampling rates\n"
    f"  Signal 1: {self.sampling_rate} Hz\n"
    f"  Signal 2: {other.sampling_rate} Hz\n"
    f"Roughness calculations require identical time bases.\n"
    f"Resample one signal to match:\n"
    f"  signal2_resampled = signal2.resample({self.sampling_rate})"
)
```

### Pattern 3: Parameter Range Validation

**Current Issues**: Usually has WHAT and WHY but missing HOW

#### Example 3.1: Cutoff Frequency Validation

**Current** (Score: 2/3)
```python
# Location: wandas/processing/filters.py:39
if self.cutoff <= 0 or self.cutoff >= self.sampling_rate / 2:
    limit = self.sampling_rate / 2
    raise ValueError(f"Cutoff frequency must be between 0 Hz and {limit} Hz")
```

**Issues:**
- ✓ Has WHAT (range requirement)
- ✓ Has WHY (shows Nyquist limit)
- ✗ Missing HOW (suggestion for user)
- ✗ Doesn't show actual value

**Improved** (Score: 3/3)
```python
if self.cutoff <= 0 or self.cutoff >= self.sampling_rate / 2:
    nyquist = self.sampling_rate / 2
    raise ValueError(
        f"Cutoff frequency out of valid range\n"
        f"  Got: {self.cutoff} Hz\n"
        f"  Valid range: 0 to {nyquist} Hz (Nyquist frequency)\n"
        f"  Sampling rate: {self.sampling_rate} Hz\n"
        f"For higher cutoff frequencies, increase the sampling rate or use a different filter.\n"
        f"Common sampling rates: 8000, 16000, 22050, 44100, 48000 Hz"
    )
```

#### Example 3.2: Overlap Parameter Validation

**Current** (Score: 2/3)
```python
# Location: wandas/frames/roughness.py:137
raise ValueError(f"overlap must be in [0.0, 1.0], got {overlap}")
```

**Issues:**
- ✓ Has WHAT (value shown)
- ✓ Has WHY (range specified)
- ✗ Missing HOW (what overlap means)
- ✗ Minimal context

**Improved** (Score: 3/3)
```python
raise ValueError(
    f"Overlap parameter out of valid range\n"
    f"  Got: {overlap}\n"
    f"  Valid range: 0.0 to 1.0 (0% to 100% overlap)\n"
    f"Overlap controls how much successive windows overlap:\n"
    f"  - 0.0 = No overlap (windows are adjacent)\n"
    f"  - 0.5 = 50% overlap (common for spectral analysis)\n"
    f"  - 0.75 = 75% overlap (high resolution)\n"
    f"Set overlap to a decimal value between 0.0 and 1.0."
)
```

### Pattern 4: Type Validation

**Current Issues**: Missing examples and context

#### Example 4.1: Channel Type Error

**Current** (Score: 0/3)
```python
# Location: wandas/frames/channel.py:764
raise TypeError("channel must be int, list, or None")
```

**Issues:**
- ✗ No actual type shown
- ✗ No explanation of each type
- ✗ No examples

**Improved** (Score: 3/3)
```python
raise TypeError(
    f"Invalid channel specification type\n"
    f"  Got: {type(channel).__name__}\n"
    f"  Value: {channel}\n"
    f"  Expected types:\n"
    f"    - int: Single channel index (e.g., channel=0 for first channel)\n"
    f"    - list[int]: Multiple channels (e.g., channel=[0, 2] for channels 0 and 2)\n"
    f"    - None: All channels (default)\n"
    f"Examples:\n"
    f"  frame.select_channel(0)        # First channel\n"
    f"  frame.select_channel([0, 1])   # First two channels\n"
    f"  frame.select_channel(None)     # All channels"
)
```

#### Example 4.2: Operation Type Error

**Current** (Score: 1/3)
```python
# Location: wandas/frames/channel.py:316
raise TypeError(
    f"Unsupported operand type for +: ChannelFrame and {type(other).__name__}"
)
```

**Issues:**
- ✓ Shows actual type
- ✗ Doesn't explain valid types
- ✗ No examples or suggestions

**Improved** (Score: 3/3)
```python
raise TypeError(
    f"Cannot add ChannelFrame with {type(other).__name__}\n"
    f"  Left operand: ChannelFrame (n_channels={self.n_channels}, n_samples={self.n_samples})\n"
    f"  Right operand: {type(other).__name__}\n"
    f"Supported operations:\n"
    f"  - ChannelFrame + ChannelFrame: Element-wise addition\n"
    f"  - ChannelFrame + numpy.ndarray: Add array to signal\n"
    f"  - ChannelFrame + scalar: Add constant to all samples\n"
    f"Example:\n"
    f"  result = signal1 + signal2  # Both ChannelFrame\n"
    f"  result = signal + 0.5       # Add DC offset"
)
```

### Pattern 5: File Operations

**Current Issues**: Missing troubleshooting information

#### Example 5.1: File Not Found

**Current** (Score: 1/3)
```python
# Location: wandas/frames/channel.py:720
raise FileNotFoundError(f"File not found: {path}")
```

**Issues:**
- ✓ Has WHAT (file not found)
- ✗ No absolute path shown
- ✗ No troubleshooting steps

**Improved** (Score: 3/3)
```python
raise FileNotFoundError(
    f"Audio file not found\n"
    f"  Path: {path}\n"
    f"  Absolute path: {Path(path).absolute()}\n"
    f"  Current directory: {Path.cwd()}\n"
    f"Troubleshooting:\n"
    f"  1. Verify the file exists at the specified path\n"
    f"  2. Check for typos in the filename\n"
    f"  3. Ensure you have read permissions\n"
    f"  4. Use absolute paths to avoid confusion: Path('/full/path/to/file.wav')"
)
```

#### Example 5.2: Folder Not Found

**Current** (Score: 1/3)
```python
# Location: wandas/utils/frame_dataset.py:92
raise FileNotFoundError(f"Folder does not exist: {self.folder_path}")
```

**Issues:**
- ✓ Has WHAT
- ✗ No troubleshooting
- ✗ No suggestions

**Improved** (Score: 3/3)
```python
raise FileNotFoundError(
    f"Dataset folder not found\n"
    f"  Path: {self.folder_path}\n"
    f"  Absolute path: {Path(self.folder_path).absolute()}\n"
    f"Ensure the folder exists and contains audio files.\n"
    f"To create a dataset:\n"
    f"  1. Create the folder: Path('{self.folder_path}').mkdir(parents=True, exist_ok=True)\n"
    f"  2. Add audio files to the folder\n"
    f"  3. Reload the dataset: dataset = FrameDataset('{self.folder_path}')"
)
```

### Pattern 6: Index/Key Errors

**Current Issues**: Missing context about valid range/keys

#### Example 6.1: Index Out of Range

**Current** (Score: 0/3)
```python
# Location: wandas/frames/channel.py:1138
raise IndexError(f"index {key} out of range")
```

**Issues:**
- ✗ No valid range shown
- ✗ No context about what is being indexed
- ✗ No suggestion

**Improved** (Score: 3/3)
```python
raise IndexError(
    f"Channel index out of range\n"
    f"  Requested index: {key}\n"
    f"  Valid range: 0 to {self.n_channels - 1}\n"
    f"  Total channels: {self.n_channels}\n"
    f"Use frame.n_channels to check the number of available channels.\n"
    f"Example: frame[0] for first channel, frame[-1] for last channel"
)
```

#### Example 6.2: Label Not Found

**Current** (Score: 0/3)
```python
# Location: wandas/frames/channel.py:1143
raise KeyError(f"label {key} not found")
```

**Issues:**
- ✗ No available labels shown
- ✗ No suggestion

**Improved** (Score: 3/3)
```python
raise KeyError(
    f"Channel label not found\n"
    f"  Requested label: '{key}'\n"
    f"  Available labels: {list(self.labels)}\n"
    f"  Total channels: {self.n_channels}\n"
    f"Use frame.labels to see all available channel labels.\n"
    f"Or use numeric index: frame[0] instead of frame['{key}']"
)
```

### Pattern 7: NotImplementedError

**Current Issues**: No explanation or alternative

#### Example 7.1: Feature Not Implemented

**Current** (Score: 0/3)
```python
# Location: wandas/frames/noct.py:274
raise NotImplementedError(
    "Plotting for NOctFrame is not yet implemented."
)
```

**Issues:**
- ✓ Has WHAT (feature not implemented)
- ✗ No WHY (reason)
- ✗ No HOW (workaround or timeline)

**Improved** (Score: 3/3)
```python
raise NotImplementedError(
    f"Plotting for NOctFrame is not yet implemented\n"
    f"Reason: Octave band visualization requires specialized plotting logic.\n"
    f"Workarounds:\n"
    f"  1. Convert to numpy array: data = frame.to_numpy()\n"
    f"  2. Use matplotlib directly:\n"
    f"       plt.bar(range(len(data)), data)\n"
    f"       plt.xlabel('Octave Band')\n"
    f"       plt.ylabel('Level (dB)')\n"
    f"  3. Track feature request: https://github.com/kasahart/wandas/issues/XXX"
)
```

## Implementation Checklist

When improving an error message:

- [ ] Change language to English if needed
- [ ] Add WHAT element (clear problem statement)
- [ ] Add WHY element (constraint/requirement explanation)
- [ ] Add HOW element (actionable solution)
- [ ] Show actual vs expected values
- [ ] Provide examples when helpful
- [ ] Include troubleshooting steps for complex errors
- [ ] Update related tests
- [ ] Follow the template from Error Message Guide

## Quick Reference

### Template Structure
```python
raise ExceptionType(
    f"<WHAT: Problem statement>\n"
    f"  Got: {actual}\n"
    f"  Expected: {expected}\n"
    f"<HOW: Solution>\n"
    f"  Example: {code_example}"
)
```

### Common Additions
- For dimensions: Explain what each dimension represents
- For rates: Show both values in Hz
- For ranges: Show min and max with units
- For types: List all valid types with examples
- For files: Show absolute path and current directory
- For indices: Show valid range and total count

---

**Next Steps**: Use these patterns as templates when implementing Phase 2 improvements.
