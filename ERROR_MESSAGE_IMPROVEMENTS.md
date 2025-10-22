# Error Message Improvements: Before and After Examples

## 1. Low-Pass Filter - Cutoff Too High

### Before
```
ValueError: Cutoff frequency must be between 0 Hz and 8000 Hz
```

### After
```
ValueError: Cutoff frequency is too high:
  Given: 10000 Hz
  Nyquist frequency (limit): 8000 Hz
  Sampling rate: 16000 Hz

Solution:
  - Use cutoff < 8000 Hz
  - Or increase sampling rate above 20000 Hz

Background:
  Nyquist frequency is half of the sampling rate.
  Filters cannot work above this limit due to aliasing.
```

**Impact**: User now understands *why* their cutoff is too high, what the limit is, and has two concrete solutions.

---

## 2. ChannelFrame - Invalid Shape

### Before
```
ValueError: Data must be 1-dimensional or 2-dimensional. Shape: (2, 3, 4)
```

### After
```
ValueError: Data must be 1-dimensional or 2-dimensional:
  Given shape: (2, 3, 4) (3 dimensions)
  Expected: (n_samples,) or (n_channels, n_samples)

Solution:
  - For single channel: use shape (n_samples,) or (1, n_samples)
  - For multi-channel: use shape (n_channels, n_samples)
  - Reshape your data: data.reshape(n_channels, n_samples)

Background:
  ChannelFrame represents time-series data with channels.
  First dimension is channels, second is time samples.
```

**Impact**: User learns what each dimension represents and gets concrete reshape examples.

---

## 3. ChannelFrame - Invalid Sampling Rate (NEW)

### Before
No validation - error would occur later in unexpected places

### After
```
ValueError: Sampling rate must be positive:
  Given: -100 Hz

Solution:
  - Use a positive sampling rate (e.g., 44100 Hz for audio)
  - Common values: 8000, 16000, 44100, 48000 Hz

Background:
  Sampling rate defines how many samples per second.
  It must be positive to represent time correctly.
```

**Impact**: Early detection with clear guidance on valid values.

---

## 4. Band-Pass Filter - Invalid Range

### Before
```
ValueError: Lower cutoff frequency (1000 Hz) must be less than higher cutoff frequency (500 Hz)
```

### After
```
ValueError: Lower cutoff must be less than higher cutoff:
  Lower cutoff: 1000 Hz
  Higher cutoff: 500 Hz

Solution:
  - Ensure low_cutoff < high_cutoff
  - For example, use low_cutoff=250.0 Hz and high_cutoff=500 Hz

Background:
  A band-pass filter requires a valid frequency band.
  The lower cutoff defines the start, and higher cutoff defines the end.
```

**Impact**: Clear structure with example values and explanation of band-pass concept.

---

## 5. Read WAV - File Not Found

### Before
```
FileNotFoundError: File not found: /path/to/audio.wav
```

### After
```
FileNotFoundError: WAV file not found:
  Path: /path/to/audio.wav
  Absolute path: /home/user/project/path/to/audio.wav

Solution:
  - Check if the file path is correct
  - Ensure the file exists in the specified location
  - Use absolute path if relative path is not working
  - Check file permissions
```

**Impact**: Shows both relative and absolute paths, helping users debug path issues.

---

## 6. Read WAV - Invalid File Format (NEW)

### Before
Raw scipy error message (cryptic)

### After
```
ValueError: Failed to read WAV file:
  Path: /path/to/file.wav
  Error: Unexpected end of file

Solution:
  - Verify the file is a valid WAV format (not MP3, AAC, etc.)
  - Check if the file is corrupted
  - Try opening the file with other audio software
  - Convert the file to WAV format if needed

Background:
  This function only supports WAV format.
  For other formats, convert to WAV first.
```

**Impact**: Catches common mistake of using non-WAV files, with clear format guidance.

---

## 7. Resampling - Invalid Target Rate (NEW)

### Before
No validation - would fail in librosa with unclear error

### After
```
ValueError: Target sampling rate must be positive:
  Given: -8000 Hz

Solution:
  - Use a positive target sampling rate (e.g., 44100 Hz for audio)
  - Common values: 8000, 16000, 44100, 48000 Hz

Background:
  Sampling rate defines how many samples per second.
  It must be positive to represent time correctly.
```

**Impact**: Early detection prevents cryptic librosa errors.

---

## Design Principles Applied

### 1. **Actionable**
Every error message includes a "Solution:" section with concrete steps

### 2. **Educational**
"Background:" sections teach signal processing concepts:
- Nyquist frequency
- Aliasing
- Channel/sample dimensions
- Sampling rate meaning

### 3. **Consistent**
All errors follow the same format:
- Problem description
- Current state (values)
- Solution (steps)
- Background (when relevant)

### 4. **Specific**
Messages include actual values:
- Given cutoff frequency
- Nyquist limit
- Suggested alternatives
- Example reshapes

### 5. **Scannable**
Multi-line format with clear sections makes errors easy to read even when stressed

---

## User Journey Improvement

### Scenario: New User Trying to Apply Low-Pass Filter

**Before:**
1. Gets error: "Cutoff frequency must be between 0 Hz and 8000 Hz"
2. Confused - why 8000? What's special about that?
3. Googles "nyquist frequency"
4. Learns about Nyquist theorem
5. Calculates: sampling_rate / 2
6. Finally fixes their code

**After:**
1. Gets detailed error with Nyquist explanation
2. Sees two clear options: lower cutoff OR increase sampling rate
3. Learns about Nyquist frequency in context
4. Immediately fixes their code
5. Bonus: Now understands signal processing concept

**Time saved:** 5-10 minutes per error, plus educational value

---

## Validation Added

New validation that didn't exist before:
1. **ChannelFrame sampling rate** - must be positive
2. **ReSampling target_sr** - must be positive
3. **WAV file format** - better error handling for non-WAV files

These additions prevent cryptic errors deep in the call stack.

---

## Test Coverage

All improved errors now have dedicated tests verifying:
- Error is raised correctly
- Message contains "Solution:"
- Message contains "Background:" (where applicable)
- Message includes relevant values
- Error type is correct (ValueError vs FileNotFoundError)

Total new tests: 12
Updated existing tests: 2

---

## Conclusion

These improvements transform error messages from obstacles into teaching moments. Users not only fix their immediate problem faster but also gain understanding of signal processing concepts that will help them write better code in the future.
