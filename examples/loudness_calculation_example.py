"""
Example: Loudness Calculation using MoSQITo

This example demonstrates how to calculate loudness (psychoacoustic metric)
using the Wandas library with MoSQITo integration.
"""

import numpy as np
import wandas as wd
from wandas.processing import LoudnessZwtv, LoudnessZwst

# Generate a test signal: 1 kHz tone at 70 dB SPL
# Note: MoSQITo expects signals in Pascals (Pa)
# 70 dB SPL corresponds to approximately 0.0632 Pa RMS

sampling_rate = 48000
duration = 1.0  # 1 second
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
frequency = 1000.0  # Hz

# Generate sine wave with RMS amplitude of 0.0632 Pa (70 dB SPL)
rms_amplitude = 0.0632  # Pa
peak_amplitude = rms_amplitude * np.sqrt(2)
signal_data = peak_amplitude * np.sin(2 * np.pi * frequency * t)

# Create a ChannelFrame
signal = wd.from_numpy(signal_data, sampling_rate=sampling_rate)

print("Signal properties:")
print(f"  Duration: {signal.duration:.2f} s")
print(f"  Sampling rate: {signal.sampling_rate} Hz")
print(f"  Number of samples: {signal.n_samples}")
print()

# ============================================================================
# Time-varying Loudness Calculation
# ============================================================================
print("=" * 70)
print("Time-varying Loudness (ISO 532-1:2017 - Zwicker method)")
print("=" * 70)

# Create loudness operation for free-field condition
loudness_tv = LoudnessZwtv(sampling_rate=sampling_rate, field_type="free")

# Calculate loudness
result_tv = loudness_tv.process(signal._data)

print(f"\nResults:")
print(f"  Overall loudness (mean): {np.mean(result_tv['N']):.3f} sone")
print(f"  Overall loudness (max): {np.max(result_tv['N']):.3f} sone")
print(f"  Overall loudness (min): {np.min(result_tv['N']):.3f} sone")
print(f"  Number of time frames: {len(result_tv['time_axis'])}")
print(f"  Time resolution: {result_tv['time_axis'][1] - result_tv['time_axis'][0]:.4f} s")
print(f"  Bark frequency bands: {len(result_tv['bark_axis'])}")
print()

# ============================================================================
# Stationary Loudness Calculation
# ============================================================================
print("=" * 70)
print("Stationary Loudness (ISO 532-1:2017 - Zwicker method)")
print("=" * 70)

# Create loudness operation for free-field condition
loudness_st = LoudnessZwst(sampling_rate=sampling_rate, field_type="free")

# Calculate loudness
result_st = loudness_st.process(signal._data)

print(f"\nResults:")
print(f"  Overall loudness: {result_st['N']:.3f} sone")
print(f"  Specific loudness shape: {result_st['N_spec'].shape}")
print(f"  Bark frequency bands: {len(result_st['bark_axis'])}")
print(f"  Peak specific loudness: {np.max(result_st['N_spec']):.3f} sone/bark")
peak_bark_idx = np.argmax(result_st['N_spec'])
print(f"  Peak at Bark frequency: {result_st['bark_axis'][peak_bark_idx]:.2f} Bark")
print()

# ============================================================================
# Comparison: Free-field vs Diffuse-field
# ============================================================================
print("=" * 70)
print("Comparison: Free-field vs Diffuse-field")
print("=" * 70)

# Calculate for diffuse-field condition
loudness_st_diffuse = LoudnessZwst(sampling_rate=sampling_rate, field_type="diffuse")
result_st_diffuse = loudness_st_diffuse.process(signal._data)

print(f"\nStationary Loudness:")
print(f"  Free-field:   {result_st['N']:.3f} sone")
print(f"  Diffuse-field: {result_st_diffuse['N']:.3f} sone")
print(f"  Difference:    {abs(result_st['N'] - result_st_diffuse['N']):.3f} sone")
print()

# ============================================================================
# Using with real audio files
# ============================================================================
print("=" * 70)
print("Usage with WAV files")
print("=" * 70)

print("""
To use with real audio files:

```python
import wandas as wd
from wandas.processing import LoudnessZwtv, LoudnessZwst

# Read audio file
signal = wd.read_wav("audio.wav")

# Important: Ensure the signal is in Pascals (Pa)
# WAV files typically contain normalized values [-1, 1]
# You may need to calibrate or convert to Pa:
# signal_pa = signal * calibration_factor

# Calculate time-varying loudness
loudness_op = LoudnessZwtv(sampling_rate=signal.sampling_rate, field_type="free")
result = loudness_op.process(signal._data)

# Access results
overall_loudness = result['N']  # [sone] over time
specific_loudness = result['N_spec']  # [sone/bark] over time and frequency
time_axis = result['time_axis']  # [s]
bark_axis = result['bark_axis']  # [Bark]
```

Note: For accurate loudness calculations, ensure your signal is properly
calibrated to Pascals (Pa). The ISO 532-1 standard expects signals in Pa.
""")
