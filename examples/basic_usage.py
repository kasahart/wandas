# examples/basic_usage.py

from wandas.utils import generate_sample

# Generate a sample
signal = generate_sample.generate_sample(freqs=1000, duration=1)

# Apply a low-pass filter
filtered_signal = signal.low_pass_filter(cutoff=1000)

# Plot the signals
signal.plot(title="Original Signal")
filtered_signal.plot(title="Filtered Signal")

# Perform Fourier transform for spectral analysis
signal.fft().plot(title="Spectrum of Original Signal")
filtered_signal.fft().plot(title="Spectrum of Filtered Signal")

# Write the filtered signal to a WAV file
# filtered_signal.to_wav('filtered_audio.wav')
