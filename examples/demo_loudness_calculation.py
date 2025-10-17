#!/usr/bin/env python3
"""
Demonstration script for loudness calculation using wandas.

This script shows how to use the new loudness_zwtv feature to calculate
time-varying loudness for audio signals using the Zwicker method.

Requirements:
    - wandas with loudness support
    - numpy, matplotlib (for visualization)
    - An audio file (or use generated signal)
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Note: This is a demonstration script. To run it, you need wandas installed
# with all dependencies (mosqito, numpy, matplotlib, etc.)

try:
    import wandas as wd
    import matplotlib.pyplot as plt
    WANDAS_AVAILABLE = True
except ImportError:
    print("wandas not available. This is a demonstration of the API.")
    WANDAS_AVAILABLE = False


def demo_basic_usage():
    """Demonstrate basic loudness calculation."""
    print("=== Basic Usage Demo ===\n")
    
    if not WANDAS_AVAILABLE:
        print("""
# Generate a test signal
signal = wd.generate_sin(freqs=[1000], duration=2.0, sampling_rate=48000)
signal = signal * 0.063  # Scale to ~70 dB SPL

# Calculate loudness
loudness = signal.loudness_zwtv()

# Print statistics
print(f"Mean loudness: {loudness.mean():.2f} sones")
print(f"Max loudness: {loudness.max():.2f} sones")
print(f"Min loudness: {loudness.min():.2f} sones")

# Plot
loudness.plot(title="Time-varying Loudness")
plt.show()
        """)
        return
    
    # Generate a test signal
    signal = wd.generate_sin(freqs=[1000], duration=2.0, sampling_rate=48000)
    signal = signal * 0.063  # Scale to ~70 dB SPL
    
    # Calculate loudness
    loudness = signal.loudness_zwtv()
    
    # Compute to get actual values
    loudness_values = loudness.data.compute()
    
    print(f"Signal shape: {signal.shape}")
    print(f"Loudness shape: {loudness.shape}")
    print(f"Mean loudness: {np.mean(loudness_values):.2f} sones")
    print(f"Max loudness: {np.max(loudness_values):.2f} sones")
    print(f"Min loudness: {np.min(loudness_values):.2f} sones")
    print()


def demo_field_comparison():
    """Demonstrate comparison between free and diffuse field."""
    print("=== Field Type Comparison Demo ===\n")
    
    if not WANDAS_AVAILABLE:
        print("""
# Generate signal
signal = wd.generate_sin(freqs=[1000, 2000, 4000], duration=1.0, sampling_rate=48000)
signal = signal * 0.1

# Calculate for both field types
loudness_free = signal.loudness_zwtv(field_type="free")
loudness_diffuse = signal.loudness_zwtv(field_type="diffuse")

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
loudness_free.plot(ax=axes[0], title="Free Field Loudness")
loudness_diffuse.plot(ax=axes[1], title="Diffuse Field Loudness")
plt.tight_layout()
plt.show()
        """)
        return
    
    # Generate multi-frequency signal
    signal = wd.generate_sin(freqs=[1000, 2000, 4000], duration=1.0, sampling_rate=48000)
    signal = signal * 0.1
    
    # Calculate for both field types
    loudness_free = signal.loudness_zwtv(field_type="free")
    loudness_diffuse = signal.loudness_zwtv(field_type="diffuse")
    
    # Get mean values
    mean_free = np.mean(loudness_free.data.compute())
    mean_diffuse = np.mean(loudness_diffuse.data.compute())
    
    print(f"Free field mean loudness: {mean_free:.2f} sones")
    print(f"Diffuse field mean loudness: {mean_diffuse:.2f} sones")
    print(f"Ratio (free/diffuse): {mean_free/mean_diffuse:.2f}")
    print()


def demo_amplitude_dependency():
    """Demonstrate loudness dependency on amplitude."""
    print("=== Amplitude Dependency Demo ===\n")
    
    if not WANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        print("""
# Create signals at different levels
amplitudes = [0.01, 0.03, 0.1, 0.3]  # Approximately 50, 60, 70, 80 dB SPL
loudness_values = []

for amp in amplitudes:
    signal = wd.generate_sin(freqs=[1000], duration=0.5, sampling_rate=48000)
    signal = signal * amp
    loudness = signal.loudness_zwtv()
    mean_loudness = loudness.mean()
    loudness_values.append(mean_loudness)
    print(f"Amplitude {amp:.2f}: {mean_loudness:.2f} sones")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(amplitudes, loudness_values, 'o-', linewidth=2)
plt.xlabel('Signal Amplitude')
plt.ylabel('Mean Loudness (sones)')
plt.title('Loudness vs Amplitude')
plt.grid(True)
plt.show()
        """)
        return
    
    # Test at different amplitudes
    amplitudes = [0.01, 0.03, 0.1, 0.3]
    loudness_values = []
    
    for amp in amplitudes:
        signal = wd.generate_sin(freqs=[1000], duration=0.5, sampling_rate=48000)
        signal = signal * amp
        loudness = signal.loudness_zwtv()
        mean_loudness = np.mean(loudness.data.compute())
        loudness_values.append(mean_loudness)
        print(f"Amplitude {amp:.3f}: {mean_loudness:.2f} sones")
    
    print()


def demo_multichannel():
    """Demonstrate multi-channel processing."""
    print("=== Multi-channel Processing Demo ===\n")
    
    if not WANDAS_AVAILABLE:
        print("""
# Create stereo signal with different content per channel
left = wd.generate_sin(freqs=[500], duration=1.0, sampling_rate=48000) * 0.05
right = wd.generate_sin(freqs=[2000], duration=1.0, sampling_rate=48000) * 0.1

# Combine into stereo
stereo = wd.ChannelFrame.concatenate([left, right], axis=0)

# Calculate loudness
loudness = stereo.loudness_zwtv()

# Access individual channels
left_loudness = loudness[0]
right_loudness = loudness[1]

print(f"Left channel mean loudness: {left_loudness.mean():.2f} sones")
print(f"Right channel mean loudness: {right_loudness.mean():.2f} sones")

# Plot both channels
loudness.plot(overlay=True, title="Stereo Loudness Comparison")
plt.show()
        """)
        return
    
    # Create different signals for left and right
    left = wd.generate_sin(freqs=[500], duration=1.0, sampling_rate=48000) * 0.05
    right = wd.generate_sin(freqs=[2000], duration=1.0, sampling_rate=48000) * 0.1
    
    # Combine into stereo (concatenate along channel axis)
    stereo_data = np.vstack([left.data.compute()[0], right.data.compute()[0]])
    stereo = wd.ChannelFrame(
        data=wd.da.from_array(stereo_data),
        sampling_rate=48000
    )
    
    # Calculate loudness
    loudness = stereo.loudness_zwtv()
    loudness_data = loudness.data.compute()
    
    print(f"Left channel (500 Hz, lower amp): {np.mean(loudness_data[0]):.2f} sones")
    print(f"Right channel (2000 Hz, higher amp): {np.mean(loudness_data[1]):.2f} sones")
    print()


def demo_comparison_with_mosqito():
    """Demonstrate that results match MoSQITo directly."""
    print("=== Comparison with MoSQITo Direct Call ===\n")
    
    if not WANDAS_AVAILABLE:
        print("""
from mosqito.sq_metrics.loudness.loudness_zwtv import loudness_zwtv

# Create signal
signal = wd.generate_sin(freqs=[1000], duration=1.0, sampling_rate=48000)
signal = signal * 0.063

# Calculate using wandas
loudness_wandas = signal.loudness_zwtv()

# Calculate using MoSQITo directly
data = signal.data[0].compute()
N_mosqito, _, _, _ = loudness_zwtv(data, signal.sampling_rate, field_type="free")

# Compare
loudness_wandas_data = loudness_wandas.data[0].compute()
difference = np.abs(loudness_wandas_data - N_mosqito)

print(f"Max difference: {np.max(difference):.10f}")
print(f"Mean difference: {np.mean(difference):.10f}")
print("Results match!" if np.max(difference) < 1e-6 else "Results differ")
        """)
        return
    
    try:
        from mosqito.sq_metrics.loudness.loudness_zwtv import loudness_zwtv as mosqito_loudness
        
        # Create signal
        signal = wd.generate_sin(freqs=[1000], duration=1.0, sampling_rate=48000)
        signal = signal * 0.063
        
        # Calculate using wandas
        loudness_wandas = signal.loudness_zwtv()
        
        # Calculate using MoSQITo directly
        data = signal.data[0].compute()
        N_mosqito, _, _, _ = mosqito_loudness(data, signal.sampling_rate, field_type="free")
        
        # Compare
        loudness_wandas_data = loudness_wandas.data[0].compute()
        difference = np.abs(loudness_wandas_data - N_mosqito)
        
        print(f"Wandas result shape: {loudness_wandas_data.shape}")
        print(f"MoSQITo result shape: {N_mosqito.shape}")
        print(f"Max difference: {np.max(difference):.10f}")
        print(f"Mean difference: {np.mean(difference):.10f}")
        print("✓ Results match!" if np.max(difference) < 1e-6 else "✗ Results differ")
        print()
    except ImportError:
        print("MoSQITo not available for comparison")
        print()


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Wandas Loudness Calculation Demonstration")
    print("=" * 60)
    print()
    
    if not WANDAS_AVAILABLE:
        print("Note: wandas is not installed. Showing API examples only.\n")
    
    demo_basic_usage()
    demo_field_comparison()
    demo_amplitude_dependency()
    demo_multichannel()
    demo_comparison_with_mosqito()
    
    print("=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
