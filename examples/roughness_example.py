"""Example: Roughness calculation using wandas and MoSQITo.

This example demonstrates how to calculate the roughness of audio signals
using the Daniel & Weber method implemented via MoSQITo.

Roughness is a psychoacoustic metric that quantifies the perceived roughness
of a sound, related to rapid amplitude modulations in the 15-300 Hz range.
"""

import numpy as np

import wandas as wd


def main() -> None:
    """Demonstrate roughness calculation with different signals."""
    print("=" * 70)
    print("Roughness Calculation Example")
    print("=" * 70)

    # Parameters
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Example 1: Pure tone (low roughness)
    print("\n1. Pure tone (1000 Hz)")
    print("-" * 70)
    pure_tone = np.sin(2 * np.pi * 1000 * t)
    signal_pure = wd.ChannelFrame(data=pure_tone, sampling_rate=sample_rate)

    roughness_pure_time = signal_pure.roughness(method="time")
    roughness_pure_freq = signal_pure.roughness(method="freq")

    print(f"Roughness (time method): {roughness_pure_time:.3f} asper")
    print(f"Roughness (freq method): {roughness_pure_freq:.3f} asper")
    print("Note: Pure tone should have very low roughness (< 0.5 asper)")

    # Example 2: Amplitude modulated tone at 70 Hz (high roughness)
    print("\n2. Amplitude modulated tone (1000 Hz carrier, 70 Hz modulation)")
    print("-" * 70)
    carrier = np.sin(2 * np.pi * 1000 * t)
    modulator_70hz = 0.5 * (1 + np.sin(2 * np.pi * 70 * t))
    modulated_70hz = carrier * modulator_70hz

    signal_mod_70 = wd.ChannelFrame(data=modulated_70hz, sampling_rate=sample_rate)

    roughness_mod_70_time = signal_mod_70.roughness(method="time")
    roughness_mod_70_freq = signal_mod_70.roughness(method="freq")

    print(f"Roughness (time method): {roughness_mod_70_time:.3f} asper")
    print(f"Roughness (freq method): {roughness_mod_70_freq:.3f} asper")
    print("Note: 70 Hz modulation is near peak roughness frequency")

    # Example 3: Amplitude modulated tone at 100 Hz (moderate roughness)
    print("\n3. Amplitude modulated tone (1000 Hz carrier, 100 Hz modulation)")
    print("-" * 70)
    modulator_100hz = 0.5 * (1 + np.sin(2 * np.pi * 100 * t))
    modulated_100hz = carrier * modulator_100hz

    signal_mod_100 = wd.ChannelFrame(data=modulated_100hz, sampling_rate=sample_rate)

    roughness_mod_100_time = signal_mod_100.roughness(method="time")
    roughness_mod_100_freq = signal_mod_100.roughness(method="freq")

    print(f"Roughness (time method): {roughness_mod_100_time:.3f} asper")
    print(f"Roughness (freq method): {roughness_mod_100_freq:.3f} asper")

    # Example 4: Stereo signal with different modulation frequencies
    print("\n4. Stereo signal (different modulation frequencies per channel)")
    print("-" * 70)
    stereo_signal = np.array([modulated_70hz, modulated_100hz])
    signal_stereo = wd.ChannelFrame(data=stereo_signal, sampling_rate=sample_rate)

    roughness_stereo = signal_stereo.roughness(method="time")

    print(f"Roughness Channel 1 (70 Hz mod):  {roughness_stereo[0]:.3f} asper")
    print(f"Roughness Channel 2 (100 Hz mod): {roughness_stereo[1]:.3f} asper")

    # Example 5: Using overlap parameter
    print("\n5. Effect of overlap parameter (time method only)")
    print("-" * 70)
    roughness_no_overlap = signal_mod_70.roughness(method="time", overlap=0.0)
    roughness_with_overlap = signal_mod_70.roughness(method="time", overlap=0.5)

    print(f"Roughness (no overlap):   {roughness_no_overlap:.3f} asper")
    print(f"Roughness (50% overlap):  {roughness_with_overlap:.3f} asper")
    print("Note: Overlap affects temporal averaging in time method")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("- Pure tone has very low roughness")
    print("- 70 Hz modulation creates higher roughness (near peak)")
    print("- 100 Hz modulation creates moderate roughness")
    print("- Roughness varies with modulation frequency")
    print("- Both time and freq methods are available")
    print("- Unit: asper (1 asper = roughness of 1kHz tone at 60dB, 100% mod at 70Hz)")
    print("=" * 70)


if __name__ == "__main__":
    main()
