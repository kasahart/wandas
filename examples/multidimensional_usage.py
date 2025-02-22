# examples/inter_channel_operations.py
from wandas.utils import generate_sample

# Create a multi-channel signal with 3 channels
signal = generate_sample.generate_sample(
    freqs=[440, 880, 1760], label="Multi-Channel Signal"
)

# Sum across channels
summed_signal = signal.sum()
summed_signal.plot(title="Sum of Channels")

# Compute mean across channels
mean_signal = signal.mean()
mean_signal.plot(title="Mean of Channels")

# Compute difference between channels
diff_signal = signal.channel_difference(other_channel=0)
diff_signal.plot(title="Difference between Channels")

# # Compute cross-correlation between channels
# correlations = signal.correlate_channels(mode='full')

# # Print shape of the correlation matrix
# print("Correlation matrix shape:", correlations.shape)

# # Plot cross-correlation between first and second channel
# import matplotlib.pyplot as plt

# corr = correlations[0, 1]
# lags = np.arange(-signal.num_samples + 1, signal.num_samples)
# plt.figure(figsize=(10, 4))
# plt.plot(lags, corr)
# plt.title("Cross-correlation between Channel 1 and Channel 2")
# plt.xlabel("Lag")
# plt.ylabel("Correlation")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
