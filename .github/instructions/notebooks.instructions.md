---
applyTo: '**/*.ipynb'
---

# Jupyter Notebook Standards for Wandas

## Documentation

- Include markdown cells explaining each step
- Use descriptive headings (# for main sections, ## for subsections)
- Explain the "why" not just the "what"
- Show expected outputs and visualizations

## Code Organization

### Cell Structure

```python
# Good: One logical operation per cell
# Cell 1: Import and setup
import wandas as wd
import matplotlib.pyplot as plt

# Cell 2: Load data
signal = wd.read_wav("audio.wav")

# Cell 3: Process
filtered = signal.low_pass_filter(cutoff=1000)

# Cell 4: Visualize
filtered.plot()
plt.show()
```

### Avoid

```python
# Bad: Everything in one cell
import wandas as wd
signal = wd.read_wav("audio.wav")
filtered = signal.low_pass_filter(cutoff=1000)
filtered.plot()
# Hard to debug and re-run
```

## Code Quality

Follow same standards as Python modules:
- Type hints on function definitions
- Clear variable names
- Method chaining for readability
- Comments for complex operations

```python
# Good: Clear and documented
def analyze_audio(filepath: str, cutoff_freq: float = 1000) -> wd.ChannelFrame:
    """
    Load and filter audio file.

    Parameters
    ----------
    filepath : str
        Path to audio file
    cutoff_freq : float
        Low-pass filter cutoff frequency in Hz

    Returns
    -------
    ChannelFrame
        Filtered audio signal
    """
    return (
        wd.read_wav(filepath)
        .normalize()
        .low_pass_filter(cutoff=cutoff_freq)
    )

# Use the function
filtered_signal = analyze_audio("audio.wav", cutoff_freq=800)
```

## Visualization

### Japanese Text Support

Use `japanize-matplotlib` for Japanese labels:

```python
import matplotlib.pyplot as plt
import japanize_matplotlib

signal.plot(
    title="音響信号の波形",
    xlabel="時間 [秒]",
    ylabel="振幅"
)
plt.show()
```

### Clear Plots

```python
# Good: Descriptive and formatted
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

signal.plot(ax=axes[0], title="Original Signal")
filtered.plot(ax=axes[1], title="Filtered Signal (1kHz cutoff)")

plt.tight_layout()
plt.show()
```

## Best Practices

### 1. Restart and Run All

Before committing, use "Restart Kernel and Run All Cells" to ensure:
- No hidden state dependencies
- Cells execute in order
- All outputs are current

### 2. Clear Outputs (Optional)

For version control, consider clearing outputs before committing:
```bash
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

### 3. Use Markdown for Explanations

```markdown
## Signal Processing Pipeline

This notebook demonstrates a complete signal processing workflow:

1. **Load audio**: Read WAV file
2. **Normalize**: Scale to [-1, 1]
3. **Filter**: Apply low-pass filter at 1kHz
4. **Analyze**: Compute and visualize spectrum

### Expected Results

The filtered signal should show attenuation of frequencies above 1kHz.
```

### 4. Show Intermediate Results

```python
# Show shapes and metadata
print(f"Original: {signal.shape}, SR: {signal.sampling_rate} Hz")
print(f"Filtered: {filtered.shape}, SR: {filtered.sampling_rate} Hz")

# Display operation history
print("\nOperation History:")
for op in filtered.operation_history:
    print(f"  - {op.name}: {op.params}")
```

## Example Notebook Structure

```markdown
# Audio Signal Analysis

## Setup
```

```python
import wandas as wd
import numpy as np
import matplotlib.pyplot as plt
```

```markdown
## Load Data
```

```python
signal = wd.read_wav("audio.wav")
print(f"Loaded {signal.duration:.2f}s of audio at {signal.sampling_rate} Hz")
```

```markdown
## Processing

Apply low-pass filter to remove high-frequency noise.
```

```python
filtered = signal.low_pass_filter(cutoff=1000)
```

```markdown
## Visualization

Compare original and filtered signals.
```

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
signal.plot(ax=axes[0], title="Original")
filtered.plot(ax=axes[1], title="Filtered")
plt.tight_layout()
plt.show()
```

## Interactive Widgets (Optional)

For exploratory analysis, use ipywidgets:

```python
from ipywidgets import interact, FloatSlider

@interact(cutoff=FloatSlider(min=100, max=5000, step=100, value=1000))
def filter_and_plot(cutoff: float):
    """Interactively adjust filter cutoff."""
    filtered = signal.low_pass_filter(cutoff=cutoff)
    filtered.plot(title=f"Filtered at {cutoff} Hz")
    plt.show()
```
