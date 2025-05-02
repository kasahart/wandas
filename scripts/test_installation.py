import sys

import wandas as wd

try:
    print(f"Successfully imported wandas version: {wd.__version__}")
    # ここに、必要に応じてさらに基本的な機能テストを追加できます
    signal = wd.generate_sin(freqs=[5000, 1000], duration=1)
    signal.fft()

    sys.exit(0)
except ImportError:
    print("Error: Failed to import wandas.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)
