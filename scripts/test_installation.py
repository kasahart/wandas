import sys


def main() -> int:
    try:
        import wandas as wd

        print(f"Successfully imported wandas version: {wd.__version__}")
        # Keep this as a lightweight smoke test for local installation checks.
        signal = wd.generate_sin(freqs=[5000, 1000], duration=1)
        signal.fft()
        return 0
    except ImportError:
        print("Error: Failed to import wandas.")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
