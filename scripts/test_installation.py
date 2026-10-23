import sys
from pathlib import Path


def main() -> int:
    try:
        import wandas as wd

        print(f"Successfully imported wandas version: {wd.__version__} from {wd.__file__}")
        package_dir = Path(wd.__file__).parent.resolve()
        # An installed wheel lives under the interpreter's prefix; a checkout does not.
        if not package_dir.is_relative_to(Path(sys.prefix).resolve()):
            print(f"Error: wandas was imported from {package_dir}, not from the environment at {sys.prefix}.")
            return 1
        if not (package_dir / "py.typed").is_file():
            print("Error: the installed package has no py.typed marker.")
            return 1
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
