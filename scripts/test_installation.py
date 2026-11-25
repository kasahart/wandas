"""Smoke-test a core-only wandas wheel installed into a fresh environment.

Run it with the environment's interpreter from outside the checkout, so the
installed wheel rather than the source tree is imported. The PR lane
(``ci.yml``), the Full Compatibility lane and the release workflow (``cd.yml``)
all run this one script; ``tests/test_ci_route.py`` checks that they do.
"""

import importlib
import importlib.util
import pkgutil
import sys
import tempfile
from pathlib import Path

# Packages that only extras install; a core-only wheel install must not pull them in.
OPTIONAL_MODULES = ("h5py", "h5netcdf", "IPython", "librosa", "mosqito", "tensorflow", "torch")


def main() -> int:
    try:
        import numpy as np

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

        installed_optional = [name for name in OPTIONAL_MODULES if importlib.util.find_spec(name) is not None]
        if installed_optional:
            print(f"Error: a core-only install pulled in optional packages: {installed_optional}")
            return 1

        import_errors = []
        for module_info in pkgutil.walk_packages(wd.__path__, prefix=f"{wd.__name__}."):
            try:
                importlib.import_module(module_info.name)
            except Exception as exc:  # noqa: BLE001 - the smoke should report every import failure.
                import_errors.append(f"{module_info.name}: {type(exc).__name__}: {exc}")
        if import_errors:
            print("Error: submodule imports failed:\n" + "\n".join(import_errors))
            return 1

        data = np.sin(np.linspace(0, 16, 4096, dtype=float))[None, :]
        frame = wd.ChannelFrame.from_numpy(data, sampling_rate=48000)
        if not np.allclose((frame + 1).to_numpy(), data + 1):
            print("Error: frame arithmetic returned unexpected values.")
            return 1
        if frame.to_dataframe().shape != (4096, 1):
            print("Error: to_dataframe() returned an unexpected shape.")
            return 1
        if frame.fft().n_channels != 1:
            print("Error: fft() returned an unexpected channel count.")
            return 1

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "describe.png"
            frame.describe(image_save=output)
            if not output.exists():
                print("Error: describe(image_save=...) wrote no file in a core-only install.")
                return 1
        return 0
    except ImportError as exc:
        print(f"Error: Failed to import wandas: {exc}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
