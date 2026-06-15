import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import tomli

from wandas.utils.optional_imports import require_optional_attr, require_optional_dependency


def _pyproject() -> dict:
    return tomli.loads(Path("pyproject.toml").read_text())


def test_runtime_dependencies_are_balanced_core_only() -> None:
    dependencies = set(_pyproject()["project"]["dependencies"])
    names = {dep.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].split("<", 1)[0] for dep in dependencies}

    assert {"numpy", "scipy", "dask", "pydantic", "soundfile"}.issubset(names)
    assert "matplotlib" not in names
    assert "librosa" not in names
    assert "ipykernel" not in names
    assert "ipywidgets" not in names
    assert "ipympl" not in names
    assert "ipycytoscape" not in names
    assert "japanize-matplotlib" not in names
    assert "pandas" not in names
    assert "h5py" not in names
    assert "mosqito" not in names
    assert "torch" not in names
    assert "tensorflow" not in names
    assert "types-requests" not in names


def test_optional_dependency_groups_exist() -> None:
    optional = _pyproject()["project"]["optional-dependencies"]

    assert set(optional) >= {"io", "viz", "notebook", "psychoacoustic", "ml"}
    assert "pandas" in optional["io"]
    assert any(dep.startswith("h5py") for dep in optional["io"])
    assert any(dep.startswith("matplotlib") for dep in optional["viz"])
    assert "librosa" in optional["viz"]
    assert any(dep.startswith("japanize-matplotlib") for dep in optional["viz"])
    assert "ipykernel" in optional["notebook"]
    assert "ipywidgets" in optional["notebook"]
    assert any(dep.startswith("ipympl") for dep in optional["notebook"])
    assert any(dep.startswith("ipycytoscape") for dep in optional["notebook"])
    assert "mosqito" in optional["psychoacoustic"]
    assert any(dep.startswith("torch") for dep in optional["ml"])
    assert any(dep.startswith("tensorflow") for dep in optional["ml"])


def test_require_optional_dependency_imports_installed_module() -> None:
    module = require_optional_dependency("math", extra="core", feature="test feature")
    assert module.sqrt(4) == 2


def test_require_optional_dependency_error_message() -> None:
    with pytest.raises(ImportError) as exc_info:
        require_optional_dependency(
            "definitely_missing_wandas_dependency",
            extra="viz",
            feature="plot",
        )

    message = str(exc_info.value)
    assert "plot requires optional dependency 'definitely_missing_wandas_dependency'" in message
    assert 'pip install "wandas[viz]"' in message


def test_require_optional_dependency_wraps_missing_parent_package() -> None:
    with pytest.raises(ImportError) as exc_info:
        require_optional_dependency(
            "definitely_missing_wandas_dependency.submodule",
            extra="viz",
            feature="plot",
        )

    message = str(exc_info.value)
    assert "plot requires optional dependency 'definitely_missing_wandas_dependency.submodule'" in message
    assert 'pip install "wandas[viz]"' in message


def test_require_optional_dependency_reraises_transitive_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_error = ModuleNotFoundError(
        "No module named 'missing_transitive_dependency'",
        name="missing_transitive_dependency",
    )

    def raise_transitive_error(module_name: str) -> None:
        assert module_name == "installed_optional_package"
        raise original_error

    monkeypatch.setattr(
        "wandas.utils.optional_imports.importlib.import_module",
        raise_transitive_error,
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        require_optional_dependency(
            "installed_optional_package",
            extra="viz",
            feature="plot",
        )

    assert exc_info.value is original_error


def test_require_optional_attr_returns_installed_attribute() -> None:
    sqrt = require_optional_attr("math", "sqrt", extra="core", feature="test feature")
    assert sqrt(9) == 3


def test_require_optional_attr_missing_attribute_error_message() -> None:
    with pytest.raises(ImportError) as exc_info:
        require_optional_attr(
            "math",
            "definitely_missing_wandas_attr",
            extra="viz",
            feature="plot",
        )

    message = str(exc_info.value)
    assert "plot requires 'math' to provide attribute 'definitely_missing_wandas_attr'" in message
    assert 'pip install "wandas[viz]"' in message


def test_import_wandas_without_librosa_or_mosqito() -> None:
    script = """
        import importlib.abc

        BLOCKED = {"librosa", "mosqito"}

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                top_level_name = fullname.split(".", 1)[0]
                if top_level_name in BLOCKED:
                    raise ModuleNotFoundError(
                        f"No module named {top_level_name!r}",
                        name=top_level_name,
                    )
                return None

        import sys
        sys.meta_path.insert(0, BlockOptionalImports())

        import wandas

        assert wandas.read_wav is not None
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_import_wandas_and_basic_waveform_ops_without_io_dependencies() -> None:
    script = """
        import importlib.abc
        import sys

        BLOCKED = {"h5py", "tqdm"}

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                top_level_name = fullname.split(".", 1)[0]
                if top_level_name in BLOCKED:
                    raise ModuleNotFoundError(
                        f"No module named {top_level_name!r}",
                        name=top_level_name,
                    )
                return None

        sys.meta_path.insert(0, BlockOptionalImports())

        import numpy as np
        import wandas

        frame = wandas.ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0]]), sampling_rate=48000)

        assert wandas.read_wav is not None
        assert np.allclose((frame + 1).to_numpy(), [[2.0, 3.0, 4.0]])
        assert "h5py" not in sys.modules
        assert "tqdm" not in sys.modules
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_import_wandas_and_basic_waveform_ops_without_visualization_or_notebook_dependencies() -> None:
    script = """
        import importlib.abc
        import sys

        BLOCKED = {
            "IPython",
            "ipycytoscape",
            "ipympl",
            "ipywidgets",
            "japanize_matplotlib",
            "librosa",
            "matplotlib",
        }

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                top_level_name = fullname.split(".", 1)[0]
                if top_level_name in BLOCKED:
                    raise ModuleNotFoundError(
                        f"No module named {top_level_name!r}",
                        name=top_level_name,
                    )
                return None

        sys.meta_path.insert(0, BlockOptionalImports())

        import numpy as np
        import wandas

        frame = wandas.ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0]]), sampling_rate=48000)

        assert np.isclose(frame.data.mean(), 2.0)
        for module_name in BLOCKED:
            assert module_name not in sys.modules
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_plot_missing_visualization_dependencies_has_viz_extra_hint() -> None:
    script = """
        import importlib.abc
        import sys

        BLOCKED = {"librosa", "matplotlib"}

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                top_level_name = fullname.split(".", 1)[0]
                if top_level_name in BLOCKED:
                    raise ModuleNotFoundError(
                        f"No module named {top_level_name!r}",
                        name=top_level_name,
                    )
                return None

        sys.meta_path.insert(0, BlockOptionalImports())

        import numpy as np
        import wandas

        frame = wandas.ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0]]), sampling_rate=48000)

        try:
            frame.plot()
        except ImportError as exc:
            assert 'pip install "wandas[viz]"' in str(exc)
        else:
            raise AssertionError("Expected ImportError for missing visualization dependency")
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_describe_missing_ipython_has_notebook_extra_hint() -> None:
    script = """
        import importlib.abc
        import sys

        class BlockIPython(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".", 1)[0] == "IPython":
                    raise ModuleNotFoundError("No module named 'IPython'", name="IPython")
                return None

        sys.meta_path.insert(0, BlockIPython())

        import numpy as np
        import wandas

        frame = wandas.ChannelFrame.from_numpy(np.array([[0.0, 1.0, 0.0, -1.0]]), sampling_rate=48000)

        try:
            frame.describe()
        except ImportError as exc:
            assert 'pip install "wandas[notebook]"' in str(exc)
        else:
            raise AssertionError("Expected ImportError for missing IPython")
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_wdf_save_missing_h5py_has_io_extra_hint() -> None:
    script = """
        import importlib.abc
        import sys
        import tempfile
        from pathlib import Path

        class BlockH5py(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".", 1)[0] == "h5py":
                    raise ModuleNotFoundError("No module named 'h5py'", name="h5py")
                return None

        sys.meta_path.insert(0, BlockH5py())

        import numpy as np
        import wandas as wd

        frame = wd.ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0]]), sampling_rate=48000)
        path = Path(tempfile.gettempdir()) / "wandas_missing_h5py_boundary.wdf"
        if path.exists():
            path.unlink()

        try:
            frame.save(path)
        except ImportError as exc:
            assert 'pip install "wandas[io]"' in str(exc)
        else:
            raise AssertionError("Expected ImportError for missing h5py")
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_frame_dataset_import_without_tqdm() -> None:
    script = """
        import importlib.abc
        import sys

        class BlockTqdm(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".", 1)[0] == "tqdm":
                    raise ModuleNotFoundError("No module named 'tqdm'", name="tqdm")
                return None

        sys.meta_path.insert(0, BlockTqdm())

        import wandas.utils.frame_dataset as frame_dataset

        assert frame_dataset.FrameDataset is not None
        assert "tqdm" not in sys.modules
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_psychoacoustic_missing_mosqito_has_extra_hint() -> None:
    script = """
        import importlib.abc

        class BlockMosqito(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".", 1)[0] == "mosqito":
                    raise ModuleNotFoundError("No module named 'mosqito'", name="mosqito")
                return None

        import sys
        sys.meta_path.insert(0, BlockMosqito())

        import numpy as np
        import wandas as wd

        frame = wd.ChannelFrame.from_numpy(np.zeros((1, 4800), dtype=float), sampling_rate=48000)

        try:
            frame.loudness_zwst()
        except ImportError as exc:
            message = str(exc)
            assert 'pip install "wandas[psychoacoustic]"' in message
        else:
            raise AssertionError("Expected ImportError for missing mosqito")
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_processing_lazy_registry_without_mosqito() -> None:
    script = """
        import importlib.abc
        import sys

        class BlockMosqito(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".", 1)[0] == "mosqito":
                    raise ModuleNotFoundError("No module named 'mosqito'", name="mosqito")
                return None

        sys.meta_path.insert(0, BlockMosqito())

        import wandas.processing as processing
        from wandas.processing import FFT, NOctSpectrum

        assert FFT.__name__ == "FFT"
        assert NOctSpectrum.__name__ == "NOctSpectrum"
        assert processing.get_operation("noct_spectrum").__name__ == "NOctSpectrum"
        assert "mosqito" not in sys.modules
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_noct_missing_mosqito_has_extra_hint() -> None:
    script = """
        import importlib.abc
        import sys

        class BlockMosqito(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".", 1)[0] == "mosqito":
                    raise ModuleNotFoundError("No module named 'mosqito'", name="mosqito")
                return None

        sys.meta_path.insert(0, BlockMosqito())

        from wandas.processing.spectral import NOctSpectrum

        op = NOctSpectrum(48000, fmin=20, fmax=20000)
        try:
            op.calculate_output_shape((1, 48000))
        except ImportError as exc:
            assert 'pip install "wandas[psychoacoustic]"' in str(exc)
        else:
            raise AssertionError("Expected ImportError for missing mosqito")
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_roughness_spec_missing_mosqito_has_extra_hint() -> None:
    script = """
        import importlib.abc
        import sys

        class BlockMosqito(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".", 1)[0] == "mosqito":
                    raise ModuleNotFoundError("No module named 'mosqito'", name="mosqito")
                return None

        sys.meta_path.insert(0, BlockMosqito())

        from wandas.processing.psychoacoustic import RoughnessDwSpec

        try:
            RoughnessDwSpec(48000)
        except ImportError as exc:
            assert 'pip install "wandas[psychoacoustic]"' in str(exc)
        else:
            raise AssertionError("Expected ImportError for missing mosqito")
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
