import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tomli

from wandas.utils.optional_imports import (
    DEPENDENCY_REGISTRY,
    require_dependency,
    require_dependency_attr,
    require_ipython_display,
    require_librosa_display,
    require_matplotlib_axes_type,
    require_mosqito_sq_metric,
    require_optional_attr,
    require_optional_dependency,
)

PROJECT_PACKAGE_BY_REGISTRY_KEY = {
    "pandas": "pandas",
    "matplotlib_pyplot": "matplotlib",
    "matplotlib_gridspec": "matplotlib",
    "matplotlib_axes": "matplotlib",
    "matplotlib_figure": "matplotlib",
    "matplotlib_lines": "matplotlib",
    "h5py": "h5py",
    "librosa": "librosa",
    "librosa_display": "librosa",
    "librosa_effects": "librosa",
    "mosqito_sq_metrics": "mosqito",
    "mosqito_sound_level_meter": "mosqito",
    "mosqito_center_freq": "mosqito",
    "ipython_display": "ipykernel",
    "torch": "torch",
    "tensorflow": "tensorflow",
}


def _pyproject() -> dict:
    return tomli.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def _dependency_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        name = requirement
        for separator in ("[", ">", "=", "<", "~", "!", ";"):
            name = name.split(separator, 1)[0]
        names.add(name.strip())
    return names


def _run_isolated_script(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_dependencies_are_balanced_core_only() -> None:
    names = _dependency_names(_pyproject()["project"]["dependencies"])

    assert {"numpy", "scipy", "dask", "soundfile", "pandas", "xarray"}.issubset(names)
    assert "matplotlib" in names
    assert "librosa" not in names
    assert "ipykernel" not in names
    assert "ipywidgets" not in names
    assert "ipympl" not in names
    assert "ipycytoscape" not in names
    assert "japanize-matplotlib" not in names
    assert "h5py" not in names
    assert "mosqito" not in names
    assert "torch" not in names
    assert "tensorflow" not in names
    assert "types-requests" not in names


def test_optional_dependency_groups_exist() -> None:
    optional = _pyproject()["project"]["optional-dependencies"]

    assert set(optional) >= {"io", "viz", "notebook", "psychoacoustic", "ml"}
    assert any(dep.startswith("h5py") for dep in optional["io"])
    assert "librosa" in optional["viz"]
    assert any(dep.startswith("japanize-matplotlib") for dep in optional["viz"])
    assert "ipykernel" in optional["notebook"]
    assert "ipywidgets" in optional["notebook"]
    assert any(dep.startswith("ipympl") for dep in optional["notebook"])
    assert any(dep.startswith("ipycytoscape") for dep in optional["notebook"])
    assert "mosqito" in optional["psychoacoustic"]
    assert any(dep.startswith("torch") for dep in optional["ml"])
    assert any(dep.startswith("tensorflow") for dep in optional["ml"])


def test_dependency_registry_matches_project_dependency_groups() -> None:
    pyproject = _pyproject()["project"]
    dependency_names_by_extra = {
        "core": _dependency_names(pyproject["dependencies"]),
        **{
            extra: _dependency_names(requirements) for extra, requirements in pyproject["optional-dependencies"].items()
        },
    }

    assert set(PROJECT_PACKAGE_BY_REGISTRY_KEY) == set(DEPENDENCY_REGISTRY)

    for key, spec in DEPENDENCY_REGISTRY.items():
        assert spec.extra in dependency_names_by_extra, key
        assert spec.import_name, key
        assert PROJECT_PACKAGE_BY_REGISTRY_KEY[key] in dependency_names_by_extra[spec.extra], key


def test_dependency_registry_install_hints_match_registered_extras() -> None:
    for key, spec in DEPENDENCY_REGISTRY.items():
        expected_hint = 'pip install "wandas"' if spec.extra == "core" else f'pip install "wandas[{spec.extra}]"'
        assert (spec.install_hint or expected_hint) == expected_hint, key


def test_require_dependency_imports_registered_module() -> None:
    module = require_dependency("pandas", feature="dataframe export")
    assert module.__name__ == "pandas"


def test_require_dependency_error_message_uses_registered_module_and_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")

    def raise_missing_matplotlib(module_name: str) -> None:
        assert module_name == "matplotlib.pyplot"
        raise original_error

    monkeypatch.setattr(
        "wandas.utils.optional_imports.importlib.import_module",
        raise_missing_matplotlib,
    )

    with pytest.raises(ImportError) as exc_info:
        require_dependency("matplotlib_pyplot", feature="roughness plot")

    message = str(exc_info.value)
    assert "roughness plot requires core dependency 'matplotlib.pyplot'" in message
    assert 'pip install "wandas"' in message


def test_require_dependency_attr_uses_registered_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = ModuleNotFoundError("No module named 'IPython'", name="IPython")

    def raise_missing_ipython(module_name: str) -> None:
        assert module_name == "IPython.display"
        raise original_error

    monkeypatch.setattr(
        "wandas.utils.optional_imports.importlib.import_module",
        raise_missing_ipython,
    )

    with pytest.raises(ImportError) as exc_info:
        require_dependency_attr("ipython_display", "Audio", feature="describe")

    message = str(exc_info.value)
    assert "describe requires optional dependency 'IPython.display'" in message
    assert 'pip install "wandas[notebook]"' in message


def test_require_dependency_reraises_registered_transitive_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = ModuleNotFoundError(
        "No module named 'missing_transitive_dependency'",
        name="missing_transitive_dependency",
    )

    def raise_transitive_error(module_name: str) -> None:
        assert module_name == "librosa"
        raise original_error

    monkeypatch.setattr(
        "wandas.utils.optional_imports.importlib.import_module",
        raise_transitive_error,
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        require_dependency("librosa", feature="spectrogram plot")

    assert exc_info.value is original_error


def test_require_dependency_attr_missing_attribute_error_message() -> None:
    with pytest.raises(ImportError) as exc_info:
        require_dependency_attr("pandas", "definitely_missing_wandas_attr", feature="dataframe export")

    message = str(exc_info.value)
    assert "dataframe export requires 'pandas' to provide attribute 'definitely_missing_wandas_attr'" in message
    assert 'pip install "wandas"' in message


def test_convenience_helpers_use_registered_import_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []

    def import_module(module_name: str) -> ModuleType:
        imported.append(module_name)
        module = ModuleType(module_name)
        if module_name == "matplotlib.axes":
            setattr(module, "Axes", object())
        elif module_name == "IPython.display":
            setattr(module, "display", object())
            setattr(module, "Audio", object())
        elif module_name == "mosqito.sq_metrics":
            setattr(module, "loudness_zwtv", object())
        return module

    monkeypatch.setattr("wandas.utils.optional_imports.importlib.import_module", import_module)

    assert require_librosa_display("spectrogram plot").__name__ == "librosa.display"
    assert require_matplotlib_axes_type("waveform plot") is not None
    display, audio = require_ipython_display("describe")
    assert display is not None
    assert audio is not None
    assert require_mosqito_sq_metric("loudness_zwtv", "loudness_zwtv") is not None

    assert imported == [
        "librosa.display",
        "matplotlib.axes",
        "IPython.display",
        "mosqito.sq_metrics",
    ]


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


def test_require_optional_dependency_reraises_import_error_without_missing_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = ModuleNotFoundError("optional package import failed")

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


def test_sharpness_din_tv_wrapper_missing_mosqito_names_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    from wandas.processing.psychoacoustic import sharpness_din_tv_mosqito

    original_error = ModuleNotFoundError("No module named 'mosqito'", name="mosqito")

    def raise_missing_mosqito(module_name: str) -> None:
        assert module_name == "mosqito.sq_metrics"
        raise original_error

    monkeypatch.setattr(
        "wandas.utils.optional_imports.importlib.import_module",
        raise_missing_mosqito,
    )

    with pytest.raises(ImportError) as exc_info:
        sharpness_din_tv_mosqito([0.0], 48000)

    message = str(exc_info.value)
    assert "sharpness_din_tv requires optional dependency 'mosqito.sq_metrics'" in message
    assert 'pip install "wandas[psychoacoustic]"' in message


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

    _run_isolated_script(script)


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

    _run_isolated_script(script)


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

    _run_isolated_script(script)


def test_spectrogram_plot_missing_librosa_has_viz_extra_hint() -> None:
    script = """
        import importlib.abc
        import sys

        BLOCKED = {"librosa"}

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

        data = np.sin(np.linspace(0, 16, 4096, dtype=float))[None, :]
        frame = wandas.ChannelFrame.from_numpy(data, sampling_rate=48000)

        try:
            frame.stft().plot()
        except ImportError as exc:
            assert 'pip install "wandas[viz]"' in str(exc)
        else:
            raise AssertionError("Expected ImportError for missing visualization dependency")
    """

    _run_isolated_script(script)


def test_describe_missing_ipython_has_notebook_extra_hint() -> None:
    script = """
        import importlib.abc
        import sys

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                top_level_name = fullname.split(".", 1)[0]
                if top_level_name in {"IPython", "librosa"}:
                    raise ModuleNotFoundError(f"No module named {top_level_name!r}", name=top_level_name)
                return None

        sys.meta_path.insert(0, BlockOptionalImports())

        import numpy as np
        import wandas

        data = np.sin(np.linspace(0, 16, 4096, dtype=float))[None, :]
        frame = wandas.ChannelFrame.from_numpy(data, sampling_rate=48000)

        try:
            frame.describe()
        except ImportError as exc:
            assert 'pip install "wandas[notebook]"' in str(exc)
        else:
            raise AssertionError("Expected ImportError for missing IPython")
    """

    _run_isolated_script(script)


def test_describe_return_figures_without_ipython_does_not_require_notebook_extra() -> None:
    script = """
        import importlib.abc
        import sys

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                top_level_name = fullname.split(".", 1)[0]
                if top_level_name in {"IPython", "librosa"}:
                    raise ModuleNotFoundError(f"No module named {top_level_name!r}", name=top_level_name)
                return None

        sys.meta_path.insert(0, BlockOptionalImports())

        import numpy as np
        import wandas

        data = np.sin(np.linspace(0, 16, 4096, dtype=float))[None, :]
        frame = wandas.ChannelFrame.from_numpy(data, sampling_rate=48000)

        figures = frame.describe(is_close=False)
        assert len(figures) == 1
    """

    _run_isolated_script(script)


def test_describe_image_save_without_ipython_does_not_require_notebook_extra() -> None:
    script = """
        import importlib.abc
        import sys
        import tempfile
        from pathlib import Path

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                top_level_name = fullname.split(".", 1)[0]
                if top_level_name in {"IPython", "librosa"}:
                    raise ModuleNotFoundError(f"No module named {top_level_name!r}", name=top_level_name)
                return None

        sys.meta_path.insert(0, BlockOptionalImports())

        import numpy as np
        import wandas

        data = np.sin(np.linspace(0, 16, 4096, dtype=float))[None, :]
        frame = wandas.ChannelFrame.from_numpy(data, sampling_rate=48000)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "describe.png"
            frame.describe(image_save=output)
            assert output.exists()
    """

    _run_isolated_script(script)


def test_hpss_harmonic_missing_librosa_effects_raises_at_init(monkeypatch: pytest.MonkeyPatch) -> None:
    from wandas.processing.effects import HpssHarmonic

    def raise_missing_librosa(feature: str) -> None:
        assert feature == "hpss_harmonic"
        raise ImportError(
            f"{feature} requires optional dependency 'librosa.effects'.\nInstall it with: pip install \"wandas[viz]\""
        )

    monkeypatch.setattr(
        "wandas.processing.effects.require_librosa_effects",
        raise_missing_librosa,
    )

    with pytest.raises(ImportError) as exc_info:
        HpssHarmonic(48000)

    message = str(exc_info.value)
    assert "hpss_harmonic requires optional dependency 'librosa.effects'" in message
    assert 'pip install "wandas[viz]"' in message


def test_remote_csv_missing_pandas_raises_before_download(monkeypatch: pytest.MonkeyPatch) -> None:
    from wandas.frames.channel import ChannelFrame

    def raise_missing_pandas(feature: str) -> None:
        assert feature == "CSV file reading"
        raise ImportError(
            "CSV file reading requires core dependency 'pandas'.\nInstall it with: pip install \"wandas\""
        )

    def fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("remote CSV should check pandas before download")

    monkeypatch.setattr("wandas.frames.channel._pandas", raise_missing_pandas)
    monkeypatch.setattr("wandas.frames.channel._download_url", fail_download)

    with pytest.raises(ImportError) as exc_info:
        ChannelFrame.from_file("https://example.com/data.csv")

    assert 'pip install "wandas"' in str(exc_info.value)


@pytest.mark.parametrize(
    ("method_name", "kwargs", "feature"),
    [
        ("loudness_zwtv", {"field_type": "free"}, "loudness_zwtv"),
        ("roughness_dw", {"overlap": 0.5}, "roughness_dw"),
        ("sharpness_din", {"weighting": "din", "field_type": "free"}, "sharpness_din"),
    ],
)
def test_lazy_psychoacoustic_missing_mosqito_raises_before_graph_build(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    kwargs: dict[str, object],
    feature: str,
) -> None:
    from wandas.frames.channel import ChannelFrame

    def raise_missing_mosqito(name: str, actual_feature: str) -> None:
        assert actual_feature == feature
        raise ImportError(
            f"{actual_feature} requires optional dependency 'mosqito.sq_metrics'.\n"
            'Install it with: pip install "wandas[psychoacoustic]"'
        )

    frame = ChannelFrame.from_numpy(np.zeros((1, 4800), dtype=float), sampling_rate=48000)
    monkeypatch.setattr("wandas.processing.psychoacoustic._sq_metric", raise_missing_mosqito)

    with pytest.raises(ImportError) as exc_info:
        getattr(frame, method_name)(**kwargs)

    assert 'pip install "wandas[psychoacoustic]"' in str(exc_info.value)


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

    _run_isolated_script(script)


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

    _run_isolated_script(script)


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

    _run_isolated_script(script)


def test_processing_getattr_unknown_name_raises_attribute_error() -> None:
    import wandas.processing as processing

    with pytest.raises(AttributeError, match="definitely_missing_operation"):
        processing.__getattr__("definitely_missing_operation")


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

    _run_isolated_script(script)


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

    _run_isolated_script(script)


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

    _run_isolated_script(script)
