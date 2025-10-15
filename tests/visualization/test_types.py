"""Tests for visualization type definitions."""

from typing import Any

from wandas.visualization.types import (
    DescribeParams,
    SpectralConfig,
    WaveformConfig,
)


class TestWaveformConfig:
    """Tests for WaveformConfig TypedDict."""

    def test_waveform_config_all_fields(self) -> None:
        """Test WaveformConfig with all fields."""
        config: WaveformConfig = {
            "xlabel": "Time [s]",
            "ylabel": "Amplitude",
            "xlim": (0, 10),
            "ylim": (-1, 1),
        }

        assert config["xlabel"] == "Time [s]"
        assert config["ylabel"] == "Amplitude"
        assert config["xlim"] == (0, 10)
        assert config["ylim"] == (-1, 1)

    def test_waveform_config_partial_fields(self) -> None:
        """Test WaveformConfig with partial fields (total=False)."""
        config: WaveformConfig = {
            "ylabel": "Sound Pressure [Pa]",
        }

        assert config["ylabel"] == "Sound Pressure [Pa]"
        assert "xlabel" not in config

    def test_waveform_config_empty(self) -> None:
        """Test WaveformConfig can be empty (total=False)."""
        config: WaveformConfig = {}
        assert len(config) == 0

    def test_waveform_config_xlim_tuple(self) -> None:
        """Test WaveformConfig xlim accepts tuple."""
        config: WaveformConfig = {
            "xlim": (0.0, 5.5),
        }

        assert isinstance(config["xlim"], tuple)
        assert config["xlim"][0] == 0.0
        assert config["xlim"][1] == 5.5


class TestSpectralConfig:
    """Tests for SpectralConfig TypedDict."""

    def test_spectral_config_all_fields(self) -> None:
        """Test SpectralConfig with all fields."""
        config: SpectralConfig = {
            "xlabel": "Frequency [Hz]",
            "ylabel": "Magnitude [dB]",
            "xlim": (-80, -20),
            "ylim": (20, 20000),
        }

        assert config["xlabel"] == "Frequency [Hz]"
        assert config["ylabel"] == "Magnitude [dB]"
        assert config["xlim"] == (-80, -20)
        assert config["ylim"] == (20, 20000)

    def test_spectral_config_partial_fields(self) -> None:
        """Test SpectralConfig with partial fields (total=False)."""
        config: SpectralConfig = {
            "xlabel": "Freq [Hz]",
            "xlim": (-90, -10),
        }

        assert config["xlabel"] == "Freq [Hz]"
        assert config["xlim"] == (-90, -10)
        assert "ylabel" not in config

    def test_spectral_config_empty(self) -> None:
        """Test SpectralConfig can be empty (total=False)."""
        config: SpectralConfig = {}
        assert len(config) == 0


class TestDescribeParams:
    """Tests for DescribeParams TypedDict."""

    def test_describe_params_all_fields(self) -> None:
        """Test DescribeParams with all fields."""
        waveform: WaveformConfig = {"ylabel": "Amplitude"}
        spectral: SpectralConfig = {"ylabel": "Magnitude"}

        params: DescribeParams = {
            "fmin": 100.0,
            "fmax": 5000.0,
            "cmap": "viridis",
            "vmin": -80.0,
            "vmax": -20.0,
            "xlim": (0, 10),
            "ylim": (20, 20000),
            "Aw": True,
            "waveform": waveform,
            "spectral": spectral,
            "normalize": True,
            "is_close": False,
            "axis_config": {"time_plot": {}},
            "cbar_config": {"vmin": -80},
        }

        assert params["fmin"] == 100.0
        assert params["fmax"] == 5000.0
        assert params["cmap"] == "viridis"
        assert params["vmin"] == -80.0
        assert params["vmax"] == -20.0
        assert params["xlim"] == (0, 10)
        assert params["ylim"] == (20, 20000)
        assert params["Aw"] is True
        assert params["waveform"]["ylabel"] == "Amplitude"
        assert params["spectral"]["ylabel"] == "Magnitude"
        assert params["normalize"] is True
        assert params["is_close"] is False

    def test_describe_params_minimal(self) -> None:
        """Test DescribeParams with minimal fields."""
        params: DescribeParams = {
            "fmin": 20,
            "fmax": 20000,
        }

        assert params["fmin"] == 20
        assert params["fmax"] == 20000
        assert "cmap" not in params

    def test_describe_params_frequency_only(self) -> None:
        """Test DescribeParams with frequency parameters only."""
        params: DescribeParams = {
            "fmin": 100,
            "fmax": 5000,
            "cmap": "jet",
        }

        assert params["fmin"] == 100
        assert params["fmax"] == 5000
        assert params["cmap"] == "jet"

    def test_describe_params_with_a_weighting(self) -> None:
        """Test DescribeParams with A-weighting parameter."""
        params: DescribeParams = {
            "Aw": True,
            "vmin": -80,
            "vmax": -20,
        }

        assert params["Aw"] is True
        assert params["vmin"] == -80
        assert params["vmax"] == -20

    def test_describe_params_with_axis_limits(self) -> None:
        """Test DescribeParams with axis limit parameters."""
        params: DescribeParams = {
            "xlim": (0, 5),
            "ylim": (20, 20000),
        }

        assert params["xlim"] == (0, 5)
        assert params["ylim"] == (20, 20000)

    def test_describe_params_with_subplot_configs(self) -> None:
        """Test DescribeParams with nested subplot configurations."""
        waveform_config: WaveformConfig = {
            "xlabel": "Time [s]",
            "ylabel": "Sound Pressure [Pa]",
            "xlim": (0, 10),
        }

        spectral_config: SpectralConfig = {
            "xlabel": "Frequency [Hz]",
            "ylabel": "SPL [dB]",
            "xlim": (-80, -20),
        }

        params: DescribeParams = {
            "fmin": 20,
            "fmax": 20000,
            "waveform": waveform_config,
            "spectral": spectral_config,
        }

        assert params["waveform"]["xlabel"] == "Time [s]"
        assert params["waveform"]["ylabel"] == "Sound Pressure [Pa]"
        assert params["spectral"]["xlabel"] == "Frequency [Hz]"
        assert params["spectral"]["ylabel"] == "SPL [dB]"

    def test_describe_params_deprecated_fields(self) -> None:
        """Test DescribeParams with deprecated fields for backward compatibility."""
        axis_config: dict[str, Any] = {
            "time_plot": {"ylabel": "Custom"},
            "freq_plot": {"xlim": (-80, -20)},
        }

        cbar_config: dict[str, Any] = {
            "vmin": -90,
            "vmax": -10,
        }

        params: DescribeParams = {
            "axis_config": axis_config,
            "cbar_config": cbar_config,
        }

        assert "axis_config" in params
        assert "cbar_config" in params
        assert params["axis_config"]["time_plot"]["ylabel"] == "Custom"
        assert params["cbar_config"]["vmin"] == -90

    def test_describe_params_empty(self) -> None:
        """Test DescribeParams can be empty (total=False)."""
        params: DescribeParams = {}
        assert len(params) == 0

    def test_describe_params_can_be_expanded(self) -> None:
        """Test DescribeParams can be expanded with ** operator."""
        params: DescribeParams = {
            "fmin": 100,
            "fmax": 5000,
            "Aw": True,
        }

        # This simulates how it would be used in practice
        def mock_describe(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        result = mock_describe(**params)

        assert result["fmin"] == 100
        assert result["fmax"] == 5000
        assert result["Aw"] is True

    def test_describe_params_merge_configs(self) -> None:
        """Test merging multiple DescribeParams configurations."""
        base_config: DescribeParams = {
            "fmin": 20,
            "fmax": 20000,
            "cmap": "jet",
        }

        custom_config: DescribeParams = {
            "Aw": True,
            "vmin": -80,
            "vmax": -20,
        }

        # Merge configurations
        merged: DescribeParams = {**base_config, **custom_config}

        assert merged["fmin"] == 20
        assert merged["fmax"] == 20000
        assert merged["cmap"] == "jet"
        assert merged["Aw"] is True
        assert merged["vmin"] == -80
        assert merged["vmax"] == -20

    def test_describe_params_override_values(self) -> None:
        """Test overriding values in DescribeParams."""
        base: DescribeParams = {
            "fmin": 20,
            "fmax": 20000,
            "cmap": "jet",
            "Aw": False,
        }

        # Override with new values
        modified: DescribeParams = {
            **base,
            "cmap": "viridis",
            "Aw": True,
        }

        assert modified["fmin"] == 20  # unchanged
        assert modified["fmax"] == 20000  # unchanged
        assert modified["cmap"] == "viridis"  # changed
        assert modified["Aw"] is True  # changed

    def test_describe_params_none_values(self) -> None:
        """Test DescribeParams with None values for optional fields."""
        params: DescribeParams = {
            "fmin": 0,
            "fmax": None,  # Optional[float]
            "vmin": None,  # Optional[float]
            "vmax": None,  # Optional[float]
            "xlim": None,  # Optional[tuple]
            "ylim": None,  # Optional[tuple]
        }

        assert params["fmin"] == 0
        assert params["fmax"] is None
        assert params["vmin"] is None
        assert params["vmax"] is None
        assert params["xlim"] is None
        assert params["ylim"] is None


class TestTypedDictIntegration:
    """Integration tests for TypedDict usage patterns."""

    def test_nested_config_construction(self) -> None:
        """Test building complex nested configurations."""
        waveform: WaveformConfig = {
            "ylabel": "Amplitude [Pa]",
            "xlim": (0, 10),
        }

        spectral: SpectralConfig = {
            "ylabel": "SPL [dB]",
            "xlim": (-90, -10),
        }

        params: DescribeParams = {
            "fmin": 100,
            "fmax": 5000,
            "cmap": "viridis",
            "Aw": True,
            "waveform": waveform,
            "spectral": spectral,
            "normalize": True,
            "is_close": False,
        }

        # Verify structure
        assert isinstance(params["waveform"], dict)
        assert isinstance(params["spectral"], dict)
        assert params["waveform"]["ylabel"] == "Amplitude [Pa]"
        assert params["spectral"]["ylabel"] == "SPL [dB]"

    def test_config_from_dict(self) -> None:
        """Test creating TypedDict from plain dict (e.g., JSON load)."""
        # Simulate loading from JSON
        json_data: dict[str, Any] = {
            "fmin": 100,
            "fmax": 5000,
            "cmap": "viridis",
            "Aw": True,
            "vmin": -80,
            "vmax": -20,
        }

        # Cast to TypedDict
        params: DescribeParams = json_data  # type: ignore

        assert params["fmin"] == 100
        assert params["fmax"] == 5000
        assert params["cmap"] == "viridis"
        assert params["Aw"] is True

    def test_partial_config_update(self) -> None:
        """Test updating config with partial values."""
        base: DescribeParams = {
            "fmin": 20,
            "fmax": 20000,
            "cmap": "jet",
        }

        # Update only specific fields
        update: dict[str, Any] = {
            "Aw": True,
            "vmin": -80,
        }

        result: DescribeParams = {**base, **update}  # type: ignore

        assert result["fmin"] == 20  # preserved
        assert result["Aw"] is True  # added
        assert result["vmin"] == -80  # added

    def test_typeddict_with_real_channelframe(self) -> None:
        """Test TypedDict parameters with actual ChannelFrame instance."""
        from unittest import mock

        import numpy as np

        import wandas as ws

        # Create test signal
        t = np.linspace(0, 1, 16000)
        signal = np.sin(2 * np.pi * 440 * t)
        cf = ws.ChannelFrame.from_numpy(data=signal.reshape(1, -1), sampling_rate=16000)

        # Create config with TypedDict
        config: DescribeParams = {
            "fmin": 100,
            "fmax": 5000,
            "cmap": "viridis",
            "Aw": True,
        }

        # Mock display to avoid showing plots in tests
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            # This should work without errors
            cf.describe(**config)  # type: ignore

    def test_typeddict_parameter_validation(self) -> None:
        """Test that TypedDict helps catch parameter errors."""
        # Correct parameters
        valid_config: DescribeParams = {
            "fmin": 100,
            "fmax": 5000,
            "Aw": True,
        }

        assert valid_config["fmin"] == 100
        assert valid_config["fmax"] == 5000
        assert valid_config["Aw"] is True

        # Type checker would catch these errors (not runtime errors)
        # invalid_config: DescribeParams = {
        #     "fmin": "100",  # Should be float, not str
        #     "Aw": "yes",    # Should be bool, not str
        # }

    def test_typeddict_reusability(self) -> None:
        """Test reusing TypedDict configurations across multiple calls."""
        from unittest import mock

        import numpy as np

        import wandas as ws

        # Create multiple signals
        signals = []
        for freq in [440, 880, 1320]:
            t = np.linspace(0, 1, 16000)
            signal = np.sin(2 * np.pi * freq * t)
            cf = ws.ChannelFrame.from_numpy(
                data=signal.reshape(1, -1), sampling_rate=16000, label=f"{freq}Hz"
            )
            signals.append(cf)

        # Single config for all signals
        shared_config: DescribeParams = {
            "fmin": 100,
            "fmax": 2000,
            "cmap": "viridis",
            "Aw": True,
            "vmin": -80,
            "vmax": -20,
        }

        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            # Apply same config to all signals
            for signal in signals:
                signal.describe(**shared_config)  # type: ignore

        # All completed successfully
        assert len(signals) == 3

    def test_typeddict_config_variants(self) -> None:
        """Test creating config variants from base configuration."""
        from unittest import mock

        import numpy as np

        import wandas as ws

        # Create test signal
        t = np.linspace(0, 1, 16000)
        signal = np.sin(2 * np.pi * 440 * t)
        cf = ws.ChannelFrame.from_numpy(data=signal.reshape(1, -1), sampling_rate=16000)

        # Base configuration
        base_config: DescribeParams = {
            "fmin": 20,
            "fmax": 20000,
            "cmap": "jet",
        }

        # Variant 1: With A-weighting
        acoustic_config: DescribeParams = {
            **base_config,
            "Aw": True,
            "vmin": -80,
            "vmax": -20,
        }

        # Variant 2: Different colormap
        dark_config: DescribeParams = {
            **base_config,
            "cmap": "magma",
        }

        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            # Test all variants
            cf.describe(**base_config)  # type: ignore
            cf.describe(**acoustic_config)  # type: ignore
            cf.describe(**dark_config)  # type: ignore

        # All variants executed successfully
