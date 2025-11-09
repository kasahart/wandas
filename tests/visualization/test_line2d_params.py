"""Tests for Line2D parameter filtering in plotting functions.

This module tests how matplotlib Line2D parameters are filtered
when passed to plotting functions via filter_kwargs.
"""

import pytest
from matplotlib.lines import Line2D

from wandas.utils.introspection import filter_kwargs


class TestLine2DParameterFiltering:
    """Test filtering of Line2D parameters."""

    def test_filter_line2d_valid_params(self) -> None:
        """Test filter_kwargs accepts valid Line2D parameters."""
        # Common Line2D parameters
        kwargs = {
            "color": "red",
            "linewidth": 2.0,
            "linestyle": "--",
            "marker": "o",
            "markersize": 5,
            "alpha": 0.8,
            "label": "test",
        }

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # All these are valid Line2D parameters, so they should be included
        assert "color" in filtered
        assert "linewidth" in filtered
        assert "linestyle" in filtered
        assert "marker" in filtered
        assert "markersize" in filtered
        assert "alpha" in filtered
        assert "label" in filtered

    def test_filter_line2d_rejects_invalid_params(self) -> None:
        """Test filter_kwargs rejects invalid Line2D parameters in strict mode."""
        # Mix of valid and invalid parameters
        kwargs = {
            "color": "blue",  # Valid
            "linewidth": 1.5,  # Valid
            "xlabel": "Time",  # Invalid for Line2D
            "ylabel": "Amplitude",  # Invalid for Line2D
            "title": "Plot Title",  # Invalid for Line2D
            "grid": True,  # Invalid for Line2D
        }

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # Valid parameters should be included
        assert "color" in filtered
        assert "linewidth" in filtered

        # Invalid parameters should be excluded in strict mode
        assert "xlabel" not in filtered
        assert "ylabel" not in filtered
        assert "title" not in filtered
        assert "grid" not in filtered

    def test_filter_line2d_color_param(self) -> None:
        """Test filtering of color parameter for Line2D."""
        kwargs = {"color": "red", "c": "blue"}

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # Both 'color' and 'c' are valid aliases in matplotlib
        # Check that at least one is included
        assert "color" in filtered or "c" in filtered

    def test_filter_line2d_linewidth_variants(self) -> None:
        """Test filtering of linewidth parameter variants."""
        kwargs = {"linewidth": 2.0, "lw": 3.0}

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # Both 'linewidth' and 'lw' are valid aliases
        # Check that at least one is included
        assert "linewidth" in filtered or "lw" in filtered

    def test_filter_line2d_linestyle_variants(self) -> None:
        """Test filtering of linestyle parameter variants."""
        kwargs = {"linestyle": "--", "ls": ":"}

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # Both 'linestyle' and 'ls' are valid aliases
        # Check that at least one is included
        assert "linestyle" in filtered or "ls" in filtered

    def test_filter_line2d_marker_params(self) -> None:
        """Test filtering of marker-related parameters."""
        kwargs = {
            "marker": "o",
            "markersize": 5,
            "markerfacecolor": "red",
            "markeredgecolor": "black",
            "markeredgewidth": 1.0,
        }

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # All marker-related parameters should be included
        assert "marker" in filtered
        assert "markersize" in filtered
        assert "markerfacecolor" in filtered
        assert "markeredgecolor" in filtered
        assert "markeredgewidth" in filtered

    def test_filter_line2d_alpha_param(self) -> None:
        """Test filtering of alpha (transparency) parameter."""
        kwargs = {"alpha": 0.5}

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        assert "alpha" in filtered
        assert filtered["alpha"] == 0.5

    def test_filter_line2d_label_param(self) -> None:
        """Test filtering of label parameter for legend."""
        kwargs = {"label": "Test Line"}

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        assert "label" in filtered
        assert filtered["label"] == "Test Line"

    def test_filter_line2d_empty_kwargs(self) -> None:
        """Test filtering with empty kwargs dictionary."""
        kwargs = {}

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        assert filtered == {}
        assert len(filtered) == 0

    def test_filter_line2d_preserves_values(self) -> None:
        """Test that filter_kwargs preserves parameter values correctly."""
        kwargs = {
            "color": "#FF5733",
            "linewidth": 2.5,
            "linestyle": "-.",
            "marker": "^",
            "alpha": 0.75,
        }

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # Verify values are preserved for included parameters
        for key in filtered:
            assert filtered[key] == kwargs[key]

    def test_filter_line2d_drawstyle_param(self) -> None:
        """Test filtering of drawstyle parameter."""
        kwargs = {"drawstyle": "steps-post"}

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # drawstyle is a valid Line2D parameter
        assert "drawstyle" in filtered
        assert filtered["drawstyle"] == "steps-post"

    def test_filter_line2d_with_axes_params(self) -> None:
        """Test that Axes-specific parameters are excluded from Line2D filtering."""
        kwargs = {
            "color": "green",  # Valid for Line2D
            "linewidth": 1.0,  # Valid for Line2D
            "xlim": (0, 10),  # Axes parameter
            "ylim": (-1, 1),  # Axes parameter
            "xscale": "log",  # Axes parameter
            "yscale": "linear",  # Axes parameter
        }

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # Line2D parameters should be included
        assert "color" in filtered
        assert "linewidth" in filtered

        # Axes parameters should be excluded
        assert "xlim" not in filtered
        assert "ylim" not in filtered
        assert "xscale" not in filtered
        assert "yscale" not in filtered

    def test_filter_line2d_zorder_param(self) -> None:
        """Test filtering of zorder parameter for drawing order."""
        kwargs = {"zorder": 10}

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # zorder is a valid Line2D parameter
        assert "zorder" in filtered
        assert filtered["zorder"] == 10

    def test_filter_line2d_antialiased_param(self) -> None:
        """Test filtering of antialiased parameter."""
        kwargs = {"antialiased": True}

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # antialiased (or aa) is a valid Line2D parameter
        assert "antialiased" in filtered or "aa" in filtered

    def test_filter_line2d_dash_params(self) -> None:
        """Test filtering of dash-related parameters."""
        kwargs = {
            "dashes": [5, 2, 2, 2],
            "dash_capstyle": "round",
            "dash_joinstyle": "round",
        }

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # These are valid Line2D parameters for customizing dashes
        assert "dashes" in filtered or "dash_capstyle" in filtered or "dash_joinstyle" in filtered


class TestLine2DParameterIntegration:
    """Integration tests for Line2D parameter filtering in context."""

    def test_plotting_with_line2d_params(self) -> None:
        """Test that Line2D parameters can be properly filtered for plotting."""
        # Simulate a typical plotting scenario
        user_kwargs = {
            "color": "red",
            "linewidth": 2.0,
            "xlabel": "Time [s]",  # Not a Line2D param
            "ylabel": "Amplitude",  # Not a Line2D param
        }

        plot_kwargs = filter_kwargs(Line2D, user_kwargs, strict_mode=True)

        # Should only contain Line2D parameters
        assert "color" in plot_kwargs
        assert "linewidth" in plot_kwargs
        assert "xlabel" not in plot_kwargs
        assert "ylabel" not in plot_kwargs

    def test_separate_filtering_for_plot_and_axes(self) -> None:
        """Test that kwargs can be split between Line2D and Axes parameters."""
        from matplotlib.axes import Axes

        kwargs = {
            "color": "blue",  # Line2D
            "linewidth": 1.5,  # Line2D
            "xlabel": "Time",  # Axes
            "ylabel": "Signal",  # Axes
            "xlim": (0, 10),  # Axes
        }

        line2d_kwargs = filter_kwargs(Line2D, kwargs, strict_mode=True)
        axes_kwargs = filter_kwargs(Axes.set, kwargs, strict_mode=True)

        # Line2D parameters
        assert "color" in line2d_kwargs
        assert "linewidth" in line2d_kwargs
        assert "xlabel" not in line2d_kwargs

        # Axes parameters
        assert "xlabel" in axes_kwargs
        assert "ylabel" in axes_kwargs
        assert "xlim" in axes_kwargs
        assert "color" not in axes_kwargs  # Not an Axes.set parameter

    def test_alpha_parameter_handling(self) -> None:
        """Test that alpha parameter is correctly filtered for Line2D."""
        kwargs = {"alpha": 0.6, "color": "green"}

        filtered = filter_kwargs(Line2D, kwargs, strict_mode=True)

        # alpha is popped from kwargs in the plotting code before filtering
        # but when passed to filter_kwargs, it should be recognized as valid
        assert "alpha" in filtered
        assert "color" in filtered
        assert filtered["alpha"] == 0.6
