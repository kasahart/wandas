"""Numeric plotting contracts for dedicated pairwise spectral Frames."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from matplotlib.axes import Axes

from tests.frame_helpers import channel_first_values
from tests.pairwise_test_helpers import make_pairwise_source
from wandas.frames.pairwise import CoherenceFrame, CrossSpectralFrame, TransferFunctionFrame
from wandas.visualization.plotting import _typed_frequency_values, _typed_matrix_entries, create_operation


def _axes_list(result: Axes | Iterator[Axes]) -> list[Axes]:
    return [result] if isinstance(result, Axes) else list(result)


def test_csd_frequency_views_plot_exact_numeric_values_and_custom_axes() -> None:
    frame = make_pairwise_source(n_channels=2).csd(
        n_fft=32,
        win_length=32,
        hop_length=16,
        window="boxcar",
        scaling="density",
    )
    raw = channel_first_values(frame)
    views = {
        "magnitude": np.abs(raw),
        "phase": np.angle(raw),
        "level": np.asarray(frame.level_db),
    }

    for view, expected in views.items():
        axes = _axes_list(frame.plot(view=view))
        assert len(axes) == frame.n_pairs
        assert all("CSD" in ax.get_ylabel() for ax in axes)
        for ax, expected_row in zip(axes, expected, strict=True):
            np.testing.assert_allclose(ax.lines[0].get_ydata(), expected_row, equal_nan=True)

    figure, axis = plt.subplots()
    result = frame.plot(
        ax=axis,
        overlay=True,
        view="phase",
        title="CSD phase",
        xlabel="Custom frequency",
        ylabel="Custom phase",
        alpha=0.5,
        xlim=(0.0, 100.0),
        ylim=(-4.0, 4.0),
        color="tab:red",
    )
    assert result is axis
    assert axis.get_title() == "CSD phase"
    assert axis.get_xlabel() == "Custom frequency"
    assert axis.get_ylabel() == "Custom phase"
    assert axis.get_xlim() == (0.0, 100.0)
    assert axis.get_ylim() == (-4.0, 4.0)
    assert len(axis.lines) == frame.n_pairs
    np.testing.assert_allclose(np.asarray(axis.lines[1].get_ydata()), np.asarray(np.angle(raw[1])), equal_nan=True)
    plt.close(figure)


def test_transfer_frequency_views_plot_gain_phase_and_both_level_definitions() -> None:
    frame = make_pairwise_source(
        n_channels=2,
        units=("V", "V"),
        references=(2.0, 5.0),
    ).transfer_function(
        n_fft=32,
        win_length=32,
        hop_length=16,
        window="boxcar",
        scaling="spectrum",
    )
    raw = channel_first_values(frame)
    views = {
        "gain": np.abs(raw),
        "phase": np.angle(raw),
        "gain_db": np.asarray(frame.gain_db),
        "transfer_level_db": np.asarray(frame.transfer_level_db),
    }

    for view, expected in views.items():
        axes = _axes_list(frame.plot(view=view))
        assert len(axes) == frame.n_pairs
        for ax, expected_row in zip(axes, expected, strict=True):
            np.testing.assert_allclose(ax.lines[0].get_ydata(), expected_row, equal_nan=True)

    assert frame.definition == "canonical_input_denominator"
    assert all("Transfer" in ax.get_ylabel() for ax in _axes_list(frame.plot(view="gain")))


def test_pairwise_matrix_plot_uses_typed_output_input_cells_and_leaves_sparse_cells_empty() -> None:
    frame = make_pairwise_source(n_channels=3).csd(
        n_fft=32,
        win_length=32,
        hop_length=16,
        window="boxcar",
        scaling="density",
    )
    subset = frame.select_pairs([6, 0, 8])
    raw = channel_first_values(subset)
    axes = _axes_list(subset.plot_matrix(view="magnitude"))

    assert len(axes) == 9
    for position, axis in enumerate(axes):
        output_index, input_index = divmod(position, 3)
        matching = [
            row
            for row, pair in enumerate(subset.pairs)
            if (pair.output.index, pair.input.index) == (output_index, input_index)
        ]
        if matching:
            assert len(axis.lines) == 1
            np.testing.assert_allclose(
                np.asarray(axis.lines[0].get_ydata()),
                np.asarray(np.abs(raw[matching[0]])),
                equal_nan=True,
            )
        else:
            assert not axis.lines


def test_pairwise_matrix_plot_materializes_values_once() -> None:
    frame = make_pairwise_source(n_channels=2).csd(
        n_fft=32,
        win_length=32,
        hop_length=16,
        window="boxcar",
        scaling="density",
    )
    original_compute = DaArray.compute
    with patch.object(DaArray, "compute", autospec=True, side_effect=original_compute) as compute:
        _axes_list(frame.plot_matrix(view="magnitude"))
    assert compute.call_count == 1


@pytest.mark.parametrize(
    "frame",
    [
        make_pairwise_source().csd(n_fft=32, win_length=32, hop_length=16, window="boxcar", scaling="density"),
        make_pairwise_source().transfer_function(
            n_fft=32,
            win_length=32,
            hop_length=16,
            window="boxcar",
            scaling="spectrum",
        ),
    ],
)
def test_csd_and_transfer_plotting_rejects_a_weighting(frame: CrossSpectralFrame | TransferFunctionFrame) -> None:
    with pytest.raises(ValueError, match="A-weighting"):
        frame.plot(Aw=True)
    with pytest.raises(ValueError, match="A-weighting"):
        frame.plot_matrix(Aw=True)


def test_coherence_plotting_is_typed_and_not_a_spectral_amplitude_view() -> None:
    frame = make_pairwise_source(n_channels=2).coherence(
        n_fft=32,
        win_length=32,
        hop_length=16,
        window="boxcar",
    )
    assert type(frame) is CoherenceFrame
    values = channel_first_values(frame)
    axes = _axes_list(frame.plot())
    assert all(axis.get_ylabel() == "Coherence" for axis in axes)
    for axis, expected in zip(axes, values, strict=True):
        np.testing.assert_allclose(axis.lines[0].get_ydata(), expected, equal_nan=True)
    with pytest.raises(ValueError, match="A-weighting"):
        frame.plot(Aw=True)


def test_typed_plot_views_reject_unknown_views_and_single_pair_matrix_is_two_dimensional() -> None:
    coherence = make_pairwise_source(n_channels=1).coherence(
        n_fft=32,
        win_length=32,
        hop_length=16,
        window="boxcar",
    )
    csd = make_pairwise_source(n_channels=2).csd(
        n_fft=32,
        win_length=32,
        hop_length=16,
        window="boxcar",
        scaling="density",
    )
    transfer = make_pairwise_source(n_channels=2).transfer_function(
        n_fft=32,
        win_length=32,
        hop_length=16,
        window="boxcar",
        scaling="spectrum",
    )

    with pytest.raises(ValueError, match="only the 'coherence'"):
        coherence.plot(view="unknown")
    with pytest.raises(ValueError, match="magnitude.*phase.*level"):
        csd.plot(view="unknown")
    with pytest.raises(ValueError, match="gain.*phase.*gain_db"):
        transfer.plot(view="unknown")

    axes = _axes_list(coherence.plot_matrix())
    assert len(axes) == 1
    assert len(axes[0].lines) == 1
    plt.close("all")


def test_plot_dispatch_rejects_noncallable_and_incomplete_typed_hooks() -> None:
    class BadFrequencyHook:
        _plot_frequency_values = object()

    class BadMatrixHook:
        _matrix_plot_entries = object()

    with pytest.raises(TypeError, match="_plot_frequency_values must be callable"):
        _typed_frequency_values(BadFrequencyHook(), view=None, aw=False)
    with pytest.raises(TypeError, match="_matrix_plot_entries must be callable"):
        _typed_matrix_entries(BadMatrixHook(), view=None, aw=False)

    class Shadowable:
        @staticmethod
        def _matrix_plot_entries(*, view: str | None, Aw: bool) -> tuple:  # noqa: N803
            del view, Aw
            return ()

        @staticmethod
        def _plot_frequency_values(*, view: str | None, Aw: bool) -> tuple[np.ndarray[Any, Any], str]:  # noqa: N803
            del view, Aw
            return np.array([0.0]), "Typed"

    shadowable = Shadowable()
    setattr(shadowable, "_matrix_plot_entries", object())
    setattr(shadowable, "_plot_frequency_values", object())
    assert _typed_matrix_entries(shadowable, view=None, aw=False) == ()
    values = _typed_frequency_values(shadowable, view=None, aw=False)
    assert values is not None
    np.testing.assert_array_equal(values[0], [0.0])
    assert values[1] == "Typed"

    class EntriesOnly:
        n_source_channels = 1

        @staticmethod
        def _matrix_plot_entries(*, view: str | None, Aw: bool) -> tuple:  # noqa: N803
            del view, Aw
            return ()

    class EmptyTyped:
        n_source_channels = 0

        @staticmethod
        def _matrix_plot_entries(*, view: str | None, Aw: bool) -> tuple:  # noqa: N803
            del view, Aw
            return ()

        @staticmethod
        def _plot_frequency_values(*, view: str | None, Aw: bool) -> tuple[np.ndarray[Any, Any], str]:  # noqa: N803
            del view, Aw
            return np.array([0.0]), "Typed"

    with pytest.raises(TypeError, match="frequency plotting contract"):
        create_operation("matrix").plot(EntriesOnly())
    with pytest.raises(ValueError, match="at least one source channel"):
        create_operation("matrix").plot(EmptyTyped())

    spectral = make_pairwise_source(n_channels=1).fft(n_fft=32)
    with pytest.raises(ValueError, match="supported only by a typed"):
        spectral.plot(view="magnitude")
    with pytest.raises(ValueError, match="supported only by a typed"):
        spectral.plot(plot_type="matrix", view="magnitude")
