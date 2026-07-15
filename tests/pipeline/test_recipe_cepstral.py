"""Recipe v2 round-trip coverage for the public cepstral workflow."""

from unittest.mock import patch

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from tests.pipeline.recipe_test_helpers import RECIPE_SAMPLE_RATE, make_recipe_source
from wandas.frames.cepstral import CepstralFrame
from wandas.frames.spectral import SpectralFrame
from wandas.pipeline import RecipePlan


def test_cepstral_workflow_extracts_serializes_loads_and_applies_without_compute() -> None:
    source = make_recipe_source(16, offset=0.25)
    replay_source = make_recipe_source(32, offset=1.5)

    with patch.object(DaArray, "compute", autospec=True, side_effect=AssertionError("unexpected compute")):
        processed = (
            source.cepstrum(window="boxcar")
            .lifter(
                cutoff=2 / RECIPE_SAMPLE_RATE,
            )
            .to_spectral_envelope()
        )
        plan = RecipePlan.from_frame(processed, input_names=("signal",))
        payload = plan.to_dict()
        loaded = RecipePlan.from_dict(payload)
        replayed = loaded.apply({"signal": replay_source})
        expected = (
            replay_source.cepstrum(window="boxcar")
            .lifter(
                cutoff=2 / RECIPE_SAMPLE_RATE,
            )
            .to_spectral_envelope()
        )

    assert loaded.to_dict() == payload
    assert isinstance(replayed, SpectralFrame)
    assert isinstance(replayed._data, da.Array)
    assert replayed.n_fft == 32
    assert replayed.metadata == replay_source.metadata
    np.testing.assert_array_equal(replayed.source_time_offset, replay_source.source_time_offset)
    np.testing.assert_allclose(replayed.compute(), expected.compute(), rtol=1e-12, atol=1e-12)
    assert [entry["operation"] for entry in replayed.operation_history] == [
        "wandas.audio.cepstrum",
        "wandas.cepstral.lifter",
        "wandas.cepstral.to_spectral_envelope",
    ]


def test_recipe_intermediate_output_keeps_cepstral_frame_type() -> None:
    source = make_recipe_source(16)
    processed = source.cepstrum(n_fft=32, window="boxcar")

    replayed = RecipePlan.from_dict(RecipePlan.from_frame(processed, input_names=("signal",)).to_dict()).apply(
        {"signal": source}
    )

    assert isinstance(replayed, CepstralFrame)
    assert replayed.n_fft == 32
    np.testing.assert_array_equal(replayed.quefrencies, np.arange(32) / RECIPE_SAMPLE_RATE)
