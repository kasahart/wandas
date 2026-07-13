import numpy as np

from tests.pipeline.custom_recipe_fixtures import custom_scale
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan


def _frame(value: float = 1.0) -> ChannelFrame:
    return ChannelFrame.from_numpy(np.full((1, 32), value), sampling_rate=8000)


def test_typed_frame_transition_replays() -> None:
    source = _frame()
    processed = source.fft(n_fft=16)
    replayed = RecipePlan.from_frame(processed).apply({"input_0": source})

    assert type(replayed) is type(processed)
    assert replayed.shape == processed.shape


def test_add_channel_frame_and_array_replay() -> None:
    source = _frame()
    added = _frame(2)
    frame_plan = RecipePlan.from_frame(source.add_channel(added, label="added"), input_names=("base", "added"))
    array_plan = RecipePlan.from_frame(
        source.add_channel(np.full((1, 32), 3.0), label="raw"), input_names=("base", "data")
    )

    assert frame_plan.apply({"base": source, "added": added}).n_channels == 2
    assert array_plan.apply({"base": source, "data": np.full((1, 32), 3.0)}).n_channels == 2


def test_custom_function_replays_by_stable_path() -> None:
    source = _frame()
    processed = source.apply(custom_scale, gain=2.0)
    replayed = RecipePlan.from_frame(processed).apply({"input_0": source})

    np.testing.assert_allclose(replayed.compute(), processed.compute())


def test_true_multi_input_replays_in_role_order() -> None:
    signal = _frame(1)
    noise = _frame(2)
    processed = signal.add(noise, snr=3.0)
    plan = RecipePlan.from_frame(processed, input_names=("signal", "noise"))
    replayed = plan.apply({"signal": signal, "noise": noise})

    np.testing.assert_allclose(replayed.compute(), processed.compute())


def test_metadata_and_source_time_offset_are_preserved() -> None:
    source = _frame()
    source.metadata["domain"] = "test"
    source.source_time_offset = [0.25]
    processed = source[:, 4:12]
    replayed = RecipePlan.from_frame(processed).apply({"input_0": source})

    assert replayed.metadata == processed.metadata
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)
