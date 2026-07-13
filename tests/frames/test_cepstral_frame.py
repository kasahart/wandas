from typing import Any, cast

import dask.array as da
import numpy as np
import pytest

from wandas.core.metadata import ChannelMetadata
from wandas.frames.cepstral import CepstralFrame
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame


class TestCepstralFrame:
    def setup_method(self) -> None:
        self.sampling_rate = 16000
        self.n_fft = 1024
        self.data = da.from_array(np.zeros((2, self.n_fft), dtype=np.float64), chunks=(1, -1))
        self.frame = CepstralFrame(
            data=self.data,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            channel_metadata=[
                ChannelMetadata(label="ch0", ref=1.0),
                ChannelMetadata(label="ch1", ref=1.0),
            ],
            label="ceps",
        )

    def test_constructor_rejects_higher_dimensional_data(self) -> None:
        data = da.zeros((2, 3, 4), chunks=(1, -1, -1))

        with pytest.raises(ValueError, match=r"must be 1D or 2D"):
            CepstralFrame(data=data, sampling_rate=self.sampling_rate, n_fft=4)

    def test_constructor_rejects_complex_data(self) -> None:
        data = da.zeros((1, 4), chunks=(1, -1), dtype=np.complex128)

        with pytest.raises(TypeError, match=r"real-valued coefficients"):
            CepstralFrame(data=data, sampling_rate=self.sampling_rate, n_fft=4)

    def test_quefrencies_property_matches_sampling_rate(self) -> None:
        expected = np.arange(self.n_fft) / self.sampling_rate
        np.testing.assert_allclose(self.frame.quefrencies, expected)

    def test_quefrency_axis_matches_sliced_data_length(self) -> None:
        sliced = self.frame[:, :8]

        assert sliced.shape == (2, 8)
        assert sliced.quefrencies.shape == (8,)
        assert sliced.to_dataframe().index.shape == (8,)
        assert sliced.to_xarray().sizes["quefrency"] == 8

    @pytest.mark.parametrize("axis_slice", [slice(4, 8), slice(None, None, 2)])
    def test_quefrency_axis_preserves_slice_coordinates(self, axis_slice: slice) -> None:
        sliced = self.frame[:, axis_slice]
        expected = self.frame.quefrencies[axis_slice]

        np.testing.assert_allclose(sliced.quefrencies, expected)
        np.testing.assert_allclose(sliced.to_dataframe().index.to_numpy(), expected)
        np.testing.assert_allclose(sliced.to_xarray().coords["quefrency"].to_numpy(), expected)

    def test_lifter_rejects_sliced_quefrency_coordinates(self) -> None:
        sliced = self.frame[:, 4:12:2]

        with pytest.raises(ValueError, match=r"requires a complete, unsliced quefrency axis"):
            sliced.lifter(cutoff=1 / self.sampling_rate, mode="low")

    @pytest.mark.parametrize("index", [4, -1])
    def test_integer_quefrency_selection_preserves_axis(self, index: int) -> None:
        result = self.frame[:, index]

        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.quefrencies, [self.frame.quefrencies[index]])

    def test_numpy_integer_quefrency_selection_preserves_axis(self) -> None:
        result = self.frame[:, np.int64(4)]

        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.quefrencies, [self.frame.quefrencies[4]])

    @pytest.mark.parametrize("index", [1024, -1025])
    def test_integer_quefrency_selection_rejects_out_of_range(self, index: int) -> None:
        with pytest.raises(IndexError, match=r"Quefrency index out of range"):
            _ = self.frame[:, index]

    def test_xarray_coordinate_mutation_does_not_change_frame(self) -> None:
        expected = self.frame.quefrencies
        public = self.frame.to_xarray()

        public.coords["quefrency"].values[0] = 123.0

        np.testing.assert_allclose(self.frame.quefrencies, expected)

    def test_lifter_returns_new_frame_with_history_and_labels(self) -> None:
        result = self.frame.lifter(cutoff=0.001, mode="low")

        assert isinstance(result, CepstralFrame)
        assert result is not self.frame
        assert isinstance(result._data, da.Array)
        assert result.operation_history[-1] == {"operation": "lifter", "params": {"cutoff": 0.001, "mode": "low"}}
        assert result.labels == ["ch0", "ch1"]
        np.testing.assert_allclose(self.frame.compute(), np.zeros((2, self.n_fft)))

    def test_to_spectral_envelope_returns_spectral_frame(self) -> None:
        result = self.frame.to_spectral_envelope()

        assert isinstance(result, SpectralFrame)
        assert result.n_fft == self.n_fft
        assert result.operation_history[-1] == {
            "operation": "spectral_envelope",
            "params": {"window": "hann", "window_length": 1024},
        }
        expected = np.full((2, self.n_fft // 2 + 1), 2.0 / np.sum(np.hanning(self.n_fft + 1)[:-1]))
        expected[..., 0] *= 0.5
        expected[..., -1] *= 0.5
        np.testing.assert_allclose(result.compute().real, expected)

    def test_cepstral_workflow_preserves_source_file_metadata(self) -> None:
        frame = ChannelFrame.from_numpy(
            np.ones((1, 1024), dtype=np.float64),
            self.sampling_rate,
            metadata={"tag": "voice", "_source_file": "speech.wav"},
        )

        cepstrum = frame.cepstrum(n_fft=1024, window="boxcar")
        low_ceps = cepstrum.lifter(cutoff=0.001, mode="low")
        envelope = low_ceps.to_spectral_envelope()

        assert cepstrum.metadata["_source_file"] == "speech.wav"
        assert low_ceps.metadata["_source_file"] == "speech.wav"
        assert envelope.metadata["_source_file"] == "speech.wav"
        assert cepstrum.operation_history[-1]["operation"] == "cepstrum"
        assert low_ceps.operation_history[-1] == {"operation": "lifter", "params": {"cutoff": 0.001, "mode": "low"}}
        assert envelope.operation_history[-1] == {
            "operation": "spectral_envelope",
            "params": {"window": "boxcar", "window_length": 1024},
        }

    def test_padded_cepstrum_preserves_analysis_window_scaling(self) -> None:
        rng = np.random.default_rng(43)
        frame = ChannelFrame.from_numpy(rng.normal(size=(1, 64)), self.sampling_rate)

        cepstrum = frame.cepstrum(n_fft=128, window="boxcar")
        envelope = cepstrum.to_spectral_envelope()
        spectrum = frame.fft(n_fft=128, window="boxcar")

        assert cepstrum.analysis_length == 64
        np.testing.assert_allclose(envelope.compute().real, np.abs(spectrum.compute()), rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("attribute", ["ifft", "noct_synthesis", "magnitude", "phase", "dB", "freqs"])
    def test_spectral_api_is_not_exposed(self, attribute: str) -> None:
        assert not hasattr(self.frame, attribute)

    def test_plot_rejects_spectral_matrix_strategy(self) -> None:
        with pytest.raises(ValueError, match=r"supports only plot_type='raw'"):
            self.frame.plot(plot_type="matrix")

    def test_plot_single_channel_uses_full_coefficient_vector(self) -> None:
        import matplotlib.pyplot as plt

        frame = self.frame[0]
        ax = cast(Any, frame.plot())

        np.testing.assert_allclose(np.asarray(ax.lines[0].get_xdata()), frame.quefrencies)
        np.testing.assert_allclose(np.asarray(ax.lines[0].get_ydata()), np.zeros(self.n_fft))
        plt.close("all")

    def test_dataframe_uses_quefrency_index(self) -> None:
        result = self.frame.to_dataframe()

        assert result.index.name == "quefrency"
        np.testing.assert_allclose(result.index.to_numpy(), self.frame.quefrencies)

    def test_info_reports_quefrency_domain(self, capsys: pytest.CaptureFixture[str]) -> None:
        self.frame.info()

        output = capsys.readouterr().out
        assert "CepstralFrame Information:" in output
        assert "Quefrency bins: 1024" in output

    def test_xarray_uses_quefrency_dimension(self) -> None:
        result = self.frame.to_xarray()

        assert result.dims == ("channel", "quefrency")

    def test_spectral_envelope_rejects_sliced_quefrency_axis(self) -> None:
        sliced = self.frame[:, 4:8]

        with pytest.raises(ValueError, match=r"requires a complete, unsliced quefrency axis"):
            sliced.to_spectral_envelope()

    def test_binary_operations_reject_spectral_domain_mixing(self) -> None:
        spectral = SpectralFrame(
            data=da.ones((2, self.n_fft), chunks=(1, -1), dtype=np.complex128),
            sampling_rate=self.sampling_rate,
            n_fft=(self.n_fft - 1) * 2,
        )

        with pytest.raises(TypeError):
            _ = spectral + self.frame
        with pytest.raises(TypeError, match=r"require another CepstralFrame"):
            _ = self.frame + spectral

    def test_binary_operations_reject_cepstrum_as_right_hand_time_domain_operand(self) -> None:
        time_frame = ChannelFrame.from_numpy(np.ones((2, self.n_fft)), self.sampling_rate)

        with pytest.raises(TypeError, match=r"matching frame types"):
            _ = time_frame + self.frame

    @pytest.mark.parametrize(
        ("window", "analysis_length"),
        [("boxcar", 1024), ("hann", 512)],
    )
    def test_binary_operations_require_matching_analysis_metadata(self, window: str, analysis_length: int) -> None:
        other = CepstralFrame(
            data=self.data,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            window=window,
            analysis_length=analysis_length,
        )

        with pytest.raises(ValueError, match=r"analysis metadata must match"):
            _ = self.frame + other

    def test_binary_operations_require_matching_quefrency_coordinates(self) -> None:
        left = self.frame[:, :4]
        right = self.frame[:, 4:8]

        with pytest.raises(ValueError, match=r"coordinates must match"):
            _ = left + right

    def test_scalar_binary_operations_remain_supported(self) -> None:
        added = self.frame + 1.0
        multiplied = 2.0 * self.frame

        np.testing.assert_allclose(added.compute(), np.ones(self.frame.shape))
        np.testing.assert_allclose(multiplied.compute(), np.zeros(self.frame.shape))

    def test_sampling_rate_is_immutable(self) -> None:
        expected = self.frame.quefrencies

        with pytest.raises(AttributeError, match=r"sampling_rate is immutable"):
            self.frame.sampling_rate = self.sampling_rate / 2

        assert self.frame.sampling_rate == self.sampling_rate
        np.testing.assert_allclose(self.frame.quefrencies, expected)

    @pytest.mark.parametrize(
        "operand",
        [
            1j,
            np.ones(1024, dtype=np.complex128),
            da.ones(1024, chunks=-1, dtype=np.complex128),
        ],
    )
    def test_binary_operations_reject_complex_operands(self, operand: object) -> None:
        with pytest.raises(TypeError, match=r"real-valued operands"):
            _ = self.frame + cast(Any, operand)

    @pytest.mark.parametrize(
        "operation_name",
        ["cepstrum", "lifter", "spectral_envelope", "ifft", "normalize"],
    )
    def test_generic_operations_outside_cepstral_contract_are_rejected(self, operation_name: str) -> None:
        with pytest.raises(ValueError, match=r"not available through CepstralFrame.apply_operation"):
            self.frame.apply_operation(operation_name)
