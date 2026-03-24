import dask.array as da
import numpy as np

from wandas.core.metadata import ChannelMetadata
from wandas.frames.cepstral import CepstralFrame
from wandas.frames.spectral import SpectralFrame

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestCepstralFrame:
    def setup_method(self) -> None:
        self.sampling_rate = 16000
        self.n_fft = 1024
        self.data = _da_from_array(np.zeros((2, self.n_fft), dtype=np.float64), chunks=(1, -1))
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

    def test_quefrencies_property_matches_sampling_rate(self) -> None:
        expected = np.arange(self.n_fft) / self.sampling_rate
        np.testing.assert_allclose(self.frame.quefrencies, expected)

    def test_lifter_returns_new_frame_with_history_and_labels(self) -> None:
        result = self.frame.lifter(cutoff=0.001, mode="low")

        assert isinstance(result, CepstralFrame)
        assert result is not self.frame
        assert isinstance(result._data, da.Array)
        assert result.operation_history[-1] == {"operation": "lifter", "params": {"cutoff": 0.001, "mode": "low"}}
        assert result.labels == ["lifter(ch0)", "lifter(ch1)"]
        np.testing.assert_allclose(self.frame.compute(), np.zeros((2, self.n_fft)))

    def test_to_spectral_envelope_returns_spectral_frame(self) -> None:
        result = self.frame.to_spectral_envelope()

        assert isinstance(result, SpectralFrame)
        assert result.n_fft == self.n_fft
        assert result.operation_history[-1] == {"operation": "spectral_envelope", "params": {"n_fft": self.n_fft}}
        np.testing.assert_allclose(result.compute().real, np.ones((2, self.n_fft // 2 + 1)))
