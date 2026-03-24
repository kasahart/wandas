import dask.array as da
import numpy as np
import pytest

from wandas.core.metadata import ChannelMetadata, FrameMetadata
from wandas.frames.cepstral import CepstralFrame
from wandas.frames.channel import ChannelFrame
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
        np.testing.assert_allclose(self.frame.freqs, expected)

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

    def test_cepstrum_lifter_and_envelope_preserve_frame_metadata_source_file(self) -> None:
        frame = ChannelFrame.from_numpy(
            np.ones((1, 1024), dtype=np.float64),
            self.sampling_rate,
            metadata=FrameMetadata({"tag": "voice"}, source_file="speech.wav"),
        )

        cepstrum = frame.cepstrum(n_fft=1024, window="boxcar")
        low_ceps = cepstrum.lifter(cutoff=0.001, mode="low")
        envelope = low_ceps.to_spectral_envelope()

        assert isinstance(cepstrum.metadata, FrameMetadata)
        assert isinstance(low_ceps.metadata, FrameMetadata)
        assert isinstance(envelope.metadata, FrameMetadata)
        assert cepstrum.metadata.source_file == "speech.wav"
        assert low_ceps.metadata.source_file == "speech.wav"
        assert envelope.metadata.source_file == "speech.wav"

    def test_ifft_on_cepstral_frame_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match=r"IFFT is not supported for cepstral data"):
            self.frame.ifft()
