"""Tests for channel-wise chunking policy in ChannelFrame."""

import dask.array as da
import numpy as np

from wandas.frames.channel import ChannelFrame


class TestChannelWiseChunking:
    """Test that channel-wise chunking (1, -1) is enforced consistently."""

    def test_from_numpy_mono_channel_chunks(self):
        """Test that from_numpy creates (1, -1) chunks for mono signal."""
        data = np.random.rand(44100)
        cf = ChannelFrame.from_numpy(data, sampling_rate=44100)

        # Channel axis should be chunked as 1
        assert cf._data.chunks[0] == (1,), (
            f"Expected channel chunks (1,), got {cf._data.chunks[0]}"
        )
        # Sample axis should be unchunked (-1 becomes full length)
        assert cf._data.chunks[1] == (44100,), (
            f"Expected sample chunks (44100,), got {cf._data.chunks[1]}"
        )

    def test_from_numpy_stereo_channel_chunks(self):
        """Test that from_numpy creates (1, 1, ...) chunks for stereo signal."""
        data = np.random.rand(2, 44100)
        cf = ChannelFrame.from_numpy(data, sampling_rate=44100)

        # Each channel should be chunked separately
        assert cf._data.chunks[0] == (1, 1), (
            f"Expected channel chunks (1, 1), got {cf._data.chunks[0]}"
        )
        # Sample axis should be unchunked
        assert cf._data.chunks[1] == (44100,), (
            f"Expected sample chunks (44100,), got {cf._data.chunks[1]}"
        )

    def test_from_numpy_multichannel_chunks(self):
        """Test channel-wise chunking for multi-channel signals."""
        n_channels = 4
        n_samples = 48000
        data = np.random.rand(n_channels, n_samples)
        cf = ChannelFrame.from_numpy(data, sampling_rate=48000)

        # Each channel should be chunked as 1
        expected_channel_chunks = tuple([1] * n_channels)
        assert cf._data.chunks[0] == expected_channel_chunks, (
            f"Expected channel chunks {expected_channel_chunks}, "
            f"got {cf._data.chunks[0]}"
        )
        # Sample axis unchunked
        assert cf._data.chunks[1] == (n_samples,), (
            f"Expected sample chunks ({n_samples},), got {cf._data.chunks[1]}"
        )

    def test_add_channel_preserves_chunking(self):
        """Test that add_channel maintains channel-wise chunking."""
        cf = ChannelFrame.from_numpy(np.random.rand(2, 44100), sampling_rate=44100)
        new_channel_data = np.random.rand(44100)

        cf_with_added = cf.add_channel(new_channel_data, label="new_ch")

        # Should now have 3 channels, each chunked as 1
        assert cf_with_added._data.chunks[0] == (1, 1, 1), (
            f"Expected (1, 1, 1) after add_channel, got {cf_with_added._data.chunks[0]}"
        )
        assert cf_with_added._data.chunks[1] == (44100,), (
            f"Expected sample chunks (44100,), got {cf_with_added._data.chunks[1]}"
        )

    def test_from_file_lazy_chunks(self, tmp_path):
        """Test that from_file uses channel-wise chunks for lazy loading."""
        # Create a temporary WAV file
        import soundfile as sf

        audio_path = tmp_path / "test_audio.wav"
        data = np.random.rand(2, 44100).astype(np.float32)
        sf.write(str(audio_path), data.T, 44100)

        # Load with from_file
        cf = ChannelFrame.from_file(audio_path)

        # Should have channel-wise chunking before compute
        assert cf._data.chunks[0] == (1, 1), (
            f"Expected channel chunks (1, 1) from lazy load, got {cf._data.chunks[0]}"
        )
        assert cf._data.chunks[1] == (44100,), (
            f"Expected sample chunks (44100,), got {cf._data.chunks[1]}"
        )

    def test_rechunk_custom_sample_axis(self):
        """Test BaseFrame auto-rechunking behavior.

        BaseFrame.__init__ automatically rechunks data to channel-wise format,
        which overrides custom chunking. This is the designed behavior to ensure
        consistent channel-wise parallelism across all frames.
        """
        cf = ChannelFrame.from_numpy(np.random.rand(2, 44100), sampling_rate=44100)

        # Try to rechunk sample axis to smaller chunks
        rechunked_data = cf._data.rechunk((1, 16384))  # type: ignore

        # Verify the rechunking was applied before passing to _create_new_instance
        assert len(rechunked_data.chunks[1]) > 1, (
            "rechunked_data should have multiple sample chunks"
        )

        # Create new instance - BaseFrame will re-apply channel-wise chunking
        cf_rechunked = cf._create_new_instance(data=rechunked_data)

        # BaseFrame.__init__ re-chunks to (1, -1), so we get back to single chunk
        # This is expected behavior - BaseFrame enforces channel-wise chunking
        assert cf_rechunked._data.chunks[0] == (1, 1)
        # After BaseFrame rechunking, sample axis is back to single chunk
        assert cf_rechunked._data.chunks[1] == (44100,), (
            f"Expected sample chunks (44100,) after BaseFrame rechunk, "
            f"got {cf_rechunked._data.chunks[1]}"
        )

    def test_binary_operations_preserve_chunking(self):
        """Test that binary operations maintain channel-wise chunking."""
        cf1 = ChannelFrame.from_numpy(np.random.rand(2, 44100), sampling_rate=44100)
        cf2 = ChannelFrame.from_numpy(np.random.rand(2, 44100), sampling_rate=44100)

        cf_result = cf1 + cf2

        # Result should maintain channel-wise chunking
        assert cf_result._data.chunks[0] == (1, 1), (
            f"Expected (1, 1) after binary op, got {cf_result._data.chunks[0]}"
        )
        assert cf_result._data.chunks[1] == (44100,), (
            f"Expected sample chunks (44100,), got {cf_result._data.chunks[1]}"
        )


class TestBaseFrameRechunking:
    """Test rechunking logic in BaseFrame initialization."""

    def test_1d_array_rechunking(self) -> None:
        """Test rechunking of 1D dask arrays."""
        # Create a 1D dask array
        data_1d = np.random.random(1000)
        dask_1d = da.from_array(data_1d, chunks=100)

        # Create a ChannelFrame (which uses BaseFrame)
        # This should trigger the 1D rechunking path (line 90)
        frame = ChannelFrame(data=dask_1d, sampling_rate=16000)

        # Verify the frame was created successfully
        assert frame.n_channels == 1
        assert frame.n_samples == 1000
        # After reshaping, it should be 2D
        assert frame._data.ndim == 2

    def test_rechunking_exception_fallback(self) -> None:
        """Test that rechunking exceptions are handled gracefully."""
        # Create a normal array
        data = np.random.random((2, 1000))
        dask_data = da.from_array(data, chunks=(1, 500))

        # This test is checking if the frame can be created normally
        # The exception handling path (lines 99-102) is difficult to trigger
        # in practice but exists as a safety net
        frame = ChannelFrame(data=dask_data, sampling_rate=16000)

        # Verify the frame was created successfully
        assert frame.n_channels == 2
        assert frame.n_samples == 1000
