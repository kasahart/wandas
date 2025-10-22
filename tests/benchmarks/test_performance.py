"""Performance benchmark tests for wandas operations.

This module contains benchmark tests for key signal processing operations
to monitor performance and detect regressions.
"""

import pytest

from wandas.frames.channel import ChannelFrame


class TestFFTPerformance:
    """Benchmark tests for FFT operation."""

    def test_fft_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark FFT computation with default parameters.

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(benchmark_signal.fft, n_fft=2048)
        assert result is not None
        assert result.n_channels == benchmark_signal.n_channels

    def test_fft_large_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark FFT computation with larger FFT size.

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(benchmark_signal.fft, n_fft=4096)
        assert result is not None
        assert result.n_channels == benchmark_signal.n_channels


class TestSTFTPerformance:
    """Benchmark tests for STFT operation."""

    def test_stft_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark STFT computation with default parameters.

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(benchmark_signal.stft, n_fft=2048, hop_length=512)
        assert result is not None
        assert result.n_channels == benchmark_signal.n_channels

    def test_stft_small_hop_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark STFT with smaller hop length (more frames).

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(benchmark_signal.stft, n_fft=2048, hop_length=256)
        assert result is not None
        assert result.n_channels == benchmark_signal.n_channels


class TestWelchPerformance:
    """Benchmark tests for Welch method."""

    def test_welch_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark Welch method with default parameters.

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(
            benchmark_signal.welch, n_fft=2048, win_length=2048, hop_length=512
        )
        assert result is not None
        assert result.n_channels == benchmark_signal.n_channels

    def test_welch_large_window_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark Welch method with larger window.

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(
            benchmark_signal.welch, n_fft=4096, win_length=4096, hop_length=1024
        )
        assert result is not None
        assert result.n_channels == benchmark_signal.n_channels


class TestFilterPerformance:
    """Benchmark tests for filter operations."""

    def test_lowpass_filter_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark low-pass filter application.

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(benchmark_signal.low_pass_filter, cutoff=1000, order=4)
        assert result is not None
        assert result.n_channels == benchmark_signal.n_channels

    def test_highpass_filter_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark high-pass filter application.

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(benchmark_signal.high_pass_filter, cutoff=100, order=4)
        assert result is not None
        assert result.n_channels == benchmark_signal.n_channels

    def test_lowpass_high_order_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark low-pass filter with higher order.

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(benchmark_signal.low_pass_filter, cutoff=1000, order=8)
        assert result is not None
        assert result.n_channels == benchmark_signal.n_channels

    def test_highpass_high_order_performance(
        self, benchmark: pytest.fixture, benchmark_signal: ChannelFrame  # type: ignore [valid-type]
    ) -> None:
        """Benchmark high-pass filter with higher order.

        Args:
            benchmark: pytest-benchmark fixture
            benchmark_signal: Sample signal for benchmarking
        """
        result = benchmark(benchmark_signal.high_pass_filter, cutoff=100, order=8)
        assert result is not None
        assert result.n_channels == benchmark_signal.n_channels
