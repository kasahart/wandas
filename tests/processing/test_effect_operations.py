import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation
from wandas.processing.effects import (
    AddWithSNR,
    HpssHarmonic,
    HpssPercussive,
    Normalize,
)
from wandas.utils.dask_helpers import da_from_array

_SR = 16000


class TestHpssHarmonic:
    """HPSS harmonic extraction: Layer 1 (unit) + Layer 2 (domain) + Layer 3."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_hpss_harmonic_init_stores_params(self) -> None:
        """Test HpssHarmonic stores sampling rate and custom margin."""
        hpss = HpssHarmonic(_SR)
        assert hpss.sampling_rate == _SR

        hpss_custom = HpssHarmonic(_SR, margin=2.0)
        assert hpss_custom.kwargs.get("margin") == 2.0

    def test_hpss_harmonic_registry_returns_correct_class(self) -> None:
        """Test HpssHarmonic is registered as 'hpss_harmonic'."""
        assert get_operation("hpss_harmonic") == HpssHarmonic
        hpss_op = create_operation("hpss_harmonic", _SR, margin=3.0)
        assert isinstance(hpss_op, HpssHarmonic)
        assert hpss_op.kwargs.get("margin") == 3.0

    # -- Layer 2: Domain (shape + immutability + spectral flatness) ---------

    def test_hpss_harmonic_preserves_shape_and_immutability(
        self, mixed_harmonic_percussive_dask: tuple[DaArray, int]
    ) -> None:
        """Shape preserved; input unchanged; Dask graph maintained."""
        dask_input, sr = mixed_harmonic_percussive_dask
        hpss = HpssHarmonic(sr)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = hpss.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Shape
        assert result_da.compute().shape == input_copy.shape

    def test_hpss_harmonic_reduces_spectral_flatness(self, mixed_harmonic_percussive_dask: tuple[DaArray, int]) -> None:
        """Harmonic extraction should produce a less flat (more peaked) spectrum.

        Spectral flatness = geometric_mean(S) / arithmetic_mean(S).
        Lower flatness = more harmonic content.
        """
        dask_input, sr = mixed_harmonic_percussive_dask
        hpss = HpssHarmonic(sr)
        raw = dask_input.compute()
        result = hpss.process(dask_input).compute()

        n_fft = 2048
        orig_spec = np.abs(np.fft.rfft(raw[0], n_fft))
        result_spec = np.abs(np.fft.rfft(result[0], n_fft))

        # Spectral flatness (Wiener entropy) — lower = more harmonic
        orig_flatness = np.exp(np.mean(np.log(orig_spec + 1e-10))) / np.mean(orig_spec)
        result_flatness = np.exp(np.mean(np.log(result_spec + 1e-10))) / np.mean(result_spec)

        assert result_flatness < orig_flatness, "Harmonic extraction must reduce spectral flatness"


class TestHpssPercussive:
    """HPSS percussive extraction: Layer 1 (unit) + Layer 2 (domain)."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_hpss_percussive_init_stores_params(self) -> None:
        """Test HpssPercussive stores sampling rate and custom margin."""
        hpss = HpssPercussive(_SR)
        assert hpss.sampling_rate == _SR

        hpss_custom = HpssPercussive(_SR, margin=2.0)
        assert hpss_custom.kwargs.get("margin") == 2.0

    def test_hpss_percussive_registry_returns_correct_class(self) -> None:
        """Test HpssPercussive is registered as 'hpss_percussive'."""
        assert get_operation("hpss_percussive") == HpssPercussive
        hpss_op = create_operation("hpss_percussive", _SR, margin=3.0)
        assert isinstance(hpss_op, HpssPercussive)
        assert hpss_op.kwargs.get("margin") == 3.0

    # -- Layer 2: Domain (shape + immutability + spectral flatness) ---------

    def test_hpss_percussive_preserves_shape_and_immutability(
        self, mixed_harmonic_percussive_dask: tuple[DaArray, int]
    ) -> None:
        """Shape preserved; input unchanged; Dask graph maintained."""
        dask_input, sr = mixed_harmonic_percussive_dask
        hpss = HpssPercussive(sr)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = hpss.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Shape
        assert result_da.compute().shape == input_copy.shape

    def test_hpss_percussive_increases_spectral_flatness(
        self, mixed_harmonic_percussive_dask: tuple[DaArray, int]
    ) -> None:
        """Percussive extraction should produce a flatter (more noise-like) spectrum.

        Higher flatness = less harmonic (more broadband/percussive) content.
        """
        dask_input, sr = mixed_harmonic_percussive_dask
        hpss = HpssPercussive(sr)
        raw = dask_input.compute()
        result = hpss.process(dask_input).compute()

        n_fft = 2048
        orig_spec = np.abs(np.fft.rfft(raw[0], n_fft))
        result_spec = np.abs(np.fft.rfft(result[0], n_fft))

        orig_flatness = np.exp(np.mean(np.log(orig_spec + 1e-10))) / np.mean(orig_spec)
        result_flatness = np.exp(np.mean(np.log(result_spec + 1e-10))) / np.mean(result_spec)

        assert result_flatness > orig_flatness, "Percussive extraction must increase spectral flatness"


class TestAddWithSNR:
    """AddWithSNR: Layer 1 (unit) + Layer 2 (domain) + Layer 3 (SNR verification)."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_add_with_snr_init_stores_params(self) -> None:
        """Test AddWithSNR stores sampling rate and SNR."""
        np.random.seed(42)
        noise = da_from_array(np.random.randn(1, _SR), chunks=(1, -1))
        op = AddWithSNR(_SR, noise, 10.0)
        assert op.sampling_rate == _SR
        assert op.snr == 10.0

    def test_add_with_snr_registry_returns_correct_class(self) -> None:
        """Test AddWithSNR is registered as 'add_with_snr'."""
        assert get_operation("add_with_snr") == AddWithSNR

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_add_with_snr_preserves_shape_and_immutability(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Shape preserved; input unchanged after SNR addition."""
        dask_input, sr = pure_sine_440hz_dask
        np.random.seed(42)  # Reproducible noise
        noise = da_from_array(np.random.randn(1, sr), chunks=(1, -1))
        op = AddWithSNR(sr, noise, 10.0)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = op.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Shape
        result = result_da.compute()
        assert result.shape == input_copy.shape

    # -- Layer 3: Integration (SNR verification) ---------------------------

    def test_add_with_snr_actual_snr_matches_target(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Actual SNR of output must match target within 10% (rtol=0.1).

        Tolerance rationale: finite-length signals cause small power estimation errors.
        """
        dask_input, sr = pure_sine_440hz_dask
        target_snr = 10.0
        np.random.seed(42)
        noise = da_from_array(np.random.randn(1, sr), chunks=(1, -1))
        op = AddWithSNR(sr, noise, target_snr)

        clean = dask_input.compute()
        result = op.process(dask_input).compute()

        from wandas.utils import util

        clean_power = util.calculate_rms(clean) ** 2
        noise_component = result - clean
        noise_power = util.calculate_rms(noise_component) ** 2
        actual_snr = 10 * np.log10(clean_power / noise_power)

        np.testing.assert_allclose(
            actual_snr,
            target_snr,
            rtol=0.1,  # 10% tolerance for finite-length signal power estimation
        )

    def test_add_with_snr_stereo_preserves_shape(self) -> None:
        """Stereo clean + stereo noise preserves (2, N) shape."""
        t = np.linspace(0, 1, _SR, endpoint=False)
        clean = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)])
        np.random.seed(42)
        noise = np.random.randn(2, _SR)

        dask_clean = da_from_array(clean, chunks=(1, -1))
        dask_noise = da_from_array(noise, chunks=(1, -1))
        op = AddWithSNR(_SR, dask_noise, 10.0)

        result = op.process(dask_clean).compute()
        assert result.shape == clean.shape


class TestNormalize:
    """Normalize operation: Layer 1 + Layer 2 + Layer 3 (theoretical values)."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_normalize_init_default_params(self) -> None:
        """Test Normalize default norm=inf, axis=-1."""
        n = Normalize(_SR)
        assert n.sampling_rate == _SR
        assert n.norm == np.inf
        assert n.axis == -1

    def test_normalize_init_custom_params(self) -> None:
        """Test Normalize stores custom norm and axis."""
        n = Normalize(_SR, norm=2, axis=0)
        assert n.norm == 2
        assert n.axis == 0

    def test_normalize_registry_returns_correct_class(self) -> None:
        """Test Normalize is registered as 'normalize'."""
        assert get_operation("normalize") == Normalize
        n = create_operation("normalize", _SR, norm=2, axis=0)
        assert isinstance(n, Normalize)
        assert n.norm == 2
        assert n.axis == 0

    def test_normalize_invalid_norm_raises_error(self) -> None:
        """Test that invalid norm type provides WHAT/WHY/HOW error."""
        with pytest.raises(ValueError) as exc_info:
            Normalize(sampling_rate=44100, norm="invalid")  # ty: ignore[invalid-argument-type]

        error_msg = str(exc_info.value)
        assert "Invalid normalization method" in error_msg
        assert "float, int, np.inf" in error_msg
        assert "Common values:" in error_msg

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_normalize_preserves_shape_and_immutability(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Shape preserved; input unchanged after normalization."""
        dask_input, sr = pure_sine_440hz_dask
        normalize = Normalize(sr)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = normalize.process(dask_input)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Shape
        assert result_da.compute().shape == input_copy.shape

    # -- Layer 3: Theoretical value verification ---------------------------

    def test_normalize_inf_norm_max_abs_equals_one(self) -> None:
        """With norm=inf, max|x| must equal 1.0 (theoretical).

        Tolerance: rtol=1e-10 — float64 arithmetic precision.
        """
        # Signal with known max of 2.0
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig = (2.0 * np.sin(2 * np.pi * 440 * t)).reshape(1, -1)
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=-1)

        result = normalize.process(dask_sig).compute()
        max_val = np.max(np.abs(result))
        np.testing.assert_allclose(
            max_val,
            1.0,
            rtol=1e-10,  # float64 precision for simple division
        )

    def test_normalize_l1_norm_sum_abs_equals_one(self) -> None:
        """With norm=1, sum|x| must equal 1.0 (theoretical).

        Tolerance: rtol=1e-10 — float64 summation precision.
        """
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig = (2.0 * np.sin(2 * np.pi * 440 * t)).reshape(1, -1)
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=1, axis=-1)

        result = normalize.process(dask_sig).compute()
        l1 = np.sum(np.abs(result), axis=-1)
        np.testing.assert_allclose(
            l1,
            1.0,
            rtol=1e-10,  # float64 summation precision
        )

    def test_normalize_l2_norm_euclidean_equals_one(self) -> None:
        """With norm=2, ||x||_2 must equal 1.0 (theoretical).

        Tolerance: rtol=1e-10 — float64 precision.
        """
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig = (2.0 * np.sin(2 * np.pi * 440 * t)).reshape(1, -1)
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=2, axis=-1)

        result = normalize.process(dask_sig).compute()
        l2 = np.sqrt(np.sum(result**2, axis=-1))
        np.testing.assert_allclose(
            l2,
            1.0,
            rtol=1e-10,  # float64 precision
        )

    def test_normalize_multichannel_independent_inf_norm(self) -> None:
        """Each channel independently normalized: max|ch_i| == 1.0."""
        t = np.linspace(0, 1, _SR, endpoint=False)
        multi = np.stack(
            [
                2.0 * np.sin(2 * np.pi * 440 * t),  # max=2.0
                3.0 * np.sin(2 * np.pi * 880 * t),  # max=3.0
            ]
        )
        dask_multi = da_from_array(multi, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=-1)

        result = normalize.process(dask_multi).compute()
        for ch in range(result.shape[0]):
            np.testing.assert_allclose(
                np.max(np.abs(result[ch])),
                1.0,
                rtol=1e-10,  # float64 precision per channel
            )

    def test_normalize_multichannel_global_preserves_ratio(self) -> None:
        """Global normalization preserves inter-channel amplitude ratio."""
        t = np.linspace(0, 1, _SR, endpoint=False)
        multi = np.stack(
            [
                2.0 * np.sin(2 * np.pi * 440 * t),
                3.0 * np.sin(2 * np.pi * 880 * t),
            ]
        )
        dask_multi = da_from_array(multi, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=None)

        result = normalize.process(dask_multi).compute()

        # Global max should be 1.0
        np.testing.assert_allclose(
            np.max(np.abs(result)),
            1.0,
            rtol=1e-10,
        )

        # Inter-channel ratio preserved
        orig_ratio = np.max(np.abs(multi[0])) / np.max(np.abs(multi[1]))
        result_ratio = np.max(np.abs(result[0])) / np.max(np.abs(result[1]))
        np.testing.assert_allclose(
            orig_ratio,
            result_ratio,
            rtol=1e-10,  # float64 division precision
        )

    def test_normalize_zero_signal_remains_zero(self) -> None:
        """Zero signal normalized with norm=inf stays zero."""
        zero = np.zeros((1, _SR))
        dask_zero = da_from_array(zero, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=-1)

        result = normalize.process(dask_zero).compute()
        np.testing.assert_allclose(result, 0.0)

    def test_normalize_threshold_prevents_amplification(self) -> None:
        """Signal below threshold is not normalized to 1.0."""
        small = np.full((1, _SR), 1e-12)
        dask_small = da_from_array(small, chunks=(1, -1))

        # threshold=1e-10 means signals with max < 1e-10 are left as-is
        normalize = Normalize(_SR, norm=np.inf, axis=-1, threshold=1e-10)
        result = normalize.process(dask_small).compute()
        assert np.max(np.abs(result)) < 1.0

    def test_normalize_fill_true_makes_zero_nonzero(self) -> None:
        """With fill=True, zero vector is filled so max|x| == 1.0."""
        zero = np.zeros((1, _SR))
        dask_zero = da_from_array(zero, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=-1, fill=True)

        result = normalize.process(dask_zero).compute()
        assert result.shape == zero.shape
        assert not np.allclose(result, 0.0)
        np.testing.assert_allclose(
            np.max(np.abs(result)),
            1.0,
            rtol=1e-10,  # fill produces exact normalized values
        )

    def test_negative_norm_error_message(self) -> None:
        """Test that negative norm (except -np.inf) provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            Normalize(sampling_rate=44100, norm=-2)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid normalization method" in error_msg
        assert "-2" in error_msg
        # Check WHY
        assert "Non-negative value" in error_msg
        # Check HOW
        assert "Common values:" in error_msg

    def test_negative_threshold_error_message(self) -> None:
        """Test that negative threshold provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            Normalize(sampling_rate=44100, threshold=-0.5)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid threshold for normalization" in error_msg
        assert "-0.5" in error_msg
        # Check WHY
        assert "Non-negative value" in error_msg
        # Check HOW
        assert "Typical values:" in error_msg
