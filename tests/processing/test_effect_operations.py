from typing import Any

import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation
from wandas.processing.effects import (
    AddWithSNR,
    Fade,
    HpssHarmonic,
    HpssPercussive,
    Normalize,
    RemoveDC,
    _normalize_array,
)
from wandas.utils import util
from wandas.utils.dask_helpers import da_from_array

_SR: int = 16000


def _as_dask(data: Any) -> DaArray:
    if isinstance(data, DaArray):
        return data
    chunks = (1, *(-1,) * (np.ndim(data) - 1)) if np.ndim(data) > 1 else (-1,)
    return da_from_array(data, chunks=chunks)


def _compute_process(operation: Any, data: Any, *inputs: Any) -> Any:
    return operation.process(_as_dask(data), *(_as_dask(input_data) for input_data in inputs)).compute()


def _assert_lazy_metadata_matches_compute(operation: Any, data: Any, *inputs: Any) -> None:
    result_da = operation.process(_as_dask(data), *(_as_dask(input_data) for input_data in inputs))
    result = result_da.compute()

    assert result_da.shape == result.shape
    assert result_da.dtype == result.dtype


class TestHpssHarmonic:
    """HPSS harmonic extraction: Layer 1 (unit) + Layer 2 (domain) + Layer 3."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_hpss_harmonic_init_stores_params(self) -> None:
        """Test HpssHarmonic stores sampling rate and custom margin."""
        hpss = HpssHarmonic(_SR)
        assert hpss.sampling_rate == _SR

        hpss_custom = HpssHarmonic(_SR, margin=2.0)
        assert hpss_custom.kwargs.get("margin") == 2.0

    def test_hpss_harmonic_kwargs_are_defensive_copies(self) -> None:
        """Mutating exposed HPSS kwargs must not change operation config."""
        hpss = HpssHarmonic(_SR, margin=2.0)

        hpss.kwargs["margin"] = 8.0

        assert hpss.kwargs["margin"] == 2.0

    def test_hpss_harmonic_empty_kwargs_are_defensive_copies(self) -> None:
        hpss = HpssHarmonic(_SR)

        hpss.kwargs["margin"] = 8.0

        assert hpss.kwargs == {}
        assert object.__getattribute__(hpss, "kwargs") == {}

    def test_hpss_harmonic_empty_kwargs_reassignment_is_blocked(self) -> None:
        hpss = HpssHarmonic(_SR)

        with pytest.raises(AttributeError):
            setattr(hpss, "kwargs", {"margin": 8.0})

        assert hpss.kwargs == {}
        assert object.__getattribute__(hpss, "kwargs") == {}

    def test_hpss_harmonic_kwargs_reassignment_to_non_mapping_is_blocked(self) -> None:
        hpss = HpssHarmonic(_SR, margin=2.0)

        with pytest.raises(AttributeError):
            setattr(hpss, "kwargs", None)

        assert hpss.kwargs == {"margin": 2.0}
        assert object.__getattribute__(hpss, "kwargs") == {"margin": 2.0}

    def test_hpss_harmonic_kwargs_snapshot_caller_owned_mutable_values(self) -> None:
        """Grouped kwargs should not retain caller-owned mutable values."""
        margin = [2.0, 3.0]
        hpss = HpssHarmonic(_SR, margin=margin)

        margin[0] = 8.0

        assert hpss.kwargs["margin"] == [2.0, 3.0]

    def test_hpss_harmonic_to_params_returns_defensive_snapshot(self) -> None:
        margin = [2.0, 3.0]
        hpss = HpssHarmonic(_SR, margin=margin)

        params = hpss.to_params()
        params["margin"][0] = 8.0

        assert hpss.to_params()["margin"] == [2.0, 3.0]

    def test_hpss_harmonic_params_and_kwargs_share_defensive_base_config(self) -> None:
        hpss = HpssHarmonic(_SR, margin=[2.0, 3.0], kernel_size={"harmonic": 31})

        hpss.params["margin"][0] = 8.0
        hpss.to_params()["kernel_size"]["harmonic"] = 99
        hpss.kwargs["margin"][1] = 9.0

        assert hpss.params["margin"] == [2.0, 3.0]
        assert hpss.to_params()["kernel_size"] == {"harmonic": 31}
        assert hpss.kwargs == {"margin": [2.0, 3.0], "kernel_size": {"harmonic": 31}}

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
        result = _compute_process(hpss, dask_input)

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
        result = _compute_process(hpss, dask_input)

        n_fft = 2048
        orig_spec = np.abs(np.fft.rfft(raw[0], n_fft))
        result_spec = np.abs(np.fft.rfft(result[0], n_fft))

        orig_flatness = np.exp(np.mean(np.log(orig_spec + 1e-10))) / np.mean(orig_spec)
        result_flatness = np.exp(np.mean(np.log(result_spec + 1e-10))) / np.mean(result_spec)

        assert result_flatness > orig_flatness, "Percussive extraction must increase spectral flatness"


def test_normalize_stores_all_lineage_parameters() -> None:
    normalize = Normalize(_SR, norm=2.0, axis=1, threshold=0.01, fill=False)

    assert normalize.norm == 2.0
    assert normalize.axis == 1
    assert normalize.threshold == 0.01
    assert normalize.fill is False
    assert normalize.to_params() == {"norm": 2.0, "axis": 1, "threshold": 0.01, "fill": False}


def test_add_with_snr_and_fade_expose_lineage_parameters() -> None:
    add = AddWithSNR(_SR, snr=12.0)
    fade = Fade(_SR, fade_ms=25)

    assert add.snr == 12.0
    assert add.to_params() == {"snr": 12.0}
    assert fade.fade_ms == 25.0
    assert fade.to_params() == {"fade_ms": 25.0}


class TestAddWithSNR:
    """AddWithSNR: Layer 1 (unit) + Layer 2 (domain) + Layer 3 (SNR verification)."""

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_add_with_snr_init_stores_params(self) -> None:
        """Test AddWithSNR stores sampling rate and SNR."""
        op = AddWithSNR(_SR, 10.0)
        assert op.sampling_rate == _SR
        assert op.snr == 10.0
        assert op.params == {"snr": 10.0}
        assert not any(isinstance(value, DaArray) for value in op._config.values())

    def test_add_with_snr_registry_returns_correct_class(self) -> None:
        """Test AddWithSNR is registered as 'add_with_snr'."""
        assert get_operation("add_with_snr") == AddWithSNR

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_add_with_snr_preserves_shape_and_immutability(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Shape preserved; input unchanged after SNR addition."""
        dask_input, sr = pure_sine_440hz_dask
        rng = np.random.default_rng(42)
        noise = da_from_array(rng.standard_normal((1, sr)), chunks=(1, -1))
        op = AddWithSNR(sr, 10.0)
        input_copy = dask_input.compute().copy()

        # Act
        result_da = op.process(dask_input, noise)

        # Assert 1: Immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)

        # Assert 2: Dask graph preserved
        assert isinstance(result_da, DaArray)

        # Assert 3: Shape
        result = result_da.compute()
        assert result.shape == input_copy.shape

    def test_add_with_snr_int16_clean_float32_noise_uses_float32_dtype(self) -> None:
        """Integer clean data is promoted to at least float32 for SNR math."""
        clean = da_from_array(np.array([[1000, -1000, 500, -500]], dtype=np.int16), chunks=(1, -1))
        noise = da_from_array(np.array([[0.5, -0.25, 0.125, -0.5]], dtype=np.float32), chunks=(1, -1))
        op = AddWithSNR(_SR, 10.0)

        result_da = op.process(clean, noise)

        assert isinstance(result_da, DaArray)
        assert result_da.dtype == np.float32
        assert result_da.compute().dtype == np.float32

    def test_add_with_snr_float32_clean_float64_noise_uses_float64_dtype(self) -> None:
        """Float64 input preserves float64 precision in SNR math."""
        clean = da_from_array(np.array([[1.0, -1.0, 0.5, -0.5]], dtype=np.float32), chunks=(1, -1))
        noise = da_from_array(np.array([[0.5, -0.25, 0.125, -0.5]], dtype=np.float64), chunks=(1, -1))
        op = AddWithSNR(_SR, 10.0)

        result_da = op.process(clean, noise)

        assert isinstance(result_da, DaArray)
        assert result_da.dtype == np.float64
        assert result_da.compute().dtype == np.float64

    def test_add_with_snr_rejects_missing_noise_before_dask_compute(self) -> None:
        """AddWithSNR declares two inputs so process() validates arity early."""
        clean = da_from_array(np.array([[1.0, -1.0, 0.5, -0.5]], dtype=np.float32), chunks=(1, -1))
        op = AddWithSNR(_SR, 10.0)

        with pytest.raises(ValueError, match="Expected exactly 2 inputs"):
            op.process(clean)

    # -- Layer 3: Integration (SNR verification) ---------------------------

    def test_add_with_snr_actual_snr_matches_target(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Actual SNR of output must match target within 10% (rtol=0.1).

        Tolerance rationale: finite-length signals cause small power estimation errors.
        """
        dask_input, sr = pure_sine_440hz_dask
        target_snr = 10.0
        rng = np.random.default_rng(42)
        noise = da_from_array(rng.standard_normal((1, sr)), chunks=(1, -1))
        op = AddWithSNR(sr, target_snr)

        clean = dask_input.compute()
        result = _compute_process(op, dask_input, noise)

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
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((2, _SR))

        dask_clean = da_from_array(clean, chunks=(1, -1))
        dask_noise = da_from_array(noise, chunks=(1, -1))
        op = AddWithSNR(_SR, 10.0)

        result_da = op.process(dask_clean, dask_noise)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        assert result.shape == clean.shape


class TestRemoveDC:
    def test_remove_dc_subtracts_channel_mean(self) -> None:
        signal = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0]])
        dask_signal = da_from_array(signal, chunks=(1, -1))
        remove_dc = RemoveDC(_SR)

        result = _compute_process(remove_dc, dask_signal)

        expected = signal - signal.mean(axis=-1, keepdims=True)
        np.testing.assert_allclose(result, expected)


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

    def test_normalize_zero_threshold_raises_error(self) -> None:
        """Threshold must be positive to avoid zero-length normalization division."""
        with pytest.raises(ValueError, match="Invalid threshold"):
            Normalize(sampling_rate=44100, threshold=0.0)

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

        result_da = normalize.process(dask_sig)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        max_val = np.max(np.abs(result))
        np.testing.assert_allclose(
            max_val,
            1.0,
            rtol=1e-10,  # float64 precision for simple division
        )

    def test_normalize_norm_none_returns_input_values(self) -> None:
        sig = np.array([[1.0, -2.0, 4.0]])
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=None)

        result = _compute_process(normalize, dask_sig)

        np.testing.assert_array_equal(result, sig)

    def test_normalize_integer_input_reports_float_dtype(self) -> None:
        sig = np.array([[1, 2]], dtype=np.int16)
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=-1)

        result_da = normalize.process(dask_sig)
        result = result_da.compute()

        assert result_da.dtype == np.float64
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, np.array([[0.5, 1.0]], dtype=np.float64))

    def test_normalize_signed_integer_min_peak_uses_float_magnitude(self) -> None:
        sig = np.array([[-32768, 0, 16384]], dtype=np.int16)
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=-1)

        result_da = normalize.process(dask_sig)
        result = result_da.compute()

        assert result_da.dtype == np.float64
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, np.array([[-1.0, 0.0, 0.5]], dtype=np.float64))

    def test_normalize_float32_peak_norm_preserves_dtype(self) -> None:
        sig = np.array([[1.0, -2.0, 4.0]], dtype=np.float32)
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=-1)

        result_da = normalize.process(dask_sig)
        result = result_da.compute()

        assert result_da.dtype == np.float32
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, np.array([[0.25, -0.5, 1.0]], dtype=np.float32))

    def test_normalize_inf_norm_values_match_numpy_reference(self) -> None:
        """Inf-norm normalization divides each channel by its max absolute value."""
        sig = np.array([[1.0, -2.0, 4.0], [0.5, -1.0, 0.25]])
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=-1)

        result = _compute_process(normalize, dask_sig)

        expected = sig / np.max(np.abs(sig), axis=-1, keepdims=True)
        np.testing.assert_allclose(result, expected)

    def test_normalize_negative_inf_norm_uses_min_abs_reference(self) -> None:
        sig = np.array([[1.0, -2.0, 4.0], [0.5, -1.0, 0.25]])
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=-np.inf, axis=-1)

        result = _compute_process(normalize, dask_sig)

        expected = sig / np.min(np.abs(sig), axis=-1, keepdims=True)
        np.testing.assert_allclose(result, expected)

    def test_normalize_l1_norm_sum_abs_equals_one(self) -> None:
        """With norm=1, sum|x| must equal 1.0 (theoretical).

        Tolerance: rtol=1e-10 — float64 summation precision.
        """
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig = (2.0 * np.sin(2 * np.pi * 440 * t)).reshape(1, -1)
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=1, axis=-1)

        result_da = normalize.process(dask_sig)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
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

        result_da = normalize.process(dask_sig)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        l2 = np.sqrt(np.sum(result**2, axis=-1))
        np.testing.assert_allclose(
            l2,
            1.0,
            rtol=1e-10,  # float64 precision
        )

    def test_normalize_integer_l2_norm_avoids_overflow(self) -> None:
        sig = np.array([[30000, -30000, 0, 0]], dtype=np.int16)
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=2, axis=-1)

        result_da = normalize.process(dask_sig)
        result = result_da.compute()

        assert result_da.dtype == np.float64
        assert result.dtype == np.float64
        np.testing.assert_allclose(np.sqrt(np.sum(result**2, axis=-1)), 1.0, rtol=1e-12)
        assert np.all(np.isfinite(result))

    def test_normalize_float32_finite_norm_reports_float64_dtype(self) -> None:
        sig = np.array([[3.0, 4.0]], dtype=np.float32)
        dask_sig = da_from_array(sig, chunks=(1, -1))
        normalize = Normalize(_SR, norm=2, axis=-1)

        result_da = normalize.process(dask_sig)
        result = result_da.compute()

        assert result_da.dtype == np.float64
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, np.array([[0.6, 0.8]], dtype=np.float64))

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

        result_da = normalize.process(dask_multi)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
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

        result_da = normalize.process(dask_multi)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()

        # Global max should be 1.0 after normalization
        np.testing.assert_allclose(
            np.max(np.abs(result)),
            1.0,
            rtol=1e-10,  # float64 normalization division precision
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

        result_da = normalize.process(dask_zero)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        np.testing.assert_allclose(result, 0.0)

    def test_normalize_threshold_prevents_amplification(self) -> None:
        """Signal below threshold is not normalized to 1.0."""
        small = np.full((1, _SR), 1e-12)
        dask_small = da_from_array(small, chunks=(1, -1))

        # threshold=1e-10 means signals with max < 1e-10 are left as-is
        normalize = Normalize(_SR, norm=np.inf, axis=-1, threshold=1e-10)
        result_da = normalize.process(dask_small)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        assert np.max(np.abs(result)) < 1.0

    def test_normalize_default_threshold_uses_float32_tiny(self) -> None:
        """Float32 subnormal values are treated as near-zero by default."""
        tiny = np.finfo(np.float32).tiny
        small_value = np.float32(tiny / 2)
        small = np.full((1, 4), small_value, dtype=np.float32)
        dask_small = da_from_array(small, chunks=(1, -1))

        normalize = Normalize(_SR, norm=np.inf, axis=-1)
        result = _compute_process(normalize, dask_small)

        assert np.max(np.abs(result)) < 1.0
        np.testing.assert_array_equal(result, small)

    def test_normalize_fill_true_makes_zero_nonzero(self) -> None:
        """With fill=True, zero vector is filled so max|x| == 1.0."""
        zero = np.zeros((1, _SR))
        dask_zero = da_from_array(zero, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=-1, fill=True)

        result_da = normalize.process(dask_zero)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        assert result.shape == zero.shape
        assert not np.allclose(result, 0.0)
        np.testing.assert_allclose(
            np.max(np.abs(result)),
            1.0,
            rtol=1e-10,  # fill produces exact normalized values
        )

    def test_normalize_fill_false_zeroes_small_signal(self) -> None:
        small = np.full((1, 4), 1e-12)
        dask_small = da_from_array(small, chunks=(1, -1))
        normalize = Normalize(_SR, norm=np.inf, axis=-1, threshold=1e-10, fill=False)

        result = _compute_process(normalize, dask_small)

        np.testing.assert_array_equal(result, np.zeros_like(small))

    def test_normalize_l2_fill_true_zero_signal_has_unit_norm(self) -> None:
        """With fill=True and norm=2, zero vectors are filled to unit L2 norm."""
        zero = np.zeros((1, 4))
        dask_zero = da_from_array(zero, chunks=(1, -1))
        normalize = Normalize(_SR, norm=2, axis=-1, fill=True)

        result = _compute_process(normalize, dask_zero)

        np.testing.assert_allclose(np.sqrt(np.sum(result**2, axis=-1)), 1.0)

    def test_normalize_norm_zero_fill_true_raises_error(self) -> None:
        zero = np.zeros((1, 4))

        with pytest.raises(ValueError, match="Cannot normalize with norm=0 and fill=True"):
            _normalize_array(zero, norm=0, axis=-1, threshold=None, fill=True)

    def test_normalize_helper_nonpositive_threshold_raises_error(self) -> None:
        values = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="threshold must be strictly positive"):
            _normalize_array(values, norm=np.inf, axis=-1, threshold=0.0, fill=None)

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
        assert "Positive value" in error_msg
        # Check HOW
        assert "Typical values:" in error_msg


class TestEffectLazyMetadata:
    def test_effect_operations_report_computed_shape_and_dtype_metadata(self) -> None:
        data = np.array([[1000, -1000, 500, -500]], dtype=np.int16)

        _assert_lazy_metadata_matches_compute(RemoveDC(_SR), data)
        _assert_lazy_metadata_matches_compute(Fade(_SR, fade_ms=0), data[0])
        _assert_lazy_metadata_matches_compute(AddWithSNR(_SR, snr=10), data, data.astype(np.float32))


class TestFade:
    def test_fade_negative_duration_raises_error(self) -> None:
        with pytest.raises(ValueError, match="fade_ms must be non-negative"):
            Fade(1000, fade_ms=-1)

    def test_fade_too_long_raises_error(self) -> None:
        fade = Fade(1000, fade_ms=5)

        with pytest.raises(ValueError, match="Fade length too long"):
            _compute_process(fade, np.ones((1, 10)))

    def test_fade_1d_input_is_reshaped_to_channel_axis(self) -> None:
        fade = Fade(1000, fade_ms=1)
        signal = np.ones(10)

        result = _compute_process(fade, signal)

        assert result.shape == (1, 10)
        assert result[0, 0] == 0.0
        assert result[0, -1] == 0.0

    def test_fade_zero_duration_reshapes_without_fading(self) -> None:
        fade = Fade(1000, fade_ms=0)
        signal = np.array([1.0, 2.0, 3.0])

        result = _compute_process(fade, signal)

        np.testing.assert_array_equal(result, signal.reshape(1, -1))
