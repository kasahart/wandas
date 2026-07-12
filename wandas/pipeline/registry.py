from __future__ import annotations

_REPLAYABLE_APPLY_OPERATIONS = frozenset(
    {
        "a_weighting",
        "abs",
        "bandpass_filter",
        "fade",
        "hpss_harmonic",
        "hpss_percussive",
        "highpass_filter",
        "lowpass_filter",
        "loudness_zwtv",
        "normalize",
        "power",
        "remove_dc",
        "roughness_dw",
        "rms_trend",
        "resampling",
        "sharpness_din",
        "sound_level",
        "trim",
    }
)
_REPLAYABLE_METHOD_OPERATIONS = {
    "channel_difference": ("channel_difference", {"other_channel": "other_channel"}),
    "fix_length": ("fix_length", {"target_length": "length"}),
    "get_channel": (
        "get_channel",
        {
            "channel_mask": "channel_mask",
            "channel_idx": "channel_idx",
            "query": "query",
            "validate_query_keys": "validate_query_keys",
        },
    ),
    "mean": ("mean", {}),
    "remove_channel": ("remove_channel", {"key": "key"}),
    "rename_channels": ("rename_channels", {"mapping_items": "mapping"}),
    "rms_trend": ("rms_trend", {"frame_length": "frame_length", "hop_length": "hop_length", "dB": "dB", "Aw": "Aw"}),
    "sound_level": (
        "sound_level",
        {"freq_weighting": "freq_weighting", "time_weighting": "time_weighting", "dB": "dB"},
    ),
    "sum": ("sum", {}),
}
_REPLAYABLE_METHOD_NAMES = frozenset(method for method, _param_names in _REPLAYABLE_METHOD_OPERATIONS.values())
_REPLAYABLE_TYPED_METHOD_OPERATIONS = {
    "cepstrum": ("cepstrum", None),
    "coherence": ("coherence", None),
    "csd": ("csd", None),
    "fft": ("fft", None),
    "get_frame_at": ("get_frame_at", {"time_idx": "time_idx"}),
    "ifft": ("ifft", {}),
    "istft": ("istft", {}),
    "lifter": ("lifter", None),
    "noct_spectrum": ("noct_spectrum", None),
    "noct_synthesis": ("noct_synthesis", None),
    "roughness_dw_spec": ("roughness_dw_spec", None),
    "stft": ("stft", None),
    "spectral_envelope": ("to_spectral_envelope", {}),
    "transfer_function": ("transfer_function", None),
    "welch": (
        "welch",
        {
            "n_fft": "n_fft",
            "hop_length": "hop_length",
            "win_length": "win_length",
            "window": "window",
            "average": "average",
        },
    ),
}
_REPLAYABLE_TYPED_METHOD_NAMES = frozenset(
    method for method, _param_names in _REPLAYABLE_TYPED_METHOD_OPERATIONS.values()
)
_REPLAYABLE_SCALAR_OPERATIONS = frozenset({"+", "-", "*", "/", "**"})
_REPLAYABLE_GETITEM_INDEXING = frozenset(
    {"boolean_mask", "channel_slice", "integer_list", "label", "label_list", "multidimensional_slice"}
)
_REPLAYABLE_TERMINAL_PROPERTIES = frozenset(
    {
        "bark_axis",
        "crest_factor",
        "dB",
        "dBA",
        "freqs",
        "magnitude",
        "phase",
        "power",
        "rms",
        "source_time",
        "source_times",
        "time",
        "times",
        "unwrapped_phase",
    }
)
_REPLAYABLE_TERMINAL_METHODS = frozenset({"loudness_zwst", "sharpness_din_st"})
_REPLAYABLE_BINARY_FRAME_OPERATIONS = frozenset({"+", "-", "*", "/", "**", "add_with_snr"})
