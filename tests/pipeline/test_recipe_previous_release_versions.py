"""The Recipe operation versions written by the previous release must still replay."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from wandas.pipeline import default_recipe_registry

ROOT = Path(__file__).resolve().parents[2]
POLICY = "docs/src/explanation/public-api-stability.md"
RULE = f"Rule: a release replays the Recipe operation versions written by the previous release (see {POLICY})"

# The release whose written versions are listed below.
CATALOG_RELEASE = "0.14.0"

# Every (operation_id, version) a public call records in CATALOG_RELEASE.
# Contracts kept only to replay older Recipes are not listed.
WRITTEN_BY_CATALOG_RELEASE: tuple[tuple[str, int], ...] = (
    ("wandas.audio.a_weighting", 2),
    ("wandas.audio.abs", 1),
    ("wandas.audio.bandpass_filter", 2),
    ("wandas.audio.cepstrum", 3),
    ("wandas.audio.channel_difference", 1),
    ("wandas.audio.coherence", 2),
    ("wandas.audio.cross_correlation", 1),
    ("wandas.audio.csd", 2),
    ("wandas.audio.envelope", 1),
    ("wandas.audio.fade", 1),
    ("wandas.audio.fft", 3),
    ("wandas.audio.fix_length", 1),
    ("wandas.audio.highpass_filter", 2),
    ("wandas.audio.hpss_harmonic", 1),
    ("wandas.audio.hpss_percussive", 1),
    ("wandas.audio.integrate", 1),
    ("wandas.audio.loudness_zwtv", 2),
    ("wandas.audio.lowpass_filter", 2),
    ("wandas.audio.mean", 2),
    ("wandas.audio.mix", 1),
    ("wandas.audio.noct_spectrum", 1),
    ("wandas.audio.normalize", 1),
    ("wandas.audio.power", 2),
    ("wandas.audio.remove_dc", 1),
    ("wandas.audio.resampling", 1),
    ("wandas.audio.rms_trend", 3),
    ("wandas.audio.roughness_dw", 2),
    ("wandas.audio.roughness_dw_spec", 2),
    ("wandas.audio.segment", 1),
    ("wandas.audio.sharpness_din", 2),
    ("wandas.audio.sound_level", 3),
    ("wandas.audio.stft", 2),
    ("wandas.audio.sum", 2),
    ("wandas.audio.transfer_function", 2),
    ("wandas.audio.trigger", 1),
    ("wandas.audio.welch", 3),
    ("wandas.cepstral.lifter", 1),
    ("wandas.cepstral.to_spectral_envelope", 1),
    ("wandas.cepstrogram.lifter", 1),
    ("wandas.cepstrogram.to_spectral_envelope", 1),
    ("wandas.cepstrogram.trim", 1),
    ("wandas.channel.add_channel", 2),
    ("wandas.channel.concat_frame", 1),
    ("wandas.channel.remove_channel", 1),
    ("wandas.channel.rename_channels", 1),
    ("wandas.channel.with_calibration", 1),
    ("wandas.cross_spectral.coherence", 1),
    ("wandas.cross_spectral.transfer_function", 1),
    ("wandas.custom.apply", 1),
    ("wandas.frame.astype", 1),
    ("wandas.frame.get_channel", 1),
    ("wandas.frame.index", 1),
    ("wandas.frame.select_pair", 1),
    ("wandas.frame.time_slice", 2),
    ("wandas.frame.with_source_time_offset", 1),
    ("wandas.numpy.ufunc", 2),
    ("wandas.operator.add", 2),
    ("wandas.operator.divide", 2),
    ("wandas.operator.multiply", 2),
    ("wandas.operator.power", 2),
    ("wandas.operator.reverse_add", 2),
    ("wandas.operator.reverse_divide", 2),
    ("wandas.operator.reverse_multiply", 2),
    ("wandas.operator.reverse_power", 2),
    ("wandas.operator.reverse_subtract", 2),
    ("wandas.operator.subtract", 2),
    ("wandas.segments.average", 1),
    ("wandas.segments.coherence", 1),
    ("wandas.segments.drop", 1),
    ("wandas.segments.select", 1),
    ("wandas.segments.transfer_function", 1),
    ("wandas.spectral.ifft", 3),
    ("wandas.spectral.noct_synthesis", 3),
    ("wandas.spectrogram.absolute", 1),
    ("wandas.spectrogram.at_time", 1),
    ("wandas.spectrogram.cepstrum", 1),
    ("wandas.spectrogram.csd", 1),
    ("wandas.spectrogram.get_frame_at", 1),
    ("wandas.spectrogram.to_channel_frame", 1),
    ("wandas.spectrogram.trim", 1),
)


def _newest_release() -> str:
    versions = [
        tuple(int(part) for part in match.groups())
        for path in (ROOT / "docs/src/release-notes").glob("v*.md")
        if (match := re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)\.md", path.name))
    ]
    return ".".join(str(part) for part in max(versions))


def _written_candidates() -> list[str]:
    """Suggest the next catalog from the contracts that public methods record."""
    return sorted(
        f'("{description.operation_id}", {description.version}),'
        for description in default_recipe_registry().describe()
        if (method := description.method) is not None and (not method.startswith("_") or method.startswith("__"))
    )


def test_catalog_lists_the_versions_written_by_the_newest_release() -> None:
    newest = _newest_release()
    candidates = "\n".join(_written_candidates())
    assert CATALOG_RELEASE == newest, (
        f"WRITTEN_BY_CATALOG_RELEASE lists the versions written by {CATALOG_RELEASE}, "
        f"but the newest release note is v{newest}.md.\n"
        f"{RULE}.\n"
        f"Fix: in {Path(__file__).relative_to(ROOT)}, set CATALOG_RELEASE = {newest!r} and replace "
        "WRITTEN_BY_CATALOG_RELEASE with the versions that release writes. Public methods record these; "
        "also keep operations recorded through private methods, such as wandas.numpy.ufunc and "
        f"wandas.segments.select:\n{candidates}"
    )


@pytest.mark.parametrize(("operation_id", "version"), WRITTEN_BY_CATALOG_RELEASE)
def test_versions_written_by_the_previous_release_stay_registered(operation_id: str, version: int) -> None:
    try:
        operation = default_recipe_registry().require(operation_id, version)
    except KeyError:
        pytest.fail(
            f"Recipe operation {operation_id!r} version {version} was written by Wandas {CATALOG_RELEASE}, "
            "so Recipes saved with it no longer replay.\n"
            f"{RULE}.\n"
            f"Fix: restore the @recipe_operation({operation_id!r}, version={version}) replay declaration "
            "(keep it under a private method name when a newer version is written). It may be dropped once "
            "CATALOG_RELEASE moves past a release that writes it; record the removal in that release's notes."
        )
    assert (operation.operation_id, operation.version) == (operation_id, version)
