# README Describe Color Range Design

## Goal

Fix the spectrogram color range in both README `describe()` examples to `-80` through `-20` dB while keeping the published sample figures and executable documentation tests synchronized.

## Scope

- Add `vmin=-80, vmax=-20` to the sample-audio `recording.describe(...)` call in `README.md` and `README.ja.md`.
- Add the same arguments to the own-data `clean.describe(...)` call in both languages.
- Use `start=0, end=15` for both README recording examples and regenerate the sample figures over 15 seconds.
- Update README tests so the published sample call must retain this exact range.
- Regenerate `images/readme_sample_audio_describe_0.png` and `images/readme_sample_audio_describe_1.png` from the documented sample-audio block.
- Do not change the `ChannelFrame.describe()` API or its defaults.

## Verification

- Run the focused range-contract test red before editing the README.
- Execute the README Python blocks and verify both sample figures are created.
- Confirm the regenerated PNG files are valid and changed only because of the documented color range.
- Run `uv run pytest tests/docs`, `uv run ruff check wandas tests`, and `uv run --extra marimo --extra psychoacoustic ty check wandas tests`.
- Check the final workspace for unintended generated artifacts.
