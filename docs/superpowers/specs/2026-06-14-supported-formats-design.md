# Supported Formats Consistency Design

## Context

`ChannelFrameDataset` and `wd.from_folder()` currently include `.mp3` in their
default file extension list. The registered readers do not support `.mp3`:

- `SoundFileReader`: `.wav`, `.flac`, `.ogg`, `.aiff`, `.aif`, `.snd`
- `CSVFileReader`: `.csv`

This means a default folder scan can collect files that the normal reader
registry cannot read. The fix should keep `wd.from_folder("data")` from
including unreadable files by default, and should expose the supported formats
through the public API.

## Scope

This change will not add MP3 decoding. MP3 can still be requested explicitly
through `file_extensions=[".mp3"]`, but it will not be part of any default
format list until a reader that can actually load it is registered.

WDF remains a separate save/load path and is not part of the folder reader
registry used by `ChannelFrameDataset`.

## Design

Use the reader registry as the single source of truth for default
`ChannelFrameDataset` file extensions.

Add a helper in `wandas/io/readers.py` that returns supported extensions from
the registered readers. The helper will:

- inspect `_file_readers`;
- collect each reader class's `supported_extensions`;
- normalize extensions to lowercase with a leading dot;
- de-duplicate extensions;
- return a stable sorted list.

Add `wd.supported_formats()` in `wandas/__init__.py`. It will delegate to the
reader helper and return the same extension list used by default folder scans.
This makes the public API and the registry agree.

Update `ChannelFrameDataset.__init__()` and
`ChannelFrameDataset.from_folder()` so their default `file_extensions=None`
path uses the reader helper. Explicit `file_extensions` values remain honored
unchanged.

Update README/docs wording so the supported folder-reader formats match the
registry: `.wav`, `.flac`, `.ogg`, `.aiff`, `.aif`, `.snd`, and `.csv`.

## Behavior

`wd.from_folder("data")` will no longer include `.mp3` files by default.

If a caller registers a custom reader with `register_file_reader()`, subsequent
calls to `wd.supported_formats()` and default dataset construction will include
that reader's extensions.

If a caller explicitly passes unsupported extensions, Wandas will still scan
for those paths. Loading may fail as it does today because the explicit request
overrides the default safety behavior.

## Tests

Add or update focused tests for:

- `wd.supported_formats()` returns registry-backed extensions and excludes
  `.mp3`;
- `ChannelFrameDataset.from_folder(tmp_path)` does not include `.mp3` files by
  default;
- top-level `wd.from_folder(tmp_path)` has the same default filtering behavior;
- registering a custom reader updates the supported format helper;
- explicit `file_extensions=[".mp3"]` remains an override.

## Acceptance Criteria

`wd.from_folder("data")` must not include files by default that the registered
readers cannot read.
