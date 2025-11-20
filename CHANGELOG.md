# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Changed

- Dask chunking defaults for channel-based frames: channel axis now uses chunk size of 1. Use `.rechunk(...)` to adjust sample-axis chunking.

### Removed

- `ChannelFrame.from_file(..., chunk_size=...)` argument removed; use `.rechunk(...)` on the resulting frame instead.
